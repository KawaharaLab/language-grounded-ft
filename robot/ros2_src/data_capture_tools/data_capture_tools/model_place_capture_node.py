"""Real-robot lift-and-lower evaluation with online cosine place detection."""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String

from .key_motion_capture_node import (
    KeyInput,
    KeyMotionCaptureNode,
    MotionState,
)
from .place_cosine_detector import CosinePlaceDetector, FORCE_COLUMNS
from .workspace_paths import resolve_workspace_path


@dataclass(frozen=True)
class PairedForceSample:
    """A timestamp-aligned bilateral sample in simulator axis convention."""

    sequence: int
    stamp_ns: int
    left_stamp_ns: int
    right_stamp_ns: int
    pair_skew_ms: float
    receipt_monotonic_ns: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class InferenceRequest:
    episode_id: int
    sequence: int
    window_start_stamp_ns: int
    window_end_stamp_ns: int
    submitted_monotonic_ns: int
    values: tuple[tuple[float, ...], ...]
    phase: str = "lowering"


@dataclass(frozen=True)
class InferenceResult:
    request: InferenceRequest
    completed_monotonic_ns: int
    hold_score: Optional[float]
    place_score: Optional[float]
    margin: Optional[float]
    inference_ms: Optional[float]
    error: Optional[str]
    fixed_surface_score: Optional[float] = None
    wobbling_surface_score: Optional[float] = None
    fixed_minus_wobbling: Optional[float] = None


class ModelPlaceCaptureNode(KeyMotionCaptureNode):
    """Run pick/lift/lower and let hold/place cosine scores stop descent."""

    _SCORE_FIELDS = [
        "episode_id",
        "phase",
        "sequence",
        "window_start_stamp_ns",
        "window_end_stamp_ns",
        "submitted_monotonic_ns",
        "completed_monotonic_ns",
        "queue_and_inference_ms",
        "inference_ms",
        "hold_score",
        "place_score",
        "margin",
        "fixed_surface_score",
        "wobbling_surface_score",
        "fixed_minus_wobbling",
        "place_positive",
        "release_eligible",
        "place_streak",
        "triggered",
        "floor_contact_stamp_ns",
        "window_end_after_contact_ms",
        "completed_after_contact_ms",
        "error",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._model_encoder_arch = str(
            self.declare_parameter("model_encoder_arch", "tokenfusion").value
        ).strip().lower()
        if self._model_encoder_arch not in {"tokenfusion", "cnn", "mlp"}:
            raise ValueError(
                "model_encoder_arch must be one of: tokenfusion, cnn, mlp"
            )
        checkpoint_parameters = {
            "tokenfusion": "model_checkpoint_path",
            "cnn": "model_cnn_checkpoint_path",
            "mlp": "model_mlp_checkpoint_path",
        }
        checkpoint_parameter = checkpoint_parameters[self._model_encoder_arch]
        checkpoint_value = str(
            self.declare_parameter(checkpoint_parameter, "").value
        ).strip()
        if not checkpoint_value:
            raise ValueError(f"{checkpoint_parameter} must not be empty")
        self._checkpoint_path = resolve_workspace_path(checkpoint_value)
        self._model_display_name = {
            "tokenfusion": "Token-Fusion PatchTST",
            "cnn": "CNN",
            "mlp": "MLP",
        }[self._model_encoder_arch]
        if self._model_encoder_arch != "tokenfusion":
            self.config.task_name = (
                f"{self.config.task_name}_{self._model_encoder_arch}"
            )
            self.config.prompt = self.config.prompt.replace(
                "PatchTST", self._model_display_name
            )
        self._normalization_path = resolve_workspace_path(
            str(self.declare_parameter("model_normalization_path", "").value)
        )
        self._hold_prototype_path = resolve_workspace_path(
            str(self.declare_parameter("hold_prototype_path", "").value)
        )
        self._place_prototype_path = resolve_workspace_path(
            str(self.declare_parameter("place_prototype_path", "").value)
        )
        self._patchtst_code = resolve_workspace_path(
            str(
                self.declare_parameter(
                    "patchtst_code_path",
                    "PatchTST/PatchTST_self_supervised",
                ).value
            )
        )
        self._model_device = str(
            self.declare_parameter("model_device", "auto").value
        )
        self._model_stride_samples = int(
            self.declare_parameter("model_stride_samples", 5).value
        )
        self._model_required_consecutive = int(
            self.declare_parameter("model_required_consecutive", 2).value
        )
        self._model_min_margin = float(
            self.declare_parameter("model_min_margin", 0.0).value
        )
        self._model_release_min_descent_sec = float(
            self.declare_parameter(
                "model_release_min_descent_sec", 0.0
            ).value
        )
        self._model_max_wait_after_contact_sec = float(
            self.declare_parameter(
                "model_max_wait_after_contact_sec", 0.5
            ).value
        )
        self._enter_reaction_compensation_sec = float(
            self.declare_parameter(
                "enter_reaction_compensation_sec", 0.3
            ).value
        )
        self._force_pair_max_skew_ms = float(
            self.declare_parameter("force_pair_max_skew_ms", 15.0).value
        )
        self._empty_offset_correction_enabled = bool(
            self.declare_parameter(
                "model_empty_offset_correction_enabled", False
            ).value
        )
        self._empty_offset_window_sec = float(
            self.declare_parameter(
                "model_empty_offset_window_sec", 1.0
            ).value
        )
        self._empty_offset_min_samples = int(
            self.declare_parameter(
                "model_empty_offset_min_samples", 50
            ).value
        )
        coordinate_sign = list(
            self.declare_parameter(
                "model_coordinate_sign",
                [
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                ],
            ).value
        )
        if len(coordinate_sign) != 12:
            raise ValueError("model_coordinate_sign must have 12 values")
        if self._model_stride_samples <= 0:
            raise ValueError("model_stride_samples must be positive")
        if self._model_required_consecutive <= 0:
            raise ValueError("model_required_consecutive must be positive")
        if self._model_max_wait_after_contact_sec <= 0.0:
            raise ValueError("model_max_wait_after_contact_sec must be positive")
        if self._model_release_min_descent_sec < 0.0:
            raise ValueError("model_release_min_descent_sec must be non-negative")
        if self._enter_reaction_compensation_sec < 0.0:
            raise ValueError(
                "enter_reaction_compensation_sec must be non-negative"
            )
        if self._empty_offset_window_sec <= 0.0:
            raise ValueError("model_empty_offset_window_sec must be positive")
        if self._empty_offset_min_samples <= 0:
            raise ValueError("model_empty_offset_min_samples must be positive")
        self._coordinate_sign = np.asarray(coordinate_sign, dtype=np.float32)

        self.get_logger().info(
            f"Loading the {self._model_display_name} place detector..."
        )
        self._detector = CosinePlaceDetector(
            checkpoint_path=self._checkpoint_path,
            normalization_path=self._normalization_path,
            hold_prototype_path=self._hold_prototype_path,
            place_prototype_path=self._place_prototype_path,
            patchtst_code=self._patchtst_code,
            device_name=self._model_device,
            expected_model_arch=self._model_encoder_arch,
        )
        if self._detector.normalization_id is not None:
            self.config.task_name = (
                f"{self.config.task_name}_norm_{self._detector.normalization_id}"
            )
        stats_sign = self._detector.normalization_metadata.get("coordinate_sign")
        if stats_sign is None or not np.allclose(
            self._coordinate_sign,
            np.asarray(stats_sign, dtype=np.float32),
        ):
            raise ValueError(
                "model_coordinate_sign does not match the normalization stats"
            )
        if (
            self._detector.normalization_metadata.get("coordinate_transform")
            != "sim_local_rz180"
        ):
            raise ValueError(
                "Expected sim_local_rz180 real-force normalization stats"
            )
        empty_offset_metadata = self._detector.normalization_metadata.get(
            "empty_offset_reference"
        )
        if self._empty_offset_correction_enabled:
            if not isinstance(empty_offset_metadata, dict):
                raise ValueError(
                    "Normalization stats have no empty_offset_reference"
                )
            reference = np.asarray(
                empty_offset_metadata.get("mean", []), dtype=np.float32
            )
            if reference.shape != (12,) or not np.isfinite(reference).all():
                raise ValueError(
                    "empty_offset_reference.mean must contain 12 finite values"
                )
            if (
                empty_offset_metadata.get("coordinate_transform")
                != "sim_local_rz180"
            ):
                raise ValueError(
                    "Empty-offset coordinate transform does not match model input"
                )
            self._empty_offset_reference = reference
        else:
            self._empty_offset_reference = np.zeros(12, dtype=np.float32)
        self._empty_offset_reference_metadata = empty_offset_metadata
        self._empty_offset_adjustment = np.zeros(12, dtype=np.float32)
        self._empty_offset_details: Optional[dict[str, object]] = None
        self.get_logger().info(
            "Place detector ready: "
            f"arch={self._detector.model_arch}, device={self._detector.device}, context="
            f"{self._detector.context_points} samples, stride="
            f"{self._model_stride_samples}, consecutive="
            f"{self._model_required_consecutive}, release gate="
            f"{self._model_release_min_descent_sec:.3f} s, empty-offset "
            f"correction={self._empty_offset_correction_enabled}, normalization="
            f"{self._detector.normalization_id or 'legacy_unversioned'} "
            f"(sha256={self._detector.normalization_sha256[:12]})."
        )

        self._model_score_topic = str(
            self.declare_parameter(
                "model_score_topic", "/data_capture/model_place_scores"
            ).value
        )
        self._model_score_pub = self.create_publisher(
            String, self._model_score_topic, 10
        )
        self._paired_samples: deque[PairedForceSample] = deque(maxlen=8192)
        self._last_paired_stamp_ns: dict[str, Optional[int]] = {
            "left": None,
            "right": None,
        }
        self._pair_sequence = 0
        self._last_submitted_sequence: Optional[int] = None
        self._request_queue: queue.Queue[Optional[InferenceRequest]] = queue.Queue(
            maxsize=1
        )
        self._result_queue: queue.Queue[InferenceResult] = queue.Queue()
        self._inference_stop = threading.Event()
        self._inference_thread = threading.Thread(
            target=self._inference_worker,
            name="place-inference",
            daemon=True,
        )
        self._inference_thread.start()

        self._place_streak = 0
        self._pending_model_trigger: Optional[InferenceResult] = None
        self._model_error: Optional[str] = None
        self._enter_input: Optional[KeyInput] = None
        self._floor_contact: Optional[KeyInput] = None
        self._run_score_csv_path: Optional[Path] = None
        self._episode_score_csv_path: Optional[Path] = None
        self._latest_result: Optional[InferenceResult] = None
        self._evaluation_context: Optional[
            tuple[str, str, int, int, Optional[InferenceResult]]
        ] = None
        self._model_completion_status: Optional[str] = None
        self._model_completion_reason: Optional[str] = None

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._inference_stop.set()
            self._replace_pending_request(None)
            self._inference_thread.join(timeout=2.0)

    def _ready_message(self, sequence: str) -> str:
        return (
            "READY (model evaluation): Right arrow records, closes, "
            f"{sequence}. {self._model_display_name} scores every "
            f"{self._model_stride_samples} "
            "force samples, ignores them for the first "
            f"{self._model_release_min_descent_sec:.1f} s of descent, and "
            "stops/opens after "
            f"{self._model_required_consecutive} consecutive place decisions. "
            "Enter only marks ground-truth floor contact; it never commands a "
            "stop. Floor contact is estimated "
            f"{self._enter_reaction_compensation_sec * 1000.0:.0f} "
            "ms before Enter. Up aborts, Space stops motion, and q exits."
        )

    def _write_capture_config_snapshot(self, destination: Path) -> None:
        """Record the effective dataset name and its exact calibration."""
        snapshot = dict(self.config.raw)
        snapshot.update(
            {
                "task_name": self.config.task_name,
                "object_name": self._object_name or None,
                "surface_condition": self._surface_condition or None,
                "prompt": self.config.prompt,
                "model_encoder_arch": self._model_encoder_arch,
                "normalization_id": self._detector.normalization_id,
                "normalization_sha256": self._detector.normalization_sha256,
                "normalization_path": str(self._detector.normalization_path),
                "normalization_snapshot": "normalization_snapshot.json",
            }
        )
        destination.write_text(
            yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self._detector.write_normalization_snapshot(
            destination.parent / "normalization_snapshot.json"
        )

    def _prepare_run(self) -> None:
        super()._prepare_run()
        if self._run_dir is None:
            return
        self._run_score_csv_path = self._run_dir / "csv" / "model_scores_all.csv"
        self._initialize_score_csv(self._run_score_csv_path)
        metadata = self._detector.metadata()
        metadata.update(
            {
                "object_name": self._object_name or None,
                "surface_condition": self._surface_condition or None,
                "task_name": self.config.task_name,
                "normalization_snapshot": "normalization_snapshot.json",
                "force_columns": FORCE_COLUMNS,
                "coordinate_sign": self._coordinate_sign.tolist(),
                "coordinate_transform": "sim_local_rz180",
                "score_stride_samples": self._model_stride_samples,
                "required_consecutive_place_scores": self._model_required_consecutive,
                "minimum_place_minus_hold_margin": self._model_min_margin,
                "release_minimum_descent_sec": self._model_release_min_descent_sec,
                "max_wait_after_estimated_contact_sec": (
                    self._model_max_wait_after_contact_sec
                ),
                "enter_reaction_compensation_sec": (
                    self._enter_reaction_compensation_sec
                ),
                "enter_semantics": (
                    "raw keyboard time; floor_contact is Enter minus reaction "
                    "compensation"
                ),
                "empty_offset_correction": {
                    "enabled": self._empty_offset_correction_enabled,
                    "window_sec": self._empty_offset_window_sec,
                    "minimum_samples": self._empty_offset_min_samples,
                    "reference": self._empty_offset_reference.tolist(),
                    "reference_metadata": self._empty_offset_reference_metadata,
                    "formula": (
                        "model_input = transformed_force + "
                        "(reference_empty - current_empty)"
                    ),
                },
            }
        )
        (self._run_dir / "model_evaluation_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _start_episode(self, trigger: KeyInput) -> None:
        self._reset_model_episode()
        super()._start_episode(trigger)
        if self._session_dir is None:
            return
        try:
            self._calibrate_empty_offset()
        except Exception:
            if self._active_episode_id is not None:
                self._discard_active_episode(
                    "model_empty_offset_calibration_failed"
                )
            raise
        self._episode_score_csv_path = self._session_dir / "csv" / "model_scores.csv"
        self._initialize_score_csv(self._episode_score_csv_path)
        if self._run_dir is not None:
            source = self._run_dir / "model_evaluation_config.json"
            destination = self._session_dir / "model_evaluation_config.json"
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    def _clear_active_episode(self) -> None:
        super()._clear_active_episode()
        self._episode_score_csv_path = None

    def _reset_model_episode(self) -> None:
        self._paired_samples.clear()
        self._last_paired_stamp_ns = {"left": None, "right": None}
        self._pair_sequence = 0
        self._last_submitted_sequence = None
        self._place_streak = 0
        self._pending_model_trigger = None
        self._model_error = None
        self._enter_input = None
        self._floor_contact = None
        self._latest_result = None
        self._evaluation_context = None
        self._model_completion_status = None
        self._model_completion_reason = None
        self._empty_offset_adjustment = np.zeros(12, dtype=np.float32)
        self._empty_offset_details = None
        self._drain_queue(self._request_queue)
        self._drain_queue(self._result_queue)

    def _calibrate_empty_offset(self) -> None:
        """Align this episode's open-gripper force offset to the training data."""
        if not self._empty_offset_correction_enabled:
            self._empty_offset_details = {
                "enabled": False,
                "adjustment": self._empty_offset_adjustment.tolist(),
            }
            return
        if self._session_dir is None or not self._paired_samples:
            raise RuntimeError("No paired force samples for empty-offset correction")

        latest_stamp_ns = self._paired_samples[-1].stamp_ns
        start_stamp_ns = latest_stamp_ns - int(
            self._empty_offset_window_sec * 1.0e9
        )
        samples = [
            sample
            for sample in self._paired_samples
            if sample.stamp_ns >= start_stamp_ns
        ]
        if len(samples) < self._empty_offset_min_samples:
            raise RuntimeError(
                "Empty-offset correction has only "
                f"{len(samples)} paired samples; "
                f"need {self._empty_offset_min_samples}"
            )
        current = np.asarray(
            [sample.values for sample in samples], dtype=np.float32
        ).mean(axis=0)
        adjustment = self._empty_offset_reference - current
        if not np.isfinite(adjustment).all():
            raise RuntimeError("Empty-offset correction produced non-finite values")
        self._empty_offset_adjustment = adjustment.astype(np.float32)
        self._empty_offset_details = {
            "enabled": True,
            "normalization_id": self._detector.normalization_id,
            "normalization_sha256": self._detector.normalization_sha256,
            "window_sec": self._empty_offset_window_sec,
            "sample_count": len(samples),
            "window_start_stamp_ns": samples[0].stamp_ns,
            "window_end_stamp_ns": samples[-1].stamp_ns,
            "reference": self._empty_offset_reference.tolist(),
            "current": current.tolist(),
            "adjustment": self._empty_offset_adjustment.tolist(),
            "formula": (
                "model_input = transformed_force + "
                "(reference_empty - current_empty)"
            ),
        }
        path = self._session_dir / "empty_offset_correction.json"
        path.write_text(
            json.dumps(
                self._empty_offset_details,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        largest = np.argsort(np.abs(adjustment))[::-1][:4]
        summary = ", ".join(
            f"{FORCE_COLUMNS[index]}={adjustment[index]:+.4f}"
            for index in largest
        )
        self.get_logger().info(
            "Empty-force offset correction calibrated from "
            f"{len(samples)} samples: {summary}"
        )

    @staticmethod
    def _drain_queue(work_queue) -> None:
        while True:
            try:
                work_queue.get_nowait()
            except queue.Empty:
                return

    def _force_cb(self, side: str, message: WrenchStamped) -> None:
        super()._force_cb(side, message)
        if not self._force_recording_active:
            return
        self._try_pair_force_samples()

    def _try_pair_force_samples(self) -> None:
        if not self._force_samples["left"] or not self._force_samples["right"]:
            return
        latest_left = self._force_samples["left"][-1]
        latest_right = self._force_samples["right"][-1]
        if (
            self._last_paired_stamp_ns["left"] is not None
            and latest_left.stamp_ns <= self._last_paired_stamp_ns["left"]
        ) or (
            self._last_paired_stamp_ns["right"] is not None
            and latest_right.stamp_ns <= self._last_paired_stamp_ns["right"]
        ):
            return
        left = latest_left
        right = latest_right
        skew_ms = abs(left.stamp_ns - right.stamp_ns) / 1.0e6
        if skew_ms > self._force_pair_max_skew_ms:
            return

        raw = np.asarray(left.values + right.values, dtype=np.float32)
        transformed = raw * self._coordinate_sign
        self._pair_sequence += 1
        paired = PairedForceSample(
            sequence=self._pair_sequence,
            stamp_ns=(left.stamp_ns + right.stamp_ns) // 2,
            left_stamp_ns=left.stamp_ns,
            right_stamp_ns=right.stamp_ns,
            pair_skew_ms=skew_ms,
            receipt_monotonic_ns=max(
                left.receipt_monotonic_ns, right.receipt_monotonic_ns
            ),
            values=tuple(float(value) for value in transformed),
        )
        self._paired_samples.append(paired)
        self._last_paired_stamp_ns["left"] = left.stamp_ns
        self._last_paired_stamp_ns["right"] = right.stamp_ns
        self._maybe_submit_inference(paired)

    def _maybe_submit_inference(self, newest: PairedForceSample) -> None:
        if self._state != MotionState.LOWERING or self._active_episode_id is None:
            return
        context = self._detector.context_points
        if len(self._paired_samples) < context:
            return
        if (
            self._last_submitted_sequence is not None
            and newest.sequence - self._last_submitted_sequence
            < self._model_stride_samples
        ):
            return
        window = list(self._paired_samples)[-context:]
        corrected_values = [
            np.asarray(sample.values, dtype=np.float32)
            + self._empty_offset_adjustment
            for sample in window
        ]
        request = InferenceRequest(
            episode_id=self._active_episode_id,
            sequence=newest.sequence,
            window_start_stamp_ns=window[0].stamp_ns,
            window_end_stamp_ns=window[-1].stamp_ns,
            submitted_monotonic_ns=time.perf_counter_ns(),
            values=tuple(
                tuple(float(value) for value in sample)
                for sample in corrected_values
            ),
            phase=self._inference_phase_label(),
        )
        self._last_submitted_sequence = newest.sequence
        self._replace_pending_request(request)

    def _replace_pending_request(
        self, request: Optional[InferenceRequest]
    ) -> None:
        try:
            self._request_queue.put_nowait(request)
            return
        except queue.Full:
            pass
        try:
            self._request_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            pass

    def _inference_worker(self) -> None:
        while not self._inference_stop.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return
            try:
                scores = self._detector.score(request.values)
                result = InferenceResult(
                    request=request,
                    completed_monotonic_ns=time.perf_counter_ns(),
                    hold_score=scores["hold_score"],
                    place_score=scores["place_score"],
                    margin=scores["margin"],
                    inference_ms=scores["inference_ms"],
                    error=None,
                    fixed_surface_score=scores.get("fixed_surface_score"),
                    wobbling_surface_score=scores.get(
                        "wobbling_surface_score"
                    ),
                    fixed_minus_wobbling=scores.get("fixed_minus_wobbling"),
                )
            except Exception as exc:  # noqa: BLE001
                result = InferenceResult(
                    request=request,
                    completed_monotonic_ns=time.perf_counter_ns(),
                    hold_score=None,
                    place_score=None,
                    margin=None,
                    inference_ms=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._result_queue.put(result)

    def _inference_phase_label(self) -> str:
        """Return a stable label stored with each asynchronous score."""
        return "lowering"

    def _accept_inference_result(self, result: InferenceResult) -> bool:
        """Allow subclasses to reject a result left over from an old phase."""
        return result.request.phase == self._inference_phase_label()

    def _drain_key_events(self) -> None:
        # Keyboard is drained first so an Enter and a model result arriving in
        # the same spin iteration preserve the ground-truth contact ordering.
        super()._drain_key_events()
        self._drain_model_results()

    def _drain_model_results(self) -> None:
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                return
            if result.request.episode_id != self._active_episode_id:
                continue
            self._latest_result = result
            if self._state != MotionState.LOWERING:
                self._write_score_result(
                    result, False, False, self._place_streak, False
                )
                continue
            if not self._accept_inference_result(result):
                self._write_score_result(
                    result, False, False, self._place_streak, False
                )
                continue
            if result.error is not None:
                self._model_error = result.error
                self._write_score_result(
                    result, False, False, self._place_streak, False
                )
                continue
            positive = bool(result.margin is not None and result.margin > self._model_min_margin)
            release_eligible = bool(
                self._lower_started_monotonic_ns is not None
                and result.completed_monotonic_ns
                - self._lower_started_monotonic_ns
                >= int(self._model_release_min_descent_sec * 1.0e9)
            )
            self._place_streak = (
                self._place_streak + 1 if positive and release_eligible else 0
            )
            triggered = self._place_streak >= self._model_required_consecutive
            self._write_score_result(
                result,
                positive,
                release_eligible,
                self._place_streak,
                triggered,
            )
            if triggered and self._pending_model_trigger is None:
                self._pending_model_trigger = result

    def _handle_key(self, event: KeyInput) -> None:
        contact_states = {
            MotionState.LOWERING,
            MotionState.FINALIZING_CAPTURE,
            MotionState.OPENING_FOR_RETURN,
        }
        if (
            event.key == "enter"
            and self._state in contact_states
            and self._active_episode_id is not None
        ):
            state = self._state.value
            if self._floor_contact is None:
                compensation_ns = int(
                    self._enter_reaction_compensation_sec * 1.0e9
                )
                self._enter_input = event
                self._floor_contact = KeyInput(
                    key="floor_contact",
                    ros_stamp_ns=event.ros_stamp_ns - compensation_ns,
                    monotonic_ns=event.monotonic_ns - compensation_ns,
                )
                self._queue_key_event(event, state, state)
                if self._enter_event_alias and self._enter_event_alias != event.key:
                    self._queue_system_event(
                        self._enter_event_alias,
                        state,
                        state,
                        timing=self._floor_contact,
                    )
                self._backfill_score_contact_columns(self._floor_contact)
                if self._state == MotionState.LOWERING:
                    self.get_logger().info(
                        "Enter recorded; estimated floor contact is "
                        f"{self._enter_reaction_compensation_sec * 1000.0:.1f} ms "
                        "earlier. The model remains in control of stopping."
                    )
                else:
                    self.get_logger().info(
                        "Enter arrived after the model halt; reaction-time "
                        "compensation is being applied to the contact estimate."
                    )
                    if self._evaluation_context is not None:
                        _, _, halt_ros_ns, halt_monotonic_ns, result = (
                            self._evaluation_context
                        )
                        outcome, status = self._classify_detection(result)
                        self._model_completion_status = status
                        self._model_completion_reason = outcome
                        self._write_model_evaluation(
                            outcome=outcome,
                            status=status,
                            halt_ros_ns=halt_ros_ns,
                            halt_monotonic_ns=halt_monotonic_ns,
                            result=result,
                        )
            else:
                self._queue_key_event(event, state, state)
                self.get_logger().warning("Duplicate floor-contact Enter ignored.")
            return
        super()._handle_key(event)

    def _classify_detection(
        self, result: Optional[InferenceResult]
    ) -> tuple[str, str]:
        if result is None:
            return "model_not_triggered", "failure"
        contact = self._floor_contact
        if contact is None:
            return "model_place_detected_before_enter", "failure"
        if result.request.window_end_stamp_ns < contact.ros_stamp_ns:
            return "model_trigger_window_ended_before_floor_contact", "failure"
        return "model_place_detected_after_floor_contact", "success"

    def _complete_active_episode(
        self,
        status_name: str = "success",
        reason: Optional[str] = None,
    ) -> None:
        if self._model_completion_status is not None:
            status_name = self._model_completion_status
        if self._model_completion_reason is not None:
            reason = self._model_completion_reason
        super()._complete_active_episode(status_name=status_name, reason=reason)

    def _control_tick(self) -> None:
        now_ns = time.perf_counter_ns()
        if self._state == MotionState.LOWERING:
            if self._pending_model_trigger is not None:
                result = self._pending_model_trigger
                self._pending_model_trigger = None
                self._finish_model_episode(result)
                return
            if self._model_error is not None:
                self._finish_without_trigger("model_inference_error")
                return
            if (
                self._floor_contact is not None
                and now_ns - self._floor_contact.monotonic_ns
                >= int(self._model_max_wait_after_contact_sec * 1.0e9)
            ):
                self._finish_without_trigger("model_detection_timeout_after_contact")
                return
            if (
                self._lower_started_monotonic_ns is not None
                and now_ns - self._lower_started_monotonic_ns
                >= int(self._max_lower_duration_sec * 1.0e9)
            ):
                self._finish_without_trigger("max_lower_duration_before_detection")
                return
        super()._control_tick()

    def _finish_model_episode(self, result: InferenceResult) -> None:
        halt_monotonic_ns = time.perf_counter_ns()
        halt_ros_ns = int(self.get_clock().now().nanoseconds)
        self._publish_motion_halt()
        self._lower_ended_monotonic_ns = halt_monotonic_ns
        previous = self._state.value
        self._state = MotionState.FINALIZING_CAPTURE
        self._queue_system_event(
            "model_place_detected", previous, self._state.value
        )

        reason, status = self._classify_detection(result)
        self._model_completion_status = status
        self._model_completion_reason = reason
        self._write_model_evaluation(
            outcome=reason,
            status=status,
            halt_ros_ns=halt_ros_ns,
            halt_monotonic_ns=halt_monotonic_ns,
            result=result,
        )
        self.get_logger().info(
            "Place detected: stopping descent immediately and opening the gripper "
            f"(hold={result.hold_score:.4f}, place={result.place_score:.4f}, "
            f"streak={self._place_streak})."
        )
        self._open_gripper_and_start_return(
            reference_monotonic_ns=halt_monotonic_ns,
            complete_episode_on_open=True,
            episode_status=status,
            episode_reason=reason,
        )

    def _finish_without_trigger(self, reason: str) -> None:
        halt_monotonic_ns = time.perf_counter_ns()
        halt_ros_ns = int(self.get_clock().now().nanoseconds)
        self._publish_motion_halt()
        self._lower_ended_monotonic_ns = halt_monotonic_ns
        previous = self._state.value
        self._state = MotionState.FINALIZING_CAPTURE
        self._queue_system_event(reason, previous, self._state.value)
        self._write_model_evaluation(
            outcome=reason,
            status="failure",
            halt_ros_ns=halt_ros_ns,
            halt_monotonic_ns=halt_monotonic_ns,
            result=None,
        )
        self.get_logger().warning(
            f"{reason}: stopping for safety, releasing, and retaining the failed trial."
        )
        self._open_gripper_and_start_return(
            reference_monotonic_ns=halt_monotonic_ns,
            complete_episode_on_open=True,
            episode_status="failure",
            episode_reason=reason,
        )

    def _initialize_score_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._SCORE_FIELDS).writeheader()

    def _backfill_score_contact_columns(self, contact: KeyInput) -> None:
        """Attach a late operator contact mark to scores already written."""
        if self._active_episode_id is None:
            return
        active_phase = self._inference_phase_label()
        for path in (self._run_score_csv_path, self._episode_score_csv_path):
            if path is None or not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows = list(reader)
            if fieldnames is None:
                continue
            changed = False
            for row in rows:
                try:
                    episode_id = int(row["episode_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if episode_id != self._active_episode_id:
                    continue
                row_phase = row.get("phase", "")
                if row_phase and row_phase != active_phase:
                    continue
                row["floor_contact_stamp_ns"] = str(contact.ros_stamp_ns)
                try:
                    row["window_end_after_contact_ms"] = str(
                        (int(row["window_end_stamp_ns"]) - contact.ros_stamp_ns)
                        / 1.0e6
                    )
                    row["completed_after_contact_ms"] = str(
                        (int(row["completed_monotonic_ns"]) - contact.monotonic_ns)
                        / 1.0e6
                    )
                except (KeyError, TypeError, ValueError):
                    pass
                changed = True
            if changed:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

    def _write_score_result(
        self,
        result: InferenceResult,
        positive: bool,
        release_eligible: bool,
        streak: int,
        triggered: bool,
    ) -> None:
        contact = self._floor_contact
        row = {
            "episode_id": result.request.episode_id,
            "phase": result.request.phase,
            "sequence": result.request.sequence,
            "window_start_stamp_ns": result.request.window_start_stamp_ns,
            "window_end_stamp_ns": result.request.window_end_stamp_ns,
            "submitted_monotonic_ns": result.request.submitted_monotonic_ns,
            "completed_monotonic_ns": result.completed_monotonic_ns,
            "queue_and_inference_ms": (
                result.completed_monotonic_ns - result.request.submitted_monotonic_ns
            )
            / 1.0e6,
            "inference_ms": result.inference_ms,
            "hold_score": result.hold_score,
            "place_score": result.place_score,
            "margin": result.margin,
            "fixed_surface_score": result.fixed_surface_score,
            "wobbling_surface_score": result.wobbling_surface_score,
            "fixed_minus_wobbling": result.fixed_minus_wobbling,
            "place_positive": positive,
            "release_eligible": release_eligible,
            "place_streak": streak,
            "triggered": triggered,
            "floor_contact_stamp_ns": contact.ros_stamp_ns if contact else None,
            "window_end_after_contact_ms": (
                (result.request.window_end_stamp_ns - contact.ros_stamp_ns) / 1.0e6
                if contact
                else None
            ),
            "completed_after_contact_ms": (
                (result.completed_monotonic_ns - contact.monotonic_ns) / 1.0e6
                if contact
                else None
            ),
            "error": result.error,
        }
        for path in (self._run_score_csv_path, self._episode_score_csv_path):
            if path is None:
                continue
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._SCORE_FIELDS).writerow(row)
        self._model_score_pub.publish(
            String(data=json.dumps(row, ensure_ascii=False, allow_nan=False))
        )

    def _write_model_evaluation(
        self,
        outcome: str,
        status: str,
        halt_ros_ns: int,
        halt_monotonic_ns: int,
        result: Optional[InferenceResult],
    ) -> None:
        if self._session_dir is None:
            return
        self._evaluation_context = (
            outcome,
            status,
            halt_ros_ns,
            halt_monotonic_ns,
            result,
        )
        contact = self._floor_contact
        interval = []
        if contact is not None:
            interval_start_ns = min(contact.ros_stamp_ns, halt_ros_ns)
            interval_end_ns = max(contact.ros_stamp_ns, halt_ros_ns)
            interval = [
                sample
                for sample in self._paired_samples
                if interval_start_ns <= sample.stamp_ns <= interval_end_ns
            ]
        force_csv = self._session_dir / "csv" / "contact_to_detection_force.csv"
        self._write_force_interval(force_csv, interval)

        metrics = {
            "episode_id": self._active_episode_id,
            "object_name": self._object_name or None,
            "surface_condition": self._surface_condition or None,
            "task_name": self.config.task_name,
            "status": status,
            "outcome": outcome,
            "floor_contact_recorded": contact is not None,
            "enter_key_stamp_ns": (
                self._enter_input.ros_stamp_ns if self._enter_input else None
            ),
            "enter_key_monotonic_ns": (
                self._enter_input.monotonic_ns if self._enter_input else None
            ),
            "enter_reaction_compensation_ms": (
                self._enter_reaction_compensation_sec * 1000.0
            ),
            "floor_contact_stamp_ns": contact.ros_stamp_ns if contact else None,
            "floor_contact_monotonic_ns": contact.monotonic_ns if contact else None,
            "halt_stamp_ns": halt_ros_ns,
            "halt_monotonic_ns": halt_monotonic_ns,
            "contact_to_halt_ms": (
                (halt_monotonic_ns - contact.monotonic_ns) / 1.0e6
                if contact
                else None
            ),
            "interval_order": (
                "contact_then_halt"
                if contact is not None and contact.ros_stamp_ns <= halt_ros_ns
                else "halt_then_contact"
                if contact is not None
                else None
            ),
            "model_triggered": result is not None,
            "place_streak_at_halt": self._place_streak,
            "model_error": self._model_error,
            "contact_to_detection_force_csv": str(force_csv),
            "force_interval": self._force_interval_metrics(interval),
            "detector": self._detector.metadata(),
            "empty_offset_correction": self._empty_offset_details,
            "decision": None,
        }
        if result is not None:
            metrics["decision"] = {
                "sequence": result.request.sequence,
                "window_start_stamp_ns": result.request.window_start_stamp_ns,
                "window_end_stamp_ns": result.request.window_end_stamp_ns,
                "hold_score": result.hold_score,
                "place_score": result.place_score,
                "margin": result.margin,
                "fixed_surface_score": result.fixed_surface_score,
                "wobbling_surface_score": result.wobbling_surface_score,
                "fixed_minus_wobbling": result.fixed_minus_wobbling,
                "inference_ms": result.inference_ms,
                "queue_and_inference_ms": (
                    result.completed_monotonic_ns
                    - result.request.submitted_monotonic_ns
                )
                / 1.0e6,
                "window_end_after_contact_ms": (
                    (result.request.window_end_stamp_ns - contact.ros_stamp_ns)
                    / 1.0e6
                    if contact
                    else None
                ),
                "compute_complete_after_contact_ms": (
                    (result.completed_monotonic_ns - contact.monotonic_ns) / 1.0e6
                    if contact
                    else None
                ),
                "halt_after_compute_complete_ms": (
                    halt_monotonic_ns - result.completed_monotonic_ns
                )
                / 1.0e6,
            }
        path = self._session_dir / "model_evaluation.json"
        path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._model_score_pub.publish(
            String(
                data=json.dumps(
                    {"type": "episode_result", **metrics},
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        )

    @staticmethod
    def _write_force_interval(path: Path, samples: list[PairedForceSample]) -> None:
        fields = [
            "sequence",
            "stamp_ns",
            "left_stamp_ns",
            "right_stamp_ns",
            "pair_skew_ms",
            *FORCE_COLUMNS,
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample in samples:
                row = {
                    "sequence": sample.sequence,
                    "stamp_ns": sample.stamp_ns,
                    "left_stamp_ns": sample.left_stamp_ns,
                    "right_stamp_ns": sample.right_stamp_ns,
                    "pair_skew_ms": sample.pair_skew_ms,
                }
                row.update(dict(zip(FORCE_COLUMNS, sample.values)))
                writer.writerow(row)

    @staticmethod
    def _force_interval_metrics(samples: list[PairedForceSample]) -> dict:
        if not samples:
            return {
                "sample_count": 0,
                "duration_ms": None,
                "left_force_norm_mean_n": None,
                "left_force_norm_max_n": None,
                "right_force_norm_mean_n": None,
                "right_force_norm_max_n": None,
                "combined_force_norm_mean_n": None,
                "combined_force_norm_max_n": None,
                "combined_force_impulse_ns": None,
            }
        values = np.asarray([sample.values for sample in samples], dtype=np.float64)
        stamps_sec = np.asarray([sample.stamp_ns for sample in samples], dtype=np.float64)
        stamps_sec = (stamps_sec - stamps_sec[0]) / 1.0e9
        left_force = np.linalg.norm(values[:, 0:3], axis=1)
        right_force = np.linalg.norm(values[:, 6:9], axis=1)
        combined = left_force + right_force
        impulse = float(np.trapz(combined, stamps_sec)) if len(samples) > 1 else 0.0
        intervals_ms = np.diff(stamps_sec) * 1000.0
        return {
            "sample_count": len(samples),
            "duration_ms": float(stamps_sec[-1] * 1000.0),
            "sample_interval_mean_ms": (
                float(intervals_ms.mean()) if intervals_ms.size else None
            ),
            "sample_interval_max_ms": (
                float(intervals_ms.max()) if intervals_ms.size else None
            ),
            "pair_skew_max_ms": max(sample.pair_skew_ms for sample in samples),
            "left_force_norm_mean_n": float(left_force.mean()),
            "left_force_norm_rms_n": float(math.sqrt(np.mean(left_force**2))),
            "left_force_norm_max_n": float(left_force.max()),
            "right_force_norm_mean_n": float(right_force.mean()),
            "right_force_norm_rms_n": float(math.sqrt(np.mean(right_force**2))),
            "right_force_norm_max_n": float(right_force.max()),
            "combined_force_norm_mean_n": float(combined.mean()),
            "combined_force_norm_max_n": float(combined.max()),
            "combined_force_impulse_ns": impulse,
            "left_fz_mean_n": float(values[:, 2].mean()),
            "left_fz_min_n": float(values[:, 2].min()),
            "left_fz_max_n": float(values[:, 2].max()),
            "right_fz_mean_n": float(values[:, 8].mean()),
            "right_fz_min_n": float(values[:, 8].min()),
            "right_fz_max_n": float(values[:, 8].max()),
            "per_axis_abs_max": {
                name: float(value)
                for name, value in zip(FORCE_COLUMNS, np.abs(values).max(axis=0))
            },
        }


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ModelPlaceCaptureNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
