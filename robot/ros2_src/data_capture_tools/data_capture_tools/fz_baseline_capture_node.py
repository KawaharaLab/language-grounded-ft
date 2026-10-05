"""Real-robot lift-and-lower evaluation using signed Fz zero crossing."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String

from .key_motion_capture_node import KeyInput, KeyMotionCaptureNode, MotionState


FORCE_COLUMNS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "left_tx",
    "left_ty",
    "left_tz",
    "right_fx",
    "right_fy",
    "right_fz",
    "right_tx",
    "right_ty",
    "right_tz",
]


@dataclass(frozen=True)
class PairedForceSample:
    """A timestamp-aligned bilateral force sample."""

    sequence: int
    stamp_ns: int
    left_stamp_ns: int
    right_stamp_ns: int
    pair_skew_ms: float
    receipt_monotonic_ns: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class FzDecision:
    """One online pre-grasp-tared signed-Fz threshold result."""

    sample: PairedForceSample
    raw_signed_fz_n: float
    corrected_signed_fz_n: float
    threshold_n: float
    threshold_met: bool
    release_eligible: bool
    consecutive_count: int


class FzBaselineCaptureNode(KeyMotionCaptureNode):
    """Stop after three signed-Fz crossings relative to the pre-grasp tare."""

    _SCORE_FIELDS = [
        "episode_id",
        "sequence",
        "stamp_ns",
        "receipt_monotonic_ns",
        "left_stamp_ns",
        "right_stamp_ns",
        "pair_skew_ms",
        "left_fz_n",
        "right_fz_n",
        "raw_signed_fz_n",
        "left_offset_fz_n",
        "right_offset_fz_n",
        "corrected_signed_fz_n",
        "threshold_n",
        "threshold_met",
        "release_eligible",
        "consecutive_count",
        "triggered",
        "floor_contact_stamp_ns",
        "sample_after_contact_ms",
    ]

    def __init__(self) -> None:
        super().__init__()
        if not self._object_name:
            raise ValueError("object_name is required for Fz baseline evaluation")

        self._run_suffix = str(
            self.declare_parameter(
                "fz_baseline_run_suffix", "fz_baseline"
            ).value
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self._run_suffix):
            raise ValueError(
                "fz_baseline_run_suffix must contain only letters, digits, "
                "underscores, or hyphens"
            )
        # Keep each baseline profile separate from PatchTST and from other Fz
        # profiles without changing the actual object label in metadata.
        condition_component = (
            f"{self._surface_condition}_" if self._surface_condition else ""
        )
        self.config.task_name = (
            f"place_{self._object_name}_{condition_component}{self._run_suffix}"
        )

        self._calibration_sec = float(
            self.declare_parameter("fz_baseline_calibration_sec", 1.0).value
        )
        self._minimum_calibration_samples = int(
            self.declare_parameter(
                "fz_baseline_min_calibration_samples", 50
            ).value
        )
        self._threshold_n = float(
            self.declare_parameter("fz_baseline_threshold_n", 0.0).value
        )
        self._required_consecutive = int(
            self.declare_parameter(
                "fz_baseline_required_consecutive", 3
            ).value
        )
        self._stride_samples = int(
            self.declare_parameter("fz_baseline_stride_samples", 5).value
        )
        self._release_min_descent_sec = float(
            self.declare_parameter(
                "fz_baseline_release_min_descent_sec", 2.0
            ).value
        )
        self._max_wait_after_contact_sec = float(
            self.declare_parameter(
                "fz_baseline_max_wait_after_contact_sec", 0.8
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
        coordinate_sign = list(
            self.declare_parameter(
                "fz_baseline_coordinate_sign",
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
        if self._calibration_sec <= 0.0:
            raise ValueError("fz_baseline_calibration_sec must be positive")
        if self._minimum_calibration_samples <= 0:
            raise ValueError(
                "fz_baseline_min_calibration_samples must be positive"
            )
        if self._required_consecutive <= 0:
            raise ValueError("fz_baseline_required_consecutive must be positive")
        if self._stride_samples <= 0:
            raise ValueError("fz_baseline_stride_samples must be positive")
        if self._release_min_descent_sec < 0.0:
            raise ValueError(
                "fz_baseline_release_min_descent_sec must be non-negative"
            )
        if self._max_wait_after_contact_sec <= 0.0:
            raise ValueError(
                "fz_baseline_max_wait_after_contact_sec must be positive"
            )
        if self._enter_reaction_compensation_sec < 0.0:
            raise ValueError("enter_reaction_compensation_sec must be non-negative")
        if self._force_pair_max_skew_ms <= 0.0:
            raise ValueError("force_pair_max_skew_ms must be positive")
        if len(coordinate_sign) != 12:
            raise ValueError("fz_baseline_coordinate_sign must have 12 values")
        self._coordinate_sign = np.asarray(coordinate_sign, dtype=np.float32)
        decision_period_ms = (
            self._stride_samples * self._control_period_sec * 1000.0
        )
        support_text = (
            f" on a {self._surface_condition} support"
            if self._surface_condition
            else ""
        )
        self.config.prompt = (
            f"place {self._object_name}{support_text}: release after "
            f"corrected bilateral signed Fz < {self._threshold_n:.3f} N for "
            f"{self._required_consecutive} consecutive "
            f"decisions evaluated about every {decision_period_ms:.0f} ms."
        )

        self._score_topic = str(
            self.declare_parameter(
                "fz_baseline_score_topic", "/data_capture/fz_baseline_scores"
            ).value
        )
        self._score_pub = self.create_publisher(String, self._score_topic, 10)
        self._paired_samples: deque[PairedForceSample] = deque(maxlen=8192)
        self._last_paired_stamp_ns: dict[str, Optional[int]] = {
            "left": None,
            "right": None,
        }
        self._pair_sequence = 0
        self._last_evaluated_sequence: Optional[int] = None
        self._left_offset_fz_n: Optional[float] = None
        self._right_offset_fz_n: Optional[float] = None
        self._calibration_start_stamp_ns: Optional[int] = None
        self._calibration_end_stamp_ns: Optional[int] = None
        self._calibration_sample_count = 0
        self._consecutive_count = 0
        self._pending_trigger: Optional[FzDecision] = None
        self._latest_decision: Optional[FzDecision] = None
        self._enter_input: Optional[KeyInput] = None
        self._floor_contact: Optional[KeyInput] = None
        self._run_score_csv_path: Optional[Path] = None
        self._episode_score_csv_path: Optional[Path] = None
        self._evaluation_context: Optional[
            tuple[int, int, Optional[FzDecision]]
        ] = None
        self._completion_status: Optional[str] = None
        self._completion_reason: Optional[str] = None

    def _ready_message(self, sequence: str) -> str:
        return (
            "READY (signed-Fz baseline): Right arrow records a pre-grasp tare "
            "with the gripper open and no object held, "
            f"closes, {sequence}, ignores decisions for the first "
            f"{self._release_min_descent_sec:.1f} s of descent, and stops/opens "
            f"after corrected left+right Fz < {self._threshold_n:.3f} N for "
            f"{self._required_consecutive} consecutive decisions evaluated every "
            f"{self._stride_samples} paired force samples. Enter only marks "
            "ground-truth floor contact. Up aborts, Space stops motion, and q exits."
        )

    def _prepare_run(self) -> None:
        super()._prepare_run()
        if self._run_dir is None:
            return
        self._run_score_csv_path = (
            self._run_dir / "csv" / "fz_baseline_scores_all.csv"
        )
        self._initialize_score_csv(self._run_score_csv_path)
        (self._run_dir / "fz_baseline_evaluation_config.json").write_text(
            json.dumps(self._detector_metadata(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _start_episode(self, trigger: KeyInput) -> None:
        self._reset_baseline_episode()
        super()._start_episode(trigger)
        if self._session_dir is None:
            return
        self._episode_score_csv_path = (
            self._session_dir / "csv" / "fz_baseline_scores.csv"
        )
        self._initialize_score_csv(self._episode_score_csv_path)
        if self._run_dir is not None:
            source = self._run_dir / "fz_baseline_evaluation_config.json"
            destination = self._session_dir / "fz_baseline_evaluation_config.json"
            destination.write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )
        try:
            self._calibrate_pregrasp_offsets()
        except RuntimeError:
            self._discard_active_episode("fz_baseline_calibration_failed")
            raise

    def _clear_active_episode(self) -> None:
        super()._clear_active_episode()
        self._episode_score_csv_path = None

    def _reset_baseline_episode(self) -> None:
        self._paired_samples.clear()
        self._last_paired_stamp_ns = {"left": None, "right": None}
        self._pair_sequence = 0
        self._last_evaluated_sequence = None
        self._left_offset_fz_n = None
        self._right_offset_fz_n = None
        self._calibration_start_stamp_ns = None
        self._calibration_end_stamp_ns = None
        self._calibration_sample_count = 0
        self._consecutive_count = 0
        self._pending_trigger = None
        self._latest_decision = None
        self._enter_input = None
        self._floor_contact = None
        self._evaluation_context = None
        self._completion_status = None
        self._completion_reason = None

    def _calibrate_pregrasp_offsets(self) -> None:
        end_ns = int(self.get_clock().now().nanoseconds)
        start_ns = end_ns - int(self._calibration_sec * 1.0e9)
        samples = [
            sample
            for sample in self._paired_samples
            if start_ns <= sample.stamp_ns <= end_ns
        ]
        if len(samples) < self._minimum_calibration_samples:
            raise RuntimeError(
                "Pre-grasp Fz calibration received only "
                f"{len(samples)} paired samples; at least "
                f"{self._minimum_calibration_samples} are required"
            )
        self._left_offset_fz_n = statistics.median(
            sample.values[2] for sample in samples
        )
        self._right_offset_fz_n = statistics.median(
            sample.values[8] for sample in samples
        )
        self._calibration_start_stamp_ns = samples[0].stamp_ns
        self._calibration_end_stamp_ns = samples[-1].stamp_ns
        self._calibration_sample_count = len(samples)
        self._write_calibration_csv(samples)
        self.get_logger().info(
            "Pre-grasp Fz tare ready: "
            f"left={self._left_offset_fz_n:.3f} N, "
            f"right={self._right_offset_fz_n:.3f} N, "
            f"sum={self._offset_sum_n():.3f} N from {len(samples)} samples."
        )

    def _force_cb(self, side: str, message: WrenchStamped) -> None:
        super()._force_cb(side, message)
        if self._force_recording_active:
            self._try_pair_force_samples()

    def _try_pair_force_samples(self) -> None:
        if not self._force_samples["left"] or not self._force_samples["right"]:
            return
        left = self._force_samples["left"][-1]
        right = self._force_samples["right"][-1]
        if (
            self._last_paired_stamp_ns["left"] is not None
            and left.stamp_ns <= self._last_paired_stamp_ns["left"]
        ) or (
            self._last_paired_stamp_ns["right"] is not None
            and right.stamp_ns <= self._last_paired_stamp_ns["right"]
        ):
            return
        skew_ms = abs(left.stamp_ns - right.stamp_ns) / 1.0e6
        if skew_ms > self._force_pair_max_skew_ms:
            return

        transformed = np.asarray(
            left.values + right.values, dtype=np.float32
        ) * self._coordinate_sign
        self._pair_sequence += 1
        sample = PairedForceSample(
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
        self._paired_samples.append(sample)
        self._last_paired_stamp_ns["left"] = left.stamp_ns
        self._last_paired_stamp_ns["right"] = right.stamp_ns
        self._maybe_evaluate_sample(sample)

    def _maybe_evaluate_sample(self, sample: PairedForceSample) -> None:
        if (
            self._state != MotionState.LOWERING
            or self._active_episode_id is None
            or self._left_offset_fz_n is None
            or self._right_offset_fz_n is None
        ):
            return
        if (
            self._last_evaluated_sequence is not None
            and sample.sequence - self._last_evaluated_sequence
            < self._stride_samples
        ):
            return
        self._last_evaluated_sequence = sample.sequence
        raw_signed_fz_n = sample.values[2] + sample.values[8]
        corrected_signed_fz_n = raw_signed_fz_n - self._offset_sum_n()
        if self._lower_started_monotonic_ns is None:
            return
        descent_elapsed_sec = (
            sample.receipt_monotonic_ns - self._lower_started_monotonic_ns
        ) / 1.0e9
        threshold_met = corrected_signed_fz_n < self._threshold_n
        release_eligible = bool(
            descent_elapsed_sec >= self._release_min_descent_sec
        )
        self._consecutive_count = (
            self._consecutive_count + 1
            if threshold_met and release_eligible
            else 0
        )
        decision = FzDecision(
            sample=sample,
            raw_signed_fz_n=raw_signed_fz_n,
            corrected_signed_fz_n=corrected_signed_fz_n,
            threshold_n=self._threshold_n,
            threshold_met=threshold_met,
            release_eligible=release_eligible,
            consecutive_count=self._consecutive_count,
        )
        self._latest_decision = decision
        triggered = self._consecutive_count >= self._required_consecutive
        self._write_score_result(decision, triggered)
        if triggered and self._pending_trigger is None:
            self._pending_trigger = decision

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
                self.get_logger().info(
                    "Enter recorded; estimated floor contact is "
                    f"{self._enter_reaction_compensation_sec * 1000.0:.1f} ms "
                    "earlier. Signed-Fz remains in control of stopping."
                )
                if self._evaluation_context is not None:
                    halt_ros_ns, halt_monotonic_ns, decision = (
                        self._evaluation_context
                    )
                    outcome, status = self._classify_detection(decision)
                    self._completion_status = status
                    self._completion_reason = outcome
                    self._write_baseline_evaluation(
                        outcome,
                        status,
                        halt_ros_ns,
                        halt_monotonic_ns,
                        decision,
                    )
            else:
                self._queue_key_event(event, state, state)
                self.get_logger().warning("Duplicate floor-contact Enter ignored.")
            return
        super()._handle_key(event)

    def _control_tick(self) -> None:
        now_ns = time.perf_counter_ns()
        if self._state == MotionState.LOWERING:
            if self._pending_trigger is not None:
                decision = self._pending_trigger
                self._pending_trigger = None
                self._finish_baseline_episode(decision)
                return
            if (
                self._floor_contact is not None
                and now_ns - self._floor_contact.monotonic_ns
                >= int(self._max_wait_after_contact_sec * 1.0e9)
            ):
                self._finish_without_trigger(
                    "fz_baseline_timeout_after_contact"
                )
                return
            if (
                self._lower_started_monotonic_ns is not None
                and now_ns - self._lower_started_monotonic_ns
                >= int(self._max_lower_duration_sec * 1.0e9)
            ):
                self._finish_without_trigger(
                    "max_lower_duration_before_fz_detection"
                )
                return
        super()._control_tick()

    def _classify_detection(
        self, decision: Optional[FzDecision]
    ) -> tuple[str, str]:
        if decision is None:
            return "fz_baseline_not_triggered", "failure"
        if self._floor_contact is None:
            return "fz_baseline_detected_before_enter", "failure"
        if decision.sample.stamp_ns < self._floor_contact.ros_stamp_ns:
            return "fz_baseline_detected_before_floor_contact", "failure"
        return "fz_baseline_detected_after_floor_contact", "success"

    def _finish_baseline_episode(self, decision: FzDecision) -> None:
        halt_monotonic_ns = time.perf_counter_ns()
        halt_ros_ns = int(self.get_clock().now().nanoseconds)
        self._publish_motion_halt()
        self._lower_ended_monotonic_ns = halt_monotonic_ns
        previous = self._state.value
        self._state = MotionState.FINALIZING_CAPTURE
        self._queue_system_event(
            "fz_zero_crossing_detected",
            previous,
            self._state.value,
        )
        outcome, status = self._classify_detection(decision)
        self._completion_status = status
        self._completion_reason = outcome
        self._write_baseline_evaluation(
            outcome,
            status,
            halt_ros_ns,
            halt_monotonic_ns,
            decision,
        )
        self.get_logger().info(
            "Signed-Fz threshold detected: stopping and opening "
            f"(corrected sum={decision.corrected_signed_fz_n:.3f} N, "
            f"threshold={decision.threshold_n:.3f} N, "
            f"streak={decision.consecutive_count})."
        )
        self._open_gripper_and_start_return(
            reference_monotonic_ns=halt_monotonic_ns,
            complete_episode_on_open=True,
            episode_status=status,
            episode_reason=outcome,
        )

    def _finish_without_trigger(self, reason: str) -> None:
        halt_monotonic_ns = time.perf_counter_ns()
        halt_ros_ns = int(self.get_clock().now().nanoseconds)
        self._publish_motion_halt()
        self._lower_ended_monotonic_ns = halt_monotonic_ns
        previous = self._state.value
        self._state = MotionState.FINALIZING_CAPTURE
        self._queue_system_event(reason, previous, self._state.value)
        self._completion_status = "failure"
        self._completion_reason = reason
        self._write_baseline_evaluation(
            reason, "failure", halt_ros_ns, halt_monotonic_ns, None
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

    def _complete_active_episode(
        self,
        status_name: str = "success",
        reason: Optional[str] = None,
    ) -> None:
        if self._completion_status is not None:
            status_name = self._completion_status
        if self._completion_reason is not None:
            reason = self._completion_reason
        # Backfilling rewrites the cumulative run CSV. Defer it until physical
        # gripper opening has completed so file I/O can never stall online force
        # callbacks or the descent stop decision.
        if self._floor_contact is not None:
            self._backfill_score_contact_columns(self._floor_contact)
        super()._complete_active_episode(status_name=status_name, reason=reason)

    def _detector_metadata(self) -> dict:
        return {
            "object_name": self._object_name,
            "surface_condition": self._surface_condition or None,
            "task_name": self.config.task_name,
            "run_suffix": self._run_suffix,
            "detector": "signed_fz_pregrasp_tared_zero_crossing",
            "formula": "(left_fz-left_pregrasp_median) + "
            "(right_fz-right_pregrasp_median) < threshold_n",
            "threshold_n": self._threshold_n,
            "score_stride_samples": self._stride_samples,
            "required_consecutive_decisions": self._required_consecutive,
            "calibration_sec": self._calibration_sec,
            "minimum_calibration_samples": self._minimum_calibration_samples,
            "calibration_state": "gripper_open_before_gripper_close_start",
            "zero_reference": (
                "per-episode median of each sensor Fz while the gripper is open "
                "and no object is held"
            ),
            "release_minimum_descent_sec": self._release_min_descent_sec,
            "max_wait_after_estimated_contact_sec": (
                self._max_wait_after_contact_sec
            ),
            "enter_reaction_compensation_sec": (
                self._enter_reaction_compensation_sec
            ),
            "force_pair_max_skew_ms": self._force_pair_max_skew_ms,
            "coordinate_transform": "sim_local_rz180",
            "coordinate_sign": self._coordinate_sign.tolist(),
            "score_topic": self._score_topic,
        }

    def _offset_sum_n(self) -> float:
        if self._left_offset_fz_n is None or self._right_offset_fz_n is None:
            raise RuntimeError("Fz baseline offsets are not calibrated")
        return self._left_offset_fz_n + self._right_offset_fz_n

    def _initialize_score_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._SCORE_FIELDS).writeheader()

    def _write_score_result(
        self, decision: FzDecision, triggered: bool
    ) -> None:
        contact = self._floor_contact
        sample = decision.sample
        row = {
            "episode_id": self._active_episode_id,
            "sequence": sample.sequence,
            "stamp_ns": sample.stamp_ns,
            "receipt_monotonic_ns": sample.receipt_monotonic_ns,
            "left_stamp_ns": sample.left_stamp_ns,
            "right_stamp_ns": sample.right_stamp_ns,
            "pair_skew_ms": sample.pair_skew_ms,
            "left_fz_n": sample.values[2],
            "right_fz_n": sample.values[8],
            "raw_signed_fz_n": decision.raw_signed_fz_n,
            "left_offset_fz_n": self._left_offset_fz_n,
            "right_offset_fz_n": self._right_offset_fz_n,
            "corrected_signed_fz_n": decision.corrected_signed_fz_n,
            "threshold_n": decision.threshold_n,
            "threshold_met": decision.threshold_met,
            "release_eligible": decision.release_eligible,
            "consecutive_count": decision.consecutive_count,
            "triggered": triggered,
            "floor_contact_stamp_ns": contact.ros_stamp_ns if contact else None,
            "sample_after_contact_ms": (
                (sample.stamp_ns - contact.ros_stamp_ns) / 1.0e6
                if contact
                else None
            ),
        }
        for path in (self._run_score_csv_path, self._episode_score_csv_path):
            if path is None:
                continue
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._SCORE_FIELDS).writerow(row)
        self._score_pub.publish(
            String(data=json.dumps(row, ensure_ascii=False, allow_nan=False))
        )

    def _backfill_score_contact_columns(self, contact: KeyInput) -> None:
        if self._active_episode_id is None:
            return
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
                    stamp_ns = int(row["stamp_ns"])
                except (KeyError, TypeError, ValueError):
                    continue
                if episode_id != self._active_episode_id:
                    continue
                row["floor_contact_stamp_ns"] = str(contact.ros_stamp_ns)
                row["sample_after_contact_ms"] = str(
                    (stamp_ns - contact.ros_stamp_ns) / 1.0e6
                )
                changed = True
            if changed:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

    def _write_calibration_csv(self, samples: list[PairedForceSample]) -> None:
        if self._session_dir is None:
            return
        path = self._session_dir / "csv" / "fz_baseline_calibration.csv"
        fields = [
            "sequence",
            "stamp_ns",
            "left_stamp_ns",
            "right_stamp_ns",
            "pair_skew_ms",
            "left_fz_n",
            "right_fz_n",
            "raw_signed_fz_n",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample in samples:
                writer.writerow(
                    {
                        "sequence": sample.sequence,
                        "stamp_ns": sample.stamp_ns,
                        "left_stamp_ns": sample.left_stamp_ns,
                        "right_stamp_ns": sample.right_stamp_ns,
                        "pair_skew_ms": sample.pair_skew_ms,
                        "left_fz_n": sample.values[2],
                        "right_fz_n": sample.values[8],
                        "raw_signed_fz_n": sample.values[2] + sample.values[8],
                    }
                )

    def _write_baseline_evaluation(
        self,
        outcome: str,
        status: str,
        halt_ros_ns: int,
        halt_monotonic_ns: int,
        decision: Optional[FzDecision],
    ) -> None:
        if self._session_dir is None:
            return
        self._evaluation_context = (halt_ros_ns, halt_monotonic_ns, decision)
        contact = self._floor_contact
        interval: list[PairedForceSample] = []
        if contact is not None:
            start_ns = min(contact.ros_stamp_ns, halt_ros_ns)
            end_ns = max(contact.ros_stamp_ns, halt_ros_ns)
            interval = [
                sample
                for sample in self._paired_samples
                if start_ns <= sample.stamp_ns <= end_ns
            ]
        force_csv = self._session_dir / "csv" / "contact_to_detection_force.csv"
        self._write_force_interval(force_csv, interval)
        metrics = {
            "episode_id": self._active_episode_id,
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
            "floor_contact_monotonic_ns": (
                contact.monotonic_ns if contact else None
            ),
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
            "baseline_triggered": decision is not None,
            "consecutive_count_at_halt": self._consecutive_count,
            "calibration": {
                "start_stamp_ns": self._calibration_start_stamp_ns,
                "end_stamp_ns": self._calibration_end_stamp_ns,
                "sample_count": self._calibration_sample_count,
                "left_offset_fz_n": self._left_offset_fz_n,
                "right_offset_fz_n": self._right_offset_fz_n,
                "offset_sum_n": self._offset_sum_n(),
            },
            "contact_to_detection_force_csv": str(force_csv),
            "force_interval": self._force_interval_metrics(interval),
            "detector": self._detector_metadata(),
            "decision": None,
        }
        if decision is not None:
            sample = decision.sample
            metrics["decision"] = {
                "sequence": sample.sequence,
                "sample_stamp_ns": sample.stamp_ns,
                "sample_receipt_monotonic_ns": sample.receipt_monotonic_ns,
                "left_fz_n": sample.values[2],
                "right_fz_n": sample.values[8],
                "raw_signed_fz_n": decision.raw_signed_fz_n,
                "corrected_signed_fz_n": decision.corrected_signed_fz_n,
                "threshold_n": decision.threshold_n,
                "consecutive_count": decision.consecutive_count,
                "sample_after_contact_ms": (
                    (sample.stamp_ns - contact.ros_stamp_ns) / 1.0e6
                    if contact
                    else None
                ),
                "halt_after_sample_receipt_ms": (
                    halt_monotonic_ns - sample.receipt_monotonic_ns
                )
                / 1.0e6,
            }
        path = self._session_dir / "fz_baseline_evaluation.json"
        path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._score_pub.publish(
            String(
                data=json.dumps(
                    {"type": "episode_result", **metrics},
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        )

    @staticmethod
    def _write_force_interval(
        path: Path, samples: list[PairedForceSample]
    ) -> None:
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
                "combined_force_norm_mean_n": None,
                "combined_force_norm_max_n": None,
                "combined_force_impulse_ns": None,
            }
        values = np.asarray(
            [sample.values for sample in samples], dtype=np.float64
        )
        stamps_sec = np.asarray(
            [sample.stamp_ns for sample in samples], dtype=np.float64
        )
        stamps_sec = (stamps_sec - stamps_sec[0]) / 1.0e9
        left_force = np.linalg.norm(values[:, 0:3], axis=1)
        right_force = np.linalg.norm(values[:, 6:9], axis=1)
        combined = left_force + right_force
        impulse = (
            float(np.trapz(combined, stamps_sec))
            if len(samples) > 1
            else 0.0
        )
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
                for name, value in zip(
                    FORCE_COLUMNS, np.abs(values).max(axis=0)
                )
            },
        }


def main(args: Optional[list[str]] = None) -> None:
    """Run the signed-Fz baseline evaluation node."""
    rclpy.init(args=args)
    node = FzBaselineCaptureNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
