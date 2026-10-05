"""Probe fixed and wobbling supports, then release on the more fixed-like one."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data

from .key_motion_capture_node import KeyInput, MotionState
from .model_place_capture_node import InferenceResult, ModelPlaceCaptureNode
from .workspace_paths import resolve_workspace_path


def select_more_fixed_surface(scores: dict[str, float]) -> str:
    """Select by direct pairwise preference, without a learned threshold."""
    if set(scores) != {"fixed", "wobble"}:
        raise ValueError("Exactly one fixed and one wobble score are required")
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("Surface scores must be finite")
    if math.isclose(scores["fixed"], scores["wobble"], abs_tol=1.0e-9):
        raise ValueError("Surface scores are exactly tied")
    return max(scores, key=scores.get)


def transform_pose_error_to_command_xy(
    error_xy: np.ndarray,
    frame_sign: np.ndarray,
) -> np.ndarray:
    """Rotate an axis-aligned XY error into the Servo command frame."""
    error = np.asarray(error_xy, dtype=np.float64)
    sign = np.asarray(frame_sign, dtype=np.float64)
    if error.shape != (2,) or sign.shape != (2,):
        raise ValueError("XY error and frame sign must each have two values")
    if not np.all(np.isfinite(error)) or not np.all(np.isin(sign, [-1.0, 1.0])):
        raise ValueError("XY error must be finite and frame signs must be +/-1")
    return error * sign


def slew_limit_vector(
    previous: np.ndarray,
    target: np.ndarray,
    rate_limit: np.ndarray,
    dt_sec: float,
) -> np.ndarray:
    """Apply a per-axis first-derivative limit to a command vector."""
    previous_value = np.asarray(previous, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    limit_value = np.asarray(rate_limit, dtype=np.float64)
    if not (
        previous_value.shape == target_value.shape == limit_value.shape
        and previous_value.ndim == 1
    ):
        raise ValueError("Slew-limit vectors must have the same 1-D shape")
    if (
        not np.all(np.isfinite(previous_value))
        or not np.all(np.isfinite(target_value))
        or not np.all(np.isfinite(limit_value))
        or np.any(limit_value <= 0.0)
        or not math.isfinite(dt_sec)
        or dt_sec <= 0.0
    ):
        raise ValueError("Slew-limit inputs must be finite and positive")
    maximum_step = limit_value * dt_sec
    return previous_value + np.clip(
        target_value - previous_value,
        -maximum_step,
        maximum_step,
    )


class TwoSurfaceSelectionCaptureNode(ModelPlaceCaptureNode):
    """Touch two supports and select one by direct prototype comparison."""

    def __init__(self) -> None:
        """Initialize the selection workflow and its motion constraints."""
        super().__init__()

        fixed_prototype = resolve_workspace_path(
            str(self.declare_parameter("fixed_surface_prototype_path", "").value)
        )
        wobbling_prototype = resolve_workspace_path(
            str(
                self.declare_parameter(
                    "wobbling_surface_prototype_path", ""
                ).value
            )
        )
        self._detector.configure_surface_prototypes(
            fixed_surface_path=fixed_prototype,
            wobbling_surface_path=wobbling_prototype,
        )

        self._surface_targets = {
            "fixed": np.asarray(
                [
                    float(
                        self.declare_parameter(
                            "fixed_surface_target_x_m", math.nan
                        ).value
                    ),
                    float(
                        self.declare_parameter(
                            "fixed_surface_target_y_m", math.nan
                        ).value
                    ),
                ],
                dtype=np.float64,
            ),
            "wobble": np.asarray(
                [
                    float(
                        self.declare_parameter(
                            "wobble_surface_target_x_m", math.nan
                        ).value
                    ),
                    float(
                        self.declare_parameter(
                            "wobble_surface_target_y_m", math.nan
                        ).value
                    ),
                ],
                dtype=np.float64,
            ),
        }
        self._horizontal_twist_command = float(
            self.declare_parameter("horizontal_twist_command", 0.8).value
        )
        self._horizontal_min_twist_command = float(
            self.declare_parameter(
                "horizontal_min_twist_command", 0.15
            ).value
        )
        self._horizontal_position_gain = float(
            self.declare_parameter("horizontal_position_gain", 20.0).value
        )
        self._horizontal_position_tolerance_m = float(
            self.declare_parameter(
                "horizontal_position_tolerance_m", 0.002
            ).value
        )
        self._surface_move_settle_sec = float(
            self.declare_parameter("surface_move_settle_sec", 0.5).value
        )
        self._probe_descent_start_height_above_home_m = float(
            self.declare_parameter(
                "probe_descent_start_height_above_home_m", 0.0757
            ).value
        )
        self._probe_vertical_twist_command = float(
            self.declare_parameter(
                "probe_vertical_twist_command", 1.0
            ).value
        )
        self._vertical_position_tolerance_m = float(
            self.declare_parameter(
                "vertical_position_tolerance_m", 0.002
            ).value
        )
        self._maximum_vertical_move_sec = float(
            self.declare_parameter(
                "maximum_vertical_move_sec", 10.0
            ).value
        )
        self._probe_enter_grace_sec = float(
            self.declare_parameter("probe_enter_grace_sec", 0.7).value
        )
        self._minimum_surface_separation_m = float(
            self.declare_parameter(
                "minimum_surface_separation_m", 0.02
            ).value
        )
        self._maximum_horizontal_move_sec = float(
            self.declare_parameter("maximum_horizontal_move_sec", 60.0).value
        )
        self._horizontal_stall_timeout_sec = float(
            self.declare_parameter(
                "horizontal_stall_timeout_sec", 5.0
            ).value
        )
        self._horizontal_min_progress_m = float(
            self.declare_parameter(
                "horizontal_min_progress_m", 0.002
            ).value
        )
        self._horizontal_status_period_sec = float(
            self.declare_parameter(
                "horizontal_status_period_sec", 5.0
            ).value
        )
        self._tcp_pose_topic = str(
            self.declare_parameter(
                "tcp_pose_topic", "/tcp_pose_broadcaster/pose"
            ).value
        )
        self._tcp_pose_timeout_sec = float(
            self.declare_parameter("tcp_pose_timeout_sec", 0.25).value
        )
        self._expected_tcp_pose_frame = str(
            self.declare_parameter("expected_tcp_pose_frame", "base").value
        )
        self._tcp_pose_xy_to_command_sign = np.asarray(
            self.declare_parameter(
                "tcp_pose_xy_to_command_sign", [-1.0, -1.0]
            ).value,
            dtype=np.float64,
        )
        self._enforce_exclusive_control_graph = bool(
            self.declare_parameter(
                "enforce_exclusive_control_graph", True
            ).value
        )
        self._control_graph_discovery_wait_sec = float(
            self.declare_parameter(
                "control_graph_discovery_wait_sec", 2.0
            ).value
        )
        self._control_graph_check_period_sec = float(
            self.declare_parameter(
                "control_graph_check_period_sec", 0.5
            ).value
        )
        self._cartesian_xy_command_slew_per_sec = float(
            self.declare_parameter(
                "cartesian_xy_command_slew_per_sec", 1.5
            ).value
        )
        self._cartesian_z_command_slew_per_sec = float(
            self.declare_parameter(
                "cartesian_z_command_slew_per_sec", 2.0
            ).value
        )

        separation = float(
            np.linalg.norm(
                self._surface_targets["fixed"]
                - self._surface_targets["wobble"]
            )
        )
        if not all(
            np.all(np.isfinite(target))
            for target in self._surface_targets.values()
        ):
            raise ValueError(
                "Finite absolute base-frame X/Y targets are required for "
                "both surfaces"
            )
        if separation < self._minimum_surface_separation_m:
            raise ValueError(
                "The fixed and wobble surface targets must be at least "
                f"{self._minimum_surface_separation_m:.3f} m apart; "
                "set the measured base-frame absolute X/Y targets"
            )
        for name, value in (
            ("horizontal_twist_command", self._horizontal_twist_command),
            ("horizontal_min_twist_command", self._horizontal_min_twist_command),
            ("horizontal_position_gain", self._horizontal_position_gain),
            ("horizontal_position_tolerance_m", self._horizontal_position_tolerance_m),
            ("maximum_horizontal_move_sec", self._maximum_horizontal_move_sec),
            (
                "probe_descent_start_height_above_home_m",
                self._probe_descent_start_height_above_home_m,
            ),
            ("probe_vertical_twist_command", self._probe_vertical_twist_command),
            ("vertical_position_tolerance_m", self._vertical_position_tolerance_m),
            ("maximum_vertical_move_sec", self._maximum_vertical_move_sec),
            ("probe_enter_grace_sec", self._probe_enter_grace_sec),
            ("horizontal_stall_timeout_sec", self._horizontal_stall_timeout_sec),
            ("horizontal_min_progress_m", self._horizontal_min_progress_m),
            ("horizontal_status_period_sec", self._horizontal_status_period_sec),
            ("tcp_pose_timeout_sec", self._tcp_pose_timeout_sec),
            (
                "control_graph_discovery_wait_sec",
                self._control_graph_discovery_wait_sec,
            ),
            (
                "control_graph_check_period_sec",
                self._control_graph_check_period_sec,
            ),
            (
                "cartesian_xy_command_slew_per_sec",
                self._cartesian_xy_command_slew_per_sec,
            ),
            (
                "cartesian_z_command_slew_per_sec",
                self._cartesian_z_command_slew_per_sec,
            ),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self._surface_move_settle_sec < 0.0:
            raise ValueError("surface_move_settle_sec must be non-negative")
        if self._horizontal_min_twist_command > self._horizontal_twist_command:
            raise ValueError(
                "horizontal_min_twist_command must not exceed "
                "horizontal_twist_command"
            )
        if self._tcp_pose_xy_to_command_sign.shape != (2,) or not np.all(
            np.isin(self._tcp_pose_xy_to_command_sign, [-1.0, 1.0])
        ):
            raise ValueError(
                "tcp_pose_xy_to_command_sign must contain two values, each "
                "equal to -1 or 1"
            )

        self._workflow_started = False
        self._probe_order: list[str] = []
        self._active_phase = "idle"
        self._active_surface: Optional[str] = None
        self._next_descent_phase: Optional[str] = None
        self._next_descent_surface: Optional[str] = None
        self._current_xy = np.zeros(2, dtype=np.float64)
        self._home_xy: Optional[np.ndarray] = None
        self._home_xyz: Optional[np.ndarray] = None
        self._tcp_pose_xy: Optional[np.ndarray] = None
        self._tcp_pose_xyz: Optional[np.ndarray] = None
        self._tcp_pose_frame: Optional[str] = None
        self._tcp_pose_receipt_monotonic_ns: Optional[int] = None
        self._descent_start_z_m: Optional[float] = None
        self._move_target_xy = np.zeros(2, dtype=np.float64)
        self._horizontal_command_xy = np.zeros(2, dtype=np.float64)
        self._horizontal_best_distance_m: Optional[float] = None
        self._horizontal_last_progress_monotonic_ns: Optional[int] = None
        self._horizontal_initial_distance_m: Optional[float] = None
        self._horizontal_last_status_monotonic_ns: Optional[int] = None
        self._custom_deadline_ns: Optional[int] = None
        self._pending_probe_result: Optional[InferenceResult] = None
        self._pending_probe_halt_ros_ns: Optional[int] = None
        self._pending_probe_halt_monotonic_ns: Optional[int] = None
        self._probe_records: list[dict] = []
        self._selected_surface: Optional[str] = None
        self._selection_scores: dict[str, float] = {}
        self._application_finished = False
        self._smoothed_twist_xyz = np.zeros(3, dtype=np.float64)
        self._last_twist_update_monotonic_ns = time.perf_counter_ns()
        self._last_control_graph_check_monotonic_ns = 0
        self._tcp_pose_subscription = self.create_subscription(
            PoseStamped,
            self._tcp_pose_topic,
            self._tcp_pose_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Two-surface selector ready: direct argmax of "
            "fixed_cosine - wobble_cosine; "
            f"fixed_base_xy={self._surface_targets['fixed'].tolist()}, "
            f"wobble_base_xy={self._surface_targets['wobble'].tolist()}, "
            "continuous closed-loop pick lift to "
            f"home Z + {self._probe_descent_start_height_above_home_m * 1000.0:.1f} mm."
        )

    def _ready_message(self, sequence: str) -> str:
        return (
            "READY (two-surface selection): Right arrow starts one trial. "
            f"The robot {sequence}, probes both supports, lifts after each "
            "probe, returns to the larger (fixed cosine - wobble cosine), "
            "touches it again, and opens. Probe order alternates every "
            "episode. Press Enter at each of the three visible contacts. "
            "Enter minus 300 ms is the estimated contact; place must be "
            "detected within 0.8 s of it. Up aborts and returns, Space stops "
            "motion, q exits."
        )

    def _prepare_run(self) -> None:
        if self._enforce_exclusive_control_graph:
            self._validate_exclusive_control_graph()
        super()._prepare_run()
        if self._run_dir is None:
            return
        path = self._run_dir / "model_evaluation_config.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["two_surface_selection"] = {
            "selection_rule": (
                "argmax over the two probes of "
                "fixed_surface_cosine - wobbling_surface_cosine"
            ),
            "threshold": None,
            "alternate_first_surface_each_episode": True,
            "surface_target_base_xy_m": {
                name: value.tolist()
                for name, value in self._surface_targets.items()
            },
            "horizontal_twist_command": self._horizontal_twist_command,
            "horizontal_min_twist_command": (
                self._horizontal_min_twist_command
            ),
            "horizontal_position_gain": self._horizontal_position_gain,
            "horizontal_position_tolerance_m": (
                self._horizontal_position_tolerance_m
            ),
            "maximum_horizontal_move_sec": self._maximum_horizontal_move_sec,
            "horizontal_stall_timeout_sec": (
                self._horizontal_stall_timeout_sec
            ),
            "horizontal_min_progress_m": self._horizontal_min_progress_m,
            "horizontal_status_period_sec": self._horizontal_status_period_sec,
            "tcp_pose_topic": self._tcp_pose_topic,
            "tcp_pose_timeout_sec": self._tcp_pose_timeout_sec,
            "surface_move_settle_sec": self._surface_move_settle_sec,
            "probe_descent_start_height_above_home_m": (
                self._probe_descent_start_height_above_home_m
            ),
            "vertical_position_tolerance_m": (
                self._vertical_position_tolerance_m
            ),
            "expected_tcp_pose_frame": self._expected_tcp_pose_frame,
            "tcp_pose_xy_to_command_sign": (
                self._tcp_pose_xy_to_command_sign.tolist()
            ),
            "exclusive_control_graph_enforced": (
                self._enforce_exclusive_control_graph
            ),
            "control_graph_discovery_wait_sec": (
                self._control_graph_discovery_wait_sec
            ),
            "control_graph_check_period_sec": (
                self._control_graph_check_period_sec
            ),
            "cartesian_xy_command_slew_per_sec": (
                self._cartesian_xy_command_slew_per_sec
            ),
            "cartesian_z_command_slew_per_sec": (
                self._cartesian_z_command_slew_per_sec
            ),
            "probe_enter_grace_sec": self._probe_enter_grace_sec,
            "enter_required_for_each_contact": True,
            "enter_role": (
                "ground-truth contact timestamp; detection must occur within "
                "0.8 s of Enter minus reaction compensation"
            ),
        }
        path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _start_episode(self, trigger: KeyInput) -> None:
        if self._enforce_exclusive_control_graph:
            self._assert_exclusive_control_graph()
        if not self._tcp_pose_is_fresh(trigger.monotonic_ns):
            raise RuntimeError(
                f"A fresh TCP pose is unavailable on {self._tcp_pose_topic}"
            )
        assert self._tcp_pose_xy is not None
        episode_home_xy = self._tcp_pose_xy.copy()
        assert self._tcp_pose_xyz is not None
        episode_home_xyz = self._tcp_pose_xyz.copy()
        super()._start_episode(trigger)
        self._reset_smoothed_twist()
        self._home_xy = episode_home_xy
        self._home_xyz = episode_home_xyz
        self._descent_start_z_m = (
            float(episode_home_xyz[2])
            + self._probe_descent_start_height_above_home_m
        )
        episode_id = int(self._active_episode_id or 0)
        self._probe_order = (
            ["fixed", "wobble"]
            if episode_id % 2 == 1
            else ["wobble", "fixed"]
        )
        self.get_logger().info(
            f"Episode {episode_id} probe order: "
            f"{self._probe_order[0]} -> {self._probe_order[1]}."
        )

    def _reset_model_episode(self) -> None:
        super()._reset_model_episode()
        self._workflow_started = False
        self._probe_order = []
        self._active_phase = "idle"
        self._active_surface = None
        self._next_descent_phase = None
        self._next_descent_surface = None
        self._current_xy = np.zeros(2, dtype=np.float64)
        self._home_xy = None
        self._home_xyz = None
        self._move_target_xy = np.zeros(2, dtype=np.float64)
        self._horizontal_command_xy = np.zeros(2, dtype=np.float64)
        self._horizontal_best_distance_m = None
        self._horizontal_last_progress_monotonic_ns = None
        self._horizontal_initial_distance_m = None
        self._horizontal_last_status_monotonic_ns = None
        self._custom_deadline_ns = None
        self._descent_start_z_m = None
        self._pending_probe_result = None
        self._pending_probe_halt_ros_ns = None
        self._pending_probe_halt_monotonic_ns = None
        self._probe_records = []
        self._selected_surface = None
        self._selection_scores = {}
        self._application_finished = False
        self._reset_smoothed_twist()

    def _tcp_pose_cb(self, message: PoseStamped) -> None:
        self._tcp_pose_xyz = np.asarray(
            [
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ],
            dtype=np.float64,
        )
        self._tcp_pose_xy = self._tcp_pose_xyz[:2]
        self._tcp_pose_frame = message.header.frame_id
        self._tcp_pose_receipt_monotonic_ns = time.perf_counter_ns()

    def _validate_exclusive_control_graph(self) -> None:
        """Reject competing Servo/controller publishers before any motion."""
        deadline_ns = time.perf_counter_ns() + int(
            self._control_graph_discovery_wait_sec * 1.0e9
        )
        while rclpy.ok() and time.perf_counter_ns() < deadline_ns:
            rclpy.spin_once(self, timeout_sec=0.05)
        self._assert_exclusive_control_graph()
        self._last_control_graph_check_monotonic_ns = time.perf_counter_ns()
        self.get_logger().info(
            "Exclusive control graph verified after discovery wait: one "
            "Twist input publisher, one Servo output publisher, and one "
            "robot-command publisher."
        )

    def _control_graph_publisher_counts(self) -> dict[str, int]:
        """Return publisher counts for every stage of the control path."""
        return {
            self._twist_command_topic: self.count_publishers(
                self._twist_command_topic
            ),
            self._servo_output_topic: self.count_publishers(
                self._servo_output_topic
            ),
            self._robot_command_topic: self.count_publishers(
                self._robot_command_topic
            ),
        }

    def _assert_exclusive_control_graph(self) -> None:
        """Raise if another node can command the same robot-control path."""
        counts = self._control_graph_publisher_counts()
        unexpected = {
            topic: count for topic, count in counts.items() if count != 1
        }
        if unexpected:
            summary = ", ".join(
                f"{topic}={count} publishers"
                for topic, count in unexpected.items()
            )
            raise RuntimeError(
                "Exclusive robot-control graph check failed: "
                f"{summary}. Stop any standalone teleop_servo launch and "
                "command_mux_node before starting this self-contained launch."
            )

    def _check_exclusive_control_graph_runtime(self, now_ns: int) -> None:
        """Stop if a competing publisher appears after startup."""
        if not self._enforce_exclusive_control_graph:
            return
        period_ns = int(self._control_graph_check_period_sec * 1.0e9)
        if now_ns - self._last_control_graph_check_monotonic_ns < period_ns:
            return
        self._last_control_graph_check_monotonic_ns = now_ns
        try:
            self._assert_exclusive_control_graph()
        except RuntimeError:
            self._publish_motion_halt()
            self._publish_joint_halt()
            raise

    def _reset_smoothed_twist(self) -> None:
        self._smoothed_twist_xyz = np.zeros(3, dtype=np.float64)
        self._last_twist_update_monotonic_ns = time.perf_counter_ns()

    def _publish_twist_xyz(
        self,
        x_command: float = 0.0,
        y_command: float = 0.0,
        z_command: float = 0.0,
    ) -> None:
        """Publish a slew-limited Cartesian velocity command."""
        now_ns = time.perf_counter_ns()
        elapsed_sec = (
            now_ns - self._last_twist_update_monotonic_ns
        ) / 1.0e9
        dt_sec = min(
            max(elapsed_sec, self._control_period_sec * 0.5),
            self._control_period_sec * 5.0,
        )
        target = np.asarray(
            [x_command, y_command, z_command], dtype=np.float64
        )
        rate_limit = np.asarray(
            [
                self._cartesian_xy_command_slew_per_sec,
                self._cartesian_xy_command_slew_per_sec,
                self._cartesian_z_command_slew_per_sec,
            ],
            dtype=np.float64,
        )
        self._smoothed_twist_xyz = slew_limit_vector(
            self._smoothed_twist_xyz,
            target,
            rate_limit,
            dt_sec,
        )
        self._last_twist_update_monotonic_ns = now_ns
        super()._publish_twist_xyz(
            x_command=float(self._smoothed_twist_xyz[0]),
            y_command=float(self._smoothed_twist_xyz[1]),
            z_command=float(self._smoothed_twist_xyz[2]),
        )

    def _publish_motion_halt(self) -> None:
        """Bypass smoothing for every safety or detection stop."""
        self._reset_smoothed_twist()
        super()._publish_twist_xyz(0.0, 0.0, 0.0)

    def _tcp_pose_is_fresh(self, now_ns: int) -> bool:
        return bool(
            self._tcp_pose_xy is not None
            and self._tcp_pose_xyz is not None
            and self._tcp_pose_frame == self._expected_tcp_pose_frame
            and self._tcp_pose_receipt_monotonic_ns is not None
            and now_ns - self._tcp_pose_receipt_monotonic_ns
            <= int(self._tcp_pose_timeout_sec * 1.0e9)
        )

    def _inference_phase_label(self) -> str:
        return getattr(self, "_active_phase", "idle")

    def _begin_lowering(self, now_ns: int) -> None:
        """Replace the first post-lift descent with a move to probe one."""
        if not self._workflow_started:
            if len(self._probe_order) != 2:
                raise RuntimeError("Probe order was not initialized")
            self._workflow_started = True
            self._begin_move_to_surface(
                self._probe_order[0], "probe_1", now_ns
            )
            return
        super()._begin_lowering(now_ns)

    def _begin_move_to_surface(
        self,
        surface: str,
        phase: str,
        now_ns: Optional[int] = None,
    ) -> None:
        if now_ns is None:
            now_ns = time.perf_counter_ns()
        if self._home_xy is None or not self._tcp_pose_is_fresh(now_ns):
            self._fail_application("tcp_pose_unavailable_before_horizontal_move")
            return
        assert self._tcp_pose_xy is not None
        target = self._surface_targets[surface]
        delta = target - self._tcp_pose_xy
        distance = float(np.linalg.norm(delta))
        self._next_descent_phase = phase
        self._next_descent_surface = surface
        self._phase_deadline_monotonic_ns = None
        self._publish_motion_halt()
        if distance <= self._horizontal_position_tolerance_m:
            self._current_xy = self._surface_targets[surface].copy()
            self._begin_surface_settle(now_ns)
            return
        self._move_target_xy = target.copy()
        self._horizontal_command_xy = (
            delta / distance * self._horizontal_twist_command
        )
        timeout_sec = self._maximum_horizontal_move_sec
        self._custom_deadline_ns = now_ns + int(timeout_sec * 1.0e9)
        self._horizontal_best_distance_m = distance
        self._horizontal_last_progress_monotonic_ns = now_ns
        self._horizontal_initial_distance_m = distance
        self._horizontal_last_status_monotonic_ns = now_ns
        previous = self._state.value
        self._state = MotionState.MOVING_LATERALLY
        self._queue_system_event(
            f"move_to_{surface}_start", previous, self._state.value
        )
        self.get_logger().info(
            f"Moving in {self._expected_tcp_pose_frame} X/Y to {surface}: "
            f"delta={delta.tolist()} m, "
            f"closed-loop tolerance={self._horizontal_position_tolerance_m * 1000.0:.1f} mm, "
            f"command limit={self._horizontal_twist_command:.2f}, "
            f"timeout={timeout_sec:.1f} s."
        )

    def _begin_surface_settle(self, now_ns: int) -> None:
        previous = self._state.value
        self._state = MotionState.SETTLING_AT_SURFACE
        self._custom_deadline_ns = now_ns + int(
            self._surface_move_settle_sec * 1.0e9
        )
        self._queue_system_event(
            "surface_move_complete", previous, self._state.value
        )

    def _begin_surface_descent(self, now_ns: int) -> None:
        phase = self._next_descent_phase
        surface = self._next_descent_surface
        if phase is None or surface is None:
            self._fail_application("surface_descent_target_missing")
            return
        self._active_phase = phase
        self._active_surface = surface
        self._next_descent_phase = None
        self._next_descent_surface = None
        self._floor_contact = None
        self._enter_input = None
        self._place_streak = 0
        self._pending_model_trigger = None
        self._model_error = None
        self._latest_result = None
        self._last_submitted_sequence = None
        self._pending_probe_result = None
        self._pending_probe_halt_ros_ns = None
        self._pending_probe_halt_monotonic_ns = None
        self._drain_queue(self._request_queue)
        super()._begin_lowering(now_ns)
        self._queue_system_event(
            f"{phase}_{surface}_descent_start",
            MotionState.SETTLING_AT_SURFACE.value,
            self._state.value,
        )
        self.get_logger().info(
            f"{phase}: lowering toward the {surface} support. "
            "Press Enter at visible contact; the model then has 0.8 s from "
            "the reaction-compensated contact time to detect place."
        )

    def _handle_key(self, event: KeyInput) -> None:
        if (
            event.key == "enter"
            and self._state
            in {MotionState.LOWERING, MotionState.WAITING_FOR_PROBE_ENTER}
            and self._active_episode_id is not None
            and self._active_phase in {"probe_1", "probe_2", "final_place"}
        ):
            state = self._state.value
            if self._floor_contact is not None:
                self._queue_key_event(event, state, state)
                self.get_logger().warning(
                    f"Duplicate Enter ignored for {self._active_phase}."
                )
                return
            compensation_ns = int(
                self._enter_reaction_compensation_sec * 1.0e9
            )
            self._enter_input = event
            self._floor_contact = KeyInput(
                key=f"floor_contact_{self._active_phase}",
                ros_stamp_ns=event.ros_stamp_ns - compensation_ns,
                monotonic_ns=event.monotonic_ns - compensation_ns,
            )
            self._queue_key_event(event, state, state)
            self._queue_system_event(
                f"floor_contact_{self._active_phase}",
                state,
                state,
                timing=self._floor_contact,
            )
            self._backfill_score_contact_columns(self._floor_contact)
            self.get_logger().info(
                f"Enter recorded for {self._active_phase}; estimated contact "
                f"is {self._enter_reaction_compensation_sec * 1000.0:.0f} ms "
                "earlier. Place must be detected within 0.8 s of that time."
            )
            if (
                self._state == MotionState.WAITING_FOR_PROBE_ENTER
                and self._pending_probe_result is not None
                and self._pending_probe_halt_ros_ns is not None
                and self._pending_probe_halt_monotonic_ns is not None
            ):
                self._complete_detected_contact(
                    self._pending_probe_result,
                    self._pending_probe_halt_ros_ns,
                    self._pending_probe_halt_monotonic_ns,
                )
            return
        super()._handle_key(event)

    def _finish_model_episode(self, result: InferenceResult) -> None:
        halt_monotonic_ns = time.perf_counter_ns()
        halt_ros_ns = int(self.get_clock().now().nanoseconds)
        self._publish_motion_halt()
        self._lower_ended_monotonic_ns = halt_monotonic_ns
        self._pending_probe_result = result
        self._pending_probe_halt_ros_ns = halt_ros_ns
        self._pending_probe_halt_monotonic_ns = halt_monotonic_ns
        if (
            result.fixed_surface_score is None
            or result.wobbling_surface_score is None
            or result.fixed_minus_wobbling is None
        ):
            self._fail_application("support_prototype_score_missing")
            return
        if self._floor_contact is None:
            previous = self._state.value
            self._state = MotionState.WAITING_FOR_PROBE_ENTER
            self._custom_deadline_ns = halt_monotonic_ns + int(
                self._probe_enter_grace_sec * 1.0e9
            )
            self._queue_system_event(
                "model_halt_waiting_for_enter", previous, self._state.value
            )
            self.get_logger().warning(
                f"{self._active_phase}: model halted before Enter. Motion is "
                f"stopped for up to {self._probe_enter_grace_sec:.2f} s so "
                "the contact timestamp can be validated."
            )
            return
        self._complete_detected_contact(result, halt_ros_ns, halt_monotonic_ns)

    def _complete_detected_contact(
        self,
        result: InferenceResult,
        halt_ros_ns: int,
        halt_monotonic_ns: int,
    ) -> None:
        contact = self._floor_contact
        if contact is None:
            self._fail_application("contact_enter_missing_after_model_halt")
            return
        detection_after_contact_ns = (
            result.completed_monotonic_ns - contact.monotonic_ns
        )
        if detection_after_contact_ns > int(
            self._model_max_wait_after_contact_sec * 1.0e9
        ):
            self._record_probe(
                result, halt_ros_ns, halt_monotonic_ns, valid=False
            )
            self._fail_application(
                f"{self._active_phase}_model_detection_timeout_after_contact",
                details={
                    "detection_after_estimated_contact_ms": (
                        detection_after_contact_ns / 1.0e6
                    ),
                    "allowed_ms": (
                        self._model_max_wait_after_contact_sec * 1000.0
                    ),
                },
            )
            return
        if result.request.window_end_stamp_ns < contact.ros_stamp_ns:
            self._record_probe(
                result, halt_ros_ns, halt_monotonic_ns, valid=False
            )
            self._fail_application("model_trigger_before_estimated_contact")
            return

        self._record_probe(result, halt_ros_ns, halt_monotonic_ns, valid=True)
        if self._active_phase == "final_place":
            selected_fixed = self._selected_surface == "fixed"
            status = "success" if selected_fixed else "failure"
            reason = (
                "selected_and_released_on_fixed_surface"
                if selected_fixed
                else "selected_and_released_on_wobbling_surface"
            )
            self._application_finished = True
            self._write_selection_evaluation(status, reason)
            self.get_logger().info(
                f"Final contact on {self._selected_surface}: opening gripper. "
                f"Selection result={status}."
            )
            self._state = MotionState.FINALIZING_CAPTURE
            self._open_gripper_and_start_return(
                reference_monotonic_ns=halt_monotonic_ns,
                complete_episode_on_open=True,
                episode_status=status,
                episode_reason=reason,
            )
            return

        self._begin_probe_lift(halt_monotonic_ns)

    def _record_probe(
        self,
        result: InferenceResult,
        halt_ros_ns: int,
        halt_monotonic_ns: int,
        valid: bool,
    ) -> None:
        contact = self._floor_contact
        interval = []
        if contact is not None:
            start_ns = min(contact.ros_stamp_ns, halt_ros_ns)
            end_ns = max(contact.ros_stamp_ns, halt_ros_ns)
        else:
            start_ns = result.request.window_start_stamp_ns
            end_ns = max(result.request.window_end_stamp_ns, halt_ros_ns)
        interval = [
            sample
            for sample in self._paired_samples
            if start_ns <= sample.stamp_ns <= end_ns
        ]
        interval_kind = "estimated_contact_to_detection"
        csv_name = f"{self._active_phase}_{interval_kind}_force.csv"
        csv_path = (
            self._session_dir / "csv" / csv_name
            if self._session_dir is not None
            else None
        )
        if csv_path is not None:
            self._write_force_interval(csv_path, interval)
        record = {
            "phase": self._active_phase,
            "surface": self._active_surface,
            "valid": valid,
            "trigger_source": "model",
            "force_interval_kind": interval_kind,
            "enter_key_stamp_ns": (
                self._enter_input.ros_stamp_ns if self._enter_input else None
            ),
            "estimated_contact_stamp_ns": (
                contact.ros_stamp_ns if contact else None
            ),
            "halt_stamp_ns": halt_ros_ns,
            "tcp_xy_at_halt_m": (
                self._tcp_pose_xy.tolist()
                if self._tcp_pose_xy is not None
                else None
            ),
            "target_tcp_xy_m": (
                self._surface_targets[self._active_surface].tolist()
                if self._active_surface is not None
                else None
            ),
            "descent_start_z_m": self._descent_start_z_m,
            "contact_to_halt_ms": (
                (halt_monotonic_ns - contact.monotonic_ns) / 1.0e6
                if contact
                else None
            ),
            "window_start_stamp_ns": result.request.window_start_stamp_ns,
            "window_end_stamp_ns": result.request.window_end_stamp_ns,
            "hold_cosine": result.hold_score,
            "place_cosine": result.place_score,
            "fixed_surface_cosine": result.fixed_surface_score,
            "wobbling_surface_cosine": result.wobbling_surface_score,
            "fixed_minus_wobbling": result.fixed_minus_wobbling,
            "inference_ms": result.inference_ms,
            "place_streak": self._place_streak,
            "force_csv": str(csv_path) if csv_path is not None else None,
            "force_interval": self._force_interval_metrics(interval),
        }
        self._probe_records.append(record)
        self._queue_system_event(
            f"{self._active_phase}_{self._active_surface}_detected",
            MotionState.LOWERING.value,
            (
                MotionState.FINALIZING_CAPTURE.value
                if self._active_phase == "final_place"
                else MotionState.LIFTING_AFTER_PROBE.value
            ),
        )
        self.get_logger().info(
            f"{self._active_phase} {self._active_surface}: "
            f"fixed={result.fixed_surface_score:.4f}, "
            f"wobble={result.wobbling_surface_score:.4f}, "
            f"fixed-wobble={result.fixed_minus_wobbling:+.4f}."
        )

    def _begin_probe_lift(self, now_ns: int) -> None:
        if self._descent_start_z_m is None:
            self._fail_application("probe_descent_start_height_missing")
            return
        self._custom_deadline_ns = now_ns + int(
            self._maximum_vertical_move_sec * 1.0e9
        )
        previous = self._state.value
        self._state = MotionState.LIFTING_AFTER_PROBE
        self._queue_system_event(
            f"{self._active_phase}_lift_start", previous, self._state.value
        )
        self.get_logger().info(
            f"Lifting clear of {self._active_surface} to the shared descent "
            f"start Z={self._descent_start_z_m:.4f} m."
        )

    def _finish_probe_lift(self, now_ns: int) -> None:
        self._publish_motion_halt()
        finished_phase = self._active_phase
        # The full approach descent has already been undone. Do not let a
        # later horizontal-move failure command that same vertical return a
        # second time.
        self._reset_descent_tracking()
        self._queue_system_event(
            f"{finished_phase}_lift_complete",
            MotionState.LIFTING_AFTER_PROBE.value,
            MotionState.MOVING_LATERALLY.value,
        )
        if finished_phase == "probe_1":
            self._begin_move_to_surface(self._probe_order[1], "probe_2", now_ns)
            return
        if finished_phase == "probe_2":
            probes = [
                record
                for record in self._probe_records
                if record["phase"] in {"probe_1", "probe_2"}
                and record["valid"]
            ]
            if len(probes) != 2:
                self._fail_application("two_valid_probe_scores_not_available")
                return
            self._selection_scores = {
                str(record["surface"]): float(record["fixed_minus_wobbling"])
                for record in probes
            }
            fixed_value = self._selection_scores["fixed"]
            wobble_value = self._selection_scores["wobble"]
            try:
                self._selected_surface = select_more_fixed_surface(
                    self._selection_scores
                )
            except ValueError as exc:
                self._fail_application(
                    "invalid_or_tied_surface_scores",
                    details={"error": str(exc)},
                )
                return
            self.get_logger().info(
                "Surface selection (no threshold): "
                f"fixed support score={fixed_value:+.4f}, "
                f"wobble support score={wobble_value:+.4f}; "
                f"selected={self._selected_surface}."
            )
            self._queue_system_event(
                f"selected_{self._selected_surface}",
                MotionState.LIFTING_AFTER_PROBE.value,
                MotionState.MOVING_LATERALLY.value,
            )
            self._begin_move_to_surface(
                self._selected_surface, "final_place", now_ns
            )
            return
        self._fail_application("unexpected_probe_lift_phase")

    def _finish_without_trigger(self, reason: str) -> None:
        self._fail_application(f"{self._active_phase}_{reason}")

    def _fail_application(
        self,
        reason: str,
        details: Optional[dict] = None,
    ) -> None:
        if self._application_finished:
            return
        self._application_finished = True
        now_ns = time.perf_counter_ns()
        self._publish_motion_halt()
        if self._state == MotionState.LOWERING:
            self._lower_ended_monotonic_ns = now_ns
        self._queue_system_event(
            reason, self._state.value, MotionState.FINALIZING_CAPTURE.value
        )
        self._write_selection_evaluation("failure", reason, details)
        self._state = MotionState.FINALIZING_CAPTURE
        self.get_logger().error(
            f"Two-surface trial failed: {reason}. Releasing and returning."
        )
        self._open_gripper_and_start_return(
            reference_monotonic_ns=now_ns,
            complete_episode_on_open=True,
            episode_status="failure",
            episode_reason=reason,
        )

    def _write_selection_evaluation(
        self,
        status: str,
        outcome: str,
        details: Optional[dict] = None,
    ) -> None:
        if self._session_dir is None:
            return
        metrics = {
            "episode_id": self._active_episode_id,
            "object_name": self._object_name or None,
            "task_name": self.config.task_name,
            "status": status,
            "outcome": outcome,
            "probe_order": self._probe_order,
            "selection_rule": (
                "argmax(fixed_surface_cosine - wobbling_surface_cosine)"
            ),
            "selection_threshold": None,
            "surface_target_base_xy_m": {
                name: value.tolist()
                for name, value in self._surface_targets.items()
            },
            "episode_home_tcp_xy_m": (
                self._home_xy.tolist() if self._home_xy is not None else None
            ),
            "episode_home_tcp_xyz_m": (
                self._home_xyz.tolist() if self._home_xyz is not None else None
            ),
            "descent_start_z_m": self._descent_start_z_m,
            "selection_scores": self._selection_scores,
            "selected_surface": self._selected_surface,
            "selection_correct": (
                self._selected_surface == "fixed"
                if self._selected_surface is not None
                else None
            ),
            "probes": self._probe_records,
            "empty_offset_correction": self._empty_offset_details,
            "detector": self._detector.metadata(),
            "details": details,
        }
        path = self._session_dir / "surface_selection_evaluation.json"
        path.write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _control_tick(self) -> None:
        now_ns = time.perf_counter_ns()
        self._check_exclusive_control_graph_runtime(now_ns)
        if self._state == MotionState.LIFTING_AFTER_GRASP:
            if (
                now_ns - self._last_motion_publish_monotonic_ns
                < int(self._control_period_sec * 1.0e9)
            ):
                return
            self._last_motion_publish_monotonic_ns = now_ns
            if self._custom_deadline_ns is None:
                self._custom_deadline_ns = now_ns + int(
                    self._maximum_vertical_move_sec * 1.0e9
                )
                self.get_logger().info(
                    "Continuous closed-loop pick lift: target Z="
                    f"{self._descent_start_z_m:.4f} m."
                )
            if self._drive_to_descent_start_height(now_ns):
                self._finish_initial_pick_lift(now_ns)
            return

        if self._state in {
            MotionState.MOVING_LATERALLY,
            MotionState.SETTLING_AT_SURFACE,
            MotionState.LIFTING_AFTER_PROBE,
            MotionState.WAITING_FOR_PROBE_ENTER,
        }:
            if (
                now_ns - self._last_motion_publish_monotonic_ns
                < int(self._control_period_sec * 1.0e9)
            ):
                return
            self._last_motion_publish_monotonic_ns = now_ns

            if self._state == MotionState.MOVING_LATERALLY:
                if not self._tcp_pose_is_fresh(now_ns):
                    self._fail_application("tcp_pose_stale_during_horizontal_move")
                    return
                assert self._tcp_pose_xy is not None
                error = self._move_target_xy - self._tcp_pose_xy
                distance = float(np.linalg.norm(error))
                self._log_horizontal_progress(distance, now_ns)
                if distance <= self._horizontal_position_tolerance_m:
                    self._publish_motion_halt()
                    assert self._next_descent_surface is not None
                    self._current_xy = self._surface_targets[
                        self._next_descent_surface
                    ].copy()
                    self._begin_surface_settle(now_ns)
                elif self._horizontal_motion_is_stalled(distance, now_ns):
                    self._fail_application(
                        "horizontal_move_stalled",
                        details={"remaining_distance_m": distance},
                    )
                elif self._custom_deadline_ns is not None and now_ns >= self._custom_deadline_ns:
                    self._fail_application(
                        "horizontal_move_timeout",
                        details={"remaining_distance_m": distance},
                    )
                else:
                    magnitude = min(
                        self._horizontal_twist_command,
                        max(
                            self._horizontal_min_twist_command,
                            distance * self._horizontal_position_gain,
                        ),
                    )
                    command = error / distance * magnitude
                    self._horizontal_command_xy = command
                    servo_command = transform_pose_error_to_command_xy(
                        command,
                        self._tcp_pose_xy_to_command_sign,
                    )
                    self._publish_twist_xyz(
                        x_command=float(servo_command[0]),
                        y_command=float(servo_command[1]),
                    )
                return

            if self._state == MotionState.SETTLING_AT_SURFACE:
                self._publish_motion_halt()
                if self._custom_deadline_ns is not None and now_ns >= self._custom_deadline_ns:
                    self._begin_surface_descent(now_ns)
                return

            if self._state == MotionState.LIFTING_AFTER_PROBE:
                if self._drive_to_descent_start_height(now_ns):
                    self._finish_probe_lift(now_ns)
                return

            self._publish_motion_halt()
            if (
                self._custom_deadline_ns is not None
                and now_ns >= self._custom_deadline_ns
            ):
                self._fail_application(
                    f"{self._active_phase}_contact_enter_not_recorded"
                )
            return

        super()._control_tick()

    def _finish_initial_pick_lift(self, now_ns: int) -> None:
        """Finish the one continuous pick lift and start the probe sequence."""
        self._publish_motion_halt()
        self._custom_deadline_ns = None
        previous = self._state.value
        if self._post_lift_hold_sec > 0.0:
            self._state = MotionState.POST_LIFT_HOLD
            self._phase_deadline_monotonic_ns = now_ns + int(
                self._post_lift_hold_sec * 1.0e9
            )
        else:
            self._begin_lowering(now_ns)
        self._queue_system_event(
            "pick_lift_complete", previous, self._state.value
        )
        if self._state == MotionState.POST_LIFT_HOLD:
            self.get_logger().info(
                "Closed-loop pick lift complete; holding before horizontal "
                "motion."
            )
        else:
            self._queue_system_event(
                "place_descent_start", previous, self._state.value
            )

    def _horizontal_motion_is_stalled(
        self,
        distance_m: float,
        now_ns: int,
    ) -> bool:
        """Detect loss of Cartesian progress without shortening valid moves."""
        if (
            self._horizontal_best_distance_m is None
            or self._horizontal_last_progress_monotonic_ns is None
        ):
            self._horizontal_best_distance_m = distance_m
            self._horizontal_last_progress_monotonic_ns = now_ns
            return False
        if (
            distance_m
            <= self._horizontal_best_distance_m - self._horizontal_min_progress_m
        ):
            self._horizontal_best_distance_m = distance_m
            self._horizontal_last_progress_monotonic_ns = now_ns
            return False
        return now_ns - self._horizontal_last_progress_monotonic_ns >= int(
            self._horizontal_stall_timeout_sec * 1.0e9
        )

    def _log_horizontal_progress(self, distance_m: float, now_ns: int) -> None:
        """Report progress so slow motion is not mistaken for a freeze."""
        if (
            self._horizontal_initial_distance_m is None
            or self._horizontal_last_status_monotonic_ns is None
            or now_ns - self._horizontal_last_status_monotonic_ns
            < int(self._horizontal_status_period_sec * 1.0e9)
        ):
            return
        progress_m = max(
            0.0,
            self._horizontal_initial_distance_m - distance_m,
        )
        self.get_logger().info(
            "Horizontal move progress: "
            f"{progress_m * 1000.0:.1f} mm travelled, "
            f"{distance_m * 1000.0:.1f} mm remaining."
        )
        self._horizontal_last_status_monotonic_ns = now_ns

    def _drive_to_descent_start_height(self, now_ns: int) -> bool:
        """Servo vertically to the one shared probe start height."""
        if self._descent_start_z_m is None:
            self._fail_application("probe_descent_start_height_missing")
            return False
        if not self._tcp_pose_is_fresh(now_ns):
            self._fail_application("tcp_pose_stale_during_vertical_move")
            return False
        assert self._tcp_pose_xyz is not None
        error_m = self._descent_start_z_m - float(self._tcp_pose_xyz[2])
        if abs(error_m) <= self._vertical_position_tolerance_m:
            self._publish_motion_halt()
            return True
        if (
            self._custom_deadline_ns is not None
            and now_ns >= self._custom_deadline_ns
        ):
            self._fail_application(
                "vertical_move_timeout",
                details={"remaining_z_error_m": error_m},
            )
            return False
        command = math.copysign(self._probe_vertical_twist_command, error_m)
        self._publish_twist(z_command=command)
        return False


def main(args: Optional[list[str]] = None) -> None:
    """Run the two-surface selection capture node."""
    rclpy.init(args=args)
    node = TwoSurfaceSelectionCaptureNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
