"""Keyboard-driven robot motion and synchronized multimodal data capture."""

from __future__ import annotations

import csv
import json
import os
import queue
import select
import shutil
import statistics
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped, WrenchStamped
from rclpy.action import ActionClient
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from robotiq_2f_gripper_msgs.action import MoveTwoFingerGripper
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, String
from std_srvs.srv import Trigger

from .bag_converter import convert_bag_to_dataset
from .data_capture_node import DataCaptureNode


class MotionState(str, Enum):
    """States that accept different keyboard commands."""

    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"
    POST_GRASP_HOLD = "post_grasp_hold"
    LIFTING_AFTER_GRASP = "lifting_after_grasp"
    POST_LIFT_HOLD = "post_lift_hold"
    LOWERING = "lowering"
    STOPPING_AFTER_ENTER = "stopping_after_enter"
    FINALIZING_CAPTURE = "finalizing_capture"
    STOPPED = "stopped"
    OPENING_FOR_RETURN = "opening_for_return"
    RETURNING = "returning"
    RETURNING_HOME = "returning_home"
    MOVING_LATERALLY = "moving_laterally"
    SETTLING_AT_SURFACE = "settling_at_surface"
    LIFTING_AFTER_PROBE = "lifting_after_probe"
    WAITING_FOR_PROBE_ENTER = "waiting_for_probe_enter"
    ERROR = "error"


@dataclass(frozen=True)
class KeyInput:
    """A key together with timestamps captured in the terminal thread."""

    key: str
    ros_stamp_ns: int
    monotonic_ns: int


@dataclass(frozen=True)
class ForceSample:
    """One force sample and both its source and subscriber timestamps."""

    stamp_ns: int
    receipt_ros_ns: int
    receipt_monotonic_ns: int
    values: tuple[float, float, float, float, float, float]
    transport_age_ms: Optional[float]


@dataclass
class PendingKeyEvent:
    """Key event waiting briefly for the nearest post-key force sample."""

    event_id: int
    input: KeyInput
    source: str
    state_before: str
    state_after: str
    episode_id: Optional[int]
    resolve_after_monotonic_ns: int


class TerminalKeyReader:
    """Read escape sequences directly from the controlling terminal."""

    _SEQUENCES = {
        b"\x1b[A": "up",
        b"\x1b[B": "down",
        b"\x1b[C": "right",
        b"\x1b[D": "left",
    }

    def __init__(self, node: "KeyMotionCaptureNode") -> None:
        """Initialize the reader for the node's controlling terminal."""
        self._node = node
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._fd: Optional[int] = None
        self._old_termios = None

    def start(self) -> None:
        """Start reading keys in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader and restore the terminal settings."""
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._restore_terminal()

    def _run(self) -> None:
        try:
            self._fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except OSError as exc:
            self._node.get_logger().error(
                f"Cannot open /dev/tty for keyboard control: {exc}"
            )
            self._node._keyboard_failed.set()
            return

        buffer = b""
        try:
            while not self._shutdown.is_set() and not self._node._stop_event.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                try:
                    chunk = os.read(self._fd, 64)
                except BlockingIOError:
                    continue
                if not chunk:
                    continue
                buffer += chunk
                buffer = self._parse_buffer(buffer)
        finally:
            self._restore_terminal()

    def _parse_buffer(self, buffer: bytes) -> bytes:
        while buffer:
            if buffer.startswith(b"\x1b"):
                if len(buffer) < 3:
                    return buffer
                sequence = buffer[:3]
                key = self._SEQUENCES.get(sequence, f"escape:{sequence.hex()}")
                self._emit(key)
                buffer = buffer[3:]
                continue

            byte = buffer[:1]
            buffer = buffer[1:]
            if byte in {b"\r", b"\n"}:
                self._emit("enter")
            elif byte == b"q":
                self._emit("q")
            elif byte == b" ":
                self._emit("space")
            elif byte.isascii() and byte.isprintable():
                self._emit(byte.decode("ascii"))
            else:
                self._emit(f"byte:{byte.hex()}")
        return buffer

    def _emit(self, key: str) -> None:
        event = KeyInput(
            key=key,
            ros_stamp_ns=int(self._node.get_clock().now().nanoseconds),
            monotonic_ns=time.perf_counter_ns(),
        )
        self._node._key_queue.put(event)

    def _restore_terminal(self) -> None:
        if self._fd is None:
            return
        if self._old_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except termios.error:
                pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None


class KeyMotionCaptureNode(DataCaptureNode):
    """Move from a configurable home pose and record key/force timing."""

    _EVENT_FIELDS = [
        "event_id",
        "episode_id",
        "source",
        "key",
        "stamp_ns",
        "timestamp_sec",
        "monotonic_ns",
        "state_before",
        "state_after",
        "left_stamp_ns",
        "left_delta_ms",
        "left_transport_age_ms",
        "left_fx",
        "left_fy",
        "left_fz",
        "left_tx",
        "left_ty",
        "left_tz",
        "right_stamp_ns",
        "right_delta_ms",
        "right_transport_age_ms",
        "right_fx",
        "right_fy",
        "right_fz",
        "right_tx",
        "right_ty",
        "right_tz",
    ]

    def __init__(self) -> None:
        """Initialize motion state, capture inputs, and hardware interfaces."""
        super().__init__()

        default_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        default_home_positions = [
            0.6006759405136108,
            -1.4743512312518519,
            1.3838186264038086,
            -1.501033131276266,
            -1.5937483946429651,
            -0.2009494940387171,
        ]
        home_joint_names = list(
            self.declare_parameter("home_joint_names", default_joint_names).value
        )
        home_positions = list(
            self.declare_parameter("home_positions_rad", default_home_positions).value
        )
        if len(home_joint_names) != len(home_positions):
            raise ValueError(
                "home_joint_names and home_positions_rad must have the same length"
            )
        self._home_pose_rad = dict(zip(home_joint_names, home_positions))

        self._joint_state_topic = self.declare_parameter(
            "joint_state_topic", "/joint_states"
        ).value
        self._robot_program_running_topic = self.declare_parameter(
            "robot_program_running_topic",
            "/io_and_status_controller/robot_program_running",
        ).value
        self._joint_command_topic = self.declare_parameter(
            "joint_command_topic", "/servo_node/delta_joint_cmds"
        ).value
        self._twist_command_topic = self.declare_parameter(
            "twist_command_topic", "/servo_node/delta_twist_cmds"
        ).value
        self._servo_start_service = self.declare_parameter(
            "servo_start_service", "/servo_node/start_servo"
        ).value
        self._command_frame = self.declare_parameter(
            "command_frame", "base_link"
        ).value

        self._startup_timeout_sec = float(
            self.declare_parameter("startup_timeout_sec", 30.0).value
        )
        self._home_timeout_sec = float(
            self.declare_parameter("home_timeout_sec", 30.0).value
        )
        self._home_deadband = float(
            self.declare_parameter("home_deadband", 0.002).value
        )
        self._home_settle_duration_sec = float(
            self.declare_parameter("home_settle_duration_sec", 0.4).value
        )
        self._home_velocity_tolerance_rad_sec = float(
            self.declare_parameter(
                "home_velocity_tolerance_rad_sec", 0.003
            ).value
        )
        self._home_min_joint_speed = float(
            self.declare_parameter("home_min_joint_speed", 0.02).value
        )
        self._home_max_joint_speed = float(
            self.declare_parameter("home_max_joint_speed", 1.0).value
        )
        self._home_velocity_gain = float(
            self.declare_parameter("home_velocity_gain", 1.5).value
        )

        self._gripper_action_name = self.declare_parameter(
            "gripper_action_name", "/robotiq_2f_gripper_action"
        ).value
        self._gripper_goal_log_topic = self.declare_parameter(
            "gripper_goal_log_topic", "/robotiq_2f_gripper_action_goal"
        ).value
        self._gripper_open_position = float(
            self.declare_parameter("gripper_open_position", 0.140).value
        )
        self._gripper_close_position = float(
            self.declare_parameter("gripper_close_position", 0.002).value
        )
        self._gripper_target_speed = float(
            self.declare_parameter("gripper_target_speed", 0.1).value
        )
        self._gripper_target_force = float(
            self.declare_parameter("gripper_target_force", 0.1).value
        )
        self._gripper_result_timeout_sec = float(
            self.declare_parameter("gripper_result_timeout_sec", 25.0).value
        )

        self._control_period_sec = float(
            self.declare_parameter("control_period_sec", 0.01).value
        )
        self._lower_twist_z = float(
            self.declare_parameter("lower_twist_command_z", -0.5).value
        )
        self._return_twist_z = float(
            self.declare_parameter("return_twist_command_z", 1.0).value
        )
        self._enter_extra_lower_sec = float(
            self.declare_parameter("enter_extra_lower_sec", 1.0).value
        )
        self._success_reason = str(
            self.declare_parameter(
                "success_reason",
                "enter_then_one_second_descent_then_gripper_open",
            ).value
        )
        self._max_lower_duration_sec = float(
            self.declare_parameter("max_lower_duration_sec", 8.0).value
        )
        self._post_start_recording_delay_sec = float(
            self.declare_parameter("post_start_recording_delay_sec", 0.5).value
        )
        self._post_grasp_hold_sec = float(
            self.declare_parameter("post_grasp_hold_sec", 0.0).value
        )
        self._pick_lift_duration_sec = float(
            self.declare_parameter("pick_lift_duration_sec", 0.0).value
        )
        self._post_lift_hold_sec = float(
            self.declare_parameter("post_lift_hold_sec", 0.0).value
        )
        self._enter_event_alias = str(
            self.declare_parameter("enter_event_alias", "").value
        ).strip()
        for parameter_name, value in (
            ("post_grasp_hold_sec", self._post_grasp_hold_sec),
            ("pick_lift_duration_sec", self._pick_lift_duration_sec),
            ("post_lift_hold_sec", self._post_lift_hold_sec),
        ):
            if value < 0.0:
                raise ValueError(f"{parameter_name} must be non-negative")
        if self._pick_lift_duration_sec > 0.0 and self._return_twist_z <= 0.0:
            raise ValueError(
                "return_twist_command_z must be positive when pick lifting is enabled"
            )

        self._left_ft_topic = self.declare_parameter(
            "left_ft_topic", "/force_torque/left"
        ).value
        self._right_ft_topic = self.declare_parameter(
            "right_ft_topic", "/force_torque/right"
        ).value
        self._key_event_topic = self.declare_parameter(
            "key_event_topic", "/data_capture/key_events"
        ).value
        self._event_force_wait_sec = float(
            self.declare_parameter("event_force_wait_sec", 0.03).value
        )
        self._camera_wrist_topic = self.declare_parameter(
            "camera_wrist_topic",
            "/camera_wrist/realsense2_camera/color/image_raw",
        ).value
        self._camera_fixed_topic = self.declare_parameter(
            "camera_fixed_topic",
            "/camera_fixed/realsense2_camera/color/image_raw",
        ).value
        self._require_cameras = bool(
            self.declare_parameter("require_cameras", True).value
        )
        self._require_force_sensors = bool(
            self.declare_parameter("require_force_sensors", True).value
        )

        self._relay_servo_commands = bool(
            self.declare_parameter("relay_servo_commands", True).value
        )
        self._servo_output_topic = self.declare_parameter(
            "servo_output_topic", "/forward_position_controller/commands_teleop"
        ).value
        self._robot_command_topic = self.declare_parameter(
            "robot_command_topic", "/forward_position_controller/commands"
        ).value

        self._state = MotionState.INITIALIZING
        self._joint_positions: dict[str, float] = {}
        self._joint_velocities: dict[str, float] = {}
        self._joint_state_ready = False
        self._robot_program_running: Optional[bool] = None
        self._key_queue: queue.Queue[KeyInput] = queue.Queue()
        self._keyboard_failed = threading.Event()
        self._key_reader = TerminalKeyReader(self)
        self._event_id = 0
        self._pending_events: list[PendingKeyEvent] = []
        self._event_lock = threading.Lock()
        self._force_samples: dict[str, deque[ForceSample]] = {
            "left": deque(maxlen=4096),
            "right": deque(maxlen=4096),
        }
        self._force_period_ms: dict[str, list[float]] = {"left": [], "right": []}
        self._force_age_ms: dict[str, list[float]] = {"left": [], "right": []}
        self._force_sample_count = {"left": 0, "right": 0}
        self._previous_force_stamp_ns: dict[str, Optional[int]] = {
            "left": None,
            "right": None,
        }
        self._camera_ready = {"wrist": False, "fixed": False}
        self._lower_started_monotonic_ns: Optional[int] = None
        self._lower_ended_monotonic_ns: Optional[int] = None
        self._stop_deadline_monotonic_ns: Optional[int] = None
        self._phase_deadline_monotonic_ns: Optional[int] = None
        self._return_deadline_monotonic_ns: Optional[int] = None
        self._pending_forced_return_ns: Optional[int] = None
        self._force_recording_active = True
        self._last_motion_publish_monotonic_ns = 0
        self._run_dir: Optional[Path] = None
        self._run_event_csv_path: Optional[Path] = None
        self._episode_summary_path: Optional[Path] = None
        self._episode_counter = 0
        self._active_episode_id: Optional[int] = None
        self._active_episode_started_ros_ns: Optional[int] = None

        command_qos = QoSProfile(depth=10)
        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._joint_state_cb,
            qos_profile_sensor_data,
        )
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = QoSReliabilityPolicy.RELIABLE
        latched_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool,
            self._robot_program_running_topic,
            self._robot_program_running_cb,
            latched_qos,
        )
        self.create_subscription(
            WrenchStamped,
            self._left_ft_topic,
            lambda message: self._force_cb("left", message),
            qos_profile_sensor_data,
        )
        self._camera_readiness_subscriptions = [
            self.create_subscription(
                Image,
                self._camera_wrist_topic,
                lambda _message: self._camera_ready.__setitem__("wrist", True),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Image,
                self._camera_fixed_topic,
                lambda _message: self._camera_ready.__setitem__("fixed", True),
                qos_profile_sensor_data,
            ),
        ]
        self.create_subscription(
            WrenchStamped,
            self._right_ft_topic,
            lambda message: self._force_cb("right", message),
            qos_profile_sensor_data,
        )
        self._joint_pub = self.create_publisher(
            JointJog, self._joint_command_topic, command_qos
        )
        self._twist_pub = self.create_publisher(
            TwistStamped, self._twist_command_topic, command_qos
        )
        self._gripper_goal_pub = self.create_publisher(
            Float32MultiArray, self._gripper_goal_log_topic, command_qos
        )
        self._key_event_pub = self.create_publisher(
            String, self._key_event_topic, command_qos
        )
        self._servo_start_client = self.create_client(
            Trigger, self._servo_start_service
        )
        self._gripper_action_client = ActionClient(
            self, MoveTwoFingerGripper, self._gripper_action_name
        )

        self._robot_command_pub = None
        if self._relay_servo_commands:
            self._robot_command_pub = self.create_publisher(
                Float64MultiArray, self._robot_command_topic, command_qos
            )
            self.create_subscription(
                Float64MultiArray,
                self._servo_output_topic,
                self._relay_servo_command,
                command_qos,
            )

    def run(self) -> None:
        """Prepare hardware and collect multiple independent episodes."""
        self._prepare_run()
        self._cleanup_existing_viewers()
        self._start_internal_viewers()
        self._start_throttles_and_compressors()

        try:
            if not self._wait_for_robot_program(self._startup_timeout_sec):
                raise RuntimeError("External Control program is not running")
            self._start_servo()
            self._send_gripper_goal(self._gripper_open_position, wait_for_result=True)
            if not self._move_to_home():
                raise RuntimeError("Failed to reach the configured initial pose")
            if not self._wait_for_capture_sources(self._startup_timeout_sec):
                raise RuntimeError("Required camera or force topics are unavailable")

            self._state = MotionState.READY
            self._key_reader.start()
            if self._pick_lift_duration_sec > 0.0:
                sequence = (
                    f"holds for {self._post_grasp_hold_sec:.3f} s, lifts for "
                    f"{self._pick_lift_duration_sec:.3f} s, holds aloft for "
                    f"{self._post_lift_hold_sec:.3f} s, then lowers"
                )
            else:
                sequence = "then lowers"
            self.get_logger().info(self._ready_message(sequence))

            while rclpy.ok() and not self._stop_event.is_set():
                rclpy.spin_once(self, timeout_sec=0.005)
                self._drain_key_events()
                self._control_tick()
                self._resolve_pending_events()
                if self._keyboard_failed.is_set():
                    raise RuntimeError("Keyboard input is unavailable")
        except KeyboardInterrupt:
            self.get_logger().info("Keyboard interrupt received; stopping safely.")
        except Exception as exc:  # noqa: BLE001
            self._state = MotionState.ERROR
            self.get_logger().error(f"Key motion capture failed: {exc}")
        finally:
            self._publish_motion_halt()
            self._publish_joint_halt()
            self._stop_event.set()
            self._key_reader.stop()
            self._resolve_pending_events(force=True)
            if self._active_episode_id is not None:
                self._discard_active_episode("node_stopped")
            self._stop_rosbag(fast=True)
            self._stop_viewers()
            self._stop_aux_processes()
            self._cleanup_existing_viewers()

    def _ready_message(self, sequence: str) -> str:
        """Return profile-specific keyboard instructions."""
        return (
            "READY: Right arrow starts a new episode, waits for physical "
            f"gripper closure, {sequence}. Enter continues descent for "
            f"{self._enter_extra_lower_sec:.3f} seconds and opens the "
            "gripper; only that path is saved. Up arrow aborts the episode "
            "and returns; q exits."
        )

    def _prepare_run(self) -> None:
        """Create one run directory that contains all successful episodes."""
        run_id = self._session_id_override or datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        run_dir = self.config.output_root / self.config.task_name / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "csv").mkdir()
        (run_dir / "logs").mkdir()
        self._write_capture_config_snapshot(run_dir / "config.yaml")
        (run_dir / "prompt.txt").write_text(
            self.config.prompt + "\n", encoding="utf-8"
        )

        self._run_dir = run_dir
        self._session_dir = run_dir
        self._bag_uri = None
        self._run_event_csv_path = run_dir / "csv" / "key_events_all.csv"
        self._episode_summary_path = run_dir / "csv" / "episodes.csv"
        with self._run_event_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.DictWriter(handle, fieldnames=self._EVENT_FIELDS).writeheader()
        with self._episode_summary_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "episode_id",
                    "status",
                    "reason",
                    "start_stamp_ns",
                    "end_stamp_ns",
                    "duration_sec",
                    "path",
                ],
            ).writeheader()
        self.get_logger().info(f"Continuous capture run: {run_dir}")

    def _start_episode(self, trigger: KeyInput) -> None:
        """Start a fresh rosbag before issuing any episode motion command."""
        if self._run_dir is None:
            raise RuntimeError("Run directory is not initialized")
        if self._active_episode_id is not None:
            raise RuntimeError("An episode is already active")

        self._episode_counter += 1
        episode_id = self._episode_counter
        episode_dir = self._run_dir / f"episode_{episode_id:06d}"
        (episode_dir / "bag").mkdir(parents=True)
        (episode_dir / "logs").mkdir()
        self._write_capture_config_snapshot(episode_dir / "config.yaml")
        (episode_dir / "prompt.txt").write_text(
            self.config.prompt + "\n", encoding="utf-8"
        )

        self._active_episode_id = episode_id
        self._active_episode_started_ros_ns = trigger.ros_stamp_ns
        self._session_dir = episode_dir
        self._bag_uri = episode_dir / "bag" / f"episode_{episode_id:06d}"
        self._stop_time_ns = None
        self._reset_force_statistics()
        self._ensure_event_file()

        try:
            self._start_rosbag()
            self._sleep_spin(self._post_start_recording_delay_sec)
            if self._bag_process is None or self._bag_process.poll() is not None:
                raise RuntimeError("rosbag2 exited during episode startup")
        except Exception as exc:  # noqa: BLE001
            self._discard_active_episode("rosbag_start_failed")
            raise RuntimeError(str(exc)) from exc

        self.get_logger().info(
            f"Episode {episode_id} recording started: {episode_dir}"
        )

    def _complete_active_episode(
        self,
        status_name: str = "success",
        reason: Optional[str] = None,
    ) -> None:
        """Stop and retain an episode after the gripper has physically opened."""
        if self._active_episode_id is None or self._session_dir is None:
            raise RuntimeError("No active episode to complete")

        episode_id = self._active_episode_id
        episode_dir = self._session_dir
        start_ns = self._active_episode_started_ros_ns
        self._record_stop_time_if_missing()
        end_ns = self._stop_time_ns
        self._write_force_latency_summary()
        self._stop_rosbag()
        if self._bag_process is not None and self._bag_process.poll() is None:
            raise RuntimeError("rosbag2 did not stop; episode cannot be finalized")
        if self._bag_uri is None or not self._bag_uri.exists():
            raise RuntimeError("Completed episode bag was not created")

        status = {
            "episode_id": episode_id,
            "status": status_name,
            "reason": reason or self._success_reason,
            "start_stamp_ns": start_ns,
            "end_stamp_ns": end_ns,
        }
        (episode_dir / "episode_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        self._append_episode_summary(
            episode_id=episode_id,
            status=status_name,
            reason=status["reason"],
            start_ns=start_ns,
            end_ns=end_ns,
            path=episode_dir,
        )

        completed_bag_uri = self._bag_uri
        self._clear_active_episode()
        if self._convert_after_record:
            try:
                convert_bag_to_dataset(
                    completed_bag_uri,
                    episode_dir,
                    self.config,
                    logger=self.get_logger(),
                    cutoff_stamp_ns=end_ns,
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f"Episode {episode_id} conversion failed; the retained "
                    f"bag is retained: {exc}"
                )

        self.get_logger().info(
            f"Episode {episode_id} {status_name.upper()}: recording kept "
            "through gripper open."
        )

    def _discard_active_episode(self, reason: str) -> None:
        """Stop recording and remove all sensor data for a failed episode."""
        if self._active_episode_id is None:
            return

        episode_id = self._active_episode_id
        episode_dir = self._session_dir
        start_ns = self._active_episode_started_ros_ns
        end_ns = int(self.get_clock().now().nanoseconds)
        self._resolve_pending_events(force=True)
        self._stop_rosbag(fast=True)
        if self._bag_process is not None and self._bag_process.poll() is None:
            self._state = MotionState.ERROR
            self.get_logger().error(
                f"Episode {episode_id}: rosbag2 is still running; refusing to "
                "remove a directory that may still be in use."
            )
            return

        if episode_dir is not None and episode_dir != self._run_dir:
            shutil.rmtree(episode_dir, ignore_errors=True)
            if episode_dir.exists():
                self._state = MotionState.ERROR
                self.get_logger().error(
                    f"Failed to remove discarded episode directory: {episode_dir}"
                )
                return
        self._append_episode_summary(
            episode_id=episode_id,
            status="discarded",
            reason=reason,
            start_ns=start_ns,
            end_ns=end_ns,
            path=None,
        )
        self.get_logger().warning(
            f"Episode {episode_id} DISCARDED: {reason}."
        )
        self._clear_active_episode()

    def _clear_active_episode(self) -> None:
        self._active_episode_id = None
        self._active_episode_started_ros_ns = None
        self._session_dir = self._run_dir
        self._bag_uri = None
        self._stop_time_ns = None

    def _append_episode_summary(
        self,
        episode_id: int,
        status: str,
        reason: str,
        start_ns: Optional[int],
        end_ns: Optional[int],
        path: Optional[Path],
    ) -> None:
        if self._episode_summary_path is None:
            return
        duration_sec = None
        if start_ns is not None and end_ns is not None:
            duration_sec = max(0, end_ns - start_ns) / 1.0e9
        row = {
            "episode_id": episode_id,
            "status": status,
            "reason": reason,
            "start_stamp_ns": start_ns,
            "end_stamp_ns": end_ns,
            "duration_sec": duration_sec,
            "path": str(path) if path is not None else "",
        }
        with self._episode_summary_path.open(
            "a", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writerow(row)

    def _reset_force_statistics(self) -> None:
        self._force_period_ms = {"left": [], "right": []}
        self._force_age_ms = {"left": [], "right": []}
        self._force_sample_count = {"left": 0, "right": 0}
        self._previous_force_stamp_ns = {"left": None, "right": None}

    def _joint_state_cb(self, message: JointState) -> None:
        for index, name in enumerate(message.name):
            if index < len(message.position):
                self._joint_positions[name] = float(message.position[index])
            if index < len(message.velocity):
                self._joint_velocities[name] = float(message.velocity[index])
        self._joint_state_ready = True

    def _robot_program_running_cb(self, message: Bool) -> None:
        self._robot_program_running = bool(message.data)

    def _force_cb(self, side: str, message: WrenchStamped) -> None:
        if not self._force_recording_active:
            return
        self._force_sample_count[side] += 1
        receipt_ros_ns = int(self.get_clock().now().nanoseconds)
        stamp_ns = int(
            message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec
        )
        if stamp_ns <= 0:
            stamp_ns = receipt_ros_ns
            age_ms = None
        else:
            age_ms = (receipt_ros_ns - stamp_ns) / 1.0e6
            self._force_age_ms[side].append(age_ms)

        previous_stamp = self._previous_force_stamp_ns[side]
        if previous_stamp is not None and stamp_ns > previous_stamp:
            self._force_period_ms[side].append((stamp_ns - previous_stamp) / 1.0e6)
        self._previous_force_stamp_ns[side] = stamp_ns

        wrench = message.wrench
        self._force_samples[side].append(
            ForceSample(
                stamp_ns=stamp_ns,
                receipt_ros_ns=receipt_ros_ns,
                receipt_monotonic_ns=time.perf_counter_ns(),
                values=(
                    float(wrench.force.x),
                    float(wrench.force.y),
                    float(wrench.force.z),
                    float(wrench.torque.x),
                    float(wrench.torque.y),
                    float(wrench.torque.z),
                ),
                transport_age_ms=age_ms,
            )
        )

    def _relay_servo_command(self, message: Float64MultiArray) -> None:
        if self._robot_command_pub is not None:
            self._robot_command_pub.publish(message)

    def _drain_key_events(self) -> None:
        while True:
            try:
                event = self._key_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_key(event)

    def _handle_key(self, event: KeyInput) -> None:
        state_before = self._state.value

        if event.key == "q":
            self._publish_motion_halt()
            self._pending_forced_return_ns = None
            self._queue_key_event(event, state_before, "stopping", force_now=True)
            if self._active_episode_id is not None:
                self._discard_active_episode("quit_key")
            self._stop_event.set()
            return

        if event.key == "space":
            self._publish_motion_halt()
            self._pending_forced_return_ns = None
            if self._state in {
                MotionState.LOWERING,
                MotionState.STOPPING_AFTER_ENTER,
            }:
                self._lower_ended_monotonic_ns = event.monotonic_ns
            self._state = MotionState.STOPPED
            self._queue_key_event(
                event, state_before, self._state.value, force_now=True
            )
            if self._active_episode_id is not None:
                self._discard_active_episode("emergency_space")
            self.get_logger().warning(
                "Emergency motion stop requested with Space; episode discarded. "
                "Press Up to open the gripper and return home."
            )
            return

        if event.key == "up":
            if self._state in {
                MotionState.OPENING_FOR_RETURN,
                MotionState.RETURNING,
                MotionState.RETURNING_HOME,
            }:
                self._queue_key_event(event, state_before, state_before)
                self.get_logger().info(
                    f"Up arrow: return is already in progress ({self._state.value})."
                )
                return

            if self._state == MotionState.CLOSING:
                self._publish_motion_halt()
                self._pending_forced_return_ns = event.monotonic_ns
                self._queue_key_event(
                    event,
                    state_before,
                    "opening_after_close_then_returning",
                )
                self.get_logger().warning(
                    "Up arrow while closing: will open as soon as the current "
                    "gripper action finishes, then return home."
                )
                return

            self._queue_key_event(event, state_before, "opening_then_returning")
            self.get_logger().warning(
                "Up arrow: aborting the episode, opening the gripper, and "
                "returning home."
            )
            self._resolve_pending_events(force=True)
            if self._active_episode_id is not None:
                self._discard_active_episode("up_arrow")
            self._open_gripper_and_start_return(
                reference_monotonic_ns=event.monotonic_ns,
            )
            return

        if event.key == "right" and self._state == MotionState.READY:
            try:
                self._start_episode(event)
            except RuntimeError as exc:
                self._state = MotionState.ERROR
                self.get_logger().error(f"Failed to start episode: {exc}")
                return
            self._state = MotionState.CLOSING
            self._pending_forced_return_ns = None
            self._queue_key_event(event, state_before, "closing_then_lowering")
            self.get_logger().info(
                f"Episode {self._active_episode_id}: closing gripper and "
                "waiting for physical motion to finish."
            )
            self._queue_system_event(
                "gripper_close_start",
                MotionState.READY.value,
                self._state.value,
            )
            try:
                self._send_gripper_goal(
                    self._gripper_close_position, wait_for_result=True
                )
            except RuntimeError as exc:
                pending_return_ns = self._pending_forced_return_ns
                self._pending_forced_return_ns = None
                if pending_return_ns is not None:
                    self.get_logger().warning(
                        f"Close action failed ({exc}); attempting forced open."
                    )
                    self._discard_active_episode("gripper_close_failed")
                    self._open_gripper_and_start_return(
                        reference_monotonic_ns=pending_return_ns,
                    )
                    return
                if self._stop_event.is_set():
                    return
                self._state = MotionState.ERROR
                self._publish_motion_halt()
                self._discard_active_episode("gripper_close_failed")
                self.get_logger().error(str(exc))
                return
            if self._stop_event.is_set():
                return
            pending_return_ns = self._pending_forced_return_ns
            self._pending_forced_return_ns = None
            if pending_return_ns is not None:
                self._resolve_pending_events(force=True)
                self._discard_active_episode("up_arrow_during_close")
                self._open_gripper_and_start_return(
                    reference_monotonic_ns=pending_return_ns,
                )
                return
            if self._state != MotionState.CLOSING:
                return
            now_ns = time.perf_counter_ns()
            if self._post_grasp_hold_sec > 0.0:
                self._state = MotionState.POST_GRASP_HOLD
                self._phase_deadline_monotonic_ns = now_ns + int(
                    self._post_grasp_hold_sec * 1.0e9
                )
            elif self._pick_lift_duration_sec > 0.0:
                self._state = MotionState.LIFTING_AFTER_GRASP
                self._phase_deadline_monotonic_ns = now_ns + int(
                    self._pick_lift_duration_sec * 1.0e9
                )
            else:
                self._begin_lowering(now_ns)
            self._queue_system_event(
                "gripper_close_complete",
                MotionState.CLOSING.value,
                self._state.value,
            )
            if self._state == MotionState.POST_GRASP_HOLD:
                self.get_logger().info(
                    "Physical gripper motion complete; holding before lift."
                )
            elif self._state == MotionState.LIFTING_AFTER_GRASP:
                self._queue_system_event(
                    "pick_lift_start",
                    MotionState.CLOSING.value,
                    self._state.value,
                )
                self.get_logger().info(
                    "Physical gripper motion complete; pick lift started."
                )
            else:
                self.get_logger().info(
                    "Physical gripper motion complete; lowering started."
                )
                self._queue_system_event(
                    "place_descent_start",
                    MotionState.CLOSING.value,
                    self._state.value,
                )
            return

        if event.key == "enter" and self._state == MotionState.LOWERING:
            self._state = MotionState.STOPPING_AFTER_ENTER
            self._stop_deadline_monotonic_ns = event.monotonic_ns + int(
                self._enter_extra_lower_sec * 1.0e9
            )
            self._queue_key_event(event, state_before, self._state.value)
            if self._enter_event_alias and self._enter_event_alias != event.key:
                self._queue_system_event(
                    self._enter_event_alias,
                    state_before,
                    self._state.value,
                    timing=event,
                )
            self.get_logger().info(
                f"Enter: continuing down for {self._enter_extra_lower_sec:.3f} s."
            )
            return

        self._queue_key_event(event, state_before, state_before)
        self.get_logger().info(
            f"Key '{event.key}' ignored while state={self._state.value}."
        )

    def _control_tick(self) -> None:
        now_ns = time.perf_counter_ns()
        if (
            now_ns - self._last_motion_publish_monotonic_ns
            < int(self._control_period_sec * 1.0e9)
        ):
            return
        self._last_motion_publish_monotonic_ns = now_ns

        if self._state == MotionState.POST_GRASP_HOLD:
            if (
                self._phase_deadline_monotonic_ns is not None
                and now_ns >= self._phase_deadline_monotonic_ns
            ):
                previous = self._state.value
                if self._pick_lift_duration_sec > 0.0:
                    self._state = MotionState.LIFTING_AFTER_GRASP
                    self._phase_deadline_monotonic_ns = now_ns + int(
                        self._pick_lift_duration_sec * 1.0e9
                    )
                else:
                    self._begin_lowering(now_ns)
                self._queue_system_event(
                    "post_grasp_hold_complete", previous, self._state.value
                )
                if self._state == MotionState.LIFTING_AFTER_GRASP:
                    self._queue_system_event(
                        "pick_lift_start", previous, self._state.value
                    )
                    self.get_logger().info("Post-grasp hold complete; lifting.")
                else:
                    self._queue_system_event(
                        "place_descent_start", previous, self._state.value
                    )
                    self.get_logger().info(
                        "Post-grasp hold complete; lowering started."
                    )
            return

        if self._state == MotionState.LIFTING_AFTER_GRASP:
            if (
                self._phase_deadline_monotonic_ns is not None
                and now_ns >= self._phase_deadline_monotonic_ns
            ):
                self._publish_motion_halt()
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
                        "Pick lift complete; holding before placement descent."
                    )
                else:
                    self._queue_system_event(
                        "place_descent_start", previous, self._state.value
                    )
                    self.get_logger().info(
                        "Pick lift complete; placement descent started."
                    )
                return
            self._publish_twist(self._return_twist_z)
            return

        if self._state == MotionState.POST_LIFT_HOLD:
            if (
                self._phase_deadline_monotonic_ns is not None
                and now_ns >= self._phase_deadline_monotonic_ns
            ):
                previous = self._state.value
                self._begin_lowering(now_ns)
                self._queue_system_event(
                    "post_lift_hold_complete", previous, self._state.value
                )
                self._queue_system_event(
                    "place_descent_start", previous, self._state.value
                )
                self.get_logger().info(
                    "Post-lift hold complete; placement descent started."
                )
            return

        if self._state in {
            MotionState.LOWERING,
            MotionState.STOPPING_AFTER_ENTER,
        }:
            if (
                self._state == MotionState.STOPPING_AFTER_ENTER
                and self._stop_deadline_monotonic_ns is not None
                and now_ns >= self._stop_deadline_monotonic_ns
            ):
                self._publish_motion_halt()
                self._lower_ended_monotonic_ns = now_ns
                previous = self._state.value
                self._state = MotionState.FINALIZING_CAPTURE
                self._queue_system_event(
                    "post_enter_descent_complete", previous, self._state.value
                )
                self.get_logger().info(
                    f"{self._enter_extra_lower_sec:.3f}-second post-Enter descent "
                    "complete; opening gripper while episode recording continues."
                )
                self._open_gripper_and_start_return(
                    reference_monotonic_ns=now_ns,
                    complete_episode_on_open=True,
                )
                return

            if (
                self._lower_started_monotonic_ns is not None
                and now_ns - self._lower_started_monotonic_ns
                >= int(self._max_lower_duration_sec * 1.0e9)
            ):
                self._publish_motion_halt()
                self._lower_ended_monotonic_ns = now_ns
                previous = self._state.value
                self._state = MotionState.STOPPED
                self._queue_system_event("max_lower_duration", previous, self._state.value)
                self.get_logger().warning(
                    "Maximum lower duration reached; discarding the episode "
                    "and returning home."
                )
                self._resolve_pending_events(force=True)
                self._discard_active_episode("max_lower_duration")
                self._open_gripper_and_start_return(reference_monotonic_ns=now_ns)
                return
            self._publish_twist(self._lower_twist_z)
            return

        if self._state == MotionState.RETURNING:
            if (
                self._return_deadline_monotonic_ns is not None
                and now_ns < self._return_deadline_monotonic_ns
            ):
                self._publish_twist(self._return_twist_z)
                return
            self._publish_motion_halt()
            self._state = MotionState.RETURNING_HOME
            if self._move_to_home():
                self._reset_descent_tracking()
                self._state = MotionState.READY
                self.get_logger().info(
                    "Initial pose restored; READY for the next episode."
                )
            else:
                self._state = MotionState.ERROR

    def _begin_lowering(self, now_ns: int) -> None:
        """Enter the placement descent state and reset descent timing."""
        self._state = MotionState.LOWERING
        self._phase_deadline_monotonic_ns = None
        self._lower_started_monotonic_ns = now_ns
        self._lower_ended_monotonic_ns = None
        self._stop_deadline_monotonic_ns = None
        self._return_deadline_monotonic_ns = None

    def _descent_duration_ns(self, reference_monotonic_ns: int) -> int:
        if self._lower_started_monotonic_ns is None:
            return 0
        end_ns = self._lower_ended_monotonic_ns
        if end_ns is None:
            end_ns = reference_monotonic_ns
            self._lower_ended_monotonic_ns = end_ns
        return max(0, end_ns - self._lower_started_monotonic_ns)

    def _open_gripper_and_start_return(
        self,
        reference_monotonic_ns: int,
        complete_episode_on_open: bool = False,
        episode_status: str = "success",
        episode_reason: Optional[str] = None,
    ) -> None:
        descent_ns = self._descent_duration_ns(reference_monotonic_ns)
        self._publish_motion_halt()
        self._stop_deadline_monotonic_ns = None
        self._state = MotionState.OPENING_FOR_RETURN
        try:
            self._send_gripper_goal(
                self._gripper_open_position,
                wait_for_result=True,
            )
        except RuntimeError as exc:
            self._state = MotionState.ERROR
            if self._active_episode_id is not None:
                self._discard_active_episode("gripper_open_failed")
            self.get_logger().error(str(exc))
            return
        if self._stop_event.is_set() or self._state != MotionState.OPENING_FOR_RETURN:
            return

        if complete_episode_on_open:
            self._queue_system_event(
                "gripper_open_complete",
                MotionState.OPENING_FOR_RETURN.value,
                f"episode_{episode_status}",
            )
            self._resolve_pending_events(force=True)
            self._sleep_spin(0.05)
            self._complete_active_episode(
                status_name=episode_status,
                reason=episode_reason,
            )

        speed_ratio = abs(self._lower_twist_z) / max(
            abs(self._return_twist_z), 1.0e-9
        )
        return_ns = int(descent_ns * speed_ratio)
        self._return_deadline_monotonic_ns = time.perf_counter_ns() + return_ns
        self._state = MotionState.RETURNING
        self.get_logger().info(
            f"Gripper open; returning upward for {return_ns / 1e9:.3f} s, "
            "then restoring the initial joint pose."
        )

    def _reset_descent_tracking(self) -> None:
        self._lower_started_monotonic_ns = None
        self._lower_ended_monotonic_ns = None
        self._stop_deadline_monotonic_ns = None
        self._phase_deadline_monotonic_ns = None
        self._return_deadline_monotonic_ns = None
        self._pending_forced_return_ns = None

    def _publish_twist_xyz(
        self,
        x_command: float = 0.0,
        y_command: float = 0.0,
        z_command: float = 0.0,
    ) -> None:
        """Publish a Cartesian Servo command in the configured base frame."""
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._command_frame
        message.twist.linear.x = float(x_command)
        message.twist.linear.y = float(y_command)
        message.twist.linear.z = float(z_command)
        self._twist_pub.publish(message)

    def _publish_twist(self, z_command: float) -> None:
        self._publish_twist_xyz(z_command=z_command)

    def _publish_motion_halt(self) -> None:
        self._publish_twist(0.0)

    def _publish_joint_halt(self) -> None:
        message = JointJog()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._command_frame
        message.joint_names = list(self._home_pose_rad.keys())
        message.velocities = [0.0] * len(message.joint_names)
        self._joint_pub.publish(message)

    def _start_servo(self) -> None:
        if not self._servo_start_client.wait_for_service(
            timeout_sec=self._startup_timeout_sec
        ):
            raise RuntimeError(
                f"Servo start service unavailable: {self._servo_start_service}"
            )
        future = self._servo_start_client.call_async(Trigger.Request())
        if not self._spin_until_future_done(future, self._startup_timeout_sec):
            raise RuntimeError("Timed out while starting MoveIt Servo")
        response = future.result()
        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            raise RuntimeError(f"MoveIt Servo failed to start: {message}")

    def _wait_for_robot_program(self, timeout_sec: float) -> bool:
        self.get_logger().info(
            "Waiting for the External Control program; press Play on the UR pendant."
        )
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self._robot_program_running is True:
                self.get_logger().info("External Control program is running.")
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(
            "Timed out waiting for External Control on "
            f"{self._robot_program_running_topic}."
        )
        return False

    def _move_to_home(self) -> bool:
        if not self._wait_for_joint_state(self._startup_timeout_sec):
            return False

        deadline = time.monotonic() + self._home_timeout_sec
        settled_since: Optional[float] = None
        while rclpy.ok() and not self._stop_event.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            message = JointJog()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self._command_frame
            errors: dict[str, float] = {}

            for name, target in self._home_pose_rad.items():
                current = self._joint_positions.get(name)
                if current is None:
                    continue
                error = target - current
                errors[name] = error
                if abs(error) < self._home_deadband:
                    continue
                command = error * self._home_velocity_gain
                sign = 1.0 if command >= 0.0 else -1.0
                magnitude = min(
                    self._home_max_joint_speed,
                    max(self._home_min_joint_speed, abs(command)),
                )
                message.joint_names.append(name)
                message.velocities.append(sign * magnitude)

            if not message.joint_names:
                self._publish_joint_halt()
                velocities = [
                    abs(self._joint_velocities.get(name, float("inf")))
                    for name in self._home_pose_rad
                ]
                max_velocity = max(velocities, default=float("inf"))
                if max_velocity <= self._home_velocity_tolerance_rad_sec:
                    if settled_since is None:
                        settled_since = now
                    if now - settled_since >= self._home_settle_duration_sec:
                        max_error = max(
                            (abs(error) for error in errors.values()),
                            default=0.0,
                        )
                        self.get_logger().info(
                            "Initial pose settled: "
                            f"max error={max_error * 1000.0:.3f} mrad, "
                            f"max velocity={max_velocity * 1000.0:.3f} mrad/s."
                        )
                        return True
                else:
                    settled_since = None
            else:
                settled_since = None
                self._joint_pub.publish(message)

            rclpy.spin_once(self, timeout_sec=0.02)
            self._drain_key_events()
            self._resolve_pending_events()

        self._publish_joint_halt()
        errors = {
            name: target - self._joint_positions.get(name, target)
            for name, target in self._home_pose_rad.items()
        }
        velocities = {
            name: abs(self._joint_velocities.get(name, float("inf")))
            for name in self._home_pose_rad
        }
        worst_error_joint = max(errors, key=lambda name: abs(errors[name]))
        worst_velocity_joint = max(velocities, key=velocities.get)
        self.get_logger().error(
            "Timed out while moving to the initial pose: "
            f"max error={abs(errors[worst_error_joint]) * 1000.0:.3f} mrad "
            f"({worst_error_joint}, limit={self._home_deadband * 1000.0:.3f}), "
            f"max velocity={velocities[worst_velocity_joint] * 1000.0:.3f} "
            f"mrad/s ({worst_velocity_joint}, limit="
            f"{self._home_velocity_tolerance_rad_sec * 1000.0:.3f})."
        )
        return False

    def _send_gripper_goal(
        self, target_position: float, wait_for_result: bool
    ) -> None:
        log_message = Float32MultiArray()
        log_message.data = [
            float(target_position),
            self._gripper_target_speed,
            self._gripper_target_force,
        ]
        self._gripper_goal_pub.publish(log_message)

        if not self._gripper_action_client.wait_for_server(
            timeout_sec=self._startup_timeout_sec
        ):
            raise RuntimeError(
                f"Gripper action unavailable: {self._gripper_action_name}"
            )

        goal = MoveTwoFingerGripper.Goal()
        goal.target_position = float(target_position)
        goal.target_speed = self._gripper_target_speed
        goal.target_force = self._gripper_target_force
        goal_future = self._gripper_action_client.send_goal_async(goal)
        if not self._spin_until_future_done(goal_future, 3.0):
            raise RuntimeError("Timed out sending the gripper goal")
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Gripper goal was rejected")

        if wait_for_result:
            result_future = goal_handle.get_result_async()
            if not self._spin_until_future_done(
                result_future, self._gripper_result_timeout_sec
            ):
                raise RuntimeError("Timed out waiting for gripper motion")
            wrapped_result = result_future.result()
            if (
                wrapped_result is None
                or wrapped_result.result is None
                or not wrapped_result.result.success
            ):
                raise RuntimeError("Gripper motion did not complete successfully")

    def _wait_for_joint_state(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self._joint_state_ready:
                missing = [
                    name
                    for name in self._home_pose_rad
                    if name not in self._joint_positions
                ]
                if missing:
                    self.get_logger().error(
                        f"Joint state is missing configured joints: {missing}"
                    )
                    return False
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(
            f"Timed out waiting for joint states on {self._joint_state_topic}."
        )
        return False

    def _wait_for_capture_sources(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            force_ready = all(self._force_samples[side] for side in ("left", "right"))
            camera_ready = all(self._camera_ready.values())
            if (
                (force_ready or not self._require_force_sensors)
                and (camera_ready or not self._require_cameras)
            ):
                for subscription in self._camera_readiness_subscriptions:
                    self.destroy_subscription(subscription)
                self._camera_readiness_subscriptions.clear()
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        missing = []
        if self._require_force_sensors:
            if not self._force_samples["left"]:
                missing.append(self._left_ft_topic)
            if not self._force_samples["right"]:
                missing.append(self._right_ft_topic)
        if self._require_cameras:
            if not self._camera_ready["wrist"]:
                missing.append(self._camera_wrist_topic)
            if not self._camera_ready["fixed"]:
                missing.append(self._camera_fixed_topic)
        self.get_logger().error(
            f"Timed out waiting for capture sources: {missing}"
        )
        return False

    def _spin_until_future_done(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while (
            rclpy.ok()
            and not future.done()
            and not self._stop_event.is_set()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.02)
            self._drain_key_events()
            self._resolve_pending_events()
        return future.done()

    def _sleep_spin(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _ensure_event_file(self) -> None:
        path = self._event_csv_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._EVENT_FIELDS).writeheader()

    def _event_csv_path(self) -> Optional[Path]:
        if self._active_episode_id is None or self._session_dir is None:
            return None
        return self._session_dir / "csv" / "key_events.csv"

    def _queue_key_event(
        self,
        event: KeyInput,
        state_before: str,
        state_after: str,
        force_now: bool = False,
    ) -> None:
        self._event_id += 1
        wait_ns = 0 if force_now else int(self._event_force_wait_sec * 1.0e9)
        self._pending_events.append(
            PendingKeyEvent(
                event_id=self._event_id,
                input=event,
                source="keyboard",
                state_before=state_before,
                state_after=state_after,
                episode_id=self._active_episode_id,
                resolve_after_monotonic_ns=event.monotonic_ns + wait_ns,
            )
        )
        self._resolve_pending_events()

    def _queue_system_event(
        self,
        name: str,
        state_before: str,
        state_after: str,
        timing: Optional[KeyInput] = None,
    ) -> None:
        if timing is None:
            event = KeyInput(
                key=name,
                ros_stamp_ns=int(self.get_clock().now().nanoseconds),
                monotonic_ns=time.perf_counter_ns(),
            )
        else:
            event = KeyInput(
                key=name,
                ros_stamp_ns=timing.ros_stamp_ns,
                monotonic_ns=timing.monotonic_ns,
            )
        self._event_id += 1
        self._pending_events.append(
            PendingKeyEvent(
                event_id=self._event_id,
                input=event,
                source="system",
                state_before=state_before,
                state_after=state_after,
                episode_id=self._active_episode_id,
                resolve_after_monotonic_ns=event.monotonic_ns,
            )
        )

    def _resolve_pending_events(self, force: bool = False) -> None:
        now_ns = time.perf_counter_ns()
        remaining: list[PendingKeyEvent] = []
        for event in self._pending_events:
            if not force and now_ns < event.resolve_after_monotonic_ns:
                remaining.append(event)
                continue
            self._write_event(event)
        self._pending_events = remaining

    def _nearest_force_sample(
        self, side: str, event_stamp_ns: int
    ) -> Optional[ForceSample]:
        samples = self._force_samples[side]
        if not samples:
            return None
        return min(samples, key=lambda sample: abs(sample.stamp_ns - event_stamp_ns))

    def _write_event(self, event: PendingKeyEvent) -> None:
        row: dict[str, object] = {
            "event_id": event.event_id,
            "episode_id": event.episode_id,
            "source": event.source,
            "key": event.input.key,
            "stamp_ns": event.input.ros_stamp_ns,
            "timestamp_sec": event.input.ros_stamp_ns / 1.0e9,
            "monotonic_ns": event.input.monotonic_ns,
            "state_before": event.state_before,
            "state_after": event.state_after,
        }
        for side in ("left", "right"):
            sample = self._nearest_force_sample(side, event.input.ros_stamp_ns)
            if sample is None:
                continue
            row[f"{side}_stamp_ns"] = sample.stamp_ns
            row[f"{side}_delta_ms"] = (
                sample.stamp_ns - event.input.ros_stamp_ns
            ) / 1.0e6
            row[f"{side}_transport_age_ms"] = sample.transport_age_ms
            for suffix, value in zip(
                ("fx", "fy", "fz", "tx", "ty", "tz"), sample.values
            ):
                row[f"{side}_{suffix}"] = value

        paths = [self._run_event_csv_path, self._event_csv_path()]
        with self._event_lock:
            for path in dict.fromkeys(path for path in paths if path is not None):
                with path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=self._EVENT_FIELDS)
                    writer.writerow(row)

        json_row = {
            key: (None if value == "" else value) for key, value in row.items()
        }
        self._key_event_pub.publish(
            String(data=json.dumps(json_row, ensure_ascii=False, allow_nan=False))
        )

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return ordered[index]

    def _write_force_latency_summary(self) -> None:
        if self._session_dir is None:
            return
        path = self._session_dir / "csv" / "force_latency_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "side",
            "sample_count",
            "measured_rate_hz",
            "period_mean_ms",
            "period_p95_ms",
            "period_max_ms",
            "transport_age_mean_ms",
            "transport_age_p95_ms",
            "transport_age_max_ms",
            "period_over_15ms_count",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for side in ("left", "right"):
                periods = self._force_period_ms[side]
                ages = self._force_age_ms[side]
                mean_period = statistics.fmean(periods) if periods else None
                row = {
                    "side": side,
                    "sample_count": self._force_sample_count[side],
                    "measured_rate_hz": (
                        1000.0 / mean_period if mean_period and mean_period > 0 else None
                    ),
                    "period_mean_ms": mean_period,
                    "period_p95_ms": self._percentile(periods, 0.95),
                    "period_max_ms": max(periods) if periods else None,
                    "transport_age_mean_ms": statistics.fmean(ages) if ages else None,
                    "transport_age_p95_ms": self._percentile(ages, 0.95),
                    "transport_age_max_ms": max(ages) if ages else None,
                    "period_over_15ms_count": sum(value > 15.0 for value in periods),
                }
                writer.writerow(row)
                if mean_period:
                    self.get_logger().info(
                        f"FT {side}: {1000.0 / mean_period:.2f} Hz, "
                        f"DDS age p95={self._percentile(ages, 0.95)} ms"
                    )


def main(args: Optional[list[str]] = None) -> None:
    """Run the keyboard motion capture node."""
    rclpy.init(args=args)
    node = KeyMotionCaptureNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
