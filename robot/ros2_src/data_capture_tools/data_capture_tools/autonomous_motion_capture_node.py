"""Record a fixed autonomous motion sequence with camera and force data."""

from __future__ import annotations

import csv
import base64
import json
import math
import os
from pathlib import Path
import select
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import cv2
import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped, WrenchStamped
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, JointState
from robotiq_2f_gripper_msgs.action import MoveTwoFingerGripper
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger

from .data_capture_node import DataCaptureNode


def _build_patchtst_model(
    patchtst_cls,
    nvars: int = 12,
    context_points: int = 80,
    patch_len: int = 10,
    stride: int = 10,
    n_layers: int = 3,
    n_heads: int = 16,
    d_model: int = 128,
    d_ff: int = 768,
):
    num_patch = (max(context_points, patch_len) - patch_len) // stride + 1
    return patchtst_cls(
        c_in=nvars,
        target_dim=96,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        shared_embedding=True,
        d_ff=d_ff,
        dropout=0.0,
        head_dropout=0.0,
        act="gelu",
        head_type="pretrain",
        res_attention=False,
    )


class AutonomousMotionCaptureNode(DataCaptureNode):
    """Run a fixed robot motion while using DataCaptureNode recording."""

    def __init__(self) -> None:
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
            math.radians(45.0),
            math.radians(-90.0),
            math.radians(90.0),
            math.radians(-90.0),
            math.radians(-90.0),
            0.0,
        ]
        home_joint_names = list(
            self.declare_parameter("home_joint_names", default_joint_names)
            .get_parameter_value()
            .string_array_value
        )
        home_positions = list(
            self.declare_parameter("home_positions_rad", default_home_positions)
            .get_parameter_value()
            .double_array_value
        )
        if len(home_joint_names) != len(home_positions):
            raise ValueError(
                "home_joint_names and home_positions_rad must have the same length"
            )
        self._home_pose_rad = dict(zip(home_joint_names, home_positions))

        self._joint_state_topic = self.declare_parameter(
            "joint_state_topic", "/joint_states"
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
        self._command_frame = self.declare_parameter("command_frame", "base_link").value

        self._home_timeout_sec = float(
            self.declare_parameter("home_timeout_sec", 20.0).value
        )
        self._home_deadband = float(
            self.declare_parameter("home_deadband", 0.01).value
        )
        self._home_min_joint_speed = float(
            self.declare_parameter("home_min_joint_speed", 0.5).value
        )
        self._home_max_joint_speed = float(
            self.declare_parameter("home_max_joint_speed", 8.0).value
        )
        self._home_velocity_gain = float(
            self.declare_parameter("home_velocity_gain", 2.0).value
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
            self.declare_parameter("gripper_target_speed", 0.01).value
        )
        self._gripper_target_force = float(
            self.declare_parameter("gripper_target_force", 0.1).value
        )
        self._gripper_result_timeout_sec = float(
            self.declare_parameter("gripper_result_timeout_sec", 5.0).value
        )

        self._pre_motion_recording_delay_sec = float(
            self.declare_parameter("pre_motion_recording_delay_sec", 1.0).value
        )
        self._closed_wait_sec = float(
            self.declare_parameter("closed_wait_sec", 5.0).value
        )
        self._post_open_wait_sec = float(
            self.declare_parameter("post_open_wait_sec", 5.0).value
        )
        self._lower_distance_m = float(
            self.declare_parameter("lower_distance_m", 0.05).value
        )
        self._lower_velocity_scale_mps = float(
            self.declare_parameter("lower_velocity_scale_mps", 0.03).value
        )
        self._lower_twist_command_z = float(
            self.declare_parameter("lower_twist_command_z", -1.0).value
        )
        self._safety_extra_lower_m = float(
            self.declare_parameter("safety_extra_lower_m", 0.04).value
        )
        self._open_on_safety_limit = bool(
            self.declare_parameter("open_on_safety_limit", True).value
        )
        self._release_mode = str(
            self.declare_parameter("release_mode", "patchtst").value
        ).strip().lower()
        if self._release_mode not in {"patchtst", "force_spike", "openai_vision"}:
            raise ValueError(
                "release_mode must be one of: patchtst, force_spike, openai_vision"
            )

        self._left_ft_topic = self.declare_parameter(
            "left_ft_topic", "/force_torque/left"
        ).value
        self._right_ft_topic = self.declare_parameter(
            "right_ft_topic", "/force_torque/right"
        ).value
        self._force_sample_period_sec = float(
            self.declare_parameter("force_sample_period_sec", 0.01).value
        )
        self._force_wait_timeout_sec = float(
            self.declare_parameter("force_wait_timeout_sec", 5.0).value
        )
        self._force_offset_sample_count = int(
            self.declare_parameter("force_offset_sample_count", 20).value
        )
        self._force_offset_sample_period_sec = float(
            self.declare_parameter("force_offset_sample_period_sec", 0.01).value
        )
        self._force_spike_threshold = float(
            self.declare_parameter("force_spike_threshold", 3.0).value
        )
        self._patchtst_enabled = bool(
            self.declare_parameter("patchtst_enabled", True).value
        )
        self._patchtst_root = resolve_workspace_path(
            self.declare_parameter(
                "patchtst_root",
                "PatchTST/PatchTST_self_supervised",
            ).value
        )
        self._patchtst_model_path = resolve_workspace_path(
            self.declare_parameter(
                "patchtst_model_path",
                "artifacts/robot/twilight-music-38_epoch4000.pth",
            ).value
        )
        self._patchtst_text_emb_dir = resolve_workspace_path(
            self.declare_parameter(
                "patchtst_text_emb_dir", "artifacts/robot/text_emb"
            ).value
        )
        self._patchtst_device = self.declare_parameter(
            "patchtst_device", "auto"
        ).value
        self._patchtst_seq_len = int(
            self.declare_parameter("patchtst_seq_len", 80).value
        )
        self._patchtst_min_samples = int(
            self.declare_parameter("patchtst_min_samples", 40).value
        )
        self._patchtst_eval_period_sec = float(
            self.declare_parameter("patchtst_eval_period_sec", 0.1).value
        )
        self._patchtst_margin = float(
            self.declare_parameter("patchtst_place_margin", 0.0).value
        )
        self._record_enter_events = bool(
            self.declare_parameter("record_enter_events", True).value
        )
        self._vision_image_topic = self.declare_parameter(
            "vision_image_topic", "/camera_wrist/realsense2_camera/color/image_raw"
        ).value
        self._vision_eval_period_sec = float(
            self.declare_parameter("vision_eval_period_sec", 0.1).value
        )
        self._vision_jpeg_quality = int(
            self.declare_parameter("vision_jpeg_quality", 75).value
        )
        self._openai_model = self.declare_parameter(
            "openai_model", "gpt-4o"
        ).value
        self._openai_api_key = self.declare_parameter("openai_api_key", "").value

        qos = QoSProfile(depth=10)
        self._joint_positions: dict[str, float] = {}
        self._joint_state_ready = False
        self._latest_left_ft: list[float] | None = None
        self._latest_right_ft: list[float] | None = None
        self._force_offset = np.zeros(12, dtype=np.float32)
        self._force_buffer: deque[list[float]] = deque(maxlen=self._patchtst_seq_len)
        self._score_trace: list[dict[str, object]] = []
        self._latest_vision_image: np.ndarray | None = None
        self._latest_vision_stamp_ns: int | None = None
        self._last_openai_image_stamp_ns: int | None = None
        self._openai_client = None
        self._enter_event_lock = threading.Lock()
        self._enter_event_thread: threading.Thread | None = None
        self._patchtst_model = None
        self._patchtst_proj = None
        self._patchtst_create_patch = None
        self._hold_vec = None
        self._place_vec = None
        self.create_subscription(
            JointState, self._joint_state_topic, self._joint_state_cb, qos
        )
        self.create_subscription(
            WrenchStamped,
            self._left_ft_topic,
            lambda msg: self._ft_cb("left", msg),
            qos,
        )
        self.create_subscription(
            WrenchStamped,
            self._right_ft_topic,
            lambda msg: self._ft_cb("right", msg),
            qos,
        )
        self.create_timer(self._force_sample_period_sec, self._sample_force)
        if self._release_mode == "openai_vision":
            self.create_subscription(
                Image,
                self._vision_image_topic,
                self._vision_image_cb,
                qos,
            )
        self._joint_pub = self.create_publisher(
            JointJog, self._joint_command_topic, qos
        )
        self._twist_pub = self.create_publisher(
            TwistStamped, self._twist_command_topic, qos
        )
        self._gripper_goal_pub = self.create_publisher(
            Float32MultiArray, self._gripper_goal_log_topic, qos
        )
        self._servo_start_client = self.create_client(
            Trigger, self._servo_start_service
        )
        self._gripper_action_client = ActionClient(
            self, MoveTwoFingerGripper, self._gripper_action_name
        )

        if self._release_mode == "patchtst" and self._patchtst_enabled:
            self._load_patchtst()
        if self._release_mode == "openai_vision":
            self._load_openai_client()

    def run(self) -> None:
        self._force_save = True
        self._prepare_session()
        self._start_enter_event_thread()
        self._cleanup_existing_viewers()
        self._start_internal_viewers()
        self._start_throttles_and_compressors()
        self._start_rosbag()

        try:
            self._sleep_spin(self._pre_motion_recording_delay_sec)
            self._start_servo()
            self._run_sequence()
        except KeyboardInterrupt:
            self.get_logger().info(
                "Keyboard interrupt received, stopping autonomous capture."
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Autonomous motion failed: {exc}")
        finally:
            self._stop_event.set()
            self._record_stop_time_if_missing()
            self._finalize()

    def _joint_state_cb(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                self._joint_positions[name] = msg.position[idx]
        self._joint_state_ready = True

    def _ft_cb(self, side: str, msg: WrenchStamped) -> None:
        values = [
            float(msg.wrench.force.x),
            float(msg.wrench.force.y),
            float(msg.wrench.force.z),
            float(msg.wrench.torque.x),
            float(msg.wrench.torque.y),
            float(msg.wrench.torque.z),
        ]
        if side == "left":
            self._latest_left_ft = values
        else:
            self._latest_right_ft = values

    def _sample_force(self) -> None:
        if self._latest_left_ft is None or self._latest_right_ft is None:
            return
        raw = np.asarray(self._latest_left_ft + self._latest_right_ft, dtype=np.float32)
        corrected = raw - self._force_offset
        self._force_buffer.append(corrected.tolist())

    def _vision_image_cb(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"Failed to decode vision image: {exc}")
            return
        self._latest_vision_image = bgr
        self._latest_vision_stamp_ns = int(
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )

    def _run_sequence(self) -> None:
        self.get_logger().info("Sequence: move to initial pose.")
        if not self._move_to_home():
            raise RuntimeError("Failed to reach initial pose")

        self.get_logger().info("Sequence: collect force-torque offset.")
        self._collect_force_offset()

        self.get_logger().info("Sequence: close gripper.")
        self._send_gripper_goal(self._gripper_close_position, wait_for_result=True)
        self.get_logger().info(
            "Sequence: gripper closed; hold still for "
            f"{self._closed_wait_sec:.2f} s before release evaluation."
        )
        self._sleep_spin(self._closed_wait_sec)
        if self._release_mode == "openai_vision":
            self._latest_vision_image = None
            self._latest_vision_stamp_ns = None
            self._last_openai_image_stamp_ns = None

        release_reason = self._adaptive_lower_and_release()
        self.get_logger().info(f"Release sequence finished: {release_reason}")
        self.get_logger().info(
            f"Waiting {self._post_open_wait_sec:.2f}s after gripper open, then ending."
        )
        self._sleep_spin(self._post_open_wait_sec)
        self.get_logger().info("Autonomous sequence complete.")
        self._write_score_trace()

    def _load_patchtst(self) -> None:
        if str(self._patchtst_root) not in sys.path:
            sys.path.insert(0, str(self._patchtst_root))
        try:
            import torch
            from src.callback.patch_mask import create_patch
            from src.models.patchTST import PatchTST
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to import PatchTST from {self._patchtst_root}: {exc}"
            ) from exc

        if not self._patchtst_model_path.is_file():
            raise FileNotFoundError(
                f"PatchTST checkpoint not found: {self._patchtst_model_path}"
            )
        if not self._patchtst_text_emb_dir.is_dir():
            raise FileNotFoundError(
                f"Text embedding dir not found: {self._patchtst_text_emb_dir}"
            )

        if self._patchtst_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(str(self._patchtst_device))

        model = _build_patchtst_model(PatchTST).to(device)
        proj = None
        ckpt_all = torch.load(self._patchtst_model_path, map_location=device)
        state = ckpt_all.get("model", ckpt_all)
        if any(key.startswith("proj.") for key in state.keys()):
            if "proj.3.weight" in state and "proj.0.weight" in state:
                in_dim = state["proj.0.weight"].shape[1]
                hidden = state["proj.0.weight"].shape[0]
                out_dim = state["proj.3.weight"].shape[0]
                proj = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.0),
                    torch.nn.Linear(hidden, out_dim),
                ).to(device)
            elif "proj.0.weight" in state:
                in_dim = state["proj.0.weight"].shape[1]
                out_dim = state["proj.0.weight"].shape[0]
                proj = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim)).to(device)

        cur = model.state_dict()
        for key, value in state.items():
            if (
                key.startswith("backbone.")
                and key in cur
                and cur[key].shape == value.shape
            ):
                cur[key] = value
        model.load_state_dict(cur, strict=False)

        if proj is not None:
            p_state = proj.state_dict()
            for key, value in state.items():
                if not key.startswith("proj."):
                    continue
                proj_key = key.replace("proj.", "")
                if proj_key in p_state and p_state[proj_key].shape == value.shape:
                    p_state[proj_key] = value
            proj.load_state_dict(p_state, strict=False)
            proj.eval()

        model.eval()
        self._patchtst_model = model
        self._patchtst_proj = proj
        self._patchtst_create_patch = create_patch
        self._hold_vec = self._load_text_embedding("hold", torch)
        self._place_vec = self._load_text_embedding("place", torch)
        self.get_logger().info(
            f"Loaded PatchTST model: {self._patchtst_model_path} on {device}"
        )

    def _load_text_embedding(self, label: str, torch):
        path = self._patchtst_text_emb_dir / f"{label}.pt"
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            tensor = tensor.float()
        else:
            tensor = torch.tensor(tensor, dtype=torch.float32)
        return tensor / (tensor.norm() + 1e-8)

    def _load_openai_client(self) -> None:
        api_key = str(self._openai_api_key).strip() or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI Vision mode requires openai_api_key or OPENAI_API_KEY"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Vision mode requires the openai package"
            ) from exc
        self._openai_client = OpenAI(api_key=api_key)
        self.get_logger().info(
            f"OpenAI Vision release mode initialized with model {self._openai_model}"
        )

    def _encode_latest_vision_image(self) -> str | None:
        if self._latest_vision_image is None:
            return None
        quality = int(max(30, min(95, self._vision_jpeg_quality)))
        ok, encoded = cv2.imencode(
            ".jpg",
            self._latest_vision_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return None
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _parse_touch_json(text: str) -> bool:
        try:
            payload = json.loads(text)
            value = payload.get("touching_ground", False)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes", "1"}
            return bool(value)
        except Exception:
            lowered = text.strip().lower()
            return "true" in lowered or "yes" in lowered

    def _classify_touching_ground_with_openai(
        self,
    ) -> tuple[bool, str, float, int, int] | None:
        if self._openai_client is None:
            return None
        image_url = self._encode_latest_vision_image()
        if image_url is None:
            return None
        prompt = (
            "You are judging a robot placement scene. "
            "Answer whether the grasped object is already touching the support "
            "surface. Return strict JSON only: "
            "{\"touching_ground\": true/false, \"confidence\": 0..1, "
            "\"reason\": \"short\"}."
        )
        request_start_ns = int(self.get_clock().now().nanoseconds)
        request_start_mono = time.perf_counter_ns()
        response = self._openai_client.responses.create(
            model=self._openai_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            temperature=0,
            max_output_tokens=120,
        )
        request_end_ns = int(self.get_clock().now().nanoseconds)
        latency_sec = float(time.perf_counter_ns() - request_start_mono) / 1e9
        text = getattr(response, "output_text", "") or ""
        return (
            self._parse_touch_json(text),
            text,
            latency_sec,
            request_start_ns,
            request_end_ns,
        )

    def _current_force_raw(self) -> np.ndarray | None:
        if self._latest_left_ft is None or self._latest_right_ft is None:
            return None
        return np.asarray(
            self._latest_left_ft + self._latest_right_ft,
            dtype=np.float32,
        )

    def _latest_corrected_force_vector(self) -> np.ndarray | None:
        raw = self._current_force_raw()
        if raw is None:
            return None
        return raw - self._force_offset

    def _collect_force_offset(self) -> None:
        if self._force_offset_sample_count <= 0:
            self._force_offset = np.zeros(12, dtype=np.float32)
            self.get_logger().info("Force-torque offset disabled.")
            return
        if not self._wait_for_force_samples():
            raise RuntimeError(
                "Force-torque samples are required for offset calibration"
            )

        samples: list[np.ndarray] = []
        deadline = time.monotonic() + self._force_wait_timeout_sec
        while (
            rclpy.ok()
            and len(samples) < self._force_offset_sample_count
            and time.monotonic() < deadline
        ):
            raw = self._current_force_raw()
            if raw is not None:
                samples.append(raw)
            rclpy.spin_once(self, timeout_sec=self._force_offset_sample_period_sec)

        if len(samples) < self._force_offset_sample_count:
            raise RuntimeError(
                "Failed to collect enough force-torque offset samples: "
                f"{len(samples)}/{self._force_offset_sample_count}"
            )

        self._force_offset = np.mean(np.stack(samples, axis=0), axis=0).astype(
            np.float32
        )
        self._force_buffer.clear()
        self._write_force_offset()
        self.get_logger().info(
            "Force-torque offset set from "
            f"{len(samples)} samples: {self._force_offset.tolist()}"
        )

    def _write_force_offset(self) -> None:
        if self._session_dir is None:
            return
        csv_dir = self._session_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        path = csv_dir / "force_torque_offset.csv"
        fieldnames = [
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
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    name: float(value)
                    for name, value in zip(fieldnames, self._force_offset)
                }
            )

    def _start_enter_event_thread(self) -> None:
        if not self._record_enter_events:
            return
        self._ensure_enter_events_file()
        if not sys.stdin.isatty():
            self.get_logger().warning(
                "stdin is not a TTY; Enter event timestamps will not be recorded."
            )
            return
        self._enter_event_thread = threading.Thread(
            target=self._enter_event_loop, daemon=True
        )
        self._enter_event_thread.start()
        self.get_logger().info(
            "Press Enter during execution to log an event timestamp."
        )

    def _ensure_enter_events_file(self) -> None:
        if self._session_dir is None:
            return
        path = self._session_dir / "csv" / "enter_events.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["stamp_ns", "timestamp_sec", "monotonic_ns"],
            )
            writer.writeheader()

    def _enter_event_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Failed to read Enter event input: {exc}")
                return
            if not readable:
                continue
            line = sys.stdin.readline()
            if line == "":
                return
            if line.strip() == "":
                self._record_enter_event()

    def _record_enter_event(self) -> None:
        try:
            stamp_ns = int(self.get_clock().now().nanoseconds)
        except Exception:
            stamp_ns = int(time.time() * 1e9)
        row = {
            "stamp_ns": stamp_ns,
            "timestamp_sec": float(stamp_ns) / 1e9,
            "monotonic_ns": time.perf_counter_ns(),
        }

        if self._session_dir is None:
            return
        path = self._session_dir / "csv" / "enter_events.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._enter_event_lock:
            file_exists = path.is_file()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["stamp_ns", "timestamp_sec", "monotonic_ns"],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        self.get_logger().info(f"Enter event recorded at {stamp_ns} ns.")

    def _start_servo(self) -> None:
        if not self._servo_start_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warning(
                f"Servo start service unavailable: {self._servo_start_service}"
            )
            return
        future = self._servo_start_client.call_async(Trigger.Request())
        self._spin_until_future_done(future, timeout_sec=2.0)

    def _move_to_home(self) -> bool:
        if not self._wait_for_joint_state(timeout_sec=5.0):
            return False

        start_time = time.monotonic()
        while rclpy.ok() and time.monotonic() - start_time < self._home_timeout_sec:
            jog = JointJog()
            jog.header.stamp = self.get_clock().now().to_msg()
            jog.header.frame_id = self._command_frame

            for name, target in self._home_pose_rad.items():
                current = self._joint_positions.get(name)
                if current is None:
                    continue
                error = target - current
                if abs(error) < self._home_deadband:
                    continue
                velocity = error * self._home_velocity_gain
                sign = 1.0 if velocity >= 0.0 else -1.0
                magnitude = min(
                    self._home_max_joint_speed,
                    max(self._home_min_joint_speed, abs(velocity)),
                )
                jog.joint_names.append(name)
                jog.velocities.append(sign * magnitude)

            if not jog.joint_names:
                self._publish_joint_halt()
                return True

            self._joint_pub.publish(jog)
            rclpy.spin_once(self, timeout_sec=0.05)

        self._publish_joint_halt()
        self.get_logger().error("Timed out while moving to initial pose.")
        return False

    def _publish_joint_halt(self) -> None:
        halt = JointJog()
        halt.header.stamp = self.get_clock().now().to_msg()
        halt.header.frame_id = self._command_frame
        halt.joint_names = list(self._home_pose_rad.keys())
        halt.velocities = [0.0] * len(halt.joint_names)
        self._joint_pub.publish(halt)

    def _adaptive_lower_and_release(self) -> str:
        if self._release_mode == "patchtst" and self._patchtst_model is None:
            raise RuntimeError("PatchTST is enabled but the model was not loaded")
        if self._release_mode in {"patchtst", "force_spike"}:
            if not self._wait_for_force_samples():
                raise RuntimeError(
                    "Force-torque samples are required for this release mode"
                )
        if self._release_mode == "openai_vision" and not self._wait_for_vision_image():
            raise RuntimeError("Vision image samples are required for OpenAI release")

        self._force_buffer.clear()
        self._score_trace.clear()
        nominal_release_distance = max(0.0, abs(self._lower_distance_m))
        max_descent = nominal_release_distance + max(0.0, self._safety_extra_lower_m)
        release_reason = "safety_limit_reached"
        released = False

        self.get_logger().info(
            "Sequence: adaptive lower; nominal release "
            f"{nominal_release_distance:.3f} m, safety limit {max_descent:.3f} m."
        )

        speed = abs(self._lower_velocity_scale_mps * self._lower_twist_command_z)
        if speed <= 0.0:
            raise RuntimeError("lower velocity must be non-zero")
        duration_sec = max_descent / speed
        start_time = time.monotonic()
        motion_elapsed_sec = 0.0
        last_motion_time = start_time
        next_eval_time = start_time
        prev_fz_sum_abs: float | None = None
        while rclpy.ok() and motion_elapsed_sec < duration_sec:
            loop_time = time.monotonic()
            motion_elapsed_sec += max(0.0, loop_time - last_motion_time)
            last_motion_time = loop_time

            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self._command_frame
            twist.twist.linear.z = self._lower_twist_command_z
            self._twist_pub.publish(twist)

            now = time.monotonic()
            if now >= next_eval_time:
                elapsed = motion_elapsed_sec
                descent = min(max_descent, motion_elapsed_sec * speed)
                stamp_ns = int(self.get_clock().now().nanoseconds)
                released_now = False
                reason = ""

                if self._release_mode == "patchtst":
                    scores = self._classify_hold_place()
                    if scores is None:
                        next_eval_time = now + self._patchtst_eval_period_sec
                        rclpy.spin_once(self, timeout_sec=0.02)
                        continue
                    hold_score, place_score = scores
                    row = {
                        "stamp_ns": stamp_ns,
                        "elapsed_sec": float(elapsed),
                        "descent_m": float(descent),
                        "hold_score": float(hold_score),
                        "place_score": float(place_score),
                        "released": False,
                        "reason": "",
                    }
                    released_now = place_score > hold_score + self._patchtst_margin
                    reason = "place_gt_hold"
                    self.get_logger().info(
                        "PatchTST scores: "
                        f"hold={hold_score:.4f}, place={place_score:.4f}, "
                        f"descent={descent:.3f} m"
                    )
                    next_eval_time = now + self._patchtst_eval_period_sec

                elif self._release_mode == "force_spike":
                    vector = self._latest_corrected_force_vector()
                    if vector is None:
                        next_eval_time = now + self._force_sample_period_sec
                        rclpy.spin_once(self, timeout_sec=0.02)
                        continue
                    fz_sum_abs = float(abs(vector[2]) + abs(vector[8]))
                    spike = (
                        0.0
                        if prev_fz_sum_abs is None
                        else float(abs(fz_sum_abs - prev_fz_sum_abs))
                    )
                    prev_fz_sum_abs = fz_sum_abs
                    row = {
                        "stamp_ns": stamp_ns,
                        "elapsed_sec": float(elapsed),
                        "descent_m": float(descent),
                        "fz_sum_abs": fz_sum_abs,
                        "spike": spike,
                        "threshold": self._force_spike_threshold,
                        "released": False,
                        "reason": "",
                    }
                    released_now = spike >= self._force_spike_threshold
                    reason = "force_spike"
                    self.get_logger().info(
                        "Force spike baseline: "
                        f"fz_sum_abs={fz_sum_abs:.4f}, spike={spike:.4f}, "
                        f"threshold={self._force_spike_threshold:.4f}"
                    )
                    next_eval_time = now + self._force_sample_period_sec

                else:
                    if (
                        self._latest_vision_image is None
                        or self._latest_vision_stamp_ns
                        == self._last_openai_image_stamp_ns
                    ):
                        rclpy.spin_once(self, timeout_sec=0.02)
                        continue
                    self._publish_twist_halt()
                    result = self._classify_touching_ground_with_openai()
                    last_motion_time = time.monotonic()
                    if result is None:
                        next_eval_time = time.monotonic() + self._vision_eval_period_sec
                        rclpy.spin_once(self, timeout_sec=0.02)
                        continue
                    image_stamp_ns = self._latest_vision_stamp_ns
                    (
                        pred_touch,
                        raw_text,
                        openai_latency_sec,
                        openai_request_start_ns,
                        openai_request_end_ns,
                    ) = result
                    row = {
                        "stamp_ns": stamp_ns,
                        "vision_image_topic": self._vision_image_topic,
                        "image_stamp_ns": image_stamp_ns,
                        "openai_request_start_ns": openai_request_start_ns,
                        "openai_request_end_ns": openai_request_end_ns,
                        "openai_latency_sec": openai_latency_sec,
                        "elapsed_sec": float(elapsed),
                        "descent_m": float(descent),
                        "pred_touch": bool(pred_touch),
                        "raw_response": raw_text,
                        "released": False,
                        "reason": "",
                    }
                    released_now = bool(pred_touch)
                    reason = "openai_visual_touch"
                    self.get_logger().info(
                        "OpenAI Vision touch prediction: "
                        f"{pred_touch}, descent={descent:.3f} m, "
                        f"latency={openai_latency_sec:.3f} s"
                    )
                    self._last_openai_image_stamp_ns = image_stamp_ns
                    self._latest_vision_image = None
                    self._latest_vision_stamp_ns = None
                    next_eval_time = time.monotonic() + self._vision_eval_period_sec

                self._score_trace.append(row)
                if released_now:
                    row["released"] = True
                    row["reason"] = reason
                    release_reason = reason
                    self._publish_twist_halt()
                    self.get_logger().info(
                        f"{reason}; stopping descent and opening gripper."
                    )
                    self._send_gripper_goal(self._gripper_open_position)
                    return release_reason

            rclpy.spin_once(self, timeout_sec=0.02)

        self._publish_twist_halt()
        if released or self._open_on_safety_limit:
            if not released:
                self._score_trace.append(
                    {
                        "stamp_ns": int(self.get_clock().now().nanoseconds),
                        "elapsed_sec": float(motion_elapsed_sec),
                        "descent_m": float(max_descent),
                        "hold_score": "",
                        "place_score": "",
                        "released": True,
                        "reason": "safety_limit_opened",
                    }
                )
                release_reason = "safety_limit_opened"
            self.get_logger().info("Sequence: open gripper.")
            self._send_gripper_goal(self._gripper_open_position)
        return release_reason

    def _classify_hold_place(self) -> tuple[float, float] | None:
        if not self._patchtst_enabled:
            return None
        if len(self._force_buffer) < self._patchtst_min_samples:
            return None

        import torch

        seq = np.asarray(list(self._force_buffer), dtype=np.float32)
        if seq.shape[0] < self._patchtst_seq_len:
            pad = np.repeat(seq[-1:, :], self._patchtst_seq_len - seq.shape[0], axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[-self._patchtst_seq_len :, :]

        x = torch.from_numpy(seq[None, ...])
        device = next(self._patchtst_model.parameters()).device
        with torch.no_grad():
            xb_patch, _ = self._patchtst_create_patch(
                x.to(device), patch_len=10, stride=10
            )
            z = self._patchtst_model.backbone(xb_patch)
            z_pool = z.mean(dim=(1, 3))
            if self._patchtst_proj is not None:
                z_pool = self._patchtst_proj(z_pool)
            z_pool = z_pool / (z_pool.norm(dim=-1, keepdim=True) + 1e-8)
        emb = z_pool[0].detach().cpu()
        hold_score = float(torch.dot(emb, self._hold_vec))
        place_score = float(torch.dot(emb, self._place_vec))
        return hold_score, place_score

    def _write_score_trace(self) -> None:
        if self._session_dir is None:
            return
        csv_dir = self._session_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        trace_names = {
            "patchtst": "patchtst_score_trace.csv",
            "force_spike": "force_spike_trace.csv",
            "openai_vision": "openai_vision_trace.csv",
        }
        score_path = csv_dir / trace_names.get(self._release_mode, "release_trace.csv")
        preferred = [
            "stamp_ns",
            "image_stamp_ns",
            "vision_image_topic",
            "openai_request_start_ns",
            "openai_request_end_ns",
            "openai_latency_sec",
            "elapsed_sec",
            "descent_m",
            "hold_score",
            "place_score",
            "fz_sum_abs",
            "spike",
            "threshold",
            "pred_touch",
            "raw_response",
            "released",
            "reason",
        ]
        keys = {key for row in self._score_trace for key in row.keys()}
        fieldnames = [name for name in preferred if name in keys]
        fieldnames.extend(sorted(keys.difference(fieldnames)))
        with score_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._score_trace)
        self.get_logger().info(f"Release trace written: {score_path}")

    def _wait_for_force_samples(self) -> bool:
        deadline = time.monotonic() + self._force_wait_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self._latest_left_ft is not None and self._latest_right_ft is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(
            "Timed out waiting for force-torque topics: "
            f"{self._left_ft_topic}, {self._right_ft_topic}"
        )
        return False

    def _wait_for_vision_image(self) -> bool:
        deadline = time.monotonic() + self._force_wait_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self._latest_vision_image is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(
            f"Timed out waiting for vision image topic: {self._vision_image_topic}"
        )
        return False

    def _publish_twist_halt(self) -> None:
        halt = TwistStamped()
        halt.header.stamp = self.get_clock().now().to_msg()
        halt.header.frame_id = self._command_frame
        self._twist_pub.publish(halt)

    def _send_gripper_goal(
        self, target_position: float, wait_for_result: bool = False
    ) -> None:
        goal_log = Float32MultiArray()
        goal_log.data = [
            float(target_position),
            float(self._gripper_target_speed),
            float(self._gripper_target_force),
        ]
        self._gripper_goal_pub.publish(goal_log)

        if not self._gripper_action_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                f"Gripper action server unavailable: {self._gripper_action_name}"
            )
        goal = MoveTwoFingerGripper.Goal()
        goal.target_position = target_position
        goal.target_speed = self._gripper_target_speed
        goal.target_force = self._gripper_target_force
        future = self._gripper_action_client.send_goal_async(goal)
        if not self._spin_until_future_done(future, timeout_sec=2.0):
            raise RuntimeError("Timed out sending gripper goal")
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Gripper goal was rejected")
        if wait_for_result:
            result_future = goal_handle.get_result_async()
            if not self._spin_until_future_done(
                result_future, timeout_sec=self._gripper_result_timeout_sec
            ):
                raise RuntimeError("Timed out waiting for gripper result")

    def _wait_for_joint_state(self, timeout_sec: float) -> bool:
        start_time = time.monotonic()
        while rclpy.ok() and time.monotonic() - start_time < timeout_sec:
            if self._joint_state_ready:
                missing = [
                    name
                    for name in self._home_pose_rad
                    if name not in self._joint_positions
                ]
                if not missing:
                    return True
                self.get_logger().warning(
                    f"Joint state missing home joints: {missing}"
                )
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(f"Timed out waiting for {self._joint_state_topic}.")
        return False

    def _sleep_spin(self, duration_sec: float) -> None:
        end_time = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _spin_until_future_done(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.done()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = AutonomousMotionCaptureNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
from .workspace_paths import resolve_workspace_path
