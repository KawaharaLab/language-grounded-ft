"""Hold initial/limit poses for visually checking gripper heights."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped

from .autonomous_motion_capture_node import AutonomousMotionCaptureNode


class HeightLimitCheckNode(AutonomousMotionCaptureNode):
    """Close the gripper at home, then descend to the safety-limit height on Enter."""

    def __init__(self) -> None:
        super().__init__()
        self._initial_hold_status_period_sec = float(
            self.declare_parameter("initial_hold_status_period_sec", 1.0).value
        )
        self._limit_hold_sec = float(
            self.declare_parameter("limit_hold_sec", -1.0).value
        )
        self._height_check_trace: list[dict[str, object]] = []

    def run(self) -> None:
        self._force_save = True
        self._prepare_session()
        self._ensure_enter_events_file()
        self._cleanup_existing_viewers()
        self._start_internal_viewers()
        self._start_throttles_and_compressors()
        self._start_rosbag()

        try:
            self._sleep_spin(self._pre_motion_recording_delay_sec)
            self._start_servo()
            self._run_sequence()
        except KeyboardInterrupt:
            self.get_logger().info("Keyboard interrupt received, stopping height check.")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Height limit check failed: {exc}")
        finally:
            self._stop_event.set()
            self._record_stop_time_if_missing()
            self._write_height_check_trace()
            self._finalize()

    def _run_sequence(self) -> None:
        self.get_logger().info("Sequence: move to initial pose.")
        if not self._move_to_home():
            raise RuntimeError("Failed to reach initial pose")

        self._collect_force_offset()

        self.get_logger().info("Sequence: close gripper at initial height.")
        self._send_gripper_goal(self._gripper_close_position, wait_for_result=True)
        self._publish_twist_halt()
        self._record_height_event("initial_closed_hold", 0.0)

        self.get_logger().info(
            "Initial height hold. Press Enter to descend to the safety-limit height."
        )
        if not self._wait_for_enter_to_descend():
            return

        max_descent = max(0.0, abs(self._lower_distance_m)) + max(
            0.0, self._safety_extra_lower_m
        )
        speed = abs(self._lower_velocity_scale_mps * self._lower_twist_command_z)
        if speed <= 0.0:
            raise RuntimeError("lower velocity must be non-zero")

        self.get_logger().info(
            f"Descending to limit: {max_descent:.3f} m at {speed:.3f} m/s."
        )
        self._descend_for_distance(max_descent, speed)
        self._publish_twist_halt()
        self._record_height_event("limit_closed_hold", max_descent)
        self.get_logger().info(
            "Limit height reached. Gripper remains closed; robot is holding still."
        )

        if self._limit_hold_sec < 0.0:
            self.get_logger().info("Press Ctrl-C when finished checking the height.")
            while rclpy.ok() and not self._stop_event.is_set():
                self._publish_twist_halt()
                rclpy.spin_once(self, timeout_sec=0.1)
        else:
            self._hold_still(self._limit_hold_sec)

    def _wait_for_enter_to_descend(self) -> bool:
        if not sys.stdin.isatty():
            self.get_logger().warning(
                "stdin is not a TTY; cannot wait for Enter. Stop with Ctrl-C."
            )
            while rclpy.ok() and not self._stop_event.is_set():
                self._publish_twist_halt()
                rclpy.spin_once(self, timeout_sec=0.1)
            return False

        last_status = 0.0
        while rclpy.ok() and not self._stop_event.is_set():
            if self._stdin_has_line():
                line = sys.stdin.readline()
                if line == "":
                    return False
                if line.strip() == "":
                    self._record_enter_event()
                    self._record_height_event("enter_pressed", 0.0)
                    return True
            now = time.monotonic()
            if now - last_status >= self._initial_hold_status_period_sec:
                self.get_logger().info("Holding initial height; waiting for Enter.")
                last_status = now
            self._publish_twist_halt()
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def _stdin_has_line(self) -> bool:
        import select

        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        return bool(readable)

    def _descend_for_distance(self, distance_m: float, speed_mps: float) -> None:
        duration_sec = distance_m / speed_mps
        start = time.monotonic()
        while rclpy.ok() and time.monotonic() - start < duration_sec:
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self._command_frame
            twist.twist.linear.z = self._lower_twist_command_z
            self._twist_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.02)

    def _hold_still(self, duration_sec: float) -> None:
        end = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < end:
            self._publish_twist_halt()
            rclpy.spin_once(self, timeout_sec=0.05)

    def _record_height_event(self, event: str, planned_descent_m: float) -> None:
        self._height_check_trace.append(
            {
                "stamp_ns": int(self.get_clock().now().nanoseconds),
                "event": event,
                "planned_descent_m": float(planned_descent_m),
                "lower_distance_m": float(abs(self._lower_distance_m)),
                "safety_extra_lower_m": float(max(0.0, self._safety_extra_lower_m)),
                "gripper_position": float(self._gripper_close_position),
            }
        )

    def _write_height_check_trace(self) -> None:
        if self._session_dir is None or not self._height_check_trace:
            return
        path = Path(self._session_dir) / "csv" / "height_limit_check_trace.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "stamp_ns",
            "event",
            "planned_descent_m",
            "lower_distance_m",
            "safety_extra_lower_m",
            "gripper_position",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._height_check_trace)
        self.get_logger().info(f"Height check trace written: {path}")


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = HeightLimitCheckNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
