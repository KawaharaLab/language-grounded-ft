#!/usr/bin/env python3
import argparse
import ast
import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import rclpy

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState


@dataclass
class JointSample:
	stamp_ns: int
	positions: List[float]


DEFAULT_JOINT_NAMES = [
	"shoulder_lift_joint",
	"elbow_joint",
	"wrist_1_joint",
	"wrist_2_joint",
	"wrist_3_joint",
	"shoulder_pan_joint",
]


def _parse_joint_positions(value: str) -> List[float]:
	parsed = ast.literal_eval(value)
	if not isinstance(parsed, (list, tuple)):
		raise ValueError("joint_positions must be a list")
	return [float(x) for x in parsed]


def read_joint_samples(csv_path: str) -> Iterable[JointSample]:
	with open(csv_path, "r", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			stamp_ns = int(row["stamp_ns"])
			positions = _parse_joint_positions(row["joint_positions"])
			yield JointSample(stamp_ns=stamp_ns, positions=positions)


def wait_for_service(client, timeout_sec: float) -> bool:
	return client.wait_for_service(timeout_sec=timeout_sec)


def compute_fk(
	node,
	client,
	joint_names: List[str],
	fk_link: str,
	frame_id: Optional[str],
	positions: List[float],
) -> PoseStamped:
	if len(positions) != len(joint_names):
		raise ValueError(f"Expected {len(joint_names)} joints, got {len(positions)}")

	joint_state = JointState()
	joint_state.name = list(joint_names)
	joint_state.position = list(positions)

	robot_state = RobotState()
	robot_state.joint_state = joint_state
	robot_state.is_diff = True

	request = GetPositionFK.Request()
	request.fk_link_names = [fk_link]
	request.robot_state = robot_state
	if frame_id:
		request.header.frame_id = frame_id

	future = client.call_async(request)
	rclpy.spin_until_future_complete(node, future)
	result = future.result()

	if result is None:
		raise RuntimeError("No response from /compute_fk")

	if not result.pose_stamped:
		raise RuntimeError("/compute_fk returned empty pose list")

	return result.pose_stamped[0]


def dh_transform(a: float, alpha: float, d: float, theta: float) -> List[List[float]]:
	ct = math.cos(theta)
	st = math.sin(theta)
	ca = math.cos(alpha)
	sa = math.sin(alpha)
	return [
		[ct, -st * ca, st * sa, a * ct],
		[st, ct * ca, -ct * sa, a * st],
		[0.0, sa, ca, d],
		[0.0, 0.0, 0.0, 1.0],
	]


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
	result = [[0.0] * 4 for _ in range(4)]
	for i in range(4):
		for j in range(4):
			result[i][j] = (
				a[i][0] * b[0][j]
				+ a[i][1] * b[1][j]
				+ a[i][2] * b[2][j]
				+ a[i][3] * b[3][j]
			)
	return result


def rotation_to_quaternion(r: List[List[float]]) -> tuple[float, float, float, float]:
	trace = r[0][0] + r[1][1] + r[2][2]
	if trace > 0.0:
		s = math.sqrt(trace + 1.0) * 2.0
		qw = 0.25 * s
		qx = (r[2][1] - r[1][2]) / s
		qy = (r[0][2] - r[2][0]) / s
		qz = (r[1][0] - r[0][1]) / s
	elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
		s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
		qw = (r[2][1] - r[1][2]) / s
		qx = 0.25 * s
		qy = (r[0][1] + r[1][0]) / s
		qz = (r[0][2] + r[2][0]) / s
	elif r[1][1] > r[2][2]:
		s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
		qw = (r[0][2] - r[2][0]) / s
		qx = (r[0][1] + r[1][0]) / s
		qy = 0.25 * s
		qz = (r[1][2] + r[2][1]) / s
	else:
		s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
		qw = (r[1][0] - r[0][1]) / s
		qx = (r[0][2] + r[2][0]) / s
		qy = (r[1][2] + r[2][1]) / s
		qz = 0.25 * s
	return qx, qy, qz, qw


def reorder_to_ur_chain(joint_names: List[str], positions: List[float]) -> List[float]:
	name_to_index = {name: i for i, name in enumerate(joint_names)}
	required = [
		"shoulder_pan_joint",
		"shoulder_lift_joint",
		"elbow_joint",
		"wrist_1_joint",
		"wrist_2_joint",
		"wrist_3_joint",
	]
	missing = [name for name in required if name not in name_to_index]
	if missing:
		raise ValueError(f"Missing joint names: {', '.join(missing)}")
	return [positions[name_to_index[name]] for name in required]


def compute_fk_ur3_dh(joint_names: List[str], positions: List[float]) -> PoseStamped:
	# UR3 standard DH parameters (meters)
	# Order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
	q1, q2, q3, q4, q5, q6 = reorder_to_ur_chain(joint_names, positions)
	d1 = 0.1519
	a2 = -0.24365
	a3 = -0.21325
	d4 = 0.11235
	d5 = 0.08535
	d6 = 0.0819

	alpha = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
	a = [0.0, a2, a3, 0.0, 0.0, 0.0]
	d = [d1, 0.0, 0.0, d4, d5, d6]
	q = [q1, q2, q3, q4, q5, q6]

	transform = [
		[1.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 0.0, 0.0],
		[0.0, 0.0, 1.0, 0.0],
		[0.0, 0.0, 0.0, 1.0],
	]
	for i in range(6):
		transform = matmul(transform, dh_transform(a[i], alpha[i], d[i], q[i]))

	pose = PoseStamped()
	pose.pose.position.x = transform[0][3]
	pose.pose.position.y = transform[1][3]
	pose.pose.position.z = transform[2][3]
	qx, qy, qz, qw = rotation_to_quaternion(transform)
	pose.pose.orientation.x = qx
	pose.pose.orientation.y = qy
	pose.pose.orientation.z = qz
	pose.pose.orientation.w = qw
	return pose


def write_output(
	output_path: str,
	rows: Iterable[tuple[int, PoseStamped]],
) -> None:
	with open(output_path, "w", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			[
				"stamp_ns",
				"x",
				"y",
				"z",
				"qx",
				"qy",
				"qz",
				"qw",
			]
		)
		for stamp_ns, pose in rows:
			writer.writerow(
				[
					stamp_ns,
					pose.pose.position.x,
					pose.pose.position.y,
					pose.pose.position.z,
					pose.pose.orientation.x,
					pose.pose.orientation.y,
					pose.pose.orientation.z,
					pose.pose.orientation.w,
				]
			)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Convert joint angles to Cartesian poses using MoveIt2 /compute_fk",
	)
	parser.add_argument(
		"path",
		help="Input joint_states.csv path or a directory containing */csv/joint_states.csv",
	)
	parser.add_argument(
		"--mode",
		choices=["auto", "moveit", "ur3_dh"],
		default="auto",
		help="FK backend: auto tries MoveIt2, then UR3 DH; moveit forces /compute_fk",
	)
	parser.add_argument(
		"--service",
		default="/compute_fk",
		help="MoveIt2 FK service name",
	)
	parser.add_argument(
		"--fk-link",
		default="tool0",
		help="Link name for FK result",
	)
	parser.add_argument(
		"--frame-id",
		default="",
		help="Frame ID to request FK in (empty = default planning frame)",
	)
	parser.add_argument(
		"--joint-names",
		default=",".join(DEFAULT_JOINT_NAMES),
		help="Comma-separated joint names matching CSV order",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		default=10.0,
		help="Seconds to wait for /compute_fk service",
	)
	return parser


def derive_output_path(input_path: str) -> str:
	root, ext = os.path.splitext(input_path)
	if ext.lower() == ".csv":
		return f"{root}_cartesian.csv"
	return f"{input_path}_cartesian.csv"


def collect_input_files(path: str) -> List[str]:
	if os.path.isdir(path):
		paths: List[str] = []
		for root, _, files in os.walk(path):
			if os.path.basename(root) != "csv":
				continue
			if "joint_states.csv" in files:
				paths.append(os.path.join(root, "joint_states.csv"))
		return sorted(paths)
	return [path]


def process_file(
	input_path: str,
	output_path: str,
	joint_names: List[str],
	fk_func,
) -> None:
	rows = []
	for sample in read_joint_samples(input_path):
		pose = fk_func(sample.positions)
		rows.append((sample.stamp_ns, pose))
	write_output(output_path, rows)


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	joint_names = [name.strip() for name in args.joint_names.split(",") if name.strip()]
	if not joint_names:
		raise SystemExit("joint-names must not be empty")

	input_paths = collect_input_files(args.path)
	if not input_paths:
		raise SystemExit("No joint_states.csv found under the given path")

	used_moveit = False

	if args.mode in {"auto", "moveit"}:
		rclpy.init()
		node = rclpy.create_node("joint_angles_to_cartesian")
		client = node.create_client(GetPositionFK, args.service)
		try:
			if not wait_for_service(client, timeout_sec=args.timeout):
				if args.mode == "moveit":
					raise SystemExit(f"Service {args.service} not available")
			else:
				used_moveit = True
				fk_func = lambda positions: compute_fk(
					node,
					client,
					joint_names,
					args.fk_link,
					args.frame_id or None,
					positions,
				)
				for input_path in input_paths:
					output_path = derive_output_path(input_path)
					process_file(input_path, output_path, joint_names, fk_func)
		finally:
			node.destroy_node()
			rclpy.shutdown()

	if not used_moveit:
		for input_path in input_paths:
			output_path = derive_output_path(input_path)
			process_file(
				input_path,
				output_path,
				joint_names,
				lambda positions: compute_fk_ur3_dh(joint_names, positions),
			)


if __name__ == "__main__":
	main()
