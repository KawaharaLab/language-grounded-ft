import random
import shutil
import subprocess
import sys
import time
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import genesis as gs
import matplotlib.pyplot as plt

import genesis.utils.geom as gu

import master_movement as mm
from make_step import (
    EarlyDropDetected,
    final_make_step,
    make_step,
    set_object_contact_targets,
    set_early_drop_monitor,
    set_intentional_release,
)


class EarlyDropSimulationError(RuntimeError):
    def __init__(self, message, paths, force_df, deform_df, segment_df, metadata):
        super().__init__(message)
        self.paths = paths
        self.force_df = force_df
        self.deform_df = deform_df
        self.segment_df = segment_df
        self.metadata = metadata


PROJECT_ROOT = Path(
    os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parent.parent.parent)
).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get("FT_GENESIS_DATA_ROOT", PROJECT_ROOT / "data")
).expanduser().resolve()

CAPTURE_IMAGES = os.environ.get("GENESIS_CAPTURE_IMAGES", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PHOTO_INTERVAL = int(os.environ.get("PHOTO_INTERVAL", "80"))
if PHOTO_INTERVAL < 1:
    raise ValueError("PHOTO_INTERVAL must be at least 1.")
MATERIAL_TYPE = "Rigid"
""""Train"""
# TARGET_CHOICES = ["drop_simple", "drop", "inclined", "gentle_simple", "firm_simple"]
# TARGET_CHOICES = ["gentle", "firm", "gentle_simple", "inclined"]
# TARGET_CHOICES = ["drop_simple", "drop", "inclined", "gentle_simple", "firm_simple"]

"""Eval"""
# TARGET_CHOICES = ["drop_simple", "drop", "inclined", "gentle_simple", "firm_simple", "gentle", "gentle_simple"]
#TARGET_CHOICES = ["drop_simple", "drop", "inclined", "step", "gentle_simple", "firm_simple"]

DEFAULT_TARGET_CHOICES = ["drop_simple", "drop", "inclined", "step", "gentle_simple", "firm_simple"]
TARGET_CHOICES = [
    choice.strip()
    for choice in os.environ.get("TARGET_CHOICES", ",".join(DEFAULT_TARGET_CHOICES)).split(",")
    if choice.strip()
]

GRASP_TO_LIFT_WAIT_STEPS = 100

MAX_PARALLEL_PROCESSES = int(os.environ.get("MAX_PARALLEL_PROCESSES", "8"))
if MAX_PARALLEL_PROCESSES < 1:
    raise ValueError("MAX_PARALLEL_PROCESSES must be at least 1.")
WORKER_SPAWN_DELAY_SECONDS = float(os.environ.get("WORKER_SPAWN_DELAY_SECONDS", "5"))
if WORKER_SPAWN_DELAY_SECONDS < 0:
    raise ValueError("WORKER_SPAWN_DELAY_SECONDS must be non-negative.")
REQUIRED_RUNS_PER_TASK = int(os.environ.get("REQUIRED_RUNS_PER_TASK", "0"))
if REQUIRED_RUNS_PER_TASK < 0:
    raise ValueError("REQUIRED_RUNS_PER_TASK must be non-negative.")
DATASET = os.environ.get("DATASET", "eval_21072026")
OUTPUT_DATASET = os.environ.get("OUTPUT_DATASET", DATASET)
DATASET_TYPE = "ycb" if "eval" in DATASET else "gso"  # Options: "ycb", "gso"
OUTPUT_VARIANT_LABEL = os.environ.get(
    "OUTPUT_VARIANT_LABEL", "vary"
).strip()  # e.g. "heavy", "floor_descent"; set "" to disable
PLACE_TARGET_X_RANGE = (0.20, 0.70)
PLACE_TARGET_Y_RANGE = (0.20, 0.70)
MIN_PLACE_DISTANCE_FROM_SPAWN = 0.14
MIN_PLACE_DISTANCE_FROM_ROBOT = 0.42
MAX_PLACE_DISTANCE_FROM_ROBOT = 0.82
TARGET_TILE_SIZE = 0.20
TARGET_TILE_THICKNESS = 0.004
TARGET_TILE_COLOR = (0.10, 0.35, 0.90, 1.0)
TARGET_RAMP_COLOR = (0.07, 0.22, 0.55, 1.0)
TARGET_RAMP_MESH = Path(__file__).resolve().parent / "assets" / "inclined_ramp.obj"
BUMP_WALL_SIZE = (0.06, 0.06, 0.035)
BUMP_WALL_COLOR = (0.90, 0.25, 0.20, 1.0)
BUMP_PUSH_STEPS = 100
TOPPLE_THRESHOLD_DEG = 35.0
INCLINED_TILE_ANGLES_DEG = (10, 20, 30)
STEP_HEIGHTS_M = {
    "step_0.5cm": 0.005,
    "step_1.0cm": 0.01,
    "step_2cm": 0.02,
}
STEP_PLATFORM_SIZE = 0.24
STEP_OVERHANG_M = 0.015
WOBBLE_DECK_HALF_SIZE = (0.120, 0.105, 0.008)
WOBBLE_PIVOT_Z_M = 0.074
WOBBLE_STIFFNESS_NM_RAD_RANGE = (5.0, 8.0)
WOBBLE_DAMPING_NMS_RAD_RANGE = (0.22, 0.35)
WOBBLE_LIMIT_DEG_RANGE = (9.0, 14.0)
WOBBLE_DECK_MASS_KG_RANGE = (0.45, 0.55)
WOBBLE_OFFSET_LOCAL_X_M_RANGE = (-0.015, 0.015)
WOBBLE_OFFSET_LOCAL_Y_ABS_M_RANGE = (0.060, 0.070)
WOBBLE_POST_CONTACT_DURATION_S_RANGE = (0.4, 1.0)
WOBBLE_DESCENT_SPEED_M_S = float(
    os.environ.get("WOBBLE_DESCENT_SPEED_M_S", "0.04")
)
if WOBBLE_DESCENT_SPEED_M_S <= 0.0:
    raise ValueError("WOBBLE_DESCENT_SPEED_M_S must be positive.")
WOBBLE_MAX_APPROACH_DESCENT_M = 0.25


def _configured_gpu_ids() -> list[int]:
    configured = os.environ.get("GENESIS_GPU_IDS")
    if configured is not None:
        configured = configured.strip().lower()
        if configured in {"", "cpu", "none"}:
            return []
        try:
            gpu_ids = [int(value.strip()) for value in configured.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("GENESIS_GPU_IDS must be comma-separated integers or 'cpu'.") from exc
        if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError("GENESIS_GPU_IDS must contain one or more non-negative GPU IDs.")
        return list(dict.fromkeys(gpu_ids))

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        visible_devices = visible_devices.strip()
        if not visible_devices or visible_devices == "-1":
            return []
        try:
            return [int(value.strip()) for value in visible_devices.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must use numeric physical GPU IDs with this launcher; "
                "set GENESIS_GPU_IDS explicitly otherwise."
            ) from exc

    device_paths = sorted(Path("/dev").glob("nvidia[0-9]*"))
    return [int(path.name.removeprefix("nvidia")) for path in device_paths]


## -------------------------- PATH SETUP -------------------------- ##
def setup_paths(object_name: str, target_choice: str) -> dict:
    input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / "model.obj"
    if DATASET_TYPE == "hugging_face":
        input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / f"{object_name}.glb"
    if not input_obj_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_obj_path}")

    output_dir = DATA_ROOT / OUTPUT_DATASET / "csv" / object_name / MATERIAL_TYPE / target_choice
    image_root = DATA_ROOT / OUTPUT_DATASET / "images" / object_name / MATERIAL_TYPE / target_choice
    image_dirs = [image_root / "camera_0", image_root / "camera_wrist"]

    output_dir.mkdir(parents=True, exist_ok=True)
    if CAPTURE_IMAGES:
        for image_dir in image_dirs:
            image_dir.mkdir(parents=True, exist_ok=True)

    return {
        "input_obj": input_obj_path,
        "output_dir": output_dir,
        "images_dir": image_root,
        "force_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_{target_choice}.csv",
        "deformation_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_deform_{target_choice}.csv",
        "segmentation_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_steps_{target_choice}.csv",
        "metadata_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_metadata_{target_choice}.json",
        "object_name": object_name,
        "plot": output_dir / f"{object_name}_force_graph",
    }


def get_obj_bounding_box(obj_path):
    """Return the axis-aligned bounding box extents for OBJ or GLB meshes."""
    min_corner, max_corner = get_obj_bounds(obj_path)
    return (max_corner - min_corner).tolist()


def get_obj_bounds(obj_path):
    """Return the axis-aligned bounding box (min, max) for OBJ or GLB meshes."""
    obj_path = Path(obj_path)
    suffix = obj_path.suffix.lower()
    if suffix == ".obj":
        return _bounds_from_obj(obj_path)
    if suffix == ".glb":
        return _bounds_from_glb(obj_path)
    raise ValueError(f"Unsupported mesh format: {suffix}")


def _bounds_from_obj(obj_path: Path):
    min_corner = np.array([np.inf, np.inf, np.inf], dtype=float)
    max_corner = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    vertex_found = False
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                x, y, z = map(float, parts[1:4])
                vertex = np.array([x, y, z], dtype=float)
                min_corner = np.minimum(min_corner, vertex)
                max_corner = np.maximum(max_corner, vertex)
                vertex_found = True
    if not vertex_found:
        raise ValueError(f"No vertex data found in OBJ file: {obj_path}")
    return min_corner, max_corner


def _bounds_from_glb(obj_path: Path):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to compute bounding boxes for .glb files.") from exc

    mesh = trimesh.load(obj_path, force="scene")
    bounds = mesh.bounds
    if bounds is None:
        raise ValueError(f"Could not compute bounds for {obj_path}")
    min_corner, max_corner = bounds
    return np.asarray(min_corner, dtype=float), np.asarray(max_corner, dtype=float)


def _scale_to_vec(scale):
    """Return a length-3 numpy array regardless of scalar/tuple scale input."""
    if np.isscalar(scale):
        return np.array([float(scale), float(scale), float(scale)], dtype=float)
    arr = np.asarray(scale, dtype=float)
    if arr.size == 1:
        return np.repeat(arr.item(), 3)
    if arr.size != 3:
        raise ValueError(f"Scale must be scalar or length 3, got shape {arr.shape}")
    return arr


def set_grasp(obj_path):
    gripper_min_width, gripper_max_width = 0.002, 0.075
    bbox = get_obj_bounding_box(obj_path)
    scale = 1.0
    if gripper_min_width < bbox[0] < gripper_max_width:
        euler = (0, 0, 90)
    elif gripper_min_width < bbox[1] < gripper_max_width:
        euler = (0, 0, 0)
    else:
        scale = (0.080 / (bbox[0] + 0.01)) if bbox[0] < bbox[1] else (0.080 / (bbox[1] + 0.01))
        euler = (0, 0, 90) if bbox[0] < bbox[1] else (0, 0, 0)
    return scale, euler


## -------------------------- SIMULATION CORE -------------------------- ##
def sample_place_target(source_xy, robot_xy=(0.0, 0.0)):
    source_xy = np.asarray(source_xy, dtype=float)
    robot_xy = np.asarray(robot_xy, dtype=float)

    def _valid_target(candidate_xy):
        d_spawn = np.linalg.norm(candidate_xy - source_xy)
        d_robot = np.linalg.norm(candidate_xy - robot_xy)
        return (
            d_spawn >= MIN_PLACE_DISTANCE_FROM_SPAWN
            and MIN_PLACE_DISTANCE_FROM_ROBOT <= d_robot <= MAX_PLACE_DISTANCE_FROM_ROBOT
        )

    for _ in range(300):
        tx = random.uniform(*PLACE_TARGET_X_RANGE)
        ty = random.uniform(*PLACE_TARGET_Y_RANGE)
        candidate = np.array([tx, ty], dtype=float)
        if _valid_target(candidate):
            return float(tx), float(ty)

    # Fallback: sample from an annulus around the robot and clamp to valid workspace.
    for _ in range(300):
        theta = random.uniform(-np.pi, np.pi)
        radius = random.uniform(MIN_PLACE_DISTANCE_FROM_ROBOT, MAX_PLACE_DISTANCE_FROM_ROBOT)
        candidate = robot_xy + radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        candidate[0] = np.clip(candidate[0], PLACE_TARGET_X_RANGE[0], PLACE_TARGET_X_RANGE[1])
        candidate[1] = np.clip(candidate[1], PLACE_TARGET_Y_RANGE[0], PLACE_TARGET_Y_RANGE[1])
        if _valid_target(candidate):
            return float(candidate[0]), float(candidate[1])

    # Final deterministic fallback near workspace center.
    fallback = np.array(
        [
            0.5 * (PLACE_TARGET_X_RANGE[0] + PLACE_TARGET_X_RANGE[1]),
            0.5 * (PLACE_TARGET_Y_RANGE[0] + PLACE_TARGET_Y_RANGE[1]),
        ],
        dtype=float,
    )
    return float(fallback[0]), float(fallback[1])


def _is_simple_mode(place_mode: str) -> bool:
    return place_mode in ("drop_simple", "gentle_simple", "firm_simple")


def _inclined_angle_deg(place_mode: str) -> float | None:
    prefix = "inclined_"
    if not place_mode.startswith(prefix):
        return None
    try:
        angle_deg = float(place_mode.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"Invalid inclined mode: {place_mode}") from exc
    if angle_deg not in INCLINED_TILE_ANGLES_DEG:
        raise ValueError(
            f"Unsupported inclined angle {angle_deg:g}; expected one of {INCLINED_TILE_ANGLES_DEG}"
        )
    return angle_deg


def _is_inclined_mode(place_mode: str) -> bool:
    return _inclined_angle_deg(place_mode) is not None


def _step_height_m(place_mode: str) -> float | None:
    return STEP_HEIGHTS_M.get(place_mode)


def _is_step_mode(place_mode: str) -> bool:
    return _step_height_m(place_mode) is not None


def _is_wobble_mode(place_mode: str) -> bool:
    return place_mode == "wobble"


def _place_descent_target_z(place_mode: str, surface_aligned_z: float) -> float:
    """For a step, descend past its top surface to the surrounding floor level."""
    step_height = _step_height_m(place_mode)
    if step_height is None:
        return float(surface_aligned_z)
    return float(surface_aligned_z) - float(step_height)


def _expand_target_choices(target_choices: list[str]) -> list[str]:
    expanded = []
    for target in target_choices:
        if target == "inclined":
            expanded.extend(f"inclined_{angle:g}" for angle in INCLINED_TILE_ANGLES_DEG)
        elif target == "step":
            expanded.extend(STEP_HEIGHTS_M)
        else:
            expanded.append(target)
    return expanded


def _plan_target_tile(place_mode: str) -> dict:
    angle_deg = _inclined_angle_deg(place_mode)
    step_height = _step_height_m(place_mode)
    if _is_wobble_mode(place_mode):
        hinge_yaw_deg = random.uniform(0.0, 360.0)
        hinge_yaw_rad = np.deg2rad(hinge_yaw_deg)
        local_offset = np.array(
            [
                random.uniform(*WOBBLE_OFFSET_LOCAL_X_M_RANGE),
                random.choice((-1.0, 1.0))
                * random.uniform(*WOBBLE_OFFSET_LOCAL_Y_ABS_M_RANGE),
            ],
            dtype=float,
        )
        rotation = np.array(
            [
                [np.cos(hinge_yaw_rad), -np.sin(hinge_yaw_rad)],
                [np.sin(hinge_yaw_rad), np.cos(hinge_yaw_rad)],
            ]
        )
        placement_offset_xy = rotation @ local_offset
        deck_half_x, deck_half_y, deck_half_z = WOBBLE_DECK_HALF_SIZE
        return {
            "surface_type": "wobble",
            "inclination_deg": 0.0,
            "euler_deg": (0.0, 0.0, float(hinge_yaw_deg)),
            # target_xy is the object center; the deck center is shifted in the
            # opposite direction so that the object lands off-center.
            "center_offset_xy": (-placement_offset_xy).tolist(),
            "placement_offset_local_xy": local_offset.tolist(),
            "placement_offset_world_xy": placement_offset_xy.tolist(),
            "size_xyz": (
                2.0 * deck_half_x,
                2.0 * deck_half_y,
                2.0 * deck_half_z,
            ),
            "center_z": float(WOBBLE_PIVOT_Z_M),
            "surface_center_z": float(WOBBLE_PIVOT_Z_M + deck_half_z),
            "hinge_yaw_deg": float(hinge_yaw_deg),
            "stiffness_nm_rad": random.uniform(*WOBBLE_STIFFNESS_NM_RAD_RANGE),
            "damping_nms_rad": random.uniform(*WOBBLE_DAMPING_NMS_RAD_RANGE),
            "limit_deg": random.uniform(*WOBBLE_LIMIT_DEG_RANGE),
            "deck_mass_kg": random.uniform(*WOBBLE_DECK_MASS_KG_RANGE),
        }
    if step_height is not None:
        yaw_deg = random.uniform(0.0, 360.0)
        yaw_rad = np.deg2rad(yaw_deg)
        center_offset_xy = (0.5 * STEP_PLATFORM_SIZE + STEP_OVERHANG_M) * np.array(
            [-np.sin(yaw_rad), np.cos(yaw_rad)]
        )
        return {
            "inclination_deg": 0.0,
            "step_height_m": float(step_height),
            "overhang_m": STEP_OVERHANG_M,
            "euler_deg": (0.0, 0.0, float(yaw_deg)),
            "center_offset_xy": center_offset_xy.tolist(),
            "size_xyz": (STEP_PLATFORM_SIZE, STEP_PLATFORM_SIZE, float(step_height)),
            "center_z": 0.5 * float(step_height),
            "surface_center_z": float(step_height),
        }
    if angle_deg is None:
        return {
            "inclination_deg": 0.0,
            "euler_deg": (0.0, 0.0, 0.0),
            "center_offset_xy": (0.0, 0.0),
            "size_xyz": (TARGET_TILE_SIZE, TARGET_TILE_SIZE, TARGET_TILE_THICKNESS),
            "center_z": TARGET_TILE_THICKNESS * 0.5,
            "surface_center_z": TARGET_TILE_THICKNESS,
        }

    signed_angle_deg = angle_deg * random.choice((-1.0, 1.0))
    yaw_deg = random.uniform(0.0, 360.0)
    angle_rad = np.deg2rad(angle_deg)
    center_z = 0.5 * (
        TARGET_TILE_SIZE * np.sin(angle_rad) + TARGET_TILE_THICKNESS * np.cos(angle_rad)
    )
    surface_center_z = center_z + 0.5 * TARGET_TILE_THICKNESS * np.cos(angle_rad)
    ramp_run = TARGET_TILE_SIZE * np.cos(angle_rad)
    ramp_rise = TARGET_TILE_SIZE * np.sin(angle_rad)
    ramp_yaw_deg = yaw_deg if signed_angle_deg > 0.0 else (yaw_deg + 180.0) % 360.0
    return {
        "inclination_deg": float(angle_deg),
        "euler_deg": (float(signed_angle_deg), 0.0, float(yaw_deg)),
        "center_offset_xy": (0.0, 0.0),
        "size_xyz": (TARGET_TILE_SIZE, TARGET_TILE_SIZE, TARGET_TILE_THICKNESS),
        "center_z": float(center_z),
        "surface_center_z": float(surface_center_z),
        "ramp_run": float(ramp_run),
        "ramp_rise": float(ramp_rise),
        "ramp_yaw_deg": float(ramp_yaw_deg),
    }


def _wobble_support_mjcf(plan: dict, center_xy: tuple[float, float]) -> str:
    """Build a marker-free single-axis compliant support from a sampled plan."""
    center_x, center_y = center_xy
    half_x, half_y, half_z = (0.5 * float(value) for value in plan["size_xyz"])
    limit_rad = np.deg2rad(float(plan["limit_deg"]))
    yaw_rad = np.deg2rad(float(plan["hinge_yaw_deg"]))
    return f"""
<mujoco model="wobble_support">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="pedestal" type="box" size="0.035 0.035 0.030"
          pos="{center_x} {center_y} 0.030" rgba="0.22 0.24 0.28 1"
          friction="1 0.005 0.0001" contype="0" conaffinity="0"/>
    <body name="wobble_deck" pos="{center_x} {center_y} {float(plan['center_z'])}"
          euler="0 0 {yaw_rad}">
      <joint name="wobble_hinge" type="hinge" axis="1 0 0" limited="true"
             range="{-limit_rad} {limit_rad}"
             stiffness="{float(plan['stiffness_nm_rad'])}"
             damping="{float(plan['damping_nms_rad'])}" armature="0.012"/>
      <geom name="deck" type="box" size="{half_x} {half_y} {half_z}"
            mass="{float(plan['deck_mass_kg'])}" rgba="0.10 0.35 0.90 1"
            friction="1.2 0.01 0.0001"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def _plan_bump_wall(source_xy, target_xy):
    source = np.asarray(source_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    delta = target - source
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        direction = np.array([1.0, 0.0], dtype=float)
        distance = 1e-6
    else:
        direction = delta / distance

    min_along = 0.08
    max_along = max(min_along, distance - 0.06)
    along = float(np.clip(distance * 0.45, min_along, max_along))
    wall_xy = source + direction * along
    return {
        "center_xy": (float(wall_xy[0]), float(wall_xy[1])),
        "size_xyz": tuple(float(v) for v in BUMP_WALL_SIZE),
        "distance_from_source": along,
    }


def create_scene(obj_path: str, place_mode: str = "gentle", gpu_id: int | None = None):
    if gpu_id == -1:
        gs.init(backend=gs.cpu)
    elif torch.cuda.is_available():
        if gpu_id is not None:
            device_count = torch.cuda.device_count()
            if gpu_id >= device_count:
                raise ValueError(f"GPU {gpu_id} is unavailable; PyTorch reports {device_count} visible GPU(s).")
            torch.cuda.set_device(gpu_id)
        gs.init(backend=gs.gpu)
    else:
        if gpu_id is not None:
            raise RuntimeError(f"GPU {gpu_id} was requested, but CUDA is unavailable.")
        gs.init(backend=gs.cpu)

    object_scale, object_euler = set_grasp(obj_path)
    obj_min_corner, obj_max_corner = get_obj_bounds(obj_path)
    scale_vec = _scale_to_vec(object_scale)
    drop_center = np.array([0.45, 0.45], dtype=float)
    if Path(obj_path).suffix.lower() == ".glb":
        local_center = 0.5 * (obj_min_corner[:2] + obj_max_corner[:2])
        drop_center -= scale_vec[:2] * local_center

    spawn_z = 0.001 + max(0.0, -scale_vec[2] * obj_min_corner[2])
    object_spawn_pos = (float(drop_center[0]), float(drop_center[1]), float(spawn_z))
    color = (0, 255, 0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-2),
        viewer_options=gs.options.ViewerOptions(camera_pos=(3, -1, 1.5), camera_lookat=(0, 0, 0), camera_fov=30),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=3.0, friction=1.0),
    )

    cam = scene.add_camera(res=(1080, 1080), pos=(0.9, 0.9, 0.35), lookat=(0.0, 0.0, 0.35), fov=30)
    cam_wrist = scene.add_camera(
        model="thinlens",
        res=(1024, 1024),
        pos=(0.0, 0.0, 0.0),  # overwritten by attach below
        lookat=(0.0, 0.0, 0.0),
        fov=80.0,
        GUI=False,
    )
    # Mount wrist camera near the gripper tip with a small forward offset and downward pitch.
    roll = np.deg2rad(-10)   # keep camera upright relative to tool flange
    pitch = np.deg2rad(180)  # 初期位置のグリッパから根本への軸周り
    yaw = np.deg2rad(-90)    # 鉛直上向きの軸周り
    R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = R_yaw @ R_pitch @ R_roll
    trans = np.array([0.14, 0.0, -0.08])  # z: 鉛直下向き
    wrist_mount_link = franka.get_link("hand")
    cam_wrist.attach(wrist_mount_link, gu.trans_R_to_T(trans, R))

    rho = 200.0 * random.uniform(1.0, 10.0)
    gso_object = scene.add_entity(
        material=gs.materials.Rigid(rho=rho),
        morph=gs.morphs.Mesh(file=obj_path, scale=object_scale, pos=object_spawn_pos, euler=object_euler),
        surface=gs.surfaces.Default(color=color),
    )

    # Simple modes: object starts already on the tile and is placed/dropped back onto it.
    if _is_simple_mode(place_mode):
        target_xy = (float(drop_center[0]), float(drop_center[1]))
    else:
        target_xy = sample_place_target(drop_center, robot_xy=(0.0, 0.0))
    target_tile_plan = _plan_target_tile(place_mode)
    tile_offset_x, tile_offset_y = target_tile_plan["center_offset_xy"]
    target_ramp = None
    if _is_inclined_mode(place_mode):
        target_ramp = scene.add_entity(
            material=gs.materials.Rigid(rho=1200, friction=1.0, coup_friction=1.0),
            morph=gs.morphs.Mesh(
                file=str(TARGET_RAMP_MESH),
                scale=(
                    TARGET_TILE_SIZE,
                    target_tile_plan["ramp_run"],
                    target_tile_plan["ramp_rise"],
                ),
                pos=(target_xy[0], target_xy[1], 0.0),
                euler=(0.0, 0.0, target_tile_plan["ramp_yaw_deg"]),
                fixed=True,
                convexify=True,
            ),
            surface=gs.surfaces.Default(color=TARGET_RAMP_COLOR),
        )
    if _is_wobble_mode(place_mode):
        deck_center_xy = (
            target_xy[0] + tile_offset_x,
            target_xy[1] + tile_offset_y,
        )
        target_tile = scene.add_entity(
            material=gs.materials.Rigid(rho=800, friction=1.2, coup_friction=1.0),
            morph=gs.morphs.MJCF(
                file=_wobble_support_mjcf(target_tile_plan, deck_center_xy)
            ),
        )
    else:
        target_tile = scene.add_entity(
            material=gs.materials.Rigid(rho=800, friction=1.0, coup_friction=1.0),
            morph=gs.morphs.Box(
                size=target_tile_plan["size_xyz"],
                pos=(
                    target_xy[0] + tile_offset_x,
                    target_xy[1] + tile_offset_y,
                    target_tile_plan["center_z"],
                ),
                euler=target_tile_plan["euler_deg"],
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=TARGET_TILE_COLOR),
        )
    wall_plan = None
    obstacle_entity = None
    if place_mode == "bump":
        wall_plan = _plan_bump_wall(drop_center, target_xy)
        wall_x, wall_y = wall_plan["center_xy"]
        wall_sx, wall_sy, wall_sz = wall_plan["size_xyz"]
        obstacle_entity = scene.add_entity(
            material=gs.materials.Rigid(rho=1200, friction=1.2, coup_friction=1.2),
            morph=gs.morphs.Box(
                size=(wall_sx, wall_sy, wall_sz),
                pos=(wall_x, wall_y, wall_sz * 0.5),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=BUMP_WALL_COLOR),
        )

    scene.build()
    return (
        scene,
        cam,
        cam_wrist,
        franka,
        gso_object,
        target_xy,
        target_tile,
        obstacle_entity,
        wall_plan,
        target_tile_plan,
        target_ramp,
    )


def _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, grip_force, steps=100):
    name = paths["object_name"]
    for _ in range(steps):
        make_step(
            scene=scene,
            cam=cam,
            franka=franka,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            gso_object=gso_object,
            name=name,
            gripper_force=grip_force,
            cam_wrist=cam_wrist,
        )


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def _flatten_quat_wxyz(quat):
    quat = _to_numpy(quat).reshape(-1)
    if quat.size >= 4:
        return quat[:4].astype(float)
    raise ValueError(f"Unexpected quaternion shape: {quat.shape}")


def _angle_diff_deg(a, b):
    delta = a - b
    return (delta + 180.0) % 360.0 - 180.0


def _compute_max_xy_tilt_change_deg(quat_ref_wxyz, quat_now_wxyz):
    e_ref = _to_numpy(gs.utils.geom.quat_to_xyz(np.asarray(quat_ref_wxyz), rpy=True, degrees=True)).reshape(-1)
    e_now = _to_numpy(gs.utils.geom.quat_to_xyz(np.asarray(quat_now_wxyz), rpy=True, degrees=True)).reshape(-1)
    dx = abs(_angle_diff_deg(float(e_now[0]), float(e_ref[0])))
    dy = abs(_angle_diff_deg(float(e_now[1]), float(e_ref[1])))
    return max(dx, dy)


def _store_placement_outcome(metadata, quat_at_release, quat_after_settle):
    post_release_change = _compute_max_xy_tilt_change_deg(quat_at_release, quat_after_settle)
    upright_reference = metadata.get("upright_reference_quat_wxyz", quat_at_release)
    final_tilt = _compute_max_xy_tilt_change_deg(upright_reference, quat_after_settle)
    outcome_tilt = max(post_release_change, final_tilt)
    metadata["post_release_tilt_change_deg"] = round(float(post_release_change), 4)
    metadata["tilt_xy_deg"] = round(float(final_tilt), 4)
    metadata["outcome_tilt_xy_deg"] = round(float(outcome_tilt), 4)
    metadata["toppled"] = bool(outcome_tilt >= TOPPLE_THRESHOLD_DEG)


def _retreat_up_to_height(
    scene,
    cam,
    cam_wrist,
    franka,
    gso_object,
    df,
    deform_csv,
    paths,
    end_effector,
    motors_dof,
    fingers_dof,
    grip_force,
    target_z,
    fixed_x,
    fixed_y,
    steps=120,
):
    name = paths["object_name"]
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    z_start = eef_pos[2]
    for step_idx in range(max(1, steps)):
        alpha = (step_idx + 1) / max(1, steps)
        z_target = (1.0 - alpha) * z_start + alpha * target_z
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([fixed_x, fixed_y, z_target]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        make_step(
            scene=scene,
            cam=cam,
            franka=franka,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            gso_object=gso_object,
            name=name,
            gripper_force=grip_force,
            cam_wrist=cam_wrist,
        )


def _wobble_angle_deg(target_tile) -> float | None:
    if target_tile is None:
        return None
    try:
        dof_position = _to_numpy(target_tile.get_dofs_position()).reshape(-1)
    except (AttributeError, RuntimeError, ValueError):
        return None
    if dof_position.size == 0:
        return None
    return float(np.rad2deg(dof_position[0]))


def _run_place_phase(
    scene,
    cam,
    cam_wrist,
    franka,
    gso_object,
    df,
    deform_csv,
    seg_df,
    paths,
    end_effector,
    place_x,
    place_y,
    place_mode,
    place_eef_z_nominal,
    retreat_target_z,
    motors_dof,
    fingers_dof,
    current_force,
    metadata,
    target_tile=None,
):
    name = paths["object_name"]

    if _is_wobble_mode(place_mode):
        seg_df.loc[len(seg_df)] = ["gentle_place", int(scene.t)]
        requested_duration_s = random.uniform(*WOBBLE_POST_CONTACT_DURATION_S_RANGE)
        requested_post_contact_steps = max(
            1, int(round(requested_duration_s / 0.01))
        )
        descent_step_m = WOBBLE_DESCENT_SPEED_M_S * 0.01
        descent_start_z = float(franka.get_links_pos([8]).tolist()[0][2])
        first_contact_step = None
        post_contact_steps = 0
        approach_steps = 0
        max_abs_angle_deg = 0.0
        final_descent_z = descent_start_z

        # Monitor the entire approach so the timer starts at the first physical
        # tile contact. The command speed deliberately remains identical before
        # and after contact; contact must not leak through an imposed speed cue.
        max_approach_steps = int(
            np.ceil(WOBBLE_MAX_APPROACH_DESCENT_M / descent_step_m)
        )
        max_total_steps = max_approach_steps + requested_post_contact_steps + 1
        for _ in range(max_total_steps):
            if first_contact_step is None and approach_steps >= max_approach_steps:
                break
            contact_was_already_detected = first_contact_step is not None
            final_descent_z -= descent_step_m
            if first_contact_step is None:
                approach_steps += 1
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=np.array([place_x, place_y, final_descent_z]),
                quat=np.array([0, 1, 0, 0]),
            )
            franka.control_dofs_position(qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
            angle_deg = _wobble_angle_deg(target_tile)
            if angle_deg is not None:
                max_abs_angle_deg = max(max_abs_angle_deg, abs(angle_deg))
            target_contact = bool(df.iloc[-1].get("obj_target_tile", False))
            if target_contact and first_contact_step is None:
                first_contact_step = int(df.iloc[-1]["step"])
                seg_df.loc[len(seg_df)] = ["press", first_contact_step]
            if contact_was_already_detected:
                post_contact_steps += 1
                if post_contact_steps >= requested_post_contact_steps:
                    break

        if first_contact_step is None:
            raise RuntimeError(
                f"Wobble deck contact was not detected within {max_approach_steps} descent steps."
            )

        metadata["wobble_contact_step"] = int(first_contact_step)
        metadata["wobble_post_contact_descent_steps"] = int(post_contact_steps)
        metadata["wobble_post_contact_descent_s"] = float(post_contact_steps * 0.01)
        metadata["wobble_approach_speed_m_s"] = float(WOBBLE_DESCENT_SPEED_M_S)
        metadata["wobble_descent_speed_m_s"] = float(WOBBLE_DESCENT_SPEED_M_S)
        metadata["wobble_post_contact_descent_m"] = float(
            post_contact_steps * descent_step_m
        )
        metadata["wobble_contact_triggered_speed_change"] = False
        metadata["wobble_motion_policy"] = "constant_descent_through_contact"
        metadata["wobble_final_eef_z"] = float(final_descent_z)
        metadata["max_abs_wobble_angle_deg_during_descent"] = float(max_abs_angle_deg)

        seg_df.loc[len(seg_df)] = ["release", int(scene.t)]
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=60,
            cam_wrist=cam_wrist,
        )
        angle_after_release = _wobble_angle_deg(target_tile)
        if angle_after_release is not None:
            metadata["wobble_angle_deg_after_release"] = float(angle_after_release)
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=120,
        )
        _settle_after_release(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            -current_force,
            steps=150,
        )
        angle_after_settle = _wobble_angle_deg(target_tile)
        if angle_after_settle is not None:
            metadata["wobble_angle_deg_after_settle"] = float(angle_after_settle)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        _store_placement_outcome(metadata, quat_at_release, quat_after_retreat)
        return

    if place_mode in ("drop", "drop_simple"):
        seg_df.loc[len(seg_df)] = ["drop", int(scene.t)]
        release_z = place_eef_z_nominal + random.uniform(0.05, 0.08)
        mm.descend_to_place_cautiously(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            z=release_z,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=120,
            cam_wrist=cam_wrist,
        )
        seg_df.loc[len(seg_df)] = ["release", int(scene.t)]
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=35,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=100,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=180)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        _store_placement_outcome(metadata, quat_at_release, quat_after_retreat)
        return

    if (
        place_mode in ("gentle", "gentle_simple")
        or _is_inclined_mode(place_mode)
        or _is_step_mode(place_mode)
    ):
        seg_df.loc[len(seg_df)] = ["gentle_place", int(scene.t)]
        mm.descend_to_place_cautiously(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            z=place_eef_z_nominal,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=320,
            cam_wrist=cam_wrist,
        )
        for _ in range(20):
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
        seg_df.loc[len(seg_df)] = ["release", int(scene.t)]
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=60,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=120,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=120)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        _store_placement_outcome(metadata, quat_at_release, quat_after_retreat)
        return

    if place_mode in ("press", "firm", "firm_simple"):
        seg_df.loc[len(seg_df)] = ["press_place", int(scene.t)]
        mm.descend_to_place_cautiously(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            z=place_eef_z_nominal,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=260,
            cam_wrist=cam_wrist,
        )

        press_depth = 0.03
        press_steps = 120
        seg_df.loc[len(seg_df)] = ["press", int(scene.t)]
        for step in range(press_steps):
            z_target = place_eef_z_nominal - press_depth * (step + 1) / press_steps
            qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([place_x, place_y, z_target]), quat=np.array([0, 1, 0, 0]))
            franka.control_dofs_position(qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
            if len(df) > 0:
                left_fz = float(df.iloc[-1]["left_fz"])
                right_fz = float(df.iloc[-1]["right_fz"])
                if abs(left_fz) + abs(right_fz) > 30.0:
                    break

        for _ in range(25):
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
        seg_df.loc[len(seg_df)] = ["release", int(scene.t)]
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=45,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=120,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=110)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        _store_placement_outcome(metadata, quat_at_release, quat_after_retreat)
        return

    if place_mode == "bump":
        seg_df.loc[len(seg_df)] = ["bump_push", int(scene.t)]
        push_eef_z = franka.get_links_pos([8]).tolist()[0][2]
        for _ in range(BUMP_PUSH_STEPS):
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=np.array([place_x, place_y, push_eef_z]),
                quat=np.array([0, 1, 0, 0]),
            )
            franka.control_dofs_position(qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )

        seg_df.loc[len(seg_df)] = ["release", int(scene.t)]
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=45,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=110,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=120)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        _store_placement_outcome(metadata, quat_at_release, quat_after_retreat)
        return

    raise ValueError(f"Unsupported place mode: {place_mode}")


def run_rotation(
    scene,
    cam,
    cam_wrist,
    franka,
    gso_object,
    df,
    deform_csv,
    seg_df,
    paths,
    place_target_xy,
    place_mode,
    metadata,
    wall_plan=None,
    target_tile_plan=None,
    target_tile=None,
):
    name = paths["object_name"]
    metadata["upright_reference_quat_wxyz"] = _flatten_quat_wxyz(gso_object.get_quat()).tolist()
    motors_dof, fingers_dof = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 1000, 1000]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 100, 100]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -10, -10]),
        np.array([87, 87, 87, 87, 12, 12, 12, 10, 10]),
    )
    end_effector = franka.get_link("hand")

    offset = 0.074
    lower_obj_bound, upper_obj_bound = gso_object.get_AABB().cpu().numpy()
    object_height = float(upper_obj_bound[2] - lower_obj_bound[2])
    obj_center = 0.5 * (lower_obj_bound + upper_obj_bound)
    x, y = obj_center[0], obj_center[1]
    z = upper_obj_bound[2] + offset
    place_x, place_y = place_target_xy
    tile_top_z = float((target_tile_plan or {}).get("surface_center_z", TARGET_TILE_THICKNESS))
    place_eef_z_surface_aligned = tile_top_z + object_height + offset
    place_eef_z_nominal = _place_descent_target_z(
        place_mode, place_eef_z_surface_aligned
    )

    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z + 0.1]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    set_intentional_release(False)
    set_early_drop_monitor(False)

    metadata["target_xy"] = {"x": float(place_x), "y": float(place_y)}
    metadata["place_mode"] = str(place_mode)
    metadata["target_tile"] = target_tile_plan or _plan_target_tile("flat")
    metadata["place_eef_z_surface_aligned"] = float(place_eef_z_surface_aligned)
    metadata["place_eef_z_target"] = float(place_eef_z_nominal)
    metadata["step_descent_to_floor"] = bool(_is_step_mode(place_mode))
    if wall_plan is not None:
        metadata["bump_wall"] = wall_plan
    seg_df.loc[len(seg_df)] = ["grasp", int(scene.t)]
    mm.set_to_pose(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        name=name,
        qpos=qpos,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        steps=20,
        cam_wrist=cam_wrist,
    )
    mm.descend_to_object(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        name=name,
        end_effector=end_effector,
        x=x,
        y=y,
        z=z,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        steps=30,
        cam_wrist=cam_wrist,
    )

    current_force = 10.0
    mm.grasp_object_position(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        name=name,
        end_effector=end_effector,
        x=x,
        y=y,
        z=z,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        grip_force=-current_force,
        steps=200,
        cam_wrist=cam_wrist,
    )
    set_early_drop_monitor(True)

    seg_df.loc[len(seg_df)] = ["hold", int(scene.t)]
    hold_qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=np.array([0, 1, 0, 0]))
    for _ in range(GRASP_TO_LIFT_WAIT_STEPS):
        franka.control_dofs_position(hold_qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        make_step(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            gripper_force=-current_force,
            cam_wrist=cam_wrist,
        )

    seg_df.loc[len(seg_df)] = ["lift", int(scene.t)]
    lift_steps = 200
    lift_step_size = 0.00075
    if place_mode == "bump":
        # Keep transport height lower so only the lower part of the object tends to hit the low wall.
        lift_steps = 120
        lift_step_size = 0.00035
    for i in range(lift_steps):
        curr_z = z + (i * lift_step_size)
        if not mm.lift_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=x,
            y=y,
            z=curr_z,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=1,
            cam_wrist=cam_wrist,
        ):
            break

    seg_df.loc[len(seg_df)] = ["move_to_target", int(scene.t)]
    if _is_simple_mode(place_mode):
        # Keep the object above the same tile: no long XY transport.
        hold_qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z + 0.12]), quat=np.array([0, 1, 0, 0]))
        for _ in range(80):
            franka.control_dofs_position(hold_qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                gso_object=gso_object,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
        place_x, place_y = x, y
    else:
        transport_steps = 300
        if place_mode == "bump":
            transport_steps = 220
        elif (
            _is_inclined_mode(place_mode)
            or _is_step_mode(place_mode)
            or _is_wobble_mode(place_mode)
        ):
            # Tall objects such as cans are more likely to slip during a long lateral transfer.
            # The placement signal does not depend on transport duration, so keep this case brief.
            transport_steps = 120
        mm.move_to_place_xy(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=transport_steps,
            cam_wrist=cam_wrist,
        )
    retreat_target_z = franka.get_links_pos([8]).tolist()[0][2]

    seg_df.loc[len(seg_df)] = ["place", int(scene.t)]
    _run_place_phase(
        scene,
        cam,
        cam_wrist,
        franka,
        gso_object,
        df,
        deform_csv,
        seg_df,
        paths,
        end_effector,
        place_x,
        place_y,
        place_mode,
        place_eef_z_nominal,
        retreat_target_z,
        motors_dof,
        fingers_dof,
        current_force,
        metadata,
        target_tile,
    )
    set_early_drop_monitor(False)

    final_make_step(
        scene=scene,
        cam=cam,
        franka=franka,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        gso_object=gso_object,
        name=name,
        cam_wrist=cam_wrist,
    )


## -------------------------- GENERATE PLOTS -------------------------- ##
def generate_plots(df, paths):
    """Generates and saves the plots for the simulation results."""
    name = paths["object_name"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].axis("off")

    force_columns = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]
    for col in force_columns:
        axs[1].plot(df["step"], df[col], marker=".", label=col)

    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Force Components Over Time")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(paths["plot"], dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {paths['plot']}")
    plt.close(fig)


def save_results(force_df, deform_df, segment_df, metadata, paths):
    print(f"💾 Saving results to {paths['output_dir']}")
    force_df.to_csv(paths["force_data"], index=False)
    deform_df.to_csv(paths["deformation_data"], index=False)
    segment_df.to_csv(paths["segmentation_data"], index=False)
    with open(paths["metadata_data"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    generate_plots(df=force_df, paths=paths)


def move_images_to_target_paths(source_paths, target_paths):
    src = Path(source_paths["images_dir"])
    dst = Path(target_paths["images_dir"])
    if not src.exists():
        return
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing image directory: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _accidental_choice_exists(object_name: str, choice_name: str) -> bool:
    csv_dir = DATA_ROOT / OUTPUT_DATASET / "csv" / object_name / MATERIAL_TYPE / choice_name
    img_dir = DATA_ROOT / OUTPUT_DATASET / "images" / object_name / MATERIAL_TYPE / choice_name
    return csv_dir.exists() or img_dir.exists()


def _append_output_variant(name: str) -> str:
    label = OUTPUT_VARIANT_LABEL.strip()
    if not label:
        return name
    return f"{name}_{label}"


def _resolve_accidental_choice(object_name: str, target_choice: str, attempt_idx: int) -> str:
    # Preferred names: accidental_drop_1_{target}, accidental_drop_2_{target}
    base = _append_output_variant(f"accidental_drop_{attempt_idx}_{target_choice}")
    if not _accidental_choice_exists(object_name, base):
        return base

    # If the preferred name already exists (e.g. previous run), avoid overwrite.
    suffix = 2
    while True:
        candidate = f"{base}_run{suffix}"
        if not _accidental_choice_exists(object_name, candidate):
            return candidate
        suffix += 1


def _choice_exists(object_name: str, choice_name: str) -> bool:
    csv_dir = DATA_ROOT / OUTPUT_DATASET / "csv" / object_name / MATERIAL_TYPE / choice_name
    img_dir = DATA_ROOT / OUTPUT_DATASET / "images" / object_name / MATERIAL_TYPE / choice_name
    return csv_dir.exists() or img_dir.exists()


def _resolve_final_choice(object_name: str, target_choice: str) -> str:
    """
    Returns a safe destination choice for successful runs.
    If target_choice already exists from a previous run, append _runN.
    """
    base = _append_output_variant(target_choice)
    if not _choice_exists(object_name, base):
        return base

    suffix = 2
    while True:
        candidate = f"{base}_run{suffix}"
        if not _choice_exists(object_name, candidate):
            return candidate
        suffix += 1


def _resolve_attempt_choice(object_name: str, target_choice: str, attempt_idx: int) -> str:
    """
    Returns an isolated working directory name for each attempt to avoid image overwrite.
    """
    base = f"_attempt_{attempt_idx}_{target_choice}"
    if not _choice_exists(object_name, base):
        return base

    suffix = 2
    while True:
        candidate = f"{base}_run{suffix}"
        if not _choice_exists(object_name, candidate):
            return candidate
        suffix += 1


def _rename_attempt_outputs(object_name: str, source_choice: str, target_choice: str):
    csv_src = DATA_ROOT / OUTPUT_DATASET / "csv" / object_name / MATERIAL_TYPE / source_choice
    img_src = DATA_ROOT / OUTPUT_DATASET / "images" / object_name / MATERIAL_TYPE / source_choice
    csv_dst = DATA_ROOT / OUTPUT_DATASET / "csv" / object_name / MATERIAL_TYPE / target_choice
    img_dst = DATA_ROOT / OUTPUT_DATASET / "images" / object_name / MATERIAL_TYPE / target_choice

    if csv_dst.exists() or img_dst.exists():
        raise FileExistsError(f"Destination already exists: csv={csv_dst}, images={img_dst}")

    if csv_src.exists():
        csv_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(csv_src), str(csv_dst))
        # Rename files to match the new deformation name for downstream scripts.
        for p in csv_dst.iterdir():
            if source_choice in p.name:
                p.rename(p.with_name(p.name.replace(source_choice, target_choice, 1)))

    if img_src.exists():
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_src), str(img_dst))


def run_single_simulation(
    object_name: str,
    target_choice: str,
    output_choice: str | None = None,
    gpu_id: int | None = None,
):
    output_choice = target_choice if output_choice is None else output_choice
    paths = setup_paths(object_name, output_choice)
    force_df = pd.DataFrame(
        columns=[
            "step",
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
            "dof_0",
            "dof_1",
            "dof_2",
            "dof_3",
            "dof_4",
            "dof_5",
            "dof_6",
            "dof_7",
            "dof_8",
            "eef_x",
            "eef_y",
            "eef_z",
            "left_finger_x",
            "left_finger_y",
            "left_finger_z",
            "right_finger_x",
            "right_finger_y",
            "right_finger_z",
            "control_left_finger",
            "control_right_finger",
            "obj_COM_x",
            "obj_COM_y",
            "obj_COM_z",
            "obj_mass",
            "obj_min_x",
            "obj_min_y",
            "obj_min_z",
            "obj_max_x",
            "obj_max_y",
            "obj_max_z",
            "obj_left_finger",
            "obj_right_finger",
            "obj_plane",
            "obj_target_tile",
            "obj_obstacle",
            "target_surface_angle_deg",
        ]
    )
    deform_df = pd.DataFrame(columns=["step", "deformations", "grip_force"])
    segment_df = pd.DataFrame(columns=["action", "step"])
    metadata = {
        "object_name": object_name,
        "material_type": MATERIAL_TYPE,
        "target_choice": target_choice,
        "output_choice": output_choice,
        "topple_threshold_deg": float(TOPPLE_THRESHOLD_DEG),
        "target_xy": None,
        "place_mode": None,
        "tilt_xy_deg": None,
        "post_release_tilt_change_deg": None,
        "outcome_tilt_xy_deg": None,
        "toppled": None,
        "contact_logging_version": 3,
        "gpu_id": (
            int(os.environ["GENESIS_ASSIGNED_GPU"])
            if "GENESIS_ASSIGNED_GPU" in os.environ
            else (gpu_id if gpu_id is not None and gpu_id >= 0 else None)
        ),
        "compute_device": (
            "cpu"
            if gpu_id == -1
            else f"physical_cuda:{os.environ['GENESIS_ASSIGNED_GPU']}"
            if "GENESIS_ASSIGNED_GPU" in os.environ
            else (f"cuda:{gpu_id}" if gpu_id is not None else "auto")
        ),
        "input_dataset": DATASET,
        "output_dataset": OUTPUT_DATASET,
        "output_variant_label": OUTPUT_VARIANT_LABEL,
        "capture_images": bool(CAPTURE_IMAGES),
    }

    scene = None
    cam = cam_wrist = franka = gso_object = None
    wall_plan = None
    try:
        (
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            place_target_xy,
            target_tile,
            obstacle_entity,
            wall_plan,
            target_tile_plan,
            target_ramp,
        ) = create_scene(str(paths["input_obj"]), target_choice, gpu_id=gpu_id)
        target_supports = [target_tile]
        if target_ramp is not None:
            target_supports.append(target_ramp)
        set_object_contact_targets(target_tile=target_supports, obstacle=obstacle_entity)
        run_rotation(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            force_df,
            deform_df,
            segment_df,
            paths,
            place_target_xy,
            target_choice,
            metadata,
            wall_plan,
            target_tile_plan,
            target_tile,
        )
    except EarlyDropDetected as exc:
        segment_df.loc[len(segment_df)] = [f"early_drop_{target_choice}", int(scene.t)]
        metadata["terminated_early_drop"] = True
        raise EarlyDropSimulationError(str(exc), paths, force_df, deform_df, segment_df, metadata) from exc
    finally:
        # Ensure each attempt fully tears down Genesis before any retry.
        set_object_contact_targets(target_tile=None, obstacle=None)
        gs.destroy()
    metadata["terminated_early_drop"] = False
    return paths, force_df, deform_df, segment_df, metadata


def main(object_name: str, target_choice: str = "gentle", gpu_id: int | None = None):
    if target_choice == "inclined":
        for expanded_target in _expand_target_choices([target_choice]):
            main(object_name, expanded_target, gpu_id=gpu_id)
        return
    if target_choice == "step":
        for expanded_target in _expand_target_choices([target_choice]):
            main(object_name, expanded_target, gpu_id=gpu_id)
        return

    assigned_gpu = os.environ.get("GENESIS_ASSIGNED_GPU")
    if gpu_id == -1:
        device_label = "CPU"
    elif assigned_gpu is not None:
        device_label = f"physical GPU {assigned_gpu} (process-local cuda:0)"
    elif gpu_id is None:
        device_label = "default device"
    else:
        device_label = f"GPU {gpu_id}"
    print(f"🚀 Starting simulation for '{object_name}' with target '{target_choice}' on {device_label}...")
    input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / "model.obj"
    if DATASET_TYPE == "hugging_face":
        input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / f"{object_name}.glb"
    if not input_obj_path.exists():
        print(f"❌ Aborting: Input file not found at: {input_obj_path}")
        return

    last_failed = None
    for attempt in range(2):
        attempt_idx = attempt + 1
        attempt_choice = _resolve_attempt_choice(object_name, target_choice, attempt_idx)
        try:
            paths, force_df, deform_df, segment_df, metadata = run_single_simulation(
                object_name,
                target_choice,
                output_choice=attempt_choice,
                gpu_id=gpu_id,
            )
            final_choice = _resolve_final_choice(object_name, target_choice)
            metadata["target_choice"] = final_choice
            metadata["source_target_choice"] = target_choice
            metadata["attempt_choice"] = attempt_choice
            save_results(force_df, deform_df, segment_df, metadata, paths)
            _rename_attempt_outputs(object_name, attempt_choice, final_choice)
            if final_choice != target_choice:
                print(
                    f"ℹ️ Destination '{target_choice}' already exists. "
                    f"Saved to '{final_choice}' instead."
                )
            print(f"✅ Finished simulation for '{object_name}'.")
            return
        except EarlyDropSimulationError as exc:
            print(f"⚠️ Early drop detected for '{object_name}' / '{target_choice}' on attempt {attempt + 1}: {exc}")
            early_choice = _resolve_accidental_choice(object_name, target_choice, attempt + 1)
            early_metadata = dict(exc.metadata)
            early_metadata["target_choice"] = early_choice
            early_metadata["source_target_choice"] = target_choice
            early_metadata["attempt_choice"] = attempt_choice
            # 1) Save all outputs into the current attempt directory.
            save_results(exc.force_df, exc.deform_df, exc.segment_df, early_metadata, exc.paths)
            # 2) Rename the whole attempt directory to accidental_drop_*.
            _rename_attempt_outputs(object_name, attempt_choice, early_choice)
            print(f"⚠️ Saved early-drop result to '{early_choice}'.")
            last_failed = (early_choice, exc.force_df, exc.deform_df, exc.segment_df, early_metadata)
            if attempt == 0:
                print(f"🔁 Retrying '{object_name}' / '{target_choice}' once after early drop.")
                continue
            print(f"❌ Early drop persisted after retry for '{object_name}' / '{target_choice}'.")
            return

    if last_failed is not None:
        print(f"❌ Finished with early-drop result for '{object_name}' / '{target_choice}'.")


def _completed_runs_by_task() -> Counter:
    """Count completed top-level invocations from their saved metadata."""
    completed = Counter()
    csv_root = DATA_ROOT / OUTPUT_DATASET / "csv"
    if not csv_root.exists():
        return completed

    for metadata_path in csv_root.glob("*/Rigid/*/*_metadata_*.json"):
        if metadata_path.parent.name.startswith("_attempt_"):
            # A working directory can contain partially saved metadata when the
            # batch is interrupted between save and final rename.
            continue
        try:
            with open(metadata_path, encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        object_name = str(metadata.get("object_name") or metadata_path.parents[2].name)
        source_target = metadata.get("source_target_choice")
        if not source_target:
            continue
        if str(metadata.get("output_variant_label") or "").strip() != OUTPUT_VARIANT_LABEL:
            continue

        if bool(metadata.get("terminated_early_drop")):
            # Attempt 1 is not a completed invocation because main() retries it.
            # A saved attempt 2 means the invocation ended after both failures.
            attempt_choice = str(metadata.get("attempt_choice") or "")
            if not attempt_choice.startswith("_attempt_2_"):
                continue

        completed[(object_name, str(source_target))] += 1

    return completed


def get_tasks_to_run():
    if 0: # DO NOT CHANGE
        return [("001_chips_can", "bump")]
        # return [("001_chips_can", "drop"), ("001_chips_can", "gentle"), ("001_chips_can", "firm"), ("002_master_chef_can", "drop"), ("002_master_chef_can", "gentle"), ("002_master_chef_can", "firm"), ("003_cracker_box", "drop"), ("003_cracker_box", "gentle"), ("003_cracker_box", "firm"), ("004_sugar_box", "drop"), ("004_sugar_box", "gentle"), ("004_sugar_box", "firm"), ("005_tomato_soup_can", "drop"), ("005_tomato_soup_can", "gentle"), ("005_tomato_soup_can", "firm")]

    tasks = []
    objects_dir = DATA_ROOT / "objects" / DATASET_TYPE
    if not objects_dir.exists():
        print(f"❌ Error: Input directory '{objects_dir}' not found.")
        return []

    object_names = [
        directory.name
        for directory in objects_dir.iterdir()
        if directory.is_dir() and (directory / "model.obj").is_file()
    ]
    print(f"🔍 Found {len(object_names)} runnable objects in '{objects_dir}'.")
    expanded_targets = _expand_target_choices(TARGET_CHOICES)
    completed_runs = _completed_runs_by_task() if REQUIRED_RUNS_PER_TASK else Counter()
    for target in expanded_targets:
        queued_for_target = 0
        for name in object_names:
            completed_count = completed_runs[(name, target)]
            if REQUIRED_RUNS_PER_TASK and completed_count >= REQUIRED_RUNS_PER_TASK:
                continue
            if REQUIRED_RUNS_PER_TASK:
                print(
                    f"  - Resuming '{name}' / '{target}' "
                    f"({completed_count}/{REQUIRED_RUNS_PER_TASK} completed)."
                )
            else:
                print(f"  - Queueing '{name}' with target '{target}' (unbounded mode).")
            tasks.append((name, target))
            queued_for_target += 1
        print(f"🔄 Target '{target}': queued {queued_for_target} object(s).")

    return tasks


if __name__ == "__main__":
    single_object = os.environ.get("GENESIS_SINGLE_OBJECT")
    single_target = os.environ.get("GENESIS_SINGLE_TARGET")
    if single_object or single_target:
        if not (single_object and single_target):
            raise SystemExit("GENESIS_SINGLE_OBJECT and GENESIS_SINGLE_TARGET must be set together.")
        child_gpu_id = 0 if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() else -1
        main(single_object, single_target, gpu_id=child_gpu_id)
        raise SystemExit(0)

    tasks_to_run = get_tasks_to_run()
    if not tasks_to_run:
        print("🎉 No new simulations to run.")
        exit()

    print(f"\nFound {len(tasks_to_run)} simulation task(s) to run.")
    gpu_ids = _configured_gpu_ids()
    if gpu_ids:
        print(f"🎛️ Using physical GPU IDs {gpu_ids} with least-loaded task assignment.")
    else:
        print("🖥️ No GPU selected; simulations will use the CPU backend.")
    targets_in_order = list(dict.fromkeys(target for _, target in tasks_to_run))
    for target_choice in targets_in_order:
        target_tasks = [task for task in tasks_to_run if task[1] == target_choice]
        print(f"\n▶ Starting full pass '{target_choice}' ({len(target_tasks)} objects).")
        processes = []
        failed_processes = 0
        for object_name, _ in target_tasks:
            while len(processes) >= MAX_PARALLEL_PROCESSES:
                still_running = []
                for process, assigned_gpu_id in processes:
                    return_code = process.poll()
                    if return_code is None:
                        still_running.append((process, assigned_gpu_id))
                    elif return_code != 0:
                        failed_processes += 1
                processes = still_running
                time.sleep(1)
            if gpu_ids:
                active_per_gpu = {
                    gpu_id: sum(assigned_gpu_id == gpu_id for _, assigned_gpu_id in processes) for gpu_id in gpu_ids
                }
                physical_gpu_id = min(gpu_ids, key=lambda gpu_id: (active_per_gpu[gpu_id], gpu_ids.index(gpu_id)))
            else:
                physical_gpu_id = None
            child_env = os.environ.copy()
            child_env["GENESIS_SINGLE_OBJECT"] = object_name
            child_env["GENESIS_SINGLE_TARGET"] = target_choice
            if physical_gpu_id is None:
                child_env["CUDA_VISIBLE_DEVICES"] = ""
                child_env.pop("GENESIS_ASSIGNED_GPU", None)
            else:
                child_env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
                child_env["GENESIS_ASSIGNED_GPU"] = str(physical_gpu_id)
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], env=child_env)
            processes.append((p, physical_gpu_id))
            time.sleep(WORKER_SPAWN_DELAY_SECONDS)

        for p, _ in processes:
            if p.wait() != 0:
                failed_processes += 1
        if failed_processes:
            print(f"⚠️ Full pass '{target_choice}' had {failed_processes} failed worker process(es).")
        print(f"✅ Completed full pass '{target_choice}'.")
    print("\n\nAll simulations completed.")
