#!/usr/bin/env python3
"""Generate shareable pick-and-place videos with synchronized force plots.

The two canonical scenarios use the same YCB object, robot motion, camera, and
rendering style.  Only the destination support changes: a fixed horizontal
tile or the compliant single-axis wobble deck used during data collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = ROOT / "src" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

import run_genesis_sim as collection  # noqa: E402


DT = 0.01
RENDER_EVERY = 5
VIDEO_FPS = int(round(1.0 / (DT * RENDER_EVERY)))
VIDEO_RESOLUTION = (960, 540)
GRIP_FORCE_N = 10.0
OBJECT_COLOR = (0, 255, 0)
PEDESTAL_COLOR = (0.10, 0.38, 0.70, 1.0)
SOURCE_XY = (0.58, -0.12)
PLACE_XY = (0.58, 0.08)
WOBBLE_CENTER_XY = (PLACE_XY[0], PLACE_XY[1] + 0.065)
STABLE_SURFACE_TOP_M = 0.016
WOBBLE_SURFACE_TOP_M = (
    collection.WOBBLE_PIVOT_Z_M + collection.WOBBLE_DECK_HALF_SIZE[2]
)
FORCE_COLUMNS = (
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
)
PHASE_LABELS = {
    "ready": "Ready",
    "approach_object": "Approach object",
    "descend_to_grasp": "Descend to grasp",
    "close_gripper": "Close gripper",
    "lift": "Lift",
    "transport": "Transport",
    "approach_surface": "Approach surface",
    "constant_speed_descent": "Place / load surface",
    "surface_contact_hold": "Maintain contact",
    "intentional_release": "Release",
    "retract": "Retract",
    "post_release_settle": "Observe after release",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    support: str
    press_depth_m: float


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            "stable_plane",
            "Stable horizontal surface",
            "stable",
            0.002,
        ),
        Scenario(
            "unstable_wobble",
            "Unstable wobbling surface",
            "wobble",
            0.016,
        ),
    )
}


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _wobble_mjcf() -> str:
    half_x, half_y, half_z = collection.WOBBLE_DECK_HALF_SIZE
    limit_rad = np.deg2rad(12.0)
    center_x, center_y = WOBBLE_CENTER_XY
    return f"""
<mujoco model="wobble_support">
  <compiler angle="radian"/>
  <worldbody>
    <body name="wobble_deck" pos="{center_x} {center_y} {collection.WOBBLE_PIVOT_Z_M}">
      <joint name="wobble_hinge" type="hinge" axis="1 0 0" limited="true"
             range="{-limit_rad} {limit_rad}" stiffness="1.0"
             damping="0.08" armature="0.012"/>
      <geom name="deck" type="box" size="{half_x} {half_y} {half_z}"
            mass="0.50" rgba="0.10 0.35 0.90 1"
            friction="1.2 0.01 0.0001"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def _add_support(scene, scenario: Scenario):
    if scenario.support == "stable":
        return scene.add_entity(
            material=gs.materials.Rigid(rho=800, friction=1.0, coup_friction=1.0),
            morph=gs.morphs.Box(
                size=(0.24, 0.21, STABLE_SURFACE_TOP_M),
                pos=(PLACE_XY[0], PLACE_XY[1], 0.5 * STABLE_SURFACE_TOP_M),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=collection.TARGET_TILE_COLOR),
        )

    scene.add_entity(
        material=gs.materials.Rigid(rho=900, friction=1.0),
        morph=gs.morphs.Box(
            size=(0.070, 0.070, 0.060),
            pos=(WOBBLE_CENTER_XY[0], WOBBLE_CENTER_XY[1], 0.030),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=PEDESTAL_COLOR),
    )
    return scene.add_entity(
        material=gs.materials.Rigid(rho=800, friction=1.2, coup_friction=1.0),
        morph=gs.morphs.MJCF(file=_wobble_mjcf()),
        surface=gs.surfaces.Default(color=collection.TARGET_TILE_COLOR),
    )


def _ik(franka, hand, position):
    return franka.inverse_kinematics(
        link=hand,
        pos=np.asarray(position, dtype=float)[None, :],
        quat=np.asarray((0.0, 1.0, 0.0, 0.0), dtype=float)[None, :],
    )


def _command(franka, motors, fingers, qpos, opening):
    franka.control_dofs_position(qpos[0, :-2], motors)
    if opening <= 0.0:
        franka.control_dofs_force(
            np.asarray((-GRIP_FORCE_N, -GRIP_FORCE_N)),
            fingers,
        )
    else:
        franka.control_dofs_position(np.asarray((opening, opening)), fingers)


def _force12(franka, finger_link_indices) -> np.ndarray:
    force = _numpy(
        franka.get_links_contact_force(finger_link_indices, sensor=True)
    ).reshape(-1, 2, 3)[0]
    torque = _numpy(
        franka.get_links_contact_torque(finger_link_indices, sensor=True)
    ).reshape(-1, 2, 3)[0]
    return np.concatenate((force[0], torque[0], force[1], torque[1])).astype(float)


def _entity_aabb(entity) -> tuple[np.ndarray, np.ndarray]:
    bounds = _numpy(entity.get_AABB()).reshape(-1, 3)
    return bounds[0], bounds[1]


def _place_mesh_on_ground(obj):
    # Use the collision AABB here.  Aligning from the visual bounds can leave
    # a decomposed collision hull slightly inside the plane and tip the object
    # during the initial settling steps.
    bounds = _numpy(obj.get_AABB()).reshape(-1, 3)
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    shift = np.asarray(
        (
            SOURCE_XY[0] - center_xy[0],
            SOURCE_XY[1] - center_xy[1],
            0.003 - bounds[0, 2],
        ),
        dtype=float,
    )
    obj.set_pos(_numpy(obj.get_pos()).reshape(-1, 3)[0] + shift, zero_velocity=True)


class Trace:
    def __init__(
        self,
        scene,
        camera,
        franka,
        hand_idx,
        finger_link_indices,
        obj,
        support,
        support_kind: str,
    ):
        self.scene = scene
        self.camera = camera
        self.franka = franka
        self.hand_idx = hand_idx
        self.finger_link_indices = finger_link_indices
        self.obj = obj
        self.support = support
        self.support_kind = support_kind
        self.phase = "ready"
        self.rows: list[dict] = []
        self.render_steps: list[int] = []
        self.grasp_assist_offset: np.ndarray | None = None
        self.grasp_assist_quat: np.ndarray | None = None

    def enable_grasp_assist(self):
        hand_pos = _numpy(self.franka.get_links_pos([self.hand_idx])).reshape(-1, 3)[0]
        object_pos = _numpy(self.obj.get_pos()).reshape(-1, 3)[0]
        self.grasp_assist_offset = object_pos - hand_pos
        self.grasp_assist_quat = _numpy(self.obj.get_quat()).reshape(-1, 4)[0].copy()

    def disable_grasp_assist(self):
        self.grasp_assist_offset = None
        self.grasp_assist_quat = None

    def _apply_grasp_assist(self):
        if self.grasp_assist_offset is None:
            return
        hand_pos = _numpy(self.franka.get_links_pos([self.hand_idx])).reshape(-1, 3)[0]
        self.obj.set_pos(hand_pos + self.grasp_assist_offset, zero_velocity=True)
        self.obj.set_quat(self.grasp_assist_quat)

    def _support_angle_deg(self) -> float:
        if self.support_kind != "wobble":
            return 0.0
        angle = _numpy(self.support.get_dofs_position()).reshape(-1)[0]
        return float(np.rad2deg(angle))

    def step(self):
        self._apply_grasp_assist()
        self.scene.step()
        self._apply_grasp_assist()
        force = _force12(self.franka, self.finger_link_indices)
        obj_pos = _numpy(self.obj.get_root_COM()).reshape(-1, 3)[0]
        row = {
            "step": len(self.rows),
            "time_s": len(self.rows) * DT,
            "phase": self.phase,
            "support_angle_deg": self._support_angle_deg(),
            "object_x": float(obj_pos[0]),
            "object_y": float(obj_pos[1]),
            "object_z": float(obj_pos[2]),
        }
        row.update({column: float(value) for column, value in zip(FORCE_COLUMNS, force)})
        self.rows.append(row)
        if row["step"] % RENDER_EVERY == 0:
            self.camera.render(rgb=True)
            self.render_steps.append(row["step"])


def _move(trace, franka, hand, hand_idx, motors, fingers, target, opening, steps):
    start = _numpy(franka.get_links_pos([hand_idx])).reshape(-1, 3)[0].astype(float)
    target = np.asarray(target, dtype=float)
    for index in range(max(1, int(steps))):
        alpha = (index + 1) / max(1, int(steps))
        position = (1.0 - alpha) * start + alpha * target
        _command(franka, motors, fingers, _ik(franka, hand, position), opening)
        trace.step()


def _hold(trace, franka, hand, motors, fingers, position, opening, steps):
    qpos = _ik(franka, hand, position)
    for _ in range(int(steps)):
        _command(franka, motors, fingers, qpos, opening)
        trace.step()


def _change_gripper(trace, franka, motors, fingers, start, end, steps):
    arm = _numpy(franka.get_dofs_position(motors)).reshape(-1)
    for index in range(int(steps)):
        alpha = (index + 1) / max(1, int(steps))
        opening = (1.0 - alpha) * start + alpha * end
        franka.control_dofs_position(arm, motors)
        franka.control_dofs_position(np.asarray((opening, opening)), fingers)
        trace.step()


def _save_csv(rows: list[dict], path: Path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _polyline(
    panel,
    values,
    current_step,
    x0,
    y0,
    width,
    height,
    low,
    high,
    color,
):
    values = np.asarray(values, dtype=float)
    count = min(len(values), current_step + 1)
    if count < 2 or high <= low:
        return
    indices = np.arange(count, dtype=float)
    xs = (x0 + indices * width / max(1, len(values) - 1)).astype(np.int32)
    normalized = np.clip((values[:count] - low) / (high - low), 0.0, 1.0)
    ys = (y0 + height - normalized * height).astype(np.int32)
    cv2.polylines(panel, [np.column_stack((xs, ys))], False, color, 2, cv2.LINE_AA)


def _magnitudes(rows: list[dict]):
    left_force = np.asarray(
        [np.linalg.norm([row["left_fx"], row["left_fy"], row["left_fz"]]) for row in rows]
    )
    right_force = np.asarray(
        [np.linalg.norm([row["right_fx"], row["right_fy"], row["right_fz"]]) for row in rows]
    )
    left_torque = np.asarray(
        [np.linalg.norm([row["left_tx"], row["left_ty"], row["left_tz"]]) for row in rows]
    )
    right_torque = np.asarray(
        [np.linalg.norm([row["right_tx"], row["right_ty"], row["right_tz"]]) for row in rows]
    )
    angle = np.asarray([row["support_angle_deg"] for row in rows])
    return left_force, right_force, left_torque, right_torque, angle


def _common_plot_limits(results: list[dict]) -> dict[str, float]:
    forces = []
    torques = []
    angles = []
    for result in results:
        left_force, right_force, left_torque, right_torque, angle = _magnitudes(result["_rows"])
        forces.extend((left_force, right_force))
        torques.extend((left_torque, right_torque))
        angles.append(angle)
    force_values = np.concatenate(forces)
    torque_values = np.concatenate(torques)
    angle_values = np.concatenate(angles)
    return {
        "force_hi": max(5.0, float(np.percentile(force_values, 99.5)) * 1.10),
        "torque_hi": max(0.05, float(np.percentile(torque_values, 99.5)) * 1.10),
        "angle_hi": max(2.0, float(np.max(np.abs(angle_values))) * 1.15),
    }


def _make_synchronized_video(result: dict, limits: dict[str, float]):
    raw_path = Path(result["raw_video"])
    out_path = Path(result["synchronized_video"])
    preview_path = Path(result["preview"])
    rows = result["_rows"]
    render_steps = result["_render_steps"]
    left_force, right_force, left_torque, right_torque, angle = _magnitudes(rows)

    capture = cv2.VideoCapture(str(raw_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or VIDEO_FPS
    temporary = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width * 2, height),
    )
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError(f"Could not open synchronized video streams for {out_path}")

    if result["scenario"]["support"] == "wobble":
        preview_step = int(np.argmax(np.abs(angle)))
    else:
        contact_steps = [
            row["step"] for row in rows if row["phase"] == "surface_contact_hold"
        ]
        preview_step = contact_steps[len(contact_steps) // 2]

    frame_index = 0
    preview_written = False
    last_composed = None
    graph_x = 82
    graph_width = width - 112
    graph_height = max(90, (height - 166) // 3)
    graph_ys = (92, 112 + graph_height, 132 + graph_height * 2)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current_step = render_steps[min(frame_index, len(render_steps) - 1)]
        row = rows[current_step]
        panel = np.full((height, width, 3), 248, dtype=np.uint8)
        cv2.putText(
            panel,
            result["scenario"]["label"],
            (24, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        phase = PHASE_LABELS.get(row["phase"], row["phase"])
        cv2.putText(
            panel,
            f"t = {row['time_s']:.2f} s   {phase}",
            (24, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(panel, "left", (width - 154, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 85, 40), 2)
        cv2.putText(panel, "right", (width - 92, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (45, 145, 45), 2)

        specifications = (
            (
                left_force,
                right_force,
                0.0,
                limits["force_hi"],
                "Fingertip force magnitude [N]",
                f"L {left_force[current_step]:.2f}   R {right_force[current_step]:.2f}",
            ),
            (
                left_torque,
                right_torque,
                0.0,
                limits["torque_hi"],
                "Fingertip torque magnitude [N m]",
                f"L {left_torque[current_step]:.3f}   R {right_torque[current_step]:.3f}",
            ),
            (
                angle,
                None,
                -limits["angle_hi"],
                limits["angle_hi"],
                "Support angle [deg]",
                f"{angle[current_step]:+.2f}",
            ),
        )
        for graph_y, specification in zip(graph_ys, specifications):
            first, second, low, high, title, current_text = specification
            cv2.rectangle(
                panel,
                (graph_x, graph_y),
                (graph_x + graph_width, graph_y + graph_height),
                (205, 205, 205),
                1,
            )
            cv2.putText(panel, title, (graph_x, graph_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (45, 45, 45), 1, cv2.LINE_AA)
            cv2.putText(panel, current_text, (width - 190, graph_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (55, 55, 55), 1, cv2.LINE_AA)
            cv2.putText(panel, f"{high:.2f}", (10, graph_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 70), 1)
            cv2.putText(panel, f"{low:.2f}", (10, graph_y + graph_height), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 70), 1)
            _polyline(panel, first, current_step, graph_x, graph_y, graph_width, graph_height, low, high, (220, 85, 40))
            if second is not None:
                _polyline(panel, second, current_step, graph_x, graph_y, graph_width, graph_height, low, high, (45, 145, 45))
            cursor_x = graph_x + int(graph_width * current_step / max(1, len(rows) - 1))
            cv2.line(panel, (cursor_x, graph_y), (cursor_x, graph_y + graph_height), (30, 30, 220), 2)

        composed = np.hstack((frame, panel))
        writer.write(composed)
        last_composed = composed
        if not preview_written and current_step >= preview_step:
            cv2.imwrite(str(preview_path), composed)
            preview_written = True
        frame_index += 1

    capture.release()
    writer.release()
    if not preview_written and last_composed is not None:
        cv2.imwrite(str(preview_path), last_composed)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
    )
    temporary.unlink()
    result["video_frames"] = frame_index


def _run_scenario(scenario: Scenario, object_path: Path, output_dir: Path) -> dict:
    scenario_dir = output_dir / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    raw_video = scenario_dir / "raw_simulation.mp4"
    synchronized_video = scenario_dir / "simulation_with_force_plot.mp4"
    trace_csv = scenario_dir / "force_trace.csv"
    preview = scenario_dir / "preview.png"

    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    camera = None
    recording = False
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT),
            show_viewer=False,
        )
        scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(coup_friction=0.0),
        )
        support = _add_support(scene, scenario)
        object_scale, object_euler = collection.set_grasp(object_path)
        object_min_corner, _ = collection.get_obj_bounds(object_path)
        scale_vector = np.asarray(object_scale, dtype=float)
        scale_z = float(scale_vector if scale_vector.ndim == 0 else scale_vector[2])
        initial_object_z = 0.003 + max(0.0, -scale_z * float(object_min_corner[2]))
        obj = scene.add_entity(
            material=gs.materials.Rigid(rho=200, friction=5.0, coup_friction=5.0),
            morph=gs.morphs.Mesh(
                file=str(object_path),
                scale=object_scale,
                # Keep the object outside the robot's initialization sweep.
                # It is moved to SOURCE_XY after the robot reaches home.
                pos=(1.50, 1.50, initial_object_z),
                euler=object_euler,
                convexify=True,
            ),
            surface=gs.surfaces.Default(color=OBJECT_COLOR),
        )
        franka = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
            material=gs.materials.Rigid(coup_friction=5.0, friction=5.0),
        )
        camera = scene.add_camera(
            res=VIDEO_RESOLUTION,
            pos=(1.38, -0.05, 0.38),
            lookat=(0.56, -0.05, 0.15),
            fov=40.0,
            near=0.03,
            GUI=False,
        )
        scene.build(n_envs=1)

        motors = np.arange(7)
        fingers = np.arange(7, 9)
        hand = franka.get_link("hand")
        hand_idx = next(index for index, link in enumerate(franka.links) if link.name == "hand")
        left_idx = next(index for index, link in enumerate(franka.links) if link.name == "left_finger")
        right_idx = next(index for index, link in enumerate(franka.links) if link.name == "right_finger")
        franka.set_dofs_kp(np.asarray([4500, 4500, 3500, 3500, 2000, 2000, 2000, 1000, 1000]))
        franka.set_dofs_kv(np.asarray([450, 450, 350, 350, 200, 200, 200, 100, 100]))
        franka.set_dofs_force_range(
            np.asarray([-87, -87, -87, -87, -12, -12, -12, -10, -10]),
            np.asarray([87, 87, 87, 87, 12, 12, 12, 10, 10]),
        )

        home_position = (0.47, -0.05, 0.34)
        home = _ik(franka, hand, home_position)
        home[0, -2:] = 0.04
        franka.set_dofs_position(home[0, :-2], motors)
        franka.set_dofs_position(home[0, -2:], fingers)
        for _ in range(80):
            _command(franka, motors, fingers, home, 0.04)
            scene.step()
        _place_mesh_on_ground(obj)
        for _ in range(60):
            _command(franka, motors, fingers, home, 0.04)
            scene.step()

        _, upper = _entity_aabb(obj)
        grasp_z = float(upper[2] + 0.055)
        hover_z = max(0.31, grasp_z + 0.10)
        transport_z = max(0.42, grasp_z + 0.16)
        support_top = (
            STABLE_SURFACE_TOP_M
            if scenario.support == "stable"
            else WOBBLE_SURFACE_TOP_M
        )
        nominal_place_z = grasp_z + support_top
        pre_contact_z = nominal_place_z + 0.050
        descent_target_z = nominal_place_z - scenario.press_depth_m

        trace = Trace(
            scene,
            camera,
            franka,
            hand_idx,
            (left_idx, right_idx),
            obj,
            support,
            scenario.support,
        )
        camera.start_recording(save_to_filename=str(raw_video), fps=VIDEO_FPS)
        recording = True

        _hold(trace, franka, hand, motors, fingers, home_position, 0.04, 30)
        trace.phase = "approach_object"
        _move(trace, franka, hand, hand_idx, motors, fingers, (*SOURCE_XY, hover_z), 0.04, 80)
        trace.phase = "descend_to_grasp"
        _move(trace, franka, hand, hand_idx, motors, fingers, (*SOURCE_XY, grasp_z), 0.04, 70)
        trace.phase = "close_gripper"
        trace.enable_grasp_assist()
        _change_gripper(trace, franka, motors, fingers, 0.04, 0.0, 200)
        _hold(trace, franka, hand, motors, fingers, (*SOURCE_XY, grasp_z), 0.0, 100)
        trace.phase = "lift"
        _move(trace, franka, hand, hand_idx, motors, fingers, (*SOURCE_XY, transport_z), 0.0, 200)
        trace.phase = "transport"
        _move(trace, franka, hand, hand_idx, motors, fingers, (*PLACE_XY, transport_z), 0.0, 120)
        trace.phase = "approach_surface"
        _move(trace, franka, hand, hand_idx, motors, fingers, (*PLACE_XY, pre_contact_z), 0.0, 100)
        trace.phase = "constant_speed_descent"
        descent_steps = int(np.ceil((pre_contact_z - descent_target_z) / (0.03 * DT)))
        _move(trace, franka, hand, hand_idx, motors, fingers, (*PLACE_XY, descent_target_z), 0.0, descent_steps)
        trace.phase = "surface_contact_hold"
        _hold(trace, franka, hand, motors, fingers, (*PLACE_XY, descent_target_z), 0.0, 90)
        trace.phase = "intentional_release"
        trace.disable_grasp_assist()
        _change_gripper(trace, franka, motors, fingers, 0.0, 0.04, 50)
        trace.phase = "retract"
        _move(trace, franka, hand, hand_idx, motors, fingers, (*PLACE_XY, pre_contact_z), 0.04, 90)
        trace.phase = "post_release_settle"
        _hold(trace, franka, hand, motors, fingers, (*PLACE_XY, pre_contact_z), 0.04, 120)

        camera.stop_recording()
        recording = False
        _save_csv(trace.rows, trace_csv)

        left_force, right_force, left_torque, right_torque, angle = _magnitudes(trace.rows)
        final_position = _numpy(obj.get_root_COM()).reshape(-1, 3)[0]
        return {
            "scenario": asdict(scenario),
            "object": object_path.parent.name,
            "duration_s": len(trace.rows) * DT,
            "simulation_steps": len(trace.rows),
            "object_scale": object_scale,
            "object_euler_deg": object_euler,
            "grasp_z_m": grasp_z,
            "grasp_stabilization": "pose-follow constraint while the gripper is closed",
            "max_abs_support_angle_deg": float(np.max(np.abs(angle))),
            "peak_left_force_n": float(np.max(left_force)),
            "peak_right_force_n": float(np.max(right_force)),
            "peak_left_torque_nm": float(np.max(left_torque)),
            "peak_right_torque_nm": float(np.max(right_torque)),
            "final_object_com_xyz": final_position.tolist(),
            "raw_video": str(raw_video.resolve()),
            "synchronized_video": str(synchronized_video.resolve()),
            "force_trace_csv": str(trace_csv.resolve()),
            "preview": str(preview.resolve()),
            "_rows": trace.rows,
            "_render_steps": trace.render_steps,
        }
    finally:
        if recording and camera is not None:
            camera.stop_recording()
        gs.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument("--object", default="001_chips_can")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "fig" / "pick_place_force_videos",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    object_path = ROOT / "data" / "objects" / "ycb" / args.object / "model.obj"
    if not object_path.exists():
        raise FileNotFoundError(f"YCB object not found: {object_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.scenarios:
        print(f"[pick-place-video] simulating {name}", flush=True)
        results.append(_run_scenario(SCENARIOS[name], object_path, args.output_dir))

    limits = _common_plot_limits(results)
    for result in results:
        print(f"[pick-place-video] composing {result['scenario']['name']}", flush=True)
        _make_synchronized_video(result, limits)

    public_results = []
    for result in results:
        public_result = {key: value for key, value in result.items() if not key.startswith("_")}
        metadata_path = Path(result["synchronized_video"]).parent / "metadata.json"
        metadata_path.write_text(json.dumps(public_result, indent=2), encoding="utf-8")
        public_result["metadata"] = str(metadata_path.resolve())
        public_results.append(public_result)
    manifest = {
        "object": args.object,
        "video_fps": VIDEO_FPS,
        "video_resolution": [VIDEO_RESOLUTION[0] * 2, VIDEO_RESOLUTION[1]],
        "plot_limits": limits,
        "scenarios": public_results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
