#!/usr/bin/env python3
"""Evaluate online place detection and fixed-vs-wobbling support recognition.

The fixed and wobbling conditions use paired seeds, identical Cartesian
commands, object properties, support geometry, and placement points.  Every
trial saves a raw video, a synchronized score video, a full trace CSV, and
metadata.  A second mode searches points on one wobbling tile and releases
according to a configurable model or baseline policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import torch


ROOT = Path(os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
WORKSPACE = Path(os.environ.get("FT_WORKSPACE_ROOT", ROOT.parent)).expanduser().resolve()
PATCHTST_ROOT = Path(os.environ.get("FT_PATCHTST_ROOT", WORKSPACE / "PatchTST")).expanduser().resolve()
PATCHTST_CODE = PATCHTST_ROOT / "PatchTST_self_supervised"
EVALUATION_CODE = ROOT / "src/evaluation"
if str(EVALUATION_CODE) not in sys.path:
    sys.path.insert(0, str(EVALUATION_CODE))
# Load this repository's evaluation module by its direct module name.  It adds
# PatchTST_self_supervised to sys.path so its own ``src.models`` imports resolve
# to the model repository, as they do when lift_and_lower_object.py is executed.
import lift_and_lower_object as lift_eval  # noqa: E402


DT = 0.01
RENDER_EVERY = 10
VIDEO_FPS = int(round(1.0 / (DT * RENDER_EVERY)))
CONTEXT = 80
CHECKPOINT = (
    PATCHTST_ROOT / "saved_models/terrain_contrastive"
    / "terrain_floor_wobble_contact_clean_20260822_234033"
    / "terrain_floor_wobble_contact_clean_tokenfusion_seed42"
    / "terrain_floor_wobble_contact_clean_tokenfusion_seed42_best.pth"
)
ACTION_PROTOTYPES = (
    PATCHTST_ROOT / "results/terrain_contrastive"
    / "terrain_floor_wobble_contact_clean_20260822_234033"
    / "prepared_eval_templates_release_merged/bert_emb/action"
)
SUPPORT_PROTOTYPES = (
    PATCHTST_ROOT / "results/wobble_place_support_eval08142026"
    / "prepared/bert_emb/support_condition"
)
FORCE_COLUMNS = lift_eval.EXPECTED_FORCE_COLS
DECK_SIZE = (0.240, 0.210, 0.016)
PIVOT_Z = 0.074
DECK_TOP_Z = PIVOT_Z + 0.5 * DECK_SIZE[2]
SOURCE_TOP_Z = 0.020
TILE_CENTER = np.asarray((0.48, 0.30), dtype=float)
SOURCE_XY = np.asarray((0.62, 0.0), dtype=float)
FRANKA_KP = np.asarray([4500, 4500, 3500, 3500, 2000, 2000, 2000, 1000, 1000])
FRANKA_KV = np.asarray([450, 450, 350, 350, 200, 200, 200, 100, 100])
FRANKA_FORCE_LOWER = np.asarray([-87, -87, -87, -87, -12, -12, -12, -10, -10])
FRANKA_FORCE_UPPER = np.asarray([87, 87, 87, 87, 12, 12, 12, 10, 10])


@dataclass(frozen=True)
class TrialPlan:
    trial_index: int
    seed: int
    offset_x_m: float
    offset_y_m: float
    offset_radius_m: float
    hinge_yaw_deg: float
    stiffness_nm_rad: float
    damping_nms_rad: float
    limit_deg: float
    deck_mass_kg: float
    post_contact_descent_m: float
    post_contact_hold_steps: int


def normalize_vector(value) -> torch.Tensor:
    tensor = torch.load(value, map_location="cpu")
    tensor = tensor.float().reshape(-1)
    return tensor / (tensor.norm() + 1e-8)


def load_prototypes(flat_embedding: Path | None = None, wobble_embedding: Path | None = None):
    flat_embedding = flat_embedding or (SUPPORT_PROTOTYPES / "0.pt")
    wobble_embedding = wobble_embedding or (SUPPORT_PROTOTYPES / "1.pt")
    return {
        "hold": normalize_vector(ACTION_PROTOTYPES / "1.pt"),
        "place": normalize_vector(ACTION_PROTOTYPES / "3.pt"),
        "flat surface": normalize_vector(flat_embedding),
        "wobbling support": normalize_vector(wobble_embedding),
    }


def make_trial_plan(index: int, seed: int, radius_range=(0.060, 0.070), support_profile="standard") -> TrialPlan:
    rng = random.Random(seed + index * 1009)
    radius = rng.uniform(*radius_range)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    offset = radius * np.asarray((math.cos(theta), math.sin(theta)))
    # Local +y of the hinge points toward the sampled placement offset.
    hinge_yaw = math.degrees(theta - 0.5 * math.pi)
    if support_profile == "standard":
        stiffness = rng.uniform(5.0, 8.0)
        damping = rng.uniform(0.22, 0.35)
        limit = rng.uniform(9.0, 14.0)
        deck_mass = rng.uniform(0.45, 0.55)
    elif support_profile == "high_wobble":
        stiffness = rng.uniform(1.5, 3.0)
        damping = rng.uniform(0.05, 0.12)
        limit = rng.uniform(20.0, 28.0)
        deck_mass = rng.uniform(0.35, 0.48)
    elif support_profile == "challenging":
        stiffness = rng.uniform(0.80, 1.50)
        damping = rng.uniform(0.04, 0.09)
        limit = rng.uniform(26.0, 32.0)
        deck_mass = rng.uniform(0.33, 0.45)
    elif support_profile == "near_tip":
        # Deliberately difficult stress test based on the near-tipping demo.
        # The hinge remains restoring and damped, but off-axis placement can
        # rotate the deck far enough to topple a slender object after release.
        stiffness = rng.uniform(0.35, 0.80)
        damping = rng.uniform(0.02, 0.06)
        limit = rng.uniform(30.0, 38.0)
        deck_mass = rng.uniform(0.30, 0.42)
    else:
        raise ValueError(f"Unsupported support profile: {support_profile}")
    return TrialPlan(
        trial_index=index,
        seed=seed + index * 1009,
        offset_x_m=float(offset[0]),
        offset_y_m=float(offset[1]),
        offset_radius_m=float(radius),
        hinge_yaw_deg=float(hinge_yaw),
        stiffness_nm_rad=stiffness,
        damping_nms_rad=damping,
        limit_deg=limit,
        deck_mass_kg=deck_mass,
        post_contact_descent_m=rng.uniform(0.007, 0.010),
        post_contact_hold_steps=rng.randint(40, 80),
    )


def support_mjcf(plan: TrialPlan, fixed: bool) -> str:
    cx, cy = TILE_CENTER
    hx, hy, hz = (0.5 * value for value in DECK_SIZE)
    if fixed:
        return f"""
<mujoco model="fixed_support"><worldbody>
  <body name="fixed_deck" pos="{cx} {cy} {PIVOT_Z}">
    <geom name="deck" type="box" size="{hx} {hy} {hz}" mass="{plan.deck_mass_kg}"
          rgba="0.10 0.35 0.90 1" friction="1.2 0.01 0.0001"/>
  </body>
</worldbody></mujoco>
""".strip()
    limit = math.radians(plan.limit_deg)
    yaw = math.radians(plan.hinge_yaw_deg)
    return f"""
<mujoco model="wobble_support"><compiler angle="radian"/><worldbody>
  <geom name="pedestal" type="box" size="0.035 0.035 0.030"
        pos="{cx} {cy} 0.030" rgba="0.22 0.24 0.28 1"
        contype="0" conaffinity="0"/>
  <body name="wobble_deck" pos="{cx} {cy} {PIVOT_Z}" euler="0 0 {yaw}">
    <joint name="wobble_hinge" type="hinge" axis="1 0 0" limited="true"
           range="{-limit} {limit}" stiffness="{plan.stiffness_nm_rad}"
           damping="{plan.damping_nms_rad}" armature="0.012"/>
    <geom name="deck" type="box" size="{hx} {hy} {hz}" mass="{plan.deck_mass_kg}"
          rgba="0.10 0.35 0.90 1" friction="1.2 0.01 0.0001"/>
  </body>
</worldbody></mujoco>
""".strip()


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def touching(entity, other) -> bool:
    contacts = entity.get_contacts(with_entity=other)
    valid = contacts.get("valid_mask")
    return bool(to_numpy(valid).astype(bool).any()) if valid is not None else False


def contact_link_indices(entity) -> set[int]:
    """Return all link indices in contacts involving ``entity``."""
    contacts = entity.get_contacts()
    indices: set[int] = set()
    for key in ("link_a", "link_b"):
        values = contacts.get(key, [])
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        for value in np.asarray(values).reshape(-1).tolist():
            try:
                indices.add(int(value))
            except (TypeError, ValueError):
                pass
    return indices


def ik(franka, hand, pos):
    return franka.inverse_kinematics(
        link=hand,
        pos=np.asarray(pos, dtype=float)[None, :],
        quat=np.asarray((0, 1, 0, 0), dtype=float)[None, :],
    )


def command(franka, motors, fingers, qpos, opening):
    franka.control_dofs_position(qpos[0, :-2], motors)
    franka.control_dofs_position(np.asarray((opening, opening)), fingers)


def force12(franka) -> np.ndarray:
    force = to_numpy(franka.get_links_contact_force([9, 10], sensor=True)).reshape(-1, 2, 3)[0]
    torque = to_numpy(franka.get_links_contact_torque([9, 10], sensor=True)).reshape(-1, 2, 3)[0]
    return np.concatenate((force[0], torque[0], force[1], torque[1])).astype(float)


def score_window(model, prototypes, buffer):
    window = lift_eval.prepare_force_window(buffer, CONTEXT)
    embedding = lift_eval.embed_force_segment(model, None, window)
    return {name: float(torch.dot(embedding, vector)) for name, vector in prototypes.items()}


class OnlineTrace:
    def __init__(
        self,
        scene,
        camera,
        franka,
        hand_idx,
        left_finger_link_idx,
        right_finger_link_idx,
        obj,
        support,
        floor,
        model,
        prototypes,
        fixed,
    ):
        self.scene, self.camera, self.franka = scene, camera, franka
        self.hand_idx, self.obj, self.support = hand_idx, obj, support
        self.left_finger_link_idx = left_finger_link_idx
        self.right_finger_link_idx = right_finger_link_idx
        self.floor = floor
        self.model, self.prototypes, self.fixed = model, prototypes, fixed
        self.phase = "initial_settle"
        self.rows: list[dict] = []
        self.render_steps: list[int] = []
        self.buffer: list[np.ndarray] = []
        self.action_enabled = False
        self.latest_scores = {name: math.nan for name in prototypes}
        self.place_detection_step = None
        self.contact_step = None
        self.contact_streak = 0
        self.place_streak = 0
        self.grasp_reference_relative = None
        self.release_commanded = False

    def reset_detection(self):
        self.place_detection_step = None
        self.contact_step = None
        self.contact_streak = 0
        self.place_streak = 0

    def step(self):
        self.scene.step()
        values = force12(self.franka)
        self.buffer.append(values)
        if len(self.buffer) > CONTEXT:
            self.buffer = self.buffer[-CONTEXT:]
        # These signals are intentionally recorded at every simulation step so
        # that pre-release grasp loss can be audited independently of outcome.
        contact_pairs = contact_link_indices(self.obj)
        left_finger_contact = self.left_finger_link_idx in contact_pairs
        right_finger_contact = self.right_finger_link_idx in contact_pairs
        contact = touching(self.obj, self.support)
        floor_contact = touching(self.obj, self.floor)
        self.contact_streak = self.contact_streak + 1 if contact else 0
        step = len(self.rows)
        if self.contact_step is None and self.contact_streak >= 3:
            self.contact_step = step - 2
        if self.action_enabled and len(self.buffer) >= CONTEXT and step % 5 == 0:
            self.latest_scores = score_window(self.model, self.prototypes, self.buffer)
            is_place = self.latest_scores["place"] > self.latest_scores["hold"]
            self.place_streak = self.place_streak + 1 if is_place else 0
            if self.place_detection_step is None and self.place_streak >= 2:
                self.place_detection_step = step
        angle = 0.0
        if not self.fixed:
            position = to_numpy(self.support.get_dofs_position()).reshape(-1)
            if len(position):
                angle = float(math.degrees(position[0]))
        obj_pos = to_numpy(self.obj.get_root_COM()).reshape(-1, 3)[0]
        hand_pos = to_numpy(self.franka.get_links_pos([self.hand_idx])).reshape(-1, 3)[0]
        relative = obj_pos - hand_pos
        relative_error = (
            math.nan
            if self.grasp_reference_relative is None
            else float(np.linalg.norm(relative - self.grasp_reference_relative))
        )
        row = {
            "step": step, "time_s": step * DT, "phase": self.phase,
            "object_touching_support": contact,
            "object_touching_floor": floor_contact,
            "left_finger_touching_object": left_finger_contact,
            "right_finger_touching_object": right_finger_contact,
            "release_commanded": self.release_commanded,
            "platform_angle_deg": angle,
            "hold_score": self.latest_scores["hold"],
            "place_score": self.latest_scores["place"],
            "flat_surface_score": self.latest_scores["flat surface"],
            "wobbling_support_score": self.latest_scores["wobbling support"],
            "object_x": float(obj_pos[0]), "object_y": float(obj_pos[1]), "object_z": float(obj_pos[2]),
            "hand_x": float(hand_pos[0]), "hand_y": float(hand_pos[1]), "hand_z": float(hand_pos[2]),
            "object_hand_relative_x": float(relative[0]),
            "object_hand_relative_y": float(relative[1]),
            "object_hand_relative_z": float(relative[2]),
            "object_hand_relative_error_m": relative_error,
        }
        row.update({name: float(value) for name, value in zip(FORCE_COLUMNS, values)})
        self.rows.append(row)
        if step % RENDER_EVERY == 0:
            self.camera.render(rgb=True)
            self.render_steps.append(step)


def move(trace, franka, hand, motors, fingers, target, opening, steps, stop_on_place=False):
    start = to_numpy(franka.get_links_pos([8])).reshape(-1, 3)[0].astype(float)
    target = np.asarray(target, dtype=float)
    for index in range(max(1, int(steps))):
        alpha = (index + 1) / max(1, int(steps))
        command(franka, motors, fingers, ik(franka, hand, (1 - alpha) * start + alpha * target), opening)
        trace.step()
        if stop_on_place and trace.place_detection_step is not None:
            return True
    return False


def hold(trace, franka, hand, motors, fingers, pos, opening, steps, stop_on_place=False):
    qpos = ik(franka, hand, pos)
    for _ in range(int(steps)):
        command(franka, motors, fingers, qpos, opening)
        trace.step()
        if stop_on_place and trace.place_detection_step is not None:
            return True
    return False


def gripper(trace, franka, motors, fingers, start, end, steps):
    arm = to_numpy(franka.get_dofs_position(motors)).reshape(-1)
    for index in range(int(steps)):
        alpha = (index + 1) / max(1, int(steps))
        franka.control_dofs_position(arm, motors)
        franka.control_dofs_position(np.asarray(((1-alpha)*start+alpha*end,) * 2), fingers)
        trace.step()


def save_rows(rows, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def polyline(panel, values, x0, y0, width, height, low, high, color):
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2 or high <= low:
        return
    xs = x0 + valid / max(1, len(values)-1) * width
    ys = y0 + height - np.clip((values[valid]-low)/(high-low), 0, 1)*height
    cv2.polylines(panel, [np.column_stack((xs, ys)).astype(np.int32)], False, color, 2, cv2.LINE_AA)


def synchronized_video(raw_path, output_path, rows, render_steps, title, tile_overlay=None):
    cap = cv2.VideoCapture(str(raw_path)); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=float(cap.get(cv2.CAP_PROP_FPS)) or VIDEO_FPS
    temp=output_path.with_suffix(".avi")
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*"MJPG"),fps,(2*width,height))
    series = {
        "hold": np.asarray([r["hold_score"] for r in rows]),
        "place": np.asarray([r["place_score"] for r in rows]),
        "flat": np.asarray([r["flat_surface_score"] for r in rows]),
        "wobble": np.asarray([r["wobbling_support_score"] for r in rows]),
        "angle": np.asarray([r["platform_angle_deg"] for r in rows]),
    }
    # Graph curves and axes are identical in every video frame.  Drawing them
    # once avoids quadratic work for long, open-ended searches.
    base_panel=np.full((height,width,3),248,np.uint8)
    cv2.putText(base_panel,title,(20,30),cv2.FONT_HERSHEY_SIMPLEX,.68,(25,25,25),2,cv2.LINE_AA)
    graphs=(("Action cosine",series["hold"],series["place"],"hold","place"),("Support cosine",series["flat"],series["wobble"],"flat","wobble"),("Support angle [deg]",series["angle"],None,"angle",""))
    margin=72; graph_w=width-(390 if tile_overlay else 100); graph_h=max(85,(height-145)//3)
    graph_specs=[]
    for gi,(label,a,b,la,lb) in enumerate(graphs):
        gy=68+gi*(graph_h+22); finite=np.r_[a[np.isfinite(a)], b[np.isfinite(b)] if b is not None else []]
        if gi<2 and len(finite): low=float(finite.min()-.03); high=float(finite.max()+.03)
        elif gi==2: high=max(2.,float(np.max(np.abs(a)))*1.1); low=-high
        else: low,high=-.2,.2
        cv2.rectangle(base_panel,(margin,gy),(margin+graph_w,gy+graph_h),(210,210,210),1)
        cv2.putText(base_panel,label,(margin,gy-6),cv2.FONT_HERSHEY_SIMPLEX,.42,(45,45,45),1)
        cv2.putText(base_panel,la,(margin+graph_w-145,gy-6),cv2.FONT_HERSHEY_SIMPLEX,.38,(220,85,40),1)
        if lb: cv2.putText(base_panel,lb,(margin+graph_w-72,gy-6),cv2.FONT_HERSHEY_SIMPLEX,.38,(45,145,45),1)
        polyline(base_panel,a,margin,gy,graph_w,graph_h,low,high,(220,85,40))
        if b is not None: polyline(base_panel,b,margin,gy,graph_w,graph_h,low,high,(45,145,45))
        graph_specs.append((gy,graph_h))
    frame_idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        panel=base_panel.copy(); step=render_steps[min(frame_idx,len(render_steps)-1)]
        cv2.putText(panel,f"t={step*DT:.2f}s  {rows[step]['phase']}",(20,54),cv2.FONT_HERSHEY_SIMPLEX,.46,(55,55,55),1,cv2.LINE_AA)
        for gy,current_graph_h in graph_specs:
            cursor=margin+int(graph_w*step/max(1,len(rows)-1)); cv2.line(panel,(cursor,gy),(cursor,gy+current_graph_h),(30,30,220),2)
        if tile_overlay:
            map_size=min(270,height-130); map_x=width-map_size-24; map_y=92
            cv2.rectangle(panel,(map_x,map_y),(map_x+map_size,map_y+map_size),(225,225,225),-1)
            cv2.rectangle(panel,(map_x,map_y),(map_x+map_size,map_y+map_size),(90,90,90),1)
            cv2.putText(panel,"Tile top view",(map_x,map_y-9),cv2.FONT_HERSHEY_SIMPLEX,.46,(35,35,35),1,cv2.LINE_AA)
            span=.145
            def map_xy(xy):
                rel=(np.asarray(xy,dtype=float)-TILE_CENTER)/span
                return (int(map_x+map_size*(.5+.5*rel[0])),int(map_y+map_size*(.5-.5*rel[1])))
            yaw=math.radians(tile_overlay["hinge_yaw_deg"]); c,s=math.cos(yaw),math.sin(yaw)
            rot=np.asarray(((c,-s),(s,c)))
            corners=np.asarray(((-DECK_SIZE[0]/2,-DECK_SIZE[1]/2),(DECK_SIZE[0]/2,-DECK_SIZE[1]/2),(DECK_SIZE[0]/2,DECK_SIZE[1]/2),(-DECK_SIZE[0]/2,DECK_SIZE[1]/2))) @ rot.T + TILE_CENTER
            cv2.polylines(panel,[np.asarray([map_xy(xy) for xy in corners],np.int32)],True,(120,75,20),2,cv2.LINE_AA)
            phase=str(rows[step]["phase"])
            for point in tile_overlay["points"]:
                xy=TILE_CENTER+np.asarray(point["offset_xy"]); px,py=map_xy(xy)
                measured=point.get("observation_end_step",10**9)<=step
                selected=tile_overlay.get("selected_point_index")==point["index"] and phase.startswith("selected_point")
                color=(40,180,40) if selected else (170,90,30) if measured else (130,130,130)
                cv2.circle(panel,(px,py),7,color,-1,cv2.LINE_AA)
                cv2.putText(panel,str(point["index"]),(px+8,py-7),cv2.FONT_HERSHEY_SIMPLEX,.40,(25,25,25),1,cv2.LINE_AA)
                if measured:
                    cv2.putText(panel,f"{point['scores']['flat surface']:.3f}",(px+8,py+10),cv2.FONT_HERSHEY_SIMPLEX,.30,(45,45,45),1,cv2.LINE_AA)
            hand_xy=(float(rows[step]["hand_x"]),float(rows[step]["hand_y"])); hx,hy=map_xy(hand_xy)
            cv2.drawMarker(panel,(hx,hy),(20,20,220),cv2.MARKER_CROSS,14,2,cv2.LINE_AA)
            cv2.putText(panel,"x: hand   number: fixed cosine",(map_x,map_y+map_size+20),cv2.FONT_HERSHEY_SIMPLEX,.34,(45,45,45),1,cv2.LINE_AA)
        writer.write(np.hstack((frame,panel))); frame_idx+=1
    cap.release(); writer.release()
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(temp),"-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p",str(output_path)],check=True)
    temp.unlink()


def run_trial(
    condition,
    plan,
    model,
    prototypes,
    output_root,
    release_on_place=True,
    post_place_observation_steps=0,
    post_place_observation_motion="hold",
):
    fixed = condition == "stable"
    trial_dir=output_root/condition/f"trial_{plan.trial_index:03d}"; trial_dir.mkdir(parents=True,exist_ok=True)
    raw=trial_dir/"raw.mp4"; sync=trial_dir/"scores.mp4"; trace_path=trial_dir/"trace.csv"
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu,logging_level="warning")
    cam=None; recording=False
    try:
        scene=gs.Scene(sim_options=gs.options.SimOptions(dt=DT),show_viewer=False)
        floor=scene.add_entity(gs.morphs.Plane(),material=gs.materials.Rigid(coup_friction=0.0))
        scene.add_entity(gs.morphs.Box(size=(.20,.20,SOURCE_TOP_Z),pos=(*SOURCE_XY,SOURCE_TOP_Z/2),fixed=True),material=gs.materials.Rigid(rho=900,friction=1.1),surface=gs.surfaces.Default(color=(.25,.30,.34,1)))
        support=scene.add_entity(gs.morphs.MJCF(file=support_mjcf(plan,fixed)),material=gs.materials.Rigid(friction=1.2))
        cube_size=.05
        obj=scene.add_entity(gs.morphs.Box(size=(cube_size,)*3,pos=(*SOURCE_XY,SOURCE_TOP_Z+cube_size/2+.001)),material=gs.materials.Rigid(rho=700,friction=1.0),surface=gs.surfaces.Default(color=(.93,.67,.22,1)))
        franka=scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),material=gs.materials.Rigid(coup_friction=3.,friction=1.))
        cam=scene.add_camera(model="thinlens",res=(960,540),pos=(1.38,-.72,.62),lookat=(.52,.17,.10),fov=42,GUI=False)
        scene.build(n_envs=1)
        motors,fingers=np.arange(7),np.arange(7,9); hand=franka.get_link("hand")
        hand_idx=next(i for i,l in enumerate(franka.links) if l.name=="hand"); left_idx=next(i for i,l in enumerate(franka.links) if l.name=="left_finger"); right_idx=next(i for i,l in enumerate(franka.links) if l.name=="right_finger")
        franka.set_dofs_kp(FRANKA_KP); franka.set_dofs_kv(FRANKA_KV)
        franka.set_dofs_force_range(FRANKA_FORCE_LOWER, FRANKA_FORCE_UPPER)
        home=ik(franka,hand,(.45,0,.30)); home[0,-2:]=.04; franka.set_dofs_position(home[0,:-2],motors); franka.set_dofs_position(home[0,-2:],fingers)
        trace=OnlineTrace(scene,cam,franka,hand_idx,int(franka.links[left_idx].idx),int(franka.links[right_idx].idx),obj,support,floor,model,prototypes,fixed)
        cam.start_recording(save_to_filename=str(raw),fps=VIDEO_FPS); recording=True
        hold(trace,franka,hand,motors,fingers,(.45,0,.30),.04,60)
        bounds=to_numpy(obj.get_AABB()).reshape(-1,3); upper=bounds[1]
        finger_len=lift_eval.infer_finger_length_from_mjcf(lift_eval.PANDA_XML_PATH)
        links=to_numpy(franka.get_links_pos([hand_idx,left_idx])).reshape(-1,3); h2r=float(links[1,2]-links[0,2]); h2t=h2r+(-1 if h2r<0 else 1)*finger_len
        grasp_z=max(float(upper[2]-.008-h2t),float(upper[2]+.008-h2r)); hover=max(.24,grasp_z+.12); lift_z=max(.30,grasp_z+.18)
        trace.phase="approach_object"; move(trace,franka,hand,motors,fingers,(*SOURCE_XY,hover),.04,100)
        trace.phase="descend_to_grasp"; move(trace,franka,hand,motors,fingers,(*SOURCE_XY,grasp_z),.04,80)
        trace.phase="close_gripper"; gripper(trace,franka,motors,fingers,.04,0.,60)
        trace.phase="lift_and_transport"; move(trace,franka,hand,motors,fingers,(*SOURCE_XY,lift_z),0.,100)
        target=TILE_CENTER+np.asarray((plan.offset_x_m,plan.offset_y_m)); move(trace,franka,hand,motors,fingers,(*target,lift_z),0.,130)
        nominal=grasp_z+(DECK_TOP_Z-SOURCE_TOP_Z); pre=nominal+.055
        trace.phase="pre_descent"; move(trace,franka,hand,motors,fingers,(*target,pre),0.,80)
        trace.phase="online_descent_30mm_s"; trace.action_enabled=True
        descent_target=nominal-plan.post_contact_descent_m
        steps=int(math.ceil((pre-descent_target)/(.03*DT)))
        detected=move(trace,franka,hand,motors,fingers,(*target,descent_target),0.,steps,stop_on_place=True)
        if not detected:
            trace.phase="post_descent_detection_wait"
            detected=hold(trace,franka,hand,motors,fingers,(*target,descent_target),0.,80,stop_on_place=True)
        detection_scores=dict(trace.latest_scores)
        observation_end_step = trace.place_detection_step
        if detected and post_place_observation_steps > 0:
            observed_pos = to_numpy(franka.get_links_pos([8])).reshape(-1, 3)[0]
            if post_place_observation_motion == "descend":
                descent_distance = 0.03 * DT * post_place_observation_steps
                observation_target = observed_pos.copy()
                observation_target[2] -= descent_distance
                trace.phase=(
                    "post_place_grasped_descent_30mm_s_"
                    f"{post_place_observation_steps * DT:.2f}s"
                )
                move(
                    trace,
                    franka,
                    hand,
                    motors,
                    fingers,
                    observation_target,
                    0.,
                    post_place_observation_steps,
                )
            else:
                # Keep both the Cartesian pose and gripper command fixed.
                trace.phase=f"post_place_grasped_hold_{post_place_observation_steps * DT:.2f}s"
                hold(
                    trace,
                    franka,
                    hand,
                    motors,
                    fingers,
                    observed_pos,
                    0.,
                    post_place_observation_steps,
                )
            observation_end_step = len(trace.rows) - 1
        observation_scores=dict(trace.latest_scores)
        if detected and release_on_place:
            trace.phase="model_triggered_release"; gripper(trace,franka,motors,fingers,0.,.04,50)
        else:
            trace.phase="no_release_cleanup"
        trace.action_enabled=False
        retreat_pos=(*target,max(pre,.25)); move(trace,franka,hand,motors,fingers,retreat_pos,.04 if detected else 0.,100)
        trace.phase="outcome_settle"; hold(trace,franka,hand,motors,fingers,retreat_pos,.04 if detected else 0.,120)
        cam.stop_recording(); recording=False
        save_rows(trace.rows,trace_path); synchronized_video(raw,sync,trace.rows,trace.render_steps,f"{condition} trial {plan.trial_index:03d}")
        final_obj=to_numpy(obj.get_root_COM()).reshape(-1,3)[0]
        delay=None if trace.contact_step is None or trace.place_detection_step is None else trace.place_detection_step-trace.contact_step
        pred="flat surface" if observation_scores["flat surface"]>=observation_scores["wobbling support"] else "wobbling support"
        detection_pred="flat surface" if detection_scores["flat surface"]>=detection_scores["wobbling support"] else "wobbling support"
        result={"condition":condition,"plan":asdict(plan),"contact_step":trace.contact_step,"place_detection_step":trace.place_detection_step,"detection_delay_steps":delay,"place_detected":trace.place_detection_step is not None,"early_place_detection":bool(delay is not None and delay<0),"post_place_observation_steps":post_place_observation_steps,"post_place_observation_s":post_place_observation_steps*DT,"post_place_observation_motion":post_place_observation_motion,"post_place_descent_distance_m":0.03*DT*post_place_observation_steps if post_place_observation_motion=="descend" else 0.0,"observation_end_step":observation_end_step,"support_prediction_at_detection":detection_pred,"scores_at_detection":detection_scores,"support_prediction":pred,"support_correct":pred==("flat surface" if fixed else "wobbling support"),"scores":observation_scores,"flat_minus_wobble":observation_scores["flat surface"]-observation_scores["wobbling support"],"final_object_xyz":final_obj.tolist(),"object_above_floor_after_release":bool(final_obj[2]>.04),"raw_video":str(raw),"score_video":str(sync),"trace_csv":str(trace_path)}
        (trial_dir/"metadata.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        return result
    finally:
        if recording and cam is not None: cam.stop_recording()
        gs.destroy()


def summarize(results):
    total=len(results); detected=[r for r in results if r["place_detected"]]; valid_delay=[r["detection_delay_steps"] for r in detected if r["detection_delay_steps"] is not None]
    return {"trials":total,"place_detection_rate":len(detected)/total if total else 0.,"early_detection_rate":sum(r["early_place_detection"] for r in results)/total if total else 0.,"median_detection_delay_steps":float(np.median(valid_delay)) if valid_delay else None,"mean_detection_delay_steps":float(np.mean(valid_delay)) if valid_delay else None,"support_accuracy_all":sum(r["support_correct"] for r in results)/total if total else 0.,"support_accuracy_detected":sum(r["support_correct"] for r in detected)/len(detected) if detected else None,"mean_flat_minus_wobble":float(np.mean([r["flat_minus_wobble"] for r in results])) if results else None}


def search_points(plan: TrialPlan, count=5, pattern="random"):
    if pattern == "center":
        return [{
            "index": 1,
            "local_xy": [0.0, 0.0],
            "offset_xy": [0.0, 0.0],
            "radius_m": 0.0,
            "hinge_moment_arm_m": 0.0,
        }]
    if pattern != "random":
        raise ValueError(f"Unsupported search point pattern: {pattern}")
    rng = random.Random(plan.seed + 700_001)
    yaw = math.radians(plan.hinge_yaw_deg)
    rotation = np.asarray(((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))))
    points = []
    # Leave enough edge clearance for the 50 mm cube while sampling across
    # the usable tile top, rather than restricting probes to a narrow band.
    while len(points) < count:
        local = np.asarray((rng.uniform(-0.080, 0.080), rng.uniform(-0.070, 0.070)))
        if np.linalg.norm(local) < 0.008:
            continue
        world = rotation @ local
        points.append({
            "index": len(points) + 1,
            "local_xy": local.tolist(),
            "offset_xy": world.tolist(),
            "radius_m": float(np.linalg.norm(world)),
            "hinge_moment_arm_m": float(abs(local[1])),
        })
    return points


def quaternion_tilt_change_deg(reference, current):
    ref_euler=to_numpy(gs.utils.geom.quat_to_xyz(np.asarray(reference),rpy=True,degrees=True)).reshape(-1)
    current_euler=to_numpy(gs.utils.geom.quat_to_xyz(np.asarray(current),rpy=True,degrees=True)).reshape(-1)
    delta=(current_euler-ref_euler+180.)%360.-180.
    return float(np.max(np.abs(delta[:2])))


def quaternion_up_axis(quaternion_wxyz):
    """Return the world-frame direction of a body's local +z axis."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    return np.asarray(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    )


def relative_upright_tilt_deg(object_quaternion, support_quaternion):
    object_up = quaternion_up_axis(object_quaternion)
    support_up = quaternion_up_axis(support_quaternion)
    cosine = float(
        np.dot(object_up, support_up)
        / ((np.linalg.norm(object_up) * np.linalg.norm(support_up)) + 1e-12)
    )
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


def run_search_trial(
    plan,
    model,
    prototypes,
    output_root,
    point_count=5,
    observation_steps=40,
    point_pattern="random",
    selection_policy="max_fixed",
    object_profile="cube",
    random_accept_probability=0.5,
    matched_probe_count=None,
):
    if selection_policy == "matched_random" and (
        matched_probe_count is None or matched_probe_count <= 0
    ):
        raise ValueError(
            "matched_random requires a positive matched_probe_count"
        )
    trial_dir=output_root/f"trial_{plan.trial_index:03d}"; trial_dir.mkdir(parents=True,exist_ok=True)
    raw=trial_dir/"raw.mp4"; sync=trial_dir/"scores.mp4"; trace_path=trial_dir/"trace.csv"
    points=search_points(plan,point_count,pattern=point_pattern)
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu,logging_level="warning")
    cam=None; recording=False
    try:
        scene=gs.Scene(sim_options=gs.options.SimOptions(dt=DT),show_viewer=False)
        floor=scene.add_entity(gs.morphs.Plane(),material=gs.materials.Rigid(coup_friction=0.0))
        scene.add_entity(gs.morphs.Box(size=(.20,.20,SOURCE_TOP_Z),pos=(*SOURCE_XY,SOURCE_TOP_Z/2),fixed=True),material=gs.materials.Rigid(rho=900,friction=1.1),surface=gs.surfaces.Default(color=(.25,.30,.34,1)))
        support=scene.add_entity(gs.morphs.MJCF(file=support_mjcf(plan,False)),material=gs.materials.Rigid(friction=1.2))
        object_sizes = {
            "cube": (.05, .05, .05),
            "tall_prism": (.04, .04, .12),
            "tall_prism_hard": (.035, .035, .16),
            "tall_prism_160": (.04, .04, .16),
            "slender_prism": (.03, .03, .12),
            "slender_prism_heavy": (.03, .03, .12),
            "narrow_prism": (.025, .025, .11),
        }
        object_size = object_sizes[object_profile]
        object_friction = (
            1.5
            if object_profile in ("tall_prism_160", "slender_prism", "slender_prism_heavy", "narrow_prism")
            else 1.0
        )
        object_density = {
            "narrow_prism": 1100,
            "slender_prism_heavy": 900,
        }.get(object_profile, 700)
        obj=scene.add_entity(gs.morphs.Box(size=object_size,pos=(*SOURCE_XY,SOURCE_TOP_Z+object_size[2]/2+.001)),material=gs.materials.Rigid(rho=object_density,friction=object_friction),surface=gs.surfaces.Default(color=(.93,.67,.22,1)))
        franka=scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),material=gs.materials.Rigid(coup_friction=3.,friction=1.))
        cam=scene.add_camera(model="thinlens",res=(960,540),pos=(1.38,-.72,.62),lookat=(.52,.17,.10),fov=42,GUI=False)
        scene.build(n_envs=1)
        motors,fingers=np.arange(7),np.arange(7,9); hand=franka.get_link("hand")
        hand_idx=next(i for i,l in enumerate(franka.links) if l.name=="hand"); left_idx=next(i for i,l in enumerate(franka.links) if l.name=="left_finger"); right_idx=next(i for i,l in enumerate(franka.links) if l.name=="right_finger")
        franka.set_dofs_kp(FRANKA_KP); franka.set_dofs_kv(FRANKA_KV)
        franka.set_dofs_force_range(FRANKA_FORCE_LOWER, FRANKA_FORCE_UPPER)
        home=ik(franka,hand,(.45,0,.30)); home[0,-2:]=.04; franka.set_dofs_position(home[0,:-2],motors); franka.set_dofs_position(home[0,-2:],fingers)
        trace=OnlineTrace(scene,cam,franka,hand_idx,int(franka.links[left_idx].idx),int(franka.links[right_idx].idx),obj,support,floor,model,prototypes,False)
        cam.start_recording(save_to_filename=str(raw),fps=VIDEO_FPS); recording=True
        hold(trace,franka,hand,motors,fingers,(.45,0,.30),.04,60)
        bounds=to_numpy(obj.get_AABB()).reshape(-1,3); upper=bounds[1]
        finger_len=lift_eval.infer_finger_length_from_mjcf(lift_eval.PANDA_XML_PATH)
        links=to_numpy(franka.get_links_pos([hand_idx,left_idx])).reshape(-1,3); h2r=float(links[1,2]-links[0,2]); h2t=h2r+(-1 if h2r<0 else 1)*finger_len
        grasp_z=max(float(upper[2]-.008-h2t),float(upper[2]+.008-h2r)); hover=max(.24,grasp_z+.12); lift_z=max(.30,grasp_z+.18)
        trace.phase="approach_object"; move(trace,franka,hand,motors,fingers,(*SOURCE_XY,hover),.04,100)
        trace.phase="descend_to_grasp"; move(trace,franka,hand,motors,fingers,(*SOURCE_XY,grasp_z),.04,80)
        trace.phase="close_gripper"; gripper(trace,franka,motors,fingers,.04,0.,60)
        trace.phase="lift"; move(trace,franka,hand,motors,fingers,(*SOURCE_XY,lift_z),0.,100)
        grasp_reference_relative=(
            to_numpy(obj.get_root_COM()).reshape(-1,3)[0]
            - to_numpy(franka.get_links_pos([hand_idx])).reshape(-1,3)[0]
        )
        trace.grasp_reference_relative = grasp_reference_relative.copy()
        nominal=grasp_z+(DECK_TOP_Z-SOURCE_TOP_Z); pre=nominal+.060; descent_target=nominal-.010
        probe_results=[]; selected=None; invalid_reason=None
        policy_rng=random.Random(plan.seed+900_003)
        for point in points:
            target=TILE_CENTER+np.asarray(point["offset_xy"])
            trace.phase=f"probe_{point['index']}_approach"; move(trace,franka,hand,motors,fingers,(*target,pre),0.,100)
            trace.reset_detection(); trace.action_enabled=True; trace.phase=f"probe_{point['index']}_online_descent"
            steps=int(math.ceil((pre-descent_target)/(.03*DT)))
            detected=move(trace,franka,hand,motors,fingers,(*target,descent_target),0.,steps,stop_on_place=True)
            if not detected:
                detected=hold(trace,franka,hand,motors,fingers,(*target,descent_target),0.,80,stop_on_place=True)
            policy_observation_steps=0 if selection_policy in ("one_shot_random","center") else observation_steps
            if detected and policy_observation_steps:
                current=to_numpy(franka.get_links_pos([hand_idx])).reshape(-1,3)[0]
                trace.phase=f"probe_{point['index']}_grasped_descent_{policy_observation_steps*DT:.2f}s"
                move(trace,franka,hand,motors,fingers,(*target,float(current[2])-.03*DT*policy_observation_steps),0.,policy_observation_steps)
            scores=dict(trace.latest_scores); contact_step=trace.contact_step; detection_step=trace.place_detection_step
            margin=scores["flat surface"]-scores["wobbling support"]
            accepted=(
                selection_policy in ("one_shot_random","center")
                or (selection_policy=="first_margin_positive" and margin>0.)
                or (selection_policy=="random_accept" and policy_rng.random()<random_accept_probability)
            )
            point.update({"place_detected":detected,"contact_step":contact_step,"place_detection_step":detection_step,"detection_delay_steps":None if contact_step is None or detection_step is None else detection_step-contact_step,"observation_end_step":len(trace.rows)-1,"observation_steps_after_place_detection":policy_observation_steps,"scores":scores,"flat_minus_wobble":margin,"accepted":bool(detected and accepted)})
            print(
                f"  probe {point['index']}/{point_count}: "
                f"margin={margin:+.3f} accepted={bool(detected and accepted)}",
                flush=True,
            )
            probe_results.append(dict(point)); trace.action_enabled=False
            trace.phase=f"probe_{point['index']}_ascend"; move(trace,franka,hand,motors,fingers,(*target,pre),0.,80); hold(trace,franka,hand,motors,fingers,(*target,pre),0.,30)
            grasp_relative=(
                to_numpy(obj.get_root_COM()).reshape(-1,3)[0]
                - to_numpy(franka.get_links_pos([hand_idx])).reshape(-1,3)[0]
            )
            grasp_relative_error=float(
                np.linalg.norm(grasp_relative-grasp_reference_relative)
            )
            object_world_tilt=float(
                math.degrees(
                    math.acos(
                        np.clip(
                            quaternion_up_axis(
                                to_numpy(obj.get_quat()).reshape(-1,4)[0]
                            )[2],
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            probe_results[-1].update(
                {
                    "post_probe_grasp_relative_error_m": grasp_relative_error,
                    "post_probe_object_world_tilt_deg": object_world_tilt,
                }
            )
            # Pre-release exclusion is strictly a grasp-loss check.  A large
            # object tilt while it is still held is not itself a dropped trial;
            # post-release tilt is evaluated separately as the task outcome.
            if grasp_relative_error > 0.060:
                invalid_reason=(
                    f"grasp_lost_after_probe_{point['index']}"
                )
                break
            if detected and accepted and selection_policy!="max_fixed":
                selected=dict(point)
                break
            if (
                selection_policy == "matched_random"
                and len(probe_results) >= matched_probe_count
            ):
                break
        detected_points=[p for p in probe_results if p["place_detected"]]
        if invalid_reason is not None:
            selected=None
        elif selection_policy=="max_fixed":
            selected=max(detected_points,key=lambda p:p["scores"]["flat surface"]) if detected_points else None
        elif selection_policy=="matched_random" and detected_points:
            selected=dict(policy_rng.choice(detected_points))
        if selected is None:
            # Strict abstention: retain the object and count the policy trial as
            # unsuccessful instead of silently falling back to an unsafe point.
            trace.action_enabled=False; trace.phase="selection_abstained"
            retreat_pos=(*SOURCE_XY,max(pre,.25)); move(trace,franka,hand,motors,fingers,retreat_pos,0.,100)
            trace.phase="abstention_settle"; hold(trace,franka,hand,motors,fingers,retreat_pos,0.,120)
            cam.stop_recording(); recording=False; save_rows(trace.rows,trace_path)
            final_obj=to_numpy(obj.get_root_COM()).reshape(-1,3)[0]
            result={"plan":asdict(plan),"object_profile":object_profile,"object_size_m":list(object_size),"selection_policy":selection_policy,"random_accept_probability":random_accept_probability,"matched_probe_count":matched_probe_count,"candidate_budget":point_count,"points":probe_results,"trial_valid":invalid_reason is None,"invalid_before_release":invalid_reason is not None,"invalid_reason":invalid_reason,"release_commanded":False,"selection_abstained":invalid_reason is None,"selection_score":"fixed minus wobble > 0" if selection_policy=="first_margin_positive" else selection_policy,"observation_reference":"place detection","observation_steps":observation_steps,"observation_s":observation_steps*DT,"selected_point_index":None,"selected_center_distance_rank":None,"selected_is_closest_to_center":False,"selected_radius_m":None,"selected_hinge_distance_rank":None,"selected_is_closest_to_hinge":False,"selected_hinge_moment_arm_m":None,"final_place_detected":False,"final_scores":{},"post_release_tilt_change_deg":None,"topple_threshold_deg":35.,"final_toppled":False,"final_object_xyz":final_obj.tolist(),"final_object_retained":False,"final_stable":False,"raw_video":str(raw),"score_video":str(sync),"trace_csv":str(trace_path)}
            synchronized_video(raw,sync,trace.rows,trace.render_steps,f"{selection_policy} {plan.trial_index:03d}",tile_overlay={"points":probe_results,"selected_point_index":None,"hinge_yaw_deg":plan.hinge_yaw_deg})
            (trial_dir/"metadata.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
        selected_target=TILE_CENTER+np.asarray(selected["offset_xy"])
        trace.phase="selected_point_approach"; move(trace,franka,hand,motors,fingers,(*selected_target,pre),0.,100)
        trace.reset_detection(); trace.action_enabled=True; trace.phase="selected_point_final_descent"
        steps=int(math.ceil((pre-descent_target)/(.03*DT))); final_detected=move(trace,franka,hand,motors,fingers,(*selected_target,descent_target),0.,steps,stop_on_place=True)
        if not final_detected: final_detected=hold(trace,franka,hand,motors,fingers,(*selected_target,descent_target),0.,80,stop_on_place=True)
        final_scores=dict(trace.latest_scores)
        release_quat=to_numpy(obj.get_quat()).reshape(-1,4)[0].astype(float)
        release_support_quat=to_numpy(support.get_quat()).reshape(-1,4)[0].astype(float)
        release_relative_tilt=relative_upright_tilt_deg(
            release_quat, release_support_quat
        )
        pre_release_relative=(
            to_numpy(obj.get_root_COM()).reshape(-1,3)[0]
            - to_numpy(franka.get_links_pos([hand_idx])).reshape(-1,3)[0]
        )
        pre_release_grasp_error=float(
            np.linalg.norm(pre_release_relative-grasp_reference_relative)
        )
        # A tilted object can still be firmly held while the support rocks.
        # That is precisely an unsafe-placement case and must be released and
        # scored, not excluded as a pre-release drop.  Exclusion is based on
        # loss of the calibrated object-hand relation only.
        pre_release_valid=bool(pre_release_grasp_error <= 0.060)
        release_commanded=bool(final_detected and pre_release_valid)
        if release_commanded:
            trace.release_commanded = True
            trace.phase="selected_point_release"; gripper(trace,franka,motors,fingers,0.,.04,50)
        trace.action_enabled=False; trace.phase="retreat"; move(trace,franka,hand,motors,fingers,(*selected_target,max(pre,.25)),.04 if release_commanded else 0.,100)
        trace.phase="outcome_settle"; hold(trace,franka,hand,motors,fingers,(*selected_target,max(pre,.25)),.04 if release_commanded else 0.,180)
        cam.stop_recording(); recording=False; save_rows(trace.rows,trace_path)
        final_obj=to_numpy(obj.get_root_COM()).reshape(-1,3)[0]
        final_quat=to_numpy(obj.get_quat()).reshape(-1,4)[0].astype(float)
        final_support_quat=to_numpy(support.get_quat()).reshape(-1,4)[0].astype(float)
        tilt_change=quaternion_tilt_change_deg(release_quat,final_quat)
        final_relative_tilt=relative_upright_tilt_deg(
            final_quat, final_support_quat
        )
        support_up=quaternion_up_axis(final_support_quat)
        height_above_support_pivot=float(
            np.dot(
                final_obj-np.asarray((TILE_CENTER[0],TILE_CENTER[1],PIVOT_Z)),
                support_up,
            )
        )
        retained_on_support=bool(
            touching(obj,support)
            and height_above_support_pivot > 0.45*min(object_size[:2])
        )
        toppled=bool(final_relative_tilt>=35.)
        # Rank against the full pre-sampled candidate budget, including points
        # not visited after an early sequential acceptance.
        ranked=sorted(points,key=lambda p:p["radius_m"]); center_rank=next(i+1 for i,p in enumerate(ranked) if p["index"]==selected["index"])
        hinge_ranked=sorted(points,key=lambda p:p["hinge_moment_arm_m"]); hinge_rank=next(i+1 for i,p in enumerate(hinge_ranked) if p["index"]==selected["index"])
        result={"plan":asdict(plan),"object_profile":object_profile,"object_size_m":list(object_size),"selection_policy":selection_policy,"random_accept_probability":random_accept_probability,"matched_probe_count":matched_probe_count,"candidate_budget":point_count,"points":probe_results,"trial_valid":pre_release_valid,"invalid_before_release":not pre_release_valid,"invalid_reason":None if pre_release_valid else "grasp_lost_before_final_release","pre_release_grasp_relative_error_m":pre_release_grasp_error,"release_commanded":release_commanded,"selection_abstained":False,"selection_score":"fixed minus wobble > 0" if selection_policy=="first_margin_positive" else selection_policy,"observation_reference":"place detection","observation_steps":observation_steps,"observation_s":observation_steps*DT,"selected_point_index":selected["index"],"selected_center_distance_rank":center_rank,"selected_is_closest_to_center":center_rank==1,"selected_radius_m":selected["radius_m"],"selected_hinge_distance_rank":hinge_rank,"selected_is_closest_to_hinge":hinge_rank==1,"selected_hinge_moment_arm_m":selected["hinge_moment_arm_m"],"final_place_detected":final_detected,"final_scores":final_scores,"release_quat_wxyz":release_quat.tolist(),"release_support_quat_wxyz":release_support_quat.tolist(),"release_relative_tilt_deg":release_relative_tilt,"final_quat_wxyz":final_quat.tolist(),"final_support_quat_wxyz":final_support_quat.tolist(),"post_release_tilt_change_deg":tilt_change,"final_relative_tilt_deg":final_relative_tilt,"topple_definition":"object local +z relative to support local +z","topple_threshold_deg":35.,"final_toppled":toppled if release_commanded else False,"final_object_xyz":final_obj.tolist(),"final_height_above_support_pivot_m":height_above_support_pivot,"final_object_retained":retained_on_support if release_commanded else False,"final_stable":bool(pre_release_valid and release_commanded and retained_on_support and not toppled),"raw_video":str(raw),"score_video":str(sync),"trace_csv":str(trace_path)}
        synchronized_video(raw,sync,trace.rows,trace.render_steps,f"{selection_policy} search {plan.trial_index:03d}",tile_overlay={"points":probe_results,"selected_point_index":selected["index"],"hinge_yaw_deg":plan.hinge_yaw_deg})
        (trial_dir/"metadata.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
    finally:
        if recording and cam is not None: cam.stop_recording()
        gs.destroy()


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("paired","search"),default="paired")
    parser.add_argument("--conditions",nargs="+",choices=("stable","wobble"),default=["stable","wobble"])
    parser.add_argument("--trials",type=int,default=30); parser.add_argument("--seed",type=int,default=42000)
    parser.add_argument(
        "--target-valid-trials",
        type=int,
        default=None,
        help=(
            "Continue attempts until this many non-excluded trials are saved; "
            "--trials remains the maximum number of attempted trials."
        ),
    )
    parser.add_argument("--points",type=int,default=5)
    parser.add_argument(
        "--point-pattern",
        choices=("random", "center"),
        default="random",
    )
    parser.add_argument(
        "--selection-policy",
        choices=(
            "max_fixed",
            "first_margin_positive",
            "random_accept",
            "matched_random",
            "one_shot_random",
            "center",
        ),
        default="max_fixed",
    )
    parser.add_argument(
        "--object-profile",
        choices=(
            "cube",
            "tall_prism",
            "tall_prism_hard",
            "tall_prism_160",
            "slender_prism",
            "slender_prism_heavy",
            "narrow_prism",
        ),
        default="cube",
    )
    parser.add_argument("--random-accept-probability",type=float,default=0.5)
    parser.add_argument(
        "--matched-policy-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing trial_*/metadata.json from the model policy. "
            "matched_random probes the same number of points in each paired trial."
        ),
    )
    parser.add_argument(
        "--support-profile",
        choices=("standard", "high_wobble", "challenging", "near_tip"),
        default="standard",
    )
    parser.add_argument("--post-place-observation-s",type=float,default=0.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed trial_*/metadata.json files in the output directory.",
    )
    parser.add_argument(
        "--post-place-observation-motion",
        choices=("hold", "descend"),
        default="hold",
    )
    parser.add_argument("--output-dir",type=Path,default=None); parser.add_argument("--checkpoint",type=Path,default=CHECKPOINT)
    parser.add_argument(
        "--flat-prototype-embedding",
        type=Path,
        default=SUPPORT_PROTOTYPES / "0.pt",
    )
    parser.add_argument(
        "--wobble-prototype-embedding",
        type=Path,
        default=SUPPORT_PROTOTYPES / "1.pt",
    )
    parser.add_argument(
        "--flat-template",
        default="Placing an object on a flat surface.",
    )
    parser.add_argument(
        "--wobble-template",
        default="Placing an object on a wobbling support.",
    )
    return parser.parse_args()


def main():
    args=parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.post_place_observation_s < 0:
        raise ValueError("--post-place-observation-s must be non-negative")
    observation_steps=int(round(args.post_place_observation_s/DT))
    model,_=lift_eval.load_contrastive_patchtst(args.checkpoint,device)
    prototypes=load_prototypes(
        args.flat_prototype_embedding,
        args.wobble_prototype_embedding,
    )
    output=args.output_dir or ROOT/"data/evaluation_outputs"/f"wobble_lift_and_lower_{datetime.now():%Y%m%d_%H%M%S}"; output.mkdir(parents=True,exist_ok=True)
    if args.mode=="search":
        matched_probe_counts = {}
        if args.selection_policy == "matched_random":
            if args.matched_policy_dir is None:
                raise ValueError(
                    "--matched-policy-dir is required for matched_random"
                )
            for metadata_path in sorted(
                args.matched_policy_dir.glob("trial_*/metadata.json")
            ):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                trial_index = int(metadata["plan"]["trial_index"])
                matched_probe_counts[trial_index] = len(metadata["points"])
        results=[]
        for index in range(1,args.trials+1):
            valid_so_far=sum(r.get("trial_valid",True) for r in results)
            if (
                args.target_valid_trials is not None
                and valid_so_far >= args.target_valid_trials
            ):
                break
            if (
                args.selection_policy == "matched_random"
                and index not in matched_probe_counts
            ):
                raise ValueError(
                    f"Missing matched model metadata for trial index {index}; "
                    "extend the model-policy evaluation first."
                )
            completed_metadata=output/f"trial_{index:03d}"/"metadata.json"
            if args.resume and completed_metadata.exists():
                print(f"[search] {index}/{args.trials} resume",flush=True)
                results.append(
                    json.loads(completed_metadata.read_text(encoding="utf-8"))
                )
                continue
            plan=make_trial_plan(index,args.seed,support_profile=args.support_profile)
            target_text=(
                f" valid={valid_so_far}/{args.target_valid_trials}"
                if args.target_valid_trials is not None else ""
            )
            print(f"[search] {index}/{args.trials}{target_text}",flush=True)
            results.append(
                run_search_trial(
                    plan,
                    model,
                    prototypes,
                    output,
                    args.points,
                    observation_steps=observation_steps,
                    point_pattern=args.point_pattern,
                    selection_policy=args.selection_policy,
                    object_profile=args.object_profile,
                    random_accept_probability=args.random_accept_probability,
                    matched_probe_count=matched_probe_counts.get(index),
                )
            )
            (output/"partial_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
        valid_results=[r for r in results if r.get("trial_valid",True)]
        selected_results=[
            r
            for r in valid_results
            if not r.get("selection_abstained",False)
        ]
        ranks=[r["selected_center_distance_rank"] for r in selected_results]; hinge_ranks=[r["selected_hinge_distance_rank"] for r in selected_results]
        actual_points=1 if args.point_pattern=="center" else args.points
        valid_count=len(valid_results)
        report={"checkpoint":str(args.checkpoint),"support_profile":args.support_profile,"point_pattern":args.point_pattern,"selection_policy":args.selection_policy,"object_profile":args.object_profile,"random_accept_probability":args.random_accept_probability,"matched_policy_dir":str(args.matched_policy_dir) if args.matched_policy_dir else None,"trials":len(results),"valid_trials":valid_count,"invalid_before_release_trials":len(results)-valid_count,"invalid_before_release_rate":(len(results)-valid_count)/len(results),"points_per_trial_max":args.points,"selection_rate":len(selected_results)/valid_count if valid_count else None,"abstention_rate":1.-len(selected_results)/valid_count if valid_count else None,"mean_probes":float(np.mean([len(r["points"]) for r in results])),"probe_observation_reference":"place detection","probe_observation_s":observation_steps*DT,"probe_descent_speed_m_s":.03,"gripper_control":{"finger_kp":FRANKA_KP[-2:].tolist(),"finger_kv":FRANKA_KV[-2:].tolist(),"finger_force_lower":FRANKA_FORCE_LOWER[-2:].tolist(),"finger_force_upper":FRANKA_FORCE_UPPER[-2:].tolist()} ,"templates":{"flat surface":args.flat_template,"wobbling support":args.wobble_template},"prototype_embeddings":{"flat surface":str(args.flat_prototype_embedding),"wobbling support":str(args.wobble_prototype_embedding)},"closest_point_selection_rate":float(np.mean([r["selected_is_closest_to_center"] for r in selected_results])) if selected_results else None,"mean_selected_center_rank":float(np.mean(ranks)) if ranks else None,"selected_center_rank_counts":{str(rank):ranks.count(rank) for rank in range(1,actual_points+1)},"closest_hinge_selection_rate":float(np.mean([r["selected_is_closest_to_hinge"] for r in selected_results])) if selected_results else None,"mean_selected_hinge_rank":float(np.mean(hinge_ranks)) if hinge_ranks else None,"selected_hinge_rank_counts":{str(rank):hinge_ranks.count(rank) for rank in range(1,actual_points+1)},"all_probe_place_detection_rate":float(np.mean([p["place_detected"] for r in results for p in r["points"]])),"final_place_detection_rate":float(np.mean([r["final_place_detected"] for r in valid_results])) if valid_results else None,"final_retention_rate":float(np.mean([r["final_object_retained"] for r in valid_results])) if valid_results else None,"final_non_topple_rate":float(np.mean([not r["final_toppled"] for r in selected_results])) if selected_results else None,"final_stability_rate":float(np.mean([r["final_stable"] for r in valid_results])) if valid_results else None,"conditional_stability_rate":float(np.mean([r["final_stable"] for r in selected_results])) if selected_results else None,"results":results}
        (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2)); return
    results=[]
    for condition in args.conditions:
        for index in range(1,args.trials+1):
            plan=make_trial_plan(index,args.seed,support_profile=args.support_profile)
            print(f"[paired] {condition} {index}/{args.trials} r={plan.offset_radius_m:.3f}",flush=True)
            result=run_trial(
                condition,
                plan,
                model,
                prototypes,
                output,
                post_place_observation_steps=observation_steps,
                post_place_observation_motion=args.post_place_observation_motion,
            )
            results.append(result)
            (output/"partial_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    report={"checkpoint":str(args.checkpoint),"post_place_observation_s":observation_steps*DT,"post_place_observation_motion":args.post_place_observation_motion,"post_place_descent_speed_m_s":0.03 if args.post_place_observation_motion=="descend" else 0.0,"gripper_control":{"finger_kp":FRANKA_KP[-2:].tolist(),"finger_kv":FRANKA_KV[-2:].tolist(),"finger_force_lower":FRANKA_FORCE_LOWER[-2:].tolist(),"finger_force_upper":FRANKA_FORCE_UPPER[-2:].tolist()},"templates":{"hold":"Holding an object.","place":"Placing an object.","flat surface":args.flat_template,"wobbling support":args.wobble_template},"prototype_embeddings":{"flat surface":str(args.flat_prototype_embedding),"wobbling support":str(args.wobble_prototype_embedding)},"overall":summarize(results),"by_condition":{condition:summarize([r for r in results if r["condition"]==condition]) for condition in args.conditions},"results":results}
    (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({"output":str(output),"overall":report["overall"],"by_condition":report["by_condition"]},indent=2))


if __name__=="__main__": main()
