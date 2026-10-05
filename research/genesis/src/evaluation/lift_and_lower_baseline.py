import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import genesis as gs
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PATCHTST_ROOT = ROOT / "external" / "PatchTST_tmp" / "PatchTST_self_supervised"
if str(PATCHTST_ROOT) not in sys.path:
    sys.path.append(str(PATCHTST_ROOT))

LO_PATH = ROOT / "src" / "evaluation" / "lift_and_lower_object.py"
spec = importlib.util.spec_from_file_location("lift_and_lower_object", LO_PATH)
lo = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(lo)

DT = lo.DT
PANDA_XML_PATH = lo.PANDA_XML_PATH
PROJECT_ROOT = lo.ROOT
YCB_TARGET_OBJ_PATH = lo.YCB_TARGET_OBJ_PATH
control_pose = lo.control_pose
entities_are_touching = lo.entities_are_touching
get_force_vector_12 = lo.get_force_vector_12
ik_pose = lo.ik_pose
infer_finger_length_from_mjcf = lo.infer_finger_length_from_mjcf
move_ee = lo.move_ee
sample_object_params = lo.sample_object_params
scaled_steps = lo.scaled_steps
set_gripper = lo.set_gripper
sim_step = lo.sim_step
set_grasp_for_obj = lo.set_grasp_for_obj
_scale_to_vec = lo._scale_to_vec

TERRAIN_RULE_THRESHOLDS = {
    "slope_angle_deg": 10.0,
    "slope_force_asymmetry": 0.45,
    "step_force_asymmetry": 0.30,
    "step_moment_arm_m": 0.004,
}


def estimate_terrain_from_force_window(force_window) -> dict:
    window = np.asarray(force_window, dtype=np.float32)
    if window.ndim != 2 or window.shape[0] < 2 or window.shape[1] != len(lo.EXPECTED_FORCE_COLS):
        return {
            "prediction": None,
            "features": None,
            "window_samples": int(window.shape[0]) if window.ndim >= 1 else 0,
            "reason": "insufficient_force_window",
        }

    baseline_end = max(1, window.shape[0] - 10)
    baseline = np.median(window[:baseline_end], axis=0)
    response = np.mean(window[-min(3, window.shape[0]):], axis=0) - baseline
    left_force, left_torque = response[:3], response[3:6]
    right_force, right_torque = response[6:9], response[9:12]
    net_force = left_force + right_force
    total_finger_force = np.linalg.norm(left_force) + np.linalg.norm(right_force)
    vertical_force = abs(float(net_force[2])) + 1e-6
    surface_normal_angle_deg = float(
        np.degrees(np.arctan2(np.linalg.norm(net_force[:2]), vertical_force))
    )
    force_asymmetry = float(
        np.linalg.norm(left_force - right_force) / (total_finger_force + 1e-6)
    )
    moment_arm_m = float(
        np.linalg.norm(left_torque + right_torque) / (total_finger_force + 1e-6)
    )

    if (
        surface_normal_angle_deg >= TERRAIN_RULE_THRESHOLDS["slope_angle_deg"]
        or force_asymmetry >= TERRAIN_RULE_THRESHOLDS["slope_force_asymmetry"]
    ):
        prediction = "slope"
    elif (
        force_asymmetry >= TERRAIN_RULE_THRESHOLDS["step_force_asymmetry"]
        or moment_arm_m >= TERRAIN_RULE_THRESHOLDS["step_moment_arm_m"]
    ):
        prediction = "step"
    else:
        prediction = "flat surface"

    return {
        "prediction": prediction,
        "features": {
            "surface_normal_angle_deg": surface_normal_angle_deg,
            "force_asymmetry": force_asymmetry,
            "moment_arm_m": moment_arm_m,
        },
        "window_samples": int(window.shape[0]),
        "thresholds": TERRAIN_RULE_THRESHOLDS,
        "method": "force_normal_and_load_asymmetry_rule_v1",
    }


def compute_release_by_spike(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    hand_idx,
    left_finger_idx,
    obj_entity,
    support_entity,
    finger_len,
    obj_xy,
    target_xy,
    source_surface_z,
    target_surface_z,
    threshold,
    detector,
    confirmation_steps,
    motion_speed,
    descent_speed_m_s,
    terrain_ground_truth,
    max_wait_steps,
    cam=None,
):
    compute_grasp_heights = lo.compute_grasp_heights
    hold_pose = lo.hold_pose

    x, y = obj_xy
    hover_z = 0.24
    grasp_z, clamp_z, safe_retract_z, lift_z, lower_back_z = compute_grasp_heights(
        franka=franka,
        hand_idx=hand_idx,
        left_finger_idx=left_finger_idx,
        obj_entity=obj_entity,
        finger_len=finger_len,
        hover_z=hover_z,
    )

    lo.set_trace_phase("approach_open")
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(150, motion_speed), cam, False)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), 0.04, scaled_steps(100, motion_speed), cam, False)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.04, scaled_steps(35, motion_speed), cam, False)
    lo.set_trace_phase("close_gripper")
    set_gripper(scene, franka, motors_dof, fingers_dof, 0.04, 0.0, scaled_steps(60, motion_speed), cam, False)
    lo.set_trace_phase("lift_from_source")
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(50, motion_speed), cam, False)
    surface_delta_z = float(target_surface_z) - float(source_surface_z)
    target_safe_retract_z = safe_retract_z + surface_delta_z
    target_lower_back_z = lower_back_z + surface_delta_z
    transport_z = max(lift_z, target_safe_retract_z + 0.10)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, transport_z), 0.0, scaled_steps(120, motion_speed), cam, False)
    lo.set_trace_phase("transport_hold")
    hold_pose(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, transport_z), 0.0, int(round(1.0 / DT)), cam, False)
    lo.start_grasp_monitor(franka, obj_entity, hand_idx)
    target_x, target_y = map(float, target_xy)
    reference_descent_steps = scaled_steps(lo.EXPLORATION_DESCENT_REFERENCE_STEPS, motion_speed)
    reference_distance = abs(float(target_safe_retract_z) - float(target_lower_back_z))
    if descent_speed_m_s is None:
        effective_descent_speed_m_s = reference_distance / (reference_descent_steps * DT)
        base_descent_speed_m_s = effective_descent_speed_m_s / max(float(motion_speed), 1e-3)
        descent_speed_source = "derived_from_legacy_exploration_segment"
    else:
        base_descent_speed_m_s = float(descent_speed_m_s)
        effective_descent_speed_m_s = base_descent_speed_m_s * max(float(motion_speed), 1e-3)
        descent_speed_source = "fixed_speed"
    lo.set_trace_phase("transport_to_target")
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (target_x, target_y, transport_z), 0.0, scaled_steps(180, motion_speed), cam, False)
    lo.set_trace_phase("approach_target_closed")
    approach_start = franka.get_links_pos([8])[0].cpu().numpy().reshape(3)
    approach_target = np.array([target_x, target_y, target_safe_retract_z], dtype=float)
    approach_steps = lo.constant_speed_steps(
        approach_start, approach_target, base_descent_speed_m_s, motion_speed
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof,
        approach_target, 0.0, approach_steps, cam, False,
    )

    # Preserve the commanded trajectory across the phase boundary; restarting
    # from the lagging measured pose would introduce an artificial speed jump.
    start_pos = approach_target.copy()
    x, y = target_x, target_y
    target_pos = np.array([x, y, target_lower_back_z], dtype=float)
    descend_steps = (
        reference_descent_steps
        if descent_speed_m_s is None
        else lo.constant_speed_steps(start_pos, target_pos, base_descent_speed_m_s, motion_speed)
    )

    floor_touch_step = None
    release_step = None
    spike_trace = []
    prev_fz = None
    confirmation_streak = 0
    force_buffer = []

    lo.set_trace_phase("adaptive_descent_closed")
    for i in range(descend_steps):
        alpha = (i + 1) / descend_steps
        interp = (1.0 - alpha) * start_pos + alpha * target_pos
        qpos = ik_pose(franka, end_effector, interp)
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=0.0)
        sim_step(scene, cam=cam, record=False)

        v = get_force_vector_12(franka)
        force_buffer.append(v)
        force_buffer = force_buffer[-lo.SEQ_LEN:]
        fz = float(np.abs(v[2]) + np.abs(v[8]))
        signed_fz = float(v[2] + v[8])
        spike = 0.0 if prev_fz is None else float(abs(fz - prev_fz))
        prev_fz = fz
        threshold_met = signed_fz < threshold if detector == "signed_fz" else spike >= threshold
        confirmation_streak = confirmation_streak + 1 if threshold_met else 0
        touching_support = entities_are_touching(obj_entity, support_entity)
        spike_trace.append(
            {
                "step": int(scene.t),
                "fz_sum_abs": fz,
                "fz_sum_signed": signed_fz,
                "spike": spike,
                "threshold_met": bool(threshold_met),
                "confirmation_streak": int(confirmation_streak),
                "touching_support": bool(touching_support),
            }
        )

        if touching_support and floor_touch_step is None:
            floor_touch_step = int(scene.t)

        if confirmation_streak >= confirmation_steps:
            release_step = int(scene.t)
            break

    if release_step is None:
        lo.set_trace_phase("post_descent_wait_closed")
        qpos_hold = ik_pose(franka, end_effector, (x, y, target_lower_back_z))
        for _ in range(max_wait_steps):
            control_pose(franka, motors_dof, fingers_dof, qpos_hold, gripper_opening=0.0)
            sim_step(scene, cam=cam, record=False)
            v = get_force_vector_12(franka)
            force_buffer.append(v)
            force_buffer = force_buffer[-lo.SEQ_LEN:]
            fz = float(np.abs(v[2]) + np.abs(v[8]))
            signed_fz = float(v[2] + v[8])
            spike = 0.0 if prev_fz is None else float(abs(fz - prev_fz))
            prev_fz = fz
            threshold_met = signed_fz < threshold if detector == "signed_fz" else spike >= threshold
            confirmation_streak = confirmation_streak + 1 if threshold_met else 0
            touching_support = entities_are_touching(obj_entity, support_entity)
            spike_trace.append(
                {
                    "step": int(scene.t),
                    "fz_sum_abs": fz,
                    "fz_sum_signed": signed_fz,
                    "spike": spike,
                    "threshold_met": bool(threshold_met),
                    "confirmation_streak": int(confirmation_streak),
                    "touching_support": bool(touching_support),
                }
            )
            if touching_support and floor_touch_step is None:
                floor_touch_step = int(scene.t)
            if confirmation_streak >= confirmation_steps:
                release_step = int(scene.t)
                break

    released = release_step is not None
    terrain_result = estimate_terrain_from_force_window(force_buffer)
    if terrain_result is not None:
        terrain_result.update(
            {
                "ground_truth": terrain_ground_truth,
                "correct": bool(terrain_result["prediction"] == terrain_ground_truth),
                "window_end_step": int(release_step if release_step is not None else scene.t),
                "trigger": "threshold_release" if released else "final_window_no_release",
            }
        )
    if released:
        lo.mark_intentional_release()
        lo.set_trace_phase("intentional_release_opening")
        set_gripper(scene, franka, motors_dof, fingers_dof, 0.0, 0.04, scaled_steps(40, motion_speed), None, False)

    delay_sec = None
    if released and floor_touch_step is not None:
        delay_sec = (release_step - floor_touch_step) * DT

    lo.mark_intentional_release()
    lo.set_trace_phase("post_trial_retract")
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(100, motion_speed), cam, False)
    return {
        "released": released,
        "release_step": release_step,
        "floor_touch_step": floor_touch_step,
        "floor_touched_at_release": bool(released and floor_touch_step is not None and release_step >= floor_touch_step),
        "delay_sec_after_touch": delay_sec,
        "spike_max": float(max([r["spike"] for r in spike_trace], default=0.0)),
        "detector": detector,
        "threshold": float(threshold),
        "confirmation_steps": int(confirmation_steps),
        "descent_motion": {
            "commanded_speed_m_s": float(effective_descent_speed_m_s),
            "speed_source": descent_speed_source,
            "legacy_exploration_reference_steps": int(reference_descent_steps),
            "approach_steps": int(approach_steps),
            "adaptive_descent_steps_planned": int(descend_steps),
        },
        "terrain_classification": terrain_result,
        "_terrain_force_window": (
            None if terrain_result is None else np.asarray(force_buffer, dtype=np.float32).tolist()
        ),
        "spike_trace": spike_trace,
        "spike_trace_tail": spike_trace[-20:],
        **lo.drop_monitor_metrics(),
    }


def run_trial(args, params: dict, threshold: float) -> dict:
    if torch.cuda.is_available():
        gs.init(backend=gs.gpu, logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.6, -1.1, 1.3), camera_lookat=(0.55, 0.0, 0.20), camera_fov=35, max_FPS=60),
        show_viewer=args.show_viewer,
    )
    floor_entity = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))
    cam = scene.add_camera(
        model="thinlens",
        res=(960, 960),
        pos=(1.7, 0.0, 0.7),
        lookat=(0.60, 0.0, 0.16),
        fov=35,
        GUI=False,
    )

    obj_xy = (0.62, 0.0)
    target_xy = (0.48, 0.32)
    source_entity = scene.add_entity(
        material=gs.materials.Rigid(rho=800, friction=1.0, coup_friction=1.0),
        morph=gs.morphs.Box(
            size=(lo.TARGET_TILE_SIZE, lo.TARGET_TILE_SIZE, lo.TARGET_TILE_THICKNESS),
            pos=(obj_xy[0], obj_xy[1], lo.TARGET_TILE_THICKNESS * 0.5),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=lo.TARGET_TILE_COLOR),
    )
    support_top_z = lo.TARGET_TILE_THICKNESS
    trial_index = int(getattr(args, "_trial_index", 0))
    surface_plan = lo.plan_evaluation_surface(
        args.surface_mode, random.Random(args.seed + 10_000 + trial_index)
    )
    support_entities = lo.add_evaluation_support(scene, target_xy, surface_plan)
    if args.object_type == "cube":
        cube_size = params["size"]
        obj_entity = scene.add_entity(
            material=gs.materials.Rigid(rho=params["rho"], friction=1.0),
            morph=gs.morphs.Box(
                size=(cube_size, cube_size, cube_size),
                pos=(obj_xy[0], obj_xy[1], support_top_z + cube_size * 0.5 + 0.001),
            ),
            surface=gs.surfaces.Default(color=(0.35, 0.75, 0.95, 1.0)),
        )
    else:
        object_scale, object_euler, min_corner = set_grasp_for_obj(YCB_TARGET_OBJ_PATH)
        scale_vec = _scale_to_vec(object_scale)
        spawn_z = support_top_z + 0.001 + max(0.0, -scale_vec[2] * min_corner[2])
        obj_entity = scene.add_entity(
            material=gs.materials.Rigid(rho=params["rho"], friction=1.0),
            morph=gs.morphs.Mesh(
                file=str(YCB_TARGET_OBJ_PATH),
                scale=object_scale,
                pos=(obj_xy[0], obj_xy[1], float(spawn_z)),
                euler=object_euler,
            ),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0, 1.0)),
        )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=3.0, friction=1.0),
    )
    scene.build(n_envs=1)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")
    hand_idx = next(i for i, link in enumerate(franka.links) if link.name == "hand")
    left_finger_idx = next(i for i, link in enumerate(franka.links) if link.name == "left_finger")
    right_finger_idx = next(i for i, link in enumerate(franka.links) if link.name == "right_finger")
    finger_len = infer_finger_length_from_mjcf(PANDA_XML_PATH)

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]), np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))

    home_q = ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
    home_q[0, -2:] = 0.04
    franka.set_dofs_position(home_q[0, :-2], motors_dof)
    franka.set_dofs_position(home_q[0, -2:], fingers_dof)
    lo.STEP_TRACE_RECORDER.update(
        {
            "enabled": True,
            "rows": [],
            "franka": franka,
            "obj_entity": obj_entity,
            "target_support": support_entities,
            "source_support": source_entity,
            "floor_entity": floor_entity,
            "hand_idx": hand_idx,
            "left_finger_link_idx": int(franka.links[left_finger_idx].idx),
            "right_finger_link_idx": int(franka.links[right_finger_idx].idx),
            "phase": "initial_settle",
            "grasp_monitor_active": False,
            "intentional_release_commanded": False,
            "grasp_reference_relative_xyz": None,
            "relative_distance_threshold_m": float(args.drop_relative_distance_threshold),
            "drop_confirmation_steps": int(args.drop_confirmation_steps),
            "drop_streak": 0,
            "dropped_before_release": False,
            "drop_detected_step": None,
            "drop_detected_phase": None,
        }
    )
    trial_dir = getattr(args, "_current_trial_dir", None)
    if trial_dir is not None and args.save_trial_images:
        img_dir = Path(trial_dir) / "images" / "camera_0"
        img_dir.mkdir(parents=True, exist_ok=True)
        lo.TRACE_RECORDER["enabled"] = True
        lo.TRACE_RECORDER["images_dir"] = img_dir
        lo.TRACE_RECORDER["image_stride"] = 20
    else:
        lo.TRACE_RECORDER["enabled"] = False

    for _ in range(120):
        control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
        sim_step(scene, cam=cam, record=False)

    metrics = compute_release_by_spike(
        scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx,
        obj_entity, support_entities, finger_len, obj_xy,
        target_xy, lo.TARGET_TILE_THICKNESS, surface_plan["surface_center_z"],
        threshold=threshold,
        detector=getattr(args, "detector", "spike"),
        confirmation_steps=max(1, int(getattr(args, "confirmation_steps", 1))),
        motion_speed=args.motion_speed,
        descent_speed_m_s=args.descent_speed_m_s,
        terrain_ground_truth=lo.surface_mode_to_terrain_label(args.surface_mode),
        max_wait_steps=args.post_descend_wait_steps,
        cam=cam,
    )
    out = {
        "object_params": params,
        "surface_plan": surface_plan,
        "metrics": metrics,
    }
    if trial_dir is not None:
        csv_dir = Path(trial_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        terrain_window_csv = csv_dir / "terrain_release_window.csv"
        terrain_window = out["metrics"].pop("_terrain_force_window", None)
        terrain_window_rows = (
            [dict(zip(lo.EXPECTED_FORCE_COLS, row)) for row in terrain_window]
            if terrain_window is not None else []
        )
        lo.save_dict_rows_to_csv(terrain_window_rows, terrain_window_csv)
        if out["metrics"].get("terrain_classification") is not None:
            out["metrics"]["terrain_classification"]["window_csv"] = str(terrain_window_csv)
        out["terrain_window_csv"] = str(terrain_window_csv)
        lo.save_dict_rows_to_csv(metrics["spike_trace"], csv_dir / "spike_trace.csv")
        lo.save_dict_rows_to_csv([metrics], csv_dir / "trial_metrics.csv")
        lo.save_dict_rows_to_csv(lo.STEP_TRACE_RECORDER["rows"], csv_dir / "step_trace.csv")
        steps = [r["step"] for r in metrics["spike_trace"]]
        fz = [r["fz_sum_abs"] for r in metrics["spike_trace"]]
        signed_fz = [r["fz_sum_signed"] for r in metrics["spike_trace"]]
        sp = [r["spike"] for r in metrics["spike_trace"]]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(steps, fz, label="fz_sum_abs")
        ax.plot(steps, signed_fz, label="fz_sum_signed")
        ax.plot(steps, sp, label="spike")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(csv_dir / "force_plot.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    lo.STEP_TRACE_RECORDER["enabled"] = False
    gs.destroy()
    return out


def success_from_metrics(m: dict) -> bool:
    if not m["released"]:
        return False
    if not m["floor_touched_at_release"]:
        return False
    d = m["delay_sec_after_touch"]
    return d is not None and 0.0 <= d <= 0.5


def classify_outcome(m: dict) -> str:
    if not m["released"]:
        return "no_release"
    if not m["floor_touched_at_release"]:
        return "early_release"
    d = m["delay_sec_after_touch"]
    if d is not None and 0.0 <= d <= 0.5:
        return "success"
    return "late_or_other"


def replay_threshold(metrics: dict, threshold: float) -> dict:
    crossing = next((row for row in metrics["spike_trace"] if row["spike"] >= threshold), None)
    release_step = None if crossing is None else int(crossing["step"])
    support_step = metrics["floor_touch_step"]
    released = release_step is not None
    touched_at_release = bool(released and support_step is not None and release_step >= support_step)
    delay_sec = None
    if touched_at_release:
        delay_sec = (release_step - support_step) * DT
    return {
        "released": released,
        "release_step": release_step,
        "floor_touch_step": support_step,
        "floor_touched_at_release": touched_at_release,
        "delay_sec_after_touch": delay_sec,
    }


def evaluate_threshold(threshold: float, calibration_results: list[dict]) -> dict:
    results = [replay_threshold(result["metrics"], threshold) for result in calibration_results]
    counts = {"success": 0, "early_release": 0, "no_release": 0, "late_or_other": 0}
    for metrics in results:
        counts[classify_outcome(metrics)] += 1
    n = max(1, len(results))
    rates = {k: counts[k] / n for k in counts}
    return {
        "threshold": float(threshold),
        "trials": len(results),
        "counts": counts,
        "rates": rates,
    }


def select_best_threshold(args, history: list[dict]) -> tuple[dict, str]:
    feasible = [
        h
        for h in history
        if h["rates"]["early_release"] <= args.max_early_release_rate and h["rates"]["no_release"] <= args.max_no_release_rate
    ]

    if feasible:
        best = max(feasible, key=lambda h: (h["rates"]["success"], h["threshold"]))
        selection_mode = "feasible_max_success"
    else:
        # Fallback: minimize constraint violation first, then maximize success, then prefer larger threshold.
        def key(h):
            early_excess = max(0.0, h["rates"]["early_release"] - args.max_early_release_rate)
            no_release_excess = max(0.0, h["rates"]["no_release"] - args.max_no_release_rate)
            violation = early_excess + no_release_excess
            return (-violation, h["rates"]["success"], h["threshold"])

        best = max(history, key=key)
        selection_mode = "fallback_min_violation"
    return best, selection_mode


def collect_calibration_results(args) -> list[dict]:
    rng = random.Random(args.seed + 1000)
    results = []
    for trial_index in range(args.search_trials):
        args._trial_index = trial_index
        params = sample_object_params("cube", rng)
        # An infinite threshold records the complete trajectory. Every finite
        # threshold can then be replayed from the same paired traces.
        result = run_trial(args, params, float("inf"))
        results.append(result)
        print(
            f"[calibration {trial_index + 1:02d}/{args.search_trials}] "
            f"spike_max={result['metrics']['spike_max']:.4f} "
            f"support_step={result['metrics']['floor_touch_step']}"
        )
    return results


def evaluate_grid(candidates: np.ndarray, calibration_results: list[dict], stage: str) -> list[dict]:
    history = []
    for index, threshold in enumerate(candidates):
        record = evaluate_threshold(float(threshold), calibration_results)
        record["stage"] = stage
        record["candidate_index"] = index + 1
        history.append(record)
    return history


def select_threshold_with_constraints(args, calibration_results: list[dict]) -> tuple[float, dict]:
    coarse_candidates = np.linspace(args.threshold_min, args.threshold_max, args.coarse_search_points)
    coarse_history = evaluate_grid(coarse_candidates, calibration_results, "coarse")
    coarse_best, coarse_mode = select_best_threshold(args, coarse_history)

    coarse_index = int(coarse_best["candidate_index"]) - 1
    lower_index = max(0, coarse_index - 1)
    upper_index = min(len(coarse_candidates) - 1, coarse_index + 1)
    fine_min = float(coarse_candidates[lower_index])
    fine_max = float(coarse_candidates[upper_index])
    fine_candidates = np.linspace(fine_min, fine_max, args.fine_search_points)
    fine_history = evaluate_grid(fine_candidates, calibration_results, "fine")
    fine_best, fine_mode = select_best_threshold(args, fine_history)

    metadata = {
        "coarse": {
            "range": [float(args.threshold_min), float(args.threshold_max)],
            "points": int(args.coarse_search_points),
            "selection_mode": coarse_mode,
            "selected_record": coarse_best,
            "history": coarse_history,
        },
        "fine": {
            "range": [fine_min, fine_max],
            "points": int(args.fine_search_points),
            "selection_mode": fine_mode,
            "selected_record": fine_best,
            "history": fine_history,
        },
        "trial_accounting": {
            "physical_calibration_trials": len(calibration_results),
            "coarse_candidate_outcomes": len(coarse_history) * len(calibration_results),
            "fine_candidate_outcomes": len(fine_history) * len(calibration_results),
            "total_candidate_outcomes": (len(coarse_history) + len(fine_history)) * len(calibration_results),
            "additional_physical_trials_for_fine_search": 0,
        },
    }
    return float(fine_best["threshold"]), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "data" / "evaluation_outputs"))
    parser.add_argument("--object-type", choices=["cube", "ycb_wood_block"], default="cube")
    parser.add_argument("--surface-mode", choices=lo.SURFACE_MODES, default="flat")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument(
        "--save-trial-images",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument(
        "--descent-speed-m-s",
        type=float,
        default=lo.DEFAULT_DESCENT_SPEED_M_S,
        help=(
            "Constant Cartesian command speed for both closed-gripper descent phases. "
            "Defaults to 0.03 m/s (30 mm/s)."
        ),
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--drop-relative-distance-threshold", type=float, default=0.02)
    parser.add_argument("--drop-confirmation-steps", type=int, default=5)
    parser.add_argument("--post-descend-wait-steps", type=int, default=1200)
    parser.add_argument("--detector", choices=["signed_fz", "spike"], default="signed_fz")
    parser.add_argument("--signed-fz-threshold", type=float, default=0.0)
    parser.add_argument(
        "--confirmation-steps",
        type=int,
        default=None,
        help="Consecutive threshold crossings; defaults to 2 for signed_fz and 1 for spike.",
    )
    parser.add_argument("--coarse-search-points", type=int, default=21)
    parser.add_argument("--fine-search-points", type=int, default=21)
    parser.add_argument("--search-trials", type=int, default=10)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=20.0)
    parser.add_argument("--max-early-release-rate", type=float, default=0.10)
    parser.add_argument("--max-no-release-rate", type=float, default=0.20)
    args = parser.parse_args()
    if args.descent_speed_m_s is not None and args.descent_speed_m_s <= 0.0:
        parser.error("--descent-speed-m-s must be positive")
    if args.confirmation_steps is None:
        args.confirmation_steps = 2 if args.detector == "signed_fz" else 1

    if args.detector == "spike":
        calibration_results = collect_calibration_results(args)
        chosen, search_meta = select_threshold_with_constraints(args, calibration_results)
        for stage in ("coarse", "fine"):
            print(f"{stage} threshold search history:")
            for record in search_meta[stage]["history"]:
                print(
                    f"idx={record['candidate_index']} th={record['threshold']:.4f} "
                    f"succ={record['rates']['success']:.3f} early={record['rates']['early_release']:.3f} "
                    f"norelease={record['rates']['no_release']:.3f}"
                )
        print(
            f"coarse_chosen={search_meta['coarse']['selected_record']['threshold']:.4f} "
            f"fine_range={search_meta['fine']['range']} chosen_threshold={chosen:.4f}"
        )
    else:
        calibration_results = []
        chosen = float(args.signed_fz_threshold)
        search_meta = {
            "strategy": "fixed_physical_zero_crossing",
            "trial_accounting": {
                "physical_calibration_trials": 0,
                "total_candidate_outcomes": 0,
            },
        }
        print(
            f"signed_fz detector: threshold={chosen:.4f} N "
            f"confirmation_steps={args.confirmation_steps}; no threshold tuning"
        )

    rng = random.Random(args.seed)
    run_root = Path(args.output_dir) / f"lift_and_lower_baseline_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    final_results = []
    excluded_results = []
    attempt_index = 0
    while len(final_results) < args.trials and attempt_index < args.max_attempts:
        args._trial_index = attempt_index
        trial_dir = run_root / f"attempt_{attempt_index+1:03d}"
        args._current_trial_dir = trial_dir
        if args.object_type == "cube":
            params = sample_object_params("cube", rng)
        else:
            params = {"rho": float(200.0 * rng.uniform(1.0, 10.0)), "size": None, "scale": None}
        r = run_trial(args, params, chosen)
        r["attempt_index"] = attempt_index + 1
        if r["metrics"]["dropped_before_release"]:
            r["excluded_reason"] = "dropped_before_release"
            excluded_results.append(r)
            print(
                f"[attempt {attempt_index + 1:03d}] EXCLUDED dropped_before_release "
                f"step={r['metrics']['drop_detected_step']} phase={r['metrics']['drop_detected_phase']}"
            )
            attempt_index += 1
            continue
        r["valid_trial_index"] = len(final_results) + 1
        final_results.append(r)
        terrain = r["metrics"].get("terrain_classification")
        print(
            f"[valid {len(final_results):02d}/{args.trials}, attempt {attempt_index+1:03d}] "
            f"released={r['metrics']['released']} "
            f"floor_at_release={r['metrics']['floor_touched_at_release']} "
            f"delay={r['metrics']['delay_sec_after_touch']} "
            f"terrain={terrain['prediction'] if terrain else 'None'}"
        )
        attempt_index += 1

    if len(final_results) < args.trials:
        raise RuntimeError(
            f"Only {len(final_results)} valid trials after {attempt_index} attempts; "
            f"increase --max-attempts (currently {args.max_attempts})."
        )

    success = sum(1 for r in final_results if success_from_metrics(r["metrics"]))
    no_release = sum(1 for r in final_results if not r["metrics"]["released"])
    early_release = sum(
        1 for r in final_results if r["metrics"]["released"] and not r["metrics"]["floor_touched_at_release"]
    )
    late_or_other = args.trials - success - no_release - early_release
    delays = [
        float(result["metrics"]["delay_sec_after_touch"])
        for result in final_results
        if result["metrics"]["delay_sec_after_touch"] is not None
    ]
    delay_stats = {
        "count": len(delays),
        "mean": float(np.mean(delays)) if delays else None,
        "std": float(np.std(delays)) if delays else None,
        "min": float(np.min(delays)) if delays else None,
        "max": float(np.max(delays)) if delays else None,
    }
    terrain_results = [
        result["metrics"]["terrain_classification"]
        for result in final_results
        if result["metrics"].get("terrain_classification") is not None
    ]
    terrain_summary = {
        "count": len(terrain_results),
        "correct_count": sum(bool(result["correct"]) for result in terrain_results),
        "accuracy": (
            sum(bool(result["correct"]) for result in terrain_results) / len(terrain_results)
            if terrain_results else None
        ),
        "prediction_counts": {
            label: sum(result["prediction"] == label for result in terrain_results)
            for label in lo.TERRAIN_TEMPLATES
        },
        "method": "force_normal_and_load_asymmetry_rule_v1",
        "thresholds": TERRAIN_RULE_THRESHOLDS,
    }
    payload = {
        "method": f"{args.detector}_baseline",
        "surface_mode": args.surface_mode,
        "descent_speed_m_s": args.descent_speed_m_s,
        "terrain_baseline": {
            "input": "release-ending force window only",
            "uses_ground_truth_geometry_for_inference": False,
            "summary": terrain_summary,
        },
        "threshold_search": {
            "strategy": (
                "two_stage_linear_grid_paired_trace_replay"
                if args.detector == "spike"
                else "fixed_physical_zero_crossing"
            ),
            "search_trials": len(calibration_results),
            "constraints": {
                "max_early_release_rate": args.max_early_release_rate,
                "max_no_release_rate": args.max_no_release_rate,
            },
            **search_meta,
            "chosen_threshold": chosen,
            "confirmation_steps": int(args.confirmation_steps),
        },
        "final_eval": {
            "trials": args.trials,
            "success_count": success,
            "success_rate": success / max(1, args.trials),
        },
        "trial_results": final_results,
        "excluded_trials": excluded_results,
        "summary": {
            "trials": args.trials,
            "physical_attempts": attempt_index,
            "excluded_trial_count": len(excluded_results),
            "success_count": success,
            "success_rate": success / max(1, args.trials),
            "no_release_count": no_release,
            "early_release_count": early_release,
            "late_or_other_count": late_or_other,
            "delay_sec_after_touch": delay_stats,
            "terrain_classification": terrain_summary,
            "physical_search_trials": len(calibration_results),
            "physical_final_eval_trials": attempt_index,
            "total_physical_trials": len(calibration_results) + attempt_index,
        },
    }

    out_dir = Path(args.output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"lift_and_lower_baseline_{args.object_type}_{args.surface_mode}_summary_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
