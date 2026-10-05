#!/usr/bin/env python3
"""Plot audited qualitative examples from the actual balanced training windows.

The overview uses raw physical units, no filtering or per-window normalization.
The companion shows simulation metadata; these are NOT model input channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv"
OUTPUT = ROOT / "fig/semantic_ft_examples"
COLORS = ["#0072B2", "#D55E00", "#009E73"]
STYLES = ["-", "-", "-"]
FORCE = [f"{side}_f{axis}" for side in ("left", "right") for axis in "xyz"]
WRENCH = [f"{side}_{kind}{axis}" for side in ("left", "right") for kind in "ft" for axis in "xyz"]
SUPPORT = ["obj_plane", "obj_target_tile", "obj_obstacle"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_path(obj, run):
    return ROOT / "data/train_08142026/csv" / obj / "Rigid" / run / f"{obj}_Rigid_{run}.csv"


def load_example(spec, dataset):
    path = episode_path(spec["object"], spec["run"])
    raw = pd.read_csv(path)
    start = spec["start"]
    frame = raw.iloc[start:start + 80].copy()
    matches = dataset[(dataset.csv_path == str(path)) & (dataset.start == start)]
    assert len(matches) == 1, (path, start, "not uniquely in balanced train CSV")
    row = matches.iloc[0]
    assert len(frame) == 80 and np.isfinite(frame[WRENCH]).all().all()
    assert np.allclose(np.diff(frame.step), 1)
    assert float(raw[FORCE].abs().max().max()) < 100
    raw_com = raw[["obj_COM_" + a for a in "xyz"]].to_numpy()
    assert np.isfinite(raw_com).all()
    max_com_step = float(np.linalg.norm(np.diff(raw_com, axis=0), axis=1).max())
    assert max_com_step <= .05
    assert np.linalg.norm(raw_com[:, :2], axis=1).max() <= 2
    assert raw_com[:, 2].min() >= -.1 and raw_com[:, 2].max() <= 2
    assert row.action == spec["action"] and row.interaction == spec["interaction"]
    t = (frame.step.to_numpy() - frame.step.iloc[0]) * .01
    left = frame[["left_finger_" + a for a in "xyz"]].to_numpy()
    right = frame[["right_finger_" + a for a in "xyz"]].to_numpy()
    com = frame[["obj_COM_" + a for a in "xyz"]].to_numpy()
    contact = (frame.obj_left_finger > 0) & (frame.obj_right_finger > 0)
    support = (frame[SUPPORT] > 0).any(axis=1)
    axis = np.median((right - left)[contact], axis=0)
    axis /= np.linalg.norm(axis)
    relative = com - (left + right) / 2
    tangent = relative - np.outer(relative @ axis, axis)
    ids = np.flatnonzero(contact)
    excursion = np.full(80, np.nan)
    excursion[ids[0]:ids[-1] + 1] = np.linalg.norm(tangent[ids[0]:ids[-1] + 1] - tangent[ids[0]], axis=1) * 1000
    speed = float(np.nanmax(excursion) / t[-1])
    assert np.isclose(speed / 1000, row.slip_tangential_speed_mps, atol=1e-9)
    assert (speed >= 5) == (row.interaction == "slip")
    if spec.get("airborne"):
        assert not support.any()
    if spec.get("full_bimanual"):
        assert contact.all()
    if spec["action"] == "grasp":
        assert not contact.iloc[0] and contact.iloc[-1]
    if spec["action"] == "press":
        assert support.all() and frame.eef_z.iloc[-1] < frame.eef_z.iloc[0]
    if spec["action"] == "place":
        assert not support.iloc[0] and support.any()
    if spec.get("weight"):
        assert row.weight == spec["weight"]
        assert (frame.obj_mass.iloc[0] >= .2) == (row.weight == "heavy")
    meta_path = path.parent / f'{spec["object"]}_Rigid_metadata_{spec["run"]}.json'
    metadata = json.loads(meta_path.read_text())
    if "target_surface_angle_deg" in frame:
        angle = frame.target_surface_angle_deg.to_numpy()
        angle_source = "logged target_surface_angle_deg"
    else:
        assert metadata["target_tile"]["inclination_deg"] == 0 or spec.get("airborne")
        angle = np.full(80, float(metadata["target_tile"]["inclination_deg"]))
        angle_source = "fixed tile inclination from metadata (not a measured channel)"
    eef = frame[["eef_" + a for a in "xyz"]].to_numpy()
    if spec["run"] == "wobble_uniform_descent":
        assert metadata["wobble_motion_policy"] == "constant_descent_through_contact"
        assert metadata["wobble_contact_triggered_speed_change"] is False
    frame["window_time_s"] = t
    frame["relative_tangential_excursion_mm"] = excursion
    frame["diagnostic_support_angle_deg"] = angle
    frame["diagnostic_bimanual_contact"] = contact.astype(int)
    frame["diagnostic_support_contact"] = support.astype(int)
    frame["diagnostic_finger_separation_mm"] = np.linalg.norm(right - left, axis=1) * 1000
    frame["diagnostic_eef_z_change_mm"] = (eef[:, 2] - eef[0, 2]) * 1000
    details = {
        **spec,
        "csv_path": str(path), "source_sha256": sha256(path),
        "metadata_path": str(meta_path), "metadata_sha256": sha256(meta_path),
        "balanced_csv_row_zero_based": int(matches.index[0]),
        "source_rows_zero_based_inclusive": [start, start + 79],
        "source_simulation_steps_inclusive": [int(frame.step.iloc[0]), int(frame.step.iloc[-1])],
        "annotation": row.annotation,
        "support_surface": None if pd.isna(row.support_surface) else row.support_surface,
        "mass_g": float(frame.obj_mass.mean() * 1000),
        "episode_max_abs_force_n": float(raw[FORCE].abs().max().max()),
        "episode_max_com_step_m": max_com_step,
        "initial_object_bbox_size_m": [float(raw[f"obj_max_{a}"].iloc[0] - raw[f"obj_min_{a}"].iloc[0]) for a in "xyz"],
        "initial_orientation_wxyz": metadata.get("upright_reference_quat_wxyz"),
        "target_xy": metadata.get("target_xy"),
        "target_tile_geometry": metadata.get("target_tile"),
        "recorded_finger_effort_range_n": frame[["control_left_finger", "control_right_finger"]].agg(["min", "max"]).to_dict(),
        "bimanual_steps": int(contact.sum()), "support_contact_steps": int(support.sum()),
        "first_support_contact_s": float(t[np.flatnonzero(support)[0]]) if support.any() else None,
        "slip_proxy_mm_s": speed, "max_relative_tangential_excursion_mm": float(np.nanmax(excursion)),
        "eef_mean_velocity_mm_s": ((eef[-1] - eef[0]) / t[-1] * 1000).tolist(),
        "eef_path_speed_mm_s": float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum() / t[-1] * 1000),
        "support_angle_source": angle_source, "support_angle_range_deg": [float(angle.min()), float(angle.max())],
        "force_torque_statistics": frame[WRENCH].agg(["mean", "std", "min", "max"]).to_dict(),
    }
    return frame, details


def signal(frame, key):
    if key == "left_force_norm":
        return np.linalg.norm(frame[["left_fx", "left_fy", "left_fz"]], axis=1)
    if key == "sum_fz":
        return (frame.left_fz + frame.right_fz).to_numpy()
    if key == "left_tx_mNm":
        return frame.left_tx.to_numpy() * 1000
    return frame[key].to_numpy()


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": .7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def render_diagnostics(panels, examples, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), layout="constrained")
    for ax, panel in zip(axes.flat, panels):
        key = "diagnostic_signal"
        ylabel = panel["diagnostic_ylabel"]
        all_y = []
        for i, spec in enumerate(panel["examples"]):
            frame, details = examples[spec["id"]]
            y = signal(frame, panel[key])
            all_y.append(y)
            ax.plot(frame.window_time_s, y, color=COLORS[i], linestyle=STYLES[i],
                    lw=1.25, label=spec["label"])
            if panel["id"] == "surface":
                ax.axvline(details["first_support_contact_s"], ymax=.73, color=COLORS[i], ls="-", lw=.7, alpha=.8)
        ax.set_title(panel["title"], loc="left", fontweight="bold", pad=6)
        ax.set_xlabel("Time within window [s]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, .8)
        ax.set_xticks([0, .2, .4, .6, .8])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".88", lw=.55)
        values = np.concatenate(all_y)
        bottom, top = float(np.nanmin(values)), float(np.nanmax(values))
        span = max(top - bottom, .1)
        ax.set_ylim(bottom - .07 * span, top + .37 * span)
        ax.legend(loc="upper left", frameon=False, ncol=3 if panel["id"] == "action" else 1,
                  handlelength=1.6, columnspacing=.85, borderaxespad=.25, handletextpad=.4)
        if panel["id"] == "surface":
            contact_mean = np.mean([examples[s["id"]][1]["first_support_contact_s"]
                                    for s in panel["examples"]])
            ax.text(contact_mean, .035, "contact", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=".35")
        if panel["id"] == "interaction":
            ax.axhline(5 * .79, color=".5", lw=.7, ls="-")
            ax.text(.79, 5 * .79 + .2, "3.95 mm window threshold", ha="right", fontsize=7, color=".35")
    stem = "semantic_ft_diagnostics"
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def render_publication(panels, examples, output, selection):
    """A repeated three-channel layout, sized for a two-column paper figure.

    Each comparison has its own physical-unit limits, shared by its two traces.
    Channel choice, sample count and display transforms are identical throughout.
    Contact is marked at the two actual sample times, never time-aligned by warp.
    """
    plot_specs = selection["publication"]["columns"]
    panel_by_id = {p["id"]: p for p in panels}
    channels = ["left_fx", "left_fy", "left_fz"]
    with plt.rc_context({
        "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9.5,
        "mathtext.fontset": "stix", "axes.linewidth": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": .6, "ytick.major.width": .6,
    }):
        fig, axes = plt.subplots(3, 3, figsize=(7.15, 3.7), sharex=True)
        fig.subplots_adjust(left=.106, right=.969, bottom=.13, top=.765,
                            wspace=.32, hspace=.15)
        for col, column in enumerate(plot_specs):
            panel = panel_by_id[column["id"]]
            assert len(panel["examples"]) == 2
            bbox = axes[0, col].get_position()
            fig.text(bbox.x0, .969, column["title"], ha="left", va="top",
                     fontsize=11, fontweight="bold")
            handles = [Line2D([], [], color=COLORS[i], linestyle=STYLES[i],
                              linewidth=1.15) for i in range(2)]
            fig.legend(handles, column["labels"], loc="upper left",
                       bbox_to_anchor=(bbox.x0, .922), frameon=False,
                       borderaxespad=0, handlelength=2, handletextpad=.5,
                       labelspacing=.22)
            for row, channel in enumerate(channels):
                ax = axes[row, col]
                values = []
                for i, spec in enumerate(panel["examples"]):
                    frame, _ = examples[spec["id"]]
                    y = frame[channel].to_numpy()
                    values.append(y)
                    ax.plot(frame.window_time_s, y, color=COLORS[i],
                            linestyle=STYLES[i], lw=1.05, zorder=3,
                            solid_capstyle="round", dash_capstyle="round")
                if column["id"] == "surface":
                    contacts = [examples[s["id"]][1]["first_support_contact_s"]
                                for s in panel["examples"]]
                    if np.isclose(min(contacts), max(contacts)):
                        ax.axvline(contacts[0], color=".68", lw=.65, zorder=1)
                    else:
                        ax.axvspan(min(contacts), max(contacts), color=".68",
                                   alpha=.28, lw=0, zorder=1)
                    if row == 0:
                        ax.annotate("Contact", xy=(np.mean(contacts), 1),
                                    xycoords=("data", "axes fraction"),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", color=".35",
                                    fontsize=8)
                low, high = np.min(values), np.max(values)
                span = max(float(high - low), .1)
                ax.set_ylim(low - .1 * span, high + .1 * span)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color(".4")
                ax.grid(axis="y", color=".9", lw=.45, zorder=0)
                ax.set_xlim(0, .8)
                ax.set_xticks([0, .4, .8])
                if col == 0:
                    ax.set_ylabel(rf"$F_{{{'xyz'[row]}}}$ (N)", labelpad=3)
                if row < 2:
                    ax.tick_params(axis="x", bottom=False)
            axes[-1, col].set_xlabel("Time (s)", labelpad=3)
        # Explicit final-size check for the renderer: no labels leave the page.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        for artist in [*fig.texts, *fig.legends,
                       *(a.xaxis.label for a in axes.flat),
                       *(a.yaxis.label for a in axes.flat),
                       *(t for a in axes.flat for t in a.get_xticklabels()),
                       *(t for a in axes.flat for t in a.get_yticklabels())]:
            extent = artist.get_window_extent(renderer)
            assert extent.x0 >= -1 and extent.y0 >= -1, (artist, extent)
            assert extent.x1 <= bounds.x1 + 1 and extent.y1 <= bounds.y1 + 1, (artist, extent)
        fig.savefig(output / "semantic_ft_examples.pdf")
        fig.savefig(output / "semantic_ft_examples.png", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=OUTPUT / "selection.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    dataset = pd.read_csv(DATASET)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    examples = {}
    for panel in selection["panels"]:
        for spec in panel["examples"]:
            frame, details = load_example(spec, dataset)
            frame.to_csv(output / f'{spec["id"]}_trace.csv', index=False)
            details["exported_trace"] = f'{spec["id"]}_trace.csv'
            examples[spec["id"]] = frame, details
    light, heavy = examples["weight_light"][1], examples["weight_heavy"][1]
    assert np.allclose(light["initial_object_bbox_size_m"], heavy["initial_object_bbox_size_m"], atol=1e-6)
    assert light["initial_orientation_wxyz"] == heavy["initial_orientation_wxyz"]
    assert light["target_xy"] == heavy["target_xy"]
    assert light["target_tile_geometry"] == heavy["target_tile_geometry"]
    assert abs(light["eef_mean_velocity_mm_s"][2] - heavy["eef_mean_velocity_mm_s"][2]) < .25
    fixed, wobble = examples["surface_fixed"][1], examples["surface_wobble"][1]
    surface_panel = next(p for p in selection["panels"] if p["id"] == "surface")
    limits = surface_panel.get("pair_validation_limits", {})
    assert fixed["object"] == wobble["object"]
    assert fixed["initial_orientation_wxyz"] == wobble["initial_orientation_wxyz"]
    assert abs(fixed["mass_g"] - wobble["mass_g"]) <= limits.get("max_mass_difference_g", .1)
    if "max_initial_bbox_relative_difference" in limits:
        bbox_a = np.asarray(fixed["initial_object_bbox_size_m"])
        bbox_b = np.asarray(wobble["initial_object_bbox_size_m"])
        assert np.max(np.abs(bbox_a - bbox_b) / np.maximum(bbox_a, bbox_b)) <= limits["max_initial_bbox_relative_difference"]
    contacts = [fixed["first_support_contact_s"], wobble["first_support_contact_s"]]
    assert abs(contacts[0] - contacts[1]) <= limits.get("max_first_contact_difference_s", .011)
    contact_display = (f"Coincident first support contact is marked by a gray line at {contacts[0]:.2f} s."
                       if np.isclose(*contacts) else
                       f"Actual support-contact times bound the gray {min(contacts):.2f}-{max(contacts):.2f} s band.")
    configure_style()
    render_publication(selection["panels"], examples, output, selection)
    render_diagnostics(selection["panels"], examples, output)
    manifest = {
        "dataset": str(DATASET), "dataset_sha256": sha256(DATASET),
        "generator": str(Path(__file__).resolve()), "generator_sha256": sha256(Path(__file__)),
        "selection_sha256": sha256(args.selection),
        "window": {"samples": 80, "sampling_hz": 100, "nominal_duration_s": .8, "first_to_last_sample_s": .79},
        "display": "Main figure: left-finger sensor-local Fx, Fy, Fz only, in N, identical channel choice in all three comparisons. No smoothing, detrending, normalization, aggregation, cropping within windows or time warping. Both traces share one physical-unit y-axis in each subplot; y limits differ across comparisons. " + contact_display,
        "publication_size_inches": [7.15, 3.7],
        "line_style": "solid for all curves and reference lines; conditions distinguished by color",
        "publication_example_ids": [s["id"] for p in selection["panels"] if p["id"] in {c["id"] for c in selection["publication"]["columns"]} for s in p["examples"]],
        "diagnostics": "Contact, pose, mass and support angle validate labels; they are not force-encoder inputs.",
        "selection": selection,
        "examples": {key: details for key, (_, details) in examples.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: {j: v[j] for j in ("start", "mass_g", "slip_proxy_mm_s", "bimanual_steps", "support_contact_steps")} for k, (_, v) in examples.items()}, indent=2))


if __name__ == "__main__":
    main()
