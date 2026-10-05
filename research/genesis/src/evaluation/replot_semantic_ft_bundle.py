#!/usr/bin/env python3
"""Replot a portable semantic-F/T editing bundle (no Genesis install needed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

COLORS = ["#0072B2", "#D55E00"]
STYLES = ["-", "-"]


def plot_item(item_dir, metadata, channels=("x", "y", "z"), destination=None):
    """PDF text is embedded; SVG text stays text with meaningful group IDs."""
    frames = [pd.read_csv(item_dir / e["window_csv"]) for e in metadata["examples"]]
    assert all(len(f) == e.get("sample_count", 80)
               for f, e in zip(frames, metadata["examples"]))
    single = len(channels) == 1
    colors = metadata.get("colors", COLORS)
    styles = metadata.get("line_styles", STYLES)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.linewidth": .6,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "semantic-ft-editing",
        "path.simplify": False,
    }):
        fig, axes = plt.subplots(len(channels), 1, squeeze=False, sharex=True,
                                 figsize=(2.65, 1.5 if single else 3.7))
        axes = axes[:, 0]
        fig.subplots_adjust(left=.28, right=.94, bottom=.31 if single else .13,
                            top=.91 if single else .765, hspace=.15)
        if not single:
            fig.text(.28, .969, metadata["title"], fontsize=11, fontweight="bold",
                     ha="left", va="top", gid="panel_title")
            handles = [Line2D([], [], color=c, linestyle=s, lw=1.15)
                       for c, s in zip(colors, styles)]
            legend = fig.legend(handles, [e["display_label"] for e in metadata["examples"]],
                                loc="upper left", bbox_to_anchor=(.28, .922),
                                frameon=False, borderaxespad=0, handlelength=2,
                                handletextpad=.5, labelspacing=.22)
            legend.set_gid("condition_legend")
        for row, (axis, channel) in enumerate(zip(axes, channels)):
            axis.set_gid(f'{metadata["id"]}_left_f{channel}')
            values = []
            for frame, example, color, style in zip(frames, metadata["examples"], colors, styles):
                y = frame[f'{metadata.get("force_column_prefix", "")}left_f{channel}'].to_numpy()
                values.append(y)
                display_time = frame.window_time_s + example.get("display_time_shift_s", 0.0)
                axis.plot(display_time, y, color=color, linestyle=style, lw=1.05,
                          zorder=3, solid_capstyle="round", dash_capstyle="round",
                          gid=f'curve_{example["id"]}_left_f{channel}')
            if metadata["id"] == "surface" and metadata.get("show_contact", True):
                contacts = [e["first_support_contact_s"] + e.get("display_time_shift_s", 0.0)
                            for e in metadata["examples"]]
                if metadata.get("contact_marker_style") == "per_example":
                    for contact, color in zip(contacts, colors):
                        axis.axvline(contact, color=color, alpha=.5, lw=.65, zorder=1,
                                     gid=f"estimated_contact_{color.lstrip('#')}")
                elif np.isclose(min(contacts), max(contacts)):
                    axis.axvline(contacts[0], color=".68", lw=.65, zorder=1,
                                 gid="first_contact_time")
                else:
                    axis.axvspan(min(contacts), max(contacts), color=".68", alpha=.28,
                                 lw=0, zorder=1, gid="first_contact_times_band")
                if row == 0 and not single:
                    axis.annotate(metadata.get("contact_label", "Contact"), xy=(np.mean(contacts), 1),
                                  xycoords=("data", "axes fraction"), xytext=(0, 3),
                                  textcoords="offset points", ha="center", va="bottom",
                                  fontsize=8, color=".35", gid="contact_label")
            all_values = np.concatenate(values)
            lo, hi = np.min(all_values), np.max(all_values)
            pad = .1 * max(float(hi - lo), .1)
            axis.set_ylim(lo - pad, hi + pad)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color(".4")
            axis.grid(axis="y", color=".9", lw=.45, zorder=0)
            axis.set_xlim(0, .8)
            axis.set_xticks([0, .4, .8])
            symbol = r"\Delta F" if metadata.get("force_is_change", False) else "F"
            axis.set_ylabel(rf"${symbol}_{{{channel}}}$ (N)", labelpad=3)
            if row < len(channels) - 1:
                axis.tick_params(axis="x", bottom=False)
        axes[-1].set_xlabel("Time (s)", labelpad=3)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Include figure-level titles and legends as well as axes labels/ticks.
        artists = [*fig.texts, *fig.legends]
        for axis in axes:
            artists.extend([axis.xaxis.label, axis.yaxis.label, *axis.texts])
            # Locators also return ticks outside the view interval; Matplotlib
            # does not actually draw these, so do not count them as clipping.
            artists.extend(label for value, label in zip(axis.get_xticks(), axis.get_xticklabels())
                           if axis.get_xlim()[0] <= value <= axis.get_xlim()[1])
            artists.extend(label for value, label in zip(axis.get_yticks(), axis.get_yticklabels())
                           if axis.get_ylim()[0] <= value <= axis.get_ylim()[1])
        for artist in artists:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (artist, box)
            assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (artist, box)
        stem = destination or item_dir / metadata["id"]
        for extension in ("pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{extension}"))
        if not single:
            fig.savefig(stem.with_suffix(".png"), dpi=300)
        plt.close(fig)


def render_bundle(bundle):
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    for item in manifest["items"]:
        item_dir = bundle / item["directory"]
        metadata = json.loads((item_dir / "metadata.json").read_text())
        plot_item(item_dir, metadata)
        axes_dir = item_dir / "axes"
        axes_dir.mkdir(exist_ok=True)
        for channel in "xyz":
            plot_item(item_dir, metadata, channels=(channel,),
                      destination=axes_dir / f"F{channel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    render_bundle(args.bundle)
    print("Replotted PDFs/SVGs in", args.bundle)
