#!/usr/bin/env python3
"""Export real fixed/wobble examples; source bags are opened read-only.

Run with ~/ur3_robotiq_ros2/.venv/bin/python. No ROS node is started.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geometry_msgs.msg import WrenchStamped
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("FT_WORKSPACE_ROOT", ROOT.parent)).expanduser().resolve()
ROBOT = Path(os.environ.get("FT_ROBOT_ROOT", WORKSPACE / "ur3_robotiq_ros2")).expanduser().resolve()
SOURCE = Path(os.environ.get("FT_ROBOT_DATA_ROOT", ROBOT / "data")).expanduser().resolve()
OUT = ROOT / "fig/real_surface_ft_examples_20260915"
SPECS = [
    ("fixed", "Fixed support", "place_rectangle_fz_baseline/20260827_195736_740533/episode_000003"),
    ("wobble", "Wobbling support", "place_rectangle_wobble_fz_baseline/20260827_234339_773604/episode_000005"),
]
CHANNELS = [f"{side}_{axis}" for side in ("left", "right")
            for axis in ("fx", "fy", "fz", "tx", "ty", "tz")]
LIMITS = (-.35, .45)
COLORS = ["#0072B2", "#D55E00"]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_forces(database):
    rows = {"left": [], "right": []}
    query = """SELECT topics.name,messages.data FROM messages
        JOIN topics ON topics.id=messages.topic_id
        WHERE topics.name IN ('/force_torque/left','/force_torque/right')
        ORDER BY messages.timestamp,messages.id"""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        for topic, blob in connection.execute(query):
            msg = deserialize_message(blob, WrenchStamped)
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            f, t = msg.wrench.force, msg.wrench.torque
            rows[topic.rsplit("/", 1)[1]].append((stamp, f.x, f.y, f.z, t.x, t.y, t.z))
    result = {}
    for side, values in rows.items():
        columns = [f"{side}_stamp_ns", *[f"{side}_{axis}" for axis in ("fx", "fy", "fz", "tx", "ty", "tz")]]
        frame = pd.DataFrame(values, columns=columns).sort_values(columns[0])
        assert not frame[columns[0]].duplicated().any()
        # Match the capture node's float32 conversion, before sign transformation.
        frame[columns[1:]] = frame[columns[1:]].astype(np.float32)
        result[side] = frame
    return result


def exact_pairs(stamps, streams, signs):
    paired = stamps[["left_stamp_ns", "right_stamp_ns"]].copy()
    for side in ("left", "right"):
        paired = paired.merge(streams[side], on=f"{side}_stamp_ns", how="left", validate="many_to_one")
    assert not paired[CHANNELS].isna().any().any()
    paired[CHANNELS] = paired[CHANNELS].to_numpy() * signs
    paired["stamp_ns"] = (paired.left_stamp_ns + paired.right_stamp_ns) // 2
    paired["pair_skew_ms"] = abs(paired.left_stamp_ns - paired.right_stamp_ns) / 1e6
    assert paired.pair_skew_ms.max() <= 15
    return paired


def negative_run(frame, column="sum_fz_n", count=5):
    after = frame.loc[frame.time_from_estimated_contact_s >= 0].reset_index(drop=True)
    hits = np.convolve((after[column].to_numpy() < 0).astype(int), np.ones(count, int), "valid")
    indexes = np.flatnonzero(hits == count)
    if not len(indexes):
        return None
    idx = int(indexes[0])
    return dict(onset_s=float(after.time_from_estimated_contact_s.iloc[idx]),
                confirmation_s=float(after.time_from_estimated_contact_s.iloc[idx + count - 1]))


def extract(spec):
    key, label, relative = spec
    episode = SOURCE / relative
    evaluation_path = episode / "fz_baseline_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    contact = evaluation["floor_contact_stamp_ns"]
    assert evaluation["enter_key_stamp_ns"] - contact == 300_000_000
    halt = (evaluation["halt_stamp_ns"] - contact) / 1e9
    assert LIMITS[1] < halt
    databases = list((episode / "bag").glob("*/*.db3"))
    assert len(databases) == 1
    streams = read_forces(databases[0])
    signs = np.asarray(evaluation["detector"]["coordinate_sign"])
    csv_dir = episode / "csv"
    calibration_path = csv_dir / "fz_baseline_calibration.csv"
    recorded_calibration = pd.read_csv(calibration_path)
    calibration = exact_pairs(recorded_calibration, streams, signs)
    offsets = calibration[CHANNELS].median()
    for side in ("left", "right"):
        np.testing.assert_allclose(offsets[f"{side}_fz"], evaluation["calibration"][f"{side}_offset_fz_n"], atol=1e-7)
    # Keep the capture node's exact pairing after contact (independently checked
    # against raw bag values). Before contact use nearest unique native samples.
    recorded_path = csv_dir / "contact_to_detection_force.csv"
    recorded = pd.read_csv(recorded_path)
    post = exact_pairs(recorded, streams, signs)
    np.testing.assert_allclose(post[CHANNELS], recorded[CHANNELS], atol=1e-7)
    left = streams["left"]
    pre_left = left.loc[left.left_stamp_ns.between(contact - 400_000_000, int(post.left_stamp_ns.min()) - 1)]
    nearest = pd.merge_asof(pre_left[["left_stamp_ns"]], streams["right"][["right_stamp_ns"]],
                            left_on="left_stamp_ns", right_on="right_stamp_ns",
                            direction="nearest", tolerance=15_000_000)
    assert not nearest.right_stamp_ns.isna().any()
    assert not nearest.right_stamp_ns.duplicated().any()
    assert nearest.right_stamp_ns.max() < post.right_stamp_ns.min()
    pre = exact_pairs(nearest, streams, signs)
    all_pairs = pd.concat([pre, post], ignore_index=True).sort_values("stamp_ns")
    assert not all_pairs.stamp_ns.duplicated().any()
    all_pairs["time_from_estimated_contact_s"] = (all_pairs.stamp_ns - contact) / 1e9
    for column in CHANNELS:
        all_pairs[f"{column}_untared"] = all_pairs[column]
        all_pairs[column] -= offsets[column]
    all_pairs["sum_fz_n"] = all_pairs.left_fz + all_pairs.right_fz
    window = all_pairs.loc[all_pairs.time_from_estimated_contact_s.between(*LIMITS)].copy()
    assert len(window) >= 75 and len(window) <= 85
    assert window.stamp_ns.diff().dropna().max() < 20_000_000
    assert window[["left_stamp_ns", "right_stamp_ns"]].max().max() < evaluation["halt_stamp_ns"]
    score_path = csv_dir / "fz_baseline_scores.csv"
    scores = pd.read_csv(score_path)
    reference = scores.merge(all_pairs[["left_stamp_ns", "right_stamp_ns", "sum_fz_n"]],
                             on=["left_stamp_ns", "right_stamp_ns"])
    assert len(reference) >= 10
    np.testing.assert_allclose(reference.corrected_signed_fz_n, reference.sum_fz_n, atol=2e-6)
    pre_values = all_pairs.loc[all_pairs.time_from_estimated_contact_s.between(-.30, -.05), "sum_fz_n"]
    post_values = all_pairs.loc[all_pairs.time_from_estimated_contact_s.between(0, .40), "sum_fz_n"]
    info = dict(id=key, label=label, episode=str(episode), resolved_episode=str(episode.resolve()),
                object="rectangle (wooden rectangular object)",
                contact_stamp_ns=contact, contact_definition="operator Enter minus 300 ms; estimated, not sensor ground truth",
                halt_command_after_estimated_contact_s=halt, calibration=evaluation["calibration"],
                calibration_offsets_all_channels=offsets.to_dict(), coordinate_sign=signs.tolist(),
                samples_in_window=len(window), pair_skew_max_ms=float(window.pair_skew_ms.max()),
                precontact_sum_fz_mean_n=float(pre_values.mean()), post_0_to_0p4_sum_fz_mean_n=float(post_values.mean()),
                five_samples_negative=negative_run(all_pairs),
                last_observation_after_estimated_contact_s=float(all_pairs.time_from_estimated_contact_s.max()),
                source_files={str(p): sha256(p) for p in [evaluation_path, calibration_path, recorded_path, score_path, episode / "config.yaml"]},
                bag=dict(path=str(databases[0]), bytes=databases[0].stat().st_size),
                post_contact_source_matches_bag=True, baseline_score_matches=len(reference))
    return window, all_pairs, calibration, info


def draw(frames, metadata, columns, destination):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "real-support-20260915", "path.simplify": False}):
        fig, axes = plt.subplots(len(columns), 1, figsize=(3.5, 2.55 if len(columns) == 1 else 5.2),
                                 squeeze=False, sharex=True)
        axes = axes[:, 0]
        fig.subplots_adjust(left=.25, right=.98, bottom=.23 if len(columns) == 1 else .12,
                            top=.75 if len(columns) == 1 else .87, hspace=.12)
        for axis, (column, ylabel) in zip(axes, columns):
            for frame, info, color in zip(frames, metadata, COLORS):
                axis.plot(frame.time_from_estimated_contact_s, frame[column], color=color, lw=1.2,
                          linestyle="-", label=info["label"], gid=f'curve_{info["id"]}_{column}')
            axis.axvline(0, color=".65", lw=.7, linestyle="-", gid="estimated_contact")
            axis.axhline(0, color=".80", lw=.6, linestyle="-", zorder=0, gid="zero_force")
            axis.set_xlim(*LIMITS)
            axis.set_xticks([-.3, 0, .3])
            axis.set_ylabel(ylabel, labelpad=3)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color=".92", linewidth=.5)
        axes[0].annotate("Estimated contact", xy=(0, 1), xycoords=("data", "axes fraction"),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                         fontsize=9, color=".4")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", frameon=False,
                   ncol=2, handlelength=1.7, columnspacing=1.3, bbox_to_anchor=(.55, 1.0))
        axes[-1].set_xlabel("Time from estimated contact (s)", labelpad=3)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = [*fig.texts, *fig.legends]
        for axis in axes:
            artists.extend([axis.xaxis.label, axis.yaxis.label, *axis.texts])
            artists.extend(label for value, label in zip(axis.get_xticks(), axis.get_xticklabels())
                           if axis.get_xlim()[0] <= value <= axis.get_xlim()[1])
            artists.extend(label for value, label in zip(axis.get_yticks(), axis.get_yticklabels())
                           if axis.get_ylim()[0] <= value <= axis.get_ylim()[1])
        for artist in artists:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (artist, box)
            assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (artist, box)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(destination.with_suffix(f".{extension}"), dpi=300)
        plt.close(fig)


def export_camera_review(information):
    destination = OUT / "camera_review"
    destination.mkdir()
    records = []
    for info in information:
        contact = info["contact_stamp_ns"]
        with sqlite3.connect(f'file:{info["bag"]["path"]}?mode=ro', uri=True) as connection:
            query = """SELECT messages.data FROM messages JOIN topics ON topics.id=messages.topic_id
                WHERE topics.name='/camera_fixed/realsense2_camera/color/image_raw/compressed'
                AND messages.timestamp BETWEEN ? AND ? ORDER BY messages.timestamp"""
            frames = []
            for (payload,) in connection.execute(query, (contact - 2_000_000_000, contact + 2_000_000_000)):
                message = deserialize_message(payload, CompressedImage)
                stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                frames.append((stamp, bytes(message.data)))
        for target in ([-1., 0., .4] if info["id"] == "fixed" else [-1., 0., .4, .78]):
            stamp, payload = min(frames, key=lambda x: abs(x[0] - (contact + int(target * 1e9))))
            filename = f'{info["id"]}_{target:+.2f}s.jpg'
            (destination / filename).write_bytes(payload)
            records.append(dict(file=filename, episode=info["episode"], stamp_ns=stamp,
                                time_from_estimated_contact_s=(stamp-contact)/1e9,
                                target_relative_s=target, original_compressed_bytes=True))
    (destination / "metadata.json").write_text(json.dumps(records, indent=2) + "\n")


README = """# 実機：固定面／wobble面の接触前後の力

surface_fz_sum.pdf / SVG：推奨の左右合計Fzの比較。surface_left.pdf は左Fx/Fy/Fz、
axes/ は単独軸、surface_fz_components.pdf は左右それぞれと合計の確認用。
CSVには元時刻、左右時刻差、12chの空荷補正後の値と補正前の値を保存。
metadata.jsonに試行、校正値、元データのハッシュと検証結果を記録。

同日の矩形木片のFz-baseline試行から、fixed episode 3、wobble episode 5を選択。
Fixedの持続的Fz反転時刻はコホート中央値付近、wobbleは接触前荷重が近い試行。
主図は各試行の推定接触時刻に揃えた−0.35〜+0.45秒。両試行の停止・開指指令より前。
これは例示用の0.8秒区間であり、保存されたモデル入力窓そのものではない。

接触時刻は操作者のEnter−300 msという推定値。物理接触の厳密な真値ではないため、
試行間の接触遅延を数十ms精度で定量比較する根拠にはしない。
力は記録された座標変換後、把持前・空グリッパ時の約1秒の中央値を各chで引いたもの。
Fzの中央値はbaseline制御で実際に使用した値と一致。既存動画のreference-empty補正とは異なる。
左右のFz分担が異なるため、荷重移行の比較には左右合計を推奨する。
グリッパの把持力や慣性等も含むセンサ測定であり、床反力そのもの／正確なmgとは言わない。

接触後は保存された100Hz力CSVの左右ペアをそのまま使用し、bagと値を全数照合。
接触前だけはbagの左サンプルに時刻最近傍の右サンプルを組み合わせる。
右サンプルの重複、15ms超の時刻ずれ、20ms以上のサンプル間隔がないことを確認。
平滑化・補間・振幅の正規化・波形に合わせた時間ずらしはしない。線はすべて実線。
負値5サンプル連続は診断用であり、実機制御の20Hz×3回規則や学習ラベルの変更ではない。

固定面は正から負に移るがwobbleは観測中に反転しない。
これは荷重移行の違いの実機例であって、wobbleの反力発生が遅れるという証明ではない。
板の角度や真の接触を力だけから断定できず、同じ質量・実軌道を完全に統制した比較でもない。
同じFz-baselineプロトコルだが、historical motion設定全体と実下降速度の一致は未検証。
各条件の異なる停止時刻以降は主図に含めない。

## 映像・コホートの確認

camera_review/ はbagの未加工JPEG。wobbleは約−1秒に隙間が見え、推定接触付近以降は
物体と板が接し、約+0.4秒に板の縁が上がる様子が見える。解像度・遮蔽の制限があるため、
これだけで最初の接触時刻をフレーム精度で確定しない。
real_surface_cohort_20260915.json は同プロトコルの固定20試行、wobble11試行の監査。
固定20/20で負値への移行、wobble0/11で観測中の負値なし。後者は観測打ち切りを伴う。
選定ペアだけに依存する傾向ではないが、一般的なwobble面すべてに成立すると主張しない。
コホート監査の接触前平均は保存済み約20Hzスコアを用い、主図metadataの接触前平均は
bagの約100Hzを用いるため数値が少し異なる。主図の平均はfixed約10.54N、wobble約10.44N。
主図は人手推定接触を基準とした力の推移を示し、厳密な物理接触からの遅延測定ではない。

実機元データ、学習データ、原稿は変更していない。
再生成はソーススクリプト内のOUTを新しい出力先にして実行（既存出力上書きを拒否）。
"""


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing assets: {OUT}")
    results = [extract(spec) for spec in SPECS]
    OUT.mkdir(parents=True)
    (OUT / "axes").mkdir()
    frames, information = [], []
    for frame, full, calibration, info in results:
        key = info["id"]
        frame.to_csv(OUT / f"{key}_window.csv", index=False)
        full.to_csv(OUT / f"{key}_precontact_to_halt.csv", index=False)
        calibration.to_csv(OUT / f"{key}_calibration.csv", index=False)
        frames.append(frame)
        information.append(info)
    draw(frames, information, [("sum_fz_n", r"Tared $F_{z,L}+F_{z,R}$ (N)")], OUT / "surface_fz_sum")
    draw(frames, information, [(f"left_f{axis}", rf"Left $F_{axis}$ (N)") for axis in "xyz"], OUT / "surface_left")
    draw(frames, information, [("left_fz", r"Left $F_z$ (N)"), ("right_fz", r"Right $F_z$ (N)"),
                              ("sum_fz_n", r"$F_{z,L}+F_{z,R}$ (N)")], OUT / "surface_fz_components")
    for axis in "xyz":
        draw(frames, information, [(f"left_f{axis}", rf"Left $F_{axis}$ (N)")], OUT / "axes" / f"F{axis}")
    manifest = dict(generator=str(Path(__file__)), generator_sha256=sha256(Path(__file__)),
                    window_limits_s=LIMITS, examples=information,
                    source_data_read_only=True, interpolation=False, smoothing=False)
    (OUT / "metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(README)
    export_camera_review(information)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
