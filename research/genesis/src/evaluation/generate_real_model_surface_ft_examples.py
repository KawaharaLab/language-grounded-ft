#!/usr/bin/env python3
"""Export real model-trigger windows in the simulation surface-panel format.

Source robot data is read-only. Plotted forces are pregrasp-tared (N), not
z-scored model inputs. Both representations are preserved in the CSV.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from generate_real_surface_ft_examples import CHANNELS, sha256
from replot_semantic_ft_bundle import plot_item

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("FT_WORKSPACE_ROOT", ROOT.parent)).expanduser().resolve()
ROBOT = Path(os.environ.get("FT_ROBOT_ROOT", WORKSPACE / "ur3_robotiq_ros2")).expanduser().resolve()
SIM = ROOT / "fig/semantic_ft_examples/surface_contact_review_20260915/candidate_surface"
sys.path.insert(0, str(ROBOT / "data_processing"))
from plot_model_place_inputs import read_force_streams, align_pairs_to_decision


def load_window(episode, key):
    evaluation_path = episode / "model_evaluation.json"
    ev = json.loads(evaluation_path.read_text())
    assert ev["status"] == "success" and ev["model_triggered"]
    # A queued positive result can raise the summary streak after the trigger.
    # The authoritative decision row below must still be the third result.
    assert ev["place_streak_at_halt"] >= 3
    decision = ev["decision"]
    contact = ev["floor_contact_stamp_ns"]
    assert ev["enter_key_stamp_ns"] - contact == 300_000_000
    scores_path = episode / "csv/model_scores.csv"
    scores = pd.read_csv(scores_path)
    trigger = scores.loc[scores.sequence == decision["sequence"]]
    assert len(trigger) == 1
    trigger = trigger.iloc[0]
    assert str(trigger.triggered).lower() == "true" and int(trigger.place_streak) == 3
    for column in ("window_start_stamp_ns", "window_end_stamp_ns"):
        assert int(trigger[column]) == decision[column]
    for column in ("hold_score", "place_score", "margin"):
        np.testing.assert_allclose(trigger[column], decision[column], atol=1e-9)
    normalization_path = Path(ev["detector"]["normalization_path"])
    norm = json.loads(normalization_path.read_text())
    assert norm["force_cols"] == CHANNELS
    signs = np.asarray([-1., -1., 1., -1., -1., 1.] * 2, dtype=np.float32)
    assert ev["detector"]["normalization_coordinate_transform"] == "sim_local_rz180"
    dbs = list((episode / "bag").glob("*/*.db3"))
    assert len(dbs) == 1
    streams = read_force_streams(dbs[0])
    recovered = align_pairs_to_decision(streams, signs, decision["window_end_stamp_ns"],
                                        decision["window_end_stamp_ns"], 80, 15.)[:80]
    assert len(recovered) == 80
    assert recovered[0].stamp_ns == decision["window_start_stamp_ns"], "Inexact start: reject candidate"
    assert recovered[-1].stamp_ns == decision["window_end_stamp_ns"], "Inexact end: reject candidate"
    frame = pd.DataFrame([dict(sequence=decision["sequence"] - 79 + i, stamp_ns=p.stamp_ns,
                               left_stamp_ns=p.left_stamp_ns, right_stamp_ns=p.right_stamp_ns,
                               pair_skew_ms=p.skew_ms, **dict(zip(CHANNELS, p.values)))
                          for i, p in enumerate(recovered)])
    frame["window_time_s"] = (frame.stamp_ns - frame.stamp_ns.iloc[0]) / 1e9
    frame["time_from_estimated_contact_s"] = (frame.stamp_ns - contact) / 1e9
    anchors = scores.loc[scores.sequence.between(frame.sequence.min(), frame.sequence.max())]
    anchor_matches = anchors.merge(frame[["sequence", "stamp_ns"]], on="sequence", validate="one_to_one")
    assert len(anchor_matches) >= 15
    np.testing.assert_array_equal(anchor_matches.window_end_stamp_ns, anchor_matches.stamp_ns)
    assert frame.window_time_s.is_monotonic_increasing
    assert frame.window_time_s.diff().dropna().max() < .02
    assert frame[["left_stamp_ns", "right_stamp_ns"]].max().max() < ev["halt_stamp_ns"]
    recorded_path = episode / "csv/contact_to_detection_force.csv"
    recorded = pd.read_csv(recorded_path)
    recorded = recorded.loc[recorded.sequence.between(frame.sequence.min(), frame.sequence.max())]
    overlap = frame.merge(recorded, on="sequence", suffixes=("", "_recorded"), validate="one_to_one")
    assert len(overlap) > 0 and len(overlap) == len(recorded)
    for column in ("stamp_ns", "left_stamp_ns", "right_stamp_ns"):
        np.testing.assert_array_equal(overlap[column], overlap[f"{column}_recorded"])
    for column in CHANNELS:
        np.testing.assert_allclose(overlap[column], overlap[f"{column}_recorded"], atol=1e-7)
    correction = ev.get("empty_offset_correction", {})
    adjustment = np.asarray(correction.get("adjustment", [0.] * 12), dtype=np.float32)
    if correction.get("enabled"):
        tare = np.asarray(correction["current"], dtype=np.float32)
        tare_info = dict(method="recorded pregrasp empty-force mean (current)",
                         start_ns=correction["window_start_stamp_ns"], end_ns=correction["window_end_stamp_ns"],
                         sample_count=correction["sample_count"])
    else:
        events = pd.read_csv(episode / "csv/key_events.csv")
        close = int(events.loc[events.key == "gripper_close_start", "stamp_ns"].iloc[0])
        values, counts = [], {}
        for side, sign in (("left", signs[:6]), ("right", signs[6:])):
            calibration = [p.values for p in streams[side] if close - 1_000_000_000 <= p.stamp_ns < close]
            assert len(calibration) >= 90
            counts[side] = len(calibration)
            values.extend((np.asarray(calibration, dtype=np.float32) * sign).mean(axis=0))
        tare = np.asarray(values, dtype=np.float32)
        tare_info = dict(method="offline per-sensor mean in 1 s before gripper_close_start; display only",
                         start_ns=close - 1_000_000_000, end_ns=close, sample_counts=counts)
    raw = frame[CHANNELS].to_numpy(dtype=np.float32)
    corrected = raw + adjustment
    normalized = (corrected - np.asarray(norm["mean"], dtype=np.float32)) / np.asarray(norm["scale"], dtype=np.float32)
    for i, column in enumerate(CHANNELS):
        frame[f"transformed_{column}"] = raw[:, i]
        frame[f"model_corrected_{column}"] = corrected[:, i]
        frame[f"normalized_{column}"] = normalized[:, i]
        frame[column] = raw[:, i].astype(float) - float(tare[i])
    frame["sum_fz_n"] = frame.left_fz + frame.right_fz
    info = dict(id=key, display_label="Fixed flat surface" if key == "fixed" else "Wobbling surface",
                window_csv=f"{key}_window.csv", episode=str(episode), decision=decision,
                first_support_contact_s=(contact - decision["window_start_stamp_ns"]) / 1e9,
                contact_definition="estimated: operator Enter minus 300 ms",
                halt_time_in_window_s=(ev["halt_stamp_ns"]-decision["window_start_stamp_ns"]) / 1e9,
                model_success_definition="place_score > hold_score, three consecutive eligible results; NOT a surface-class prediction",
                tare=tare_info, tare_all_channels=tare.tolist(), model_input_adjustment=adjustment.tolist(),
                model_input_adjustment_enabled=bool(correction.get("enabled", False)),
                left_fz_peak_to_peak_n=float(np.ptp(frame.left_fz)),
                left_fz_first10_mean_n=float(frame.left_fz.iloc[:10].mean()),
                left_fz_last10_mean_n=float(frame.left_fz.iloc[-10:].mean()),
                authoritative_postcontact_pairs_checked=len(overlap), strict_start_end_ns_match=True,
                model_score_timestamp_anchors_checked=len(anchor_matches),
                pair_skew_max_ms=float(frame.pair_skew_ms.max()),
                max_sample_interval_ms=float(frame.window_time_s.diff().max() * 1000),
                checkpoint=ev["detector"]["checkpoint_path"], detector=ev["detector"],
                source_hashes={str(p): sha256(p) for p in [evaluation_path, scores_path, recorded_path, normalization_path]},
                bag_path=str(dbs[0]))
    return frame, info


README = """# 成功したモデル判定窓：実機支持面比較

surface.pdf/png/svg は左Fx・Fy・Fzの3段。シミュレーションcandidate_surfaceと
同じ2.65×3.7inch、字体、文字サイズ、色、実線、0〜0.8秒の横軸に揃えている。
simulation_reference.* は比較元の変更なしコピー。axes/ に各軸単体。
wobble_detail/ は同じwobble窓の単独拡大図（縦軸範囲は比較図と異なる、単位はNのまま）。

実機の「成功」はhold/placeの対比でplace優勢が3回続いて停止した試行であり、
支持面クラスwobbleの正解率や、Fzだけに基づく判断を意味しない。
各CSVの80点は最後の成功判定（停止を発火させた3回目）の窓をbagから復元。
開始・終了時刻はログとns単位で一致。接触後のペアは保存済みsequence・左右時刻・
12ch値と全数照合。停止／開指後のデータを加えていない。
推定接触時刻はEnter−300ms。灰帯は二試行の窓内推定時刻の間であり、
接触持続時間や推定誤差の信頼区間ではない。波形を接触に合わせて移動していない。

図は把持前の空荷値を引いたN単位の力。wobbleは記録済みcurrent平均、fixedで
オンライン空荷補正がない場合は閉指直前1秒から表示用に平均を算出。
この一定値の減算は波形の変化量を変えない。振幅正規化・補間・平滑化なし。
CSVには図の値のほか、変換後センサ値、当時の空荷補正後のモデル入力、正規化後入力を
それぞれtransformed_*, model_corrected_*, normalized_*として区別して保存。
接頭辞のないleft_fxなどは図用の空荷補正済み値であり、生センサ値ではない。
モデルの正規化・補正は当時の記録を用い、図のtareとは混同しない。

この例は変化が比較的大きい成功窓を意図的に選んだ定性的例示であり、無作為抽出ではない。
成功したwobbleモデル試行21件を調べ、元の左右端点から80点を一意に復元できる16件を比較。
残る5件は区間内に81〜82サンプルあり、オンラインで落ちたサンプルの追加特定なしでは
厳密な80点として扱えないため、ランキングから除外した。
今回のwobbleは20260827_230524_074937/episode_000005、sequence 2008。
左Fzの最大−最小は0.841N、最終10点平均−最初10点平均は−0.392N。
固定面は同checkpointの20260827_153945_906240/episode_000004、sequence 2004。
実機のwobble成功窓にはシミュレーション例のような数N規模の急なFz反転は見つかっていない。
小さなFz変動を確認できても、モデルがFzの変動を根拠に成功したと因果的に主張しない。
モデルは左右の6軸、計12chを入力しており、この図はそのうち左の3力成分のみ。
実機・シミュレーションで物体、把持、支持機構が異なるため、絶対値・反転時刻の一致も主張しない。

## 成功判定の独立再現

real_model_surface_replay_20260915.json にCPUでの独立再現とcheckpoint・正規化・
言語プロトタイプのSHA-256を記録。固定面のplace−holdは記録0.034726／再現0.034640、
wobbleは記録0.046513／再現0.046526で、どちらも同じ正判定。
個々のcosineの差は最大0.000245（CPU/GPU再計算差を含む）。
左右の端点位相を固定して別途復元した80×12の変換後・補正後・正規化後行列は、
同梱CSVをfloat32へ戻した値と6/6行列でビット単位一致。
元の成功ログだけでなく、図に使った窓と入力処理の対応も確認している。

元実機データ、以前のbaseline図、シミュレーション図、原稿には変更していない。
replot_semantic_ft_bundle.py のplot_item関数は同梱CSVからROSなしでも再作図可能。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument("--wobble", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "fig/real_model_surface_ft_examples_20260915")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    results = [load_window(args.fixed, "fixed"), load_window(args.wobble, "wobble")]
    assert results[0][1]["checkpoint"] == results[1][1]["checkpoint"]
    args.out.mkdir(parents=True)
    metadata = dict(id="surface", title="(d) Real robot", contact_label="Est. contact",
                    examples=[info for _, info in results], generator=str(Path(__file__)),
                    generator_sha256=sha256(Path(__file__)), plotter_sha256=sha256(Path(plot_item.__code__.co_filename)))
    for frame, info in results:
        frame.to_csv(args.out / info["window_csv"], index=False)
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (args.out / "README.md").write_text(README)
    plot_item(args.out, metadata)
    (args.out / "axes").mkdir()
    for channel in "xyz":
        plot_item(args.out, metadata, channels=(channel,), destination=args.out / "axes" / f"F{channel}")
    detail = args.out / "wobble_detail"
    detail.mkdir()
    shutil.copy2(args.out / "wobble_window.csv", detail / "wobble_window.csv")
    detail_info = dict(id="surface", title="Wobbling (real)", contact_label="Est. contact",
                       colors=["#D55E00"], examples=[results[1][1]])
    (detail / "metadata.json").write_text(json.dumps(detail_info, indent=2) + "\n")
    plot_item(detail, detail_info)
    for extension in ("pdf", "png", "svg"):
        shutil.copy2(SIM / f"surface.{extension}", args.out / f"simulation_reference.{extension}")
    shutil.copy2(Path(plot_item.__code__.co_filename), args.out / "replot_semantic_ft_bundle.py")
    shutil.copy2(__file__, args.out / Path(__file__).name)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
