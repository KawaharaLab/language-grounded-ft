#!/usr/bin/env python3
"""Package the published six F/T windows, editable vector panels and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from replot_semantic_ft_bundle import render_bundle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fig/semantic_ft_examples"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def export(source, bundle):
    # Never overwrite a bundle that may already contain the user's manual edits.
    if bundle.exists() or bundle.with_suffix(".zip").exists():
        raise FileExistsError(f"Choose a new --output-dir; preserving existing assets: {bundle}")
    original = json.loads((source / "manifest.json").read_text())
    assert sha256(Path(original["dataset"])) == original["dataset_sha256"]
    assert sha256(source / "selection.json") == original["selection_sha256"]
    columns = original["selection"]["publication"]["columns"]
    panels = {p["id"]: p for p in original["selection"]["panels"]}
    verified = {}
    for key in original["publication_example_ids"]:
        info = original["examples"][key]
        raw_path, meta_path = Path(info["csv_path"]), Path(info["metadata_path"])
        assert sha256(raw_path) == info["source_sha256"]
        assert sha256(meta_path) == info["metadata_sha256"]
        raw = pd.read_csv(raw_path)
        frame = pd.read_csv(source / info["exported_trace"])
        expected = raw.iloc[info["start"]:info["start"] + 80].reset_index(drop=True)
        assert len(frame) == len(expected) == 80
        np.testing.assert_allclose(frame[raw.columns], expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(frame.window_time_s, np.arange(80) / 100, atol=1e-12)
        verified[key] = (info, frame)
    bundle.mkdir(parents=True)
    sources_dir = bundle / "source_episodes"
    sources_dir.mkdir()
    shutil.copy2(source / "semantic_ft_examples.pdf", bundle / "combined_reference.pdf")
    shutil.copy2(source / "semantic_ft_examples.png", bundle / "combined_reference.png")
    shutil.copy2(source / "manifest.json", bundle / "original_figure_manifest.json")
    shutil.copy2(source / "selection.json", bundle / "original_selection.json")
    shutil.copy2(Path(__file__).with_name("replot_semantic_ft_bundle.py"), bundle / "replot.py")
    source_copies, summary_rows, items = {}, [], []
    for column in columns:
        panel_id = column["id"]
        item_dir = bundle / panel_id
        item_dir.mkdir()
        records, pair = [], {"window_time_s": np.arange(80) / 100}
        for spec, display_label in zip(panels[panel_id]["examples"], column["labels"]):
            info, frame = verified[spec["id"]]
            key = info["id"]
            window_name = f"{key}_window.csv"
            shutil.copy2(source / info["exported_trace"], item_dir / window_name)
            for channel in "xyz":
                pair[f"{key}_left_f{channel}_N"] = frame[f"left_f{channel}"].to_numpy()
            if info["csv_path"] not in source_copies:
                raw_copy = sources_dir / f"{key}_episode.csv"
                meta_copy = sources_dir / f"{key}_episode_metadata.json"
                shutil.copy2(info["csv_path"], raw_copy)
                shutil.copy2(info["metadata_path"], meta_copy)
                source_copies[info["csv_path"]] = (raw_copy, meta_copy)
            raw_copy, meta_copy = source_copies[info["csv_path"]]
            record = dict(info, display_label=display_label, window_csv=window_name,
                          bundled_episode_csv=f"../source_episodes/{raw_copy.name}",
                          bundled_episode_metadata=f"../source_episodes/{meta_copy.name}",
                          window_sha256=sha256(item_dir / window_name))
            records.append(record)
            summary_rows.append({
                "item": panel_id, "example_id": key, "display_label": display_label,
                "object": info["object"], "run": info["run"], "action": info["action"],
                "interaction": info["interaction"], "mass_g": info["mass_g"],
                "slip_proxy_mm_s": info["slip_proxy_mm_s"],
                "source_row_start_zero_based": info["start"],
                "source_row_stop_exclusive": info["start"] + 80,
                "simulation_step_first": info["source_simulation_steps_inclusive"][0],
                "simulation_step_last": info["source_simulation_steps_inclusive"][1],
                "first_support_contact_s": info["first_support_contact_s"],
                "annotation": info["annotation"],
                "window_csv": f"{panel_id}/{window_name}",
                "bundled_episode_csv": f"source_episodes/{raw_copy.name}",
                "original_episode_csv": info["csv_path"],
            })
        pd.DataFrame(pair).to_csv(item_dir / "plot_data.csv", index=False, float_format="%.17g")
        metadata = {"id": panel_id, "title": column["title"],
                    "selection_rationale": panels[panel_id]["selection_rationale"],
                    "examples": records}
        write_json(item_dir / "metadata.json", metadata)
        info_lines = [f'# {column["title"]}', "", "図：同名のPDF/SVG。再作図用：plot_data.csv。", ""]
        for record in records:
            first, last = record["source_rows_zero_based_inclusive"]
            steps = record["source_simulation_steps_inclusive"]
            info_lines.extend([
                f'## {record["display_label"]}', "",
                f'- 物体ID：`{record["object"]}`', f'- 試行ディレクトリ：`{record["run"]}`',
                f'- 元データの0始まり行：{first}〜{last}（`iloc[{first}:{last + 1}]`）',
                f'- シミュレーションstep：{steps[0]}〜{steps[1]}',
                f'- 質量：{record["mass_g"]:.6f} g',
                f'- action / interaction：{record["action"]} / {record["interaction"]}',
                f'- 窓単位の滑り指標：{record["slip_proxy_mm_s"]:.6f} mm/s',
                f'- アノテーション：{record["annotation"]}',
                f'- [80点のCSV]({record["window_csv"]})',
                f'- [元episodeの全CSV]({record["bundled_episode_csv"]})',
                f'- [元episodeのmetadata]({record["bundled_episode_metadata"]})', "",
            ])
        (item_dir / "source_info.md").write_text("\n".join(info_lines))
        items.append({"id": panel_id, "directory": panel_id})
    pd.DataFrame(summary_rows).to_csv(bundle / "sources.csv", index=False)
    write_json(bundle / "bundle_manifest.json", {
        "items": items, "source_figure_directory": str(source),
        "original_dataset": original["dataset"], "dataset_sha256": original["dataset_sha256"],
        "examples": original["publication_example_ids"], "sampling_hz": 100,
        "samples_per_window": 80, "plot_columns": ["left_fx", "left_fy", "left_fz"],
        "force_unit": "N", "coordinate_frame": "left-finger sensor local",
    })
    (bundle / "README.md").write_text(README)
    archive = finalize_bundle(bundle)
    print(json.dumps({"bundle": str(bundle), "zip": str(archive),
                      "items": len(items), "windows": len(summary_rows),
                      "original_episodes": len(source_copies)}, indent=2))


def finalize_bundle(bundle):
    archive = bundle.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Preserving existing archive: {archive}")
    render_bundle(bundle)
    checksums = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256(path)}  {path.relative_to(bundle).as_posix()}")
    (bundle / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=f"{bundle.name}/{path.relative_to(bundle)}")
    return archive


README = """# 力履歴図：編集用パッケージ

現在の3列図を、Weight / Interaction / Support surfaceごとに分割した。
合成図・個別図とも曲線はすべて実線で、比較対象は色分けする。論文原稿・学習データは変更していない。

## 最初に開くファイル

| 項目 | 編集用PDF | 編集用SVG | 出典 |
| --- | --- | --- | --- |
| 重量 | [weight.pdf](weight/weight.pdf) | [weight.svg](weight/weight.svg) | [出典](weight/source_info.md) |
| 滑り | [interaction.pdf](interaction/interaction.pdf) | [interaction.svg](interaction/interaction.svg) | [出典](interaction/source_info.md) |
| 支持面 | [surface.pdf](surface/surface.pdf) | [surface.svg](surface/surface.svg) | [出典](surface/source_info.md) |

各項目のPNGはプレビュー。PDF/SVGの曲線はベクトルで、SVGの文字はtext要素として保存した。
フォントはDejaVu SansとSTIX。編集環境に無い場合はフォント置換が必要になる場合がある。
Illustrator自体での読み込み検証は行っていない。数学添字の文字は別々のtext/tspanになる場合がある。
SVGでは曲線に `curve_weight_light_left_fx` などのIDを付け、軸・凡例もグループ名を付けている。
`axes/Fx.pdf(svg)`, `Fy`, `Fz` は各成分をさらに個別にした部品。タイトル・凡例を省き、同じ線種・色を使用する。
個別図には全て縦軸ラベルを付け、元の図と同じ生値・縦軸範囲を維持する。

## CSVの使い分け

- 各項目の `plot_data.csv`：80行×7列。先頭がwindow_time_s、続いて2条件それぞれの左Fx/Fy/Fz [N]。表計算ソフトなどでの再作図用。
- 各項目の `*_window.csv`：表示した80点の全元列＋明示的な診断列。右指・トルク・姿勢・接触情報も残す。
- `source_episodes/`：元episodeの全CSVと元JSONをそのままコピー。6窓のうち滑り比較2窓は同一episodeなので、元episodeは5本。
- `sources.csv`：物体ID・試行名・行位置・質量・アノテーション・ファイルパスの6行一覧。
- 各項目の `metadata.json`：元パス、SHA-256、接触数、滑り指標、初接地時刻などを含む詳細。

## 時刻・単位・注意点

- 全例100 Hz、80点。横軸は窓内相対時刻0.00〜0.79 s。0.8 s位置に81点目を追加していない。
- 元行番号はヘッダーを除いた0始まり。例えば560:640は560〜639行目、通常のCSVテキスト行では562〜641行目に相当する。
- 元CSVの `step` と行番号は別の量として記録する。両者はこの6窓では1ずれている。
- 力 `left_f*`, `right_f*` はN、トルク `left_t*`, `right_t*` はN m。元CSVのobj_massはkg、出典表のmass_gはg。
- グラフは左指センサ局所座標の生の力3成分。平滑化・左右和・窓ごとの正規化・時間位置合わせは行わない。
- 同じ比較内の2本は縦軸を共有する。別の比較間は縦軸範囲が異なる。
- Support surfaceの灰色マーカーは実際の初接地時刻。2試行で一致する場合は1本の縦線、一致しない場合は両時刻の間の帯であり、接触継続時間ではない。各時刻はmetadata.jsonとsources.csvに記録する。
- 滑り速度は窓内最大接線相対変位÷0.79 sというアノテーション用指標。瞬間速度ではない。
- window CSV中のdiagnostic_*やrelative_tangential_excursion_mm、物理状態列は出典確認用。モデルの入力は左右F/Tの12成分のみ。
- Weightの2例は窓内で下降速度が変わる。Surfaceの可動面は主に一方向に傾く例で、周期振動を実証するものではない。
- 手動選択の具体例であり、全データの代表性、因果関係、モデルが使った特徴をこの図だけで主張しない。

## 再作図

同梱のreplot.pyはこのフォルダ内のCSVだけを読み、元のGenesis環境は不要。
Pythonとnumpy/pandas/matplotlibを使用する。作成環境のMatplotlibは3.10.6。

```bash
python replot.py
```

PDF/SVG/PNGは上書きされるので、手修正後のファイルは別名で保存してから実行する。
CSVと原稿は変更しない。手修正・再作図後はSHA256SUMS.txtと一致しなくなるため、これは納品時点の検証値として扱う。
`original_figure_manifest.json`と`original_selection.json`は来歴記録（過去のAction補助例も記載）であり、このパッケージの掲載対象はbundle_manifest.jsonの6窓のみ。
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=SOURCE / "editing_bundle")
    args = parser.parse_args()
    export(args.source_dir, args.output_dir)
