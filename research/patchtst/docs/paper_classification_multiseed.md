# PAPER分類比較の3シード評価

対象は `${FT_WORKSPACE_ROOT}/Genesis/paper.tex` の `tab:model_comparison` に載せた
Token-Fusion PatchTST、CNN、MLP、Supervised Token-Fusion の4モデル。
seed 42 の既存学習を再利用し、seed 43・44を追加する（計8学習）。

## 実行

```bash
cd ${FT_WORKSPACE_ROOT}/PatchTST
GROUP=paper_classification_3seed_20260909 \
  GPUS="0 1" WANDB_MODE=online \
  bash scripts/run_paper_classification_multiseed.sh
```

同名のtmuxセッション内で実行する。GPU 0がseed 43、GPU 1がseed 44を担当し、
それぞれ Token-Fusion → Supervised Token-Fusion → CNN → MLP の順に学習・評価する。
実行済み学習には完了マーカーを保存し、同じGROUPで再実行すると完了済み処理をスキップする。
途中で中断した学習はそのモデル・seedの先頭から再実行する。

```bash
tmux attach -t paper_classification_3seed_20260909
tail -f results/paper_classification_multiseed/paper_classification_3seed_20260909/logs/paper_tokenfusion_seed43.train.log
```

## 比較条件

- 学習CSV: `train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv`（437,244窓、250種類のannotation）。
- 学習embedding: `bert_emb_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms`。
- seed 42 checkpointに保存された引数を再利用する。最大400 epoch、BS 512、LR 2e-3、warmup 10 epoch、cosine decay、early stopping patience 75。
- 対照学習は双方向InfoNCE。教師ありは `terrain0814_eval` のmulti-task分類、class weightはbalanced。
- 初期化、学習時のサンプル順序、train/validation分割の乱数を各seedで変える。正規化統計は各seedの学習用90%だけから計算する。
- best選択基準は既存実験を維持する：対照学習はvalidation loss、教師ありはvalidationのタスク平均Macro-F1。評価データでcheckpointやseedを選択しない。
- bestとlastの両方を評価し、原稿にはbestを使用する。教師ありのlastはファイル名が `_final.pth`。

評価集合はseed間で固定し、action 10,008例（各1,668）、weight 6,000例（各3,000）、
interaction 3,675例（各1,225）。actionではreleaseとrelease midairをreleaseに統合済み。
accidental-drop episodeとtransition windowのaction除外条件も原稿掲載時のCSVを維持する。

対照学習の分類とUMAPは同じ定型文embeddingを使用する。
releaseは `Releasing an object.` の1文であり、embedding平均を取らない。
今回の原稿条件ではstableは `The object has stable contact.` を使用する。
全定義を結果ディレクトリの `prepared_eval/prototype_templates.json` に保存する。

## 保存先と原稿反映

`results/paper_classification_multiseed/<GROUP>/` に以下を保存する。

- `manifest.json`: 元checkpoint、全コマンド、入力ハッシュ、seed、選択基準。
- `code/`: 実行に使用するPythonコードのコピー（実行中の編集によるseed間差を避ける）。
- `status.json` / `runner.log` / `logs/`: 実行状況と各学習・評価ログ。
- `evaluation/<model>/seed<seed>/<best|last>/`: UMAP、混同行列、予測CSV、各評価の一覧。
- `ARTIFACTS.md`: 全モデル・seed・checkpointの成果物リンク。
- `per_seed_metrics.csv`: 丸める前の予測CSVから再計算した各seedのAccuracy / Macro-F1。
- `summary.md` / `summary.json`: best/last別の3シード平均・標準偏差。
- `model_comparison_3seed.tex`: 原稿用の表。

標準偏差は標本標準偏差（`ddof=1`、分母 `n-1`）とする。
サンプルを3シード分まとめて評価せず、各seedのAccuracyとMacro-F1を算出してから平均する。
3シードすべての評価とUMAP・混同行列が揃うまでは、平均±標準偏差の表を原稿へ反映しない。

完了時に `${FT_WORKSPACE_ROOT}/Genesis/paper.tex` の対象表、学習条件の説明、当該結果の本文を自動更新する。
実行中にそれらの部分を編集した場合は自動置換を止め、生成したLaTeXを保存する。
それ以外の原稿の編集は維持する。更新前の原稿も結果ディレクトリへバックアップする。

集計・原稿反映だけを再実行する場合：

```bash
uv run python scripts/paper_classification_multiseed.py summarize \
  --group paper_classification_3seed_20260909
```
