# Terrain学習・評価手順

この文書は、`train_08142026`のterrain annotationを用いてToken-Fusion PatchTST、CNN、MLPを対照学習し、`eval_08142026`で評価する際の標準手順をまとめたものである。

## 評価規約

対照学習モデルの評価では、特別な指定がない限り次を必ず守る。

- クラスprototypeには単語ではなく、下記の定型文のS-BERT embeddingを使う。
- 分類、UMAP上のtext marker、混同行列で同じ定型文embeddingを使う。
- `release`と`release midair`は1クラスの`release`へ統合する。
- release prototypeは`Releasing an object.`の1文だけとし、embedding平均は行わない。
- action、weight、interactionはそれぞれ独立に均衡化した評価CSVで評価する。
- best checkpointとlast checkpointの両方を評価する。
- UMAP、混同行列、サンプル別予測CSV、Accuracy / Macro-F1一覧を保存する。
- `terrain0814_release_merged_*`は旧単語prototype評価なので使用しない。`terrain0814_templates_release_merged_*`を使用する。

標準prototypeは以下である。

| Target | Class | Prototype |
| --- | --- | --- |
| Action | grasp | `Grasping an object.` |
| Action | hold | `Holding an object.` |
| Action | lift | `Lifting an object.` |
| Action | place | `Placing an object.` |
| Action | press | `Pressing an object.` |
| Action | release | `Releasing an object.` |
| Weight | heavy | `The object is heavy.` |
| Weight | light | `The object is light.` |
| Interaction | accidental drop | `The object was accidentally dropped.` |
| Interaction | slip | `The object is slipping.` |
| Interaction | stable | `The object is stable.` |

定義元は`results/template_prototypes_eval04272026/prototypes/templates.json`である。releaseだけは上記の文へ明示的に差し替えられ、実行時の全prototypeは各runの`prepared_eval_templates_release_merged/prototype_templates.json`へ保存される。

## 前提データ

標準設定では以下を使用する。

- 学習ディレクトリ: `${FT_WORKSPACE_ROOT}/Genesis/data/train_08142026`
- 学習CSV: `train_08142026_terrain_balanced.csv`
- 学習text embedding: `bert_emb_terrain`
- 評価ディレクトリ: `${FT_WORKSPACE_ROOT}/Genesis/data/eval_08142026`
- 評価CSV: `eval_08142026_terrain_{action,weight,interaction}_balanced.csv`

学習CSVを変更した場合は、`csv_path`、`start`、`annotation`、`emb_index`が存在し、`emb_index`と`bert_emb_terrain/*.pt`が対応していることを確認する。annotation自体を変更した場合は、学習前にGenesis側でembeddingを再生成する。

```bash
cd ${FT_WORKSPACE_ROOT}/Genesis

DATA_TYPE=train_08142026 \
TRAIN_CSV=train_08142026_terrain_balanced.csv \
EMB_DIR=bert_emb_terrain \
MAPPING_NAME=annotation_index_map_terrain.csv \
.venv/bin/python src/processing/rigid/3_bert_train.py
```

## 標準学習

`scripts/run_terrain_contrastive_0814.sh`はtmux sessionを作り、GPU 0でToken-Fusion、GPU 1でCNNとMLPを順番に学習する。学習後はbest/lastを自動評価する。

```bash
cd ${FT_WORKSPACE_ROOT}/PatchTST

SESSION=terrain_my_experiment \
GROUP=terrain_contrastive_my_experiment_$(date +%Y%m%d_%H%M%S) \
EPOCHS=400 \
BATCH_SIZE=512 \
PEAK_LR=2e-3 \
SEED=42 \
bash scripts/run_terrain_contrastive_0814.sh
```

起動後は次で状態を確認する。

```bash
tmux attach -t terrain_my_experiment
```

`GROUP`は実験ごとに新しくする。同じ`GROUP`を再利用すると、既存checkpointや評価結果が`SKIP_EXISTING=1`によってskipされる。

## 条件の変更

環境変数で変更できる主な項目は次の通り。

```bash
SESSION=terrain_lr_test \
GROUP=terrain_lr_test_$(date +%Y%m%d_%H%M%S) \
EPOCHS=600 \
BATCH_SIZE=256 \
PEAK_LR=1e-3 \
MIN_LR=1e-5 \
WARMUP_EPOCHS=10 \
PATIENCE=75 \
SEED=43 \
TRAIN_CSV=train_08142026_terrain_balanced.csv \
TRAIN_EMB_DIR=bert_emb_terrain \
bash scripts/run_terrain_contrastive_0814.sh
```

次の変更は`scripts/run_terrain_contrastive_0814.sh`内で行う。

- PatchTSTサイズ、patch長、stride: `ARCH_ARGS`
- validationを含む共通学習条件とloss: `COMMON_ARGS`
- Token-Fusion固有条件: `tokenfusion)`のcommand
- CNN/MLP固有条件: `cnn)`、`mlp)`のcommand
- GPU割り当て: 末尾の`train_model 0 tokenfusion`と`train_model 1 ...`

annotation種類がバッチサイズより少ない場合、通常InfoNCEは同じannotationを持つサンプルをfalse negativeにする。loss比較では`--loss_type infonce_multi_positive_numerator`または`--loss_type infonce_multi_positive`を優先候補とする。

## 自動評価

標準スクリプトは学習後に次を行う。

1. `release midair`を`release`へ置換する。
2. 定型文prototype用のruntime directoryを作る。
3. action、weight、interactionの独立均衡CSVを結合する。
4. Token-Fusion、CNN、MLPのbest/lastを評価する。
5. Accuracy / Macro-F1一覧を`summary.md`へ保存する。

出力先は以下である。

```text
saved_models/terrain_contrastive/<GROUP>/
results/terrain_contrastive/<GROUP>/summary.md
results/terrain_contrastive/<GROUP>/prepared_eval_templates_release_merged/
logs/terrain_contrastive/<GROUP>/
data/terrain0814/terrain0814_templates_release_merged_<RUN>_<best|last>/
```

モデルごとの評価directoryには次が存在することを確認する。

```text
umap_metrics_summary.md
umap/action_umap.png
umap/weight_umap.png
umap/interaction_umap.png
heatmap/action_heatmap.png
heatmap/weight_heatmap.png
heatmap/interaction_heatmap.png
pred/action_pred.csv
pred/weight_pred.csv
pred/interaction_pred.csv
pred/*_pred_label_order.txt
```

`--skip_plots`を指定するとUMAPと混同行列が作られないため、通常の最終評価では指定しない。

## 既存checkpointの手動再評価

評価条件だけを変更する場合はcheckpointを再学習せず、`patchtst_eval_normalized.py`を直接使う。以下はToken-Fusion bestの例である。

```bash
cd ${FT_WORKSPACE_ROOT}/PatchTST

GROUP=terrain_contrastive_20260819_121436
RUN=terrain_tokenfusion_seed42
EVAL_RUNTIME=results/terrain_contrastive/${GROUP}/prepared_eval_templates_release_merged

CUDA_VISIBLE_DEVICES=0 uv run PatchTST_self_supervised/patchtst_eval_normalized.py \
  --model_path saved_models/terrain_contrastive/${GROUP}/${RUN}/${RUN}_best.pth \
  --ckpt_type contrastive \
  --model_arch patchtst \
  --arch auto \
  --pool mean \
  --patch_len 10 \
  --stride 10 \
  --n_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 768 \
  --head_dropout 0.2 \
  --token_fusion_layers 1 \
  --token_fusion_heads 8 \
  --token_fusion_ff_mult 4 \
  --out_embed_dim 768 \
  --eval_dir "${EVAL_RUNTIME}" \
  --eval_csv eval_08142026_terrain_independent_balanced.csv \
  --targets action,weight,interaction \
  --text_type bert \
  --zero_force_eps 1e-8 \
  --output_name terrain0814_templates_release_merged_${RUN}_best_rerun
```

CNNでは`--model_arch cnn`、MLPでは`--model_arch mlp`へ変更する。lastを評価するときはcheckpointと`output_name`の`best`を`last`へ変更する。既存結果を誤って上書きしないよう、再評価時の`output_name`には目的が分かるsuffixを付ける。

## Place地形識別の追加評価

平面、30度斜面、2.0 cm段差だけを比較するときは次を使う。

```bash
cd ${FT_WORKSPACE_ROOT}/PatchTST
MODEL_GROUP=<GROUP> SEED=42 bash scripts/evaluate_terrain_place_flat_30deg_2cm.sh
```

この評価でも各クラス1つの定型文を使用し、embedding平均は行わない。

- 平面: `Placing an object on a flat surface.`
- 30度斜面: `Placing an object on a steep slope.`
- 2.0 cm段差: `Placing an object on a large step.`

結果は`data/terrainplacespecific/terrainplacespecific_<MODEL>_seed<SEED>_best/`へ保存される。

## 実行前後のチェックリスト

### 実行前

- 新しい`GROUP`とtmux `SESSION`を付けた。
- 使用GPUが空いている。
- 学習CSVと`bert_emb_terrain`が同じannotation版である。
- 評価CSVがaction、weight、interactionごとに均衡化されている。
- loss、batch size、model sizeをログへ残せるrun名にした。

### 実行後

- bestとlastのcheckpointがある。
- `prototype_templates.json`が想定した定型文になっている。
- actionはgrasp、hold、lift、place、press、releaseの6クラスである。
- `release midair`が予測ラベル順に残っていない。
- UMAP、混同行列、予測CSVが3タスクすべてにある。
- `summary.md`にAccuracy / Macro-F1がある。
- 結果を報告するときはUMAP、混同行列、予測CSVへのパスも併記する。
