# 論文原稿 RQ1–RQ3：データ・モデル・コード・評価結果の対応表

本公開版の `paper_*` は可搬な実験名です。既存アーカイブの名前が異なる場合は、ローカルのシンボリックリンクで対応付けてください。原稿ソース・図・実験アーカイブは公開コードに含まれません。

調査日：2026-09-14（JST）

**2026-09-17 RQ1更新：** 原稿の教師あり参照を、Surfaceも教師に含むMLP（seed 42/43/44）へ差し替えた。[新しい結果と条件](paper_review_20260916/rq1_surface_supervised/summary.md)、[再検証記録](paper_review_20260916/rq1_surface_supervised/audit.json)を参照。`rq1_language_composition/rq1_20260916_235337/jobs/supervised_full_seed{42,43,44}/best.pt` の既存全量学習モデルを再利用し、原稿と同じYCB評価窓の全予測をCPU推論で再確認した。Macro F1はaction/weight/interaction = **89.31±0.34 / 91.12±0.50 / 75.34±0.82%**。新参照は4 head・episode-level validationであり、以下の旧教師ありPatchTST（3 head・window-level validation）は過去結果として保存する。対照学習・SSL・RQ2/RQ3のモデルは変更していない。

**同日追補：** 本資料を起点とした追加監査・原稿修正は [REPRODUCIBILITY_AUDIT.md](REPRODUCIBILITY_AUDIT.md) を参照。固定支持面の初期encoder実験には把持前オフセット補正がなく、後続wobble実験等で導入されたことを保存設定から確認し、第6節を訂正した。以下の原稿対応・hashは原則として初回調査時点のもの。

## 1. 対象と読み方

- 本文は論文実験の対応表です。原稿ソースは別途管理し、このリポジトリには同梱しません。
- 現在の原稿の表・本文・実際に読み込まれる図を、保存済みのmanifest、checkpoint、集計CSV、実機ログと照合した。
- **確認済み**は保存物との対応を確認できたもの、**推定**は状況証拠による対応、**未特定**は元ファイル・厳密な条件を特定できなかったものを指す。近い名前の実験を根拠なく代用していない。
- 以下の `$変数/相対パス` は第2節で定義する保存先からのパス。`{action,weight,interaction}`、`<seed>` などは複数ファイルのパステンプレートであり、文字どおりのファイル名ではない。
- 本調査では再学習・再評価はしていない。原稿、実験コード、データ、既存結果も変更していない。

### 最初に開くファイル

| 内容 | パス |
| --- | --- |
| RQ1：対照学習3モデル＋教師ありPatchTST、3 seedsの集計 | `/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/summary.md` |
| RQ1：全12モデルのcheckpoint・評価先の対応 | `/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/manifest.json` |
| RQ1：UMAP・混同行列・予測CSVへのリンク | `/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/ARTIFACTS.md` |
| RQ1：BYOLと他SSLの比較 | `/home/user/PatchTST/results/tokenfusion_byol/tokenfusion_byol_20260912_194926/summary.md` |
| RQ2：固定支持面の接触検知集計 | `/home/user/ur3_robotiq_ros2/artifacts/fixed_support_model_vs_fz_baseline/summary.csv` |
| RQ2：CNN・MLPの実機比較 | `/home/user/ur3_robotiq_ros2/artifacts/cnn_mlp_remount_evaluation_20260910/README.md` |
| RQ3：教師ありモデル追加後の最終結果 | `/home/user/PatchTST/results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852/SUMMARY.md` |
| RQ3：checkpoint・予測・図へのリンク | `/home/user/PatchTST/results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852/ARTIFACTS.md` |

**先に知っておくべき注意点：** RQ2の支持面分類31/40の元集計は未特定。RQ2の短時間Fz評価には試行数の対応に未解決点がある。RQ3は原稿の図に教師ありモデルが含まれるが、現在の本文の手法列挙からは抜けている。詳細は第8節。

## 2. 保存先の略記とディスクの実体

以下はパスを短く表記するための定義であり、学習を開始するコマンドではない。

```bash
G=/home/user/Genesis
P=/home/user/PatchTST
R=/home/user/ur3_robotiq_ros2
TRAIN="$G/data/train_08142026"
EVAL="$G/data/eval_08142026"
V=terrain_floor_wobble_uniform_descent_contact_clean_slip5mms
CG=terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352
SG=tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515
BG=tokenfusion_byol_20260912_194926
C42="$P/saved_models/terrain_contrastive/$CG"
CR42="$P/results/terrain_contrastive/$CG"
S42="$P/saved_models/tokenfusion_supervised/$SG"
SSL42="$P/saved_models/tokenfusion_ssl/$SG"
SSLR="$P/results/tokenfusion_supervised_ssl/$SG"
BYOL="$P/results/tokenfusion_byol/$BG"
M3="$P/results/paper_classification_multiseed/paper_classification_3seed_20260909"
M3CK="$P/saved_models/paper_classification_multiseed/paper_classification_3seed_20260909"
TD="$P/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827/dataset"
TA="$P/results/terminal_insertion_adaptation_efficiency/terminal_insertion_adaptation_efficiency_enter_minus030s_20260827_150802"
TB="$P/results/terminal_insertion_byol/terminal_insertion_byol_20260913_103313"
TS="$P/results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852"
```

大容量データは一部symlinkになっている。

| 見かけのパス | 実体 |
| --- | --- |
| `/home/user/Genesis/data` | `/mnt/nvme1/user/data/Genesis` |
| `/home/user/PatchTST/saved_models` | `/mnt/nvme1/user/data/PatchTST/saved_models` |
| `/home/user/ur3_robotiq_ros2/data` | `/mnt/nvme1/user/data/ur2_robotiq_ros2` |

最後の実体名は **`ur2_robotiq_ros2`** である。誤記として勝手に `ur3` に直さないこと。結果ディレクトリ間にも過去実験へのsymlinkがあり、移動・バックアップ時はリンク先を含めて保存する必要がある。

## 3. 共通のシミュレーションデータ：収集からアノテーションまで

### 3.1 原稿に対応するCSV

`08142026` というディレクトリ名だけでは版を特定できない。原稿に対応するのは **`$V` 版**であり、以前のterrain、floor normal/diverse、uniform descent導入前のCSVではない。

| 用途 | 実ファイルの指定 | 行数 | episode数 | 物体数 |
| --- | --- | ---: | ---: | ---: |
| 品質フィルタ後、均衡化前の学習集合 | `$TRAIN/train_08142026_${V}.csv` | 1,531,185 | 40,878 | — |
| 実際に学習へ渡した集合 | `$TRAIN/train_08142026_${V}_balanced.csv` | 437,244 | 40,851 | 892 |
| 評価の全候補集合 | `$EVAL/eval_08142026_${V}.csv` | 111,355 | 2,173 | 75 |
| Action評価 | `$EVAL/eval_08142026_${V}_action_balanced.csv` | 10,008 | — | — |
| Weight評価 | `$EVAL/eval_08142026_${V}_weight_balanced.csv` | 6,000 | — | — |
| Interaction評価 | `$EVAL/eval_08142026_${V}_interaction_balanced.csv` | 3,675 | — | — |

物体数・episode数はCSVの `csv_path` から再集計して確認した。原稿では学習物体をGSO、評価物体を未使用のYCBとしている。19,683は3タスクの評価行数の和であり、必ずしも重複なしの物理的window数ではない。

実際の学習CSVの絶対パス：

```text
/home/user/Genesis/data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv
```

関連ファイル：

- 生成・アノテーション条件：`$TRAIN/train_08142026_${V}_metadata.json`、`$EVAL/eval_08142026_${V}_metadata.json`。
- 学習用文章とembedding ID：`$TRAIN/annotation_index_map_${V}.csv`。
- 学習文章embedding：`$TRAIN/bert_emb_${V}/`。250種類のアノテーションに対応。
- 学習データ監査：`$SSLR/data_audit.md`、`$SSLR/data_audit.json`。CSVのSHA256、サンプル数、欠損ラベル等を追える。
- 学習CSVのSHA256：`b4f38c4b43d68848bb084fef6d8e7a0e58719b491039df0f4ec9d8df704c1570`。

### 3.2 生時系列・メタデータ・画像

生データのディレクトリ構造：

```text
$TRAIN/csv/<object>/Rigid/<scenario>/
  <object>_Rigid_<scenario>.csv
  <object>_Rigid_metadata_<scenario>.json
  ...steps/deform等の補助CSV...
  ..._annotations_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms.csv
$TRAIN/images/
```

評価側も `$EVAL/csv/` 以下に同様の構造を持つ。学習CSVの `csv_path` と `start` が、元のepisodeと80点windowの対応を指定する。

確認済みの例：

```text
/home/user/Genesis/data/train_08142026/csv/Cascadia_8_Color_AquariusHibscsBearingSeaBlk_Size_50/Rigid/accidental_drop_1_step_0.5cm_floor_descent/
/home/user/Genesis/data/eval_08142026/csv/001_chips_can/Rigid/accidental_drop_1_drop_simple_vary/
```

### 3.3 コードの対応

| 処理 | コード |
| --- | --- |
| Genesisシミュレーション、F/T・状態・metadata保存 | `$G/src/generation/run_genesis_sim.py` |
| ロボット動作 | `$G/src/generation/master_movement.py` |
| episode内のwindowに意味属性・文章を付与 | `$G/src/processing/rigid/1_create_annotations.py` |
| episode別annotation CSVの結合 | `$G/src/processing/rigid/2_create_train_csv.py` |
| 学習集合の重複なしサブサンプリング | `$G/src/processing/rigid/create_train_balanced_set.py` |
| 分類軸ごとの独立した評価集合作成 | `$G/src/processing/rigid/create_independent_eval_sets.py` |
| 学習文章のS-BERT embedding生成 | `$G/src/processing/rigid/3_bert_train.py` |
| Genesis側の評価embedding生成コード | `$G/src/processing/rigid/3_bert_eval.py` |
| F/T windowの品質チェック | `$G/src/processing/rigid/force_window_filter.py` |

**注意：** 最終的なRQ1の文章prototypeはGenesis側の旧評価embeddingをそのまま使ったものではなく、第4.3節のprepared evaluationを使う。現在のソースのデフォルト引数と当時の実行条件は同一とは限らない。版の特定にはCSV名、metadata、checkpoint内設定を優先する。

### 3.4 データ版を区別する主要条件

metadata・コードから確認できる主要な条件は以下。

- 両指の6軸ずつ、合計12軸。順序は `left_fx,left_fy,left_fz,left_tx,left_ty,left_tz,right_fx,right_fy,right_fz,right_tx,right_ty,right_tz`。
- 100 Hz、80点＝公称0.8秒のwindow、通常の開始位置strideは10点。early-dropイベントが通常gridから漏れる場合はepisode末尾に合わせた80点windowを追加する（最終CSVで10点grid外14,383窓、全てaccidental drop）。**モデルのpatch strideとは別の設定**。
- force絶対値100 N以上、非有限COM、1 stepで5 cmを超えるCOM移動、範囲外位置等のepisodeを品質条件で除外。
- `inclined_50`、`step_5cm`、`step_8cm`、指定のuniform descent版ではない旧wobble収集等を除外。
- `uniform_descent` のwobble追加収集は一定下降速度0.04 m/s。全ての歴代シナリオが一律にこの生成法だったという意味ではない。
- slipは物体COMと指中点の相対運動を用いる。最初～最後の左右同時接触間の最大接線変位をwindow全体の経過時間（通常0.79秒）で割った量が0.005 m/s以上、左右同時接触が5点以上連続することを条件とする。瞬間速度ではない。
- 環境が物体を完全に支えているwindowではweightを省略し、把持接触が十分でない場合はstable/slipを省略する。他属性が有効ならwindow自体は残す。
- 対照学習の文章にはaction/weight/interactionに加えて支持面の記述が入る。一方、RQ1の表はaction/weight/interactionのみ。
- force全軸ゼロのwindowを学習・評価から除く処理もモデル側にある。該当設定はcheckpoint内の引数と `$P/PatchTST_self_supervised/datautils.py` で追える。

### 3.5 正規化・validationの重要な区別

RQ1は学習CSVからtrain/validationを分け、**train部分だけ**で各軸の正規化統計を計算する。統計はcheckpointの `normalization` に保存され、評価側で再計算しない。

- 対照学習・SSLのseed 42：window単位のランダム90/10 split、train 393,520、validation 43,724。
- 教師あり：actionを用いた層化90/10 window split。train 393,519、validation 43,725。欠損属性のlossはマスクする。
- seed 43/44ではsplit・初期化・データ順も変わる。3 seedsは同一固定splitで初期化だけを変えた実験ではない。
- **このvalidationは物体単位・episode単位の分離ではない。** YCB評価集合をvalidationに使っていないことと、学習内validationの物体が完全に未使用であることは別。原稿の “held-out GSO subset” をobject-disjointの意味で読ませないよう注意する。

## 4. RQ1：未使用物体の状態識別

対応する原稿ラベル：`tab:model_comparison`、`subsec:classification`。

### 4.1 アーキテクチャ・学習条件

| モデル | 保存設定に基づく概要 |
| --- | --- |
| Token-Fusion PatchTST | **raw入力、deltaなし**。patch length/stride 10/10、`d_model=128`、3層、16 heads、`d_ff=768`。Channel-Independent時間処理後に1層・8 headsのToken-Fusionを適用し、768次元へ。約1.11M規模。 |
| CNN | 12軸を入力channelとする通常の**1D temporal CNN**。4段、channel数81/162/324/324、kernel 5、BatchNorm/GELU/MaxPool、dropout 0.2、global average pooling＋768次元Linear。1,109,172 parameters。12×80を1枚の画像として扱う2D CNNではない。 |
| MLP | 全80×12点をflattenし、960→429→429→429→768。GELU/dropout 0.2、poolingなし。1,111,449 parameters。 |
| Supervised PatchTST | 同じraw Token-Fusion encoder＋action 6 / weight 2 / interaction 3クラスの分類head。元の46クラス分類器ではない。 |

- 対照学習は両方向InfoNCE。言語側は固定した `sentence-transformers/all-mpnet-base-v2`、768次元。
- BS 512、peak LR 0.002、最大400 epochs、warmup 10、cosine decay、min LR 1e-5、early stopping patience 75。
- 対照学習のbestはvalidation loss、教師ありのbestはvalidationの平均Macro F1で選択。testをbest選択に使っていない。
- 教師ありはbalanced class weights付きcross entropy、属性欠損をマスクしたmulti-task学習。
- 原稿の表は各seedの**best**。lastの結果も保存されているが、表の値とは区別する。

### 4.2 全checkpointの場所

**seed 42：既存の原稿用実験を再利用。**

対照学習の `<arch>` は `tokenfusion`、`cnn`、`mlp`。

```text
$C42/terrain_floor_wobble_uniform_descent_slip5mms_<arch>_seed42/
  terrain_floor_wobble_uniform_descent_slip5mms_<arch>_seed42_best.pth
  terrain_floor_wobble_uniform_descent_slip5mms_<arch>_seed42_last.pth

$S42/tokenfusion_supervised_multitask_seed42/
  tokenfusion_supervised_multitask_seed42_best.pth
  tokenfusion_supervised_multitask_seed42_final.pth
```

RQ2・RQ3にも使用する対照学習Token-Fusion seed 42 bestの絶対パス：

```text
/home/user/PatchTST/saved_models/terrain_contrastive/terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42_best.pth
```

**seed 43/44：3-seed比較用に追加学習。** `<model>` は `tokenfusion`、`cnn`、`mlp`、`supervised`。

```text
$M3CK/paper_<model>_seed<seed>/paper_<model>_seed<seed>_best.pth
```

lastは対照学習で `_last.pth`、教師ありで `_final.pth`。完全な絶対パス一覧は `$M3/manifest.json` の `runs[].checkpoints` が正本。

| モデル | seed 42 best epoch | seed 43 best epoch | seed 44 best epoch | last epoch |
| --- | ---: | ---: | ---: | ---: |
| Token-Fusion | 358 | 356 | 396 | 400 |
| CNN | 377 | 348 | 377 | 400 |
| MLP | 383 | 367 | 382 | 400 |
| Supervised Token-Fusion | 348 | 382 | 375 | 400 |

### 4.3 実際に評価へ渡したCSV・定型文prototype

正本は **`$M3/prepared_eval/`**。seed 42の元は `$CR42/prepared_eval_templates_release_merged/`。

```text
$M3/prepared_eval/
  eval_08142026_terrain_action_balanced.csv
  eval_08142026_terrain_weight_balanced.csv
  eval_08142026_terrain_interaction_balanced.csv
  eval_08142026_terrain_independent_balanced.csv
  action_index_map.csv
  weight_index_map.csv
  interaction_index_map.csv
  prototype_templates.json
  bert_emb/<action|weight|interaction>/<index>.pt
  csv/  -> 評価側の生CSV
```

`release midair` は **`release`へ統合**。6 action各1,668例、2 weight各3,000例、3 interaction各1,225例。統合前のsource CSVでreleaseとrelease midairの件数が違っていても、統合後の6クラスがこの件数になる。

| 分類軸 | クラス | 実際の文章prototype |
| --- | --- | --- |
| action | grasp | Grasping an object. |
| action | hold | Holding an object. |
| action | lift | Lifting an object. |
| action | place | Placing an object. |
| action | press | Pressing an object. |
| action | release | Releasing an object. |
| weight | heavy | The object is heavy. |
| weight | light | The object is light. |
| interaction | accidental drop | The object was accidentally dropped. |
| interaction | slip | The object is slipping. |
| interaction | stable | The object has stable contact. |

**単語prototypeでも、release/release midairのembedding平均でもない。分類とUMAPに同じ文章embeddingを使う。** `prototype_templates.json` に余分な `drop` 項目があっても、実際のactionクラス集合は `action_index_map.csv` の6クラスで決まる。

### 4.4 数値・UMAP・混同行列

集計：`$M3/summary.md`、`$M3/summary.json`、`$M3/per_seed_metrics.csv`。

原稿のbest結果。各セルはAccuracy / Macro F1、百分率、3 seeds（42,43,44）の平均±標本標準偏差（ddof=1）。

| モデル | Action | Weight | Interaction |
| --- | --- | --- | --- |
| Token-Fusion | 94.02±0.11 / 94.02±0.11 | 93.00±0.50 / 93.00±0.50 | 79.08±0.53 / 78.14±0.73 |
| CNN | 93.27±0.19 / 93.28±0.19 | 93.97±0.38 / 93.97±0.38 | 82.10±0.40 / 81.24±0.46 |
| MLP | 94.12±0.31 / 94.12±0.31 | 93.38±0.50 / 93.38±0.50 | 82.93±0.28 / 82.57±0.34 |
| Supervised Token-Fusion | 93.59±0.19 / 93.56±0.19 | 92.17±0.05 / 92.17±0.05 | 74.51±0.29 / 74.46±0.28 |

対照学習3モデルの `<target>` はaction/weight/interaction、`<selection>` はbest/last。

```text
$M3/evaluation/<tokenfusion|cnn|mlp>/seed<42|43|44>/<selection>/
  umap_metrics_summary.md
  pred/<target>_pred.csv
  umap/<target>_umap.png
  heatmap/<target>_heatmap.png
```

教師ありは配置が異なる。

```text
$M3/evaluation/supervised/seed<42|43|44>/<selection>/
  balanced_eval_summary.md
  balanced_eval_summary.json
  <target>_pred.csv
  <target>_umap.png
  <target>_confusion_matrix.png
```

すぐに開ける例：

- Token-Fusion action UMAP：`/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/evaluation/tokenfusion/seed42/best/umap/action_umap.png`
- 同混同行列：`/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/evaluation/tokenfusion/seed42/best/heatmap/action_heatmap.png`
- 教師ありaction混同行列：`/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/evaluation/supervised/seed42/best/action_confusion_matrix.png`

### 4.5 学習・評価コードと再現の入口

| 用途 | パス |
| --- | --- |
| seed 42 terrain対照学習の起動 | `$P/scripts/run_terrain_contrastive_0814.sh` |
| 3-seed比較の起動・管理 | `$P/scripts/run_paper_classification_multiseed.sh`、`$P/scripts/paper_classification_multiseed.py` |
| PatchTST対照学習 | `$P/PatchTST_self_supervised/patchtst_bert_normalized.py`、`patchtst_bert.py` |
| CNN/MLP正規化学習、モデル定義 | `$P/PatchTST_self_supervised/patchtst_baseline_normalized.py`、`baseline_models.py` |
| 対照学習モデルの評価・可視化 | `$P/PatchTST_self_supervised/patchtst_eval_normalized.py`、`patchtst_eval.py` |
| 教師あり学習・評価 | `$P/PatchTST_self_supervised/patchtst_supervised.py`、`evaluate_supervised_balanced.py` |
| 正規化・クラス統合 | `$P/PatchTST_self_supervised/force_normalization.py`、`classification_taxonomy.py` |
| 追加seed実行時のコードsnapshot | `$M3/code/PatchTST_self_supervised/` |
| ログ・コマンド情報 | `$M3/logs/`、`$M3/manifest.json`、`$M3/runner.log` |
| 操作説明 | `$P/docs/paper_classification_multiseed.md`、`$P/docs/terrain_training_and_evaluation.md` |

同じセル内でファイル名だけを併記したものは、直前のファイルと同じディレクトリにある。

## 5. RQ1：自己教師あり表現の線形読み出し

対応する原稿ラベル：`tab:ssl_linear_probe`。

### 5.1 使用モデル・プロトコル

- 全方式で**raw Token-Fusion PatchTST**を使用。旧CNN-SSLや旧JEPAの数値ではない。
- pretraining seedは42の1個。共通の437,244-window CSVから10% validationを分割し、train側のみで正規化。
- SSLは最大400 epochs、BS512、LR0.002、warmup10、cosine、validation lossによるearly stopping。
- validation-best encoderを凍結し、各属性につき1つの線形分類層だけを学習。プローブは最大200 epochs、LR0.01、BS512、weight decay 0、patience30、balanced class weights。
- SSL線形プローブは教師ありheadによる読み出しなので、評価に文章prototypeは使わない。UMAPはF/T表現を示す。
- Masked reconstructionはマスク時系列の復元、VAEは復元＋KL、VICRegはview間不変性・分散・共分散制約。
- BYOLはonline/EMA target encoderとprojector/predictorを使う2-viewの自己蒸留。追加projector/predictorはpretraining用で、線形評価は共通の768次元encoder出力を使う。

### 5.2 checkpoint・線形head・結果

`<method>` は `masked_reconstruction`、`vae`、`vicreg`。

```text
$SSL42/tokenfusion_ssl_<method>_seed42/tokenfusion_ssl_<method>_seed42_best.pth
$SSLR/linear_probes/<method>/
  linear_probe_summary.json
  linear_probe_summary.md
  <target>_linear_head.pth
  <target>_history.csv
  <target>_pred.csv
  <target>_umap.png
  <target>_confusion_matrix.png
```

BYOLは別group：

```text
$P/saved_models/tokenfusion_ssl/$BG/tokenfusion_ssl_byol_seed42/tokenfusion_ssl_byol_seed42_best.pth
$BYOL/linear_probes/<best|last>/
  linear_probe_summary.json
  linear_probe_summary.md
  <target>_linear_head.pth
  <target>_history.csv
  <target>_pred.csv
  <target>_umap.png
  <target>_confusion_matrix.png
```

| 方式 | encoder best epoch | encoder last epoch | Action Acc/F1 [%] | Weight Acc/F1 [%] | Interaction Acc/F1 [%] |
| --- | ---: | ---: | --- | --- | --- |
| BYOL | 8 | 83 | 77.01 / 76.88 | 83.22 / 83.21 | 74.18 / 73.84 |
| Masked reconstruction | 387 | 400 | 85.20 / 85.24 | 86.82 / 86.81 | 78.80 / 78.30 |
| VAE | 338 | 400 | 82.39 / 82.41 | 83.77 / 83.71 | 77.63 / 77.19 |
| VICReg | 161 | 236 | 80.26 / 80.14 | 73.98 / 73.95 | 73.14 / 73.49 |

表はbest encoderに対する線形プローブ。1 seedなので標準偏差は付けない。encoderのbest epochと、線形headのbest epochは別で、後者は `linear_probe_summary.json` / 各headのhistoryを参照する。

UMAP・混同行列の例：

- Masked reconstruction：`/home/user/PatchTST/results/tokenfusion_supervised_ssl/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515/linear_probes/masked_reconstruction/action_umap.png`
- 同混同行列：`/home/user/PatchTST/results/tokenfusion_supervised_ssl/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515/linear_probes/masked_reconstruction/action_confusion_matrix.png`
- BYOLの全リンク：`$BYOL/ARTIFACTS.md`。

### 5.3 コード・ログ

- 起動：`$P/scripts/run_tokenfusion_supervised_ssl.sh`、`$P/scripts/run_tokenfusion_byol.sh`。
- encoder・SSL：`$P/PatchTST_self_supervised/token_fusion_encoder.py`、`token_fusion_force_ssl.py`、`byol_force.py`。
- 線形プローブ：`$P/PatchTST_self_supervised/cnn_ssl_linear_probe.py`。**ファイル名にCNNとあるがToken-Fusionにも対応する**。
- BYOL実行記録：`$BYOL/manifest.json`、`commands.sh`、`data_audit.json`、`logs/`、`code/`。
- 詳しいBYOLの実装・augmentation・学習条件：`$P/docs/tokenfusion_byol.md`。
- 他3方式の比較：`$SSLR/linear_probe_comparison.md`。学習側checkpointディレクトリと各probe summaryにも元checkpointが記録されている。

## 6. RQ2：encoderを更新しない実機接触認識

対応する原稿ラベル：`subsec:real_transfer`、`tab:real_contact_detection`、`tab:real_support_classification`。

### 6.1 共通モデル・正規化・座標変換

- 主モデルは第4.2節の**対照学習Token-Fusion seed 42 best、epoch358**。追加の実機encoder学習はしない。
- UR3＋Robotiq 2F＋左右MMS101で12軸F/Tを収集。
- `sim_local_rz180`：各センサの `Fx,Fy,Tx,Ty` を反転し、`Fz,Tz` は維持する。実装・設定中の12軸符号は `[-1,-1,1,-1,-1,1]` を左右に並べたもの。
- 正規化はRQ1のシミュレーション統計をそのまま使うのではなく、**実機pick-and-place 10試行から得た統計**を使用。**初期の固定支持面Token-Fusion実験60試行には把持前オフセット補正がない。** 後続wobble本実験20試行・追加13試行、CNN/MLP remount各20試行では、把持前1秒の無負荷平均を校正時の無負荷参照値に合わせる補正を行う。Fz baselineの把持前median tareとは別処理。
- 保存JSONの座標変換は確認できるが、物理的なセンサ取付け姿勢を確定する完全なTF・外部計測記録は未特定。保存済みの符号変換と、幾何学的に完全に検証された座標校正を混同しない。

| 用途 | 校正保存先 | 生データ |
| --- | --- | --- |
| 8月のToken-Fusion主要接触実験 | `$R/artifacts/pick_place_normalization/` | `$R/data/keyboard_pick_and_place/` 内、manifestに列挙された10 episodes |
| 9月10日のCNN/MLP比較 | `$R/artifacts/pick_place_normalization_remount_20260910/` | `$R/data/keyboard_pick_and_place/20260910_112259_931378/` の10 episodes |

両保存先に `pick_place_normalization.json`、`pick_place_manifest.json`、`pick_place_force_windows.npz`、`pick_place_phase_stats.json` がある。**校正手順は共通だが統計は同一ではない**。

旧校正の10 episodesの完全な選択は `pick_place_manifest.json` が正本。収集runは `20260827_105243_477939`、`20260827_105637_961759`、`20260827_113629_939423`、`20260827_115029_672083` にまたがる。

正規化JSONのSHA256：

- 旧校正：`3f35e8207e2789b63a7f933f22b87d83f47e57ca76a397dc110b4cbc20622c97`
- remount校正：`a1abf26900e7041244c635105ca791342b0a24b2df734a1afde3668ce2a368f9`

### 6.2 オンライン接触検知の条件とコード

- 80点の履歴を5点＝50 msごとに評価する。
- 下降開始後2秒のgate、place cosine > hold cosineが3回連続でstop。
- 文章は **Holding an object. / Placing an object.**。保存されたprototypeは `$CR42/prepared_eval_templates_release_merged/bert_emb/action/1.pt` と `3.pt`。
- operatorのEnterから0.300秒を引いた時刻を参照接触とし、原稿のsuccessは0.8秒以内の停止。平均delayは成功試行のみ。
- Fz baselineは左右の把持前tare補正後Fz和の符号遷移で判定。保存式は `(left_fz-left_pregrasp_median) + (right_fz-right_pregrasp_median) < 0`、3回連続。

| 役割 | コード・設定 |
| --- | --- |
| learned detector・モデル読み込み | `$R/data_capture_tools/data_capture_tools/place_cosine_detector.py` |
| learned detectorを使う実機収集 | `$R/data_capture_tools/data_capture_tools/model_place_capture_node.py` |
| Fz baseline | `$R/data_capture_tools/data_capture_tools/fz_baseline_capture_node.py` |
| pick-and-place校正の起動 | `$R/data_capture_tools/launch/pick_place_motion_capture.launch.py` |
| 固定面／wobbleのモデル検知起動 | `$R/data_capture_tools/launch/lift_and_lower_model_capture.launch.py`、`lift_and_lower_wobble_model_capture.launch.py` |
| Fz検知起動 | `$R/data_capture_tools/launch/lift_and_lower_fz_baseline_capture.launch.py`、`lift_and_lower_wobble_fz_baseline_capture.launch.py` |
| wobble 5秒Fz検知 | `$R/data_capture_tools/launch/lift_and_lower_wobble_fz_long_horizon_capture.launch.py` |
| 校正統計作成 | `$R/data_processing/compute_pick_place_force_normalization.py` |
| 反応時間補正 | `$R/data_processing/apply_enter_reaction_compensation.py` |
| 固定面の集計・図 | `$R/data_processing/plot_fixed_support_model_vs_fz_baseline.py` |
| CNN/MLP remount集計 | `$R/data_processing/summarize_remount_model_evaluations.py` |

対応するYAMLは `$R/data_capture_tools/config/` にある。主なものは `pick_place_motion_capture.yaml`、`pick_place_motion_data_capture.yaml`、`lift_and_lower_model_motion_capture.yaml`、`lift_and_lower_model_data_capture.yaml`、`lift_and_lower_wobble_model_motion_capture.yaml`、`lift_and_lower_wobble_model_data_capture.yaml`、`lift_and_lower_fz_baseline_motion_capture.yaml`、`lift_and_lower_wobble_fz_baseline_motion_capture.yaml`、`lift_and_lower_wobble_fz_long_horizon_motion_capture.yaml`。

**当時の条件は各episodeの `config.yaml`、`model_evaluation_config.json` / `fz_baseline_evaluation_config.json` を優先する。現在のYAMLには再取付け後の変更が含まれる。**

### 6.3 固定支持面：原稿の元試行と集計

主集計先：`$R/artifacts/fixed_support_model_vs_fz_baseline/`。

- `manifest.json`：採用run・除外理由。
- `trials.csv`：試行別の判定・delay。
- `summary.csv`：条件別集計。
- `model_vs_fz_baseline_overview.pdf` / `.png`：集計図。

| 条件・手法 | 生データrun（`$R/data/`からの相対パス） | 原稿Success | Delay [s] |
| --- | --- | ---: | ---: |
| 木製直方体・Token-Fusion | `place_rectangle_retest/20260827_152109_994319` ＋ `place_rectangle_retest/20260827_153945_906240` | 20/20 | 0.348 |
| コルク円柱・Token-Fusion | `place_cylinder/20260827_162707_684887` | 19/20 | 0.430 |
| 軟らかいボール・Token-Fusion | `place_baseball/20260827_164827_013733` ＋ `place_baseball/20260827_170059_207659` | 11/20 | 0.631 |
| 木製直方体・Fz | `place_rectangle_fz_baseline/20260827_195736_740533` | 20/20 | 0.492 |
| コルク円柱・Fz | `place_cylinder_fz_baseline/20260827_201609_654573` | 20/20 | 0.547 |
| 軟らかいボール・Fz | `place_baseball_fz_baseline/20260827_203609_924573` | 0/20 | — |

原稿の合計はToken-Fusion 50/60、Fz 40/60。

**保存集計と原稿の差：** コルクの `episode_000005` はEnterなしのため `summary.csv` の有効参照時刻付き集計から除外され、19/19となる。一方、原稿はこの失敗試行を分母に入れて19/20としている。delayは19成功試行から計算する。

**プロトコルの小差：** 木製直方体の前半11試行は停止待ち上限0.5秒設定、後半9試行は0.8秒。前半は全て0.5秒未満で成功しているため掲載成否は変わらないが、全20試行が文字どおり同一timeoutだったわけではない。

### 6.4 wobble：元試行と未解決の分母

| 内容 | 生データrun（`$R/data/`からの相対パス） | 確認できた内容 |
| --- | --- | --- |
| Token-Fusion本実験 | `place_rectangle_wobble/20260827_230202_819708` | 3試行、0成功 |
| 同 | `place_rectangle_wobble/20260827_230524_074937` | 12試行、10成功 |
| 同 | `place_rectangle_wobble/20260827_231650_521705` | 5試行、2成功 |
| 支持面分類用と考えられる追加成功収集 | `place_rectangle_add_wobble/20260827_233058_989610` | 13試行、8成功 |
| Fz短時間収集 | `place_rectangle_wobble_fz_baseline/20260827_234339_773604` | 保存された評価JSONは**11件、0成功** |
| Fz 5秒収集 | `place_rectangle_wobble_fz_baseline_long_horizon/20260829_142525_572228` | 20試行、15成功 |

Token-Fusion本実験は20件のJSONから12/20、成功平均delay 0.436867秒を再集計でき、原稿の12/20・0.437秒と一致する。

Fz 5秒収集は15/20、成功平均3.889727秒で原稿と一致する。成功delayの最小値でも3.342秒なので、この20試行を0.8秒で打ち切って数えるなら0/20になる。しかし、**原稿の短時間0/20がこの再集計なのか、別の20試行なのかは特定できなかった**。短時間用ディレクトリの11件だけを「20件確認済み」とは扱わない。

### 6.5 支持面分類31/40：部分的にのみ特定

原稿の設定：成功した木製直方体の固定20件・wobble20件で、次の2文章を比較する。

- `Placing an object on a fixed flat surface.`
- `Placing an object on a wobbling surface.`

原稿の数値：固定18/20（cosine平均0.263 / 0.145）、wobble13/20（0.061 / 0.081）、合計31/40。

**推定できるデータ候補：** 固定は第6.3節の20成功、wobbleは第6.4節の最初の12成功＋追加8成功。原稿のコメントにも後者の説明が残る。

**未特定：** この40件の厳密な採用リスト、読み出しwindowの時刻、2文章の保存embedding、各試行cosineの元CSV/JSON、31/40を生成した実行スクリプト・コマンド。今回の調査ではこの集計を保存物から独立に再現確認できなかった。

以下は名前が似ているが、自動的にこの31/40の根拠にはしない。

- `$P/scripts/evaluate_wobble_place_support.sh` と `$P/results/wobble_place_support_eval08142026/`：シミュレーションの支持面評価。
- `$R/data_processing/analyze_two_surface_selection.py`、`$R/data/place_rectangle_surface_selection/`：別のtwo-surface selection実験。
- `$G/src/evaluation/evaluate_wobble_lift_and_lower.py`：Genesis側のシミュレーション評価。

### 6.6 CNN・MLPの実機比較

原稿冒頭のアーキテクチャ選択理由にあるCNN 11/20、MLP 17/20は次の保存物と一致する。

```text
$R/data/place_rectangle_cnn_norm_remount_20260910/20260910_115223_636682/
$R/data/place_rectangle_mlp_norm_remount_20260910/20260910_121547_246046/
$R/artifacts/cnn_mlp_remount_evaluation_20260910/
  README.md
  summary.json
  episodes.csv
```

モデルは第4.2節の各CNN/MLP seed 42 best。成功平均delayはCNN約0.599秒、MLP約0.483秒。

**比較上の注意：** Token-Fusion 20/20は8月の旧校正・把持前補正導入前、CNN/MLPは9月10日のセンサ再取付け後の新校正・把持前補正あり。データ取得日、正規化統計、オフセット補正の有無が異なるため、完全に同条件のarchitecture-only比較とは言えない。

### 6.7 実機データ1試行の中身

```text
$R/data/<experiment>/<run>/episode_<number>/
  config.yaml
  episode_status.json
  bag/episode_<number>/metadata.yaml
  bag/episode_<number>/episode_<number>_0.db3
  csv/key_events.csv
  csv/force_latency_summary.csv
  model_evaluation.json                # learned detectorの場合
  model_evaluation_config.json
  empty_offset_correction.json
  csv/model_scores.csv
  csv/contact_to_detection_force.csv
```

Fz版は `fz_baseline_evaluation.json` / `fz_baseline_evaluation_config.json` を参照する。run直下の結合CSVもある。ファイルの有無は収集方式に依存する。

RQ2の主要指標はオンラインstopの成否・delayであり、RQ1のような全run共通のUMAP/混同行列一式は特定していない。接触後波形は `csv/contact_to_detection_force.csv`、時系列スコアは `csv/model_scores.csv`、再描画候補は `$R/data_processing/plot_model_place_inputs.py` / `render_place_ft_cosine_video.py`。

## 7. RQ3：未学習の嵌合完了イベントへの適応

対応する原稿ラベル：`subsec:fewshot_insertion`、`fig:terminal_insertion_linear_vs_scratch`、`fig:terminal_insertion_lora_vs_scratch`。

### 7.1 生実機データと保存済み学習dataset

**現在の生データの正しいroot：** `/home/user/ur3_robotiq_ros2/data/insertion_binary/`。

dataset manifestの125 episodesについて、episodeディレクトリ、bag metadata、key event CSVの存在を確認し、保存された3イベントのtimestampも生CSVと照合して一致した。

| 収集run | manifestに採用されたepisode数 |
| --- | ---: |
| `20260826_145900_931570` | 13 |
| `20260826_152221_323003` | 2 |
| `20260826_153919_041643` | 1 |
| `20260826_154316_123637` | 90 |
| `20260826_181110_252278` | 19 |
| 合計 | 125 |

**古いパスに注意：** manifest内の `data_root=/mnt/nvme1/user/data/ur2_robotiq_ros2/keyboard_descent` は現在存在しない。当時のtask名は `keyboard_descent` だが、現在は `insertion_binary` にデータがある。再構築時はrootを読み替える必要がある。

加工済みdatasetの正本：

```text
$TD/terminal_insertion_windows.npz
$TD/dataset_manifest.json
$TD/axis_audit.json
$TD/reaction_delay_relabel_audit.json
```

`$TS/dataset/` にも同じdatasetが保存され、NPZのSHA256は両者で一致する：

```text
67223c7dd1bfe18470d37ff649f784b83f29850022413e3bb9f1b94f155662c0
```

### 7.2 加工・split・ラベル定義

- ROS `WrenchStamped` のheader時刻を使用し、左右を100 Hzへ補間・同期。座標符号変換は `sim_local_rz180`。
- 把持完了1秒後から `post_enter_descent_complete` までを使用。
- 80点の因果window、stride10点。labelは**window終端**が `Enter - 0.300 s` 以降かどうか。嵌合完了後の観測も全てpositive。
- 125成功episodesを固定の100 train / 5 validation / 20 testにepisode単位で分割。
- 実際のwindow数は次表。RQ1と違い、ここはepisode-disjoint split。

| split | episodes | windows | before | after |
| --- | ---: | ---: | ---: | ---: |
| train | 100 | 5,018 | 3,710 | 1,308 |
| validation | 5 | 250 | 184 | 66 |
| test | 20 | 979 | 718 | 261 |
| 合計 | 125 | 6,247 | 4,612 | 1,635 |

- 入力正規化は**100 train episodes全て**の重複なし同期サンプル57,604点から作った統計を、全N・全手法で固定使用する。N=1でも「正規化まで1 episodeのみ」ではない。
- Frozen linearではさらに、選んだN個の学習episodesのfeatureで標準化する。これは入力F/Tの正規化とは別。
- 0.3秒補正により375 windowsがbefore→afterへ変更された。元波形・episode splitは変更されていない。
- 旧未補正dataset `$P/results/terminal_insertion/terminal_insertion_20260826_190335/dataset/` の数値とは混ぜない。

### 7.3 収集・加工・学習・作図コード

| 処理 | パス |
| --- | --- |
| 実機収集node | `$R/data_capture_tools/data_capture_tools/key_motion_capture_node.py` |
| 収集launch・設定 | `$R/data_capture_tools/launch/key_motion_capture.launch.py`、`$R/data_capture_tools/config/key_motion_capture.yaml`、`key_motion_data_capture.yaml` |
| bagからdataset作成 | `$P/PatchTST_self_supervised/prepare_terminal_insertion_dataset.py` |
| Enter反応時間補正 | `$P/PatchTST_self_supervised/relabel_terminal_insertion_reaction_delay.py` |
| 凍結encoderのデータ効率実験 | `$P/PatchTST_self_supervised/train_terminal_insertion_data_efficiency.py` |
| LoRA・scratch学習 | `$P/PatchTST_self_supervised/train_terminal_insertion_probe.py` |
| 全手法集計・百分率の図 | `$P/PatchTST_self_supervised/summarize_terminal_insertion_pretrained_adaptation.py` |
| BYOL下流実験 | `$P/scripts/run_terminal_insertion_byol.sh` |
| 教師あり下流実験 | `$P/scripts/run_terminal_insertion_supervised.sh` |
| 結果検証・UMAP/混同行列・原稿用図の整理 | `$P/scripts/finalize_terminal_insertion_byol.py` |
| 実行時コードsnapshot・起動スクリプト | `$TS/code/`、`$TS/run_terminal_insertion_byol.sh`、`$TS/run_terminal_insertion_supervised.sh` |

履歴上の起動コードとして `run_terminal_insertion_pretrained_data_efficiency_corrected.sh`、`run_terminal_insertion_adaptation_efficiency.sh`、`run_terminal_insertion_all_pretrained_adaptation.sh` も `$P/scripts/` にある。ただし旧JEPA・full fine-tuning等も含む場合があり、現在の原稿条件をそのまま再現する入口とは限らない。

### 7.4 使用encoderと適応条件

pretrained encoderは全て**RQ1で使用したseed 42 best**：第4節の言語対照学習Token-Fusion、教師ありToken-Fusion、第5節のMasked reconstruction / VAE / VICReg / BYOL。教師ありモデルは元の3分類headを捨て、768次元表現を使う。

- 下流のrandom seeds：42,43,44,45,46。**pretrainingを5回やり直したものではない**。
- 学習episode数：N=1,2,5,10,25,50,100。同一seedでは手法間で同一のnested subsets。
- Frozen Linear：encoder凍結、769 trainable parameters。
- LoRA：attention Q/V、rank4、alpha8、分類headを含め8,961 trainable parameters。
- Scratch End-to-End：ランダム初期化から1,111,937 parameters全てを学習。
- 原則最大300 epochs、BS256、AdamW、weight decay1e-4、patience30。Frozen LR0.002、LoRA LR0.0005。validation lossでbestを選択。
- testの二値判定閾値は0.5固定。window-level Macro F1を5 seedsの平均±標本標準偏差で表示。保存CSVのmetricは0–1、原稿・図は百分率。
- **履歴上の例外：** 元のadaptation sweepではseed42・N100の代表条件にpatience0で300 epochsまで回す設定がある。全保存runが一律patience30と断言せず、該当起動scriptとhistoryを参照する。

### 7.5 結果groupの依存関係

```text
$TA  元の対照学習・Masked・VAE・VICReg・scratchのadaptation結果
  ↓ 既存結果を再利用
$TB  BYOL追加後の比較
  ↓ 既存結果を再利用
$TS  教師ありToken-Fusion追加後の比較（現在の原稿PDFの生成元）
```

`$TS/manifest.json` の `reference_results` は `$TB`。さらに過去結果へのリンクを含む。最終groupだけをsymlinkのままコピーすると旧モデルの実体を失う可能性がある。

主要ファイル：

```text
$TS/manifest.json
$TS/validation.json
$TS/publication.json
$TS/SUMMARY.md
$TS/REPORT_ja.md
$TS/ARTIFACTS.md
$TS/pretrained_linear_vs_scratch_end_to_end_all_seed_results.csv
$TS/pretrained_linear_vs_scratch_end_to_end_aggregate.csv
$TS/pretrained_lora_vs_end_to_end_all_seed_results.csv
$TS/pretrained_lora_vs_end_to_end_aggregate.csv
```

各条件の `<seed>` は42–46、`<NNN>` は001/002/005/010/025/050/100。

```text
$TS/pretrained_frozen/<contrastive|masked_reconstruction|vae|vicreg|byol|supervised>/seed_<seed>/n_<NNN>/
$TS/pretrained_adaptation/<masked_reconstruction|vae|vicreg|byol|supervised>/lora/seed_<seed>/n_<NNN>/
$TS/lora_r4/seed_<seed>/n_<NNN>/       # 対照学習encoderのLoRA
$TS/scratch/seed_<seed>/n_<NNN>/
```

各条件で `best.pth`、`metrics.json`、`history.csv`、`test_predictions.csv` を参照する。`metrics.json` には元encoder checkpoint、選択したtrain episode IDs、best epoch、test混同行列等が含まれる。

追加BYOL・教師ありの各70条件には混同行列が保存されている。UMAPは各方式のseed42・N=1/5/100の代表条件で保存。**全seed・全NにUMAPがあるという意味ではない**。

```text
$TS/pretrained_frozen/supervised/seed_42/n_100/umap.png
$TS/pretrained_frozen/supervised/seed_42/n_100/confusion_matrix.png
$TS/pretrained_adaptation/supervised/lora/seed_42/n_100/umap.png
$TS/pretrained_adaptation/supervised/lora/seed_42/n_100/confusion_matrix.png
```

歴代条件の可視化有無・直接リンクは `$TS/ARTIFACTS.md`、BYOL単体は `$TB/ARTIFACTS.md` を参照。

### 7.6 原稿の図との対応

| 原稿が読み込むPDF | 生成元（`$TS/` 内） |
| --- | --- |
| `$G/fig/terminal_insertion_pretrained_linear_vs_scratch_end_to_end.pdf` | `pretrained_linear_vs_scratch_end_to_end_macro_f1_5seed.pdf` |
| `$G/fig/terminal_insertion_pretrained_lora_vs_end_to_end.pdf` | `pretrained_lora_vs_end_to_end_macro_f1_5seed.pdf` |

それぞれ内容のSHA256が一致することを確認した。

- Frozen図：`e3da36042640c5752b225a1da408731a79ab0482177282243127dce51493a7d0`
- LoRA図：`57987a070f676a4e37bfa11a981fb93ea463480c0f21637aa9391c88787b7013`

`pretrained_vs_scratch_patchtst_macro_f1_5seed.pdf` という旧名のファイルも残るが、上記2図と同一物として扱わない。

原稿の数値との照合例（Macro F1 [%]）：

| 条件 | 数値 |
| --- | ---: |
| Frozen contrastive、N100 | 91.90±1.02 |
| Scratch、N100 | 98.34±0.19 |
| Contrastive LoRA、N1 | 91.66±2.77 |
| Contrastive LoRA、N5 | 95.29±1.33 |
| Contrastive LoRA、N10 | 96.93±1.17 |
| Contrastive LoRA、N100 | 98.99±0.17 |

**原稿の不整合：** 現在の2図は教師ありToken-Fusionを含む7系列（6 pretrained＋scratch）だが、現在のRQ3本文のAdaptation Methodsは言語対照学習＋4 SSLしか列挙していない。教師ありを図から落とした版ではない。本文側へ記述を戻すか、図側を変更するかは別途判断が必要。

## 8. 未特定・不一致・原稿未掲載の項目

| 項目 | 今回確認できた範囲／未解決点 |
| --- | --- |
| RQ2支持面分類31/40 | 原稿値と候補の生データrunは特定。40件の確定リスト、読み出しwindow、prototype保存先、元スコアCSV、集計コード・コマンドは**未特定**。 |
| RQ2 wobble Fz短時間0/20 | 短時間用保存JSONは11件。5秒20試行を0.8秒で数えれば0/20だが、原稿がどちらに由来するか**未特定**。 |
| RQ2コルク19/20 | 元20試行中Enterなし1件をsummaryが除外し19/19、原稿は失敗を含め19/20。分母の扱いが異なる。 |
| RQ2 architecture比較 | Token-Fusionは8月旧校正、CNN/MLPは9月remount校正。完全に同じ校正・収集セッションではない。 |
| 0.300秒の反応時間の独立測定 | コード・ラベル・ログへの補正適用は確認済み。原稿がいう「別途測定」の生計測データ・測定手順は**未特定**。 |
| 物理センサ取付けの座標校正 | 保存されたRz180符号変換は確認済み。物理取付け姿勢を裏付ける完全なTF/計測記録は**未特定**。 |
| RQ1 validation | window単位。物体・episode単位のvalidation holdoutではない。原稿表現で混同させないこと。 |
| RQ3 raw dataset root | manifestの旧 `keyboard_descent` rootは現在無い。`insertion_binary` に125件があり、イベントtimestampを照合できた。 |
| RQ3教師ありモデル | 最終PDFには含まれるが、現在の本文の手法列挙からは抜けている。 |
| 原稿が参照するJPG 2枚 | `fig/overview.jpg`、`fig/ft_compare.jpg` は原稿に参照があるが、現在のGenesisの該当パスには存在しない。3リポジトリの通常ファイル検索でも同名ファイルを特定できなかった。 |
| Genesisの歴史的ソース版 | 現在の生成・annotationコードと版別metadataは特定。全収集・全annotationの実行当時の完全なGit commit＋dirty diff＋起動コマンド一式は**未特定**。 |

### 8.1 完了済みだが現原稿のRQ1表に未掲載の教師ありCNN/MLP

最新の追加実験は完了しているが、現在の `tab:model_comparison` の4モデルとは区別する。

```text
/home/user/PatchTST/results/paper_supervised_baselines/rq1_supervised_cnn_mlp_3seed_20260914_001528/
  status.json
  manifest.json
  summary.md
  aggregate_metrics.csv
  ARTIFACTS.md
```

3 seedsのbest結果、Accuracy / Macro F1 [%]：

| 追加モデル | Action | Weight | Interaction |
| --- | --- | --- | --- |
| Supervised CNN | 93.20±0.26 / 93.18±0.25 | 93.24±0.28 / 93.24±0.28 | 79.46±0.38 / 79.20±0.30 |
| Supervised MLP | 93.32±0.09 / 93.28±0.08 | 92.77±0.21 / 92.77±0.21 | 76.73±0.45 / 76.78±0.43 |

checkpointは `$P/saved_models/paper_supervised_baselines/rq1_supervised_cnn_mlp_3seed_20260914_001528/` 以下。完全な個別パスはこのgroupの `manifest.json`、UMAP・混同行列は `ARTIFACTS.md` を参照。

## 9. 原稿のF/T例示図と監査時点

### 9.1 RQ1付近のF/T例示図

- 原稿が参照する画像：`$G/fig/ft_compare.jpg`。**現在このパスに実ファイルはない**。
- 波形選択と元episode情報：`$G/fig/semantic_ft_examples/manifest.json`、`selection.json`、`README.md`。
- 編集用一式：`$G/fig/semantic_ft_examples/editing_bundle/`。`bundle_manifest.json`、`original_figure_manifest.json`、`original_selection.json`、`sources.csv`、`SHA256SUMS.txt`、`source_episodes/`、各軸PDFがある。
- 編集用合成参照：`$G/fig/semantic_ft_examples/editing_bundle/combined_reference.pdf`。
- 作図コード：`$G/src/evaluation/generate_semantic_ft_examples.py`、`export_semantic_ft_editing_bundle.py`、`replot_semantic_ft_bundle.py`。
- 編集用の元波形・PDFの対応はmanifestで追えるが、**最終JPGそのものと、その手編集・レイアウト合成の全履歴は未特定**。同様に原稿の概要図の参照先 `$G/fig/overview.jpg` も現在存在しない。

### 9.2 調査時点の識別情報

対象原稿SHA256：

```text
ef715d6c0fa43ce3a7ce22a25481f251330238a714e7091dc7f5139c5e04ca94
```

調査時点のGit HEAD：

| リポジトリ | HEAD |
| --- | --- |
| Genesis | `e8f52067b4773935677a5742a2873096ec3df135` |
| PatchTST | `d745c289295fd1365dd211eea553c8ec3f75923f` |
| ur3_robotiq_ros2 | `bdf04375035159f6616df2ab8c4ca0744b49250d` |

**全リポジトリに未コミット変更があるため、このHEADだけで実験時のコードを復元できるわけではない。** 再現にはrun内の `code/` snapshot、checkpoint設定、manifestのhash、当時のepisode設定を優先する。

今回の確認範囲：RQ1の12 runsについてbest/last checkpointと3属性の予測・UMAP・混同行列、合計240パスを確認。RQ3の追加BYOL・教師ありについて、140条件のcheckpoint・指標・history・予測・混同行列と代表UMAP、合計712パスを確認した。いずれも欠損なし。RQ3の125生episodesは保存イベント時刻を照合し、原稿のRQ3 PDF 2枚は最終結果groupとhash一致を確認した。これは保存物の監査であり、全モデルの再推論を行ったという意味ではない。

### 9.3 再実行・移設時に守ること

1. 原稿と同じ版のCSV・checkpoint・prototype・正規化統計をまとめて指定する。単に「最新」「terrain」「best」で探索し直さない。
2. 対照学習の評価・UMAPには同じ定型文prototypeを使う。releaseは1クラス・1文章とし、embedding平均を再導入しない。
3. RQ1はbest/lastとseedを区別し、予測CSV・UMAP・混同行列・集計MDを保存する。表はbest、mean±sample SD、百分率。
4. RQ3は0.3秒補正済みdatasetを使い、episode split、各seedのnested subset、100 train episodes由来の正規化を維持する。
5. RQ2の再集計では無効参照時刻・早期停止・timeoutの分母規則を明記する。未特定の試行を勝手に補って論文値に合わせない。
6. 既存runを上書きせず新groupへ出力する。symlinkのリンク先、原稿PDFの生成元CSV、実行時ソースも一緒に保存する。
