# PatchTST research component

The public PatchTST source is pinned at `204c21efe0b39603ad6e2ca640ef5896646ab1a9`, the exact merge base of the local research branch and the current public `main`. The Apache-2.0 license is retained in both the upstream checkout and the research overlay. The local source was inspected read-only at commit `d745c289295fd1365dd211eea553c8ec3f75923f`; the overlay also captures its modified and untracked model, training, evaluation, runner, and experiment-documentation files.

After repository bootstrap, `workspace/PatchTST` combines `third_party/patchtst` with `research/patchtst`. Set `FT_WORKSPACE_ROOT` only when the generated workspace is elsewhere. Historical CSV and checkpoint metadata can retain host paths: the loader translates recorded `/home/user/Genesis`, `/home/user/PatchTST`, `/home/user/ur3_robotiq_ros2`, and `/mnt/nvme1/user/data` prefixes into workspace siblings at read time.

The source overlay excludes datasets, checkpoints, logs, predictions, and generated result trees. Their exact source paths, portable destinations, and measured byte counts are in `manifests/learning_patchtst.json`. The two Genesis train/evaluation directories alone require about 196 GB, so artifact attachment uses external symlinks via tools/link_local.py.

The publication overlay is intentionally limited to the current paper dependency closure. It retains the final Token-Fusion PatchTST, CNN and MLP contrastive models; supervised four-head reference; masked reconstruction, BYOL, VAE and VICReg support; final RQ1 classification/evaluation runners; terminal-insertion preparation, LoRA, frozen-probe, full-finetune and scratch paths; shared normalization, taxonomy, checkpoint and loader code; and the final experiment documentation. Shared baseline classes remain even when a standalone exploratory runner was removed because archived checkpoints and evaluators import those symbols.

The pruning removed 31 files devoted to the unused CLIP/image experiment, old forecasting-style pretraining entry points, TCN/LSTM and iTransformer architecture screens, delta/loss/LR hyperparameter sweeps, early 0.5-million-window trials, duplicate supervised launchers, old label rendering/slip scratch analysis, and sweep-only summarizers. The immutable public upstream submodule remains complete; pruning applies only to the research overlay and therefore does not remove upstream PatchTST code.

Run the offline checks with:

```bash
python3 -m unittest tests.test_learning_patchtst
python3 -m compileall -q workspace/PatchTST/PatchTST_self_supervised workspace/PatchTST/scripts
```

The smoke test exercises 80-sample force windows, upstream patch extraction, an encoder forward pass, normalized embeddings, strict checkpoint restoration, and legacy-path translation without downloading a model or dataset.

The same suite also reads one real 0814 training row whose byte-preserved `csv_path` still records `/home/user/Genesis/...`. It loads that force window and its real text embedding through the generated workspace, forms an actual one-item batch, and runs under the repository runtime guard with direct access to original source code denied and registered artifact links allowed. This directly checks the main contrastive-training loader rather than relying only on cached RQ1 arrays.

## Reproducible learning environment

The source environment used CPython 3.10.0 and uv 0.8.3. `env/learning-requirements.lock` is the hash-locked full environment for training, evaluation, plotting, language-model embedding, UMAP, and W&B workflows. Every version comes from the source `uv.lock`. The smaller `env/learning-replay-requirements.lock` contains only the NumPy, pandas, PyTorch, and scikit-learn closure needed by the authoritative CPU replay.

Create it from the repository root:

```bash
tools/setup_env.sh learning
```

This lock deliberately retains the source PyTorch 2.8.0 CUDA dependency set on Linux. The replay itself forces CPU inference and does not require a GPU. The source environment was created by uv and had no `pip` module, so `pip freeze` is not a reliable record; the versions were recovered from its immutable `uv.lock` and checked against `uv tree`.

For an audit-only environment, substitute `env/learning-replay-requirements.lock` in the `uv pip sync` command.

## Authoritative RQ1 replay

The current manuscript reference is the September 17 four-head supervised MLP, not the older three-head PatchTST baseline. It has a shared 1,111,449-parameter MLP and separate action, weight, interaction, and support-surface heads. Seeds 42, 43, and 44 use episode-level validation and their own minimum four-head validation loss. The fixed best-checkpoint Macro F1 values are 89.31 ± 0.34% for action, 91.12 ± 0.50% for weight, and 75.34 ± 0.82% for interaction.

Import `artifacts/PatchTST/results/rq1_language_composition/rq1_20260916_235337` and the Genesis train/evaluation artifacts first. The experiment directory is 3,338,175,800 bytes. Then run the existing audit through the portable adapter:

```bash
tools/run.py learning -- "$PWD/env/learning-replay-rq1.py" \
  --workspace "$PWD/workspace" \
  --out "$PWD/workspace/reproduced/rq1_guarded_replay"
```

The adapter runs `workspace/Genesis/src/experiments/audit_rq1_surface_supervised.py`, translates paths embedded in both JSON values and dictionary keys, checks source and checkpoint hashes, reconstructs episode-disjoint splits and validation-only selection, and replays all six best/last checkpoints on CPU. Results go to `workspace/reproduced/rq1_guarded_replay`. `tools/run.py` enables the runtime guard, which rejects direct dependencies on the original source-code trees while permitting registered artifact links. The replay performs no training and does not select a checkpoint from YCB performance.

The authoritative best checkpoint hashes are:

| Seed | SHA-256 |
|---:|---|
| 42 | `cb5210e3f9817871d1c9e75250553431dbae8b9cf69cfc1035fd55d011237b63` |
| 43 | `b77d0e7017ea18f85e10ef90478367bbebde3fc2a6012bbb67f57f9f62f05303` |
| 44 | `77c96aa53985e8e1c7cd199c2de1cfe896fb79d21b80cfb4aec49d98668316f3` |

The portable replay was executed on the imported artifacts and passed for best and last checkpoints from all three seeds. It reproduced action, weight, and interaction Macro F1 means of 0.893059, 0.911216, and 0.753444 for the best selection, matching the manuscript percentages above.

The public source overlay was also scanned for embedded credentials. Legacy hardcoded W&B credentials in the callback and four training CLI defaults were removed from this repository; they now use the normal `WANDB_API_KEY` environment variable or an existing W&B login. `tests.test_learning_credentials` combines text-pattern checks with Python AST checks for assignments, mappings, keyword arguments, and `argparse` defaults. A direct equality scan against the credential retained only in the read-only original found zero occurrences in the overlay. The original repository was not changed. Linked artifact snapshots remain unchanged at their original locations and excluded from Git; their archived `code/` directories require an explicit credential review before any publication export.

## RQ1 contrastive training and fixed-prototype evaluation

The command below reproduces the paper's Token-Fusion PatchTST architecture and optimization for one seed. It writes a fresh checkpoint directory and disables external logging. Run seeds 42, 43, and 44 by changing `--seed` and the two output names.

```bash
mkdir -p workspace/reproduced/rq1_contrastive/seed42
WANDB_MODE=disabled tools/run.py learning -- patchtst_bert_normalized.py \
  --data_dir "$PWD/workspace/Genesis/data/train_08142026" \
  --index_csv train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv \
  --emb_dir bert_emb_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms \
  --epochs 400 --batch_size 512 --peak_lr 2e-3 --min_lr 1e-5 \
  --warmup_epochs 10 --early_stopping_patience 75 \
  --early_stopping_min_delta 1e-4 --val_from_train_pct 0.10 \
  --num_workers 4 --zero_force_eps 1e-8 --loss_type infonce \
  --seed 42 --from_scratch --patch_len 10 --stride 10 \
  --n_layers 3 --n_heads 16 --d_model 128 --d_ff 768 \
  --dropout 0.2 --head_dropout 0.2 --token_fusion_layers 1 \
  --token_fusion_heads 8 --token_fusion_ff_mult 4 --out_embed_dim 768 \
  --arch token_fusion --pool mean --input_features raw \
  --run_name rq1_tokenfusion_seed42_reproduced \
  --save_dir "$PWD/workspace/reproduced/rq1_contrastive/seed42"
```

This uses the final terrain/floor/wobble/contact-clean/slip-5-mm/s annotation and embedding variant. To regenerate and audit the fixed caption bank and language prototypes into a fresh directory, use the already frozen language-composition preparation stage with a local all-mpnet-base-v2 model:

```bash
export FT_TEXT_MODEL="$PWD/artifacts/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/<snapshot>"
tools/run.py simulation -- src/experiments/rq1_language_composition/prepare.py \
  --run-dir "$PWD/workspace/reproduced/rq1_language_prototypes"
```

The preparation stage checks the 437,244-row source hash, freezes episode splits and all four annotation heads, builds the 250 canonical meanings and 32 diverse captions per meaning, and verifies canonical evaluation embeddings against the archived prototypes. It does not modify the imported source data.

Evaluate the reproduced best checkpoint against the exact fixed prototype archive used by the paper:

```bash
tools/run.py learning -- patchtst_eval_normalized.py \
  --model_path "$PWD/workspace/reproduced/rq1_contrastive/seed42/rq1_tokenfusion_seed42_reproduced_best.pth" \
  --ckpt_type contrastive --model_arch patchtst --arch auto --pool mean \
  --patch_len 10 --stride 10 --n_layers 3 --n_heads 16 \
  --d_model 128 --d_ff 768 --head_dropout 0.2 \
  --token_fusion_layers 1 --token_fusion_heads 8 \
  --token_fusion_ff_mult 4 --out_embed_dim 768 \
  --eval_dir "$PWD/workspace/artifacts/PatchTST/results/terrain_contrastive/terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/prepared_eval_templates_release_merged" \
  --eval_csv eval_08142026_terrain_independent_balanced.csv \
  --targets action,weight,interaction --text_type bert \
  --zero_force_eps 1e-8 --skip_plots \
  --output_name rq1_reproduced_seed42_best
```

The evaluator writes predictions and metrics below `workspace/PatchTST/PatchTST_self_supervised/data/rq1/rq1_reproduced_seed42_best`. The archived prototype descriptions merge release and release-midair as specified by the paper; no prompt is selected using test scores.

## Primary RQ3 contrastive-LoRA and scratch comparison

The main RQ3 adaptation result uses the seed-42 contrastive Token-Fusion encoder, then trains Q/V LoRA and a binary head. The matched scratch baseline trains the same Token-Fusion architecture end to end. These commands reproduce seed 42 at N=5 and write fresh outputs; both perform validation-only checkpoint selection and then evaluate the selected model exactly once on the fixed test split.

```bash
DATASET="$PWD/workspace/artifacts/PatchTST/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827/dataset/terminal_insertion_windows.npz"
MANIFEST="$PWD/workspace/artifacts/PatchTST/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827/dataset/dataset_manifest.json"
CONTRASTIVE="$PWD/workspace/artifacts/PatchTST/saved_models/terrain_contrastive/terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42_best.pth"

tools/run.py learning -- train_terminal_insertion_probe.py \
  --dataset "$DATASET" --manifest "$MANIFEST" \
  --model contrastive --checkpoint "$CONTRASTIVE" \
  --pretrained-adaptation lora --lora-rank 4 --lora-alpha 8 \
  --train-episode-count 5 \
  --output-dir "$PWD/workspace/reproduced/rq3_primary/lora_r4/seed_42/n_005" \
  --epochs 300 --patience 30 --min-delta 1e-5 \
  --learning-rate 5e-4 --weight-decay 1e-4 \
  --batch-size 256 --num-workers 4 --seed 42 --device cuda \
  --fixed-threshold 0.5 --wandb-project terminal-insertion-reproduction \
  --wandb-group rq3_primary_reproduced --wandb-mode disabled

tools/run.py learning -- train_terminal_insertion_probe.py \
  --dataset "$DATASET" --manifest "$MANIFEST" \
  --model scratch_tokenfusion --train-episode-count 5 \
  --output-dir "$PWD/workspace/reproduced/rq3_primary/scratch/seed_42/n_005" \
  --epochs 300 --patience 30 --min-delta 1e-5 \
  --learning-rate 3e-4 --weight-decay 1e-4 \
  --batch-size 256 --num-workers 4 --seed 42 --device cuda \
  --fixed-threshold 0.5 --wandb-project terminal-insertion-reproduction \
  --wandb-group rq3_primary_reproduced --wandb-mode disabled
```

For the full main comparison, run both commands for seeds 42–46 and episode counts 1, 2, 5, 10, 25, 50, and 100. Preserve the historical seed-42 N=100 exceptions: contrastive-LoRA and scratch use `--patience 0`; their saved runs consequently used scheduler patience 3 instead of 7. All other conditions use the arguments above. Do not replace the reported five-seed comparison with the four-seed sensitivity analysis.

As configuration anchors, the archived seed-42 N=5 LoRA run used 262 training windows, LR 0.0005, 8,961 trainable parameters, fixed threshold 0.5, and selected epoch 300. Its scratch counterpart used the same 262 windows, LR 0.0003, 1,111,937 trainable parameters, fixed threshold 0.5, and selected epoch 19. These values come from each imported `metrics.json`; they are checks on the recipe, not targets for checkpoint selection.

## RQ3 supervised-pretrained LoRA training and evaluation

The final RQ3 supervised-transfer comparison uses the seed-42 four-task supervised Token-Fusion encoder, Q/V LoRA rank 4 with alpha 8, fixed threshold 0.5, and the Enter−300 ms terminal-insertion dataset. This one-seed, full-100-episode command writes a fresh result and performs validation selection followed by its fixed test evaluation:

```bash
tools/run.py learning -- train_terminal_insertion_probe.py \
  --dataset "$PWD/workspace/artifacts/PatchTST/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827/dataset/terminal_insertion_windows.npz" \
  --manifest "$PWD/workspace/artifacts/PatchTST/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827/dataset/dataset_manifest.json" \
  --model supervised \
  --checkpoint "$PWD/workspace/artifacts/PatchTST/saved_models/tokenfusion_supervised/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515/tokenfusion_supervised_multitask_seed42/tokenfusion_supervised_multitask_seed42_best.pth" \
  --pretrained-adaptation lora --lora-rank 4 --lora-alpha 8 \
  --train-episode-count 100 \
  --output-dir "$PWD/workspace/reproduced/rq3_supervised_lora/seed42/n_100" \
  --epochs 300 --patience 30 --min-delta 1e-5 \
  --batch-size 256 --num-workers 4 --seed 42 --device cuda \
  --fixed-threshold 0.5 --wandb-project terminal-insertion-reproduction \
  --wandb-group rq3_supervised_lora_reproduced --wandb-mode disabled
```

Repeat with seeds 43–46 and episode counts 1, 2, 5, 10, 25, 50, and 100 for the complete data-efficiency grid. Each checkpoint records its exact episode subset, normalization, LoRA targets, optimizer settings, threshold, predictions, and test metrics. The historical two seed-42 N=100 optimizer exceptions belong to the older contrastive-LoRA and scratch curves; they do not change this supervised-pretrained command.

Validate the helper without artifacts or long inference using:

```bash
python3 -m unittest tests.test_learning_replay tests.test_learning_credentials
```
