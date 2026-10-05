# RQ1: supervised CNN and MLP

## Launch

```bash
cd ${FT_WORKSPACE_ROOT}/PatchTST
bash scripts/run_paper_supervised_baselines.sh
```

The launcher starts a detached tmux session. GPUs 0 and 1 must be available; each GPU runs one job at a time, taking the next available job from a shared queue. Override `GPUS="0"` to use only one GPU. Six new training runs use seeds 42, 43, 44 independently for each architecture. W&B uses the `patchtst_tokenfusion_supervised` project and the new group name; use `WANDB_MODE=disabled` if needed.

This extends the RQ1 supervised classification comparison, not the terminal-insertion adaptation experiment. No pretraining weights are loaded for the new CNN/MLP runs.

## Matched architectures

The encoder dimensions come from the exact contrastive CNN/MLP checkpoints used in RQ1, not a fresh parameter search.

| Model | Encoder | Trainable encoder parameters | Trainable parameters including three classification heads |
| --- | --- | ---: | ---: |
| CNN | 4 Conv1d stages, channels 81 / 162 / 324 / 324, kernel 5, BatchNorm, GELU, MaxPool(2), dropout 0.2; global average pooling and linear 324→768 | 1,109,172 | 1,117,631 |
| MLP | Flatten 80×12 → 960; 960→429→429→429→768, GELU/dropout 0.2 after each hidden layer; no pooling | 1,111,449 | 1,119,908 |
| Existing supervised PatchTST | Raw Token-Fusion encoder → 768 | 1,111,168 | 1,119,627 |

The PatchTST implementation also retains three frozen parameters; its total stored parameter count is 1,119,630. The counts above consistently refer to trainable parameters.

All supervised models apply head dropout 0.2, then three linear heads: action (6), weight (2), interaction (3). Classification uses logits and cross entropy, not text prototypes. Unlike contrastive encoders, supervised features are not L2-normalized; this matches the existing supervised Token-Fusion model. The default contrastive CNN/MLP behavior and checkpoint parameter names remain unchanged.

## Data and training

- Training: `${FT_WORKSPACE_ROOT}/Genesis/data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv`, 437,244 usable windows.
- Same `terrain0814_eval` label taxonomy as the RQ1 supervised reference, including release / release midair merged into release and the existing transition-label canonicalization.
- Validation: action-stratified 10% split from the training CSV, 393,519 train / 43,725 validation windows. Seed controls splitting, initialization and data order. This deliberately preserves RQ1's window-level split; train and validation are not episode-disjoint.
- Per-axis normalization statistics use only that seed's training subset. The runner verifies exact equality with the corresponding RQ1 supervised PatchTST statistics and checks the saved split indices against the original splitting rule.
- Mean cross entropy over the three tasks; inverse-frequency class weights computed on the training subset only. No label smoothing.
- Adam, peak LR 2e-3, batch size 512, weight decay 0, clipping 1; warmup 10 epochs, cosine to 1e-5 at 400 epochs; scheduler steps at epoch end.
- Best and early stopping: validation mean task Macro F1, patience 75, min delta 1e-4, exactly as the existing supervised PatchTST. Test data are not used for selection.
- Independently balanced RQ1 evaluation: action 10,008 windows, weight 6,000, interaction 3,675. The runner copies the existing prepared evaluation CSVs and class maps; all 23 source index/prototype hashes must match the original RQ1 manifest.

## Outputs

Results: `results/paper_supervised_baselines/<GROUP>/`. Checkpoints: `saved_models/paper_supervised_baselines/<GROUP>/`.

- `manifest.json`: exact commands, architecture settings, source checkpoints, input/code hashes. Training and evaluation run from the frozen `code/` snapshot.
- `status.json`, `runner.log`, `logs/*.status.json`, `logs/*.train.log`: progress. Each checkpoint directory also contains `history.csv`, `split_indices.npz`, normalization JSON, best and final checkpoints.
- `evaluation/supervised_{cnn,mlp}/seed{42,43,44}/{best,last}/`: `action`, `weight`, `interaction` each have `*_pred.csv`, `*_umap.png`, `*_confusion_matrix.png`; metrics in `balanced_eval_summary.md` / `.json`.
- `ARTIFACTS.md`: direct links to predictions, UMAP and confusion matrices, including the existing RQ1 methods.
- `summary.md`: six-model comparison, Accuracy / Macro F1 in percent, mean ± sample standard deviation (ddof=1), for both best and last.
- `per_seed_metrics.csv`, `aggregate_metrics.csv`: unrounded metrics on the 0–1 scale.
- `validation.json`, `logs/*.train.audit.json`: split/normalization checks and unchanged reference-metric verification.

Final means are generated only when all three seeds and both checkpoints have valid predictions and all figures. Existing PatchTST / contrastive results and the manuscript are not overwritten. Review the complete results before modifying the paper.

To refresh the aggregate or artifacts without training:

```bash
.venv/bin/python scripts/paper_supervised_baselines.py summarize --group <GROUP>
```

The same group may retry evaluation after completed training. It refuses to overwrite a partial training directory silently; use a new group or explicitly resolve the partial run first.

Completed fits with both checkpoints and `summary.json` are reused even if a post-training audit failed before its completion marker was written. The split audit encodes masked/transition action labels with the training code's `IGNORE_INDEX`, rather than passing missing values to stratification, and mirrors the training fallback for singleton classes. Missing action labels remain masked; neither input data nor the learned weights are changed by this audit.
