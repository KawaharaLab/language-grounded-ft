# Token-Fusion BYOL experiment

## Method and scope

This experiment replaces the custom JEPA baseline with BYOL for the manuscript's
frozen-encoder linear evaluation. It does not relabel or overwrite previous JEPA
checkpoints, change manuscript numbers, or rerun terminal-insertion adaptation.

Reference: Grill et al., *Bootstrap Your Own Latent*, NeurIPS 2020:
https://arxiv.org/abs/2006.07733

Official implementation:
https://github.com/google-deepmind/deepmind-research/tree/master/byol

The online encoder processes two independently augmented views. A separate
projector and online-only predictor each use Linear–BatchNorm–ReLU–Linear, with
hidden dimension 4096 and projection dimension 256. The objective is the sum of
the two cross-view squared distances between L2-normalized predictions and
stop-gradient target projections. No variance penalty or negative pairs are
added. Both encoder and projector parameters in the target branch follow an EMA
whose momentum increases from 0.996 to 1 by a cosine schedule over optimizer
steps. As in the official implementation, the target branch runs in training
mode during pretraining: its BatchNorm buffers update from its own batches,
without gradients. Validation puts both branches in evaluation mode and does
not update their running statistics.

F/T-specific adaptations are explicit: the common 1,111,171-parameter Raw
Token-Fusion encoder replaces the image encoder, and the two views use the same
augmentations as the existing VICReg baseline: per-channel Gaussian gain
variation (standard deviation 0.1), additive normalized noise (0.03), independent
time-point masking (0.4, shared across channels at a time point), and channel
dropout (0.1). Channel identity and temporal order are not permuted. The training
optimizer and budget follow the manuscript SSL protocol rather than the
original ImageNet LARS setup. This is BYOL adapted to F/T, not an ImageNet
reproduction. The extra projector/predictor are training-only; the common
768-dimensional encoder output is used for linear probing.

## Fixed comparison protocol

- Seed 42; normalized 80-by-12 inputs; no delta features.
- Same 437,244-window training CSV as the manuscript SSL results; 10% validation
  split reproduced from seed 42. Normalization uses only the training split;
  its counts and statistics must match the reference experiment before training.
- AdamW, batch size 512, peak learning rate 0.002, weight decay 0.0001.
- Up to 400 epochs, 10-epoch warmup, cosine decay, patience 75, minimum validation
  loss improvement 0.00001. Early stopping may end before the cosine endpoint.
- Best selection uses only validation BYOL loss. Fixed validation augmentation
  randomness is reused each epoch. Representation spread is logged to W&B for
  collapse monitoring; test accuracy is never used for checkpoint selection.
- For both best and last: freeze the encoder; train one linear head per task,
  up to 200 epochs, AdamW learning rate 0.01, weight decay 0, batch size 512, patience 30, balanced
  class weights, and the same training/validation split as pretraining.
- Action: 10,008 windows / 6 classes; weight: 6,000 / 2; interaction: 3,675 / 3.
  `release midair` is merged into `release`. The exact existing evaluation CSVs
  are reused. These supervised linear probes do not use text prototypes.

## Run and outputs

```bash
cd ${FT_WORKSPACE_ROOT}/PatchTST
GROUP=tokenfusion_byol_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s "$GROUP" \
  "cd ${FT_WORKSPACE_ROOT}/PatchTST && GPU=0 GROUP=$GROUP bash scripts/run_tokenfusion_byol.sh"
tmux attach -t "$GROUP"
```

Check that the chosen GPU is free before launching. W&B is online by default;
set `WANDB_MODE=offline` explicitly if needed. The launcher refuses to overwrite
an existing output group, verifies reference CSV hashes, snapshots Python code
and index CSVs, and links the raw data read-only by usage (raw CSVs themselves are
not copied; do not modify them during the run).

Outputs under `results/tokenfusion_byol/<GROUP>/`:

- `logs/pipeline.log`, `logs/train.log`, `logs/linear_probe_{best,last}.log`.
- `status.json`, `manifest.json`, `commands.sh`, `data_audit.json`.
- `linear_probes/{best,last}/{action,weight,interaction}_umap.png`.
- `linear_probes/{best,last}/{action,weight,interaction}_confusion_matrix.png`.
- Per-task prediction CSVs, linear heads, learning curves, and summary Markdown.
- `summary.md`: best/last BYOL and the three existing manuscript SSL baselines.
- `ARTIFACTS.md`: direct links to every UMAP, confusion matrix, and prediction CSV.

Checkpoints and pretraining curves are under
`saved_models/tokenfusion_ssl/<GROUP>/tokenfusion_ssl_byol_seed42/`.
Training curves are saved each epoch, and W&B logs encoder/projector/target
normalized feature standard deviations, encoder norms, prediction agreement,
EMA momentum, and training/validation losses. Low BYOL loss alone is not proof
of a useful representation. Review spread diagnostics and linear-probe results
before updating the manuscript.

## Completed run and downstream follow-up

`tokenfusion_byol_20260912_194926` finished with validation-best epoch 8 and last
epoch 83 (early stopping). Best-encoder Accuracy / Macro F1, in percent:
action 77.01 / 76.88, weight 83.22 / 83.21, interaction 74.18 / 73.84.
These are newly trained BYOL results, not renamed JEPA results.

The separate terminal-insertion follow-up uses the best encoder and the exact
historical five-seed adaptation protocol. See
[terminal_insertion_byol.md](terminal_insertion_byol.md) for its launcher,
frozen/LoRA protocols, comparison figures and diagnostics.
