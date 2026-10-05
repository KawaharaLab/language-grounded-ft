# RQ1 supervised encoder in the terminal-insertion comparison

## Run

```bash
cd ${FT_WORKSPACE_ROOT}/PatchTST
GROUP=terminal_insertion_supervised_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s "$GROUP" "GROUP=$GROUP bash scripts/run_terminal_insertion_supervised.sh"
```

Both GPUs 0 and 1 must be available. The entry point uses the shared runner in `run_terminal_insertion_byol.sh` with `MODEL=supervised`. It refuses an existing output group, snapshots the training code and dataset, records hashes, and logs to W&B under `terminal-insertion-adaptation-efficiency`.

## What is transferred

Use exactly the supervised seed 42 validation-best checkpoint listed in the RQ1 three-seed manifest:

`saved_models/tokenfusion_supervised/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515/tokenfusion_supervised_multitask_seed42/tokenfusion_supervised_multitask_seed42_best.pth`

This is epoch 348 of the normalized supervised multi-task run. RQ1 selects it using validation classification performance, not terminal-insertion test performance. Its encoder has 1,111,171 parameters and returns 768-dimensional features. The original action (6 classes), weight (2 classes), and interaction (3 classes) heads are discarded; their logits are not used as downstream features. No supervised pretraining is repeated.

## Matched downstream protocol

- Same corrected real-robot dataset as the existing comparisons: insertion reference = Enter minus 0.30 s; train/validation/test = 100/5/20 episodes. No new robot data collection.
- Same nested labeled episode subsets: N = 1, 2, 5, 10, 25, 50, 100; downstream seeds 42–46. All pretraining methods use their seed 42 checkpoint, rather than five separate pretrained encoders.
- Fixed input normalization from all 100 real training episodes for every N.
- Frozen encoder + a new 769-parameter linear head: AdamW LR 2e-3; feature-coordinate standardization uses only the selected N training episodes, exactly as in the other frozen baselines.
- Q/V LoRA, rank 4 and alpha 8, plus a new binary head: 8,961 trainable parameters; AdamW LR 5e-4; pretrained BatchNorm statistics frozen. The LoRA pipeline does not standardize encoder features.
- Both adaptations: batch size 256, weight decay 1e-4, max 300 epochs, validation-loss early stopping patience 30 / min_delta 1e-5, ReduceLROnPlateau, validation-best checkpoint, fixed probability threshold 0.5.

The 70 new fits are compared with the existing Contrastive, Masked reconstruction, VAE, VICReg, BYOL, and Scratch results. Those reference checkpoints/results are linked, not modified or retrained. JEPA is not reintroduced.

## Results and checks

Outputs: `results/terminal_insertion_supervised/<GROUP>/`.

- `status.json` and `logs/`: progress and failures.
- `SUMMARY.md`: mean ± sample standard deviation (ddof=1) of Macro F1 in percent.
- `pretrained_linear_vs_scratch_end_to_end_macro_f1_5seed.pdf` / `.png` and `pretrained_lora_vs_end_to_end_macro_f1_5seed.pdf` / `.png`: seven-method comparison plots, also in percent.
- `*_all_seed_results.csv` / `*_aggregate.csv`: unrounded results on the original 0–1 scale.
- `pretrained_frozen/supervised/seed_<seed>/n_<NNN>/` and `pretrained_adaptation/supervised/lora/seed_<seed>/n_<NNN>/`: checkpoints, histories, test predictions, and confusion matrices for all 70 fits.
- `umap.png` and `umap_coordinates.csv` for the prespecified seed 42 and N = 1, 5, 100 in both adaptation modes; `ARTIFACTS.md` links all diagnostic plots.

The finalizer re-encodes the 979 test windows and verifies predictions, Accuracy and Macro F1 against saved checkpoints, CSVs, episode IDs and timing. It also requires all retained comparison rows and episode subsets to match the prior experiment. Training scripts do not automatically overwrite the manuscript; figures and RQ3 text are published after reviewing complete, verified results, with backups.
