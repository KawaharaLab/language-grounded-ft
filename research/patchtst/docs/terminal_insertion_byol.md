# BYOL: terminal-insertion comparison

Run `bash scripts/run_terminal_insertion_byol.sh` in tmux with GPUs 0 and 1 available.
The script refuses to overwrite an existing output directory. `GROUP` selects a new run name; `WANDB_MODE=offline` is available if needed.

## Fixed protocol

- BYOL checkpoint: `saved_models/tokenfusion_ssl/tokenfusion_byol_20260912_194926/tokenfusion_ssl_byol_seed42/tokenfusion_ssl_byol_seed42_best.pth` (pretraining seed 42, validation-best epoch 8).
- Corrected insertion dataset: Enter minus 0.30 seconds; 125 episodes split into train 100, validation 5, test 20. Input statistics always come from all 100 training episodes, not from validation/test or simulated data.
- Downstream seeds 42–46, each using nested N = 1, 2, 5, 10, 25, 50, 100 training episodes. These are five downstream runs, not five independent pretrained encoders.
- Frozen encoder: 768-dimensional output, per-feature standardization using the selected N training episodes, one 769-parameter linear classifier, AdamW LR 2e-3.
- LoRA: Q/V projections in every attention block, rank 4, alpha 8; adapters plus classifier total 8,961 trainable parameters. AdamW LR 5e-4; frozen BatchNorm statistics; no feature standardization beyond the common input normalization.
- Both: max 300 epochs, batch size 256, weight decay 1e-4, validation loss early stopping with patience 30 and min_delta 1e-5, ReduceLROnPlateau, validation-best checkpoint, fixed probability threshold 0.5.
- The existing Contrastive, Masked reconstruction, VAE, VICReg and Scratch comparisons are linked read-only by convention into the new results tree. Their trained checkpoints and result files are not modified. JEPA remains preserved in the historical results but is excluded from the new comparison figures.

## Execution and outputs

The runner snapshots Python training sources and the corrected dataset, records SHA-256 hashes, fits 35 frozen probes and 35 LoRA models, and checks episode subsets against the historical scratch runs. It uses the original frozen-probe trainer (including feature standardization), rather than the generic trainer's different frozen mode.

`results/terminal_insertion_byol/<GROUP>/status.json` records progress or failures; `logs/pipeline.log`, `logs/frozen.log` and `logs/lora_seed*_n*.log` contain training progress. W&B uses project `terminal-insertion-adaptation-efficiency` and group `<GROUP>`.

After fitting:

- `SUMMARY.md`: BYOL frozen, BYOL LoRA and scratch Macro F1, mean ± sample SD (ddof=1), for every N.
- `pretrained_linear_vs_scratch_end_to_end_macro_f1_5seed.pdf` / `.png`: frozen comparison without JEPA.
- `pretrained_lora_vs_end_to_end_macro_f1_5seed.pdf` / `.png`: LoRA comparison without JEPA.
- Both comparison figures display Macro F1 in percent, including the standard-deviation bands and error bars. Stored metric and aggregate CSVs retain the original 0–1 scale; rendering multiplies means and standard deviations by 100 without modifying those files.
- `*_all_seed_results.csv`, `*_aggregate.csv`: all comparison values.
- `pretrained_frozen/byol/seed_<seed>/n_<NNN>/` and `pretrained_adaptation/byol/lora/seed_<seed>/n_<NNN>/`: checkpoints, histories, predictions and confusion matrices.
- UMAP is generated for the prespecified seed 42 and N = 1, 5, 100 in both modes. These six diagnostic conditions are selected before viewing results. They contain `umap.png` and `umap_coordinates.csv`; every condition has a confusion matrix.
- `ARTIFACTS.md`: clickable diagnostic paths; `validation.json`: completion of saved-checkpoint prediction and historical-comparison audits.

The finalizer re-encodes all test windows and recomputes probabilities from saved checkpoints, verifies Macro F1 against prediction CSVs, and requires all non-BYOL comparison rows to match the historical figures to numerical precision. Test results never select epochs or thresholds. Neither the training runner nor finalizer overwrites the Genesis manuscript: after visual review, publish the two PDFs and update the manuscript separately, with backups.

The corresponding simulation SSL result table and its UMAP/confusion plots are in `results/tokenfusion_byol/tokenfusion_byol_20260912_194926/summary.md` and `linear_probes/{best,last}/`.
