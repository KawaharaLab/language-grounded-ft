# Genesis overlay modifications

This directory is an overlay for Genesis at commit
`8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44`, licensed under Apache License
2.0. The copied `LICENSE` applies to the upstream-derived files.

The following upstream files are modified and carry a notice at the top of
the file:

- `genesis/utils/array_class.py`: adds per-link `contact_torque` state.
- `genesis/engine/solvers/rigid/constraint/solver.py`: resets and accumulates
  direct-contact torque about each link origin.
- `genesis/engine/entities/rigid_entity/rigid_entity.py`: exposes root center
  of mass and local/world per-link contact force and torque accessors.

`src/`, the focused contact-torque test, and the annotation tests capture the
research code from local fork commit `e8f52067b4773935677a5742a2873096ec3df135`
plus the relevant uncommitted working-tree changes present on 2026-09-30.
Those files implement the current paper simulation data generation, annotation,
dataset construction, publication figures, and RQ1 audit experiments.

The copied research tree is unchanged except for portable path resolution in
the following source files and two new runner/smoke files:

- generation: `run_genesis_sim.py`, `run_rigid_two_rounds.sh`; new
  `run_one_episode.sh` and `smoke_cpu.py`
- processing: `1_create_annotations.py`, `2_create_train_csv.py`,
  `3_bert_eval.py`, `3_bert_train.py`, `create_independent_eval_sets.py`,
  `create_support_contrast_sets.py`, and `create_train_balanced_set.py`
- evaluation: `export_semantic_ft_editing_bundle.py`,
  `generate_semantic_ft_examples.py`, `replot_semantic_ft_bundle.py`, the RQ2
  `audit_real_object_contact_windows.py`,
  `evaluate_terrain_force_baseline_offline.py`,
  `evaluate_wobble_lift_and_lower.py`, `lift_and_lower_baseline.py`,
  `lift_and_lower_object.py`, and their real-log loaders
  `generate_real_model_surface_ft_examples.py` and
  `generate_real_surface_ft_examples.py`; supplementary-video source
  `generate_pick_place_force_videos.py`; and
  `reexport_terminal_insertion_fonts.py`,
  `replot_terminal_insertion_errorbars.py`, and
  `replot_terminal_insertion_without_supervised.py`
- experiments: `audit_rq1_surface_supervised.py`,
  `rq1_language_composition/common.py`, `prepare.py`, `prepare_data.py`,
  `run_queue.py`, `test_captions.py`, and `test_training.py`

`src/workspace_paths.py` is new and centralizes the environment-based path
resolution. These portability edits replace machine-specific absolute paths;
they do not alter labels, filtering rules, window sizes, physics settings, or
experiment defaults. `run_rigid_two_rounds.sh` retains the recorded 0814
defaults while exposing them as environment overrides.

The overlay is intentionally limited to the current paper dependency closure.
It excludes the fork's deformable/FEM and rigid-elastic benchmarks, SAP-only
engine patch, OpenAI-vision runner, VLA work, exploratory manipulation
policies and parameter sweeps, old dataset utilities, and unpublished RQ1
action-interaction and semantic-geometry experiments. The public upstream
checkout remains complete; only research-fork additions outside the paper
scope are omitted.

Large generated data, checkpoints, paper figures, videos, caches, the OpenPI
submodule, and VLA source are intentionally outside this overlay.
