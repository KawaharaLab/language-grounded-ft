# Simulation and annotation overlay

The source-only overlay in `research/genesis/` applies to the public Genesis
checkout pinned at `8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44`.
It captures the current paper simulation, annotation, dataset preparation,
RQ1 audit, RQ2 offline evaluation, publication-figure, and supplementary-video
dependency closure from the research fork at
`e8f52067b4773935677a5742a2873096ec3df135`, including the relevant dirty
working-tree files. The bootstrap process copies the overlay into
`workspace/Genesis` after checking out the public pin.

## External paths

Set `FT_WORKSPACE_ROOT` to the directory containing the bootstrapped sibling
repositories. It defaults to the parent of `workspace/Genesis`. These optional
overrides are supported:

| Variable | Default |
| --- | --- |
| `FT_GENESIS_ROOT` | `$FT_WORKSPACE_ROOT/Genesis` |
| `FT_GENESIS_DATA_ROOT` | `$FT_GENESIS_ROOT/data` |
| `FT_PATCHTST_ROOT` | `$FT_WORKSPACE_ROOT/PatchTST` |
| `FT_ROBOT_ROOT` | `$FT_WORKSPACE_ROOT/ur3_robotiq_ros2` |
| `FT_ROBOT_DATA_ROOT` | `$FT_ROBOT_ROOT/data` |
| `FT_PYTHON` | the relevant repository's `.venv/bin/python` |
| `GS_CACHE_FILE_PATH` | repository-local `.cache/genesis` |
| `FT_TEXT_MODEL` | `sentence-transformers/all-mpnet-base-v2` |
| `FT_TEXT_MODEL_REVISION` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |

The simulation expects object meshes at
`$FT_GENESIS_DATA_ROOT/objects/gso/<object>/model.obj` for training and
`$FT_GENESIS_DATA_ROOT/objects/ycb/<object>/model.obj` for evaluation. The
committed evaluation defaults specifically reference YCB `009_gelatin_box`
and `010_potted_meat_can`. The Panda MJCF and meshes come from the pinned
Genesis checkout at `genesis/assets/xml/franka_emika_panda/`. The custom ramp
is included as `src/generation/assets/inclined_ramp.obj`.

## Collect one episode

After materializing the workspace and installing the contained simulation
environment, this command collects one headless CPU episode into a new dataset:

```bash
workspace/Genesis/src/generation/run_one_episode.sh \
  001_chips_can gentle_simple eval_08142026 smoke_one_episode
```

The input mesh must exist at
`workspace/Genesis/data/objects/ycb/001_chips_can/model.obj`. Outputs are
written under `workspace/Genesis/data/smoke_one_episode/csv/`. Images default
off in this one-episode wrapper. Set `GENESIS_CAPTURE_IMAGES=1` to retain them.
The wrapper defaults to CPU; for a GPU run set `GENESIS_DEVICE=gpu` and
`CUDA_VISIBLE_DEVICES` explicitly. `gentle_simple` is a direct target and does
not expand into multiple terrain variants. The simulator can retry once after
an early drop, so a failed manipulation may leave an audited accidental-drop
attempt before the final episode.

Annotate and merge the collected episode without changing the raw CSV:

```bash
env DATASET=smoke_one_episode MAX_ANNOTATION_PROCESSES=1 \
  tools/run.py simulation -- src/processing/rigid/1_create_annotations.py

env MODE=smoke_one_episode OUTPUT_NAME=smoke_one_episode \
  tools/run.py simulation -- src/processing/rigid/2_create_train_csv.py
```

The first command writes an annotation CSV beside the raw episode. The second
writes `workspace/Genesis/data/smoke_one_episode/smoke_one_episode.csv` and
filters all-zero force windows. Embeddings are a separate optional stage:

```bash
env DATA_TYPE=smoke_one_episode TRAIN_CSV=smoke_one_episode.csv \
  tools/run.py simulation -- src/processing/rigid/3_bert_train.py
```

## Recorded 0814 collection configuration

`src/generation/run_rigid_two_rounds.sh` exposes the collection campaign
defaults without requiring source edits:

```bash
env GENESIS_DATASETS=eval_08142026,train_08142026 \
  TARGET_CHOICES=drop_simple,drop,inclined,step,gentle_simple,firm_simple \
  REQUIRED_RUNS_PER_TASK=2 GENESIS_GPU_IDS=0,1 \
  MAX_PARALLEL_PROCESSES=10 WORKER_SPAWN_DELAY_SECONDS=1 \
  workspace/Genesis/src/generation/run_rigid_two_rounds.sh
```

`inclined` and `step` expand to the three angle or height variants implemented
by `run_genesis_sim.py`. The corresponding offline processing order is
`1_create_annotations.py`, `2_create_train_csv.py`,
`create_train_balanced_set.py`, `create_independent_eval_sets.py`, and the two
`3_bert_*.py` embedding scripts. Use the exact dataset names and metadata
recorded in the paper artifact manifest rather than relying on script defaults.

The retained evaluation closure also contains the RQ2 indexed-window force
baseline, fixed-versus-wobbling support simulation, real contact-window audit
and its real-log loaders. `generate_pick_place_force_videos.py` is retained
because its fixed and wobbling YCB outputs are source clips for the paper
supplementary video. The semantic F/T and terminal-insertion plotting scripts are
the source programs identified for the manuscript figures.

This is a runnable reconstruction of the preserved source and settings, not a
claim of bitwise regeneration of the published 0814 corpus. The local research
checkout was dirty, and a single immutable historical source commit and full
historical dependency lock were not saved with every rollout. The preserved
CSV metadata, artifact hashes, checkpoint settings, and the historical
SentenceTransformer revision remain the authority for matching paper results.

RQ1 experiment scripts read the balanced 0814 training CSV and raw-window
cache from the Genesis artifact tree, prepared evaluation prototypes and
checkpoints from the PatchTST sibling, and optionally the audited robot logs.
The historical SentenceTransformer revision was
`e8c3b32edf5434bc2275fc9bab85f82640a19130`; the portable default resolves
that exact revision. `FT_TEXT_MODEL` may instead point to an already downloaded
snapshot.

## Scientific behavior

The overlay preserves 100 Hz acquisition, 80-sample force/torque windows, the
label rules and exclusion filters, the stored train/evaluation file names, and
the original checkpoint/prototype directory structure. Portability edits only
replace machine-specific absolute paths. The engine additions expose the net
direct-contact wrench on each rigid link. The unrelated SAP/FEM engine patch
and deformable benchmark are outside the paper collector dependency closure.
Removed research-fork files cover FEM/rigid-elastic method benchmarks,
OpenAI-vision and VLA work, unused manipulation baselines, parameter sweeps,
old dataset utilities, and unpublished RQ1 side experiments. None is imported
by the retained collector, annotation, RQ1 replay, RQ2 offline evaluation,
paper-figure, or supplementary-video source paths.

Install the pinned Genesis dependencies plus
`research/genesis/requirements-research.txt`. The retained RQ2 real-log audit
loaders also need the host ROS 2 Python packages. Annotation tests are CPU-only:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_processing_rigid_annotations.py \
  tests/test_generation_make_step_contacts.py
```

Run the standalone engine smoke test through the contained environment:

```bash
tools/run.py simulation -- src/generation/smoke_cpu.py
```

It creates a floor and an off-center falling box, advances 100 CPU steps, and
checks that the exposed direct-contact force and torque are finite and that the
torque is nonzero. The fuller regression is `tests/rigid/test_contact_torque.py`.
Both compile a small CPU scene on first use.

Validation on 2026-09-30 used the contained Python 3.11.13 environment and the
materialized public pin plus overlay. The smoke scene produced four contacts,
a force norm of `47.08799999633895`, and a nonzero torque norm of
`14.126399998901704` after 100 CPU steps.

The one-episode wrapper was also exercised headlessly on CPU with YCB
`001_chips_can`. It built the Panda/object scene, recorded force/torque data,
and saved two audited accidental-drop attempts after contact was lost (steps
480 and 673). This validates the real collection and retry path; it is not a
successful gentle-placement result. Offline annotation produced 20 and 40
windows from the two attempts, and the merge wrote a 60-row, 14-column CSV.
The retained outputs, environment record, and checksums are under
`workspace/reproduced/simulation_smoke/`.
