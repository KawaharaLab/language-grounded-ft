# Preprint source validation

The public code uses neutral paper/experiment names. Existing local archives can
be connected through local symlink aliases without copying or altering their
contents. Manuscript sources are managed separately from this code release.

Validation after the naming update: 22 workspace/source/license checks passed,
all six learning checks passed, and all six saved RQ1 best/last checkpoints
replayed with unchanged metrics. Python and shell syntax checks passed. No
real-hardware execution or full retraining was performed.

# Public source preparation (2026-10-05)

- The public snapshot has a separate parentless history. Earlier local migration
  branches are retained only locally because they contain driver patches omitted
  from this release. Audit/push only the public branch, not all local refs.
- Data, checkpoints, recorded trials, local figure links, environments, and
  private driver overlays are excluded from the public source tree.
- A fresh single-branch source clone initialized the pinned Genesis and PatchTST
  upstreams and passed `materialize --without-robot` and `doctor --without-robot`
  with **zero artifact mounts** and no Robotiq dependency.
- The README learning check in that fresh source tree ran six tests: **three
  synthetic tests passed and three artifact-dependent tests were skipped**.
  It reused the already installed, pinned learning environment; package downloads
  were not repeated. The same command with local artifacts passed all six tests.
- Workspace, simulation-overlay, robot-license/configuration, and public-tree
  checks: **22 tests passed**. The release-tree and public-history audit found
  no detected credentials, private trial/model files, or excluded source copies.
- Driver patch exclusion changes real-robot action/preemption/completion behavior
  relative to the private experimental setup. The earlier robot build checks
  below included those patches; they do not certify identical behavior of the
  public-only driver. See [robot integration](robot-setup.md).
- No full training or real-hardware execution was performed during publication
  preparation. Prior model and simulation checks remain recorded below.

# Consolidation validation (2026-09-30)

The implementations were prepared with GPT-5.6-sol subagents as requested.
Original checkouts were treated as read-only. Local execution uses the new
repository and its declared artifact mounts.

## Verified

- Public upstream commits are fixed by three Git submodule entries; Robotiq's
  serial dependency is pinned by its own upstream submodule entry.
- Fresh contained Python installations: CPython 3.10.0 for learning, CPython
  3.11.13 for simulation, with separate hash-locked dependency files.
- RQ1: six best/last four-head MLP checkpoints across three seeds replayed on
  the fixed evaluation windows. Maximum absolute metric difference against
  recorded metrics: **1.11e-16**. The complete replay also passed with original-directory reads blocked, including archived source-hash checks. Full provenance:
  [learning_replay_verification.json](learning_replay_verification.json).
- Model checks: real Token-Fusion backward/optimizer update; strict restored
  contrastive-pretrained and supervised-pretrained LoRA state dictionaries;
  finite CPU inference; real recorded CSV/text-embedding batch loading under a guard that rejects access to the old directories. Ten learning tests passed.
- Language encoder: the copied historical all-mpnet-base-v2 snapshot loads
  with network access disabled, producing a normalized 768-dimensional vector.
- Genesis: 100 CPU simulation steps produce four contacts, force norm
  47.088 and nonzero contact-torque norm 14.1264 in the contained environment.
- Actual Panda/YCB collection and offline annotation/merge produced 60 windows
  from two recorded early-drop attempts (not a successful placement). Thirty
  focused simulation/annotation tests passed. Evidence is in
  `workspace/reproduced/simulation_smoke/verification.json`.
- A fresh Git clone with independently initialized submodules materialized at
  a different path and passed the workspace dependency audit.
- Robot package compilation and non-actuating launch-argument parsing passed;
  robot source and license contract checks passed. A second build from an empty
  environment confirmed all external ROS packages come from `/opt/ros/humble`;
  generated setup scripts do not include former workspace prefixes. The actual learned-placement detector also loaded the contained checkpoint,
  normalization and prototypes under ROS system Python and produced finite
  scores for an 80×12 window. PyTorch/NumPy came from the contained learning
  environment, and rclpy from `/opt/ros/humble`. No physical robot ran.
- Migration-tool tests cover cross-repository link translation, relocation of
  relative links, and refusal to replace actual data with a symlink.
- All eight required placement model/prototype/normalization hashes match
  `manifests/robot-artifacts.json`.
- Public tracked-tree credential/large-file checks pass. An embedded legacy
  W&B key was removed from the overlay, and its absence has a regression test.

## Current artifact storage: external symlinks

The user requested existing SSD artifacts instead of independent copies.
`tools/link_local.py` now registers only current paper groups in
`artifacts/local-links.json`. The 244.4 GiB independent copy from the first
migration was removed. Dataset/model contents remain excluded from GitHub;
future distribution is planned through Hugging Face.

The historical independent-copy replay above was verified before this storage
change. Current runs allow declared artifact roots (including their canonical
SSD paths) and still reject accidental dependency on old live-code trees.
After the switch, all ten learning tests and all six RQ1 checkpoint replays passed again with unchanged metrics. The original datasets and model files are unchanged. New experiment outputs
belong under `workspace/reproduced/`, not in a linked input directory.

The nine already-missing YCB meshes remain missing in the original collection;
none is part of the recorded evaluation episode directories. They have not been
substituted. Historical raw code snapshots may contain legacy credentials and
remain outside the Git release. The removed W&B key should be revoked if active.

## Scope and limits

Full multi-day data collection/pretraining and all adaptation-grid runs were not
repeated. Exact historical data-generation source commits were not fully recorded
by the original experiments; the copied datasets and their hashes are the numeric
replay reference. Archived scientific ambiguities remain documented in the prior
experiment audit; this consolidation does not silently repair or reinterpret them.

Python virtual environments and ROS build products should be recreated after
moving to a different absolute path or operating system. Code/data relative links
remain portable. Real-robot operation still requires the documented ROS/system
packages and physical configuration. Unrelated OpenPI actuation and exploratory experiment launchers are excluded.

No GitHub push or data upload was performed. Original research code is Apache-2.0;
upstream Robotiq hardware/messages license declarations remain unresolved. Public
artifact hosting and per-artifact release terms are not set by this local copy.
