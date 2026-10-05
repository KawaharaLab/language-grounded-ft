# Licensing scope

This is a multi-component research workspace, **not a single-license redistribution**.
Original research contributions and the workspace tooling are licensed under Apache-2.0, as authorized by the authors on 2026-09-30. See the root `LICENSE`.
This file does not grant a new license to code whose original permissions are unspecified.

- `third_party/genesis` and `third_party/patchtst`: Apache-2.0; the exact upstream license and copyright notices remain in the pinned submodules.
- `research/genesis` and `research/patchtst`: research changes derived from those projects; upstream Apache-2.0 texts and attribution are retained. See their source/overlay inventories for changed files. Original research contributions are Apache-2.0.
- `third_party/robotiq`: a pinned Git submodule reference to an independently
  maintained repository. The Git link does not incorporate that repository's
  contents into this project's Apache-2.0 work. Do not bundle the submodule in a
  source archive: its description package is ROS-Industrial BSD and its
  interfaces package is BSD-3-Clause, but its hardware and message packages have
  no granted license at the pinned revision. Its nested serial submodule has
  separate MIT and BSD-style notices. See `THIRD_PARTY_NOTICES.md`.
- `robot`: original research contributions are Apache-2.0, except for the
  identified third-party files and derivatives. The Robotiq visualization files
  below `robot/ros2_src/teleop_servo/config/robotiq_2f_140_gripper_visualization/`
  retain the ROS-Industrial BSD license stored beside them. Experiment-specific
  derivatives of upstream Robotiq hardware source are intentionally excluded
  from this public tree; the optional ignored local location and resulting
  reproduction limit are documented in `robot/overlays/README.md` and
  `docs/robot-setup.md`.
- `artifacts` (ignored by Git): datasets, object models, weights, photographs and videos have separate terms. Their presence in a local workspace does not relicense them.
- `publication`: manuscript and bibliography retain their authors' rights; no software license is applied automatically.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream sources and data attribution.
