# Optional local Robotiq runtime overlay

The experimental Robotiq hardware patches are intentionally omitted from this
public source tree because the corresponding upstream hardware package does not
state a redistribution license. A private working copy may place the two files
at:

```text
.local/robot-overlays/robotiq_2f_gripper_ros2/
  robotiq_2f_gripper_node.cpp
  robotiq_2f_gripper_node.hpp
```

`python tools/workspace.py materialize` reports whether it applies that local
overlay. When the directory is absent, materialization leaves the pinned
upstream driver unchanged.

The omitted experiment version accepts a new goal while an earlier action is
running, waits briefly for the physical gripper status to change, requires two
consecutive stopped samples before reporting success, handles cancellation while
waiting, aborts on a configured timeout, and preserves the commanded speed in
fake-hardware state. The pinned upstream instead rejects overlapping goals and
uses its original single-sample completion loop. Consequently, the public tree
does not exactly reproduce rapid command preemption, cancellation, or completion
timing from the real-robot experiments. Simulation, learning, and recorded-data
evaluation do not require this local overlay.
