# Robot experiment reproduction

Validate the source without hardware from the repository root:

```bash
source env/robot.sh
python3 -m compileall -q robot/ros2_src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_robot_bundle.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="robot/ros2_src/data_capture_tools:${PYTHONPATH:-}" pytest -q \
  robot/ros2_src/data_capture_tools/test/test_workspace_paths.py \
  robot/ros2_src/data_capture_tools/test/test_surface_selection.py
```

After bootstrap and dependency installation, compile the materialized source without contacting hardware:

```bash
source env/robot.sh
cd "$FT_ROBOT_WORKSPACE"
colcon list
colcon build --symlink-install --packages-up-to \
  data_capture_tools mms101_driver gripper_client realsense_dual_launch \
  teleop_servo robotiq_2f_gripper_hardware
source install/setup.bash
ros2 pkg executables data_capture_tools
ros2 launch data_capture_tools lift_and_lower_model_capture.launch.py --show-args
```

`colcon build`, package discovery, and `--show-args` do not start a ROS node. Do not remove `--show-args` during offline verification: an actual experiment launch can command physical motion.

`env/robot.sh` clears inherited ROS overlay variables before sourcing `/opt/ros/humble` and this workspace. On the verified ROS Humble installation, `moveit_servo`, `ur_robot_driver`, `ur_client_library`, and `realsense2_camera` all resolve from `/opt/ros/humble`; none is supplied by an original development workspace. Check a new machine before building:

```bash
source env/robot.sh
for package in moveit_servo ur_robot_driver ur_client_library realsense2_camera; do
  ros2 pkg prefix "$package"
done
```

The same environment exposes `.venvs/learning/lib/python3.10/site-packages` to ROS Humble's `/usr/bin/python3`. This supplies the repository-managed PyTorch build while retaining ROS Python packages. Verify the complete learned-placement load and one CPU-only 80-sample readout with:

```bash
source env/robot.sh
/usr/bin/python3 tests/test_robot_detector.py
```

This test constructs `CosinePlaceDetector` directly and creates no ROS node or hardware connection.

Robot model parameters resolve against `FT_WORKSPACE_ROOT`, whose contract is the generated `<repository>/workspace` directory. Model configurations expect `PatchTST/PatchTST_self_supervised` and `artifacts/robot` below it. The capture output default also resolves below this generated workspace.

The PAPER insertion readout is reproduced offline from the PatchTST workspace and imported result trees:

```bash
source env/robot.sh
cd "$FT_WORKSPACE_ROOT/PatchTST"
python PatchTST_self_supervised/summarize_terminal_insertion_pretrained_adaptation.py --help
```

The complete run directories and dataset are enumerated in `manifests/learning_patchtst.json`. Use their snapshotted runner and configuration for numeric reproduction.

The Robotiq third-party checkout must be pinned first, including its nested `serial-ros2` submodule. Experiment-authorized private workspaces may place the two local runtime files in `.local/robot-overlays/robotiq_2f_gripper_ros2`; `tools/workspace.py` applies them to the matching hardware include and source directories and logs the action. Those files are omitted from the public tree because upstream does not state redistribution terms for the hardware package. Without them, materialization uses the pinned driver unchanged and does not exactly reproduce rapid goal preemption, cancellation while waiting, or the experiment's more conservative physical-motion completion timing. See `robot/overlays/README.md`.
