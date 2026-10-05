# Real-robot setup and safety boundary

The source under `robot/ros2_src` reproduces the UR3, Robotiq 2F-140, dual MMS101, dual RealSense, MoveIt Servo, Fz baseline, learned placement, surface-selection, and data acquisition used by the experiments. It deliberately contains no recorded datasets, build products, device credentials, or robot calibration changes.

Run `source env/robot.sh` from any directory. It sets `FT_REPO_ROOT`, defines `FT_WORKSPACE_ROOT` as `<repository>/workspace`, and sources ROS Humble and the robot install space when present. The bootstrap materializes the colcon workspace at `workspace/ur3_robotiq_ros2` and links the packages from `robot/ros2_src` into its `src` directory. Runtime ROS dependencies are ROS 2 Humble, `ur_robot_driver`/`ur_client_library`, MoveIt Servo, `realsense2_camera`, `cv_bridge`, OpenCV, NumPy, PyYAML, and the pinned Robotiq driver described in `manifests/robot-dependencies.json`. The Robotiq driver itself has a nested `serial-ros2` dependency.

Robot setup is optional for the simulation and learning workflows. To materialize
it, initialize the pinned Robotiq checkout and its nested serial dependency, then
run the default materializer:

```bash
git submodule update --init --recursive third_party/robotiq
python3 tools/workspace.py materialize
python3 tools/workspace.py doctor
```

The main code-only path may instead use `materialize --without-robot` and
`doctor --without-robot`; those commands neither initialize nor validate the
Robotiq checkout.

The committed defaults preserve the experimental IP address, device aliases, camera serials, home pose, velocities, force thresholds, and target coordinates. Review them against the physical cell before any launch. No automated check can establish that the workspace is clear, the UR program is loaded, the emergency stop works, or the sensor axes match the installed hardware. Start the persistent UR driver separately; experiment launch files default to attaching to it (`start_robot_driver:=false`). Use `use_fake_hardware:=true` for launch-graph checks.

The Fz and learned placement entry points are the `lift_and_lower_*_capture.launch.py` files. `two_surface_selection_capture.launch.py` performs the fixed-versus-wobbling support experiment. The current paper terminal-insertion result is an offline PatchTST binary readout over already captured force windows: frozen linear and Q/V LoRA variants are trained and evaluated by the code and artifacts covered by `manifests/learning_patchtst.json`.

The authors have authorized the original research contributions under Apache-2.0. `data_capture_tools`, `mms101_driver`, `gripper_client`, `realsense_dual_launch`, and `teleop_servo` carry Apache-2.0 declarations and full package license texts. Third-party rights remain separate. In particular, the Robotiq meshes and adapted Xacro files vendored below `teleop_servo/config/robotiq_2f_140_gripper_visualization` retain the ROS-Industrial BSD `LICENCE`, attribution, and modification notices in that directory; the package's Apache-2.0 declaration does not relicense those assets. The twelve STL files were verified byte-identical to the pinned description package, and the four Xacro files identify their copied or adapted status.

The pinned Robotiq checkout is commit `17533fe6faa9688f40a7ccc1663f59054c6dc3e2`. Its `robotiq_2f_gripper_description` package carries the ROS-Industrial BSD notice and `robotiq_2f_gripper_interfaces` carries PickNik's BSD-3-Clause text. Its `robotiq_2f_gripper_hardware` and `robotiq_2f_gripper_msgs` manifests instead contain `TODO: License declaration`, with no license grant found for those packages at the pin. Nested `serial-ros2` is pinned at `3fd8bd0fcb6cfd2bd6d5268fc3b33859e4182f6a`; its MIT notice is embedded in `include/serial/serial.h`, and `include/serial/v8stdint.h` retains a separate V8-project BSD-style notice.

Two experiment files are modified copies of that unlicensed Robotiq hardware package. They are excluded from the public tree and, for authorized private reproduction, may be placed only under the ignored `.local/robot-overlays/robotiq_2f_gripper_ros2` path. Workspace materialization reports whether it applies them; otherwise it leaves the pinned upstream driver unchanged. The omitted version adds rapid goal preemption, cancellation during the completion wait, delayed and two-sample physical stop confirmation, explicit timeout aborts, and faithful fake-hardware speed state. Public real-robot reproduction therefore does not exactly match those action and completion semantics. See `robot/overlays/README.md`. Redistribution of those omitted driver patches requires explicit permission or a licensed replacement; attribution alone does not resolve the missing permission.

The `third_party/robotiq` entry in this repository is a Gitlink to the external project, not a copy of its files in the superproject commit. It may remain as a pinned provenance reference, but do not bundle its contents in a release archive or claim that a recursive checkout is wholly Apache-2.0. Fetching the submodule still exposes users to the unresolved upstream hardware/message terms. See `THIRD_PARTY_NOTICES.md` for the release boundary and complete component inventory.
