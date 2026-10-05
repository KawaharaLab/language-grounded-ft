# Third-party provenance and notices

Pinned sources are recorded in `.gitmodules` and the Git submodule entries. Upstream
source, copyright and license notices are preserved. The generated `workspace/`
applies overlays to a copy; it never edits the pinned submodule itself.

| Component | Original source | Terms / handling |
|---|---|---|
| Genesis | https://github.com/Genesis-Embodied-AI/Genesis | Apache-2.0; `third_party/genesis/LICENSE` |
| PatchTST | https://github.com/yuqinie98/PatchTST | Apache-2.0; `third_party/patchtst/LICENSE` |
| Robotiq ROS 2 driver | https://github.com/jannishaberhausen/robotiq_2f_gripper_ros2, commit `17533fe6faa9688f40a7ccc1663f59054c6dc3e2` | Gitlink/reference only. Description: ROS-Industrial BSD; interfaces: PickNik BSD-3-Clause. Hardware and messages: `package.xml` says `TODO: License declaration`, no license file applies to those packages, and no permission to redistribute them is inferred. Do not bundle the full submodule in a release archive. |
| Robotiq visualization files copied into `robot/ros2_src/teleop_servo/config/robotiq_2f_140_gripper_visualization/` | Robotiq description package at the pin above | Twelve STL files are byte-identical upstream copies; four Xacro files are copied/adapted derivatives. ROS-Industrial BSD notice and disclaimer are retained in the adjacent `LICENCE`; local changes and the exact pin are recorded in the adjacent `NOTICE` and file headers. |
| Optional local Robotiq hardware overlay | Robotiq hardware package at the pin above | Modified experiment files are excluded from the public tree. `robot/overlays/README.md` documents their ignored local location and the behavior unavailable without them. They may be published only after obtaining an explicit license from the rightsholder or replacing/reimplementing them from a licensed source. |
| serial-ros2 | https://github.com/jannishaberhausen/serial-ros2, commit `3fd8bd0fcb6cfd2bd6d5268fc3b33859e4182f6a` (nested submodule) | MIT notice is embedded in `include/serial/serial.h`; the implementation files retain author notices. `include/serial/v8stdint.h` retains the V8-project BSD-style notice. Preserve these notices in any bundled copy. |
| Google Scanned Objects conversion | https://github.com/kevinzakka/mujoco_scanned_objects | OBJ/PNG assets: CC BY 4.0; MJCF/XML: MIT (Kevin Zakka, 2022). Do not apply the MIT code license to the meshes/textures. |
| YCB objects | https://www.ycbbenchmarks.com/object-models/ | Source collection attribution required; local processed mesh provenance must be checked before redistribution |
| all-mpnet-base-v2 | https://huggingface.co/sentence-transformers/all-mpnet-base-v2 | Apache-2.0 model card; model revision is recorded by simulation tooling |

The Apache-2.0 redistribution and modification requirements are available at
https://www.apache.org/licenses/LICENSE-2.0. Preserve upstream notices and identify
modified files. `LICENSES/Apache-2.0.txt` retains the complete license text.

## Public source-release boundary

A normal checkout of this superproject records submodules as commit identifiers;
it does not contain the submodule source. That reference is useful provenance,
but neither a recursive clone nor a release archive assembled with submodule
contents should be described as wholly Apache-2.0. In particular, the pinned
Robotiq hardware and message packages have no stated redistribution permission.

The two experiment-specific hardware overlay files are omitted from this public
tree and may exist only under the ignored `.local/` path described in
`robot/overlays/README.md`. Keeping a Gitlink to the external Robotiq repository
is distinct from redistributing its contents, but users who fetch it receive
components with unresolved terms. A
redistributable robot stack requires one of: written permission or an upstream
license covering those packages; replacement with a compatible licensed
driver/messages implementation; or independent reimplementation without copying
the unlicensed source. Attribution cannot substitute for permission.

Locally imported artifacts remain outside Git. Public data/weight release locations
and dataset/model license choices have not yet been assigned. Do not describe
this repository's code license as covering all imported artifacts.
