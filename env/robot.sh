#!/usr/bin/env bash
# Source this file; it only configures paths and never starts ROS nodes.

_ft_robot_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FT_REPO_ROOT="$(cd "${_ft_robot_env_dir}/.." && pwd)"
export FT_WORKSPACE_ROOT="${FT_REPO_ROOT}/workspace"
export FT_ROBOT_WORKSPACE="${FT_WORKSPACE_ROOT}/ur3_robotiq_ros2"

# Drop inherited ROS overlays before establishing this repository's two-prefix
# environment. This prevents an interactive shell from silently supplying
# packages from an older development workspace.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH
unset PYTHONPATH LD_LIBRARY_PATH

if [[ -r /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [[ -r "${FT_ROBOT_WORKSPACE}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${FT_ROBOT_WORKSPACE}/install/setup.bash"
fi

# ROS Humble nodes run with system Python 3.10. Reuse the repository-managed
# learning environment's binary wheels without activating its non-ROS Python.
_ft_learning_site="${FT_REPO_ROOT}/.venvs/learning/lib/python3.10/site-packages"
if [[ -d "${_ft_learning_site}" ]]; then
  export PYTHONPATH="${_ft_learning_site}${PYTHONPATH:+:${PYTHONPATH}}"
fi

unset _ft_robot_env_dir _ft_learning_site
