"""Keep the UR3 driver and ros2_control stack running independently."""

import os
from xml.etree import ElementTree

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch the upstream UR driver with the controller used by data capture."""
    urcl_prefix = get_package_prefix("ur_client_library")
    urcl_package_xml = os.path.join(
        get_package_share_directory("ur_client_library"), "package.xml"
    )
    urcl_version = ElementTree.parse(urcl_package_xml).getroot().findtext("version")
    version_numbers = tuple(int(part) for part in urcl_version.split(".")[:3])
    if version_numbers < (2, 14, 0):
        raise RuntimeError(
            f"ur_client_library {urcl_version} has a one-second Primary "
            "Interface startup timeout. Update the ROS Humble UR packages to "
            "2.14.0 or newer before launching."
        )

    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": LaunchConfiguration("ur_type"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "initial_joint_controller": "forward_position_controller",
            "activate_joint_controller": "true",
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "controller_spawner_timeout": LaunchConfiguration(
                "controller_spawner_timeout"
            ),
        }.items(),
    )

    return LaunchDescription(
        [
            LogInfo(msg=f"Using ur_client_library {urcl_version}: {urcl_prefix}"),
            DeclareLaunchArgument("ur_type", default_value="ur3"),
            DeclareLaunchArgument("robot_ip", default_value="192.168.1.102"),
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument(
                "controller_spawner_timeout",
                default_value="30",
                description="Wait longer for controller_manager during UR startup.",
            ),
            ur_driver,
        ]
    )
