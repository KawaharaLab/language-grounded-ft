"""Run the long-horizon signed-Fz baseline on a wobbling support."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Combine the raised wobble pose with the five-second Fz horizon."""
    object_name = LaunchConfiguration("object_name")
    wobble_baseline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("data_capture_tools"),
                    "launch",
                    "lift_and_lower_wobble_fz_baseline_capture.launch.py",
                ]
            )
        ),
        launch_arguments={
            "object_name": object_name,
            "motion_config": PathJoinSubstitution(
                [
                    FindPackageShare("data_capture_tools"),
                    "config",
                    "lift_and_lower_wobble_fz_long_horizon_motion_capture.yaml",
                ]
            ),
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "object_name",
                description="Object label, for example baseball.",
            ),
            wobble_baseline,
        ]
    )
