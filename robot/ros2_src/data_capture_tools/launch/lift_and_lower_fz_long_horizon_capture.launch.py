"""Run a longer-horizon signed-Fz zero-crossing experiment."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Reuse the normal Fz launch with the isolated long-horizon profile."""
    object_name = LaunchConfiguration("object_name")
    baseline_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("data_capture_tools"),
                    "launch",
                    "lift_and_lower_fz_baseline_capture.launch.py",
                ]
            )
        ),
        launch_arguments={
            "object_name": object_name,
            "motion_config": PathJoinSubstitution(
                [
                    FindPackageShare("data_capture_tools"),
                    "config",
                    "lift_and_lower_fz_long_horizon_motion_capture.yaml",
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
            baseline_launch,
        ]
    )
