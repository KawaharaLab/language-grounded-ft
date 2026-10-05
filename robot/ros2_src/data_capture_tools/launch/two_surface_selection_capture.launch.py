"""Probe fixed and wobbling supports and place on the selected support."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Attach to the persistent UR driver and run two-support selection."""
    defaults = {
        "ur_type": "ur3",
        "robot_ip": "192.168.1.102",
        "use_fake_hardware": "false",
        "gripper_port": "/dev/ttyUSB0",
        "force_left_port": "/dev/mms101_left",
        "force_right_port": "/dev/mms101_right",
        "camera_wrist_serial": "143122066280",
        "camera_fixed_serial": "141722062040",
        "camera_profile": "640x360x15",
        "start_robot_driver": "false",
        "start_gripper_driver": "true",
        "start_cameras": "true",
        "start_force_sensors": "true",
        "start_servo": "true",
        "capture_executable": "two_surface_selection_capture",
        "surface_condition": "surface_selection",
        "fixed_surface_target_x_m": "-0.3718813400688677",
        "fixed_surface_target_y_m": "-0.19589684937240429",
        "wobble_surface_target_x_m": "-0.15115650678458373",
        "wobble_surface_target_y_m": "-0.4116110457184974",
    }
    arguments = [
        DeclareLaunchArgument(name, default_value=value)
        for name, value in defaults.items()
    ]
    arguments.extend(
        [
            DeclareLaunchArgument(
                "object_name",
                description="Object label without the place_ prefix.",
            ),
            DeclareLaunchArgument(
                "motion_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("data_capture_tools"),
                        "config",
                        "two_surface_selection_motion_capture.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "capture_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("data_capture_tools"),
                        "config",
                        "two_surface_selection_data_capture.yaml",
                    ]
                ),
            ),
        ]
    )
    forwarded = list(defaults) + [
        "object_name",
        "motion_config",
        "capture_config",
    ]
    capture = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("data_capture_tools"),
                    "launch",
                    "key_motion_capture.launch.py",
                ]
            )
        ),
        launch_arguments={
            name: LaunchConfiguration(name) for name in forwarded
        }.items(),
    )
    return LaunchDescription(arguments + [capture])
