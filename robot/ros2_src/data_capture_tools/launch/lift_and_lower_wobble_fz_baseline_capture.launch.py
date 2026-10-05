"""Run the signed-Fz baseline on the raised wobbling-support pose."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Use the wobble model pose while keeping the Fz baseline protocol."""
    argument_defaults = {
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
        "capture_executable": "fz_baseline_capture",
        "surface_condition": "wobble",
    }
    arguments = [
        DeclareLaunchArgument(name, default_value=value)
        for name, value in argument_defaults.items()
    ]
    arguments.extend(
        [
            DeclareLaunchArgument(
                "object_name",
                description=(
                    "Object label without place_; output is written below "
                    "data/place_<object_name>_wobble_fz_baseline."
                ),
            ),
            DeclareLaunchArgument(
                "motion_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("data_capture_tools"),
                        "config",
                        "lift_and_lower_wobble_fz_baseline_motion_capture.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "capture_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("data_capture_tools"),
                        "config",
                        "lift_and_lower_fz_baseline_data_capture.yaml",
                    ]
                ),
            ),
        ]
    )
    forwarded = list(argument_defaults) + [
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
