"""Start the real lift-and-lower evaluation with PatchTST place detection."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Reuse persistent robot attachment and start the model capture node."""
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
        "capture_executable": "model_place_capture",
        "model_encoder_arch": "tokenfusion",
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
                    "Object label used for data/place_<object_name>; pass the "
                    "label without the place_ prefix."
                ),
            ),
            DeclareLaunchArgument(
                "surface_condition",
                default_value="",
                description=(
                    "Optional support condition suffix and metadata label. Leave "
                    "empty for the existing fixed-floor protocol."
                ),
            ),
            DeclareLaunchArgument(
                "motion_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("data_capture_tools"),
                        "config",
                        "lift_and_lower_model_motion_capture.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "capture_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("data_capture_tools"),
                        "config",
                        "lift_and_lower_model_data_capture.yaml",
                    ]
                ),
            ),
        ]
    )
    forwarded = list(argument_defaults) + [
        "object_name",
        "surface_condition",
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
