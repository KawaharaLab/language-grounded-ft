"""Attach to a persistent robot driver and start sensors and key capture."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _include(package: str, filename: str, condition: str, arguments: dict):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", filename])
        ),
        condition=IfCondition(LaunchConfiguration(condition)),
        launch_arguments=arguments.items(),
    )


def generate_launch_description():
    """Start capture devices while attaching to a persistent UR driver by default."""
    motion_config = LaunchConfiguration("motion_config")
    capture_config = LaunchConfiguration("capture_config")

    robot_driver = _include(
        "ur_robot_driver",
        "ur_control.launch.py",
        "start_robot_driver",
        {
            "ur_type": LaunchConfiguration("ur_type"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "launch_rviz": "false",
            "initial_joint_controller": "forward_position_controller",
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
        },
    )
    gripper_driver = _include(
        "robotiq_2f_gripper_hardware",
        "robotiq_2f_gripper_launch.py",
        "start_gripper_driver",
        {
            "serial_port": LaunchConfiguration("gripper_port"),
            "fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "rviz2": "false",
        },
    )
    cameras = _include(
        "realsense_dual_launch",
        "realsense_dual.launch.py",
        "start_cameras",
        {
            "serial1": LaunchConfiguration("camera_wrist_serial"),
            "serial2": LaunchConfiguration("camera_fixed_serial"),
            "color_profile1": LaunchConfiguration("camera_profile"),
            "color_profile2": LaunchConfiguration("camera_profile"),
            "enable_depth1": "false",
            "enable_depth2": "false",
        },
    )
    force_sensors = _include(
        "mms101_driver",
        "mms101_dual.launch.py",
        "start_force_sensors",
        {
            "port1": LaunchConfiguration("force_left_port"),
            "port2": LaunchConfiguration("force_right_port"),
            "sample_interval_us": "10000",
            "low_latency": "true",
        },
    )
    servo = _include(
        "teleop_servo",
        "servo_ur.launch.py",
        "start_servo",
        {"launch_teleop": "false"},
    )

    capture_node = Node(
        package="data_capture_tools",
        executable=LaunchConfiguration("capture_executable"),
        output="screen",
        emulate_tty=True,
        parameters=[
            motion_config,
            {
                "config": capture_config,
                "object_name": ParameterValue(
                    LaunchConfiguration("object_name"), value_type=str
                ),
                "surface_condition": ParameterValue(
                    LaunchConfiguration("surface_condition"), value_type=str
                ),
                "model_encoder_arch": ParameterValue(
                    LaunchConfiguration("model_encoder_arch"), value_type=str
                ),
                "fixed_surface_target_x_m": ParameterValue(
                    LaunchConfiguration("fixed_surface_target_x_m"), value_type=float
                ),
                "fixed_surface_target_y_m": ParameterValue(
                    LaunchConfiguration("fixed_surface_target_y_m"), value_type=float
                ),
                "wobble_surface_target_x_m": ParameterValue(
                    LaunchConfiguration("wobble_surface_target_x_m"), value_type=float
                ),
                "wobble_surface_target_y_m": ParameterValue(
                    LaunchConfiguration("wobble_surface_target_y_m"), value_type=float
                ),
            },
        ],
    )

    stop_launch_when_capture_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=capture_node,
            on_exit=[Shutdown(reason="key motion capture exited")],
        )
    )

    arguments = [
        DeclareLaunchArgument("ur_type", default_value="ur3"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.102"),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("gripper_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("force_left_port", default_value="/dev/mms101_left"),
        DeclareLaunchArgument("force_right_port", default_value="/dev/mms101_right"),
        DeclareLaunchArgument("camera_wrist_serial", default_value="143122066280"),
        DeclareLaunchArgument("camera_fixed_serial", default_value="141722062040"),
        DeclareLaunchArgument("camera_profile", default_value="640x360x15"),
        DeclareLaunchArgument(
            "start_robot_driver",
            default_value="false",
            description=(
                "Start the UR driver inside this launch. Keep false when using the "
                "persistent ur3_control.launch.py terminal."
            ),
        ),
        DeclareLaunchArgument("start_gripper_driver", default_value="true"),
        DeclareLaunchArgument("start_cameras", default_value="true"),
        DeclareLaunchArgument("start_force_sensors", default_value="true"),
        DeclareLaunchArgument("start_servo", default_value="true"),
        DeclareLaunchArgument(
            "capture_executable",
            default_value="key_motion_capture",
            description="Keyboard capture node executable to run.",
        ),
        DeclareLaunchArgument(
            "object_name",
            default_value="",
            description=(
                "Optional object label. When set, data is saved under "
                "data/place_<object_name>."
            ),
        ),
        DeclareLaunchArgument(
            "surface_condition",
            default_value="",
            description=(
                "Optional support condition appended to the dataset name, for "
                "example wobble -> data/place_<object_name>_wobble."
            ),
        ),
        DeclareLaunchArgument(
            "model_encoder_arch",
            default_value="tokenfusion",
            description=(
                "Force-language encoder used by model_place_capture: "
                "tokenfusion, cnn, or mlp."
            ),
        ),
        DeclareLaunchArgument("fixed_surface_target_x_m", default_value="0.0"),
        DeclareLaunchArgument("fixed_surface_target_y_m", default_value="0.0"),
        DeclareLaunchArgument("wobble_surface_target_x_m", default_value="0.0"),
        DeclareLaunchArgument("wobble_surface_target_y_m", default_value="0.0"),
        DeclareLaunchArgument(
            "motion_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("data_capture_tools"), "config", "key_motion_capture.yaml"]
            ),
        ),
        DeclareLaunchArgument(
            "capture_config",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("data_capture_tools"),
                    "config",
                    "key_motion_data_capture.yaml",
                ]
            ),
        ),
    ]

    # The capture node waits for all services/topics, so modest staggering is
    # enough and avoids a CPU spike from starting every component at once.
    return LaunchDescription(
        arguments
        + [
            robot_driver,
            gripper_driver,
            cameras,
            force_sensors,
            TimerAction(period=2.0, actions=[servo]),
            TimerAction(period=3.0, actions=[capture_node]),
            stop_launch_when_capture_exits,
        ]
    )
