from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    port1_arg = DeclareLaunchArgument(
        'port1', default_value='/dev/mms101_left', description='Serial port for MMS101 #1'
    )
    port2_arg = DeclareLaunchArgument(
        'port2', default_value='/dev/mms101_right', description='Serial port for MMS101 #2'
    )
    frame1_arg = DeclareLaunchArgument(
        'frame1', default_value='ft_sensor_left', description='frame_id for MMS101 #1'
    )
    frame2_arg = DeclareLaunchArgument(
        'frame2', default_value='ft_sensor_right', description='frame_id for MMS101 #2'
    )
    force_scale_arg = DeclareLaunchArgument(
        'force_scale', default_value='0.001', description='Force scale (shared)'
    )
    torque_scale_arg = DeclareLaunchArgument(
        'torque_scale', default_value='0.00001', description='Torque scale (shared)'
    )
    sample_interval_arg = DeclareLaunchArgument(
        'sample_interval_us', default_value='10000',
        description='Sensor sample interval in microseconds (10000 = 100 Hz)'
    )
    low_latency_arg = DeclareLaunchArgument(
        'low_latency', default_value='true',
        description='Request the Linux low-latency serial flag when supported'
    )

    node1 = Node(
        package='mms101_driver',
        executable='mms101_node',
        name='mms101_driver_1',
        output='screen',
        remappings=[('force_torque', LaunchConfiguration('topic1'))],
        parameters=[
            {
                'port': LaunchConfiguration('port1'),
                'frame_id': LaunchConfiguration('frame1'),
                'force_scale': ParameterValue(
                    LaunchConfiguration('force_scale'), value_type=float
                ),
                'torque_scale': ParameterValue(
                    LaunchConfiguration('torque_scale'), value_type=float
                ),
                'sample_interval_us': ParameterValue(
                    LaunchConfiguration('sample_interval_us'), value_type=int
                ),
                'low_latency': ParameterValue(
                    LaunchConfiguration('low_latency'), value_type=bool
                ),
            }
        ],
    )

    node2 = Node(
        package='mms101_driver',
        executable='mms101_node',
        name='mms101_driver_2',
        output='screen',
        remappings=[('force_torque', LaunchConfiguration('topic2'))],
        parameters=[
            {
                'port': LaunchConfiguration('port2'),
                'frame_id': LaunchConfiguration('frame2'),
                'force_scale': ParameterValue(
                    LaunchConfiguration('force_scale'), value_type=float
                ),
                'torque_scale': ParameterValue(
                    LaunchConfiguration('torque_scale'), value_type=float
                ),
                'sample_interval_us': ParameterValue(
                    LaunchConfiguration('sample_interval_us'), value_type=int
                ),
                'low_latency': ParameterValue(
                    LaunchConfiguration('low_latency'), value_type=bool
                ),
            }
        ],
    )

    return LaunchDescription([
        port1_arg,
        port2_arg,
        frame1_arg,
        frame2_arg,
        force_scale_arg,
        torque_scale_arg,
        sample_interval_arg,
        low_latency_arg,
        DeclareLaunchArgument(
            'topic1', default_value='/force_torque/left',
            description='Output topic for sensor 1'
        ),
        DeclareLaunchArgument(
            'topic2', default_value='/force_torque/right',
            description='Output topic for sensor 2'
        ),
        node1,
        node2,
    ])
