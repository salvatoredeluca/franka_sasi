from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    return LaunchDescription([

        # Haptic device 1 (n70 - 192.168.9.16) --- (left n70)
        Node(
            package='haption_raptor_api',
            executable='raptor_api_wrapper',
            name='raptor_api_wrapper',
            namespace='left',
            output='screen',
        ),
        Node(
            package='test_impedance',
            executable='test_impedance',
            name='test_impedance',
            namespace='left',
            output='screen',
            parameters=[{
                'channel':            'SimpleChannelUDP',
                'ff_device_ip_address': '192.168.9.16',
                'ff_device_param_file': '/etc/Haption/Connector/desktop_6D_n70.param',
                'local_ip_address':   '192.168.9.101',
                'speed_factor':       1.0,
                'force_factor':       1.0,
                'max_force':          10.0,
                'horizontal_table_z': -0.05,
                'penalty_stiffness':  100.0,
            }],
        ),

        # Haptic device 2 (n71 - 192.168.9.17) --- (right n71)
                Node(
                    package='haption_raptor_api',
                    executable='raptor_api_wrapper',
                    name='raptor_api_wrapper',
                    namespace='right',
                    output='screen',
                ),
                Node(
                    package='test_impedance',
                    executable='test_impedance',
                    name='test_impedance',
                    namespace='right',
                    output='screen',
                    parameters=[{
                        'channel':              'SimpleChannelUDP',
                        'ff_device_ip_address': '192.168.9.17',
                        'ff_device_param_file': '/etc/Haption/Connector/desktop_6D_n71.param',
                        'local_ip_address':     '192.168.9.102',
                        'speed_factor':         1.0,
                        'force_factor':         1.0,
                        'max_force':            10.0,
                        'horizontal_table_z':   -0.05,
                        'penalty_stiffness':    100.0,
                    }],
                ),
    ])