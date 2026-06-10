from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # --- Common parameters ---
    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='fr3_link0',
        description='Franka base frame'
    )
    linear_scale_arg = DeclareLaunchArgument(
        'linear_scale',
        default_value='2.5',
        description='Linear velocity scale factor'
    )
    angular_scale_arg = DeclareLaunchArgument(
        'angular_scale',
        default_value='1.0',
        description='Angular velocity scale factor'
    )
    linear_deadband_arg = DeclareLaunchArgument(
        'linear_deadband',
        default_value='0.001',
        description='Linear velocity deadband (m/s)'
    )
    angular_deadband_arg = DeclareLaunchArgument(
        'angular_deadband',
        default_value='0.005',
        description='Angular velocity deadband (rad/s)'
    )
    pose_mode_arg = DeclareLaunchArgument(
        'pose_mode',
        default_value='both',
        description='Pose mode: absolute | relative | both'
    )

    # TF statiche: world → haptic base frames 
    tf_haptic_left_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='haptic_left_base',
        arguments=['0', '0', '0', '0', '0', '0', '1',
                   'world', 'haptic_left_base']
        #          x   y   z   qx  qy  qz  qw
    )

    tf_haptic_right_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='haptic_right_base',
        arguments=['0', '0', '0', '0', '0', '0', '1',
                   'world', 'haptic_right_base']
    )

    # Device 1 (haptic1) Left N70
    haptic_teleop_left = Node(
        package='haptic_control',
        executable='haptic_teleop_node',
        name='haptic_teleop_node',
        namespace='left',
        output='screen',
        parameters=[{
            'base_frame':          LaunchConfiguration('base_frame'),
            'haptic_base_frame':   'haptic_left_base',   # parent TF per RViz
            'ee_frame':            'haptic_ee_left',
            'linear_scale':        LaunchConfiguration('linear_scale'),
            'angular_scale':       LaunchConfiguration('angular_scale'),
            'linear_deadband':     LaunchConfiguration('linear_deadband'),
            'angular_deadband':    LaunchConfiguration('angular_deadband'),
            'pose_mode':           LaunchConfiguration('pose_mode'),
            'haptic_rotation_matrix': [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        }],
    )

    # --- Device 2 (haptic2) --- Right N71
    haptic_teleop_right = Node(
        package='haptic_control',
        executable='haptic_teleop_node',
        name='haptic_teleop_node',
        namespace='right',
        output='screen',
        parameters=[{
            'base_frame':          LaunchConfiguration('base_frame'),
            'haptic_base_frame':   'haptic_right_base', 
            'ee_frame':            'haptic_ee_right',
            'linear_scale':        LaunchConfiguration('linear_scale'),
            'angular_scale':       LaunchConfiguration('angular_scale'),
            'linear_deadband':     LaunchConfiguration('linear_deadband'),
            'angular_deadband':    LaunchConfiguration('angular_deadband'),
            'pose_mode':           LaunchConfiguration('pose_mode'),

            'haptic_rotation_matrix': [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        }],
    )

    return LaunchDescription([
        base_frame_arg,
        linear_scale_arg,
        angular_scale_arg,
        linear_deadband_arg,
        angular_deadband_arg,
        pose_mode_arg,
        tf_haptic_left_base,   
        tf_haptic_right_base,  
        haptic_teleop_left,
        haptic_teleop_right,
    ])