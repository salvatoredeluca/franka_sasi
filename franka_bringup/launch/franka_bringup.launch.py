# Copyright 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################################
# Unified launch file for Franka robots - real hardware AND Gazebo simulation.
#
# Gazebo is launched with a zero-gravity world.  This mirrors the real
# Franka behaviour where the firmware compensates gravity internally,
# so user controllers never see gravitational effects.  The same
# controller code works identically on real hardware and in simulation.
#
# Priority: real hardware (use_fake_hardware: "false") always takes
# precedence over Gazebo, regardless of the gazebo flag.
#
# Bringup phases (Gazebo):
#   1) Spawn ALL models in parallel
#   2) ALL JSBs in parallel (each triggered by its own spawn)
#   3) User controllers SERIALISED after last JSB
#      (Pinocchio segfaults if two models are built from URDF
#       concurrently — must be serialised)
############################################################################

import os
import sys
import tempfile

import xacro
import yaml
import franka_bringup.launch_utils as launch_utils

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CONTROLLER_EXAMPLE = 'controller'

load_yaml = launch_utils.load_yaml
get_parameter_for_config = launch_utils.get_parameter_for_config


def generate_rviz_config(configs, use_gazebo_any):
    displays = []
    displays.append({
        'Alpha': 0.5, 'Cell Size': 1,
        'Class': 'rviz_default_plugins/Grid',
        'Color': '160; 160; 164', 'Enabled': True,
        'Line Style': {'Line Width': 0.03, 'Value': 'Lines'},
        'Name': 'Grid', 'Normal Cell Count': 0,
        'Offset': {'X': 0, 'Y': 0, 'Z': 0},
        'Plane': 'XY', 'Plane Cell Count': 10,
        'Reference Frame': '<Fixed Frame>', 'Value': True,
    })
    for robot_key, config in configs.items():
        namespace = str(config.get('namespace', ''))
        arm_prefix = str(config.get('arm_prefix', ''))
        topic = f'/{namespace}/robot_description' if namespace else '/robot_description'
        display_name = f'RobotModel_{arm_prefix}' if arm_prefix else f'RobotModel_{robot_key}'
        displays.append({
            'Alpha': 1, 'Class': 'rviz_default_plugins/RobotModel',
            'Collision Enabled': False, 'Description File': '',
            'Description Source': 'Topic',
            'Description Topic': {
                'Depth': 5, 'Durability Policy': 'Volatile',
                'History Policy': 'Keep Last', 'Reliability Policy': 'Reliable',
                'Value': topic,
            },
            'Enabled': True,
            'Links': {
                'All Links Enabled': True, 'Expand Joint Details': False,
                'Expand Link Details': False, 'Expand Tree': False,
                'Link Tree Style': 'Links in Alphabetic Order',
            },
            'Name': display_name, 'TF Prefix': '',
            'Update Interval': 0, 'Value': True, 'Visual Enabled': True,
        })
    displays.append({
        'Class': 'rviz_default_plugins/TF', 'Enabled': False,
        'Frame Timeout': 15, 'Frames': {'All Enabled': True},
        'Marker Scale': 0.5, 'Name': 'TF',
        'Show Arrows': True, 'Show Axes': True, 'Show Names': False,
        'Tree': {}, 'Update Interval': 0, 'Value': True,
    })
    fixed_frame = 'world' if use_gazebo_any else 'base'
    return {
        'Panels': [
            {'Class': 'rviz_common/Displays', 'Help Height': 78, 'Name': 'Displays',
             'Property Tree Widget': {'Expanded': ['/Global Options1', '/Status1'], 'Splitter Ratio': 0.5},
             'Tree Height': 814},
            {'Class': 'rviz_common/Selection', 'Name': 'Selection'},
            {'Class': 'rviz_common/Views', 'Name': 'Views', 'Splitter Ratio': 0.5},
        ],
        'Visualization Manager': {
            'Class': '', 'Displays': displays, 'Enabled': True,
            'Global Options': {'Background Color': '48; 48; 48', 'Fixed Frame': fixed_frame, 'Frame Rate': 30},
            'Name': 'root',
            'Tools': [
                {'Class': 'rviz_default_plugins/Interact', 'Hide Inactive Objects': True},
                {'Class': 'rviz_default_plugins/MoveCamera'},
                {'Class': 'rviz_default_plugins/FocusCamera'},
            ],
            'Transformation': {'Current': {'Class': 'rviz_default_plugins/TF'}},
            'Value': True,
            'Views': {
                'Current': {
                    'Class': 'rviz_default_plugins/Orbit', 'Distance': 3.0,
                    'Enable Stereo Rendering': {'Value': False},
                    'Focal Point': {'X': 0.5, 'Y': 0.0, 'Z': 0.3},
                    'Focal Shape Fixed Size': True, 'Focal Shape Size': 0.05,
                    'Invert Z Axis': False, 'Name': 'Current View',
                    'Near Clip Distance': 0.01, 'Pitch': 0.4,
                    'Target Frame': '<Fixed Frame>', 'Value': 'Orbit (rviz)', 'Yaw': 1.58,
                },
                'Saved': None,
            },
        },
        'Window Geometry': {
            'Displays': {'collapsed': False}, 'Height': 1043,
            'Hide Left Dock': False, 'Hide Right Dock': False,
            'Selection': {'collapsed': False}, 'Views': {'collapsed': False},
            'Width': 1848, 'X': 72, 'Y': 0,
        },
    }


def _generate_gazebo_robot_description(config):
    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_gazebo_bringup'),
        'urdf', 'franka_arm.gazebo.xacro',
    )
    robot_description_config = xacro.process_file(
        franka_xacro_file,
        mappings={
            'robot_type': str(config['robot_type']),
            'arm_prefix': str(config.get('arm_prefix', '')),
            'hand': str(config.get('load_gripper', 'false')),
            'ros2_control': 'true', 'gazebo': 'true',
            'ee_id': str(config.get('franka_hand', 'franka_hand')),
            'gazebo_effort': str(config.get('gazebo_effort', 'true')),
            'xyz': str(config.get('base_xyz', '0 0 0')),
            'rpy': str(config.get('base_rpy', '0 0 0')),
            'robot_namespace': str(config.get('namespace', '')),
        },
    )
    return {'robot_description': robot_description_config.toxml()}


def _gazebo_nodes_for_robot(config):
    namespace = str(config.get('namespace', ''))
    robot_description = _generate_gazebo_robot_description(config)
    controller_name = str(config.get('controller', 'gravity_compensation_example_controller'))
    model_name = namespace if namespace else str(config.get('arm_prefix', 'franka'))
    arm_prefix = str(config.get('arm_prefix', ''))
    robot_type = str(config.get('robot_type', 'fr3'))

    description_topic = (
        f'/{namespace}/robot_description' if namespace
        else '/robot_description'
    )

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', namespace=namespace,
        output='screen', parameters=[robot_description],
    )
    spawn = Node(
        package='ros_gz_sim', executable='create',
        name=f'gz_spawn_{model_name}',
        arguments=['-topic', description_topic, '-name', model_name],
        output='screen',
    )
    jsb = Node(
        package='controller_manager', executable='spawner',
        namespace=namespace, name=f'jsb_spawner_{model_name}',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '30'],
        output='screen',
    )

    ctrl_override = {
        '/**': {controller_name: {'ros__parameters': {
            'arm_prefix': arm_prefix, 'robot_type': robot_type,
        }}}
    }
    ctrl_yaml_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False,
        prefix=f'franka_gz_ctrl_{model_name}_',
    )
    yaml.dump(ctrl_override, ctrl_yaml_tmp, default_flow_style=False)
    ctrl_yaml_tmp.close()

    ctrl = Node(
        package='controller_manager', executable='spawner',
        namespace=namespace, name=f'ctrl_spawner_{model_name}',
        arguments=[
            controller_name, '--controller-manager-timeout', '30',
            '--param-file', ctrl_yaml_tmp.name,
        ],
        output='screen',
    )

    
    return {'rsp': rsp, 'spawn': spawn, 'jsb': jsb, 'ctrl': ctrl}


def generate_robot_nodes(context):
    config_file = LaunchConfiguration('robot_config_file').perform(context)
    robot_ips = LaunchConfiguration('robot_ips').perform(context)
    gz_gui = LaunchConfiguration('gz_gui').perform(context).lower() == 'true'
    configs = load_yaml(config_file)
    nodes = []

    # Gazebo is used only if gazebo: "true" AND use_fake_hardware is not "false".
    # Real hardware always takes priority.
    use_gazebo_any = any(
        str(config.get('gazebo', 'false')).lower() == 'true'
        and str(config.get('use_fake_hardware', 'true')).lower() != 'false'
        for config in configs.values()
    )

    if use_gazebo_any:
        os.environ['GZ_SIM_RESOURCE_PATH'] = os.path.dirname(
            get_package_share_directory('franka_description')
        )

        no_gravity_world = os.path.join(
            get_package_share_directory('franka_gazebo_bringup'),
            'worlds', 'empty_no_gravity.sdf',
        )
        gz_args = (
            f'-r {no_gravity_world}' if gz_gui
            else f'-s -r {no_gravity_world}'
        )

        nodes.append(IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py',
            ]),
            launch_arguments={'gz_args': gz_args}.items(),
        ))
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='world_to_base_static_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--roll', '0', '--pitch', '0', '--yaw', '0',
                       '--frame-id', 'world', '--child-frame-id', 'base'],
            output='screen',
        ))
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_object_tf_publisher',
            arguments=[
                '--x', '-0.36',
                '--y', '-0.27',
                '--z', '0.57',
                '--yaw', '0.0',
                '--pitch', '0.0',
                '--roll', '0.0',
                '--frame-id', 'world',
                '--child-frame-id', 'box'
            ]
        ))
        # nodes.append(RegisterEventHandler(OnShutdown(on_shutdown=[
        #     ExecuteProcess(cmd=['pkill', '-SIGINT', '-f', 'gz sim'],
        #                    name='gz_sim_graceful_shutdown'),
        # ])))

    gz_robot_node_dicts = []

    for index, (robot_key, config) in enumerate(configs.items()):
        namespace = config.get('namespace', '')
        # Real hardware always takes priority over Gazebo.
        uses_real_hardware = str(config.get('use_fake_hardware', 'true')).lower() == 'false'
        is_gazebo = (
            str(config.get('gazebo', 'false')).lower() == 'true'
            and not uses_real_hardware
        )

        if is_gazebo:
            robot_nodes = _gazebo_nodes_for_robot(config)
            nodes.append(robot_nodes['rsp'])
            gz_robot_node_dicts.append(robot_nodes)
        else:
            if robot_ips:
                robot_ip = get_parameter_for_config(
                    robot_ips, num_configs=len(configs), config_index=index)
            else:
                robot_ip = str(config['robot_ip'])

            nodes.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([
                    FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py',
                ])),
                launch_arguments={
                    'robot_type': str(config['robot_type']),
                    'arm_prefix': str(config['arm_prefix']),
                    'namespace': str(namespace),
                    'robot_ip': robot_ip,
                    'load_gripper': str(config['load_gripper']),
                    'use_fake_hardware': str(config['use_fake_hardware']),
                    'fake_sensor_commands': str(config['fake_sensor_commands']),
                    'joint_state_rate': str(config['joint_state_rate']),
                    'base_xyz': str(config.get('base_xyz', '0 0 0')),
                    'base_rpy': str(config.get('base_rpy', '0 0 0')),
                }.items(),
            ))

            controller_name = str(
                config.get('controller', 'gravity_compensation_example_controller'))

            if CONTROLLER_EXAMPLE in controller_name:
                nodes.append(Node(
                    package='controller_manager', executable='spawner',
                    namespace=namespace,
                    arguments=[controller_name, '--controller-manager-timeout', '30'],
                    parameters=[PathJoinSubstitution([
                        FindPackageShare('franka_bringup'), 'config', 'controllers.yaml',
                    ])],
                    output='screen',
                ))
            else:
                nodes.append(Node(
                    package='franka_example_controllers', executable=controller_name,
                    namespace=namespace, output='screen',
                ))

    # Launch RealSense cameras for each robot that has use_camera: "true" and uses real hardware
    for robot_key, config in configs.items():
        #uses_real_hw = str(config.get('use_fake_hardware', 'true')).lower() == 'false'
        use_camera = str(config.get('use_camera', 'false')).lower() == 'true'
        if use_camera:
            arm_prefix = str(config.get('arm_prefix', ''))
            namespace = str(config.get('namespace', ''))
            serial_no = str(config.get('camera_serial_no', ''))
            nodes.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('realsense2_camera'),
                        'launch',
                        'rs_launch.py'
                    )
                ),
                launch_arguments={
                    'publish_tf': 'false',
                    'serial_no': f"'{serial_no}'",
                    'camera_name': f'camera_{arm_prefix}',
                    'camera_namespace': namespace,
                    'rgb_camera.color_profile': '640x480x30',
                    'enable_depth': 'false',
                }.items(),
            ))

    # ------------------------------------------------------------------
    # Gazebo bringup
    #
    # Phase 1: ALL spawns in parallel.
    # Phase 2: ALL JSBs in parallel (each triggered by its own spawn).
    # Phase 3: User controllers SERIALISED after LAST JSB.
    #          (Pinocchio segfaults on concurrent URDF parsing)
    #
    # Timeline:
    #   spawn_1 ──→ jsb_1 ──┐
    #   spawn_2 ──→ jsb_2 ──┤
    #                       └──→ ctrl_1 → ctrl_2
    # ------------------------------------------------------------------
    if gz_robot_node_dicts:

        # --- Phase 1: all spawns in parallel ---
        all_spawns = [rd['spawn'] for rd in gz_robot_node_dicts]
        nodes.extend(all_spawns)

        # --- Phase 2: JSBs serialised (ClassLoader is not thread-safe) ---
        # First JSB triggered by first spawn
        nodes.append(RegisterEventHandler(event_handler=OnProcessExit(
            target_action=gz_robot_node_dicts[0]['spawn'],
            on_exit=[gz_robot_node_dicts[0]['jsb']],
        )))

        # Subsequent JSBs chained after previous JSB
        for i in range(1, len(gz_robot_node_dicts)):
            nodes.append(RegisterEventHandler(event_handler=OnProcessExit(
                target_action=gz_robot_node_dicts[i - 1]['jsb'],
                on_exit=[gz_robot_node_dicts[i]['jsb']],
            )))

        # --- Phase 3: controllers serialised after last JSB ---
        last_jsb = gz_robot_node_dicts[-1]['jsb']

        nodes.append(RegisterEventHandler(event_handler=OnProcessExit(
            target_action=last_jsb,
            on_exit=[gz_robot_node_dicts[0]['ctrl']],
        )))

        for i in range(1, len(gz_robot_node_dicts)):
            nodes.append(RegisterEventHandler(event_handler=OnProcessExit(
                target_action=gz_robot_node_dicts[i - 1]['ctrl'],
                on_exit=[gz_robot_node_dicts[i]['ctrl']],
            )))

    if any(str(config.get('use_rviz', 'false')).lower() == 'true'
           for config in configs.values()):
        rviz_config = generate_rviz_config(configs, use_gazebo_any)
        rviz_tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.rviz', delete=False, prefix='franka_multi_')
        yaml.dump(rviz_config, rviz_tmp, default_flow_style=False)
        rviz_tmp.close()
        nodes.append(Node(
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['--display-config', rviz_tmp.name], output='screen',
        ))
    nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/right_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
                '/right_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
                '/right_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            ],
            output='screen',
        ))
    nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['--x', '-0.338', '--y', '-0.453', '--z', '0',
                    '--roll', '0', '--pitch', '0', '--yaw', '1.570796',
                    '--frame-id', 'world',
                    '--child-frame-id', 'left_fr3_link0'],
            output='screen',
        ))
    nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['--x', '-0.343', '--y', '0.416', '--z', '0',
                    '--roll', '0', '--pitch', '0', '--yaw', '-1.570796',
                    '--frame-id', 'world',
                    '--child-frame-id', 'right_fr3_link0'],
            output='screen',
        ))


    return nodes





def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('franka_bringup'), 'config', 'franka.config.yaml',
            ]),
            description='Path to the robot configuration file to load',
        ),
        DeclareLaunchArgument(
            'robot_ips', default_value='',
            description='Comma-separated list of IP addresses (optional).',
        ),
        DeclareLaunchArgument(
            'gz_gui', default_value='true',
            description='Set to "true" to launch Gazebo with GUI.',
        ),
        OpaqueFunction(function=generate_robot_nodes),
    ])