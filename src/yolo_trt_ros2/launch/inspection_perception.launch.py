from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory('yolo_trt_ros2')
    default_config = os.path.join(pkg_share, 'config', 'inspection_perception.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to detector and coordinate projector parameter YAML file.',
    )
    web_ui_arg = DeclareLaunchArgument(
        'webUI',
        default_value='false',
        description='Enable the in-process HTTP WebUI and preview generation.',
    )
    automation_arg = DeclareLaunchArgument(
        'automation',
        default_value='false',
        description='Start the status-gated cabinet automation bridge.',
    )
    automation_execute_arg = DeclareLaunchArgument(
        'automation_execute',
        default_value='false',
        description='Allow the automation bridge to send commands to the LowCmd worker.',
    )
    confidence_arg = DeclareLaunchArgument(
        'confidence',
        default_value='0.30',
        description='Unified YOLOE 2D and coordinate-projector confidence threshold.',
    )
    camera_serial_arg = DeclareLaunchArgument(
        'cam_serial',
        default_value='',
        description='RealSense serial number. Always set this when multiple cameras are connected.',
    )
    camera_index_arg = DeclareLaunchArgument(
        'cam_index',
        default_value='-1',
        description='Optional RealSense index override; -1 keeps the YAML value.',
    )
    handeye_mode_arg = DeclareLaunchArgument(
        'handeye_mode',
        default_value='eye-in-hand',
        description='eye-in-hand for wrist camera or eye-to-hand for torso camera.',
    )
    handeye_path_arg = DeclareLaunchArgument(
        'handeye_npy_path',
        default_value='/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/outputs/eye_in_hand_20260726_115420.json',
        description='Hand-eye JSON or NPY directory.',
    )
    handeye_target_frame_arg = DeclareLaunchArgument(
        'handeye_target_frame',
        default_value='torso_link',
        description='Reference frame declared by the calibration.',
    )
    base_link_arg = DeclareLaunchArgument(
        'base_link',
        default_value='torso_link',
        description='FK reference frame used for the target arm.',
    )
    hand_link_arg = DeclareLaunchArgument(
        'hand_link',
        default_value='right_wrist_yaw_link',
        description='Target wrist link; may be switched to left_wrist_yaw_link.',
    )
    dex1_tip_arg = DeclareLaunchArgument(
        'dex1_tip_from_wrist_xyz',
        default_value='[0.14, 0.01, 0.012]',
        description='Dex1-1 fingertip/contact point offset from right_wrist_yaw_link, meters.',
    )
    apply_tip_compensation_arg = DeclareLaunchArgument(
        'apply_tip_compensation',
        default_value='true',
        description='Apply Dex1 tip-to-control-frame target compensation.',
    )
    blue_point_offset_arg = DeclareLaunchArgument(
        'blue_point_target_world_offset_xyz',
        default_value='[0.0, 0.001, -0.004]',
        description='World-frame offset applied only to the lock force/contact point, meters.',
    )
    handeye_mount_offset_arg = DeclareLaunchArgument(
        'handeye_mount_offset_from_wrist_xyz',
        default_value='[0.0, 0.0, 0.0]',
        description='Calibration hand-frame offset from wrist; zero for right_wrist_yaw_link calibration.',
    )

    # Global ROS parameter overrides are harmless for the other in-process
    # nodes: only coordinate_projector declares these parameter names.
    ros_arguments = [
        '--ros-args',
        '--params-file',
        LaunchConfiguration('config_file'),
        '-p',
        ["camera_serial_override:='", LaunchConfiguration('cam_serial'), "'"],
        '-p',
        ['camera_index_override:=', LaunchConfiguration('cam_index')],
        '-p',
        ['conf_thres:=', LaunchConfiguration('confidence')],
        '-p',
        ['min_confidence:=', LaunchConfiguration('confidence')],
        '-p',
        ['handeye_mode:=', LaunchConfiguration('handeye_mode')],
        '-p',
        ['handeye_npy_path:=', LaunchConfiguration('handeye_npy_path')],
        '-p',
        ['handeye_target_frame:=', LaunchConfiguration('handeye_target_frame')],
        '-p',
        ['base_link:=', LaunchConfiguration('base_link')],
        '-p',
        ['hand_link:=', LaunchConfiguration('hand_link')],
        '-p',
        ['dex1_tip_from_wrist_xyz:=', LaunchConfiguration('dex1_tip_from_wrist_xyz')],
        '-p',
        ['apply_tip_compensation:=', LaunchConfiguration('apply_tip_compensation')],
        '-p',
        ['ik_end_effector_offset_xyz:=', LaunchConfiguration('handeye_mount_offset_from_wrist_xyz')],
        '-p',
        ['blue_point_target_world_offset_xyz:=', LaunchConfiguration('blue_point_target_world_offset_xyz')],
        '-p',
        ['handeye_mount_offset_from_wrist_xyz:=', LaunchConfiguration('handeye_mount_offset_from_wrist_xyz')],
    ]
    process_env = {
        'LD_LIBRARY_PATH': [
            '/opt/ros/humble/lib:',
            EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
        ],
    }
    process_prefix = ['/usr/bin/python3', '-m', 'yolo_trt_ros2.integrated_perception_node']

    integrated_node = ExecuteProcess(
        cmd=process_prefix + ros_arguments,
        additional_env=process_env,
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('webUI')),
    )

    integrated_node_with_web = ExecuteProcess(
        cmd=process_prefix + ['--webUI'] + ros_arguments,
        additional_env=process_env,
        output='screen',
        condition=IfCondition(LaunchConfiguration('webUI')),
    )
    automation_node = ExecuteProcess(
        cmd=[
            '/usr/bin/python3',
            '-m',
            'yolo_trt_ros2.cabinet_automation_node',
            '--ros-args',
            '--params-file',
            LaunchConfiguration('config_file'),
            '-p',
            ['execute_enabled:=', LaunchConfiguration('automation_execute')],
            '-p',
            ['min_confidence:=', LaunchConfiguration('confidence')],
        ],
        additional_env=process_env,
        output='screen',
        condition=IfCondition(LaunchConfiguration('automation')),
    )

    return LaunchDescription([
        config_arg,
        web_ui_arg,
        automation_arg,
        automation_execute_arg,
        confidence_arg,
        camera_serial_arg,
        camera_index_arg,
        handeye_mode_arg,
        handeye_path_arg,
        handeye_target_frame_arg,
        base_link_arg,
        hand_link_arg,
        dex1_tip_arg,
        apply_tip_compensation_arg,
        blue_point_offset_arg,
        handeye_mount_offset_arg,
        integrated_node,
        integrated_node_with_web,
        automation_node,
    ])
