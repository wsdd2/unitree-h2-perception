#!/usr/bin/env python3
import json
import math
import os
import re
import sys
import time
import ctypes
from collections import deque
from pathlib import Path


def _bootstrap_h2_python_paths():
    # H2 daily launch runs from system Python. Keep these paths local to this
    # node so console-script metadata issues and stripped PYTHONPATH do not
    # break rclpy or Unitree SDK imports.
    for path in (
        '/opt/ros/humble/local/lib/python3.10/dist-packages',
        '/opt/ros/humble/lib/python3.10/site-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/detector_msgs/local/lib/python3.10/dist-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/detector_msgs/lib/python3.10/site-packages',
        '/home/unitree/MscapeTech/unitree_sdk2_python',
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    ros_lib_dirs = []
    ros_lib_root = '/opt/ros/humble/lib'
    if os.path.isdir(ros_lib_root):
        ros_lib_dirs.append(ros_lib_root)
        for root, _dirs, files in os.walk(ros_lib_root):
            if 'librcl_action.so' in files and root not in ros_lib_dirs:
                ros_lib_dirs.append(root)

    old_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    old_ld_parts = [part for part in old_ld_library_path.split(':') if part]
    missing_ld_dirs = [path for path in ros_lib_dirs if path not in old_ld_parts]
    if missing_ld_dirs:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(missing_ld_dirs + old_ld_parts)
        # The dynamic loader reads LD_LIBRARY_PATH at process startup. If this
        # process was spawned with a stripped environment, restart once so
        # rclpy's C extension can resolve librcl_action.so and friends.
        if os.environ.get('YOLO_TRT_ROS2_LD_BOOTSTRAPPED') != '1':
            os.environ['YOLO_TRT_ROS2_LD_BOOTSTRAPPED'] = '1'
            os.execv(sys.executable, [sys.executable] + sys.argv)

    for library in (
        'librcl_action.so',
        'librcl.so',
        'librmw.so',
        'librcutils.so',
    ):
        for directory in ros_lib_dirs:
            path = os.path.join(directory, library)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    pass


_bootstrap_h2_python_paths()

import numpy as np
import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

try:
    import tf2_ros
except ImportError:  # pragma: no cover - tf2_ros is provided by ROS.
    tf2_ros = None

from detector_msgs.msg import Object3D, Object3DArray, RobotInspectionStatus
from detector_msgs.msg import Object2DArray
from yolo_trt_ros2.work_tag_detector import (
    is_hang_hook_class,
    is_work_tag_grasp_class,
    stage_prefers_hang_hook,
    stage_prefers_work_tag_grasp,
)


G1_H2_JOINT_INDEX = {
    'waist_yaw_joint': 12,
    'waist_roll_joint': 13,
    'waist_pitch_joint': 14,
    'left_shoulder_pitch_joint': 15,
    'left_shoulder_roll_joint': 16,
    'left_shoulder_yaw_joint': 17,
    'left_elbow_joint': 18,
    'left_wrist_roll_joint': 19,
    'left_wrist_pitch_joint': 20,
    'left_wrist_yaw_joint': 21,
    'right_shoulder_pitch_joint': 22,
    'right_shoulder_roll_joint': 23,
    'right_shoulder_yaw_joint': 24,
    'right_elbow_joint': 25,
    'right_wrist_roll_joint': 26,
    'right_wrist_pitch_joint': 27,
    'right_wrist_yaw_joint': 28,
}

H2_XR_DUAL_ARM_JOINTS = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]


class UnitreeLowStateJointReader:
    """Tiny Unitree lowstate reader used only when eye-in-hand FK is enabled."""

    def __init__(self, network_interface='', domain_id=0, lowstate_topic='rt/lowstate'):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self._latest = None
        self.topic = lowstate_topic or 'rt/lowstate'
        if network_interface:
            ChannelFactoryInitialize(int(domain_id), str(network_interface))
        else:
            ChannelFactoryInitialize(int(domain_id))
        self._subscriber = ChannelSubscriber(self.topic, LowState_)
        self._subscriber.Init(self._on_lowstate, 10)

    def _on_lowstate(self, msg):
        self._latest = msg

    def wait(self, timeout_sec=2.0):
        deadline = time.time() + float(timeout_sec)
        while self._latest is None and time.time() < deadline:
            time.sleep(0.02)
        return self._latest is not None

    def joint_positions_by_name(self):
        if self._latest is None:
            return {}
        return {
            name: float(self._latest.motor_state[index].q)
            for name, index in G1_H2_JOINT_INDEX.items()
        }


class CoordinateProjectorNode(Node):
    """Project 2D detector centers into camera-frame 3D coordinates."""

    def __init__(self, subscribe_perception_inputs=True, parameter_overrides=None):
        node_kwargs = {}
        if parameter_overrides is not None:
            node_kwargs['parameter_overrides'] = parameter_overrides
        super().__init__('coordinate_projector', **node_kwargs)
        self._declare_parameters()

        self.objects_topic = self.get_parameter('objects_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.objects_3d_topic = self.get_parameter('objects_3d_topic').value
        self.target_point_topic = self.get_parameter('target_point_topic').value
        self.target_pose_topic = self.get_parameter('target_pose_topic').value
        self.target_joint_state_topic = self.get_parameter('target_joint_state_topic').value
        self.current_joint_state_topic = self.get_parameter('current_joint_state_topic').value
        self.current_ee_point_topic = self.get_parameter('current_ee_point_topic').value
        self.objects_ik_topic = self.get_parameter('objects_ik_topic').value
        self.pixel_query_topic = self.get_parameter('pixel_query_topic').value
        self.pixel_query_result_topic = self.get_parameter('pixel_query_result_topic').value
        self.robot_status_topic = self.get_parameter('robot_status_topic').value
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.handeye_npy_path = str(self.get_parameter('handeye_npy_path').value)
        self.handeye_mode = str(self.get_parameter('handeye_mode').value).strip().lower().replace('_', '-')
        self.handeye_target_frame = str(self.get_parameter('handeye_target_frame').value)
        self.robot_model = str(self.get_parameter('robot_model').value).strip().lower()
        self.workspace_root = str(self.get_parameter('workspace_root').value)
        self.urdf_path = str(self.get_parameter('urdf_path').value)
        self.base_link = str(self.get_parameter('base_link').value)
        self.hand_link = str(self.get_parameter('hand_link').value)
        self.camera_link = str(self.get_parameter('camera_link').value)
        self.network_interface = str(self.get_parameter('network_interface').value)
        self.domain_id = int(self.get_parameter('domain_id').value)
        self.lowstate_topic = str(self.get_parameter('lowstate_topic').value)
        self.joint_json_path = str(self.get_parameter('joint_json_path').value)
        self.lock_waist = bool(self.get_parameter('lock_waist').value)
        self.handeye_mount_offset_from_wrist_xyz = self._get_float_list_parameter(
            'handeye_mount_offset_from_wrist_xyz',
            [0.0, 0.0, 0.0],
            3,
        )
        self.xr_ee_offset_from_wrist_xyz = self._get_float_list_parameter(
            'xr_ee_offset_from_wrist_xyz',
            [0.0, 0.0, 0.0],
            3,
        )
        self.dex1_tip_from_wrist_xyz = self._get_float_list_parameter('dex1_tip_from_wrist_xyz', [0.05, 0.0, 0.0], 3)
        self.apply_tip_compensation = bool(self.get_parameter('apply_tip_compensation').value)
        self.blue_point_target_world_offset_xyz = self._get_float_list_parameter(
            'blue_point_target_world_offset_xyz',
            [0.0, 0.0, 0.0],
            3,
        )
        self.fk_backend_requested = str(self.get_parameter('fk_backend').value).strip().lower()
        self.fk_backend_active = 'none'
        self.h2_xr_scripts_dir = str(self.get_parameter('h2_xr_scripts_dir').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.depth_radius = int(self.get_parameter('depth_radius').value)
        self.depth_min_valid_count = int(self.get_parameter('depth_min_valid_count').value)
        self.depth_percentile = float(self.get_parameter('depth_percentile').value)
        self.press_max_depth_m = float(self.get_parameter('press_max_depth_m').value)
        self.depth_fallback = str(self.get_parameter('depth_fallback').value).strip().lower()
        self.depth_roi_scale = float(self.get_parameter('depth_roi_scale').value)
        self.depth_expanded_roi_scale = float(self.get_parameter('depth_expanded_roi_scale').value)
        self.handle_depth_grasp_fallback = bool(self.get_parameter('handle_depth_grasp_fallback').value)
        self.handle_depth_search_left_px = int(self.get_parameter('handle_depth_search_left_px').value)
        self.handle_depth_search_right_px = int(self.get_parameter('handle_depth_search_right_px').value)
        self.handle_depth_search_y_px = int(self.get_parameter('handle_depth_search_y_px').value)
        self.handle_depth_near_delta_m = float(self.get_parameter('handle_depth_near_delta_m').value)
        self.handle_depth_max_blue_distance_px = float(self.get_parameter('handle_depth_max_blue_distance_px').value)
        self.handle_depth_sticky_px = float(self.get_parameter('handle_depth_sticky_px').value)
        self.handle_mid_right_offset_m = float(self.get_parameter('handle_mid_right_offset_m').value)
        self.press_point_ground_ratio = max(
            0.0,
            min(1.0, float(self.get_parameter('press_point_ground_ratio').value)),
        )
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.foreground_max_depth_m = float(self.get_parameter('foreground_max_depth_m').value)
        self.handle_max_depth_m = float(self.get_parameter('handle_max_depth_m').value)
        self.control_row_z_tolerance_m = float(
            self.get_parameter('control_row_z_tolerance_m').value
        )
        self.pixel_query_max_depth_m = float(self.get_parameter('pixel_query_max_depth_m').value)
        self.publish_invalid = bool(self.get_parameter('publish_invalid').value)
        self.publish_target_pose = bool(self.get_parameter('publish_target_pose').value)
        self.publish_target_joint_state = bool(self.get_parameter('publish_target_joint_state').value)
        self.publish_current_joint_state = bool(self.get_parameter('publish_current_joint_state').value)
        self.publish_objects_ik_json = bool(self.get_parameter('publish_objects_ik_json').value)
        self.publish_failed_ik_solution = bool(self.get_parameter('publish_failed_ik_solution').value)
        self.stale_depth_sec = float(self.get_parameter('stale_depth_sec').value)
        self.depth_sync_tolerance_sec = float(self.get_parameter('depth_sync_tolerance_sec').value)
        self.depth_cache_size = int(self.get_parameter('depth_cache_size').value)
        self.transform_timeout_sec = float(self.get_parameter('transform_timeout_sec').value)
        self.diagnostic_3d_log_enabled = bool(self.get_parameter('diagnostic_3d_log_enabled').value)
        self.diagnostic_3d_log_interval_sec = max(
            0.0,
            float(self.get_parameter('diagnostic_3d_log_interval_sec').value),
        )
        self.diagnostic_3d_log_change_m = max(
            1e-6,
            float(self.get_parameter('diagnostic_3d_log_change_m').value),
        )
        self.class_filter = set(self._get_string_list_parameter('class_filter'))
        self.pose_orientation_xyzw = self._get_float_list_parameter(
            'target_pose_orientation_xyzw',
            [0.0, 0.0, 0.0, 1.0],
            4,
        )
        self.ik_target_link = str(self.get_parameter('ik_target_link').value)
        self.ik_active_joints = self._get_string_list_parameter('ik_active_joints')
        self.ik_end_effector_offset_xyz = self._get_float_list_parameter('ik_end_effector_offset_xyz', [0.0, 0.0, 0.0], 3)
        self.ik_target_position_offset_xyz = self._get_float_list_parameter('ik_target_position_offset_xyz', [0.0, 0.0, 0.0], 3)
        self.ik_max_iterations = int(self.get_parameter('ik_max_iterations').value)
        self.ik_tolerance_position = float(self.get_parameter('ik_tolerance_position').value)
        self.ik_tolerance_orientation = float(self.get_parameter('ik_tolerance_orientation').value)
        self.ik_damping = float(self.get_parameter('ik_damping').value)
        self.ik_step_scale = float(self.get_parameter('ik_step_scale').value)
        self.ik_position_weight = float(self.get_parameter('ik_position_weight').value)
        self.ik_orientation_weight = float(self.get_parameter('ik_orientation_weight').value)
        self.max_ik_objects = int(self.get_parameter('max_ik_objects').value)

        self.bridge = CvBridge()
        self.latest_depth_stamp_ns = 0
        self.latest_depth = None
        self.depth_cache = deque(maxlen=max(2, self.depth_cache_size))
        self.latest_camera_info = None
        self.handeye_transform = None
        self.T_cam2hand = None
        self._fk_model = None
        self._xr_ik = None
        self._xr_current_ee_poses = None
        self._ik_solver = None
        self._joint_reader = None
        self._fk_error = ''
        self._ik_error = ''
        self._last_ik_warning_sec = 0.0
        self._last_fk_warning_sec = 0.0
        self._last_depth_sync_warning_sec = 0.0
        self._last_3d_diagnostic_log_sec = 0.0
        self._last_3d_diagnostic_snapshot = None
        self._last_depth_handle_grasp = None
        self._frame_handle_reference = None
        self.latest_robot_status = None
        self._configure_handeye_and_fk()

        self.tf_buffer = None
        self.tf_listener = None
        if self.target_frame and self.handeye_transform is None and self.T_cam2hand is None:
            if tf2_ros is None:
                self.get_logger().warn('tf2_ros is not available; target_frame transform disabled.')
            else:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.objects_3d_pub = self.create_publisher(Object3DArray, self.objects_3d_topic, 10)
        self.target_point_pub = self.create_publisher(PointStamped, self.target_point_topic, 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)
        self.target_joint_state_pub = self.create_publisher(JointState, self.target_joint_state_topic, 10)
        self.current_joint_state_pub = self.create_publisher(JointState, self.current_joint_state_topic, 10)
        self.current_ee_point_pub = self.create_publisher(PointStamped, self.current_ee_point_topic, 10)
        self.objects_ik_pub = self.create_publisher(String, self.objects_ik_topic, 10)
        self.pixel_query_result_pub = self.create_publisher(String, self.pixel_query_result_topic, 10)

        self.depth_sub = None
        self.camera_info_sub = None
        self.objects_sub = None
        if subscribe_perception_inputs:
            self.depth_sub = self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self._camera_info_callback,
                10,
            )
            self.objects_sub = self.create_subscription(Object2DArray, self.objects_topic, self._objects_callback, 10)
        self.create_subscription(PointStamped, self.pixel_query_topic, self._pixel_query_callback, 10)
        self.create_subscription(RobotInspectionStatus, self.robot_status_topic, self._robot_status_callback, 10)
        self.create_timer(0.1, self._current_joint_timer_callback)

        self.get_logger().info(
            'Coordinate projector started: input=%s, objects=%s, depth=%s, camera_info=%s, target_frame=%s, handeye_mode=%s, handeye=%s'
            % (
                'ROS topics' if subscribe_perception_inputs else 'in-process RGB-D',
                self.objects_topic,
                self.depth_topic,
                self.camera_info_topic,
                self._output_frame_name(),
                self.handeye_mode or '<none>',
                self.handeye_npy_path or '<none>',
            )
        )
        effective_tip = (
            np.asarray(self.dex1_tip_from_wrist_xyz, dtype=np.float64)
            - np.asarray(self.handeye_mount_offset_from_wrist_xyz, dtype=np.float64)
        )
        self.get_logger().info(
            'Target compensation active: dex1_tip_from_wrist=%s '
            'handeye_mount_offset=%s xr_ee_offset=%s effective_tip_from_mount=%s '
            'apply_tip_compensation=%s lock_world_offset=%s depth_scale=%.6f'
            % (
                self._format_xyz(self.dex1_tip_from_wrist_xyz),
                self._format_xyz(self.handeye_mount_offset_from_wrist_xyz),
                self._format_xyz(self.xr_ee_offset_from_wrist_xyz),
                self._format_xyz(effective_tip),
                self.apply_tip_compensation,
                self._format_xyz(self.blue_point_target_world_offset_xyz),
                self.depth_scale,
            )
        )

    def _declare_parameters(self):
        self.declare_parameter('objects_topic', '/detector/objects')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('objects_3d_topic', '/detector/objects_3d')
        self.declare_parameter('target_point_topic', '/detector/target_point')
        self.declare_parameter('target_pose_topic', '/detector/target_pose')
        self.declare_parameter('target_joint_state_topic', '/detector/target_joint_state')
        self.declare_parameter('current_joint_state_topic', '/detector/current_joint_state')
        self.declare_parameter('current_ee_point_topic', '/detector/current_ee_point')
        self.declare_parameter('objects_ik_topic', '/detector/objects_ik_json')
        self.declare_parameter('pixel_query_topic', '/detector/pixel_query')
        self.declare_parameter('pixel_query_result_topic', '/detector/pixel_query_result_json')
        self.declare_parameter('robot_status_topic', '/robot/inspection_status')
        self.declare_parameter('target_frame', '')
        self.declare_parameter('handeye_npy_path', '')
        self.declare_parameter('handeye_mode', 'eye-to-hand')
        self.declare_parameter('handeye_target_frame', 'base_link')
        self.declare_parameter('robot_model', 'h2')
        self.declare_parameter('workspace_root', '/home/unitree/MscapeTech')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('base_link', '')
        self.declare_parameter('hand_link', '')
        self.declare_parameter('camera_link', '')
        self.declare_parameter('network_interface', '')
        self.declare_parameter('domain_id', 0)
        self.declare_parameter('lowstate_topic', 'rt/lowstate')
        self.declare_parameter('joint_json_path', '')
        self.declare_parameter('lock_waist', True)
        self.declare_parameter('handeye_mount_offset_from_wrist_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('xr_ee_offset_from_wrist_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('dex1_tip_from_wrist_xyz', [0.05, 0.0, 0.0])
        self.declare_parameter('apply_tip_compensation', True)
        self.declare_parameter('blue_point_target_world_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('fk_backend', 'auto')
        self.declare_parameter('h2_xr_scripts_dir', '/home/unitree/H2_joint_cartesian/scripts')
        self.declare_parameter('class_filter', '')
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('depth_radius', 3)
        self.declare_parameter('depth_min_valid_count', 8)
        self.declare_parameter('depth_percentile', 35.0)
        self.declare_parameter('press_max_depth_m', 1.2)
        self.declare_parameter('depth_fallback', 'bbox')
        self.declare_parameter('depth_roi_scale', 0.55)
        self.declare_parameter('depth_expanded_roi_scale', 1.8)
        self.declare_parameter('handle_depth_grasp_fallback', True)
        self.declare_parameter('handle_depth_search_left_px', 360)
        self.declare_parameter('handle_depth_search_right_px', 18)
        self.declare_parameter('handle_depth_search_y_px', 70)
        self.declare_parameter('handle_depth_near_delta_m', 0.015)
        self.declare_parameter('handle_depth_max_blue_distance_px', 180.0)
        self.declare_parameter('handle_depth_sticky_px', 45.0)
        self.declare_parameter('handle_mid_right_offset_m', 0.01)
        self.declare_parameter('press_point_ground_ratio', 0.5)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('foreground_max_depth_m', 1.2)
        self.declare_parameter('handle_max_depth_m', 1.0)
        self.declare_parameter('control_row_z_tolerance_m', 0.04)
        self.declare_parameter('pixel_query_max_depth_m', 1.2)
        self.declare_parameter('publish_invalid', False)
        self.declare_parameter('publish_target_pose', True)
        self.declare_parameter('publish_target_joint_state', False)
        self.declare_parameter('publish_current_joint_state', True)
        self.declare_parameter('publish_objects_ik_json', True)
        self.declare_parameter('publish_failed_ik_solution', False)
        self.declare_parameter('target_pose_orientation_xyzw', [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('ik_target_link', '')
        self.declare_parameter('ik_active_joints', '')
        self.declare_parameter('ik_end_effector_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('ik_target_position_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('ik_max_iterations', 120)
        self.declare_parameter('ik_tolerance_position', 0.005)
        self.declare_parameter('ik_tolerance_orientation', 3.2)
        self.declare_parameter('ik_damping', 0.001)
        self.declare_parameter('ik_step_scale', 0.4)
        self.declare_parameter('ik_position_weight', 1.0)
        self.declare_parameter('ik_orientation_weight', 0.0)
        self.declare_parameter('max_ik_objects', 24)
        self.declare_parameter('stale_depth_sec', 0.5)
        self.declare_parameter('depth_sync_tolerance_sec', 0.08)
        self.declare_parameter('depth_cache_size', 60)
        self.declare_parameter('transform_timeout_sec', 0.05)
        self.declare_parameter('diagnostic_3d_log_enabled', True)
        self.declare_parameter('diagnostic_3d_log_interval_sec', 5.0)
        self.declare_parameter('diagnostic_3d_log_change_m', 0.001)

    def _get_string_list_parameter(self, name):
        value = self.get_parameter(name).value
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def _get_float_list_parameter(self, name, default, expected_len):
        value = self.get_parameter(name).value
        if value is None:
            return default
        try:
            values = [float(item) for item in value]
        except TypeError:
            values = default
        if len(values) != expected_len:
            self.get_logger().warn('%s must contain %d values; using default.' % (name, expected_len))
            return default
        return values

    def _configure_handeye_and_fk(self):
        if not self.handeye_npy_path:
            return

        if self.handeye_mode in ('eye-in-hand', 'eyeinhand'):
            self.T_cam2hand = self._load_eye_in_hand_transform(self.handeye_npy_path)
            if self.T_cam2hand is not None:
                self._configure_fk()
            return

        self.handeye_transform = self._load_handeye_transform(self.handeye_npy_path)
        if self.handeye_transform is not None and (self.publish_target_joint_state or self.publish_objects_ik_json):
            self._configure_fk()

    def _workspace_path(self, *parts):
        return Path(os.path.expanduser(self.workspace_root)).joinpath(*parts)

    def _resolve_urdf_path(self):
        if self.urdf_path:
            return Path(os.path.expanduser(self.urdf_path))
        if self.robot_model == 'h2':
            return self._workspace_path('unitree_ros', 'robots', 'h2_description', 'H2.urdf')
        return self._workspace_path('unitree_ros', 'robots', 'g1_description', 'g1_29dof_rev_1_0.urdf')

    def _default_base_link(self):
        if self.base_link:
            return self.base_link
        if self.robot_model == 'h2':
            return 'torso_link'
        return 'pelvis'

    def _default_hand_link(self):
        if self.hand_link:
            return self.hand_link
        return 'right_wrist_yaw_link'

    def _configure_fk(self):
        urdf = self._resolve_urdf_path()
        if not urdf.is_file() and self.fk_backend_requested in ('auto', 'urdf'):
            self._fk_error = 'URDF does not exist: %s' % urdf
            self.get_logger().warn(self._fk_error)
            if self.fk_backend_requested == 'urdf':
                return

        try:
            if urdf.is_file():
                workspace = Path(os.path.expanduser(self.workspace_root))
                robot_kinematics_dir = workspace / 'robot_kinematics'
                joint_to_pose_dir = robot_kinematics_dir / 'joint_to_pose'
                pose_to_joint_dir = robot_kinematics_dir / 'pose_to_joint'
                for path in (robot_kinematics_dir, joint_to_pose_dir, pose_to_joint_dir):
                    text = str(path)
                    if text not in sys.path:
                        sys.path.insert(0, text)
                from fk_urdf import URDFFK

                self._fk_model = URDFFK(str(urdf))
                self.urdf_path = str(urdf)
            self.base_link = self._default_base_link()
            self.hand_link = self._default_hand_link()
            self.get_logger().info(
                'Eye-in-hand FK enabled: robot=%s, backend=%s, urdf=%s, base=%s, hand=%s'
                % (self.robot_model, self.fk_backend_requested, self.urdf_path or '<none>', self.base_link, self.hand_link)
            )
        except Exception as exc:
            self._fk_error = 'Failed to initialize FK: %s' % exc
            self.get_logger().warn(self._fk_error)
            if self.fk_backend_requested == 'urdf':
                return

        if self.publish_target_joint_state or self.publish_objects_ik_json:
            try:
                from ik_urdf import URDFIK

                self._ik_solver = URDFIK(str(urdf))
                if not self.ik_target_link:
                    self.ik_target_link = self.hand_link
                self.get_logger().info(
                    'IK target joint publisher enabled: topic=%s, base=%s, target_link=%s, active_joints=%s'
                    % (
                        self.target_joint_state_topic,
                        self.base_link,
                        self.ik_target_link,
                        ','.join(self.ik_active_joints) if self.ik_active_joints else '<chain-default>',
                    )
                )
            except Exception as exc:
                self._ik_error = 'Failed to initialize IK: %s' % exc
                self.get_logger().warn(self._ik_error)

        if self.joint_json_path:
            return

        try:
            self._joint_reader = UnitreeLowStateJointReader(
                self.network_interface,
                self.domain_id,
                self.lowstate_topic,
            )
            if not self._joint_reader.wait(timeout_sec=2.0):
                self.get_logger().warn(
                    'No %s received yet; point_target will stay camera-only until joints arrive.'
                    % self.lowstate_topic
                )
        except Exception as exc:
            self._fk_error = 'Failed to initialize Unitree lowstate reader: %s' % exc
            self.get_logger().warn(self._fk_error)

    def _ensure_xr_fk(self):
        if self._xr_ik is not None and self._xr_current_ee_poses is not None:
            return
        if self.base_link not in ('pelvis', 'torso', 'torso_link'):
            raise RuntimeError(
                'xr_pinocchio FK supports base_link=pelvis or torso_link, got %s'
                % self.base_link
            )
        candidates = [
            Path(os.path.expanduser(self.h2_xr_scripts_dir)) if self.h2_xr_scripts_dir else None,
            Path(os.path.expanduser(self.workspace_root)) / 'H2_joint_control' / 'H2_joint_cartesian' / 'scripts',
            Path('/home/unitree/H2_joint_cartesian/scripts'),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)

        module_path = None
        for candidate in candidates:
            if candidate is None:
                continue
            path = candidate / 'h2_xr_official_ik_demo.py'
            if path.is_file():
                module_path = path
                break
        if module_path is None:
            raise FileNotFoundError('Cannot locate h2_xr_official_ik_demo.py in configured candidates.')

        import importlib.util
        import inspect

        module_name = '_yolo_trt_ros2_h2_xr_runtime'
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise ImportError('Cannot load xr_pinocchio module from %s' % module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        current_ee_poses = module.current_ee_poses
        signature = inspect.signature(current_ee_poses)
        required_parameters = {'waist_q', 'reference_frame'}
        missing_parameters = required_parameters.difference(signature.parameters)
        if missing_parameters:
            raise RuntimeError(
                'Waist/torso-aware current_ee_poses is required; missing %s in signature %s from %s'
                % (sorted(missing_parameters), signature, module_path)
            )

        self._xr_ik = module.H2CompatibleIK()
        self._xr_current_ee_poses = current_ee_poses
        self.get_logger().info(
            'xr_pinocchio FK initialized: module=%s current_ee_poses%s'
            % (module_path, signature)
        )

    def _resolve_handeye_input(self, path):
        resolved = Path(os.path.expanduser(path))
        if resolved.suffix == '.json':
            npy_dir = resolved.with_name(resolved.stem + '_npy')
            if npy_dir.is_dir():
                return npy_dir
        return resolved

    def _load_matrix_from_candidates(self, root, candidates):
        root = self._resolve_handeye_input(root)
        if root.is_file():
            try:
                transform = np.load(str(root)).astype(np.float64)
            except Exception as exc:
                self.get_logger().warn('Failed to load transform npy %s: %s' % (root, exc))
                return None
            if transform.shape != (4, 4):
                self.get_logger().warn('transform must be 4x4, got shape=%s from %s' % (transform.shape, root))
                return None
            return transform, str(root)

        if root.is_dir():
            for name in candidates:
                candidate = root / name
                if candidate.is_file():
                    try:
                        transform = np.load(str(candidate)).astype(np.float64)
                    except Exception as exc:
                        self.get_logger().warn('Failed to load transform npy %s: %s' % (candidate, exc))
                        return None
                    if transform.shape != (4, 4):
                        self.get_logger().warn(
                            'transform must be 4x4, got shape=%s from %s' % (transform.shape, candidate)
                        )
                        return None
                    return transform, str(candidate)

        self.get_logger().warn('No supported transform found at: %s' % root)
        return None

    def _load_handeye_transform(self, path):
        if not path:
            return None

        loaded = self._load_matrix_from_candidates(
            path,
            ['T_cam2base.npy', 'T_cam2world.npy', 'T_camera2base.npy', 'T_camera2world.npy'],
        )
        if loaded is None:
            return None
        transform, source = loaded

        self.get_logger().info('Loaded hand-eye camera-to-target transform: %s' % source)
        return transform

    def _load_eye_in_hand_transform(self, path):
        loaded = self._load_matrix_from_candidates(
            path,
            ['T_cam2hand.npy', 'T_cam2ee.npy', 'T_camera2hand.npy', 'T_camera2ee.npy'],
        )
        if loaded is None:
            return None
        transform, source = loaded
        self.get_logger().info('Loaded eye-in-hand camera-to-hand transform: %s' % source)
        return transform

    def _output_frame_name(self):
        if self.T_cam2hand is not None:
            return self.handeye_target_frame or self.base_link or 'base_link'
        if self.handeye_transform is not None:
            return self.handeye_target_frame or self.target_frame or 'base_link'
        return self.target_frame or '<camera>'

    def _depth_callback(self, depth_msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error('Failed to convert depth image: %s' % exc)
            return

        self.ingest_depth(np.asarray(depth), depth_msg.header.stamp)

    def ingest_depth(self, depth, stamp):
        """Cache one in-memory aligned depth frame without ROS Image conversion."""
        depth_array = np.asarray(depth)
        stamp_ns = self._stamp_to_ns(stamp)
        self.latest_depth_stamp_ns = stamp_ns
        self.latest_depth = depth_array
        self.depth_cache.append((stamp_ns, depth_array))

    def _camera_info_callback(self, camera_info):
        self.ingest_camera_info(camera_info)

    def ingest_camera_info(self, camera_info):
        """Set color intrinsics supplied by the in-process camera owner."""
        self.latest_camera_info = camera_info

    def process_objects(self, objects_msg):
        """Project an in-memory Object2DArray using the synchronized depth cache."""
        self._objects_callback(objects_msg)

    def _robot_status_callback(self, msg):
        self.latest_robot_status = msg

    def _pixel_query_callback(self, msg):
        request_id = int(round(float(msg.point.z)))
        u = float(msg.point.x)
        v = float(msg.point.y)
        result = {
            'request_id': request_id,
            'valid': False,
            'pixel': [u, v],
            'depth_m': None,
            'depth_source': '',
            'point_camera': None,
            'point_world': None,
            'point_target': None,
            'source_frame': str(msg.header.frame_id),
            'target_frame': self._output_frame_name(),
            'message': '',
        }

        if self.latest_camera_info is None:
            result['message'] = 'missing CameraInfo'
            self._publish_pixel_query_result(result)
            return

        depth_match = self._select_depth_for_header(msg.header)
        if depth_match is None:
            result['message'] = 'no synchronized depth frame for selected image'
            self._publish_pixel_query_result(result)
            return

        self.latest_depth_stamp_ns, self.latest_depth = depth_match
        depth_result = None
        used_radius = self.depth_radius
        for radius in (self.depth_radius, max(self.depth_radius + 2, 6), max(self.depth_radius + 5, 10)):
            depth_result = self._sample_depth_m(
                u,
                v,
                max_depth_m=self.pixel_query_max_depth_m,
                radius=radius,
            )
            if depth_result is not None:
                used_radius = radius
                break
        if depth_result is None:
            result['message'] = 'no valid foreground depth near selected pixel'
            self._publish_pixel_query_result(result)
            return

        depth_m, valid_count = depth_result
        try:
            point_camera = self._deproject(u, v, depth_m)
        except RuntimeError as exc:
            result['message'] = str(exc)
            self._publish_pixel_query_result(result)
            return

        point_world = point_camera
        point_target = point_camera
        target_frame = str(msg.header.frame_id)
        transform_message = 'camera_frame'
        if self.T_cam2hand is not None:
            transformed = self._transform_eye_in_hand_with_pose(point_camera)
            if transformed is None:
                result['message'] = 'eye_in_hand_unavailable: %s' % (self._fk_error or 'missing joints/FK')
                self._publish_pixel_query_result(result)
                return
            point_world, T_base_hand = transformed
            point_target, target_message = self._action_target_for_detection(None, point_world, T_base_hand)
            target_frame = self.handeye_target_frame or self.base_link or 'base_link'
            transform_message = 'eye_in_hand_fk; ' + target_message
        elif self.handeye_transform is not None:
            point_world = self._apply_transform(self.handeye_transform, point_camera)
            joints = self._current_joint_values()
            if not joints:
                result['message'] = 'eye_to_hand target compensation unavailable: missing joints'
                self._publish_pixel_query_result(result)
                return
            try:
                T_base_hand = self._compute_base_to_hand(joints)
            except Exception as exc:
                result['message'] = 'eye_to_hand target compensation FK failed: %s' % exc
                self._publish_pixel_query_result(result)
                return
            point_target, target_message = self._action_target_for_detection(
                None,
                point_world,
                T_base_hand,
            )
            target_frame = self.handeye_target_frame or self.target_frame or 'base_link'
            transform_message = 'eye_to_hand; ' + target_message
        elif self.target_frame:
            transformed = self._transform_point(point_camera, msg.header.frame_id, self.target_frame)
            if transformed is None:
                result['message'] = 'TF transform unavailable'
                self._publish_pixel_query_result(result)
                return
            point_world = transformed
            point_target = point_world
            target_frame = self.target_frame
            transform_message = 'transformed'

        result.update(
            {
                'valid': True,
                'depth_m': float(depth_m),
                'depth_source': 'pixel_patch_r%d_percentile' % int(used_radius),
                'depth_valid_count': int(valid_count),
                'point_camera': [float(value) for value in point_camera],
                'point_world': [float(value) for value in point_world],
                'point_target': [float(value) for value in point_target],
                'target_frame': target_frame,
                'message': transform_message,
            }
        )
        self._publish_pixel_query_result(result)

    def _publish_pixel_query_result(self, result):
        msg = String()
        msg.data = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        self.pixel_query_result_pub.publish(msg)
        if result.get('valid'):
            self.get_logger().info(
                'PIXEL_3D pixel=(%.1f,%.1f) depth_m=%.4f source=%s '
                'point_world=%s point_target=%s'
                % (
                    float(result['pixel'][0]),
                    float(result['pixel'][1]),
                    float(result['depth_m']),
                    result['depth_source'],
                    self._format_xyz(result['point_world']),
                    self._format_xyz(result['point_target']),
                )
            )
        else:
            self.get_logger().warn(
                'PIXEL_3D failed pixel=(%.1f,%.1f): %s'
                % (
                    float(result['pixel'][0]),
                    float(result['pixel'][1]),
                    result.get('message', 'unknown error'),
                )
            )

    def _objects_callback(self, objects_msg):
        out_msg = Object3DArray()
        out_msg.header = objects_msg.header

        if self.latest_depth is None or self.latest_camera_info is None:
            out_msg.objects = [self._invalid_object(obj, objects_msg.header, 'missing depth or CameraInfo')
                               for obj in objects_msg.objects if self.publish_invalid]
            self.objects_3d_pub.publish(out_msg)
            return

        depth_match = self._select_depth_for_header(objects_msg.header)
        if depth_match is None:
            now = time.monotonic()
            if now - self._last_depth_sync_warning_sec >= 1.0:
                self._last_depth_sync_warning_sec = now
                self.get_logger().warn(
                    'No synchronized depth for detector frame; refusing stale 3D target '
                    '(tolerance=%.3fs, cache=%d).'
                    % (self.depth_sync_tolerance_sec, len(self.depth_cache))
                )
            out_msg.objects = [
                self._invalid_object(obj, objects_msg.header, 'no synchronized depth frame')
                for obj in objects_msg.objects
                if self.publish_invalid
            ]
            self.objects_3d_pub.publish(out_msg)
            return

        self.latest_depth_stamp_ns, self.latest_depth = depth_match
        self._frame_handle_reference = self._best_handle_reference(objects_msg.objects)

        # Keep all projected objects for perception/grasp diagnostics.
        # Default action target remains the circular push point; hang-tag
        # stages override that with the grasp point or cabinet hook.
        best_blue_obj = None
        best_grasp_obj = None
        best_hook_obj = None
        for obj in objects_msg.objects:
            if not self._passes_filter(obj):
                continue

            self._apply_ground_ratio_to_red_sticker(obj)
            object_3d = self._project_object(obj, objects_msg.header)
            if object_3d.valid or self.publish_invalid:
                out_msg.objects.append(object_3d)
            if not object_3d.valid:
                continue
            if self._is_blue_point_detection(obj) and (
                best_blue_obj is None
                or obj.confidence > best_blue_obj.detection.confidence
            ):
                best_blue_obj = object_3d
            if is_work_tag_grasp_class(obj.class_name) and (
                best_grasp_obj is None
                or obj.confidence > best_grasp_obj.detection.confidence
            ):
                best_grasp_obj = object_3d
            if is_hang_hook_class(obj.class_name) and (
                best_hook_obj is None
                or obj.confidence > best_hook_obj.detection.confidence
            ):
                best_hook_obj = object_3d

        self._apply_3d_control_row_semantics(out_msg.objects)
        self._log_3d_diagnostics_if_changed(out_msg.objects)
        self.objects_3d_pub.publish(out_msg)
        joints = self._current_joint_values()
        if joints:
            self._publish_current_joint_state(joints, objects_msg.header)
        self._publish_objects_ik_json(out_msg, objects_msg.header, joints)

        target_obj = self._select_action_target(best_blue_obj, best_grasp_obj, best_hook_obj)
        if target_obj is not None:
            self._publish_target(target_obj)

    def _select_action_target(self, best_blue_obj, best_grasp_obj, best_hook_obj):
        status = self._robot_status_payload(getattr(self, 'latest_robot_status', None))
        stage_id = int(status.get('stage_id', 0) or 0)
        stage_name = str(status.get('stage_name', '') or '')
        if stage_prefers_work_tag_grasp(stage_id, stage_name) and best_grasp_obj is not None:
            return best_grasp_obj
        if stage_prefers_hang_hook(stage_id, stage_name) and best_hook_obj is not None:
            return best_hook_obj
        return best_blue_obj

    def _apply_3d_control_row_semantics(self, objects):
        button_z = [
            float(obj.point_target.z)
            for obj in objects
            if obj.valid
            and str(obj.detection.class_name).lower()
            in ('red push button', 'green push button')
        ]
        if not button_z:
            return
        reference_z = float(np.median(button_z))
        tolerance = max(0.005, self.control_row_z_tolerance_m)
        for obj in objects:
            if not obj.valid:
                continue
            detection = obj.detection
            if 'rotary selector switch' not in str(detection.class_name).lower():
                continue
            delta_z = float(obj.point_target.z) - reference_z
            detection.spatial_relation = (
                'top_row' if abs(delta_z) <= tolerance else 'middle_row'
            )
            parts = [str(detection.class_name)]
            parts.append(str(detection.spatial_relation))
            label = str(getattr(detection, 'label_text', '')).strip()
            if label:
                parts.append(label)
            elif bool(getattr(detection, 'label_tag_present', False)):
                parts.append('with_tag')
            detection.semantic_name = '/'.join(parts)
            suffix = '/'.join(parts[1:])
            class_id = re.sub(
                r'[^a-z0-9]+',
                '_',
                str(detection.class_name).lower(),
            ).strip('_')
            detection.control_id = '%s/%s' % (class_id, suffix)

    def _passes_filter(self, obj):
        if obj.confidence < self.min_confidence:
            return False
        if self.class_filter and obj.class_name not in self.class_filter:
            return False
        return True
    def _publish_pixel_query_result(self, result):
        msg = String()
        msg.data = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        self.pixel_query_result_pub.publish(msg)
        if result.get('valid'):
            self.get_logger().info(
                'PIXEL_3D pixel=(%.1f,%.1f) depth_m=%.4f source=%s '
                'point_world=%s point_target=%s'
                % (
                    float(result['pixel'][0]),
                    float(result['pixel'][1]),
                    float(result['depth_m']),
                    result['depth_source'],
                    self._format_xyz(result['point_world']),
                    self._format_xyz(result['point_target']),
                )
            )
        else:
            self.get_logger().warn(
                'PIXEL_3D failed pixel=(%.1f,%.1f): %s'
                % (
                    float(result['pixel'][0]),
                    float(result['pixel'][1]),
                    result.get('message', 'unknown error'),
                )
            )

    def _log_3d_diagnostics_if_changed(self, objects):
        if not self.diagnostic_3d_log_enabled:
            return

        change_m = self.diagnostic_3d_log_change_m
        entries = []
        for obj in objects:
            if not obj.valid:
                continue
            detection = obj.detection
            depth_source = 'unknown'
            for part in str(obj.message).split(';'):
                part = part.strip()
                if part.startswith('depth_source='):
                    depth_source = part.split('=', 1)[1].strip() or 'unknown'
                    break
            camera = (
                float(obj.point_camera.x),
                float(obj.point_camera.y),
                float(obj.point_camera.z),
            )
            target = (
                float(obj.point_target.x),
                float(obj.point_target.y),
                float(obj.point_target.z),
            )
            identity = (
                str(detection.class_name),
                int(round(float(detection.cx) / 8.0)),
                int(round(float(detection.cy) / 8.0)),
            )
            quantized = tuple(
                int(round(value / change_m))
                for value in (float(obj.depth_m),) + camera + target
            )
            entries.append((identity, depth_source, quantized, obj))

        entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        snapshot = tuple(
            (identity, depth_source, quantized)
            for identity, depth_source, quantized, _ in entries
        )
        if snapshot == self._last_3d_diagnostic_snapshot:
            return

        now = time.monotonic()
        if now - self._last_3d_diagnostic_log_sec < self.diagnostic_3d_log_interval_sec:
            return

        waist_q = np.asarray(getattr(self, '_last_waist_q', np.zeros(3)), dtype=np.float64)
        lines = [
            '3D_DIAG changed objects=%d waist_yaw_roll_pitch_rad=[%.5f %.5f %.5f]'
            % (len(entries), float(waist_q[0]), float(waist_q[1]), float(waist_q[2]))
        ]
        for identity, depth_source, _, obj in entries:
            camera = obj.point_camera
            target = obj.point_target
            lines.append(
                'class=%s center_px=(%.1f,%.1f) depth_m=%.4f depth_source=%s '
                'point_camera=[%.4f %.4f %.4f] point_target=[%.4f %.4f %.4f]'
                % (
                    identity[0],
                    float(obj.detection.cx),
                    float(obj.detection.cy),
                    float(obj.depth_m),
                    depth_source,
                    float(camera.x),
                    float(camera.y),
                    float(camera.z),
                    float(target.x),
                    float(target.y),
                    float(target.z),
                )
            )
        self.get_logger().info('\n'.join(lines))
        self._last_3d_diagnostic_snapshot = snapshot
        self._last_3d_diagnostic_log_sec = now

    def _depth_is_stale(self, objects_header):
        if self.stale_depth_sec <= 0.0 or self.latest_depth_stamp_ns <= 0:
            return False
        object_ns = self._stamp_to_ns(objects_header.stamp)
        if object_ns <= 0:
            return False
        return abs(object_ns - self.latest_depth_stamp_ns) * 1e-9 > self.stale_depth_sec

    def _select_depth_for_header(self, objects_header):
        if not self.depth_cache:
            return None
        object_ns = self._stamp_to_ns(objects_header.stamp)
        if object_ns <= 0:
            return self.depth_cache[-1]
        best = min(self.depth_cache, key=lambda item: abs(item[0] - object_ns))
        delta_sec = abs(best[0] - object_ns) * 1e-9
        if delta_sec > max(0.0, self.depth_sync_tolerance_sec):
            return None
        return best

    def _stamp_to_ns(self, stamp):
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)

    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _project_object(self, obj, header):
        depth_result = self._sample_object_depth_m(obj)
        if depth_result is None:
            return self._invalid_object(obj, header, 'no valid depth near center or bbox roi')
        depth_m, depth_source, depth_valid_count = depth_result

        try:
            point_camera = self._deproject(float(obj.cx), float(obj.cy), depth_m)
        except RuntimeError as exc:
            return self._invalid_object(obj, header, str(exc))
        point_object = point_camera
        point_target = point_camera
        target_frame = header.frame_id
        message = 'camera_frame'

        if self.T_cam2hand is not None:
            transformed = self._transform_eye_in_hand_with_pose(point_camera)
            if transformed is None:
                return self._invalid_object(
                    obj,
                    header,
                    'eye_in_hand_unavailable: %s' % (self._fk_error or 'missing joints/FK'),
                )
            else:
                point_object, T_base_hand = transformed
                point_target, target_message = self._action_target_for_detection(obj, point_object, T_base_hand)
                target_frame = self.handeye_target_frame or self.base_link or 'base_link'
                message = 'eye_in_hand_fk; ' + target_message
        elif self.handeye_transform is not None:
            point_object = self._apply_transform(self.handeye_transform, point_camera)
            joints = self._current_joint_values()
            if not joints:
                return self._invalid_object(
                    obj,
                    header,
                    'eye_to_hand target compensation unavailable: missing joints',
                )
            try:
                T_base_hand = self._compute_base_to_hand(joints)
            except Exception as exc:
                return self._invalid_object(
                    obj,
                    header,
                    'eye_to_hand target compensation FK failed: %s' % exc,
                )
            point_target, target_message = self._action_target_for_detection(
                obj,
                point_object,
                T_base_hand,
            )
            target_frame = self.handeye_target_frame or self.target_frame or 'base_link'
            message = 'eye_to_hand; ' + target_message
        elif self.target_frame:
            transformed = self._transform_point(point_camera, header.frame_id, self.target_frame)
            if transformed is None:
                return self._invalid_object(obj, header, 'TF transform unavailable')
            point_object = transformed
            point_target = point_object
            target_frame = self.target_frame
            message = 'transformed'

        msg = Object3D()
        msg.header = header
        msg.detection = obj
        msg.valid = True
        msg.cached = False
        msg.source_frame = header.frame_id
        msg.target_frame = target_frame
        msg.depth_m = float(depth_m)
        msg.point_camera = self._point_to_msg(point_camera)
        msg.point_target = self._point_to_msg(point_target)
        msg.message = '%s; depth_source=%s; depth_valid_count=%d' % (
            message,
            depth_source,
            int(depth_valid_count),
        )
        return msg

    def _invalid_object(self, obj, header, message):
        msg = Object3D()
        msg.header = header
        msg.detection = obj
        msg.valid = False
        msg.cached = False
        msg.source_frame = header.frame_id
        if self.T_cam2hand is not None:
            msg.target_frame = self._output_frame_name()
        elif self.handeye_transform is not None:
            msg.target_frame = self._output_frame_name()
        else:
            msg.target_frame = self.target_frame or header.frame_id
        msg.depth_m = 0.0
        msg.point_camera = Point()
        msg.point_target = Point()
        msg.message = message
        return msg

    def _sample_object_depth_m(self, obj):
        # A blue press point is a near-field eye-in-hand target. If its own
        # depth is missing, never substitute a multi-meter background surface.
        is_press = self._is_blue_point_detection(obj)
        max_depth_m = min(self.max_depth_m, self.foreground_max_depth_m)
        if self._is_handle_detection(obj):
            max_depth_m = min(max_depth_m, self.handle_max_depth_m)
        if is_press:
            max_depth_m = min(max_depth_m, self.press_max_depth_m)

        center = self._sample_depth_m(obj.cx, obj.cy, max_depth_m=max_depth_m)
        if center is not None:
            return center[0], 'center_percentile', center[1]

        if self.depth_fallback not in ('bbox', 'roi'):
            return None

        bbox = (float(obj.xmin), float(obj.ymin), float(obj.xmax), float(obj.ymax))
        roi = self._sample_bbox_depth_m(
            float(obj.cx),
            float(obj.cy),
            bbox,
            self.depth_roi_scale,
            max_depth_m=max_depth_m,
        )
        if roi is not None:
            return roi[0], 'bbox_roi_percentile', roi[1]

        if is_press:
            if self._frame_handle_reference is not None:
                handle = self._frame_handle_reference
                handle_bbox = (
                    float(handle.xmin),
                    float(handle.ymin),
                    float(handle.xmax),
                    float(handle.ymax),
                )
                # Never fit a press-point plane from a YOLO handle that has no
                # foreground depth support of its own. This prevents a distant
                # false handle from pulling near random pixels into the fit.
                sampled = self._sample_depth_m(
                    float(handle.cx),
                    float(handle.cy),
                    max_depth_m=max_depth_m,
                )
                if sampled is None:
                    sampled = self._sample_bbox_depth_m(
                        float(handle.cx),
                        float(handle.cy),
                        handle_bbox,
                        self.depth_roi_scale,
                        max_depth_m=max_depth_m,
                    )
                if sampled is not None:
                    fitted = self._fit_handle_plane_depth(obj, handle)
                    if fitted is not None:
                        return fitted[0], 'handle_plane_fit', fitted[1]
                    return sampled[0], 'handle_supported_percentile', sampled[1]

            # Never expand a press-point ROI into unrelated background. A
            # missing foreground depth must suppress the action target.
            return None

        expanded = self._sample_bbox_depth_m(
            float(obj.cx),
            float(obj.cy),
            bbox,
            self.depth_expanded_roi_scale,
            max_depth_m=max_depth_m,
        )
        if expanded is not None:
            return expanded[0], 'bbox_expanded_percentile', expanded[1]
        return None

    def _best_handle_reference(self, objects):
        handles = [obj for obj in objects if self._is_handle_detection(obj)]
        if not handles:
            return None
        return max(handles, key=lambda obj: float(obj.confidence))

    def _fit_handle_plane_depth(self, target, handle):
        """Fit inverse-depth plane over handle→button ROI and evaluate at target."""
        depth = self.latest_depth
        if depth is None or depth.size == 0:
            return None
        depth_h, depth_w = depth.shape[:2]

        handle_w = max(4.0, abs(float(handle.xmax) - float(handle.xmin)))
        handle_h = max(4.0, abs(float(handle.ymax) - float(handle.ymin)))
        pad_x = max(6.0, handle_w * 0.45)
        pad_y = max(6.0, handle_h * 0.12)
        left = min(float(handle.xmin), float(handle.xmax), float(target.cx)) - pad_x
        right = max(float(handle.xmin), float(handle.xmax), float(target.cx)) + pad_x
        top = min(float(handle.ymin), float(handle.ymax), float(target.cy)) - pad_y
        bottom = max(float(handle.ymin), float(handle.ymax), float(target.cy)) + pad_y
        x0, y0 = self._map_pixel_to_depth_image(left, top, depth_w, depth_h)
        x1, y1 = self._map_pixel_to_depth_image(right, bottom, depth_w, depth_h)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x1 = min(depth_w, x1 + 1)
        y1 = min(depth_h, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        roi_m = self._depth_patch_to_meters(np.asarray(depth[y0:y1, x0:x1]))
        upper = min(self.max_depth_m, self.press_max_depth_m)
        valid_mask = (
            np.isfinite(roi_m)
            & (roi_m >= self.min_depth_m)
            & (roi_m <= upper)
        )
        ys, xs = np.nonzero(valid_mask)
        values = roi_m[ys, xs]
        min_valid = max(20, self.depth_min_valid_count)
        if values.size < min_valid:
            return None

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        band = max(0.025, 3.5 * 1.4826 * mad)
        keep = np.abs(values - median) <= band
        xs = xs[keep].astype(np.float64) + float(x0)
        ys = ys[keep].astype(np.float64) + float(y0)
        values = values[keep]
        if values.size < min_valid:
            return None

        target_x, target_y = self._map_pixel_to_depth_image(
            float(target.cx), float(target.cy), depth_w, depth_h
        )
        sx = max(1.0, float(x1 - x0))
        sy = max(1.0, float(y1 - y0))
        xn = (xs - float(target_x)) / sx
        yn = (ys - float(target_y)) / sy
        design = np.column_stack((xn, yn, np.ones_like(xn)))
        inv_depth = 1.0 / values
        inliers = np.ones(values.size, dtype=bool)
        coeff = None
        for _ in range(3):
            if int(np.count_nonzero(inliers)) < min_valid:
                return None
            coeff, _, _, _ = np.linalg.lstsq(
                design[inliers], inv_depth[inliers], rcond=None
            )
            residual = inv_depth - design @ coeff
            residual_mad = float(np.median(np.abs(residual[inliers])))
            threshold = max(1e-6, 3.5 * 1.4826 * residual_mad)
            inliers = np.abs(residual) <= threshold

        if coeff is None or float(coeff[2]) <= 0.0:
            return None
        target_depth = 1.0 / float(coeff[2])
        if not self.min_depth_m < target_depth < upper:
            return None
        return float(target_depth), int(np.count_nonzero(inliers))

    def _sample_depth_m(self, u, v, max_depth_m=None, radius=None):
        depth = self.latest_depth
        if depth is None or depth.size == 0:
            return None

        h, w = depth.shape[:2]
        u_depth, v_depth = self._map_pixel_to_depth_image(u, v, w, h)
        sample_radius = self.depth_radius if radius is None else max(0, int(radius))
        x0 = max(0, u_depth - sample_radius)
        x1 = min(w, u_depth + sample_radius + 1)
        y0 = max(0, v_depth - sample_radius)
        y1 = min(h, v_depth + sample_radius + 1)
        patch = np.asarray(depth[y0:y1, x0:x1])
        if patch.size == 0:
            return None

        return self._valid_depth_percentile(patch, max_depth_m=max_depth_m)

    def _sample_bbox_depth_m(self, cx, cy, bbox_xyxy, scale, max_depth_m=None):
        depth = self.latest_depth
        if depth is None or depth.size == 0:
            return None
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        scale = max(0.01, float(scale))
        half_w = max(float(self.depth_radius + 1), 0.5 * max(1.0, x2 - x1) * scale)
        half_h = max(float(self.depth_radius + 1), 0.5 * max(1.0, y2 - y1) * scale)

        left = cx - half_w
        right = cx + half_w
        top = cy - half_h
        bottom = cy + half_h
        dx0, dy0 = self._map_pixel_to_depth_image(left, top, w, h)
        dx1, dy1 = self._map_pixel_to_depth_image(right, bottom, w, h)
        x0, x1 = sorted((dx0, dx1))
        y0, y1 = sorted((dy0, dy1))
        x1 = min(w, x1 + 1)
        y1 = min(h, y1 + 1)
        patch = np.asarray(depth[y0:y1, x0:x1])
        if patch.size == 0:
            return None
        return self._valid_depth_percentile(patch, max_depth_m=max_depth_m)

    def _valid_depth_percentile(self, patch, max_depth_m=None):
        depth_m = self._depth_patch_to_meters(patch)
        upper = self.max_depth_m if max_depth_m is None else min(self.max_depth_m, float(max_depth_m))
        valid = depth_m[
            np.isfinite(depth_m)
            & (depth_m >= self.min_depth_m)
            & (depth_m <= upper)
        ]
        if valid.size < max(1, self.depth_min_valid_count):
            return None
        percentile = max(0.0, min(100.0, self.depth_percentile))
        return float(np.percentile(valid, percentile)), int(valid.size)

    def _map_pixel_to_depth_image(self, u, v, depth_w, depth_h):
        info = self.latest_camera_info
        image_w = int(info.width) if info is not None and info.width else depth_w
        image_h = int(info.height) if info is not None and info.height else depth_h
        if image_w <= 0 or image_h <= 0:
            image_w, image_h = depth_w, depth_h
        x = int(round(float(u) * float(depth_w) / float(image_w)))
        y = int(round(float(v) * float(depth_h) / float(image_h)))
        return max(0, min(depth_w - 1, x)), max(0, min(depth_h - 1, y))

    def _depth_patch_to_meters(self, patch):
        if np.issubdtype(patch.dtype, np.integer):
            return patch.astype(np.float64) * self.depth_scale
        return patch.astype(np.float64)

    def _deproject(self, u, v, depth_m):
        k = self.latest_camera_info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError('CameraInfo contains invalid focal length.')
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        return np.array([x, y, depth_m], dtype=np.float64)

    def _transform_point(self, point, source_frame, target_frame):
        if self.tf_buffer is None:
            return None
        if not source_frame:
            self.get_logger().warn('Object header has empty frame_id; cannot transform.')
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.transform_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn('TF lookup failed: %s' % exc)
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        rotated = self._rotate_vector(point, [q.x, q.y, q.z, q.w])
        return rotated + np.array([t.x, t.y, t.z], dtype=np.float64)

    def _apply_transform(self, transform, point):
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = np.asarray(point, dtype=np.float64).reshape(3)
        return (transform @ point_h)[:3]

    def _transform_eye_in_hand(self, point_camera):
        transformed = self._transform_eye_in_hand_with_pose(point_camera)
        if transformed is None:
            return None
        return transformed[0]

    def _transform_eye_in_hand_with_pose(self, point_camera):
        if self.T_cam2hand is None:
            return None
        joints = self._current_joint_values()
        if not joints:
            self._fk_error = 'missing joint values'
            return None
        try:
            T_base_hand = self._compute_base_to_hand(joints)
        except Exception as exc:
            self._fk_error = 'FK failed: %s' % exc
            now = time.monotonic()
            if now - self._last_fk_warning_sec >= 1.0:
                self._last_fk_warning_sec = now
                self.get_logger().warn(self._fk_error)
            return None
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = np.asarray(point_camera, dtype=np.float64).reshape(3)
        point_hand = self.T_cam2hand @ point_h
        point_target = T_base_hand @ point_hand
        return point_target[:3], T_base_hand

    def _action_target_for_detection(self, detection, point_object, T_base_hand=None):
        contact = np.asarray(point_object, dtype=np.float64).reshape(3)
        messages = []
        if self._is_blue_point_detection(detection):
            offset = np.asarray(self.blue_point_target_world_offset_xyz, dtype=np.float64).reshape(3)
            if np.linalg.norm(offset) > 0.0:
                contact = contact + offset
                messages.append('blue_point_world_offset=%s' % self._format_xyz(offset))

        tip_from_wrist = np.asarray(self.dex1_tip_from_wrist_xyz, dtype=np.float64).reshape(3)
        mount_offset = np.asarray(self.handeye_mount_offset_from_wrist_xyz, dtype=np.float64).reshape(3)
        tip_from_mount = tip_from_wrist - mount_offset
        target = contact
        if not self.apply_tip_compensation:
            messages.append('dex1_tip_compensation=disabled')
        elif T_base_hand is not None:
            rotation = np.asarray(T_base_hand, dtype=np.float64)[:3, :3]
            target = contact - rotation @ tip_from_mount
            messages.append('direct_copy_target')
        else:
            messages.append('contact_target_no_fk_rotation')
        messages.append('contact=%s' % self._format_xyz(contact))
        messages.append('dex1_tip_from_wrist=%s' % self._format_xyz(tip_from_wrist))
        messages.append('mount_offset_from_wrist=%s' % self._format_xyz(mount_offset))
        return target, '; '.join(messages)

    def _apply_ground_ratio_to_red_sticker(self, detection):
        name = str(getattr(detection, 'class_name', '')).lower()
        if 'red sticker push point' not in name and 'lock point' not in name:
            return
        if self.T_cam2hand is None or self.latest_camera_info is None:
            return
        joints = self._current_joint_values()
        if not joints:
            return
        try:
            T_base_hand = np.asarray(self._compute_base_to_hand(joints), dtype=np.float64)
            T_base_camera = T_base_hand @ np.asarray(self.T_cam2hand, dtype=np.float64)
        except Exception as exc:
            self._fk_error = 'ground-direction FK failed: %s' % exc
            return

        k = self.latest_camera_info.k
        fx, fy = float(k[0]), float(k[4])
        ppx, ppy = float(k[2]), float(k[5])
        if fx == 0.0 or fy == 0.0:
            return

        center_x = 0.5 * (float(detection.xmin) + float(detection.xmax))
        center_y = 0.5 * (float(detection.ymin) + float(detection.ymax))
        ray_x = (center_x - ppx) / fx
        ray_y = (center_y - ppy) / fy
        ground_camera = T_base_camera[:3, :3].T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        image_direction = np.array(
            [
                fx * (ground_camera[0] - ray_x * ground_camera[2]),
                fy * (ground_camera[1] - ray_y * ground_camera[2]),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(image_direction))
        if norm < 1e-9:
            return
        direction = image_direction / norm

        def distance_to_edge(dx, dy):
            distances = []
            if dx > 1e-9:
                distances.append((float(detection.xmax) - center_x) / dx)
            elif dx < -1e-9:
                distances.append((float(detection.xmin) - center_x) / dx)
            if dy > 1e-9:
                distances.append((float(detection.ymax) - center_y) / dy)
            elif dy < -1e-9:
                distances.append((float(detection.ymin) - center_y) / dy)
            positive = [value for value in distances if value >= 0.0]
            return min(positive) if positive else 0.0

        distance_ground = distance_to_edge(direction[0], direction[1])
        distance_up = distance_to_edge(-direction[0], -direction[1])
        ratio = self.press_point_ground_ratio
        start = np.array([center_x, center_y], dtype=np.float64) - direction * distance_up
        target = start + direction * ratio * (distance_up + distance_ground)
        detection.cx = float(np.clip(target[0], float(detection.xmin), float(detection.xmax)))
        detection.cy = float(np.clip(target[1], float(detection.ymin), float(detection.ymax)))

    def _is_blue_point_detection(self, detection):
        name = str(getattr(detection, 'class_name', '')).lower()
        return (
            'blue' in name
            or 'circle push point' in name
            or 'white square push point' in name
            or 'red sticker push point' in name
            or 'lock point' in name
        )

    def _is_handle_detection(self, detection):
        name = str(getattr(detection, 'class_name', '')).lower()
        return any(token in name for token in ('handle', 'lever', 'pull', 'cabinet door'))

    def _format_xyz(self, values):
        return '[%.4f, %.4f, %.4f]' % tuple(float(v) for v in np.asarray(values, dtype=np.float64).reshape(3))

    def _current_joint_values(self):
        if self.joint_json_path:
            try:
                import json
                with open(os.path.expanduser(self.joint_json_path), 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and 'joint_values' in payload:
                    payload = payload['joint_values']
                if isinstance(payload, dict) and 'joints' in payload:
                    payload = payload['joints']
                return {str(k): float(v) for k, v in payload.items()}
            except Exception as exc:
                self._fk_error = 'failed to load joint_json_path: %s' % exc
                return {}
        if self._joint_reader is None:
            return {}
        return self._joint_reader.joint_positions_by_name()

    def _compute_base_to_hand(self, joint_values):
        backend = self.fk_backend_requested or 'auto'
        if backend in ('auto', 'xr_pinocchio'):
            try:
                return self._compute_base_to_hand_xr(joint_values)
            except Exception as exc:
                self._fk_error = 'xr_pinocchio failed: %s' % exc
                if backend == 'xr_pinocchio':
                    raise
        if backend in ('auto', 'urdf'):
            return self._compute_base_to_hand_urdf(joint_values)
        raise RuntimeError('Unknown FK backend: %s' % backend)

    def _compute_base_to_hand_xr(self, joint_values):
        self._ensure_xr_fk()
        q = np.asarray(
            [float(joint_values.get(name, 0.0)) for name in H2_XR_DUAL_ARM_JOINTS],
            dtype=np.float64,
        )
        waist_q = np.asarray(
            [
                0.0 if self.lock_waist else float(joint_values.get(name, 0.0))
                for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
            ],
            dtype=np.float64,
        )
        self._last_waist_q = waist_q.copy()
        left_pose, right_pose = self._xr_current_ee_poses(
            self._xr_ik,
            q,
            waist_q,
            reference_frame=self.base_link,
        )
        pose = left_pose if self.hand_link.startswith('left_') else right_pose
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(pose.rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        if self.hand_link.endswith('wrist_yaw_link'):
            # Kept configurable for compatibility with older H2 IK scripts.
            # Current H2CompatibleIK places its control frame at wrist_yaw_link,
            # so the configured offset is zero.
            ee_offset = np.asarray(
                self.xr_ee_offset_from_wrist_xyz,
                dtype=np.float64,
            ).reshape(3)
            transform[:3, 3] = (
                transform[:3, 3] - transform[:3, :3] @ ee_offset
            )
        self.fk_backend_active = 'xr_pinocchio'
        self._fk_error = ''
        return transform

    def _compute_base_to_hand_urdf(self, joint_values):
        if self._fk_model is None:
            raise RuntimeError('URDF FK model is not initialized')
        targets = [self.hand_link]
        poses = self._fk_model.compute_link_poses(
            joint_values=self._lock_joint_values(joint_values),
            targets=targets,
            base_link=self.base_link or self._default_base_link(),
            base_pose=np.eye(4, dtype=np.float64),
            clamp_to_limits=False,
        )
        transform = np.asarray(poses[self.hand_link].matrix, dtype=np.float64)
        offset = np.asarray(self.handeye_mount_offset_from_wrist_xyz, dtype=np.float64).reshape(3)
        if np.linalg.norm(offset) > 0.0:
            transform = transform.copy()
            transform[:3, 3] = transform[:3, 3] + transform[:3, :3] @ offset
        self.fk_backend_active = 'urdf'
        self._fk_error = ''
        return transform

    def _lock_joint_values(self, joint_values):
        out = dict(joint_values)
        if self.lock_waist:
            for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'):
                out[name] = 0.0
        return out

    def _rotate_vector(self, vector, quat_xyzw):
        x, y, z, w = [float(v) for v in quat_xyzw]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return np.asarray(vector, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        qvec = np.array([x, y, z], dtype=np.float64)
        vec = np.asarray(vector, dtype=np.float64)
        uv = np.cross(qvec, vec)
        uuv = np.cross(qvec, uv)
        return vec + 2.0 * (w * uv + uuv)

    def _publish_current_joint_state(self, joints, header):
        if not self.publish_current_joint_state:
            return
        msg = JointState()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.base_link or self._default_base_link()
        names = sorted(joints.keys())
        msg.name = names
        msg.position = [float(joints[name]) for name in names]
        self.current_joint_state_pub.publish(msg)

    def _current_joint_timer_callback(self):
        if not self.publish_current_joint_state:
            return
        joints = self._current_joint_values()
        if not joints:
            return
        stamp = self.get_clock().now().to_msg()
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_link or self._default_base_link()
        names = sorted(joints.keys())
        msg.name = names
        msg.position = [float(joints[name]) for name in names]
        self.current_joint_state_pub.publish(msg)

        try:
            transform = np.asarray(self._compute_base_to_hand(joints), dtype=np.float64)
        except Exception as exc:
            self._fk_error = 'current EE FK failed: %s' % exc
            return
        point_msg = PointStamped()
        point_msg.header.stamp = stamp
        point_msg.header.frame_id = self.base_link or self._default_base_link()
        point_msg.point = self._point_to_msg(transform[:3, 3])
        self.current_ee_point_pub.publish(point_msg)

    def _publish_objects_ik_json(self, objects_3d_msg, header, joints):
        if not self.publish_objects_ik_json:
            return

        payload = {
            'header': {
                'stamp': {
                    'sec': int(header.stamp.sec),
                    'nanosec': int(header.stamp.nanosec),
                },
                'frame_id': str(header.frame_id),
            },
            'base_link': self.base_link or self._default_base_link(),
            'target_link': self.ik_target_link or self.hand_link or self._default_hand_link(),
            'current_joint_values_rad': {str(k): float(v) for k, v in (joints or {}).items()},
            'robot_status': self._robot_status_payload(self.latest_robot_status),
            'objects': [],
            'message': 'ok',
        }

        if not joints:
            payload['message'] = 'missing current joint values'
            self._publish_json_string(self.objects_ik_pub, payload)
            return
        if self._ik_solver is None:
            payload['message'] = 'IK solver unavailable: %s' % (self._ik_error or 'not initialized')
            self._publish_json_string(self.objects_ik_pub, payload)
            return

        solved = 0
        for index, object_3d in enumerate(objects_3d_msg.objects):
            item = {
                'object_id': self._object_id(index, object_3d.detection.class_name),
                'class_name': str(object_3d.detection.class_name),
                'label_text': str(getattr(object_3d.detection, 'label_text', '')),
                'label_confidence': float(getattr(object_3d.detection, 'label_confidence', 0.0)),
                'semantic_name': str(getattr(object_3d.detection, 'semantic_name', '')),
                'control_id': str(getattr(object_3d.detection, 'control_id', '')),
                'spatial_relation': str(getattr(object_3d.detection, 'spatial_relation', '')),
                'label_tag_present': bool(getattr(object_3d.detection, 'label_tag_present', False)),
                'class_id': int(object_3d.detection.class_id),
                'confidence': float(object_3d.detection.confidence),
                'bbox_xyxy': [
                    int(object_3d.detection.xmin),
                    int(object_3d.detection.ymin),
                    int(object_3d.detection.xmax),
                    int(object_3d.detection.ymax),
                ],
                'center_px': [float(object_3d.detection.cx), float(object_3d.detection.cy)],
                'valid_3d': bool(object_3d.valid),
                'target_frame': str(object_3d.target_frame),
                'point_target': self._point_payload(object_3d.point_target),
                'ik': None,
            }
            item.update(self._handle_grasp_payload(object_3d.detection, joints))
            if not object_3d.valid:
                item['ik'] = {'success': False, 'message': object_3d.message or 'invalid 3D object'}
                payload['objects'].append(item)
                continue
            if solved >= max(0, self.max_ik_objects):
                item['ik'] = {'success': False, 'message': 'skipped: max_ik_objects reached'}
                payload['objects'].append(item)
                continue
            try:
                solution = self._solve_object_ik(object_3d, joints)
                item['ik'] = {
                    'success': bool(solution.success),
                    'message': str(solution.message),
                    'iterations': int(solution.iterations),
                    'position_error_m': float(solution.final_position_error_norm),
                    'orientation_error_rad': float(solution.final_orientation_error_norm),
                    'active_joints': [str(name) for name in solution.active_joints],
                    'joint_values_rad': {
                        str(name): float(solution.joint_values[name])
                        for name in solution.active_joints
                    },
                }
                solved += 1
            except Exception as exc:
                item['ik'] = {'success': False, 'message': 'IK failed: %s' % exc}
            payload['objects'].append(item)

        self._publish_json_string(self.objects_ik_pub, payload)

    def _handle_grasp_payload(self, detection, joints):
        edge_px = [float(v) for v in getattr(detection, 'handle_grasp_edge_px', [])]
        center_px = [float(v) for v in getattr(detection, 'handle_grasp_center_px', [])]
        payload = {}
        if len(edge_px) != 4 and self.handle_depth_grasp_fallback and self._is_blue_point_detection(detection):
            fallback = self._estimate_depth_handle_grasp_edge_px(detection)
            if fallback:
                edge_px = [float(v) for point in fallback.get('handle_grasp_edge_px', []) for v in point]
                center_px = [float(v) for v in fallback.get('handle_grasp_center_px', [])]
                payload.update(fallback)
        if len(edge_px) == 4:
            payload['handle_grasp_edge_px'] = [
                [edge_px[0], edge_px[1]],
                [edge_px[2], edge_px[3]],
            ]
        if len(center_px) == 2:
            payload['handle_grasp_center_px'] = [center_px[0], center_px[1]]
        mid_px = [float(v) for v in payload.get('handle_mid_px', [])]
        if len(mid_px) != 2 and len(center_px) == 2:
            mid_px = [center_px[0], center_px[1]]
        width_px = float(getattr(detection, 'handle_grasp_width_px', 0.0))
        if width_px > 0.0:
            payload['handle_grasp_width_px'] = width_px
        source = str(getattr(detection, 'handle_grasp_source', ''))
        if source:
            payload['handle_grasp_source'] = source
        if len(edge_px) != 4 or not joints:
            return payload

        try:
            T_base_hand = self._compute_base_to_hand(joints)
        except Exception as exc:
            payload['handle_grasp_message'] = 'FK unavailable for grasp endpoints: %s' % exc
            return payload

        if len(mid_px) == 2:
            mid_right = self._project_mid_right_air_with_pose(mid_px[0], mid_px[1], T_base_hand)
            if mid_right is not None:
                air_world, air_target, surface_world, depth_m, depth_count = mid_right
                payload['handle_mid_px'] = [float(mid_px[0]), float(mid_px[1])]
                payload['handle_mid_right_offset_m'] = float(self.handle_mid_right_offset_m)
                payload['handle_mid_surface_world_m'] = [float(x) for x in surface_world]
                payload['handle_mid_right_air_world_m'] = [float(x) for x in air_world]
                payload['handle_mid_right_air_target_m'] = [float(x) for x in air_target]
                payload['handle_mid_right_air_detail'] = {
                    'valid': True,
                    'depth_m': float(depth_m),
                    'depth_valid_count': int(depth_count),
                    'message': 'middle right air target',
                }

        endpoints_world = []
        endpoint_targets = []
        endpoint_details = []
        for u, v in ((edge_px[0], edge_px[1]), (edge_px[2], edge_px[3])):
            projected = self._project_pixel_with_pose(float(u), float(v), T_base_hand)
            if projected is None:
                endpoint_details.append({'valid': False, 'message': 'no valid depth near endpoint'})
                continue
            contact, depth_m, depth_count = projected
            target, target_message = self._action_target_for_detection(detection, contact, T_base_hand)
            endpoints_world.append([float(x) for x in contact])
            endpoint_targets.append([float(x) for x in target])
            endpoint_details.append(
                {
                    'valid': True,
                    'depth_m': float(depth_m),
                    'depth_valid_count': int(depth_count),
                    'message': target_message,
                }
            )

        payload['handle_grasp_endpoint_details'] = endpoint_details
        if len(endpoints_world) == 2:
            ep0 = np.asarray(endpoints_world[0], dtype=np.float64)
            ep1 = np.asarray(endpoints_world[1], dtype=np.float64)
            center = 0.5 * (ep0 + ep1)
            center_target, center_message = self._action_target_for_detection(detection, center, T_base_hand)
            payload['handle_grasp_endpoints_world_m'] = endpoints_world
            payload['handle_grasp_endpoint_targets_m'] = endpoint_targets
            payload['handle_grasp_center_world_m'] = [float(x) for x in center]
            payload['handle_grasp_ree_target_m'] = [float(x) for x in center_target]
            payload['handle_grasp_width_m'] = float(np.linalg.norm(ep1 - ep0))
            payload['handle_grasp_message'] = center_message
        return payload

    def _estimate_depth_handle_grasp_edge_px(self, detection):
        depth = self.latest_depth
        info = self.latest_camera_info
        if depth is None or depth.size == 0 or info is None:
            return None

        image_w = int(info.width) if int(info.width) > 0 else 1280
        image_h = int(info.height) if int(info.height) > 0 else 720
        cx = float(getattr(detection, 'cx', 0.0))
        cy = float(getattr(detection, 'cy', 0.0))
        x1 = max(0.0, cx - float(self.handle_depth_search_left_px))
        x2 = min(float(image_w - 1), cx - float(self.handle_depth_search_right_px))
        y1 = max(0.0, cy - float(self.handle_depth_search_y_px))
        y2 = min(float(image_h - 1), cy + float(self.handle_depth_search_y_px))
        if x2 <= x1 + 8.0 or y2 <= y1 + 8.0:
            return None

        h, w = depth.shape[:2]
        dx0, dy0 = self._map_pixel_to_depth_image(x1, y1, w, h)
        dx1, dy1 = self._map_pixel_to_depth_image(x2, y2, w, h)
        dx0, dx1 = sorted((dx0, dx1))
        dy0, dy1 = sorted((dy0, dy1))
        dx1 = min(w, dx1 + 1)
        dy1 = min(h, dy1 + 1)
        patch = np.asarray(depth[dy0:dy1, dx0:dx1])
        if patch.size == 0:
            return None

        depth_m = self._depth_patch_to_meters(patch)
        valid = np.isfinite(depth_m) & (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m)
        if int(np.count_nonzero(valid)) < 40:
            return None

        vals = depth_m[valid]
        background = float(np.percentile(vals, 70.0))
        near_cut = min(float(np.percentile(vals, 25.0)), background - max(0.002, self.handle_depth_near_delta_m))
        mask = valid & (depth_m <= near_cut)
        if int(np.count_nonzero(mask)) < 20:
            near_cut = float(np.percentile(vals, 18.0))
            mask = valid & (depth_m <= near_cut)
        if int(np.count_nonzero(mask)) < 20:
            return None

        mask_u8 = (mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        sx = float(image_w) / float(w)
        sy = float(image_h) / float(h)
        blue_pt = np.asarray([cx, cy], dtype=np.float32)
        scored = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 18.0:
                continue
            contour_color = contour.astype(np.float32)
            contour_color[:, 0, 0] = (contour_color[:, 0, 0] + float(dx0)) * sx
            contour_color[:, 0, 1] = (contour_color[:, 0, 1] + float(dy0)) * sy
            rect = cv2.minAreaRect(contour_color)
            rect_w, rect_h = float(rect[1][0]), float(rect[1][1])
            rect_short = max(1.0, min(rect_w, rect_h))
            rect_long = max(rect_w, rect_h)
            if rect_long < 18.0 or rect_short < 5.0:
                continue

            angle = np.deg2rad(float(rect[2]))
            axis_w = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            axis_h = np.asarray([-np.sin(angle), np.cos(angle)], dtype=np.float32)
            if rect_w >= rect_h:
                long_axis = axis_w
                short_axis = axis_h
                long_len = rect_w
                short_len = rect_h
            else:
                long_axis = axis_h
                short_axis = axis_w
                long_len = rect_h
                short_len = rect_w

            center = np.asarray(rect[0], dtype=np.float32)
            end_a = center - long_axis * (long_len * 0.5)
            end_b = center + long_axis * (long_len * 0.5)
            use_a = np.linalg.norm(end_a - blue_pt) <= np.linalg.norm(end_b - blue_pt)
            tip = end_a if use_a else end_b
            inward = long_axis if use_a else -long_axis
            target_center = tip + inward * (long_len * 0.10)
            cross_half = short_axis * (short_len * 0.5)
            endpoints = np.stack([target_center - cross_half, target_center + cross_half], axis=0)
            tip_dist = float(np.linalg.norm(tip - blue_pt))
            center_dist = float(np.linalg.norm(target_center - blue_pt))
            if tip_dist > self.handle_depth_max_blue_distance_px:
                continue
            distance_score = -3.0 * tip_dist - 0.5 * center_dist
            right_side_score = 0.25 * float(target_center[0])
            area_score = min(25.0, 0.004 * float(area))
            length_score = min(20.0, 0.15 * float(rect_long))
            scored.append(
                (
                    distance_score + right_side_score + area_score + length_score,
                    endpoints,
                    target_center,
                    rect_long,
                    area,
                    near_cut,
                    background,
                    tip_dist,
                    center,
                )
            )

        if not scored:
            return None
        _, endpoints, center, long_len, area, near_cut, background, tip_dist, rect_center = max(scored, key=lambda item: item[0])
        endpoints[:, 0] = np.clip(endpoints[:, 0], 0.0, float(image_w - 1))
        endpoints[:, 1] = np.clip(endpoints[:, 1], 0.0, float(image_h - 1))
        center[0] = np.clip(center[0], 0.0, float(image_w - 1))
        center[1] = np.clip(center[1], 0.0, float(image_h - 1))
        rect_center = np.asarray(rect_center, dtype=np.float32)
        rect_center[0] = np.clip(rect_center[0], 0.0, float(image_w - 1))
        rect_center[1] = np.clip(rect_center[1], 0.0, float(image_h - 1))
        endpoints_i = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in endpoints]
        result = {
            'handle_grasp_edge_px': endpoints_i,
            'handle_grasp_center_px': [int(round(float(center[0]))), int(round(float(center[1])))],
            'handle_mid_px': [int(round(float(rect_center[0]))), int(round(float(rect_center[1])))],
            'handle_grasp_width_px': float(np.linalg.norm(endpoints[1] - endpoints[0])),
            'handle_grasp_source': 'depth_near_blue_left_roi',
            'handle_depth_near_cut_m': float(near_cut),
            'handle_depth_background_m': float(background),
            'handle_depth_mask_area_px': float(area),
            'handle_grasp_long_axis_length_px': float(long_len),
            'handle_depth_tip_distance_px': float(tip_dist),
        }
        now = time.time()
        previous = self._last_depth_handle_grasp
        if previous is not None and now - float(previous.get('time', 0.0)) < 1.0:
            prev_center = np.asarray(previous.get('center_px', []), dtype=np.float64)
            new_center = np.asarray(result['handle_grasp_center_px'], dtype=np.float64)
            if prev_center.shape == (2,) and np.linalg.norm(prev_center - new_center) > self.handle_depth_sticky_px:
                prev_result = dict(previous.get('result', {}))
                if prev_result:
                    prev_result['handle_grasp_source'] = str(prev_result.get('handle_grasp_source', 'depth_near_blue_left_roi')) + '_sticky'
                    return prev_result
        self._last_depth_handle_grasp = {
            'time': now,
            'center_px': result['handle_grasp_center_px'],
            'result': dict(result),
        }
        return result

    def _project_pixel_with_pose(self, u, v, T_base_hand):
        if self.T_cam2hand is None:
            return None
        depth = self._sample_depth_m(u, v)
        if depth is None:
            return None
        depth_m, depth_count = depth
        point_camera = self._deproject(float(u), float(v), float(depth_m))
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = point_camera
        point_hand = self.T_cam2hand @ point_h
        point_base = T_base_hand @ point_hand
        return point_base[:3], float(depth_m), int(depth_count)

    def _project_mid_right_air_with_pose(self, u, v, T_base_hand):
        projected = self._project_pixel_with_pose(float(u), float(v), T_base_hand)
        if projected is None:
            return None
        surface_world, depth_m, depth_count = projected

        info = self.latest_camera_info
        image_w = int(info.width) if info is not None and int(info.width) > 0 else 1280
        du = 10.0
        u_right = min(float(image_w - 1), float(u) + du)
        point_camera = self._deproject(float(u), float(v), float(depth_m))
        point_camera_right = self._deproject(u_right, float(v), float(depth_m))
        direction_camera = point_camera_right - point_camera
        if np.linalg.norm(direction_camera) <= 1e-9:
            return None

        rotation_cam_to_base = np.asarray(T_base_hand, dtype=np.float64)[:3, :3] @ np.asarray(self.T_cam2hand, dtype=np.float64)[:3, :3]
        direction_world = rotation_cam_to_base @ direction_camera
        norm = np.linalg.norm(direction_world)
        if norm <= 1e-9:
            return None
        direction_world = direction_world / norm
        air_world = np.asarray(surface_world, dtype=np.float64).reshape(3) + direction_world * float(self.handle_mid_right_offset_m)
        air_target, _ = self._action_target_for_detection(None, air_world, T_base_hand)
        return air_world, air_target, np.asarray(surface_world, dtype=np.float64).reshape(3), float(depth_m), int(depth_count)

    def _publish_json_string(self, publisher, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)

    def _robot_status_payload(self, msg):
        if msg is None:
            return {
                'available': False,
                'stage_id': 0,
                'stage_name': '',
                'current_action': '',
                'motion_active': False,
                'progress': 0.0,
                'has_error': False,
                'error_code': '',
                'error_message': '',
                'emergency_stop': False,
                'target_reachable': False,
                'reachability_message': 'no /robot/inspection_status received',
                'target_id': '',
            }
        return {
            'available': True,
            'header': {
                'stamp': {
                    'sec': int(msg.header.stamp.sec),
                    'nanosec': int(msg.header.stamp.nanosec),
                },
                'frame_id': str(msg.header.frame_id),
            },
            'stage_id': int(msg.stage_id),
            'stage_name': str(msg.stage_name),
            'current_action': str(msg.current_action),
            'motion_active': bool(msg.motion_active),
            'progress': float(msg.progress),
            'has_error': bool(msg.has_error),
            'error_code': str(msg.error_code),
            'error_message': str(msg.error_message),
            'emergency_stop': bool(msg.emergency_stop),
            'target_reachable': bool(msg.target_reachable),
            'reachability_message': str(msg.reachability_message),
            'target_id': str(msg.target_id),
        }

    def _solve_object_ik(self, object_3d, joints):
        target_pose = self._ik_target_pose_matrix(object_3d.point_target, joints)
        return self._ik_solver.solve(
            target_link=self.ik_target_link or self.hand_link or self._default_hand_link(),
            target_pose=target_pose,
            initial_joint_values=self._lock_joint_values(joints),
            active_joints=self.ik_active_joints or None,
            base_link=self.base_link or self._default_base_link(),
            base_pose=np.eye(4, dtype=np.float64).tolist(),
            max_iterations=self.ik_max_iterations,
            tolerance_position=self.ik_tolerance_position,
            tolerance_orientation=self.ik_tolerance_orientation,
            damping=self.ik_damping,
            step_scale=self.ik_step_scale,
            position_weight=self.ik_position_weight,
            orientation_weight=self.ik_orientation_weight,
            clamp_to_limits=True,
        )

    def _object_id(self, index, class_name):
        safe = ''.join(ch if ch.isalnum() else '_' for ch in str(class_name).lower()).strip('_')
        return '%02d_%s' % (int(index), safe or 'object')

    def _point_payload(self, point):
        return {'x': float(point.x), 'y': float(point.y), 'z': float(point.z)}

    def _publish_target(self, object_3d):
        header = object_3d.header
        header.frame_id = object_3d.target_frame or object_3d.source_frame

        point_msg = PointStamped()
        point_msg.header = header
        point_msg.point = object_3d.point_target
        self.target_point_pub.publish(point_msg)

        if self.publish_target_pose:
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.pose.position = object_3d.point_target
            qx, qy, qz, qw = self.pose_orientation_xyzw
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            self.target_pose_pub.publish(pose_msg)

        self._publish_target_joint_state(object_3d, header)

    def _publish_target_joint_state(self, object_3d, header):
        if not self.publish_target_joint_state:
            return
        if self._ik_solver is None:
            self._warn_ik('IK solver is unavailable: %s' % (self._ik_error or 'not initialized'))
            return

        joints = self._current_joint_values()
        if not joints:
            self._warn_ik('IK skipped: missing current joint values')
            return

        try:
            target_pose = self._ik_target_pose_matrix(object_3d.point_target, joints)
            solution = self._ik_solver.solve(
                target_link=self.ik_target_link or self.hand_link or self._default_hand_link(),
                target_pose=target_pose,
                initial_joint_values=self._lock_joint_values(joints),
                active_joints=self.ik_active_joints or None,
                base_link=self.base_link or self._default_base_link(),
                base_pose=np.eye(4, dtype=np.float64).tolist(),
                max_iterations=self.ik_max_iterations,
                tolerance_position=self.ik_tolerance_position,
                tolerance_orientation=self.ik_tolerance_orientation,
                damping=self.ik_damping,
                step_scale=self.ik_step_scale,
                position_weight=self.ik_position_weight,
                orientation_weight=self.ik_orientation_weight,
                clamp_to_limits=True,
            )
        except Exception as exc:
            self._warn_ik('IK failed: %s' % exc)
            return

        if not solution.success:
            self._warn_ik(
                'IK did not converge: %s, pos_err=%.4f, rot_err=%.4f'
                % (solution.message, solution.final_position_error_norm, solution.final_orientation_error_norm)
            )
            if not self.publish_failed_ik_solution:
                return

        joint_msg = JointState()
        joint_msg.header = header
        joint_msg.name = list(solution.active_joints)
        joint_msg.position = [float(solution.joint_values[name]) for name in solution.active_joints]
        self.target_joint_state_pub.publish(joint_msg)

    def _ik_target_pose_matrix(self, point_msg, joints=None):
        qx, qy, qz, qw = self.pose_orientation_xyzw
        rotation = self._quat_to_matrix([qx, qy, qz, qw])
        if joints:
            try:
                rotation = np.asarray(self._compute_base_to_hand(joints), dtype=np.float64)[:3, :3]
            except Exception as exc:
                self._warn_ik('IK using configured orientation; current wrist rotation unavailable: %s' % exc)
        position = np.array([point_msg.x, point_msg.y, point_msg.z], dtype=np.float64)
        position = position + np.asarray(self.ik_target_position_offset_xyz, dtype=np.float64).reshape(3)
        ee_offset = np.asarray(self.ik_end_effector_offset_xyz, dtype=np.float64).reshape(3)
        link_position = position - rotation @ ee_offset

        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = rotation
        target[:3, 3] = link_position
        return target.tolist()

    def _quat_to_matrix(self, quat_xyzw):
        x, y, z, w = [float(v) for v in quat_xyzw]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _warn_ik(self, message):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_ik_warning_sec >= 2.0:
            self.get_logger().warn(message)
            self._last_ik_warning_sec = now_sec

    def _point_to_msg(self, point):
        msg = Point()
        msg.x = float(point[0])
        msg.y = float(point[1])
        msg.z = float(point[2])
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CoordinateProjectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
