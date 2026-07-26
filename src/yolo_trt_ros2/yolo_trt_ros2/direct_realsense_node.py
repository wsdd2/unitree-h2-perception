#!/usr/bin/env python3
"""Publish RealSense RGB-D topics using the same direct camera code as field tests."""

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from yolo_trt_ros2.camera_info_utils import set_camera_info_intrinsics


class DirectRealSenseNode(Node):
    def __init__(self):
        super().__init__('direct_realsense')
        self._declare_parameters()

        self.workspace_root = str(self.get_parameter('workspace_root').value)
        self.color_topic = str(self.get_parameter('color_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.color_frame_id = str(self.get_parameter('color_frame_id').value)
        self.depth_frame_id = str(self.get_parameter('depth_frame_id').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.depth_width = int(self.get_parameter('depth_width').value)
        self.depth_height = int(self.get_parameter('depth_height').value)
        self.depth_fps = int(self.get_parameter('depth_fps').value)
        self.cam_index = int(self.get_parameter('cam_index').value)
        self.cam_serial = str(self.get_parameter('cam_serial').value)
        self.frame_timeout_ms = int(self.get_parameter('frame_timeout_ms').value)
        self.publish_period_sec = float(self.get_parameter('publish_period_sec').value)
        self.restart_on_timeout_count = int(self.get_parameter('restart_on_timeout_count').value)

        self.bridge = CvBridge()
        self.camera = None
        self.camera_info = None
        self.timeout_count = 0

        self.color_pub = self.create_publisher(Image, self.color_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)

        self._start_camera()
        self.timer = self.create_timer(max(0.001, self.publish_period_sec), self._timer_callback)

    def _declare_parameters(self):
        self.declare_parameter('workspace_root', '/home/unitree/MscapeTech')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('color_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('depth_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        self.declare_parameter('depth_fps', 30)
        self.declare_parameter('cam_index', 0)
        self.declare_parameter('cam_serial', '')
        self.declare_parameter('frame_timeout_ms', 1000)
        self.declare_parameter('publish_period_sec', 0.033)
        self.declare_parameter('restart_on_timeout_count', 30)

    def _add_workspace_paths(self):
        workspace = Path(os.path.expanduser(self.workspace_root))
        paths = [
            workspace,
            workspace / 'handle_recognition',
            workspace / 'handle_recognition' / 'minimal_test',
        ]
        for path in paths:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

    def _start_camera(self):
        self._add_workspace_paths()
        try:
            from handle_recognition.realsense_stream import RealSenseRGBD

            self.camera = RealSenseRGBD(
                index=self.cam_index,
                serial=self.cam_serial,
                width=self.width,
                height=self.height,
                fps=self.fps,
                depth_width=self.depth_width,
                depth_height=self.depth_height,
                depth_fps=self.depth_fps,
            )
            self.camera.start()
            _, _, info = self.camera.color_intrinsics()
            self.camera_info = info
            self.timeout_count = 0
            self.get_logger().info(
                'Direct RealSense started: color=%s depth=%s camera_info=%s active=%s'
                % (self.color_topic, self.depth_topic, self.camera_info_topic, getattr(self.camera, 'active_config', {}))
            )
        except Exception as exc:
            self.camera = None
            self.camera_info = None
            self.get_logger().error('Failed to start direct RealSense: %s\n%s' % (exc, traceback.format_exc()))

    def _restart_camera(self):
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.camera = None
        self.camera_info = None
        self._start_camera()

    def _camera_info_msg(self, stamp):
        info = self.camera_info or {}
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.color_frame_id
        msg.width = int(info.get('width') or self.width)
        msg.height = int(info.get('height') or self.height)
        fx = float(info.get('fx') or 0.0)
        fy = float(info.get('fy') or 0.0)
        ppx = float(info.get('ppx') or 0.0)
        ppy = float(info.get('ppy') or 0.0)
        coeffs = [float(v) for v in info.get('coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])]
        msg.distortion_model = 'plumb_bob'
        set_camera_info_intrinsics(msg, fx, fy, ppx, ppy, coeffs)
        return msg

    def _timer_callback(self):
        if self.camera is None:
            self._start_camera()
            return
        try:
            frame = self.camera.fetch(timeout_ms=self.frame_timeout_ms)
        except Exception as exc:
            self.get_logger().warn('RealSense fetch failed, restarting: %s' % exc)
            self._restart_camera()
            return

        if frame is None:
            self.timeout_count += 1
            if self.timeout_count == 1 or self.timeout_count % 30 == 0:
                self.get_logger().warn('No RealSense frame received, timeout_count=%d' % self.timeout_count)
            if self.restart_on_timeout_count > 0 and self.timeout_count >= self.restart_on_timeout_count:
                self.get_logger().warn('Restarting RealSense after %d timeouts.' % self.timeout_count)
                self._restart_camera()
            return
        self.timeout_count = 0

        stamp = self.get_clock().now().to_msg()
        rgb = np.ascontiguousarray(frame['rgb'])
        depth = np.ascontiguousarray(frame['depth'])

        color_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = self.color_frame_id

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.depth_frame_id

        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(self._camera_info_msg(stamp))

    def destroy_node(self):
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DirectRealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
