#!/usr/bin/env python3
"""Single-process RealSense, YOLO, 3D projection and optional WebUI."""

import argparse
import os
import queue
import sys
import time
import traceback
from pathlib import Path


def _bootstrap_ros_python_paths():
    """Keep rclpy importable after the H2 dependency loader re-execs Python."""
    for path in (
        '/opt/ros/humble/local/lib/python3.10/dist-packages',
        '/opt/ros/humble/lib/python3.10/site-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/detector_msgs/local/lib/python3.10/dist-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/detector_msgs/lib/python3.10/site-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/yolo_trt_ros2/local/lib/python3.10/dist-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/yolo_trt_ros2/lib/python3.10/site-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2',
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_ros_python_paths()
# A merged ROS/YOLO/FK process cannot survive either dependency helper replacing
# the interpreter: the H2 Pinocchio helper removes ROS library paths, while the
# ROS helper loses launch-time Python paths. Both libraries already coexist in
# the initial Humble process, so force their guarded in-process paths.
os.environ['YOLO_TRT_ROS2_LD_BOOTSTRAPPED'] = '1'
os.environ['H2_XR_PINOCCHIO_ENV'] = 'clean'

# Import this first: its H2 bootstrap discovers ROS library directories and
# preloads librcl*.so before any direct rclpy import. This ordering must also
# hold after that bootstrap re-execs the Python process.
from yolo_trt_ros2.coordinate_projector_node import CoordinateProjectorNode

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from yolo_trt_ros2.web_dashboard_node import WebDashboardNode
from yolo_trt_ros2.yolo_detector_node import YoloDetectorNode
from yolo_trt_ros2.camera_info_utils import set_camera_info_intrinsics
from yolo_trt_ros2.perception_singleton import (
    PerceptionAlreadyRunningError,
    assert_no_conflicting_perception,
)


class IntegratedRealSenseNode(Node):
    """Own the camera and move synchronized frames through in-process APIs."""

    def __init__(self, detector, projector, web_dashboard=None):
        super().__init__('direct_realsense')
        self._declare_parameters()

        self.detector = detector
        self.projector = projector
        self.web_dashboard = web_dashboard
        self.workspace_root = str(self.get_parameter('workspace_root').value)
        self.color_frame_id = str(self.get_parameter('color_frame_id').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.depth_width = int(self.get_parameter('depth_width').value)
        self.depth_height = int(self.get_parameter('depth_height').value)
        self.depth_fps = int(self.get_parameter('depth_fps').value)
        self.cam_index = int(self.get_parameter('cam_index').value)
        self.cam_serial = str(self.get_parameter('cam_serial').value)
        serial_override = str(self.get_parameter('camera_serial_override').value).strip()
        index_override = int(self.get_parameter('camera_index_override').value)
        if serial_override:
            self.cam_serial = serial_override
        if index_override >= 0:
            self.cam_index = index_override
        self.frame_timeout_ms = int(self.get_parameter('frame_timeout_ms').value)
        self.publish_period_sec = float(self.get_parameter('publish_period_sec').value)
        self.restart_on_timeout_count = int(self.get_parameter('restart_on_timeout_count').value)

        self.camera = None
        self.camera_info = None
        self.timeout_count = 0
        self._last_queue_warning_sec = 0.0
        self._projection_queue = queue.Queue(maxsize=1)

        self._start_camera()
        self.capture_timer = self.create_timer(max(0.001, self.publish_period_sec), self._capture_callback)
        # This timer belongs to the projector node. Its default mutually-exclusive
        # callback group serializes projection with pixel-query and joint callbacks.
        self.projector.create_timer(0.01, self._drain_projection_queue)

        self.get_logger().info(
            'Integrated perception started: raw ROS image topics disabled, webUI=%s'
            % bool(self.web_dashboard)
        )

    def _declare_parameters(self):
        self.declare_parameter('workspace_root', '/home/unitree/MscapeTech')
        self.declare_parameter('color_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        self.declare_parameter('depth_fps', 30)
        self.declare_parameter('cam_index', 0)
        self.declare_parameter('cam_serial', '')
        self.declare_parameter(
            'camera_serial_override',
            '',
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter('camera_index_override', -1)
        self.declare_parameter('frame_timeout_ms', 1000)
        self.declare_parameter('publish_period_sec', 0.033)
        self.declare_parameter('restart_on_timeout_count', 30)

    def _add_workspace_paths(self):
        workspace = Path(os.path.expanduser(self.workspace_root))
        for path in (
            workspace,
            workspace / 'handle_recognition',
            workspace / 'handle_recognition' / 'minimal_test',
        ):
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
            _, _, self.camera_info = self.camera.color_intrinsics()
            self.timeout_count = 0
            self.get_logger().info(
                'Integrated RealSense started: requested_serial=%s fallback_index=%d active=%s'
                % (
                    self.cam_serial or '<empty>',
                    self.cam_index,
                    getattr(self.camera, 'active_config', {}),
                )
            )
        except Exception as exc:
            self.camera = None
            self.camera_info = None
            self.get_logger().error(
                'Failed to start integrated RealSense: %s\n%s' % (exc, traceback.format_exc())
            )

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
        msg.distortion_model = 'plumb_bob'
        set_camera_info_intrinsics(
            msg,
            fx,
            fy,
            ppx,
            ppy,
            info.get('coeffs', [0.0] * 5),
        )
        return msg

    def _capture_callback(self):
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
        rgb = np.asarray(frame['rgb'])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        objects_msg, debug_image = self.detector.process_frame(bgr, self._header(stamp), publish_objects=False)
        if objects_msg is None:
            return

        item = (
            np.asarray(frame['depth']).copy(),
            self._camera_info_msg(stamp),
            objects_msg,
            debug_image,
        )
        if self._projection_queue.full():
            try:
                self._projection_queue.get_nowait()
            except queue.Empty:
                pass
            now = time.monotonic()
            if now - self._last_queue_warning_sec >= 5.0:
                self._last_queue_warning_sec = now
                self.get_logger().warn('Projector is behind; replacing one pending perception frame.')
        self._projection_queue.put_nowait(item)

    def _header(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.color_frame_id
        return header

    def _drain_projection_queue(self):
        try:
            depth, camera_info, objects_msg, debug_image = self._projection_queue.get_nowait()
        except queue.Empty:
            return

        self.projector.ingest_camera_info(camera_info)
        self.projector.ingest_depth(depth, objects_msg.header.stamp)
        self.detector.objects_pub.publish(objects_msg)
        self.projector.process_objects(objects_msg)
        if self.web_dashboard is not None:
            if debug_image is not None:
                self.web_dashboard.ingest_bgr_image(debug_image, objects_msg.header)
            self.web_dashboard.ingest_depth(depth)

    def destroy_node(self):
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        super().destroy_node()


def _parse_app_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--webUI', action='store_true', help='Enable the in-process HTTP dashboard.')
    non_ros_args = remove_ros_args(args=argv)
    app_args = non_ros_args
    if app_args and not str(app_args[0]).startswith('-'):
        app_args = app_args[1:]
    options, _unknown = parser.parse_known_args(app_args)
    clean_argv = [arg for arg in argv if arg != '--webUI']
    return options, clean_argv


def _web_port_from_argv(argv):
    """Best-effort parse of `-p web_port:=8081` before nodes declare params."""
    tokens = [str(item) for item in argv]
    for index, token in enumerate(tokens):
        if token.startswith('web_port:='):
            value = token.split(':=', 1)[1].strip().strip("'\"")
            try:
                return int(value)
            except ValueError:
                return None
        if token in ('-p', '--param') and index + 1 < len(tokens):
            nxt = tokens[index + 1]
            if nxt.startswith('web_port:='):
                value = nxt.split(':=', 1)[1].strip().strip("'\"")
                try:
                    return int(value)
                except ValueError:
                    return None
    return None


def _endpoint_topics(node):
    """Return (publish_topics, subscribe_topics) for a live rclpy node."""
    pubs = []
    subs = []
    for pub in list(getattr(node, '_publishers', []) or []):
        topic = str(getattr(pub, 'topic_name', '') or '').strip()
        if topic:
            pubs.append(topic)
    for sub in list(getattr(node, '_subscriptions', []) or []):
        topic = str(getattr(sub, 'topic_name', '') or '').strip()
        if topic:
            subs.append(topic)
    return sorted(set(pubs)), sorted(set(subs))


def _log_integrated_topics(nodes):
    """Print publish/subscribe topics right after the integrated stack is up."""
    lines = [
        '======== Integrated perception ROS topics ========',
        'ROS_DOMAIN_ID=%s  ROS_LOCALHOST_ONLY=%s'
        % (
            os.environ.get('ROS_DOMAIN_ID', '0'),
            os.environ.get('ROS_LOCALHOST_ONLY', '<unset>'),
        ),
    ]
    all_pub = set()
    all_sub = set()
    for node in nodes:
        name = node.get_fully_qualified_name()
        pubs, subs = _endpoint_topics(node)
        all_pub.update(pubs)
        all_sub.update(subs)
        lines.append('node: %s' % name)
        lines.append('  publish (%d):' % len(pubs))
        if pubs:
            for topic in pubs:
                lines.append('    %s' % topic)
        else:
            lines.append('    (none)')
        lines.append('  subscribe (%d):' % len(subs))
        if subs:
            for topic in subs:
                lines.append('    %s' % topic)
        else:
            lines.append('    (none)')
    lines.append('---- summary ----')
    lines.append('all publish (%d): %s' % (len(all_pub), ', '.join(sorted(all_pub)) or '(none)'))
    lines.append('all subscribe (%d): %s' % (len(all_sub), ', '.join(sorted(all_sub)) or '(none)'))
    lines.append('======== end integrated topics ========')
    text = '\n'.join(lines)
    print(text, flush=True)
    if nodes:
        nodes[0].get_logger().info(text)


def main(args=None):
    argv = list(sys.argv if args is None else args)
    options, ros_argv = _parse_app_args(argv)
    web_port = _web_port_from_argv(ros_argv) if options.webUI else None
    if options.webUI and web_port is None:
        web_port = 8081
    try:
        assert_no_conflicting_perception(web_port=web_port)
    except PerceptionAlreadyRunningError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    rclpy.init(args=ros_argv)

    nodes = []
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        # Construct the projector first so ROS/FK compatibility is validated
        # before the comparatively expensive YOLO model is loaded.
        projector = CoordinateProjectorNode(subscribe_perception_inputs=False)
        nodes.append(projector)
        detector = YoloDetectorNode(subscribe_image=False, enable_debug_output=options.webUI)
        nodes.append(detector)
        web_dashboard = WebDashboardNode(subscribe_debug_image=False) if options.webUI else None
        if web_dashboard is not None:
            nodes.append(web_dashboard)
        camera = IntegratedRealSenseNode(detector, projector, web_dashboard)
        nodes.append(camera)

        for node in nodes:
            executor.add_node(node)
        _log_integrated_topics(nodes)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
