#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键：发 inspection_command + 订阅指定 class 的 3D 坐标。

在 H2 上（感知已启动）:

  source /opt/ros/humble/setup.bash
  source ~/MscapeTech/Foxy_ROS/install/setup.bash
  export ROS_DOMAIN_ID=42
  export ROS_LOCALHOST_ONLY=1
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

  # 默认类名见下方 WANT_CLASS_NAMES，无参直接跑：
  python3 ~/MscapeTech/Foxy_ROS/examples/grab_class_3d_with_command.py

  # 或命令行覆盖类名：
  python3 ~/MscapeTech/Foxy_ROS/examples/grab_class_3d_with_command.py \\
    --class-name 'push button' --class-name 'hang cord' --active-class 'push button'
"""

from __future__ import annotations

import argparse
import time
import uuid

import rclpy
from rclpy.node import Node

from detector_msgs.msg import InspectionCommand, Object3DArray

# ========== 改这里即可（也可用 --class-name 覆盖）==========
WANT_CLASS_NAMES = [
    'push button',
    'toggle switch',
    'hang cord',
]
ACTIVE_TARGET_CLASS = 'push button'  # 驱动运动的那一类；仅观察可改成 ''
REQUESTED_ACTION = 'observe_targets'  # observe_targets / move / press / ...
COMMAND_RATE_HZ = 2.0
# ==========================================================


def parse_args():
    p = argparse.ArgumentParser(
        description='Publish inspection_command and print 3D for requested classes.',
    )
    p.add_argument(
        '--class-name',
        action='append',
        default=None,
        help='Requested class (repeatable). Default: WANT_CLASS_NAMES in file.',
    )
    p.add_argument(
        '--active-class',
        default=None,
        help='active_target_class_name. Default: ACTIVE_TARGET_CLASS in file.',
    )
    p.add_argument(
        '--action',
        default=REQUESTED_ACTION,
        choices=('observe_targets', 'move', 'press', 'grasp_rotate', 'door', 'abort'),
    )
    p.add_argument('--stage', type=int, default=1)
    p.add_argument('--stage-name', default='grab_class_3d')
    p.add_argument('--command-id', default='')
    p.add_argument('--rate', type=float, default=COMMAND_RATE_HZ)
    p.add_argument('--min-confidence', type=float, default=0.20)
    p.add_argument('--once', action='store_true', help='Print first match then exit.')
    p.add_argument(
        '--no-command',
        action='store_true',
        help='Only subscribe /detector/objects, do not publish inspection_command.',
    )
    return p.parse_args()


class GrabClass3DNode(Node):
    def __init__(self, args):
        super().__init__('grab_class_3d_with_command')
        self.args = args
        self.want = list(args.class_name or WANT_CLASS_NAMES)
        self.want_set = {n.strip().lower() for n in self.want if n.strip()}
        self.active = (
            args.active_class
            if args.active_class is not None
            else ACTIVE_TARGET_CLASS
        )
        self.command_id = args.command_id or ('grab3d_%s' % uuid.uuid4().hex[:8])
        self._got_once = False
        self._last_print_key = None
        self._last_print_t = 0.0

        self.create_subscription(
            Object3DArray,
            '/detector/objects',
            self._on_objects,
            10,
        )
        self.get_logger().info(
            'subscribe=/detector/objects want=%s active=%s' % (self.want, self.active)
        )

        self.cmd_pub = None
        if not args.no_command:
            self.cmd_pub = self.create_publisher(
                InspectionCommand,
                '/robot/inspection_command',
                10,
            )
            period = 1.0 / max(0.1, float(args.rate))
            self.create_timer(period, self._publish_command)
            self.get_logger().info(
                'publish=/robot/inspection_command id=%s action=%s @ %.1f Hz'
                % (self.command_id, args.action, args.rate)
            )

    def _publish_command(self):
        msg = InspectionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'pelvis'
        msg.command_id = self.command_id
        msg.stage_id = int(self.args.stage)
        msg.stage_name = str(self.args.stage_name)
        msg.requested_action = str(self.args.action)
        msg.requested_class_names = list(self.want)
        msg.active_target_class_name = str(self.active or '')
        msg.selection_policy = 'highest_confidence'
        msg.lock_target = False
        msg.min_confidence = float(self.args.min_confidence)
        msg.required_stable_frames = 3
        msg.max_position_std_m = 0.01
        msg.timeout_sec = 30.0
        self.cmd_pub.publish(msg)

    def _on_objects(self, msg: Object3DArray):
        hits = []
        for obj in msg.objects:
            if not obj.valid:
                continue
            name = str(obj.detection.class_name).strip()
            if name.lower() not in self.want_set:
                continue
            p = obj.point_target
            hits.append(
                '%s conf=%.2f frame=%s xyz=(%.4f, %.4f, %.4f) cam=(%.4f, %.4f, %.4f)'
                % (
                    name,
                    float(obj.detection.confidence),
                    obj.target_frame or '?',
                    float(p.x),
                    float(p.y),
                    float(p.z),
                    float(obj.point_camera.x),
                    float(obj.point_camera.y),
                    float(obj.point_camera.z),
                )
            )

        if not hits:
            return

        key = '|'.join(hits)
        now = time.monotonic()
        # 降噪：同一内容 0.5s 内不重复刷
        if key == self._last_print_key and (now - self._last_print_t) < 0.5:
            return
        self._last_print_key = key
        self._last_print_t = now

        self.get_logger().info('--- matched %d ---' % len(hits))
        for line in hits:
            self.get_logger().info(line)

        if self.args.once:
            self._got_once = True
            raise SystemExit(0)


def main():
    args = parse_args()
    rclpy.init()
    node = GrabClass3DNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
