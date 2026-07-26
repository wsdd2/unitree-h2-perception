#!/usr/bin/env python3
"""Publish a typed cabinet inspection target/action request."""

import argparse

import rclpy
from rclpy.node import Node

from detector_msgs.msg import InspectionCommand


def parse_args():
    parser = argparse.ArgumentParser(description='Publish /robot/inspection_command.')
    parser.add_argument('--command-id', required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--stage-name', default='')
    parser.add_argument(
        '--action',
        choices=('observe_targets', 'move', 'press', 'grasp_rotate', 'door', 'abort'),
        required=True,
    )
    parser.add_argument('--class-name', action='append', default=[])
    parser.add_argument('--active-class', default='')
    parser.add_argument('--semantic-name', action='append', default=[])
    parser.add_argument('--active-semantic', default='')
    parser.add_argument(
        '--selection-policy',
        choices=('highest_confidence', 'nearest', 'specified_target_id'),
        default='highest_confidence',
    )
    parser.add_argument('--target-id', default='')
    parser.add_argument('--lock-target', action='store_true')
    parser.add_argument('--min-confidence', type=float, default=0.20)
    parser.add_argument('--stable-frames', type=int, default=5)
    parser.add_argument('--max-position-std-m', type=float, default=0.003)
    parser.add_argument('--timeout-sec', type=float, default=10.0)
    parser.add_argument('--rate', type=float, default=5.0)
    return parser.parse_args()


class InspectionCommandPublisher(Node):
    def __init__(self, args):
        super().__init__('inspection_command_publisher_example')
        self.args = args
        self.publisher = self.create_publisher(
            InspectionCommand,
            '/robot/inspection_command',
            10,
        )
        self.timer = self.create_timer(1.0 / max(0.1, args.rate), self._publish)

    def _publish(self):
        msg = InspectionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'pelvis'
        msg.command_id = self.args.command_id
        msg.stage_id = self.args.stage
        msg.stage_name = self.args.stage_name
        msg.requested_action = self.args.action
        msg.requested_class_names = list(self.args.class_name)
        msg.active_target_class_name = self.args.active_class
        msg.requested_semantic_names = list(self.args.semantic_name)
        msg.active_target_semantic_name = self.args.active_semantic
        msg.selection_policy = self.args.selection_policy
        msg.target_id = self.args.target_id
        msg.lock_target = bool(self.args.lock_target)
        msg.min_confidence = float(self.args.min_confidence)
        msg.required_stable_frames = max(1, int(self.args.stable_frames))
        msg.max_position_std_m = max(0.0, float(self.args.max_position_std_m))
        msg.timeout_sec = max(0.0, float(self.args.timeout_sec))
        self.publisher.publish(msg)


def main():
    args = parse_args()
    rclpy.init()
    node = InspectionCommandPublisher(args)
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
