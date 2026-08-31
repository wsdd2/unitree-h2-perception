#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运控侧最小对接：发 InspectionCommand，自动收指定 class 的 3D 回传。

协议（不是“打印全部检测”）：
  运控发  →  /robot/inspection_command   (detector_msgs/InspectionCommand)
  视觉回  →  /detector/objects           (detector_msgs/Object3DArray)
             /detector/target_point      (geometry_msgs/PointStamped，当前锁定点)

H2 上（感知已启动）:

  source /opt/ros/humble/setup.bash
  source ~/MscapeTech/Foxy_ROS/install/setup.bash
  export ROS_DOMAIN_ID=42
  export ROS_LOCALHOST_ONLY=1
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

  python3 ~/MscapeTech/Foxy_ROS/examples/ops_request_class_3d.py
"""

from __future__ import annotations

import uuid

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node

from detector_msgs.msg import InspectionCommand, Object3DArray

# ========== 运控本阶段要的类（改这里）==========
REQUESTED_CLASS_NAMES = [
    'push button',
    'hang cord',
]
ACTIVE_TARGET_CLASS = 'push button'  # 本阶段要动的那一个；仅观察可 ''
REQUESTED_ACTION = 'observe_targets'  # observe_targets / move / press / ...
STAGE_ID = 1
STAGE_NAME = 'ops_request_class_3d'
COMMAND_HZ = 5.0  # 合同建议持续发，同一请求保持相同 command_id
# ==============================================


class OpsRequestClass3D(Node):
    """运控节点模板：发请求，在回调里自动拿到回传 3D。"""

    def __init__(self):
        super().__init__('ops_request_class_3d')
        self.command_id = 'ops_%s' % uuid.uuid4().hex[:8]
        self.want = {n.strip().lower() for n in REQUESTED_CLASS_NAMES if n.strip()}

        # 1) 发请求（运控 → 视觉）
        self.cmd_pub = self.create_publisher(
            InspectionCommand, '/robot/inspection_command', 10
        )
        self.create_timer(1.0 / max(0.1, COMMAND_HZ), self._publish_command)

        # 2) 收回传（视觉 → 运控），自动进回调
        self.create_subscription(
            Object3DArray, '/detector/objects', self._on_objects_reply, 10
        )
        self.create_subscription(
            PointStamped, '/detector/target_point', self._on_target_point, 10
        )

        self.get_logger().info(
            'request id=%s classes=%s active=%s → wait /detector/objects'
            % (self.command_id, REQUESTED_CLASS_NAMES, ACTIVE_TARGET_CLASS)
        )

    def _publish_command(self):
        msg = InspectionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'pelvis'
        msg.command_id = self.command_id  # 同一请求不要换 id
        msg.stage_id = STAGE_ID
        msg.stage_name = STAGE_NAME
        msg.requested_action = REQUESTED_ACTION
        msg.requested_class_names = list(REQUESTED_CLASS_NAMES)
        msg.active_target_class_name = ACTIVE_TARGET_CLASS
        msg.selection_policy = 'highest_confidence'
        msg.lock_target = False
        msg.min_confidence = 0.20
        msg.required_stable_frames = 3
        msg.max_position_std_m = 0.01
        msg.timeout_sec = 30.0
        self.cmd_pub.publish(msg)

    def _on_objects_reply(self, msg: Object3DArray):
        """视觉自动回传的全量/当前检测；运控只取自己请求的 class。"""
        for obj in msg.objects:
            if not obj.valid:
                continue
            name = str(obj.detection.class_name).strip()
            if name.lower() not in self.want:
                continue

            p = obj.point_target  # 默认 torso_link，见 obj.target_frame
            # ★ 这里接运控：把 (name, p.x, p.y, p.z) 交给规划 / 动臂
            self.on_class_xyz(
                class_name=name,
                x=float(p.x),
                y=float(p.y),
                z=float(p.z),
                frame=str(obj.target_frame or ''),
                confidence=float(obj.detection.confidence),
            )

    def _on_target_point(self, msg: PointStamped):
        """视觉按 active_target 选出的当前锁定点（可选订）。"""
        self.get_logger().info(
            'target_point frame=%s xyz=(%.4f, %.4f, %.4f)'
            % (msg.header.frame_id, msg.point.x, msg.point.y, msg.point.z)
        )

    def on_class_xyz(self, class_name, x, y, z, frame, confidence):
        """运控业务入口：收到请求类的 3D 后自动调用。"""
        self.get_logger().info(
            'REPLY %s conf=%.2f frame=%s xyz=(%.4f, %.4f, %.4f)'
            % (class_name, confidence, frame, x, y, z)
        )
        # TODO: 调用你们的 h2_arm / IK / 轨迹，例如：
        # self.arm_goto(x, y, z, frame)


def main():
    rclpy.init()
    node = OpsRequestClass3D()
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
