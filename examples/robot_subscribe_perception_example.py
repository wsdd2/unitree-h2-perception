#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot-side example: subscribe to perception results.

Recommended topic:
  /detector/objects_ik_json  std_msgs/String(JSON)

Why JSON first:
  - It contains object class, bbox, 3D target, IK result, and robot_status.
  - Robot control code can integrate it without depending on every custom field.

Run on H2:
  cd ~/MscapeTech/Foxy_ROS
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  export ROS_DOMAIN_ID=42
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  python3 examples/robot_subscribe_perception_example.py
"""

import json
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _xyz_from_dict(point: Optional[dict[str, Any]]) -> Optional[list[float]]:
    if not point:
        return None
    return [float(point.get("x", 0.0)), float(point.get("y", 0.0)), float(point.get("z", 0.0))]


class RobotPerceptionSubscriber(Node):
    def __init__(self) -> None:
        super().__init__("robot_perception_subscriber_example")
        self.create_subscription(
            String,
            "/detector/objects_ik_json",
            self._on_objects_ik_json,
            10,
        )
        self.get_logger().info("Listening: /detector/objects_ik_json")

    def _on_objects_ik_json(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid perception JSON: {exc}")
            return

        robot_status = data.get("robot_status", {})
        if robot_status.get("available"):
            self.get_logger().info(
                "robot_status stage=%s action=%s reachable=%s error=%s"
                % (
                    robot_status.get("stage_name", ""),
                    robot_status.get("current_action", ""),
                    robot_status.get("target_reachable", False),
                    robot_status.get("has_error", False),
                )
            )

        for obj in data.get("objects", []):
            class_name = str(obj.get("class_name", "")).lower()
            object_id = str(obj.get("object_id", ""))
            point_target = _xyz_from_dict(obj.get("point_target"))
            ik = obj.get("ik") or {}

            # Stage 2: blue button. Use this target for pressing the blue point.
            if "blue" in class_name or "red sticker push point" in class_name:
                self.get_logger().info(
                    "blue target id=%s xyz=%s ik_success=%s"
                    % (object_id, point_target, ik.get("success"))
                )
                if ik.get("success"):
                    joint_values = ik.get("joint_values_rad", {})
                    # TODO(robot engineer): send joint_values to the arm controller.
                    self.get_logger().info("blue IK joints=%s" % joint_values)

            # Stage 3: raised handle. Use this target for grasping/pulling.
            if "handle" in class_name:
                self.get_logger().info(
                    "handle target id=%s xyz=%s ik_success=%s"
                    % (object_id, point_target, ik.get("success"))
                )
                if ik.get("success"):
                    joint_values = ik.get("joint_values_rad", {})
                    # TODO(robot engineer): move gripper to this IK target, then close gripper.
                    self.get_logger().info("handle IK joints=%s" % joint_values)

            # Stage 6: pick the hang tag by the white force point / cord apex.
            if "work tag grasp" in class_name:
                self.get_logger().info(
                    "work-tag grasp id=%s xyz=%s ik_success=%s"
                    % (object_id, point_target, ik.get("success"))
                )

            # Stage 7: hang the tag onto the cabinet hook.
            if "cabinet hang hook" in class_name:
                self.get_logger().info(
                    "hang-hook id=%s xyz=%s ik_success=%s"
                    % (object_id, point_target, ik.get("success"))
                )


def main() -> None:
    rclpy.init()
    node = RobotPerceptionSubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
