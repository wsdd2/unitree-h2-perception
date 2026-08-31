#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运控适配参考：/h2_arm/vision_command (std_srvs/Trigger)

流程（视觉本身没有 Trigger，由本节点/你们 C++ 包一层）:

  ros2 service call /h2_arm/vision_command std_srvs/srv/Trigger
      │
      ├─ 读 YAML（InspectionCommand 字段）
      ├─ 发  /robot/inspection_command
      ├─ 等  /detector/objects 里 requested_class_names 满足
      │     min_confidence / required_stable_frames / timeout_sec
      └─ Trigger.response: success + message(JSON，含各类 3D)

注意：
  /h2_arm/singlearmjoint 是动臂服务，和视觉无关；
  先 vision_command 拿到坐标，再自行调 singlearmjoint。

H2:

  source /opt/ros/humble/setup.bash
  source ~/MscapeTech/Foxy_ROS/install/setup.bash
  export ROS_DOMAIN_ID=42
  export ROS_LOCALHOST_ONLY=1

  # 终端1：起桥（感知已开）
  python3 ~/MscapeTech/Foxy_ROS/examples/h2_arm_vision_command_bridge.py \\
    --yaml ~/MscapeTech/Foxy_ROS/examples/vision_command_observe_lock_handle.yaml

  # 终端2：
  ros2 service call /h2_arm/vision_command std_srvs/srv/Trigger
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from detector_msgs.msg import InspectionCommand, Object3DArray

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit('需要 PyYAML: pip3 install --user pyyaml') from exc


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('YAML root must be a mapping')
    return data


class VisionCommandBridge(Node):
    def __init__(self, yaml_path: Path, service_name: str, command_topic: str, objects_topic: str):
        super().__init__('h2_arm_vision_command_bridge')
        self.yaml_path = yaml_path
        self.command_topic = command_topic
        self.objects_topic = objects_topic
        self._cg = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._busy = False

        self._want = set()
        self._min_conf = 0.3
        self._need_frames = 3
        self._max_std = 0.005
        self._timeout = 20.0
        self._stable = {}  # class_name -> list[(x,y,z)]
        self._done_event = threading.Event()
        self._result = {}

        self.cmd_pub = self.create_publisher(InspectionCommand, command_topic, 10)
        self.create_subscription(
            Object3DArray,
            objects_topic,
            self._on_objects,
            10,
            callback_group=self._cg,
        )
        self.create_service(
            Trigger,
            service_name,
            self._handle_trigger,
            callback_group=self._cg,
        )
        self.create_timer(0.2, self._republish_command, callback_group=self._cg)
        self._active_cmd = None  # InspectionCommand or None

        self.get_logger().info(
            'ready service=%s yaml=%s → pub=%s sub=%s'
            % (service_name, yaml_path, command_topic, objects_topic)
        )

    def _republish_command(self):
        # 合同建议持续发同一 command_id，直到本轮 Trigger 结束
        with self._lock:
            msg = self._active_cmd
        if msg is not None:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.cmd_pub.publish(msg)

    def _handle_trigger(self, _req, res):
        with self._lock:
            if self._busy:
                res.success = False
                res.message = 'busy: previous vision_command still running'
                return res
            self._busy = True

        try:
            cfg = _load_yaml(self.yaml_path)
            cmd = self._build_command(cfg)
            want = [str(x).strip() for x in (cfg.get('requested_class_names') or []) if str(x).strip()]
            if not want:
                raise ValueError('requested_class_names empty')

            with self._lock:
                self._want = {n.lower() for n in want}
                self._min_conf = float(cfg.get('min_confidence', 0.3))
                self._need_frames = max(1, int(cfg.get('required_stable_frames', 3)))
                self._max_std = float(cfg.get('max_position_std_m', 0.005))
                self._timeout = float(cfg.get('timeout_sec', 20.0))
                self._stable = {n: [] for n in want}
                self._result = {}
                self._done_event.clear()
                self._active_cmd = cmd

            self.get_logger().info(
                'Trigger start id=%s classes=%s timeout=%.1fs'
                % (cmd.command_id, want, self._timeout)
            )
            # 立刻发一帧
            cmd.header.stamp = self.get_clock().now().to_msg()
            self.cmd_pub.publish(cmd)

            ok = self._done_event.wait(timeout=self._timeout)
            with self._lock:
                result = dict(self._result)
                self._active_cmd = None

            if not ok or not result:
                res.success = False
                res.message = json.dumps(
                    {
                        'ok': False,
                        'command_id': cmd.command_id,
                        'error': 'timeout_or_unstable',
                        'partial': result,
                    },
                    ensure_ascii=False,
                )
                return res

            payload = {
                'ok': True,
                'command_id': cmd.command_id,
                'stage_id': int(cmd.stage_id),
                'stage_name': cmd.stage_name,
                'objects': result,
            }
            res.success = True
            res.message = json.dumps(payload, ensure_ascii=False)
            self.get_logger().info('Trigger OK: %s' % res.message)
            return res
        except Exception as exc:
            res.success = False
            res.message = 'vision_command failed: %s' % exc
            self.get_logger().error(res.message)
            return res
        finally:
            with self._lock:
                self._busy = False
                self._active_cmd = None

    def _build_command(self, cfg: dict) -> InspectionCommand:
        prefix = str(cfg.get('command_id') or 'vision_cmd').strip() or 'vision_cmd'
        # C++ 约定：前缀 + 时间戳，保证每次服务调用唯一
        command_id = '%s_%d' % (prefix, int(time.time() * 1000))

        msg = InspectionCommand()
        msg.header.frame_id = 'pelvis'
        msg.command_id = command_id
        msg.stage_id = int(cfg.get('stage_id', 0))
        msg.stage_name = str(cfg.get('stage_name') or '')
        msg.requested_action = str(cfg.get('requested_action') or 'observe_targets')
        msg.requested_class_names = [
            str(x).strip() for x in (cfg.get('requested_class_names') or []) if str(x).strip()
        ]
        msg.active_target_class_name = str(cfg.get('active_target_class_name') or '')
        msg.requested_semantic_names = [
            str(x).strip() for x in (cfg.get('requested_semantic_names') or []) if str(x).strip()
        ]
        msg.active_target_semantic_name = str(cfg.get('active_target_semantic_name') or '')
        msg.selection_policy = str(cfg.get('selection_policy') or 'highest_confidence')
        msg.target_id = str(cfg.get('target_id') or '')
        msg.lock_target = bool(cfg.get('lock_target', False))
        msg.min_confidence = float(cfg.get('min_confidence', 0.3))
        msg.required_stable_frames = max(1, int(cfg.get('required_stable_frames', 3)))
        msg.max_position_std_m = float(cfg.get('max_position_std_m', 0.005))
        msg.timeout_sec = float(cfg.get('timeout_sec', 20.0))
        return msg

    def _on_objects(self, msg: Object3DArray):
        with self._lock:
            if not self._busy or not self._want:
                return
            want = set(self._want)
            min_conf = self._min_conf
            need = self._need_frames
            max_std = self._max_std
            stable = self._stable
            best = {}

            for obj in msg.objects:
                if not obj.valid:
                    continue
                name = str(obj.detection.class_name).strip()
                if name.lower() not in want:
                    continue
                if float(obj.detection.confidence) < min_conf:
                    continue
                p = obj.point_target
                xyz = (float(p.x), float(p.y), float(p.z))
                if any(math.isnan(v) or math.isinf(v) for v in xyz):
                    continue
                conf = float(obj.detection.confidence)
                prev = best.get(name)
                if prev is None or conf > prev[0]:
                    best[name] = (conf, xyz, str(obj.target_frame or ''), conf)

            for name, (_c, xyz, frame, conf) in best.items():
                # 按显示名归档（YAML 里的原始大小写优先）
                key = name
                for k in list(stable.keys()):
                    if k.lower() == name.lower():
                        key = k
                        break
                else:
                    stable[key] = []
                hist = stable[key]
                hist.append(xyz)
                if len(hist) > need:
                    del hist[0 : len(hist) - need]

            ready = {}
            for name, hist in stable.items():
                if len(hist) < need:
                    continue
                xs = [h[0] for h in hist]
                ys = [h[1] for h in hist]
                zs = [h[2] for h in hist]
                std = max(_std(xs), _std(ys), _std(zs))
                if std > max_std:
                    continue
                # 用最近一帧；附带均值
                x, y, z = hist[-1]
                meta = best.get(name) or next(
                    (best[k] for k in best if k.lower() == name.lower()),
                    None,
                )
                frame = meta[2] if meta else ''
                conf = meta[3] if meta else 0.0
                ready[name] = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'mean_xyz': [sum(xs) / need, sum(ys) / need, sum(zs) / need],
                    'std_m': std,
                    'frame': frame,
                    'confidence': conf,
                    'stable_frames': len(hist),
                }

            # 全部 requested 类都齐才算成功（观察锁+把手）
            if len(ready) >= len(stable) and stable:
                self._result = ready
                self._done_event.set()


def _std(vals):
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--yaml',
        default=str(
            Path.home()
            / 'MscapeTech/Foxy_ROS/examples/vision_command_observe_lock_handle.yaml'
        ),
    )
    ap.add_argument('--service', default='/h2_arm/vision_command')
    ap.add_argument('--command-topic', default='/robot/inspection_command')
    ap.add_argument('--objects-topic', default='/detector/objects')
    args = ap.parse_args()

    yaml_path = Path(args.yaml).expanduser()
    if not yaml_path.is_file():
        raise SystemExit('YAML not found: %s' % yaml_path)

    rclpy.init()
    node = VisionCommandBridge(
        yaml_path=yaml_path,
        service_name=args.service,
        command_topic=args.command_topic,
        objects_topic=args.objects_topic,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
