#!/usr/bin/env python3
"""Select stable perception targets and command the exclusive H2 motion worker."""

import json
import socket
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from detector_msgs.msg import InspectionCommand, Object3DArray, RobotInspectionStatus


PRESS_CLASSES = (
    'red sticker push point',
    'red push button',
    'green push button',
    'yellow push button',
    'black push button',
    'lock point',
)
HANDLE_CLASSES = ('black cabinet door handle', 'door handle', 'handle')


class CabinetAutomationNode(Node):
    """Status-gated target selector; dry-run unless execute_enabled is true."""

    def __init__(self):
        super().__init__('cabinet_automation')
        self._declare_parameters()

        self.execute_enabled = bool(self.get_parameter('execute_enabled').value)
        self.worker_host = str(self.get_parameter('worker_host').value)
        self.worker_port = int(self.get_parameter('worker_port').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.stability_window = max(2, int(self.get_parameter('stability_window').value))
        self.stability_max_std_m = float(self.get_parameter('stability_max_std_m').value)
        self.target_max_age_sec = float(self.get_parameter('target_max_age_sec').value)
        self.press_approach_m = float(self.get_parameter('press_approach_m').value)
        self.press_hold_sec = float(self.get_parameter('press_hold_sec').value)
        self.handle_preapproach_m = float(self.get_parameter('handle_preapproach_m').value)
        self.move_duration_sec = float(self.get_parameter('move_duration_sec').value)
        self.expected_target_frame = str(self.get_parameter('expected_target_frame').value).strip()
        self.require_status_reachable = bool(self.get_parameter('require_status_reachable').value)
        self.status_max_age_sec = float(self.get_parameter('status_max_age_sec').value)

        objects_topic = str(self.get_parameter('objects_3d_topic').value)
        objects_ik_topic = str(self.get_parameter('objects_ik_topic').value)
        command_topic = str(self.get_parameter('inspection_command_topic').value)
        status_topic = str(self.get_parameter('robot_status_topic').value)
        result_topic = str(self.get_parameter('result_topic').value)

        self.status = None
        self.status_received_monotonic = 0.0
        self.command = None
        self.command_received_monotonic = 0.0
        self.latest_target = None
        self.target_key = None
        self.target_history = deque(maxlen=self.stability_window)
        self.target_received_monotonic = 0.0
        self.handle_geometries = []
        self.handle_geometry_received_monotonic = 0.0
        self.observation_tracks = {}
        self.dispatched_status_key = None
        self.active_command_id = ''
        self.command_sequence = 0
        self._last_observe_diag_monotonic = 0.0

        self.sock = None
        self.socket_buffer = b''
        self.last_connect_attempt = 0.0
        self.last_heartbeat = 0.0

        self.result_pub = self.create_publisher(String, result_topic, 10)
        self.create_subscription(Object3DArray, objects_topic, self._objects_callback, 10)
        self.create_subscription(String, objects_ik_topic, self._objects_ik_callback, 10)
        self.create_subscription(InspectionCommand, command_topic, self._command_callback, 10)
        self.create_subscription(RobotInspectionStatus, status_topic, self._status_callback, 10)
        self.create_timer(0.05, self._timer_callback)

        self.get_logger().warn(
            'Cabinet automation started in %s mode; objects=%s status=%s worker=%s:%d'
            % (
                'EXECUTE' if self.execute_enabled else 'DRY-RUN',
                objects_topic,
                status_topic,
                self.worker_host,
                self.worker_port,
            )
        )

    def _declare_parameters(self):
        self.declare_parameter('execute_enabled', False)
        self.declare_parameter('worker_host', '127.0.0.1')
        self.declare_parameter('worker_port', 8765)
        self.declare_parameter('objects_3d_topic', '/detector/objects_3d')
        self.declare_parameter('objects_ik_topic', '/detector/objects_ik_json')
        self.declare_parameter('inspection_command_topic', '/robot/inspection_command')
        self.declare_parameter('robot_status_topic', '/robot/inspection_status')
        self.declare_parameter('result_topic', '/robot/cabinet_action_result_json')
        self.declare_parameter('min_confidence', 0.20)
        self.declare_parameter('stability_window', 5)
        self.declare_parameter('stability_max_std_m', 0.003)
        self.declare_parameter('target_max_age_sec', 0.40)
        self.declare_parameter('press_approach_m', 0.06)
        self.declare_parameter('press_hold_sec', 0.35)
        self.declare_parameter('handle_preapproach_m', 0.10)
        self.declare_parameter('move_duration_sec', 2.0)
        self.declare_parameter('expected_target_frame', 'torso_link')
        self.declare_parameter('require_status_reachable', True)
        self.declare_parameter('status_max_age_sec', 3.0)

    def _command_callback(self, msg):
        action = str(msg.requested_action).strip().lower().replace('-', '_')
        command_id = str(msg.command_id).strip()
        if not command_id:
            self._publish_result({'event': 'command_rejected', 'reason': 'missing command_id'})
            return
        if action == 'abort':
            old_id = str(self.command.command_id) if self.command is not None else ''
            if command_id != old_id:
                self._send_abort('inspection_command abort id=%s' % command_id)
            self.command = msg
            self.command_received_monotonic = time.monotonic()
            return
        supported = {'observe_targets', 'move', 'press', 'grasp_rotate', 'door'}
        if action not in supported:
            self._publish_result(
                {
                    'event': 'command_rejected',
                    'command_id': command_id,
                    'reason': 'unsupported requested_action=%s' % action,
                }
            )
            return
        selection_policy = str(msg.selection_policy).strip().lower() or 'highest_confidence'
        if selection_policy != 'highest_confidence':
            self._publish_result(
                {
                    'event': 'command_rejected',
                    'command_id': command_id,
                    'reason': 'selection_policy_not_implemented=%s' % selection_policy,
                }
            )
            return
        requested = [
            str(value).strip().lower()
            for value in msg.requested_class_names
            if str(value).strip()
        ]
        requested_semantic = [
            str(value).strip().lower()
            for value in msg.requested_semantic_names
            if str(value).strip()
        ]
        active = str(msg.active_target_class_name).strip().lower()
        active_semantic = str(msg.active_target_semantic_name).strip().lower()
        effective_active = (
            active_semantic
            or active
            or (requested_semantic[0] if len(requested_semantic) == 1 else '')
            or (requested[0] if len(requested) == 1 else '')
        )
        if action == 'press' and not self._matches(effective_active, PRESS_CLASSES):
            self._publish_result(
                {
                    'event': 'command_rejected',
                    'command_id': command_id,
                    'reason': 'press requires lock point or push button class',
                }
            )
            return
        if action in {'grasp_rotate', 'door'} and not self._matches(effective_active, HANDLE_CLASSES):
            self._publish_result(
                {
                    'event': 'command_rejected',
                    'command_id': command_id,
                    'reason': 'door/grasp_rotate currently supports cabinet handle only',
                }
            )
            return
        old_id = str(self.command.command_id) if self.command is not None else ''
        self.command = msg
        self.command_received_monotonic = time.monotonic()
        if command_id != old_id:
            required = max(
                self.stability_window,
                int(msg.required_stable_frames) if int(msg.required_stable_frames) > 0 else 0,
            )
            self.target_history = deque(maxlen=required)
            self.target_key = None
            self.latest_target = None
            self.observation_tracks = {}
            self.dispatched_status_key = None
        if command_id != old_id:
            self._publish_result(
                {
                    'event': 'inspection_command_received',
                    'command_id': command_id,
                    'requested_action': action,
                    'requested_class_names': [str(value) for value in msg.requested_class_names],
                    'active_target_class_name': str(msg.active_target_class_name),
                    'requested_semantic_names': [str(value) for value in msg.requested_semantic_names],
                    'active_target_semantic_name': str(msg.active_target_semantic_name),
                }
            )

    def _status_callback(self, msg):
        old_key = self._status_key(self.status)
        self.status = msg
        self.status_received_monotonic = time.monotonic()
        new_key = self._status_key(msg)
        if new_key != old_key:
            self.target_key = None
            self.target_history.clear()
            self.latest_target = None
            if msg.stage_id in (0, 4, 5):
                self.dispatched_status_key = None
        if msg.emergency_stop or msg.has_error or msg.stage_id == 5:
            self._send_abort(
                'robot_status estop=%s has_error=%s stage=%d'
                % (msg.emergency_stop, msg.has_error, msg.stage_id)
            )

    @staticmethod
    def _status_key(status):
        if status is None:
            return None
        return (
            int(status.stage_id),
            str(status.stage_name).strip().lower(),
            str(status.current_action).strip().lower(),
            str(status.target_id).strip().lower(),
        )

    def _desired_policy(self):
        command = self.command
        if command is not None:
            action = str(command.requested_action).strip().lower().replace('-', '_')
            requested = tuple(
                str(value).strip().lower()
                for value in command.requested_class_names
                if str(value).strip()
            )
            requested_semantic = tuple(
                str(value).strip().lower()
                for value in command.requested_semantic_names
                if str(value).strip()
            )
            active = str(command.active_target_class_name).strip().lower()
            active_semantic = str(command.active_target_semantic_name).strip().lower()
            if action == 'observe_targets':
                return {
                    'skill': 'observe',
                    'patterns': requested_semantic or requested or PRESS_CLASSES + HANDLE_CLASSES,
                }
            if active_semantic:
                patterns = (active_semantic,)
            elif active:
                patterns = (active,)
            elif len(requested_semantic) == 1:
                patterns = requested_semantic
            elif len(requested) == 1:
                patterns = requested
            else:
                return None
            skill = {
                'press': 'press',
                'move': 'move',
                'grasp_rotate': 'door',
                'door': 'door',
            }.get(action)
            if skill is None:
                return None
            return {'skill': skill, 'patterns': patterns}

        status = self.status
        if status is None:
            return None
        action = '%s %s %s' % (status.stage_name, status.current_action, status.target_id)
        action = action.lower().replace('_', ' ')

        for class_name in PRESS_CLASSES:
            if class_name in action:
                return {'skill': 'press', 'patterns': (class_name,)}
        if 'handle' in action:
            skill = 'move' if status.stage_id == 1 or 'front' in action or 'approach' in action else 'door'
            return {'skill': skill, 'patterns': HANDLE_CLASSES}
        if status.stage_id == 1:
            return {'skill': 'move', 'patterns': HANDLE_CLASSES}
        if status.stage_id == 2:
            return {'skill': 'press', 'patterns': PRESS_CLASSES}
        if status.stage_id == 3:
            return {'skill': 'door', 'patterns': HANDLE_CLASSES}
        return None

    @staticmethod
    def _matches(class_name, patterns):
        name = class_name.lower()
        return any(pattern in name for pattern in patterns)

    def _objects_ik_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        geometries = []
        for obj in payload.get('objects', []):
            class_name = str(obj.get('class_name', ''))
            if not self._matches(class_name, HANDLE_CLASSES):
                continue
            edge_px = obj.get('handle_grasp_edge_px') or []
            endpoints = obj.get('handle_grasp_endpoint_targets_m') or []
            center = obj.get('handle_grasp_ree_target_m') or []
            if len(edge_px) != 2 or len(endpoints) != 2 or len(center) != 3:
                continue
            pairs = []
            for pixel, target in zip(edge_px, endpoints):
                if len(pixel) != 2 or len(target) != 3:
                    pairs = []
                    break
                pairs.append((pixel, target))
            if len(pairs) != 2:
                continue
            pairs.sort(key=lambda item: float(item[0][0]))
            bbox = obj.get('bbox_xyxy') or [0, 0, 0, 0]
            geometries.append(
                {
                    'class_name': class_name,
                    'center_px': obj.get('center_px') or [
                        0.5 * (float(bbox[0]) + float(bbox[2])),
                        0.5 * (float(bbox[1]) + float(bbox[3])),
                    ],
                    'center_target_m': [float(value) for value in center],
                    'image_left_target_m': [float(value) for value in pairs[0][1]],
                    'image_right_target_m': [float(value) for value in pairs[1][1]],
                    'image_left_px': [float(value) for value in pairs[0][0]],
                    'image_right_px': [float(value) for value in pairs[1][0]],
                    'width_m': float(obj.get('handle_grasp_width_m', 0.0)),
                }
            )
        if geometries:
            self.handle_geometries = geometries
            self.handle_geometry_received_monotonic = time.monotonic()

    def _objects_callback(self, msg):
        policy = self._desired_policy()
        if policy is None:
            return
        min_confidence = self.min_confidence
        if self.command is not None and float(self.command.min_confidence) > 0.0:
            min_confidence = float(self.command.min_confidence)
        candidates = []
        for obj in msg.objects:
            if not obj.valid:
                continue
            if float(obj.detection.confidence) < min_confidence:
                continue
            identity = '%s %s' % (
                str(obj.detection.class_name),
                str(getattr(obj.detection, 'semantic_name', '')),
            )
            if not self._matches(identity, policy['patterns']):
                continue
            if self.expected_target_frame and str(obj.target_frame) != self.expected_target_frame:
                continue
            xyz = np.array(
                [obj.point_target.x, obj.point_target.y, obj.point_target.z],
                dtype=float,
            )
            if not np.all(np.isfinite(xyz)):
                continue
            candidates.append((float(obj.detection.confidence), obj, xyz))
        if not candidates:
            return

        if policy['skill'] == 'observe':
            self._update_observation_tracks(candidates)
            return

        _confidence, obj, xyz = max(candidates, key=lambda item: item[0])
        key = (
            str(obj.detection.class_name).lower(),
            int(round(obj.detection.cx / 8.0)),
            int(round(obj.detection.cy / 8.0)),
        )
        if key != self.target_key:
            self.target_key = key
            self.target_history.clear()
        self.target_history.append(xyz)
        self.latest_target = {
            'xyz': xyz,
            'class_name': str(obj.detection.class_name),
            'label_text': str(getattr(obj.detection, 'label_text', '')),
            'label_confidence': float(getattr(obj.detection, 'label_confidence', 0.0)),
            'semantic_name': str(getattr(obj.detection, 'semantic_name', '')),
            'control_id': str(getattr(obj.detection, 'control_id', '')),
            'spatial_relation': str(getattr(obj.detection, 'spatial_relation', '')),
            'label_tag_present': bool(getattr(obj.detection, 'label_tag_present', False)),
            'confidence': float(obj.detection.confidence),
            'target_frame': str(obj.target_frame),
            'center_px': [float(obj.detection.cx), float(obj.detection.cy)],
            'policy': policy,
        }
        self.target_received_monotonic = time.monotonic()

    def _required_stable_frames(self):
        if self.command is not None and int(self.command.required_stable_frames) > 0:
            return int(self.command.required_stable_frames)
        return self.stability_window

    def _allowed_position_std_m(self):
        if self.command is not None and float(self.command.max_position_std_m) > 0.0:
            return float(self.command.max_position_std_m)
        return self.stability_max_std_m

    def _update_observation_tracks(self, candidates):
        required = self._required_stable_frames()
        now = time.monotonic()
        # Observe: one track per class_name (highest-confidence instance each frame).
        # Pixel-bin keys fragment when the bbox center jitters by a few pixels.
        best_by_class = {}
        for confidence, obj, xyz in candidates:
            class_name = str(obj.detection.class_name).lower()
            prev = best_by_class.get(class_name)
            if prev is None or confidence > prev[0]:
                best_by_class[class_name] = (confidence, obj, xyz)
        for class_name, (_confidence, obj, xyz) in best_by_class.items():
            key = class_name
            track = self.observation_tracks.get(key)
            if track is None:
                track = {'points': deque(maxlen=max(required, 3))}
                self.observation_tracks[key] = track
            if track['points'].maxlen != max(required, 3):
                track['points'] = deque(track['points'], maxlen=max(required, 3))
            track['points'].append(xyz)
            track['latest'] = {
                'class_name': str(obj.detection.class_name),
                'label_text': str(getattr(obj.detection, 'label_text', '')),
                'label_confidence': float(getattr(obj.detection, 'label_confidence', 0.0)),
                'semantic_name': str(getattr(obj.detection, 'semantic_name', '')),
                'control_id': str(getattr(obj.detection, 'control_id', '')),
                'spatial_relation': str(getattr(obj.detection, 'spatial_relation', '')),
                'label_tag_present': bool(getattr(obj.detection, 'label_tag_present', False)),
                'confidence': float(obj.detection.confidence),
                'target_frame': str(obj.target_frame),
                'center_px': [float(obj.detection.cx), float(obj.detection.cy)],
            }
            track['received_monotonic'] = now

    def _observe_track_max_age_sec(self):
        # Camera runs ~6Hz; allow a few missed frames before declaring stale.
        return max(float(self.target_max_age_sec), 1.5)

    def _stable_observations(self):
        command = self.command
        if command is None:
            return None
        requested = [
            str(value).strip().lower()
            for value in (
                command.requested_semantic_names
                if command.requested_semantic_names
                else command.requested_class_names
            )
            if str(value).strip()
        ]
        if not requested:
            return None
        required = self._required_stable_frames()
        allowed_std = self._allowed_position_std_m()
        max_age = self._observe_track_max_age_sec()
        results = []
        for requested_class in requested:
            matches = []
            for track in self.observation_tracks.values():
                latest = track.get('latest') or {}
                identity = '%s %s' % (
                    str(latest.get('class_name', '')),
                    str(latest.get('semantic_name', '')),
                )
                if requested_class not in identity.lower():
                    continue
                if time.monotonic() - float(track.get('received_monotonic', 0.0)) > max_age:
                    continue
                if len(track['points']) < required:
                    continue
                points = np.stack(tuple(track['points']), axis=0)
                std_xyz = np.std(points, axis=0)
                if float(np.max(std_xyz)) > allowed_std:
                    continue
                matches.append((float(latest.get('confidence', 0.0)), track, points, std_xyz))
            if not matches:
                return None
            _confidence, track, points, std_xyz = max(matches, key=lambda item: item[0])
            latest = track['latest']
            item = {
                'class_name': latest['class_name'],
                'label_text': latest.get('label_text', ''),
                'label_confidence': latest.get('label_confidence', 0.0),
                'semantic_name': latest.get('semantic_name', ''),
                'control_id': latest.get('control_id', ''),
                'spatial_relation': latest.get('spatial_relation', ''),
                'label_tag_present': latest.get('label_tag_present', False),
                'confidence': latest['confidence'],
                'target_frame': latest['target_frame'],
                'point_target_m': [float(value) for value in np.mean(points, axis=0)],
                'std_xyz_m': [float(value) for value in std_xyz],
            }
            if self._matches(latest['class_name'], PRESS_CLASSES):
                item['force_point_target_m'] = list(item['point_target_m'])
            if self._matches(latest['class_name'], HANDLE_CLASSES):
                geometry = self._matching_handle_geometry(latest)
                if geometry is not None:
                    item.update(
                        {
                            'handle_image_left_target_m': geometry['image_left_target_m'],
                            'handle_image_right_target_m': geometry['image_right_target_m'],
                            'handle_center_target_m': geometry['center_target_m'],
                            'handle_width_m': geometry['width_m'],
                        }
                    )
                else:
                    # Observe-only: still report center; left/right edges may come later.
                    item['handle_center_target_m'] = list(item['point_target_m'])
                    item['handle_geometry_missing'] = True
            results.append(item)
        return results

    def _observe_blocker_report(self):
        """Explain why observe_targets has not emitted inspection_observation yet."""
        command = self.command
        if command is None:
            return {'reason': 'no_command'}
        if self.status is None:
            return {'reason': 'no_inspection_status'}
        now = time.monotonic()
        if now - self.status_received_monotonic > self.status_max_age_sec:
            return {'reason': 'robot_status_stale'}
        if int(command.stage_id) >= 0 and int(command.stage_id) != int(self.status.stage_id):
            return {
                'reason': 'stage_id_mismatch',
                'command_stage_id': int(command.stage_id),
                'status_stage_id': int(self.status.stage_id),
            }
        requested = [
            str(value).strip().lower()
            for value in (
                command.requested_semantic_names
                if command.requested_semantic_names
                else command.requested_class_names
            )
            if str(value).strip()
        ]
        required = self._required_stable_frames()
        allowed_std = self._allowed_position_std_m()
        max_age = self._observe_track_max_age_sec()
        classes = []
        for requested_class in requested:
            best = {
                'requested': requested_class,
                'track_count': 0,
                'best_frames': 0,
                'best_age_sec': None,
                'best_std_m': None,
                'frame_ok': False,
            }
            for track in self.observation_tracks.values():
                latest = track.get('latest') or {}
                identity = '%s %s' % (
                    str(latest.get('class_name', '')),
                    str(latest.get('semantic_name', '')),
                )
                if requested_class not in identity.lower():
                    continue
                best['track_count'] += 1
                frames = len(track['points'])
                age = now - float(track.get('received_monotonic', 0.0))
                std_xyz = None
                if frames > 0:
                    points = np.stack(tuple(track['points']), axis=0)
                    std_xyz = float(np.max(np.std(points, axis=0)))
                if frames >= best['best_frames']:
                    best['best_frames'] = frames
                    best['best_age_sec'] = round(age, 3)
                    best['best_std_m'] = None if std_xyz is None else round(std_xyz, 4)
                    best['frame_ok'] = (
                        frames >= required
                        and age <= max_age
                        and (std_xyz is None or std_xyz <= allowed_std)
                    )
            if best['track_count'] == 0:
                best['blocker'] = (
                    'no_3d_track_for_class (web 2D!=automation 3D; check target_frame=%s conf/depth)'
                    % self.expected_target_frame
                )
            elif best['best_frames'] < required:
                best['blocker'] = 'need_%d_stable_frames_have_%d' % (required, best['best_frames'])
            elif best['best_age_sec'] is not None and best['best_age_sec'] > max_age:
                best['blocker'] = 'track_stale'
            elif best['best_std_m'] is not None and best['best_std_m'] > allowed_std:
                best['blocker'] = 'position_std_too_high'
            else:
                best['blocker'] = ''
            classes.append(best)
        return {
            'reason': 'waiting_stable_observations',
            'expected_target_frame': self.expected_target_frame,
            'required_stable_frames': required,
            'max_position_std_m': allowed_std,
            'observe_track_max_age_sec': max_age,
            'classes': classes,
        }

    def _stable_target(self):
        required = self._required_stable_frames()
        if self.latest_target is None or len(self.target_history) < required:
            return None
        if time.monotonic() - self.target_received_monotonic > self.target_max_age_sec:
            return None
        points = np.stack(tuple(self.target_history), axis=0)
        std_xyz = np.std(points, axis=0)
        if float(np.max(std_xyz)) > self._allowed_position_std_m():
            return None
        target = dict(self.latest_target)
        target['xyz'] = np.mean(points, axis=0)
        target['std_xyz'] = std_xyz
        if target['policy']['skill'] in {'move', 'door'}:
            geometry = self._matching_handle_geometry(target)
            if geometry is None:
                return None
            target['handle_geometry'] = geometry
        return target

    def _matching_handle_geometry(self, target):
        if time.monotonic() - self.handle_geometry_received_monotonic > self.target_max_age_sec:
            return None
        center = np.asarray(target.get('center_px', []), dtype=float)
        if center.shape != (2,) or not self.handle_geometries:
            return None
        return min(
            self.handle_geometries,
            key=lambda geometry: float(
                np.linalg.norm(np.asarray(geometry['center_px'], dtype=float) - center)
            ),
        )

    def _connect_worker(self):
        if not self.execute_enabled or self.sock is not None:
            return
        now = time.monotonic()
        if now - self.last_connect_attempt < 1.0:
            return
        self.last_connect_attempt = now
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        candidate.settimeout(0.2)
        try:
            candidate.connect((self.worker_host, self.worker_port))
        except OSError:
            candidate.close()
            return
        candidate.setblocking(False)
        self.sock = candidate
        self.socket_buffer = b''
        self.last_heartbeat = 0.0
        self.get_logger().info('Connected to H2 cabinet motion worker.')

    def _disconnect_worker(self, reason):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.socket_buffer = b''
        if self.active_command_id:
            self._publish_result(
                {
                    'event': 'worker_disconnected',
                    'command_id': self.active_command_id,
                    'reason': reason,
                }
            )
            self.active_command_id = ''

    def _send_json(self, payload):
        if self.sock is None:
            return False
        try:
            self.sock.sendall(
                json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n'
            )
            return True
        except OSError as exc:
            self._disconnect_worker(str(exc))
            return False

    def _send_abort(self, reason):
        if self.active_command_id and self.sock is not None:
            self._send_json({'type': 'abort', 'reason': reason})

    def _poll_worker(self):
        if self.sock is None:
            return
        try:
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    self._disconnect_worker('eof')
                    return
                self.socket_buffer += chunk
        except BlockingIOError:
            pass
        except OSError as exc:
            self._disconnect_worker(str(exc))
            return

        while b'\n' in self.socket_buffer:
            raw, self.socket_buffer = self.socket_buffer.split(b'\n', 1)
            if not raw.strip():
                continue
            try:
                event = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            self._publish_result(event)
            event_name = str(event.get('event', ''))
            command_id = str(event.get('command_id', ''))
            if command_id == self.active_command_id and event_name in {
                'completed',
                'aborted',
                'rejected',
            }:
                self.active_command_id = ''

    def _publish_result(self, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.result_pub.publish(msg)
        event = str(payload.get('event', ''))
        if event in {'aborted', 'rejected', 'worker_disconnected'}:
            self.get_logger().error(msg.data)
        else:
            self.get_logger().info(msg.data)

    def _build_command(self, target):
        self.command_sequence += 1
        status_key = self._status_key(self.status)
        skill = target['policy']['skill']
        xyz = np.array(target['xyz'], dtype=float)
        geometry = target.get('handle_geometry')
        if geometry is not None:
            xyz = np.asarray(geometry['center_target_m'], dtype=float)
        if skill == 'move':
            xyz = xyz + np.array([-self.handle_preapproach_m, 0.0, 0.0])
        source = {
            'status_key': list(status_key),
            'class_name': target['class_name'],
            'label_text': target.get('label_text', ''),
            'label_confidence': target.get('label_confidence', 0.0),
            'semantic_name': target.get('semantic_name', ''),
            'control_id': target.get('control_id', ''),
            'spatial_relation': target.get('spatial_relation', ''),
            'label_tag_present': target.get('label_tag_present', False),
            'confidence': target['confidence'],
            'target_frame': target['target_frame'],
            'force_point_target_m': [float(value) for value in target['xyz']],
            'std_xyz_m': [float(value) for value in target['std_xyz']],
        }
        if geometry is not None:
            source['handle_image_left_target_m'] = geometry['image_left_target_m']
            source['handle_image_right_target_m'] = geometry['image_right_target_m']
            source['handle_center_target_m'] = geometry['center_target_m']
            source['handle_width_m'] = geometry['width_m']
        return {
            'type': 'execute',
            'command_id': 'stage%d-%06d' % (int(self.status.stage_id), self.command_sequence),
            'skill': skill,
            'xyz': [float(value) for value in xyz],
            'move_duration': self.move_duration_sec,
            'approach_m': self.press_approach_m,
            'hold_sec': self.press_hold_sec,
            'source': source,
        }

    def _timer_callback(self):
        self._connect_worker()
        self._poll_worker()
        now = time.monotonic()
        if self.sock is not None and now - self.last_heartbeat >= 0.5:
            if self._send_json({'type': 'heartbeat', 'time': time.time()}):
                self.last_heartbeat = now

        status = self.status
        if status is None:
            return
        command = self.command
        if command is not None:
            timeout_sec = float(command.timeout_sec)
            if timeout_sec > 0.0 and now - self.command_received_monotonic > timeout_sec:
                self._send_abort('inspection_command_timeout')
                return
            if int(command.stage_id) >= 0 and int(command.stage_id) != int(status.stage_id):
                return
        if now - self.status_received_monotonic > self.status_max_age_sec:
            self._send_abort('robot_status_stale')
            return
        if status.emergency_stop or status.has_error or status.motion_active:
            return
        if self.require_status_reachable and not status.target_reachable:
            return
        status_key = self._status_key(status)
        dispatch_key = (
            ('command', str(command.command_id))
            if command is not None
            else ('status',) + status_key
        )
        if dispatch_key == self.dispatched_status_key or self.active_command_id:
            return
        if command is not None and str(command.requested_action).strip().lower().replace('-', '_') == 'observe_targets':
            observations = self._stable_observations()
            if observations is None:
                if now - self._last_observe_diag_monotonic >= 2.0:
                    self._last_observe_diag_monotonic = now
                    report = self._observe_blocker_report()
                    report.update(
                        {
                            'event': 'inspection_observation_pending',
                            'command_id': str(command.command_id),
                            'stage_id': int(command.stage_id),
                        }
                    )
                    self._publish_result(report)
                    self.get_logger().warn('observe_targets pending: %s' % json.dumps(report, ensure_ascii=False))
                return
            self.dispatched_status_key = dispatch_key
            self._publish_result(
                {
                    'event': 'inspection_observation',
                    'command_id': str(command.command_id),
                    'stage_id': int(command.stage_id),
                    'objects': observations,
                }
            )
            return
        target = self._stable_target()
        if target is None:
            return
        command = self._build_command(target)

        if not self.execute_enabled:
            self.dispatched_status_key = dispatch_key
            command['event'] = 'dry_run_command'
            self._publish_result(command)
            return
        if self.sock is None:
            self.get_logger().error('Stable target is ready but motion worker is not connected.')
            return
        if self._send_json(command):
            self.dispatched_status_key = dispatch_key
            self.active_command_id = command['command_id']
            self._publish_result({'event': 'command_sent', **command})

    def destroy_node(self):
        self._send_abort('automation_node_shutdown')
        if self.sock is not None:
            self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CabinetAutomationNode()
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
