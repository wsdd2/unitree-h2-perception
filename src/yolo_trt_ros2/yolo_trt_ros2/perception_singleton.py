# -*- coding: utf-8 -*-
"""Refuse starting a second inspection-perception stack on the same ROS domain.

Another session may already own RealSense / GPU / DDS topics. Call
assert_no_conflicting_perception() before creating camera/YOLO nodes.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Cmdline substrings that mean "inspection perception (or old split stack) is up".
PERCEPTION_CMDLINE_MARKERS: Tuple[str, ...] = (
    'integrated_perception_node',
    'yolo_trt_ros2.integrated_perception',
    'yolo_trt_ros2.yolo_detector',
    'yolo_detector_node',
    'direct_realsense_node',
    'yolo_trt_ros2.direct_realsense',
    'yolo_trt_ros2.coordinate_projector',
    'coordinate_projector_node',
    'yolo_trt_ros2.web_dashboard',
    'web_dashboard_node',
    'inspection_perception.launch',
    'cabinet_yoloseg_backend',
    'cabinet_perception_bringup',
)

# These own the RealSense; a second open fails even across Domain IDs.
CAMERA_OWNER_MARKERS: Tuple[str, ...] = (
    'integrated_perception_node',
    'yolo_trt_ros2.integrated_perception',
    'direct_realsense_node',
    'yolo_trt_ros2.direct_realsense',
)


class PerceptionAlreadyRunningError(RuntimeError):
    """Raised when another same-function perception stack is already active."""


def _decode(data: bytes) -> str:
    return data.decode('utf-8', errors='replace')


def _ancestor_pids(pid: Optional[int] = None) -> set:
    """Self + parents, so the launching shell / ros2 launch is not a 'peer'."""
    current = int(pid if pid is not None else os.getpid())
    seen = set()
    while current > 1 and current not in seen:
        seen.add(current)
        try:
            status = Path('/proc/%d/status' % current).read_text(encoding='utf-8', errors='replace')
        except OSError:
            break
        parent = None
        for line in status.splitlines():
            if line.startswith('PPid:'):
                parent = int(line.split()[1])
                break
        if parent is None:
            break
        current = parent
    return seen


def _read_proc_environ(pid: int) -> dict:
    try:
        raw = Path('/proc/%d/environ' % pid).read_bytes()
    except OSError:
        return {}
    env = {}
    for item in raw.split(b'\0'):
        if not item or b'=' not in item:
            continue
        key, value = item.split(b'=', 1)
        env[_decode(key)] = _decode(value)
    return env


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path('/proc/%d/cmdline' % pid).read_bytes()
    except OSError:
        return ''
    return _decode(raw.replace(b'\0', b' ')).strip()


def _domain_id_from_env(env: Optional[dict] = None) -> str:
    source = env if env is not None else os.environ
    value = str(source.get('ROS_DOMAIN_ID', '0')).strip()
    return value if value else '0'


def _matches_any(text: str, markers: Sequence[str]) -> Optional[str]:
    lower = text.lower()
    for marker in markers:
        if marker.lower() in lower:
            return marker
    return None


def _iter_other_pids(exclude: Iterable[int]) -> List[int]:
    excluded = {int(pid) for pid in exclude}
    pids = []
    proc = Path('/proc')
    if not proc.is_dir():
        return pids
    for entry in proc.iterdir():
        name = entry.name
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in excluded:
            continue
        pids.append(pid)
    return pids


def _port_in_use(port: int, host: str = '0.0.0.0') -> bool:
    if port <= 0:
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
        return False
    except OSError:
        return True
    finally:
        try:
            sock.close()
        except OSError:
            pass


def find_conflicting_perception(
    domain_id: Optional[str] = None,
    web_port: Optional[int] = None,
) -> List[dict]:
    """Return peer perception processes that would fight with this start."""
    my_domain = _domain_id_from_env() if domain_id is None else str(domain_id).strip() or '0'
    exclude = _ancestor_pids()
    conflicts = []

    for pid in _iter_other_pids(exclude):
        cmdline = _read_cmdline(pid)
        if not cmdline:
            continue
        marker = _matches_any(cmdline, PERCEPTION_CMDLINE_MARKERS)
        if marker is None:
            continue
        env = _read_proc_environ(pid)
        peer_domain = _domain_id_from_env(env) if env else '<unknown>'
        is_camera_owner = _matches_any(cmdline, CAMERA_OWNER_MARKERS) is not None
        same_domain = peer_domain == my_domain or peer_domain == '<unknown>'
        # Same domain → DDS/topic fight. Camera owner → device fight even if Domain differs.
        if not (same_domain or is_camera_owner):
            continue
        conflicts.append(
            {
                'pid': pid,
                'cmdline': cmdline[:300],
                'marker': marker,
                'ros_domain_id': peer_domain,
                'camera_owner': is_camera_owner,
                'reason': (
                    'same ROS_DOMAIN_ID=%s' % my_domain
                    if same_domain
                    else 'camera-owning process (Domain %s != %s)' % (peer_domain, my_domain)
                ),
            }
        )

    if web_port is not None and _port_in_use(int(web_port)):
        conflicts.append(
            {
                'pid': None,
                'cmdline': '<tcp listen>',
                'marker': 'web_port',
                'ros_domain_id': my_domain,
                'camera_owner': False,
                'reason': 'WebUI port %s already in use' % web_port,
            }
        )
    return conflicts


def format_conflict_report(
    conflicts: Sequence[dict],
    domain_id: Optional[str] = None,
) -> str:
    my_domain = _domain_id_from_env() if domain_id is None else str(domain_id).strip() or '0'
    lines = [
        'Refusing to start inspection perception: another same-function ROS2 stack is already running.',
        'This start would fight over RealSense / GPU / /detector/* topics.',
        'Local ROS_DOMAIN_ID=%s  ROS_LOCALHOST_ONLY=%s  pid=%s'
        % (
            my_domain,
            os.environ.get('ROS_LOCALHOST_ONLY', '<unset>'),
            os.getpid(),
        ),
        'Conflicting peers:',
    ]
    for item in conflicts:
        lines.append(
            '  - pid=%s domain=%s marker=%s reason=%s'
            % (
                item.get('pid'),
                item.get('ros_domain_id'),
                item.get('marker'),
                item.get('reason'),
            )
        )
        lines.append('    cmd: %s' % item.get('cmdline'))
    lines.extend(
        [
            'Fix:',
            '  1) Check who already started perception / ros2 launch on this machine.',
            '  2) Or stop the peer, then retry:',
            '       pkill -f integrated_perception_node || true',
            '       pkill -f inspection_perception || true',
            '  3) Debug override only (not for normal use): ALLOW_MULTI_PERCEPTION=1',
        ]
    )
    return '\n'.join(lines)


def assert_no_conflicting_perception(
    domain_id: Optional[str] = None,
    web_port: Optional[int] = None,
) -> None:
    """Raise PerceptionAlreadyRunningError if a conflicting stack is active."""
    override = str(os.environ.get('ALLOW_MULTI_PERCEPTION', '')).strip().lower()
    if override in ('1', 'true', 'yes', 'on'):
        return
    conflicts = find_conflicting_perception(domain_id=domain_id, web_port=web_port)
    if not conflicts:
        return
    raise PerceptionAlreadyRunningError(format_conflict_report(conflicts, domain_id=domain_id))
