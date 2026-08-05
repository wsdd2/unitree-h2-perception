# -*- coding: utf-8 -*-
"""OpenCV ArUco marker detection for inspection perception.

Publishes detections as class_name ``aruco_tag_<ID>`` with pixel center and
bbox. 3D is produced by coordinate_projector via (cx, cy) depth (and optional
PnP when marker length is configured).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SOURCE_TAG = 'opencv_aruco'
CLASS_PREFIX = 'aruco_tag_'


def is_aruco_class(class_name: str) -> bool:
    name = str(class_name or '').strip().lower()
    return name.startswith('aruco_tag_') or name.startswith('aruco_')


def parse_aruco_id(class_name: str) -> Optional[int]:
    name = str(class_name or '').strip().lower()
    for prefix in ('aruco_tag_', 'aruco_'):
        if name.startswith(prefix):
            tail = name[len(prefix) :].strip()
            if tail.isdigit():
                return int(tail)
    return None


def _dictionary_from_name(name: str):
    key = str(name or 'DICT_6X6_250').strip()
    if not key.startswith('DICT_'):
        key = 'DICT_' + key
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('cv2.aruco is unavailable in this OpenCV build')
    if not hasattr(cv2.aruco, key):
        raise ValueError('Unknown ArUco dictionary: %s' % key)
    dict_id = getattr(cv2.aruco, key)
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def _make_detector(dictionary_name: str):
    dictionary = _dictionary_from_name(dictionary_name)
    if hasattr(cv2.aruco, 'DetectorParameters') and hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        def _detect(gray):
            return detector.detectMarkers(gray)

        return _detect

    params = cv2.aruco.DetectorParameters_create()

    def _detect(gray):
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    return _detect


def detect_aruco_markers(
    image_bgr: np.ndarray,
    dictionary_name: str = 'DICT_6X6_250',
    confidence: float = 0.99,
    min_side_px: float = 8.0,
) -> List[Dict[str, Any]]:
    """Detect ArUco markers and return Object2D-compatible detection dicts."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    if not hasattr(cv2, 'aruco'):
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        detect_fn = _make_detector(dictionary_name)
        corners, ids, _rejected = detect_fn(gray)
    except Exception:
        return []

    if ids is None or len(ids) == 0:
        return []

    detections = []
    for corner, marker_id in zip(corners, ids.flatten().tolist()):
        pts = np.asarray(corner, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 4:
            continue
        x_min = float(np.min(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        x_max = float(np.max(pts[:, 0]))
        y_max = float(np.max(pts[:, 1]))
        side = min(x_max - x_min, y_max - y_min)
        if side < float(min_side_px):
            continue
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        mid_id = int(marker_id)
        corner_list = [[float(x), float(y)] for x, y in pts[:4]]
        detections.append(
            {
                'class_name': '%s%d' % (CLASS_PREFIX, mid_id),
                'class_id': mid_id,
                'confidence': float(confidence),
                'xmin': int(round(x_min)),
                'ymin': int(round(y_min)),
                'xmax': int(round(x_max)),
                'ymax': int(round(y_max)),
                'cx': cx,
                'cy': cy,
                'geometric_center_px': [cx, cy],
                # Transport 4 corners without extending Object2D.msg.
                'handle_grasp_edge_px': corner_list,
                'handle_grasp_center_px': [cx, cy],
                'handle_grasp_source': SOURCE_TAG,
                'detection_source': SOURCE_TAG,
                'semantic_name': '%s%d' % (CLASS_PREFIX, mid_id),
                'control_id': '%s%d' % (CLASS_PREFIX, mid_id),
                'label_text': 'id=%d' % mid_id,
            }
        )
    return detections


def draw_aruco_overlays(image_bgr: np.ndarray, detections: Sequence[Dict[str, Any]]) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    color = (255, 0, 255)  # magenta
    for det in detections:
        if not is_aruco_class(det.get('class_name', '')):
            continue
        corners = det.get('handle_grasp_edge_px') or []
        if len(corners) >= 4:
            pts = np.asarray(corners[:4], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image_bgr, [pts], True, color, 2)
        else:
            cv2.rectangle(
                image_bgr,
                (int(det.get('xmin', 0)), int(det.get('ymin', 0))),
                (int(det.get('xmax', 0)), int(det.get('ymax', 0))),
                color,
                2,
            )
        cx = int(round(float(det.get('cx', 0.0))))
        cy = int(round(float(det.get('cy', 0.0))))
        cv2.drawMarker(
            image_bgr,
            (cx, cy),
            color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )
        label = str(det.get('class_name', 'aruco'))
        cv2.putText(
            image_bgr,
            label,
            (cx + 6, max(14, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return image_bgr


def aruco_object_points(marker_length_m: float) -> np.ndarray:
    """Square marker corners in marker frame (meters), matching OpenCV order."""
    half = 0.5 * float(marker_length_m)
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def solve_aruco_center_camera(
    corners_px: Sequence[Sequence[float]],
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    marker_length_m: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """Return (point_camera_xyz, depth_z) from PnP, or None."""
    if marker_length_m <= 1e-6:
        return None
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    obj_pts = aruco_object_points(marker_length_m)
    img_pts = pts[:4].astype(np.float64)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist = (
        np.zeros((4, 1), dtype=np.float64)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    )
    flags = cv2.SOLVEPNP_ITERATIVE
    if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE'):
        flags = cv2.SOLVEPNP_IPPE_SQUARE
    ok, _rvec, tvec = cv2.solvePnP(obj_pts, img_pts, k, dist, flags=flags)
    if not ok:
        return None
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    depth = float(tvec[2])
    if depth <= 1e-4:
        return None
    return tvec, depth
