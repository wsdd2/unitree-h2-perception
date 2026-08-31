# -*- coding: utf-8 -*-
"""Closed-set YOLO-seg priority detector for cabinet controls / handle / hang-tag.

When robot InspectionCommand.requested_class_names overlap these families,
prefer smoothed mask targets over open-vocab YOLOE / OpenCV fallbacks.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

# Train-time class names from Yolo_Training/yolo26seg/train_yolo12_seg.py
DEFAULT_SEG_CLASS_NAMES = (
    'push button',
    'rotary selector switch',
    'rotary multiple selector switch',
    'toggle switch',
    'cabinet door handle',
    'lock point',
    'work tag',
    'hang cord',
    'hook',
)

# Map seg class -> publish name used by automation / stage contract.
PUBLISH_NAME = {
    'push button': 'push button',
    'rotary selector switch': 'black rotary selector switch',
    'rotary multiple selector switch': 'black rotary multiple selector switch',
    'toggle switch': 'toggle switch',
    'cabinet door handle': 'black cabinet door handle',
    'lock point': 'lock point',
    'work tag': 'green work tag',
    'hang cord': 'red hang cord',
    'hook': 'cabinet hang hook',
}

# requested_class_names tokens that activate each seg family.
# Keep rotary vs rotary-multiple aliases disjoint (no shared "selector switch").
REQUEST_ALIASES = {
    'push button': ('push button', 'button'),
    'rotary selector switch': (
        'black rotary selector switch',
        'rotary selector switch',
        'rotary selector',
    ),
    'rotary multiple selector switch': (
        'black rotary multiple selector switch',
        'rotary multiple selector switch',
        'rotary multiple',
        'multi selector',
    ),
    'toggle switch': ('toggle switch', 'rocker switch', 'toggle'),
    'cabinet door handle': ('cabinet door handle', 'door handle', 'handle'),
    'lock point': ('lock point', 'push point', 'sticker push'),
    'work tag': ('work tag', 'work here', 'safety tag'),
    'hang cord': ('hang cord', 'hanging cord', 'cord'),
    'hook': ('hang hook', 'cabinet hang hook', 'hook'),
}

SOURCE_TAG = 'yoloseg_mask'

# SEG classes whose publish name gets an OpenCV color prefix (red/green/yellow only).
COLORIZED_SEG_CLASSES = frozenset({'push button', 'toggle switch'})
SUPPORTED_COLORS = ('red', 'green', 'yellow')

# BGR overlay colors for debug preview.
OVERLAY_BGR = {
    'red': (0, 0, 255),
    'green': (0, 200, 0),
    'yellow': (0, 220, 255),
}


def normalize_name(name: str) -> str:
    return str(name or '').strip().lower().replace('_', ' ').replace('-', ' ')


def _color_mask_hsv(hsv: np.ndarray, color: str) -> np.ndarray:
    """Binary mask for one of red / green / yellow in HSV."""
    if color == 'red':
        low = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        high = cv2.inRange(hsv, (170, 70, 50), (179, 255, 255))
        return cv2.bitwise_or(low, high)
    if color == 'green':
        return cv2.inRange(hsv, (35, 50, 40), (95, 255, 255))
    if color == 'yellow':
        return cv2.inRange(hsv, (15, 70, 60), (35, 255, 255))
    return np.zeros(hsv.shape[:2], dtype=np.uint8)


def classify_mask_color(bgr_image, mask, min_ratio=0.08, min_pixels=24):
    """Pick red/green/yellow from YOLO-seg mask pixels via OpenCV HSV.

    Returns one of SUPPORTED_COLORS, or None when evidence is weak / ambiguous.
    """
    if bgr_image is None or mask is None or bgr_image.size == 0 or mask.size == 0:
        return None
    if bgr_image.shape[:2] != mask.shape[:2]:
        return None

    region = mask > 0
    # Shrink slightly so gray panel border does not dilute button color.
    if int(np.count_nonzero(region)) >= 64:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode((region.astype(np.uint8) * 255), kernel, iterations=1)
        if int(np.count_nonzero(eroded)) >= min_pixels:
            region = eroded > 0

    pixel_count = int(np.count_nonzero(region))
    if pixel_count < min_pixels:
        return None

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    scores = {}
    for color in SUPPORTED_COLORS:
        color_px = _color_mask_hsv(hsv, color) > 0
        scores[color] = float(np.count_nonzero(color_px & region)) / float(pixel_count)

    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]
    if best_score < float(min_ratio):
        return None
    # Require a clear winner so mixed specular reflections do not flip labels.
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) >= 2 and ranked[0] < ranked[1] * 1.25 + 0.02:
        return None
    return best_color


def colored_publish_name(seg_class: str, color: Optional[str]) -> str:
    seg_n = normalize_name(seg_class)
    if color in SUPPORTED_COLORS and seg_n in COLORIZED_SEG_CLASSES:
        return '%s %s' % (color, seg_n)
    return PUBLISH_NAME.get(seg_n, seg_class)


def available_publish_class_names(
    seg_classes: Optional[Sequence[str]] = None,
    opencv_color_enabled: bool = True,
) -> list:
    """Publish-side class names the SEG path may emit (one-shot listing)."""
    classes = list(seg_classes) if seg_classes else list(DEFAULT_SEG_CLASS_NAMES)
    names = []
    seen = set()
    for seg in classes:
        seg_n = normalize_name(seg)
        if not seg_n:
            continue
        base = PUBLISH_NAME.get(seg_n, seg_n)
        candidates = []
        if opencv_color_enabled and seg_n in COLORIZED_SEG_CLASSES:
            for color in SUPPORTED_COLORS:
                candidates.append('%s %s' % (color, seg_n))
            # Color evidence can be weak → still publish the uncolored base name.
            candidates.append(base)
        else:
            candidates.append(base)
        for name in candidates:
            key = normalize_name(name)
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
    return names


def requested_activates_seg(requested: Sequence[str], seg_class: str) -> bool:
    req = [normalize_name(item) for item in requested if str(item).strip()]
    if not req:
        return False
    aliases = REQUEST_ALIASES.get(seg_class, (seg_class,))
    for alias in aliases:
        alias_n = normalize_name(alias)
        for item in req:
            if alias_n in item or item in alias_n:
                return True
    return False


def detection_matches_active_request(
    requested: Sequence[str],
    seg_class: str,
    publish_name: str,
    opencv_color: Optional[str] = None,
) -> bool:
    """Match publish-side request names against model seg families + color.

    Requests look like ``green push button``; YOLO classes are ``push button``.
    Exact-string filters against seg_name drop every detection.
    """
    req = [str(item).strip() for item in requested if str(item).strip()]
    if not req:
        return True
    seg_n = normalize_name(seg_class)
    family_matches = [
        item for item in req if requested_activates_seg([item], seg_n)
    ]
    if not family_matches:
        return False
    color_constrained = []
    for item in family_matches:
        item_n = normalize_name(item)
        colors = [c for c in SUPPORTED_COLORS if c in item_n]
        if colors:
            color_constrained.append(colors)
    if not color_constrained:
        return True
    pub_n = normalize_name(publish_name)
    for colors in color_constrained:
        for color in colors:
            if color in pub_n:
                return True
            if opencv_color is not None and color == opencv_color:
                return True
    return False


def any_requested_covered(requested: Sequence[str], covered_classes: Sequence[str]) -> bool:
    for seg_class in covered_classes:
        if requested_activates_seg(requested, seg_class):
            return True
    return False


def detection_family_key(class_name: str) -> Optional[str]:
    name = normalize_name(class_name)
    if 'lock point' in name or 'push point' in name:
        return 'lock point'
    if 'handle' in name or 'lever' in name:
        return 'cabinet door handle'
    if 'hang hook' in name or name.endswith('hook') or name == 'hook':
        return 'hook'
    if 'hang cord' in name or 'hanging cord' in name or name == 'cord':
        return 'hang cord'
    if 'work tag' in name or 'work here' in name:
        return 'work tag'
    if 'push button' in name or name == 'button':
        return 'push button'
    if 'rotary multiple' in name or 'multiple selector' in name:
        return 'rotary multiple selector switch'
    if 'rotary' in name or 'selector' in name:
        return 'rotary selector switch'
    if 'toggle' in name or 'rocker' in name:
        return 'toggle switch'
    return None


class YoloSegPriorityBackend(object):
    """Lazy Ultralytics YOLO-seg runner with mask smoothing and target picking."""

    def __init__(
        self,
        model_path,
        class_names=None,
        conf_thres=0.25,
        iou_thres=0.45,
        imgsz=1280,
        device='',
        mask_smooth_ksize=5,
        mask_close_iters=1,
        lock_ground_ratio=0.80,
        enable_opencv_color=True,
        logger=None,
    ):
        if not model_path:
            raise ValueError('yolo_seg_model_path is required')
        resolved = os.path.expanduser(str(model_path))
        if not os.path.isfile(resolved):
            raise FileNotFoundError('yolo_seg_model_path does not exist: %s' % resolved)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'Failed to import ultralytics for YOLO-seg priority backend'
            ) from exc

        self.model_path = resolved
        self.class_names = list(class_names) if class_names else list(DEFAULT_SEG_CLASS_NAMES)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        if isinstance(imgsz, (list, tuple)):
            self.imgsz = [int(imgsz[0]), int(imgsz[1])]
        else:
            self.imgsz = int(imgsz)
        self.device = device
        self.mask_smooth_ksize = max(1, int(mask_smooth_ksize) | 1)
        self.mask_close_iters = max(0, int(mask_close_iters))
        self.lock_ground_ratio = float(np.clip(lock_ground_ratio, 0.0, 1.0))
        self.enable_opencv_color = bool(enable_opencv_color)
        self._logger = logger
        self.model = YOLO(resolved)

    def infer(self, bgr_image, active_seg_classes=None, down_uv=None):
        if bgr_image is None or bgr_image.size == 0:
            return []
        results = self.model.predict(
            source=bgr_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device or None,
            verbose=False,
            retina_masks=True,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names or {}
        masks = getattr(result, 'masks', None)
        img_h, img_w = bgr_image.shape[:2]
        down = self._normalize_down(down_uv)

        detections = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            raw_name = str(names.get(cls_id, self._class_name(cls_id)))
            seg_name = normalize_name(raw_name)
            if active_seg_classes is not None and not requested_activates_seg(
                active_seg_classes, seg_name
            ):
                continue

            xyxy = boxes.xyxy[i].cpu().numpy().astype(float)
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w - 1, x2))
            y2 = max(0, min(img_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            mask = self._extract_mask(masks, i, img_h, img_w, x1, y1, x2, y2)
            smoothed = self._smooth_mask(mask)
            target = self._pick_target(smoothed, seg_name, down)
            if target is None:
                continue
            cx, cy, extras = target
            opencv_color = None
            if self.enable_opencv_color and seg_name in COLORIZED_SEG_CLASSES:
                opencv_color = classify_mask_color(bgr_image, smoothed)
            publish_name = colored_publish_name(seg_name, opencv_color)
            if active_seg_classes is not None and not detection_matches_active_request(
                active_seg_classes, seg_name, publish_name, opencv_color
            ):
                continue
            det = {
                'class_name': publish_name,
                'class_id': cls_id,
                'confidence': conf,
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'cx': float(cx),
                'cy': float(cy),
                'handle_grasp_source': SOURCE_TAG,
                'detection_source': SOURCE_TAG,
                'seg_class_name': seg_name,
                'mask_contour': extras.get('contour'),
            }
            if opencv_color is not None:
                det['opencv_color'] = opencv_color
            det.update(extras.get('fields', {}))
            detections.append(det)
        return detections

    def _class_name(self, class_id):
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def _normalize_down(self, down_uv):
        if down_uv is None:
            return np.array([0.0, 1.0], dtype=np.float64)
        vec = np.asarray(down_uv, dtype=np.float64).reshape(-1)
        if vec.size < 2:
            return np.array([0.0, 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(vec[:2]))
        if norm < 1e-9:
            return np.array([0.0, 1.0], dtype=np.float64)
        return vec[:2] / norm

    def _extract_mask(self, masks, index, img_h, img_w, x1, y1, x2, y2):
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if masks is not None and getattr(masks, 'data', None) is not None and index < len(masks.data):
            raw = masks.data[index].detach().cpu().numpy()
            if raw.ndim == 3:
                raw = raw[0]
            raw_u8 = (raw > 0.5).astype(np.uint8) * 255
            if raw_u8.shape[0] != img_h or raw_u8.shape[1] != img_w:
                raw_u8 = cv2.resize(raw_u8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask = raw_u8
        else:
            mask[y1:y2 + 1, x1:x2 + 1] = 255
        return mask

    def _smooth_mask(self, mask):
        if mask is None or mask.size == 0:
            return mask
        out = mask.copy()
        k = self.mask_smooth_ksize
        if k >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            if self.mask_close_iters > 0:
                out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=self.mask_close_iters)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
            out = cv2.GaussianBlur(out, (k, k), 0)
            _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
        return out

    def _largest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _nearest_in_mask_point(self, mask, points, cx, cy):
        """Pick a mask pixel near (cx, cy); prefer thicker interior when tied.

        For non-convex hang cords the contour centroid often falls in the hollow;
        this snaps the action point back onto the cord mask.
        """
        if points is None or len(points) == 0:
            return None
        dx = points[:, 0] - float(cx)
        dy = points[:, 1] - float(cy)
        dist2 = dx * dx + dy * dy
        # Among the nearest ~5% mask pixels to the centroid, prefer the most interior.
        n_near = max(1, min(int(points.shape[0]), max(8, int(0.05 * points.shape[0]))))
        if n_near >= points.shape[0]:
            near_idx = np.arange(points.shape[0])
        else:
            near_idx = np.argpartition(dist2, n_near - 1)[:n_near]

        dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        xs = np.clip(points[near_idx, 0].astype(np.int32), 0, mask.shape[1] - 1)
        ys = np.clip(points[near_idx, 1].astype(np.int32), 0, mask.shape[0] - 1)
        interior = dt[ys, xs].astype(np.float64)
        # Maximize interior, then minimize distance to centroid.
        order = np.lexsort((dist2[near_idx], -interior))
        best = near_idx[int(order[0])]
        return points[best]

    def _pick_target(self, mask, seg_name, down_uv):
        contour = self._largest_contour(mask)
        if contour is None or cv2.contourArea(contour) < 8.0:
            return None

        # Light polygon smoothing for stable endpoints.
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(1.0, 0.01 * peri), True)
        if approx is not None and len(approx) >= 3:
            contour = approx

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        projections = points @ down_uv

        moments = cv2.moments(contour)
        if moments['m00'] > 1e-6:
            geom_cx = float(moments['m10'] / moments['m00'])
            geom_cy = float(moments['m01'] / moments['m00'])
        else:
            geom_cx = float(np.mean(xs))
            geom_cy = float(np.mean(ys))
        extras = {
            'contour': contour.reshape(-1, 2),
            'fields': {
                'geometric_center_px': [geom_cx, geom_cy],
            },
        }

        if seg_name == 'cabinet door handle':
            # End closer to ground: max projection along world-down image vector.
            idx = int(np.argmax(projections))
            tip = points[idx]
            extras['fields'].update(
                {
                    'handle_grasp_center_px': [float(tip[0]), float(tip[1])],
                    'handle_grasp_source': SOURCE_TAG + '_ground_end',
                }
            )
            return float(tip[0]), float(tip[1]), extras

        if seg_name == 'lock point':
            # 靠近地面的 20% 处 == from anti-ground edge, 80% toward ground.
            p_min = float(np.min(projections))
            p_max = float(np.max(projections))
            target_p = p_min + self.lock_ground_ratio * (p_max - p_min)
            idx = int(np.argmin(np.abs(projections - target_p)))
            tip = points[idx]
            extras['fields'].update(
                {
                    'handle_grasp_source': SOURCE_TAG + '_ground_ratio',
                }
            )
            return float(tip[0]), float(tip[1]), extras

        if seg_name == 'hang cord':
            # Arch/U cords: contour centroid often lies in the hollow. Snap onto mask.
            tip = self._nearest_in_mask_point(mask, points, geom_cx, geom_cy)
            if tip is None:
                return None
            extras['fields'].update(
                {
                    'handle_grasp_center_px': [float(tip[0]), float(tip[1])],
                    'handle_grasp_source': SOURCE_TAG + '_inmask_near_centroid',
                }
            )
            return float(tip[0]), float(tip[1]), extras

        # Default action target: same as geometric center.
        return geom_cx, geom_cy, extras


def resolve_publish_name(
    seg_class: str,
    requested: Sequence[str],
    opencv_color: Optional[str] = None,
) -> str:
    """Prefer a concrete requested class when it uniquely matches this family.

    When OpenCV already labeled red/green/yellow, never rewrite a different color
    just because the request list only mentions one colored class.
    """
    seg_n = normalize_name(seg_class)
    default = colored_publish_name(seg_n, opencv_color)
    matches = [
        str(item).strip()
        for item in requested
        if str(item).strip() and requested_activates_seg([item], seg_n)
    ]
    if opencv_color in SUPPORTED_COLORS and seg_n in COLORIZED_SEG_CLASSES:
        color_matches = [
            item for item in matches if opencv_color in normalize_name(item)
        ]
        if len(color_matches) == 1:
            return color_matches[0]
        if color_matches:
            return color_matches[0]
        return default
    if len(matches) == 1:
        return matches[0]
    return default


def merge_yolo_seg_priority(
    base_detections: Iterable[dict],
    seg_detections: Iterable[dict],
    requested_class_names: Sequence[str],
    mode: str = 'on_request',
):
    """Replace overlapping families with YOLO-seg when requested (or always).

    Preview of all seg detections is handled separately by the detector overlay;
    this merge only decides which detections are published as Object2D.
    """
    base = [dict(det) for det in base_detections]
    seg = [dict(det) for det in seg_detections]
    mode = str(mode or 'on_request').strip().lower()
    if not seg:
        return base, []

    prefer_families = set()
    for det in seg:
        family = detection_family_key(det.get('seg_class_name') or det.get('class_name', ''))
        if family is None:
            continue
        if mode == 'always' or requested_activates_seg(requested_class_names, family):
            prefer_families.add(family)

    if not prefer_families:
        return base, list(seg)

    kept = []
    for det in base:
        family = detection_family_key(det.get('class_name', ''))
        if family in prefer_families:
            continue
        kept.append(det)

    preferred = []
    for det in seg:
        family = detection_family_key(det.get('seg_class_name') or det.get('class_name', ''))
        if family not in prefer_families:
            continue
        det = dict(det)
        color = det.get('opencv_color')
        if color not in SUPPORTED_COLORS:
            color = None
        det['class_name'] = resolve_publish_name(
            det.get('seg_class_name') or det.get('class_name', ''),
            requested_class_names,
            opencv_color=color,
        )
        preferred.append(det)
    return preferred + kept, list(seg)


def _overlay_color_for_det(det) -> tuple:
    color = det.get('opencv_color')
    if color in OVERLAY_BGR:
        return OVERLAY_BGR[color]
    name = normalize_name(det.get('class_name', ''))
    for key, bgr in OVERLAY_BGR.items():
        if name.startswith(key + ' '):
            return bgr
    return (255, 180, 0)


def draw_yolo_seg_overlays(image, detections):
    """Draw smoothed contours + action target + geometric center for web debug."""
    if image is None or image.size == 0:
        return image
    for det in detections:
        if not (
            str(det.get('detection_source', '')).startswith('yoloseg')
            or str(det.get('handle_grasp_source', '')).startswith('yoloseg')
        ):
            continue
        contour = det.get('mask_contour')
        color = _overlay_color_for_det(det)
        if contour is not None and len(contour) >= 3:
            pts = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], True, color, 2)
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.22, image, 0.78, 0, image)
        cx = int(round(float(det.get('cx', 0.0))))
        cy = int(round(float(det.get('cy', 0.0))))
        cv2.drawMarker(
            image,
            (cx, cy),
            (0, 255, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )
        geom = det.get('geometric_center_px') or []
        if len(geom) == 2:
            gx = int(round(float(geom[0])))
            gy = int(round(float(geom[1])))
            cv2.circle(image, (gx, gy), 5, (255, 255, 0), 2)
        label = 'SEG %s' % str(det.get('class_name', ''))
        cv2.putText(
            image,
            label,
            (cx + 6, max(14, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return image
