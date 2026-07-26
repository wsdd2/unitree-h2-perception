# -*- coding: utf-8 -*-
"""OpenCV detector for the green hang tag, red cord, grasp point, and hook.

Publish classes for motion control:
  - green work tag: square reflective tag body
  - red hang cord: twisted red rope above the tag
  - work tag grasp point: white load-bearing spot on/near the cord apex
    (gripper pick target); falls back to the geometric cord apex
  - cabinet hang hook: white plastic base + metal tip (hang target)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLASS_GREEN_WORK_TAG = 'green work tag'
CLASS_RED_HANG_CORD = 'red hang cord'
CLASS_WORK_TAG_GRASP = 'work tag grasp point'
CLASS_CABINET_HANG_HOOK = 'cabinet hang hook'

CLASS_ID_TAG = 9101
CLASS_ID_CORD = 9102
CLASS_ID_GRASP = 9103
CLASS_ID_HOOK = 9104

# YOLOE / MobileCLIP prompt text -> stable publish class_name.
YOLOE_LABEL_TO_CLASS = {
    'green square work here safety tag': CLASS_GREEN_WORK_TAG,
    'green work tag': CLASS_GREEN_WORK_TAG,
    'green safety hang tag': CLASS_GREEN_WORK_TAG,
    'green reflective work tag': CLASS_GREEN_WORK_TAG,
    'red twisted hanging cord rope': CLASS_RED_HANG_CORD,
    'red hang cord': CLASS_RED_HANG_CORD,
    'red hanging rope': CLASS_RED_HANG_CORD,
    'white adhesive cabinet wall hook': CLASS_CABINET_HANG_HOOK,
    'white plastic wall hook': CLASS_CABINET_HANG_HOOK,
    'metal adhesive wall hook': CLASS_CABINET_HANG_HOOK,
    'cabinet hang hook': CLASS_CABINET_HANG_HOOK,
    'white cabinet hook': CLASS_CABINET_HANG_HOOK,
}


def normalize_yoloe_work_tag_label(class_name: str) -> Optional[str]:
    """Map a raw YOLOE label to a stable hang-tag publish class, or None."""
    name = str(class_name or '').lower().replace('_', ' ').replace('-', ' ').strip()
    if not name:
        return None
    if name in YOLOE_LABEL_TO_CLASS:
        return YOLOE_LABEL_TO_CLASS[name]
    if any(token in name for token in ('wall hook', 'adhesive hook', 'cabinet hook', 'hang hook')):
        return CLASS_CABINET_HANG_HOOK
    if 'work' in name and 'tag' in name:
        return CLASS_GREEN_WORK_TAG
    if ('cord' in name or 'rope' in name) and 'red' in name:
        return CLASS_RED_HANG_CORD
    if 'work tag grasp' in name:
        return CLASS_WORK_TAG_GRASP
    return None


def normalize_yoloe_work_tag_detection(det: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mapped = normalize_yoloe_work_tag_label(det.get('class_name', ''))
    if mapped is None:
        return None
    out = dict(det)
    out['class_name'] = mapped
    if mapped == CLASS_GREEN_WORK_TAG:
        out['class_id'] = CLASS_ID_TAG
    elif mapped == CLASS_RED_HANG_CORD:
        out['class_id'] = CLASS_ID_CORD
    elif mapped == CLASS_CABINET_HANG_HOOK:
        out['class_id'] = CLASS_ID_HOOK
    else:
        out['class_id'] = CLASS_ID_GRASP
    out['work_tag_source'] = str(out.get('work_tag_source') or 'yoloe')
    # Default action point = box center; hook tip may be refined later.
    if 'cx' not in out or 'cy' not in out:
        out['cx'] = 0.5 * (float(out.get('xmin', 0)) + float(out.get('xmax', 0)))
        out['cy'] = 0.5 * (float(out.get('ymin', 0)) + float(out.get('ymax', 0)))
    return out


def _det(
    class_name: str,
    class_id: int,
    confidence: float,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    cx: float,
    cy: float,
    **extra: Any,
) -> Dict[str, Any]:
    payload = {
        'class_name': class_name,
        'class_id': int(class_id),
        'confidence': float(min(0.99, max(0.01, confidence))),
        'xmin': int(xmin),
        'ymin': int(ymin),
        'xmax': int(xmax),
        'ymax': int(ymax),
        'cx': float(cx),
        'cy': float(cy),
    }
    payload.update(extra)
    return payload


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    low = cv2.inRange(hsv, (0, 55, 50), (16, 255, 255))
    high = cv2.inRange(hsv, (168, 55, 50), (179, 255, 255))
    mask = cv2.bitwise_or(low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _green_mask(hsv: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(hsv, (35, 40, 35), (95, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _white_mask(hsv: np.ndarray) -> np.ndarray:
    # Hook base is pale plastic on grey cabinet: low sat, mid-high value.
    mask = cv2.inRange(hsv, (0, 0, 135), (179, 55, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _contour_center(contour: np.ndarray, ox: float = 0.0, oy: float = 0.0) -> Tuple[float, float]:
    moments = cv2.moments(contour)
    if moments['m00'] != 0.0:
        return ox + float(moments['m10'] / moments['m00']), oy + float(moments['m01'] / moments['m00'])
    x, y, w, h = cv2.boundingRect(contour)
    return ox + float(x + w) * 0.5, oy + float(y + h) * 0.5


def detect_green_work_tag(image_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    if image_bgr is None or image_bgr.size == 0:
        return None
    img_h, img_w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = _green_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    min_area = max(800.0, 0.002 * img_w * img_h)
    max_area = 0.45 * img_w * img_h
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h < 40:
            continue
        aspect = float(w) / float(max(1, h))
        if aspect < 0.70 or aspect > 1.40:
            continue
        rectangularity = area / float(max(1, w * h))
        if rectangularity < 0.45:
            continue
        square_score = 1.0 - min(1.0, abs(1.0 - aspect))
        score = area * rectangularity * max(0.2, square_score)
        if score > best_score:
            best_score = score
            cx, cy = _contour_center(contour)
            best = _det(
                CLASS_GREEN_WORK_TAG,
                CLASS_ID_TAG,
                0.55 + min(0.35, 0.0000003 * score),
                x,
                y,
                x + w,
                y + h,
                cx,
                cy,
                work_tag_source='opencv_green_square',
            )
    return best


def _cord_component_from_tag(
    red: np.ndarray,
    tag: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Keep only red components that touch the top edge of the green tag."""
    img_h, img_w = red.shape[:2]
    tw = max(1, int(tag['xmax'] - tag['xmin']))
    th = max(1, int(tag['ymax'] - tag['ymin']))
    x1 = max(0, int(tag['xmin']) - int(0.08 * tw))
    x2 = min(img_w, int(tag['xmax']) + int(0.08 * tw))
    y_seed1 = max(0, int(tag['ymin']) - 4)
    y_seed2 = min(img_h, int(tag['ymin']) + max(10, int(0.08 * th)))
    y_top = max(0, int(tag['ymin']) - int(1.15 * th))

    seed = np.zeros_like(red)
    seed[y_seed1:y_seed2, x1:x2] = red[y_seed1:y_seed2, x1:x2]
    if int(cv2.countNonZero(seed)) < 20:
        return None

    search = np.zeros_like(red)
    search[y_top:y_seed2, x1:x2] = red[y_top:y_seed2, x1:x2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(search, connectivity=8)
    keep = np.zeros_like(red)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        component = labels == label
        if not np.any(component[y_seed1:y_seed2, x1:x2] & (seed[y_seed1:y_seed2, x1:x2] > 0)):
            continue
        # Reject huge skin/background blobs that dominate the ROI.
        if area > 0.12 * img_w * img_h:
            continue
        keep[component] = 255
    if int(cv2.countNonZero(keep)) < 40:
        return None
    return keep


def _row_leg_centers(row_xs: np.ndarray) -> Optional[Tuple[float, float]]:
    """Split one mask row into left/right cord legs when both exist."""
    if row_xs.size < 4:
        return None
    ordered = np.sort(row_xs.astype(np.float64))
    gaps = np.diff(ordered)
    if gaps.size == 0:
        return None
    split = int(np.argmax(gaps))
    if float(gaps[split]) < 8.0:
        return None
    left = ordered[: split + 1]
    right = ordered[split + 1 :]
    if left.size < 2 or right.size < 2:
        return None
    return float(np.median(left)), float(np.median(right))


def _apex_from_cord_mask(cord_mask: np.ndarray) -> Optional[Tuple[float, float, Tuple[int, int, int, int]]]:
    ys, xs = np.where(cord_mask > 0)
    if xs.size < 40:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bottom = int(ys.max())
    top = int(ys.min())

    # Walk upward from the tag attachments: apex is where the two legs merge.
    last_pair = None
    merge_y = None
    merge_x = None
    for y in range(bottom, top - 1, -1):
        row_xs = xs[ys == y]
        pair = _row_leg_centers(row_xs)
        if pair is not None:
            last_pair = pair
            continue
        if last_pair is not None and row_xs.size >= 3:
            merge_y = float(y)
            merge_x = float(np.median(row_xs))
            break

    if merge_x is not None and merge_y is not None:
        apex_x, apex_y = merge_x, merge_y
    elif last_pair is not None:
        # Legs stay split to the top of the ROI (e.g. hand holding the loop).
        left_x, right_x = last_pair
        apex_x = 0.5 * (left_x + right_x)
        # Use a point slightly below the image/mask top so depth stays valid.
        apex_y = float(top + max(8, int(0.08 * max(1, bottom - top))))
    else:
        mid_lo = x1 + 0.20 * (x2 - x1)
        mid_hi = x2 - 0.20 * (x2 - x1)
        between = (xs >= mid_lo) & (xs <= mid_hi)
        if not np.any(between):
            between = np.ones(xs.shape, dtype=bool)
        top_y = int(ys[between].min())
        top_band = between & (ys <= top_y + max(3, int(0.06 * max(1, y2 - y1))))
        apex_x = float(np.median(xs[top_band]))
        apex_y = float(top_y + max(6, int(0.05 * max(1, y2 - y1))))

    return apex_x, apex_y, (x1, y1, x2, y2)


def detect_red_hang_cord(
    image_bgr: np.ndarray,
    tag: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if image_bgr is None or image_bgr.size == 0 or tag is None:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red = _red_mask(hsv)
    cord_mask = _cord_component_from_tag(red, tag)
    if cord_mask is None:
        return None
    apex = _apex_from_cord_mask(cord_mask)
    if apex is None:
        return None
    apex_x, apex_y, (x1, y1, x2, y2) = apex
    area = float(cv2.countNonZero(cord_mask))
    return _det(
        CLASS_RED_HANG_CORD,
        CLASS_ID_CORD,
        0.62,
        x1,
        y1,
        x2,
        y2,
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        handle_grasp_center_px=[int(round(apex_x)), int(round(apex_y))],
        handle_grasp_source='opencv_red_cord_apex',
        cord_apex_px=[float(apex_x), float(apex_y)],
        cord_mask_area_px=area,
        _cord_mask=cord_mask,
    )


def detect_white_force_point(
    image_bgr: np.ndarray,
    cord: Optional[Dict[str, Any]] = None,
    tag: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """White load-bearing spot on the red cord (gripper pick target)."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    img_h, img_w = image_bgr.shape[:2]
    apex = None
    if cord is not None:
        apex_vals = cord.get('cord_apex_px') or cord.get('handle_grasp_center_px')
        if apex_vals is not None and len(apex_vals) >= 2:
            apex = (float(apex_vals[0]), float(apex_vals[1]))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Bright specular / white ferrule on red cord: high value, not strongly green.
    bright = cv2.inRange(hsv, (0, 0, 170), (179, 90, 255))
    cord_mask = cord.get('_cord_mask') if cord else None
    if cord_mask is not None:
        dilate = cv2.dilate(
            cord_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        bright = cv2.bitwise_and(bright, dilate)

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 8.0 or area > 900.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) > 45:
            continue
        cx, cy = _contour_center(contour)
        if tag is not None and tag['xmin'] + 12 < cx < tag['xmax'] - 12 and tag['ymin'] + 12 < cy < tag['ymax'] - 12:
            continue
        if apex is not None:
            dist = float(np.hypot(cx - apex[0], cy - apex[1]))
            if dist > 55.0:
                continue
            score = area / (1.0 + 0.08 * dist)
        else:
            score = area
        candidates.append(
            (
                score,
                _det(
                    CLASS_WORK_TAG_GRASP,
                    CLASS_ID_GRASP,
                    0.72,
                    x,
                    y,
                    x + w,
                    y + h,
                    cx,
                    cy,
                    handle_grasp_center_px=[int(round(cx)), int(round(cy))],
                    handle_grasp_source='opencv_white_force_on_cord',
                    work_tag_source='opencv_white_force_on_cord',
                ),
            )
        )

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    if apex is None:
        return None
    ax, ay = apex
    half = 12
    return _det(
        CLASS_WORK_TAG_GRASP,
        CLASS_ID_GRASP,
        0.78,
        max(0, int(ax) - half),
        max(0, int(ay) - half),
        min(img_w - 1, int(ax) + half),
        min(img_h - 1, int(ay) + half),
        ax,
        ay,
        handle_grasp_center_px=[int(round(ax)), int(round(ay))],
        handle_grasp_source='opencv_red_cord_apex_fallback',
        work_tag_source='opencv_red_cord_apex_fallback',
    )


def detect_cabinet_hang_hook(
    image_bgr: np.ndarray,
    tag: Optional[Dict[str, Any]] = None,
    cord: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Detect white plastic hook base; hang tip is metal curve under the base."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    img_h, img_w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white = _white_mask(hsv)

    # Search window: prefer around cord apex, else above tag center,
    # else upper cabinet region for an empty hook.
    if cord is not None and cord.get('cord_apex_px'):
        ax, ay = float(cord['cord_apex_px'][0]), float(cord['cord_apex_px'][1])
        x1 = max(0, int(ax) - 70)
        x2 = min(img_w, int(ax) + 70)
        y1 = max(0, int(ay) - 80)
        y2 = min(img_h, int(ay) + 40)
    elif tag is not None:
        tw = max(1, int(tag['xmax'] - tag['xmin']))
        th = max(1, int(tag['ymax'] - tag['ymin']))
        cx = 0.5 * (float(tag['xmin']) + float(tag['xmax']))
        x1 = max(0, int(cx - 0.45 * tw))
        x2 = min(img_w, int(cx + 0.45 * tw))
        y1 = max(0, int(tag['ymin']) - int(1.4 * th))
        y2 = min(img_h, int(tag['ymin']) + 8)
    else:
        x1 = int(0.12 * img_w)
        x2 = int(0.88 * img_w)
        y1 = int(0.04 * img_h)
        y2 = int(0.62 * img_h)

    search = np.zeros_like(white)
    search[y1:y2, x1:x2] = 255
    white = cv2.bitwise_and(white, search)
    # Suppress the large white circle on the tag face.
    if tag is not None:
        white[int(tag['ymin']) : int(tag['ymax']), int(tag['xmin']) : int(tag['xmax'])] = 0

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 60.0 or area > 3500.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 14 or h < 8 or w > 120 or h > 90:
            continue
        aspect = float(w) / float(max(1, h))
        if aspect < 0.85 or aspect > 3.5:
            continue
        rectangularity = area / float(max(1, w * h))
        if rectangularity < 0.30:
            continue

        tip_x = float(x) + 0.5 * float(w)
        tip_y = float(y) + 0.95 * float(h)
        refined = _refine_hook_tip(image_bgr, x, y, w, h)
        # Require a darker metal tip under the white base to reject fabric/glare.
        if refined is None:
            continue
        tip_x, tip_y = refined

        # White adhesive bases sit on grey metal, not colorful fabric.
        base_roi = image_bgr[y : y + h, x : x + w]
        if base_roi.size == 0:
            continue
        base_hsv = cv2.cvtColor(base_roi, cv2.COLOR_BGR2HSV)
        colorful = float(np.mean(base_hsv[:, :, 1] > 70))
        if colorful > 0.35:
            continue

        score = area * rectangularity
        if cord is not None and cord.get('cord_apex_px'):
            ax, ay = float(cord['cord_apex_px'][0]), float(cord['cord_apex_px'][1])
            dist = float(np.hypot(tip_x - ax, tip_y - ay))
            if dist > 90.0:
                continue
            score *= 1.0 / (1.0 + 0.03 * dist)

        half = max(10, int(0.4 * max(w, h)))
        candidates.append(
            (
                score,
                _det(
                    CLASS_CABINET_HANG_HOOK,
                    CLASS_ID_HOOK,
                    0.60 + 0.25 * rectangularity,
                    max(0, int(tip_x) - half),
                    max(0, int(tip_y) - half),
                    min(img_w - 1, int(tip_x) + half),
                    min(img_h - 1, int(tip_y) + half),
                    tip_x,
                    tip_y,
                    handle_grasp_center_px=[int(round(tip_x)), int(round(tip_y))],
                    handle_grasp_source='opencv_white_hook_base_tip',
                    work_tag_source='opencv_white_hook_base_tip',
                    hook_base_bbox_px=[int(x), int(y), int(x + w), int(y + h)],
                ),
            )
        )

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return None


def _refine_hook_tip(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> Optional[Tuple[float, float]]:
    img_h, img_w = image_bgr.shape[:2]
    rx1 = max(0, x)
    rx2 = min(img_w, x + w)
    ry1 = max(0, y + int(0.40 * h))
    ry2 = min(img_h, y + h + max(16, int(1.4 * h)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    crop = image_bgr[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Prefer darker metal relative to local median.
    local = float(np.median(gray))
    dark = (gray < max(70.0, local - 18.0)).astype(np.uint8) * 255
    col_mask = np.zeros_like(dark)
    mid = (rx2 - rx1) // 2
    half = max(3, int(0.30 * (rx2 - rx1)))
    col_mask[:, max(0, mid - half) : min(col_mask.shape[1], mid + half)] = 255
    dark = cv2.bitwise_and(dark, col_mask)
    ys, xs = np.where(dark > 0)
    if xs.size < 8:
        return None
    bottom = int(ys.max())
    band = ys >= bottom - max(2, int(0.18 * (ry2 - ry1)))
    tip_x = float(rx1) + float(np.median(xs[band]))
    tip_y = float(ry1) + float(bottom)
    return tip_x, tip_y


def detect_work_tag_scene(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    tag = detect_green_work_tag(image_bgr)
    if tag is not None:
        detections.append(tag)

    cord = detect_red_hang_cord(image_bgr, tag=tag)
    cord_pub = None
    if cord is not None:
        cord_pub = {key: value for key, value in cord.items() if key != '_cord_mask'}
        detections.append(cord_pub)

    grasp = detect_white_force_point(image_bgr, cord=cord, tag=tag)
    if grasp is not None:
        detections.append(grasp)
        if cord_pub is not None:
            cord_pub['handle_grasp_center_px'] = list(grasp.get('handle_grasp_center_px') or [])
            cord_pub['handle_grasp_source'] = str(
                grasp.get('handle_grasp_source', cord_pub.get('handle_grasp_source', ''))
            )

    hook = detect_cabinet_hang_hook(image_bgr, tag=tag, cord=cord)
    if hook is not None:
        detections.append(hook)

    return detections


def refine_hook_tip_in_bbox(
    image_bgr: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> Optional[Tuple[float, float, Tuple[int, int, int, int]]]:
    """Refine a YOLOE hook box to the metal tip under the white plastic base."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    img_h, img_w = image_bgr.shape[:2]
    x1 = max(0, int(xmin))
    y1 = max(0, int(ymin))
    x2 = min(img_w, int(xmax))
    y2 = min(img_h, int(ymax))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    # Expand slightly downward: metal tip often sits below the plastic base box.
    pad_x = max(6, int(0.20 * (x2 - x1)))
    pad_y = max(10, int(0.45 * (y2 - y1)))
    rx1 = max(0, x1 - pad_x)
    rx2 = min(img_w, x2 + pad_x)
    ry1 = max(0, y1 - pad_y)
    ry2 = min(img_h, y2 + pad_y)

    crop = image_bgr[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = _white_mask(hsv)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_base = None
    best_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 40.0 or area > 4000.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(w) / float(max(1, h))
        if aspect < 0.7 or aspect > 3.8:
            continue
        if area > best_area:
            best_area = area
            best_base = (rx1 + x, ry1 + y, w, h)

    if best_base is not None:
        bx, by, bw, bh = best_base
        refined = _refine_hook_tip(image_bgr, bx, by, bw, bh)
        if refined is not None:
            return refined[0], refined[1], (bx, by, bx + bw, by + bh)
        return float(bx + 0.5 * bw), float(by + 0.9 * bh), (bx, by, bx + bw, by + bh)

    # No white base found: use lower-center of the YOLOE box as hang tip.
    tip_x = 0.5 * (float(x1) + float(x2))
    tip_y = float(y1) + 0.78 * (float(y2) - float(y1))
    return tip_x, tip_y, (x1, y1, x2, y2)


def merge_work_tag_detections(
    image_bgr: np.ndarray,
    model_detections: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fuse YOLOE hang-tag semantics with OpenCV geometry.

    Policy:
      - tag / hook: prefer YOLOE box; refine hook tip with OpenCV
      - cord / grasp point: prefer OpenCV geometry (inverted-V + white force)
      - fill missing classes from the other source
    """
    yoloe_by_class: Dict[str, Dict[str, Any]] = {}
    for det in model_detections:
        normalized = normalize_yoloe_work_tag_detection(det)
        if normalized is None:
            continue
        name = str(normalized['class_name'])
        prev = yoloe_by_class.get(name)
        if prev is None or float(normalized.get('confidence', 0.0)) > float(prev.get('confidence', 0.0)):
            yoloe_by_class[name] = normalized

    opencv_list = detect_work_tag_scene(image_bgr)
    opencv_by_class = {str(det['class_name']): det for det in opencv_list}

    # If OpenCV missed the green tag but YOLOE found it, re-run geometric
    # cord/grasp using the YOLOE box as the tag prior.
    if CLASS_GREEN_WORK_TAG not in opencv_by_class and CLASS_GREEN_WORK_TAG in yoloe_by_class:
        tag_prior = yoloe_by_class[CLASS_GREEN_WORK_TAG]
        cord = detect_red_hang_cord(image_bgr, tag=tag_prior)
        if cord is not None:
            cord_pub = {key: value for key, value in cord.items() if key != '_cord_mask'}
            opencv_by_class[CLASS_RED_HANG_CORD] = cord_pub
            grasp = detect_white_force_point(image_bgr, cord=cord, tag=tag_prior)
            if grasp is not None:
                opencv_by_class[CLASS_WORK_TAG_GRASP] = grasp
                cord_pub['handle_grasp_center_px'] = list(grasp.get('handle_grasp_center_px') or [])
                cord_pub['handle_grasp_source'] = str(grasp.get('handle_grasp_source', ''))
            if CLASS_CABINET_HANG_HOOK not in opencv_by_class:
                hook = detect_cabinet_hang_hook(image_bgr, tag=tag_prior, cord=cord)
                if hook is not None:
                    opencv_by_class[CLASS_CABINET_HANG_HOOK] = hook

    merged: List[Dict[str, Any]] = []

    if CLASS_GREEN_WORK_TAG in yoloe_by_class:
        merged.append(yoloe_by_class[CLASS_GREEN_WORK_TAG])
    elif CLASS_GREEN_WORK_TAG in opencv_by_class:
        merged.append(opencv_by_class[CLASS_GREEN_WORK_TAG])

    if CLASS_RED_HANG_CORD in opencv_by_class:
        merged.append(opencv_by_class[CLASS_RED_HANG_CORD])
    elif CLASS_RED_HANG_CORD in yoloe_by_class:
        cord = dict(yoloe_by_class[CLASS_RED_HANG_CORD])
        cord['handle_grasp_center_px'] = [
            int(round(float(cord.get('cx', 0.0)))),
            int(round(float(cord.get('cy', 0.0)))),
        ]
        cord['handle_grasp_source'] = 'yoloe_cord_center'
        merged.append(cord)

    if CLASS_WORK_TAG_GRASP in opencv_by_class:
        merged.append(opencv_by_class[CLASS_WORK_TAG_GRASP])

    if CLASS_CABINET_HANG_HOOK in yoloe_by_class:
        hook = dict(yoloe_by_class[CLASS_CABINET_HANG_HOOK])
        refined = refine_hook_tip_in_bbox(
            image_bgr,
            int(hook.get('xmin', 0)),
            int(hook.get('ymin', 0)),
            int(hook.get('xmax', 0)),
            int(hook.get('ymax', 0)),
        )
        if refined is not None:
            tip_x, tip_y, base_bbox = refined
            hook['cx'] = float(tip_x)
            hook['cy'] = float(tip_y)
            hook['handle_grasp_center_px'] = [int(round(tip_x)), int(round(tip_y))]
            hook['handle_grasp_source'] = 'yoloe_box_opencv_tip'
            hook['work_tag_source'] = 'yoloe+opencv_tip'
            hook['hook_base_bbox_px'] = [int(v) for v in base_bbox]
        else:
            hook['handle_grasp_center_px'] = [
                int(round(float(hook.get('cx', 0.0)))),
                int(round(float(hook.get('cy', 0.0)))),
            ]
            hook['handle_grasp_source'] = 'yoloe_box_center'
        merged.append(hook)
    elif CLASS_CABINET_HANG_HOOK in opencv_by_class:
        merged.append(opencv_by_class[CLASS_CABINET_HANG_HOOK])

    return merged


def is_work_tag_class(class_name: str) -> bool:
    if normalize_yoloe_work_tag_label(class_name) is not None:
        return True
    name = str(class_name or '').lower()
    return any(
        token in name
        for token in (
            'green work tag',
            'red hang cord',
            'work tag grasp',
            'cabinet hang hook',
            'wall hook',
            'adhesive hook',
            'work here safety tag',
            'hanging cord rope',
        )
    )


def is_work_tag_grasp_class(class_name: str) -> bool:
    name = str(class_name or '').lower()
    return 'work tag grasp' in name


def is_hang_hook_class(class_name: str) -> bool:
    mapped = normalize_yoloe_work_tag_label(class_name)
    if mapped == CLASS_CABINET_HANG_HOOK:
        return True
    name = str(class_name or '').lower()
    return 'cabinet hang hook' in name or 'wall hook' in name or 'adhesive hook' in name


def stage_prefers_work_tag_grasp(stage_id: int, stage_name: str) -> bool:
    name = str(stage_name or '').lower()
    if int(stage_id) == 6:
        return True
    return any(
        token in name
        for token in (
            'pick_work_tag',
            'grasp_work_tag',
            'pick_hang_tag',
            'grasp_hang_tag',
            'work_tag_grasp',
        )
    )


def stage_prefers_hang_hook(stage_id: int, stage_name: str) -> bool:
    name = str(stage_name or '').lower()
    if int(stage_id) == 7:
        return True
    return any(
        token in name
        for token in (
            'hang_work_tag',
            'hang_tag',
            'hang_on_hook',
            'work_tag_hang',
        )
    )
