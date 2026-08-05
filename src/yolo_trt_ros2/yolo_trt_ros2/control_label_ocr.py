"""Optional Chinese OCR association for visually identical cabinet controls."""

from collections import Counter, deque
from difflib import SequenceMatcher
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np


CONTROL_NAME_TOKENS = (
    'button',
    'switch',
    'selector',
    'meter',
    'gauge',
    'handle',
    'lock point',
)


class ControlLabelOCR:
    """Attach stable label_text/semantic_name fields to detection dictionaries."""

    def __init__(
        self,
        logger,
        enabled=False,
        backend='paddleocr',
        language='ch',
        dictionary_path='',
        interval_frames=6,
        min_confidence=0.65,
        stable_frames=3,
        max_missed=30,
        max_controls_per_frame=8,
        roi_width_scale=2.6,
        roi_height_scale=1.8,
        row_alignment_tolerance_px=28.0,
        label_tag_brightness_delta=18.0,
    ):
        self.logger = logger
        self.enabled = bool(enabled)
        self.backend_name = str(backend).strip().lower()
        self.language = str(language).strip() or 'ch'
        self.interval_frames = max(1, int(interval_frames))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.stable_frames = max(1, int(stable_frames))
        self.max_missed = max(1, int(max_missed))
        self.max_controls_per_frame = max(1, int(max_controls_per_frame))
        self.roi_width_scale = max(0.5, float(roi_width_scale))
        self.roi_height_scale = max(0.5, float(roi_height_scale))
        self.row_alignment_tolerance_px = max(
            5.0,
            float(row_alignment_tolerance_px),
        )
        self.label_tag_brightness_delta = max(
            3.0,
            float(label_tag_brightness_delta),
        )
        self.frame_index = 0
        self.tracks = {}
        self.dictionary = self._load_dictionary(dictionary_path)
        self.reader = None

        if self.enabled:
            self._initialize_backend()

    @property
    def available(self):
        return self.enabled and self.reader is not None

    def _initialize_backend(self):
        if self.backend_name == 'tesseract':
            executable = shutil.which('tesseract')
            if executable is None:
                self.logger.warn(
                    'Control OCR unavailable: tesseract executable was not found.'
                )
                self.enabled = False
                return
            self.reader = executable
            self.logger.info(
                'Control label OCR ready: backend=tesseract lang=%s dictionary=%d'
                % (self.language, len(self.dictionary))
            )
            return
        if self.backend_name != 'paddleocr':
            self.logger.warn('Control OCR disabled: unsupported backend=%s' % self.backend_name)
            self.enabled = False
            return
        try:
            from paddleocr import PaddleOCR

            try:
                self.reader = PaddleOCR(
                    lang=self.language,
                    use_angle_cls=True,
                    show_log=False,
                )
            except TypeError:
                # PaddleOCR 3.x renamed some constructor arguments.
                self.reader = PaddleOCR(
                    lang=self.language,
                    use_textline_orientation=True,
                )
            self.logger.info(
                'Control label OCR ready: backend=%s lang=%s dictionary=%d'
                % (self.backend_name, self.language, len(self.dictionary))
            )
        except Exception as exc:
            self.reader = None
            self.enabled = False
            self.logger.warn(
                'Control OCR unavailable; continuing without labels: %s' % exc
            )

    def _load_dictionary(self, path):
        if not path:
            return {}
        resolved = Path(str(path)).expanduser()
        if not resolved.is_file():
            self.logger.warn('OCR label dictionary does not exist: %s' % resolved)
            return {}
        values = {}
        with resolved.open('r', encoding='utf-8') as stream:
            for raw in stream:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [part.strip() for part in line.split('|') if part.strip()]
                if not parts:
                    continue
                canonical = parts[0]
                for alias in parts:
                    normalized = self._normalize_text(alias)
                    if normalized:
                        values[normalized] = canonical
        return values

    @staticmethod
    def _normalize_text(text):
        return ''.join(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', str(text))).lower()

    @staticmethod
    def _is_control_detection(det):
        name = str(det.get('class_name', '')).lower()
        return any(token in name for token in CONTROL_NAME_TOKENS)

    @staticmethod
    def _track_key(det):
        return (
            str(det.get('class_name', '')).lower(),
            int(round(float(det.get('cx', 0.0)) / 12.0)),
            int(round(float(det.get('cy', 0.0)) / 12.0)),
        )

    def annotate(self, image, detections):
        detections = [dict(det) for det in detections]
        if image is None or image.size == 0:
            return detections
        detections = self._attach_spatial_semantics(image, detections)
        if not self.available:
            return detections

        self.frame_index += 1
        run_ocr = self.frame_index % self.interval_frames == 0
        ranked = sorted(
            (det for det in detections if self._is_control_detection(det)),
            key=lambda det: float(det.get('confidence', 0.0)),
            reverse=True,
        )
        ocr_keys = {
            self._track_key(det)
            for det in ranked[:self.max_controls_per_frame]
        }
        seen = set()
        for det in detections:
            if not self._is_control_detection(det):
                continue
            key = self._track_key(det)
            seen.add(key)
            track = self.tracks.get(key)
            if track is None:
                track = {
                    'labels': deque(maxlen=self.stable_frames),
                    'confidences': deque(maxlen=self.stable_frames),
                    'missed': 0,
                    'stable_label': '',
                    'stable_confidence': 0.0,
                }
                self.tracks[key] = track
            track['missed'] = 0

            if run_ocr and key in ocr_keys:
                result = self._read_detection_label(image, det)
                if result is not None:
                    label, confidence = result
                    track['labels'].append(label)
                    track['confidences'].append(confidence)
                    self._update_stable_track(track)

            label = str(track.get('stable_label', ''))
            if label:
                det['label_text'] = label
                det['label_confidence'] = float(track['stable_confidence'])
                self._apply_semantic_name(det)

        for key in list(self.tracks):
            if key in seen:
                continue
            self.tracks[key]['missed'] += 1
            if self.tracks[key]['missed'] > self.max_missed:
                del self.tracks[key]
        return detections

    def _attach_spatial_semantics(self, image, detections):
        buttons = [
            det
            for det in detections
            if str(det.get('class_name', '')).lower()
            in ('red push button', 'green push button', 'yellow push button')
        ]
        row_model = None
        if len(buttons) >= 2:
            xs = np.asarray([float(det.get('cx', 0.0)) for det in buttons])
            ys = np.asarray([float(det.get('cy', 0.0)) for det in buttons])
            if float(np.ptp(xs)) >= 10.0:
                slope, intercept = np.polyfit(xs, ys, 1)
                row_model = (float(slope), float(intercept))
            else:
                row_model = (0.0, float(np.median(ys)))

        for det in detections:
            name = str(det.get('class_name', '')).lower()
            relation = ''
            if 'rotary selector switch' in name and row_model is not None:
                predicted_y = (
                    row_model[0] * float(det.get('cx', 0.0))
                    + row_model[1]
                )
                delta_y = float(det.get('cy', 0.0)) - predicted_y
                if abs(delta_y) <= self.row_alignment_tolerance_px:
                    relation = 'top_row'
                elif delta_y > self.row_alignment_tolerance_px:
                    relation = 'middle_row'
            det['spatial_relation'] = relation
            det['label_tag_present'] = self._detect_label_tag(image, det)
            self._apply_semantic_name(det)
        return detections

    def _detect_label_tag(self, image, det):
        height, width = image.shape[:2]
        x1 = float(det.get('xmin', 0.0))
        y2 = float(det.get('ymax', 0.0))
        x2 = float(det.get('xmax', width - 1))
        box_w = max(8.0, x2 - x1)
        box_h = max(8.0, float(det.get('ymax', 0.0)) - float(det.get('ymin', 0.0)))
        left = max(0, int(round(x1 - 0.45 * box_w)))
        right = min(width, int(round(x2 + 0.45 * box_w)))
        top = max(0, int(round(y2)))
        bottom = min(height, int(round(y2 + 1.6 * box_h)))
        if right - left < 8 or bottom - top < 6:
            return False
        crop = image[top:bottom, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        threshold = min(
            245.0,
            float(np.mean(gray)) + self.label_tag_brightness_delta,
        )
        mask = (gray >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        crop_area = max(1.0, float(crop.shape[0] * crop.shape[1]))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = float(w * h) / crop_area
            aspect = float(w) / max(1.0, float(h))
            if 0.04 <= area_ratio <= 0.75 and 1.3 <= aspect <= 10.0:
                return True
        return False

    def _apply_semantic_name(self, det):
        parts = [str(det.get('class_name', 'unknown'))]
        relation = str(det.get('spatial_relation', '')).strip()
        if relation:
            parts.append(relation)
        label = str(det.get('label_text', '')).strip()
        if label:
            parts.append(label)
        elif bool(det.get('label_tag_present', False)):
            parts.append('with_tag')
        if len(parts) > 1:
            det['semantic_name'] = '/'.join(parts)
            det['control_id'] = self._control_id(
                det,
                '/'.join(parts[1:]),
            )

    @staticmethod
    def _control_id(det, label):
        class_part = re.sub(r'[^a-z0-9]+', '_', str(det.get('class_name', '')).lower()).strip('_')
        label_part = re.sub(r'\s+', '', str(label))
        return '%s/%s' % (class_part or 'control', label_part)

    def _update_stable_track(self, track):
        if len(track['labels']) < self.stable_frames:
            return
        counts = Counter(track['labels'])
        label, count = counts.most_common(1)[0]
        if count < self.stable_frames:
            return
        confidences = [
            confidence
            for candidate, confidence in zip(track['labels'], track['confidences'])
            if candidate == label
        ]
        track['stable_label'] = label
        track['stable_confidence'] = float(np.mean(confidences))

    def _read_detection_label(self, image, det):
        best = None
        variants = []
        for crop in self._candidate_crops(image, det):
            for rotated in (
                crop,
                cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.rotate(crop, cv2.ROTATE_180),
            ):
                variants.append(self._prepare_crop(rotated))
        montage = self._make_montage(variants)
        for raw_text, ocr_confidence in self._run_ocr(montage):
            normalized = self._normalize_text(raw_text)
            if not normalized:
                continue
            label, dictionary_score = self._match_dictionary(normalized)
            confidence = float(ocr_confidence) * float(dictionary_score)
            if confidence < self.min_confidence:
                continue
            candidate = (label, confidence)
            if best is None or candidate[1] > best[1]:
                best = candidate
        return best

    @staticmethod
    def _make_montage(images):
        images = [image for image in images if image is not None and image.size > 0]
        if not images:
            return None
        max_width = min(1200, max(image.shape[1] for image in images))
        rows = []
        for image in images:
            if image.shape[1] > max_width:
                scale = float(max_width) / float(image.shape[1])
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right = max_width - image.shape[1]
            rows.append(
                cv2.copyMakeBorder(
                    image,
                    6,
                    6,
                    6,
                    right + 6,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            )
        return np.ascontiguousarray(cv2.vconcat(rows))

    def _candidate_crops(self, image, det):
        height, width = image.shape[:2]
        x1 = float(det.get('xmin', 0))
        y1 = float(det.get('ymin', 0))
        x2 = float(det.get('xmax', width - 1))
        y2 = float(det.get('ymax', height - 1))
        box_w = max(12.0, x2 - x1)
        box_h = max(12.0, y2 - y1)
        regions = [
            # Most cabinet labels in the field image are to the right.
            (x2, y1 - 0.4 * box_h, x2 + self.roi_width_scale * box_w, y2 + 1.0 * box_h),
            # Conventional label below a control.
            (x1 - 0.6 * box_w, y2, x2 + 0.6 * box_w, y2 + self.roi_height_scale * box_h),
            # Left-side fallback.
            (x1 - self.roi_width_scale * box_w, y1 - 0.4 * box_h, x1, y2 + 1.0 * box_h),
        ]
        crops = []
        for left, top, right, bottom in regions:
            ix1 = max(0, min(width - 1, int(round(left))))
            iy1 = max(0, min(height - 1, int(round(top))))
            ix2 = max(0, min(width, int(round(right))))
            iy2 = max(0, min(height, int(round(bottom))))
            if ix2 - ix1 < 8 or iy2 - iy1 < 8:
                continue
            crops.append(np.ascontiguousarray(image[iy1:iy2, ix1:ix2]))
        return crops

    @staticmethod
    def _prepare_crop(crop):
        if crop is None or crop.size == 0:
            return crop
        scale = max(2.0, 96.0 / max(1.0, float(crop.shape[0])))
        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        return cv2.cvtColor(
            cv2.merge((lightness, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    def _run_ocr(self, crop):
        if crop is None or crop.size == 0:
            return []
        if self.backend_name == 'tesseract':
            return self._run_tesseract(crop)
        try:
            try:
                result = self.reader.ocr(crop, cls=True)
            except TypeError:
                result = self.reader.ocr(crop)
        except Exception:
            return []
        values = []

        def visit(node):
            if isinstance(node, (list, tuple)):
                if (
                    len(node) == 2
                    and isinstance(node[0], str)
                    and isinstance(node[1], (int, float))
                ):
                    values.append((node[0], float(node[1])))
                    return
                for child in node:
                    visit(child)

        visit(result)
        return values

    def _run_tesseract(self, crop):
        ok, encoded = cv2.imencode('.png', crop)
        if not ok:
            return []
        language = 'chi_sim+eng' if self.language in {'ch', 'zh', 'chi_sim'} else self.language
        try:
            completed = subprocess.run(
                [
                    str(self.reader),
                    'stdin',
                    'stdout',
                    '-l',
                    language,
                    '--psm',
                    '6',
                    'tsv',
                ],
                input=encoded.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        values = []
        text = completed.stdout.decode('utf-8', errors='ignore')
        for line in text.splitlines()[1:]:
            columns = line.split('\t')
            if len(columns) < 12:
                continue
            candidate = columns[11].strip()
            if not candidate:
                continue
            try:
                confidence = max(0.0, min(1.0, float(columns[10]) / 100.0))
            except ValueError:
                continue
            values.append((candidate, confidence))
        return values

    def _match_dictionary(self, normalized):
        if not self.dictionary:
            return normalized, 1.0
        if normalized in self.dictionary:
            return self.dictionary[normalized], 1.0
        best = None
        for alias, canonical in self.dictionary.items():
            if alias in normalized or normalized in alias:
                score = min(len(alias), len(normalized)) / max(len(alias), len(normalized))
            else:
                score = SequenceMatcher(None, normalized, alias).ratio()
            if best is None or score > best[1]:
                best = (canonical, score)
        if best is None or best[1] < 0.55:
            return normalized, 0.0
        return best
