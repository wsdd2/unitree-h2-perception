#!/usr/bin/env python3
import os
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from detector_msgs.msg import InspectionCommand, Object2D, Object2DArray
from yolo_trt_ros2.backends.mock_backend import MockBackend
from yolo_trt_ros2.control_label_ocr import ControlLabelOCR
from yolo_trt_ros2.work_tag_detector import is_work_tag_class, merge_work_tag_detections
from yolo_trt_ros2.aruco_detector import (
    detect_aruco_markers,
    draw_aruco_overlays,
    is_aruco_class,
)
from yolo_trt_ros2.yolo_seg_priority import (
    DEFAULT_SEG_CLASS_NAMES,
    YoloSegPriorityBackend,
    any_requested_covered,
    available_publish_class_names,
    draw_yolo_seg_overlays,
    merge_yolo_seg_priority,
    requested_activates_seg,
)

_CJK_FONT_CANDIDATES = (
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
)
_CJK_FONT_CACHE = {}


def _load_cjk_font(size_px):
    key = int(size_px)
    cached = _CJK_FONT_CACHE.get(key)
    if cached is not None or key in _CJK_FONT_CACHE:
        return cached
    try:
        from PIL import ImageFont
    except ImportError:
        _CJK_FONT_CACHE[key] = None
        return None
    for path in _CJK_FONT_CANDIDATES:
        if not os.path.isfile(path):
            continue
        try:
            font = ImageFont.truetype(path, key)
            _CJK_FONT_CACHE[key] = font
            return font
        except Exception:
            continue
    _CJK_FONT_CACHE[key] = None
    return None


def _put_text_bgr(image, text, org, color, font_scale=0.6, thickness=2):
    """Draw UTF-8 text; fall back to Hershey only for ASCII-only labels."""
    text = str(text)
    x, y = int(org[0]), int(org[1])
    if text.isascii():
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            color,
            int(thickness),
            cv2.LINE_AA,
        )
        return image
    font = _load_cjk_font(max(14, int(round(32 * float(font_scale)))))
    if font is None:
        # OpenCV Hershey cannot render CJK; keep a readable ASCII fallback.
        ascii_fallback = text.encode('ascii', errors='replace').decode('ascii')
        cv2.putText(
            image,
            ascii_fallback,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            color,
            int(thickness),
            cv2.LINE_AA,
        )
        return image
    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    # PIL uses top-left of text box; OpenCV putText uses baseline.
    ascent, _descent = font.getmetrics()
    fill = (int(color[2]), int(color[1]), int(color[0]))
    draw.text((x, max(0, y - ascent)), text, font=font, fill=fill)
    image[:, :, :] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return image


class YoloDetectorNode(Node):
    """Small ROS2 Foxy detector node with a swappable inference backend."""

    def __init__(self, subscribe_image=True, enable_debug_output=None):
        super().__init__('yolo_detector')

        self._declare_parameters()
        self.image_topic = self.get_parameter('image_topic').value
        self.objects_topic = self.get_parameter('objects_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.engine_path = self.get_parameter('engine_path').value
        self.model_path = self.get_parameter('model_path').value
        self.model_kind = str(self.get_parameter('model_kind').value).lower()
        self.class_names_path = self.get_parameter('class_names_path').value
        self.yoloe_classes_path = str(self.get_parameter('yoloe_classes_path').value)
        self.input_width = int(self.get_parameter('input_width').value)
        self.input_height = int(self.get_parameter('input_height').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        imgsz_height = int(self.get_parameter('imgsz_height').value)
        imgsz_width = int(self.get_parameter('imgsz_width').value)
        if imgsz_height > 0 and imgsz_width > 0:
            self.imgsz = [imgsz_height, imgsz_width]
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.iou_thres = float(self.get_parameter('iou_thres').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        if enable_debug_output is not None:
            self.publish_debug_image = bool(enable_debug_output)
        self.backend_name = str(self.get_parameter('backend').value).lower()
        self.device = str(self.get_parameter('device').value)
        self.prompt_free = bool(self.get_parameter('prompt_free').value)
        self.best_handle_only = bool(self.get_parameter('best_handle_only').value)
        self.filter_prompt_free_handles = bool(self.get_parameter('filter_prompt_free_handles').value)
        self.mobileclip_path = str(self.get_parameter('mobileclip_path').value)
        self.detect_blue_point = bool(self.get_parameter('detect_blue_point').value)
        self.detect_work_tag = bool(self.get_parameter('detect_work_tag').value)
        self.detect_aruco = bool(self.get_parameter('detect_aruco').value)
        self.aruco_dictionary = str(self.get_parameter('aruco_dictionary').value).strip()
        self.aruco_confidence = float(self.get_parameter('aruco_confidence').value)
        self.aruco_min_side_px = float(self.get_parameter('aruco_min_side_px').value)
        self.blue_point_class_name = str(self.get_parameter('blue_point_class_name').value)
        self.press_point_mode = str(self.get_parameter('press_point_mode').value).strip().lower()
        self.press_point_vertical_ratio = max(
            0.0,
            min(1.0, float(self.get_parameter('press_point_vertical_ratio').value)),
        )
        self.blue_point_min_area = float(self.get_parameter('blue_point_min_area').value)
        self.blue_point_max_area_ratio = float(self.get_parameter('blue_point_max_area_ratio').value)
        self.lock_min_dark_surround_ratio = max(
            0.0,
            min(
                1.0,
                float(self.get_parameter('lock_min_dark_surround_ratio').value),
            ),
        )
        self.lock_dark_value_max = max(
            0,
            min(255, int(self.get_parameter('lock_dark_value_max').value)),
        )
        self.roi_mean_filter_enabled = bool(self.get_parameter('roi_mean_filter_enabled').value)
        self.roi_mean_filter_window = max(1, int(self.get_parameter('roi_mean_filter_window').value))
        self.roi_mean_filter_max_jump_px = max(
            1.0,
            float(self.get_parameter('roi_mean_filter_max_jump_px').value),
        )
        self.roi_mean_filter_max_missed = max(
            0,
            int(self.get_parameter('roi_mean_filter_max_missed').value),
        )
        self.control_ocr_enabled = bool(self.get_parameter('control_ocr_enabled').value)
        self.lock_overlap_suppression_enabled = bool(
            self.get_parameter('lock_overlap_suppression_enabled').value
        )
        self.lock_overlap_suppression_padding_px = max(
            0.0,
            float(self.get_parameter('lock_overlap_suppression_padding_px').value),
        )
        self.handle_max_bbox_area_ratio = max(
            0.01,
            float(self.get_parameter('handle_max_bbox_area_ratio').value),
        )
        self.handle_max_bbox_height_ratio = max(
            0.05,
            float(self.get_parameter('handle_max_bbox_height_ratio').value),
        )
        self.detector_mode = str(self.get_parameter('detector_mode').value).strip().lower()
        self.enable_yoloe = bool(self.get_parameter('enable_yoloe').value)
        self.yolo_seg_run_mode = str(
            self.get_parameter('yolo_seg_run_mode').value
        ).strip().lower()
        # Legacy flag: false forces SEG off unless detector_mode / run_mode says otherwise.
        self.yolo_seg_enabled = bool(self.get_parameter('yolo_seg_enabled').value)
        self.enable_yoloe, self.yolo_seg_run_mode = self._resolve_detector_stack(
            self.detector_mode,
            self.enable_yoloe,
            self.yolo_seg_run_mode,
            self.yolo_seg_enabled,
        )
        self.yolo_seg_enabled = self.yolo_seg_run_mode != 'off'
        self.yolo_seg_model_path = str(self.get_parameter('yolo_seg_model_path').value)
        self.yolo_seg_priority_mode = str(
            self.get_parameter('yolo_seg_priority_mode').value
        ).strip().lower()
        self.yolo_seg_conf_thres = float(self.get_parameter('yolo_seg_conf_thres').value)
        self.yolo_seg_iou_thres = float(self.get_parameter('yolo_seg_iou_thres').value)
        self.yolo_seg_imgsz = int(self.get_parameter('yolo_seg_imgsz').value)
        self.yolo_seg_mask_smooth_ksize = int(
            self.get_parameter('yolo_seg_mask_smooth_ksize').value
        )
        self.yolo_seg_mask_close_iters = int(
            self.get_parameter('yolo_seg_mask_close_iters').value
        )
        self.yolo_seg_lock_ground_ratio = float(
            self.get_parameter('yolo_seg_lock_ground_ratio').value
        )
        self.yolo_seg_opencv_color = bool(
            self.get_parameter('yolo_seg_opencv_color').value
        )
        self.yolo_seg_inspection_command_topic = str(
            self.get_parameter('yolo_seg_inspection_command_topic').value
        )
        self.yolo_seg_covered_classes = self._parse_list_value(
            self.get_parameter('yolo_seg_covered_classes').value
        ) or list(DEFAULT_SEG_CLASS_NAMES)
        self._requested_class_names = []
        self._yolo_seg_backend = None
        self._last_yolo_seg_detections = []
        self.prompts = self._load_yoloe_classes(self.yoloe_classes_path)
        self.published_class_names = self._load_text_list(
            self.yoloe_classes_path,
            'yoloe_classes_path',
            include_publish_only=True,
        )
        if self.published_class_names:
            self.get_logger().info(
                'Canonical class registry contains %d prompt/generated names.'
                % len(self.published_class_names)
            )
        if not self.prompts:
            self.prompts = self._parse_list_value(self.get_parameter('prompts').value)

        self.bridge = CvBridge()
        self._roi_mean_tracks = {'press': [], 'handle': [], 'work_tag': []}
        self.class_names = self._load_class_names(self.class_names_path)
        self.backend = self._create_backend()
        self.control_ocr = ControlLabelOCR(
            logger=self.get_logger(),
            enabled=self.control_ocr_enabled,
            backend=str(self.get_parameter('control_ocr_backend').value),
            language=str(self.get_parameter('control_ocr_language').value),
            dictionary_path=str(self.get_parameter('control_ocr_dictionary_path').value),
            interval_frames=int(self.get_parameter('control_ocr_interval_frames').value),
            min_confidence=float(self.get_parameter('control_ocr_min_confidence').value),
            stable_frames=int(self.get_parameter('control_ocr_stable_frames').value),
            max_missed=int(self.get_parameter('control_ocr_max_missed').value),
            max_controls_per_frame=int(self.get_parameter('control_ocr_max_controls_per_frame').value),
            roi_width_scale=float(self.get_parameter('control_ocr_roi_width_scale').value),
            roi_height_scale=float(self.get_parameter('control_ocr_roi_height_scale').value),
            row_alignment_tolerance_px=float(self.get_parameter('control_row_alignment_tolerance_px').value),
            label_tag_brightness_delta=float(self.get_parameter('control_label_tag_brightness_delta').value),
        )

        self.objects_pub = self.create_publisher(Object2DArray, self.objects_topic, 10)
        self.debug_pub = None
        if self.publish_debug_image and subscribe_image:
            self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.image_sub = None
        if subscribe_image:
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self._image_callback,
                # Inference is slower than the camera. A deep queue makes YOLO
                # process old color frames whose matching depth has already gone.
                # Keep only the newest frame instead of accumulating latency.
                1,
            )
        if self.yolo_seg_run_mode in ('always', 'on_request'):
            if self.yolo_seg_run_mode == 'always':
                self._ensure_yolo_seg_backend()
            self.create_subscription(
                InspectionCommand,
                self.yolo_seg_inspection_command_topic,
                self._inspection_command_callback,
                10,
            )

        self.get_logger().info(
            'YOLO detector started: backend=%s, enable_yoloe=%s, yolo_seg_run_mode=%s, '
            'yolo_seg_opencv_color=%s, detect_aruco=%s, detector_mode=%s, input=%s, '
            'objects_topic=%s, debug=%s'
            % (
                self.backend_name if self.enable_yoloe else 'skipped',
                self.enable_yoloe,
                self.yolo_seg_run_mode,
                'ON' if self.yolo_seg_opencv_color else 'OFF',
                'ON' if self.detect_aruco else 'OFF',
                self.detector_mode or '<granular>',
                self.image_topic if subscribe_image else 'in-process RGB',
                self.objects_topic,
                self.publish_debug_image,
            )
        )
        if self.yolo_seg_enabled:
            available = available_publish_class_names(
                self.yolo_seg_covered_classes,
                opencv_color_enabled=self.yolo_seg_opencv_color,
            )
            self.get_logger().info(
                'YOLOSeg OpenCV color semantics: %s | available class names (%d): %s'
                % (
                    'ON' if self.yolo_seg_opencv_color else 'OFF',
                    len(available),
                    ', '.join(available),
                )
            )


    def camera_callback(self, image_msg):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error('Failed to convert ROS Image to OpenCV: %s' % exc)
            return

        self.process_frame(bgr_image, image_msg.header)
        if debug_image is not None and self.debug_pub is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.debug_pub.publish(debug_msg)

        return bgr_image
    def _declare_parameters(self):
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('objects_topic', '/detector/objects_2d')
        self.declare_parameter('debug_image_topic', '/detector/debug_image')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_kind', 'auto')
        self.declare_parameter('class_names_path', '')
        self.declare_parameter('yoloe_classes_path', '')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('imgsz_height', 0)
        self.declare_parameter('imgsz_width', 0)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('backend', 'mock')
        self.declare_parameter('device', '')
        self.declare_parameter('prompt_free', False)
        self.declare_parameter('best_handle_only', False)
        self.declare_parameter('filter_prompt_free_handles', False)
        self.declare_parameter('mobileclip_path', '')
        self.declare_parameter('detect_blue_point', False)
        self.declare_parameter('detect_work_tag', False)
        self.declare_parameter('detect_aruco', False)
        self.declare_parameter('aruco_dictionary', 'DICT_6X6_250')
        self.declare_parameter('aruco_confidence', 0.99)
        self.declare_parameter('aruco_min_side_px', 8.0)
        self.declare_parameter('blue_point_class_name', 'blue circle push point')
        self.declare_parameter('press_point_mode', 'blue')
        self.declare_parameter('press_point_vertical_ratio', 0.5)
        self.declare_parameter('blue_point_min_area', 40.0)
        self.declare_parameter('blue_point_max_area_ratio', 0.02)
        self.declare_parameter('lock_min_dark_surround_ratio', 0.28)
        self.declare_parameter('lock_dark_value_max', 90)
        self.declare_parameter('roi_mean_filter_enabled', False)
        self.declare_parameter('roi_mean_filter_window', 3)
        self.declare_parameter('roi_mean_filter_max_jump_px', 40.0)
        self.declare_parameter('roi_mean_filter_max_missed', 2)
        self.declare_parameter('control_ocr_enabled', False)
        self.declare_parameter('control_ocr_backend', 'tesseract')
        self.declare_parameter('control_ocr_language', 'chi_sim')
        self.declare_parameter('control_ocr_dictionary_path', '')
        self.declare_parameter('control_ocr_interval_frames', 6)
        self.declare_parameter('control_ocr_min_confidence', 0.65)
        self.declare_parameter('control_ocr_stable_frames', 3)
        self.declare_parameter('control_ocr_max_missed', 30)
        self.declare_parameter('control_ocr_max_controls_per_frame', 8)
        self.declare_parameter('control_ocr_roi_width_scale', 2.6)
        self.declare_parameter('control_ocr_roi_height_scale', 1.8)
        self.declare_parameter('control_row_alignment_tolerance_px', 28.0)
        self.declare_parameter('control_label_tag_brightness_delta', 18.0)
        self.declare_parameter('lock_overlap_suppression_enabled', True)
        self.declare_parameter('lock_overlap_suppression_padding_px', 8.0)
        self.declare_parameter('handle_max_bbox_area_ratio', 0.12)
        self.declare_parameter('handle_max_bbox_height_ratio', 0.65)
        # detector_mode shortcuts (override enable_yoloe / yolo_seg_run_mode when set):
        #   yoloe | yoloseg | hybrid | seg_on_request
        self.declare_parameter('detector_mode', '')
        self.declare_parameter('enable_yoloe', True)
        # off | always | on_request  (when to run YOLO-seg inference)
        self.declare_parameter('yolo_seg_run_mode', 'always')
        self.declare_parameter('yolo_seg_enabled', False)
        self.declare_parameter('yolo_seg_model_path', '')
        self.declare_parameter('yolo_seg_priority_mode', 'on_request')
        self.declare_parameter('yolo_seg_conf_thres', 0.25)
        self.declare_parameter('yolo_seg_iou_thres', 0.45)
        self.declare_parameter('yolo_seg_imgsz', 1280)
        self.declare_parameter('yolo_seg_mask_smooth_ksize', 5)
        self.declare_parameter('yolo_seg_mask_close_iters', 1)
        self.declare_parameter('yolo_seg_lock_ground_ratio', 0.80)
        self.declare_parameter('yolo_seg_opencv_color', False)
        self.declare_parameter('yolo_seg_inspection_command_topic', '/robot/inspection_command')
        self.declare_parameter(
            'yolo_seg_covered_classes',
            ','.join(DEFAULT_SEG_CLASS_NAMES),
        )
        self.declare_parameter(
            'prompts',
            'lever door handle,horizontal door handle,door lever handle,pull door handle',
        )

    def _parse_list_value(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def _load_yoloe_classes(self, path):
        classes = self._load_text_list(path, 'yoloe_classes_path')
        if classes:
            self.get_logger().info('Loaded %d YOLOE classes from %s' % (len(classes), path))
        return classes

    def _load_class_names(self, class_names_path):
        return self._load_text_list(class_names_path, 'class_names_path')

    def _load_text_list(self, path, label, include_publish_only=False):
        if not path:
            return []
        expanded = os.path.expanduser(str(path))
        if not os.path.exists(expanded):
            self.get_logger().warn('%s does not exist: %s' % (label, expanded))
            return []

        values = []
        with open(expanded, 'r', encoding='utf-8') as names_file:
            for line in names_file:
                text = line.strip()
                if not text or text.startswith('#'):
                    continue
                if '#' in text:
                    text = text.split('#', 1)[0].strip()
                publish_only = text.lower().startswith('publish:')
                if publish_only:
                    if not include_publish_only:
                        continue
                    text = text.split(':', 1)[1].strip()
                for item in text.split(','):
                    name = item.strip()
                    if name:
                        values.append(name)
        return values

    def _create_backend(self):
        if not self.enable_yoloe:
            self.get_logger().info('Primary YOLOE/Ultralytics backend disabled (enable_yoloe:=false).')

            class _EmptyBackend(object):
                def infer(self, _bgr_image):
                    return []

            return _EmptyBackend()

        if self.backend_name == 'mock':
            return MockBackend(self.class_names)

        if self.backend_name == 'tensorrt':
            # Import only here so mock mode does not need TensorRT installed.
            from yolo_trt_ros2.backends.tensorrt_backend import TensorRTBackend

            return TensorRTBackend(
                engine_path=self.engine_path,
                class_names=self.class_names,
                input_width=self.input_width,
                input_height=self.input_height,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )

        if self.backend_name in ('ultralytics', 'yolo', 'yoloe'):
            from yolo_trt_ros2.backends.ultralytics_backend import UltralyticsBackend

            model_kind = self.model_kind
            if self.backend_name in ('yolo', 'yoloe'):
                model_kind = self.backend_name
            return UltralyticsBackend(
                model_path=self.model_path,
                model_kind=model_kind,
                class_names=self.class_names,
                prompts=self.prompts,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                imgsz=self.imgsz,
                device=self.device,
                prompt_free=self.prompt_free,
                best_handle_only=self.best_handle_only,
                mobileclip_path=self.mobileclip_path,
                filter_prompt_free_handles=self.filter_prompt_free_handles,
            )

        raise ValueError('Unsupported backend: %s. Use mock, yoloe, yolo, ultralytics or tensorrt.' % self.backend_name)

    def _resolve_detector_stack(self, detector_mode, enable_yoloe, run_mode, legacy_seg_enabled):
        """Map CLI/yaml presets into (enable_yoloe, yolo_seg_run_mode)."""
        presets = {
            'yoloe': (True, 'off'),
            'yoloseg': (False, 'always'),
            'seg': (False, 'always'),
            'hybrid': (True, 'always'),
            'both': (True, 'always'),
            'seg_on_request': (True, 'on_request'),
            'on_request': (True, 'on_request'),
        }
        mode = str(detector_mode or '').strip().lower()
        if mode in presets:
            return presets[mode]

        resolved_run = str(run_mode or '').strip().lower()
        if resolved_run not in ('off', 'always', 'on_request'):
            resolved_run = 'always' if bool(legacy_seg_enabled) else 'off'
        return bool(enable_yoloe), resolved_run

    def _image_callback(self, image_msg):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error('Failed to convert ROS Image to OpenCV: %s' % exc)
            return

        _objects_msg, debug_image = self.process_frame(bgr_image, image_msg.header)
        if debug_image is not None and self.debug_pub is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.debug_pub.publish(debug_msg)

    def process_frame(self, bgr_image, header, publish_objects=True):
        """Run detection on an in-memory BGR frame and publish only Object2D."""
        if self.enable_yoloe:
            try:
                detections = self.backend.infer(bgr_image)
            except Exception as exc:
                self.get_logger().error('Backend inference failed: %s' % exc)
                return None, None
            detections = self._filter_implausible_handle_boxes(
                detections,
                bgr_image.shape[:2],
            )
            if self.detect_blue_point:
                # Legacy blue-marker mode: HSV owns the target and replaces any
                # model-native press detections.
                detections = [
                    det
                    for det in detections
                    if not self._is_press_point_name(det.get('class_name', ''))
                ]
                detections = list(detections) + self._detect_press_point(bgr_image, detections)
                detections = self._suppress_buttons_overlapping_lock(detections)
            else:
                # White-tape mode: keep the model detection but expose the physical
                # marker's actual color/shape semantics to downstream consumers.
                detections = [dict(det) for det in detections]
                for det in detections:
                    if self._is_press_point_name(det.get('class_name', '')):
                        det['class_name'] = self.blue_point_class_name
            if not any(self._is_handle_detection(det) for det in detections):
                press = self._best_blue_detection(detections)
                fallback_handle = self._detect_black_handle(bgr_image, press)
                if fallback_handle is not None:
                    detections = list(detections) + [fallback_handle]
            if self.detect_work_tag:
                # Keep non-hang-tag YOLOE hits; fuse hang-tag YOLOE prompts with
                # OpenCV cord/grasp geometry (YOLOE preferred for the hook).
                model_keep = [
                    det
                    for det in detections
                    if not is_work_tag_class(det.get('class_name', ''))
                ]
                detections = list(model_keep) + merge_work_tag_detections(bgr_image, detections)
        else:
            detections = []

        detections = self._merge_yolo_seg_priority(bgr_image, detections)
        if self.detect_aruco:
            aruco_dets = detect_aruco_markers(
                bgr_image,
                dictionary_name=self.aruco_dictionary,
                confidence=self.aruco_confidence,
                min_side_px=self.aruco_min_side_px,
            )
            if aruco_dets:
                # Drop any prior same-ID ArUco hits; OpenCV owns this family.
                detections = [
                    det
                    for det in detections
                    if not is_aruco_class(det.get('class_name', ''))
                ] + list(aruco_dets)
        detections = self._attach_handle_grasp_edges(bgr_image, detections)
        detections = self._smooth_detection_rois(detections)
        detections = self.control_ocr.annotate(bgr_image, detections)

        objects_msg = self._build_objects_msg(header, detections)
        if publish_objects:
            self.objects_pub.publish(objects_msg)

        debug_image = None
        if self.publish_debug_image:
            debug_image = self._draw_debug_image(bgr_image.copy(), detections)
        return objects_msg, debug_image

    def _inspection_command_callback(self, msg):
        self._requested_class_names = [
            str(value).strip()
            for value in getattr(msg, 'requested_class_names', [])
            if str(value).strip()
        ]
        active = str(getattr(msg, 'active_target_class_name', '') or '').strip()
        if active and active not in self._requested_class_names:
            self._requested_class_names.append(active)

    def _ensure_yolo_seg_backend(self):
        if self._yolo_seg_backend is not None:
            return self._yolo_seg_backend
        if not self.yolo_seg_model_path:
            self.get_logger().warn('yolo_seg_enabled but yolo_seg_model_path is empty')
            return None
        try:
            self._yolo_seg_backend = YoloSegPriorityBackend(
                model_path=self.yolo_seg_model_path,
                class_names=self.yolo_seg_covered_classes,
                conf_thres=self.yolo_seg_conf_thres,
                iou_thres=self.yolo_seg_iou_thres,
                imgsz=self.yolo_seg_imgsz,
                device=self.device,
                mask_smooth_ksize=self.yolo_seg_mask_smooth_ksize,
                mask_close_iters=self.yolo_seg_mask_close_iters,
                lock_ground_ratio=self.yolo_seg_lock_ground_ratio,
                enable_opencv_color=self.yolo_seg_opencv_color,
                logger=self.get_logger(),
            )
            self.get_logger().info(
                'YOLO-seg priority backend ready: %s mode=%s opencv_color=%s'
                % (
                    self.yolo_seg_model_path,
                    self.yolo_seg_priority_mode,
                    'ON' if self.yolo_seg_opencv_color else 'OFF',
                )
            )
        except Exception as exc:
            self.get_logger().error('Failed to init YOLO-seg priority backend: %s' % exc)
            self._yolo_seg_backend = None
        return self._yolo_seg_backend

    def _active_yolo_seg_classes(self):
        if self.yolo_seg_run_mode == 'on_request':
            requested = list(self._requested_class_names)
            return [
                name
                for name in self.yolo_seg_covered_classes
                if requested_activates_seg(requested, name)
            ]
        return list(self.yolo_seg_covered_classes)

    def _merge_yolo_seg_priority(self, bgr_image, detections):
        self._last_yolo_seg_detections = []
        if self.yolo_seg_run_mode == 'off':
            return list(detections)
        if self.yolo_seg_run_mode == 'on_request':
            if not any_requested_covered(
                self._requested_class_names,
                self.yolo_seg_covered_classes,
            ):
                return list(detections)
        active = self._active_yolo_seg_classes()
        if not active:
            return list(detections)
        backend = self._ensure_yolo_seg_backend()
        if backend is None:
            return list(detections)
        try:
            seg_dets = backend.infer(bgr_image, active_seg_classes=active)
        except Exception as exc:
            self.get_logger().warn('YOLO-seg inference failed: %s' % exc)
            return list(detections)
        # Seg-only stack: publish every SEG hit. Hybrid: replace only requested families.
        merge_mode = 'always' if not self.enable_yoloe else self.yolo_seg_priority_mode
        merged, preview = merge_yolo_seg_priority(
            detections,
            seg_dets,
            self._requested_class_names,
            mode=merge_mode,
        )
        self._last_yolo_seg_detections = list(preview)
        return merged


    def _build_objects_msg(self, header, detections):
        msg = Object2DArray()
        msg.header = header

        for det in detections:
            obj = Object2D()
            obj.class_name = str(det.get('class_name', 'unknown'))
            obj.class_id = int(det.get('class_id', -1))
            obj.confidence = float(det.get('confidence', 0.0))
            obj.xmin = int(det.get('xmin', 0))
            obj.ymin = int(det.get('ymin', 0))
            obj.xmax = int(det.get('xmax', 0))
            obj.ymax = int(det.get('ymax', 0))
            obj.cx = float(det.get('cx', float(obj.xmin + obj.xmax) * 0.5))
            obj.cy = float(det.get('cy', float(obj.ymin + obj.ymax) * 0.5))
            edge_px = det.get('handle_grasp_edge_px') or []
            center_px = det.get('handle_grasp_center_px') or []
            obj.handle_grasp_edge_px = [float(v) for point in edge_px for v in point]
            obj.handle_grasp_center_px = [float(v) for v in center_px]
            obj.handle_grasp_width_px = float(det.get('handle_grasp_width_px', 0.0))
            obj.handle_grasp_source = str(det.get('handle_grasp_source', ''))
            obj.label_text = str(det.get('label_text', ''))
            obj.label_confidence = float(det.get('label_confidence', 0.0))
            obj.semantic_name = str(det.get('semantic_name', ''))
            obj.control_id = str(det.get('control_id', ''))
            obj.spatial_relation = str(det.get('spatial_relation', ''))
            obj.label_tag_present = bool(det.get('label_tag_present', False))
            geom = det.get('geometric_center_px') or []
            if len(geom) >= 2:
                obj.geometric_center_px = [float(geom[0]), float(geom[1])]
            msg.objects.append(obj)

        return msg

    def _draw_debug_image(self, image, detections):
        for det in detections:
            xmin = int(det.get('xmin', 0))
            ymin = int(det.get('ymin', 0))
            xmax = int(det.get('xmax', 0))
            ymax = int(det.get('ymax', 0))
            class_name = str(det.get('class_name', 'unknown'))
            display_name = str(det.get('semantic_name', '') or class_name)
            confidence = float(det.get('confidence', 0.0))
            cx = int(round(float(det.get('cx', float(xmin + xmax) * 0.5))))
            cy = int(round(float(det.get('cy', float(ymin + ymax) * 0.5))))

            is_press = self._is_press_point_name(class_name)
            if is_aruco_class(class_name):
                color = (255, 0, 255)
                center_color = (255, 0, 255)
            elif is_work_tag_class(class_name):
                if 'grasp' in class_name.lower():
                    color = (0, 255, 255)
                    center_color = (0, 255, 255)
                elif 'hook' in class_name.lower():
                    color = (255, 128, 0)
                    center_color = (255, 128, 0)
                elif 'cord' in class_name.lower():
                    color = (0, 0, 255)
                    center_color = (0, 0, 255)
                else:
                    color = (0, 200, 0)
                    center_color = (0, 200, 0)
            elif is_press:
                color = (255, 128, 0)
                center_color = (255, 0, 0)
            else:
                color = (0, 255, 0)
                center_color = (0, 0, 255)
            if not is_aruco_class(class_name):
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.circle(image, (cx, cy), 5, center_color, -1)
            edge_px = det.get('handle_grasp_edge_px') or []
            if len(edge_px) == 2 and not is_aruco_class(class_name):
                p0 = (int(edge_px[0][0]), int(edge_px[0][1]))
                p1 = (int(edge_px[1][0]), int(edge_px[1][1]))
                cv2.line(image, p0, p1, (0, 0, 255), 3)
                cv2.circle(image, p0, 5, (0, 255, 255), -1)
                cv2.circle(image, p1, 5, (0, 255, 255), -1)
            center_px = det.get('handle_grasp_center_px') or []
            if len(center_px) == 2 and not is_aruco_class(class_name):
                cv2.drawMarker(
                    image,
                    (int(center_px[0]), int(center_px[1])),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=22,
                    thickness=2,
                )
            if is_aruco_class(class_name):
                continue
            label = '%s %.2f' % (display_name, confidence)
            
            if str(det.get('detection_source', '')).startswith('yoloseg') or str(
                det.get('handle_grasp_source', '')
            ).startswith('yoloseg'):
                label = 'SEG ' + label
            _put_text_bgr(
                image,
                label,
                (xmin, max(0, ymin - 8)),
                color,
                font_scale=0.6,
                thickness=2,
            )
        draw_yolo_seg_overlays(image, list(detections) + list(self._last_yolo_seg_detections))
        draw_aruco_overlays(image, detections)
        return image

    def _attach_handle_grasp_edges(self, image, detections):
        if image is None or image.size == 0:
            return list(detections)
        detections = [dict(det) for det in detections]
        blue = self._best_blue_detection(detections)
        for det in detections:
            if not self._is_handle_detection(det):
                continue
            if str(det.get('handle_grasp_source', '')).startswith('yoloseg'):
                continue
            grasp = self._estimate_handle_grasp_edge(image, det, blue)
            if grasp:
                det.update(grasp)
        return detections

    def _best_blue_detection(self, detections):
        blue = [
            det for det in detections
            if self._is_press_point_name(det.get('class_name', ''))
        ]
        if not blue:
            return None
        return max(blue, key=lambda det: float(det.get('confidence', 0.0)))

    def _is_press_point_name(self, class_name):
        name = str(class_name or '').lower().replace('_', ' ').replace('-', ' ')
        return (
            'blue' in name
            or 'circle push point' in name
            or 'white square push point' in name
            or 'red sticker push point' in name
            or 'lock point' in name
        )

    def _suppress_buttons_overlapping_lock(self, detections):
        if not self.lock_overlap_suppression_enabled:
            return list(detections)
        lock_detections = [
            det
            for det in detections
            if 'lock point' in str(det.get('class_name', '')).lower()
        ]
        if not lock_detections:
            return list(detections)

        kept = []
        for det in detections:
            name = str(det.get('class_name', '')).lower()
            if 'push button' not in name:
                kept.append(det)
                continue
            x1 = float(det.get('xmin', 0.0)) - self.lock_overlap_suppression_padding_px
            y1 = float(det.get('ymin', 0.0)) - self.lock_overlap_suppression_padding_px
            x2 = float(det.get('xmax', 0.0)) + self.lock_overlap_suppression_padding_px
            y2 = float(det.get('ymax', 0.0)) + self.lock_overlap_suppression_padding_px
            overlaps_lock = any(
                x1 <= float(lock.get('cx', 0.0)) <= x2
                and y1 <= float(lock.get('cy', 0.0)) <= y2
                for lock in lock_detections
            )
            if not overlaps_lock:
                kept.append(det)
        return kept

    def _is_handle_detection(self, det):
        name = str(det.get('class_name', '')).lower()
        return any(token in name for token in ('handle', 'lever', 'pull', 'cabinet door'))

    def _filter_implausible_handle_boxes(self, detections, image_shape):
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        image_area = max(1.0, float(image_w * image_h))
        kept = []
        for det in detections:
            if not self._is_handle_detection(det):
                kept.append(det)
                continue
            width = max(0.0, float(det.get('xmax', 0)) - float(det.get('xmin', 0)))
            height = max(0.0, float(det.get('ymax', 0)) - float(det.get('ymin', 0)))
            area_ratio = width * height / image_area
            height_ratio = height / max(1.0, float(image_h))
            if (
                area_ratio > self.handle_max_bbox_area_ratio
                or height_ratio > self.handle_max_bbox_height_ratio
            ):
                continue
            kept.append(det)
        return kept

    def _detect_black_handle(self, image, press, min_area=180.0):
        if image is None or image.size == 0:
            return None
        img_h, img_w = image.shape[:2]
        if press is not None:
            bx = int(round(float(press.get('cx', 0.0))))
            by = int(round(float(press.get('cy', 0.0))))
            rx1, ry1 = max(0, bx - 260), max(0, by - 360)
            rx2, ry2 = min(img_w, bx + 260), min(img_h, by + 360)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, img_w, img_h

        crop = image[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 85))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(min_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 25:
                continue
            aspect = float(w) / float(h)
            elongation = max(aspect, 1.0 / max(aspect, 1e-6))
            if elongation < 1.6:
                continue
            if w > img_w * 0.75 or h > img_h * 0.85:
                continue
            gx1, gy1, gx2, gy2 = rx1 + x, ry1 + y, rx1 + x + w, ry1 + y + h
            score = area * elongation
            if press is not None:
                bx = float(press.get('cx', 0.0))
                by = float(press.get('cy', 0.0))
                if gx1 - 30 <= bx <= gx2 + 30 and gy1 - 30 <= by <= gy2 + 30:
                    score *= 2.0
                distance = np.hypot(float((gx1 + gx2) * 0.5 - bx), float((gy1 + gy2) * 0.5 - by))
                score *= 1.0 / max(1.0, distance / 160.0)
            candidates.append(
                (
                    score,
                    {
                        'class_name': 'black cabinet door handle',
                        'class_id': 9002,
                        'confidence': 0.55,
                        'xmin': int(gx1),
                        'ymin': int(gy1),
                        'xmax': int(gx2),
                        'ymax': int(gy2),
                        'cx': float(gx1 + gx2) * 0.5,
                        'cy': float(gy1 + gy2) * 0.5,
                        'source': 'opencv_black',
                    },
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _smooth_detection_rois(self, detections):
        detections = [dict(det) for det in detections]
        if not self.roi_mean_filter_enabled or self.roi_mean_filter_window <= 1:
            return detections

        for tracks in self._roi_mean_tracks.values():
            for track in tracks:
                track['missed'] += 1

        used_tracks = {'press': set(), 'handle': set(), 'work_tag': set()}
        for det in detections:
            kind = self._roi_filter_kind(det)
            if kind is None:
                continue
            center = self._detection_center(det)
            tracks = self._roi_mean_tracks[kind]
            best_index = None
            best_distance = None
            for index, track in enumerate(tracks):
                if index in used_tracks[kind]:
                    continue
                distance = float(np.linalg.norm(center - track['raw_center']))
                if distance > self.roi_mean_filter_max_jump_px:
                    continue
                if best_distance is None or distance < best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None:
                track = {
                    'raw_center': center,
                    'history': deque(maxlen=self.roi_mean_filter_window),
                    'missed': 0,
                }
                tracks.append(track)
                best_index = len(tracks) - 1
            else:
                track = tracks[best_index]
                track['raw_center'] = center
                track['missed'] = 0

            used_tracks[kind].add(best_index)
            track['history'].append(self._roi_geometry(det))
            self._apply_mean_geometry(det, track['history'])

        for kind, tracks in self._roi_mean_tracks.items():
            self._roi_mean_tracks[kind] = [
                track
                for track in tracks
                if track['missed'] <= self.roi_mean_filter_max_missed
            ]
        return detections

    def _roi_filter_kind(self, det):
        if self._is_press_point_name(det.get('class_name', '')):
            return 'press'
        if self._is_handle_detection(det):
            return 'handle'
        if is_work_tag_class(det.get('class_name', '')):
            return 'work_tag'
        return None

    def _detection_center(self, det):
        return np.asarray(
            [
                float(det.get('cx', (float(det.get('xmin', 0)) + float(det.get('xmax', 0))) * 0.5)),
                float(det.get('cy', (float(det.get('ymin', 0)) + float(det.get('ymax', 0))) * 0.5)),
            ],
            dtype=np.float64,
        )

    def _roi_geometry(self, det):
        center = self._detection_center(det)
        geometry = {
            key: np.asarray([float(det.get(key, 0.0))], dtype=np.float64)
            for key in ('xmin', 'ymin', 'xmax', 'ymax')
        }
        geometry['cx'] = np.asarray([center[0]], dtype=np.float64)
        geometry['cy'] = np.asarray([center[1]], dtype=np.float64)
        edge_px = det.get('handle_grasp_edge_px') or []
        if len(edge_px) == 2:
            geometry['handle_grasp_edge_px'] = np.asarray(edge_px, dtype=np.float64).reshape(4)
        center_px = det.get('handle_grasp_center_px') or []
        if len(center_px) == 2:
            geometry['handle_grasp_center_px'] = np.asarray(center_px, dtype=np.float64).reshape(2)
        return geometry

    def _apply_mean_geometry(self, det, history):
        for key in ('xmin', 'ymin', 'xmax', 'ymax', 'cx', 'cy'):
            values = [entry[key] for entry in history if key in entry]
            if not values:
                continue
            mean_value = float(np.mean(np.stack(values, axis=0)))
            det[key] = int(round(mean_value)) if key in ('xmin', 'ymin', 'xmax', 'ymax') else mean_value

        for key, shape in (('handle_grasp_edge_px', (2, 2)), ('handle_grasp_center_px', (2,))):
            if not det.get(key):
                continue
            values = [entry[key] for entry in history if key in entry]
            if not values:
                continue
            mean_value = np.mean(np.stack(values, axis=0), axis=0).reshape(shape)
            det[key] = np.rint(mean_value).astype(int).tolist()

        edge_px = det.get('handle_grasp_edge_px') or []
        if len(edge_px) == 2:
            det['handle_grasp_width_px'] = float(
                np.linalg.norm(np.asarray(edge_px[1], dtype=np.float64) - np.asarray(edge_px[0], dtype=np.float64))
            )

    def _estimate_handle_grasp_edge(self, image, handle, blue, min_area=220.0, inset_ratio=0.10):
        img_h, img_w = image.shape[:2]
        x1 = int(handle.get('xmin', 0))
        y1 = int(handle.get('ymin', 0))
        x2 = int(handle.get('xmax', 0))
        y2 = int(handle.get('ymax', 0))
        pad_x = max(8, int((x2 - x1) * 0.08))
        pad_y = max(8, int((y2 - y1) * 0.08))
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(img_w, x2 + pad_x)
        ry2 = min(img_h, y2 + pad_y)
        crop = image[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 92))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        blue_pt = None
        if blue is not None:
            blue_pt = np.asarray(
                [
                    float(blue.get('cx', 0.0)),
                    float(blue.get('cy', 0.0)),
                ],
                dtype=np.float32,
            )

        scored = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            rect = cv2.minAreaRect(contour)
            center_local = np.asarray(rect[0], dtype=np.float32)
            rect_w, rect_h = float(rect[1][0]), float(rect[1][1])
            rect_short = max(1.0, min(rect_w, rect_h))
            rect_long = max(rect_w, rect_h)
            if rect_long / rect_short < 2.0:
                continue

            angle = np.deg2rad(float(rect[2]))
            axis_w = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            axis_h = np.asarray([-np.sin(angle), np.cos(angle)], dtype=np.float32)
            if rect_w >= rect_h:
                long_axis = axis_w
                short_axis = axis_h
                long_len = rect_w
                short_len = rect_h
            else:
                long_axis = axis_h
                short_axis = axis_w
                long_len = rect_h
                short_len = rect_w
            if long_len < 20.0 or short_len < 6.0:
                continue

            center_global = center_local + np.asarray([rx1, ry1], dtype=np.float32)
            end_a = center_global - long_axis * (long_len * 0.5)
            end_b = center_global + long_axis * (long_len * 0.5)
            if blue_pt is not None:
                use_a = np.linalg.norm(end_a - blue_pt) <= np.linalg.norm(end_b - blue_pt)
            else:
                use_a = float(end_a[0]) >= float(end_b[0])
            tip = end_a if use_a else end_b
            inward = long_axis if use_a else -long_axis
            target_center = tip + inward * (long_len * max(0.0, min(0.45, float(inset_ratio))))
            cross_half = short_axis * (short_len * 0.5)
            endpoints = np.stack([target_center - cross_half, target_center + cross_half], axis=0)
            score = -float(np.linalg.norm(tip - blue_pt)) if blue_pt is not None else float(tip[0])
            score += 0.0005 * area
            scored.append((score, endpoints, target_center, long_len, area))

        if not scored:
            return None
        _, endpoints, center, long_len, area = max(scored, key=lambda item: item[0])
        width_px = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        endpoints_i = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in endpoints]
        return {
            'handle_grasp_edge_px': endpoints_i,
            'handle_grasp_center_px': [int(round(float(center[0]))), int(round(float(center[1])))],
            'handle_grasp_width_px': width_px,
            'handle_grasp_source': 'opencv_black_end_inset_near_blue' if blue is not None else 'opencv_black_end_inset',
            'handle_grasp_mask_area_px': float(area),
            'handle_grasp_long_axis_length_px': float(long_len),
        }

    def _detect_press_point(self, image, detections):
        if self.press_point_mode in ('red', 'red_sticker', 'red-sticker'):
            return self._detect_red_sticker_point(image, detections)
        if self.press_point_mode in ('white', 'white_square', 'white-square'):
            return self._detect_white_square_point(image, detections)
        return self._detect_blue_point(image, detections)

    def _detect_red_sticker_point(self, image, detections):
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        candidates = []
        for rx1, ry1, rx2, ry2 in self._blue_search_rois(detections, img_w, img_h):
            crop = image[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask_low = cv2.inRange(hsv, (0, 70, 45), (14, 255, 255))
            mask_high = cv2.inRange(hsv, (165, 70, 45), (179, 255, 255))
            mask = cv2.bitwise_or(mask_low, mask_high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < max(100.0, self.blue_point_min_area):
                    continue
                if area > float(img_w * img_h) * max(self.blue_point_max_area_ratio, 1e-6):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 10 or h < 10:
                    continue
                aspect = float(w) / float(h)
                if aspect < 0.45 or aspect > 2.20:
                    continue
                rectangularity = area / float(max(1, w * h))
                if rectangularity < 0.45:
                    continue

                gx1, gy1 = int(rx1 + x), int(ry1 + y)
                gx2, gy2 = int(gx1 + w), int(gy1 + h)
                dark_ratio = self._dark_surround_ratio(
                    image,
                    gx1,
                    gy1,
                    gx2,
                    gy2,
                )
                if dark_ratio < self.lock_min_dark_surround_ratio:
                    continue
                cx = float(gx1 + gx2) * 0.5
                cy = float(gy1) + float(h) * self.press_point_vertical_ratio
                square_score = 1.0 - min(1.0, abs(1.0 - aspect))
                candidates.append(
                    (
                        area * rectangularity * max(0.2, square_score),
                        {
                            'class_name': self.blue_point_class_name,
                            'class_id': 9001,
                            'confidence': min(0.99, 0.50 + 0.35 * rectangularity),
                            'xmin': gx1,
                            'ymin': gy1,
                            'xmax': gx2,
                            'ymax': gy2,
                            'cx': cx,
                            'cy': cy,
                            'lock_dark_surround_ratio': float(dark_ratio),
                        },
                    )
                )

        if not candidates:
            return []
        return [max(candidates, key=lambda item: item[0])[1]]

    def _dark_surround_ratio(self, image, x1, y1, x2, y2):
        image_h, image_w = image.shape[:2]
        box_w = max(4, int(x2 - x1))
        box_h = max(4, int(y2 - y1))
        pad = max(8, int(round(0.85 * max(box_w, box_h))))
        rx1 = max(0, int(x1) - pad)
        ry1 = max(0, int(y1) - pad)
        rx2 = min(image_w, int(x2) + pad)
        ry2 = min(image_h, int(y2) + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            return 0.0
        hsv = cv2.cvtColor(image[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
        surround = np.ones(hsv.shape[:2], dtype=bool)
        ix1 = max(0, int(x1) - rx1)
        iy1 = max(0, int(y1) - ry1)
        ix2 = min(hsv.shape[1], int(x2) - rx1)
        iy2 = min(hsv.shape[0], int(y2) - ry1)
        surround[iy1:iy2, ix1:ix2] = False
        count = int(np.count_nonzero(surround))
        if count <= 0:
            return 0.0
        dark = (hsv[:, :, 2] <= self.lock_dark_value_max) & surround
        return float(np.count_nonzero(dark)) / float(count)

    def _detect_white_square_point(self, image, detections):
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        candidates = []
        for rx1, ry1, rx2, ry2 in self._blue_search_rois(detections, img_w, img_h):
            crop_hsv = hsv_full[ry1:ry2, rx1:rx2]
            if crop_hsv.size == 0:
                continue
            mask = cv2.inRange(crop_hsv, (0, 0, 145), (179, 95, 255))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < max(100.0, self.blue_point_min_area):
                    continue
                if area > float(img_w * img_h) * max(self.blue_point_max_area_ratio, 1e-6):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 10 or h < 10:
                    continue
                aspect = float(w) / float(h)
                if aspect < 0.60 or aspect > 1.55:
                    continue
                rectangularity = area / float(max(1, w * h))
                if rectangularity < 0.52:
                    continue

                gx1, gy1, gx2, gy2 = rx1 + x, ry1 + y, rx1 + x + w, ry1 + y + h
                pad = max(6, int(round(max(w, h) * 0.30)))
                sx1, sy1 = max(0, gx1 - pad), max(0, gy1 - pad)
                sx2, sy2 = min(img_w, gx2 + pad), min(img_h, gy2 + pad)
                value_region = hsv_full[sy1:sy2, sx1:sx2, 2]
                if value_region.size == 0:
                    continue
                ring_mask = np.ones(value_region.shape, dtype=bool)
                ring_mask[gy1 - sy1:gy2 - sy1, gx1 - sx1:gx2 - sx1] = False
                ring_values = value_region[ring_mask]
                inner_values = hsv_full[gy1:gy2, gx1:gx2, 2]
                if ring_values.size == 0 or inner_values.size == 0:
                    continue
                contrast = float(np.mean(inner_values)) - float(np.median(ring_values))
                if contrast < 22.0:
                    continue

                moments = cv2.moments(contour)
                if moments['m00'] != 0.0:
                    cx = rx1 + float(moments['m10'] / moments['m00'])
                    cy = ry1 + float(moments['m01'] / moments['m00'])
                else:
                    cx, cy = float(gx1 + gx2) * 0.5, float(gy1 + gy2) * 0.5
                square_score = 1.0 - min(1.0, abs(1.0 - aspect))
                candidates.append(
                    (
                        area * rectangularity * max(0.2, square_score) * contrast,
                        {
                            'class_name': self.blue_point_class_name,
                            'class_id': 9001,
                            'confidence': min(0.99, 0.50 + 0.25 * rectangularity + 0.002 * contrast),
                            'xmin': int(gx1),
                            'ymin': int(gy1),
                            'xmax': int(gx2),
                            'ymax': int(gy2),
                            'cx': float(cx),
                            'cy': float(cy),
                        },
                    )
                )

        if not candidates:
            return []
        return [max(candidates, key=lambda item: item[0])[1]]

    def _detect_blue_point(self, image, detections):
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        rois = self._blue_search_rois(detections, img_w, img_h)
        candidates = []
        for rx1, ry1, rx2, ry2 in rois:
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            crop = image[ry1:ry2, rx1:rx2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (78, 45, 35), (112, 255, 255))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.blue_point_min_area:
                    continue
                if area > float(img_w * img_h) * max(self.blue_point_max_area_ratio, 1e-6):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 2 or h <= 2:
                    continue
                aspect = float(w) / float(h)
                if aspect < 0.45 or aspect > 2.2:
                    continue
                perimeter = float(cv2.arcLength(contour, True))
                circularity = 0.0 if perimeter <= 0.0 else 4.0 * 3.14159 * area / (perimeter * perimeter)
                if circularity < 0.25:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] != 0.0:
                    cx = rx1 + float(moments['m10'] / moments['m00'])
                    cy = ry1 + float(moments['m01'] / moments['m00'])
                else:
                    cx = rx1 + float(x + w) * 0.5
                    cy = ry1 + float(y + h) * 0.5
                candidates.append(
                    {
                        'class_name': self.blue_point_class_name,
                        'class_id': 9001,
                        'confidence': min(0.99, 0.45 + 0.45 * min(1.0, circularity)),
                        'xmin': int(rx1 + x),
                        'ymin': int(ry1 + y),
                        'xmax': int(rx1 + x + w),
                        'ymax': int(ry1 + y + h),
                        'cx': float(cx),
                        'cy': float(cy),
                        'score': area * max(0.2, circularity),
                    }
                )

        if not candidates:
            return []
        best = max(candidates, key=lambda det: det['score'])
        best.pop('score', None)
        return [best]

    def _blue_search_rois(self, detections, img_w, img_h):
        handle_rois = []
        for det in detections:
            name = str(det.get('class_name', '')).lower()
            if not any(token in name for token in ('handle', 'lever', 'pull', 'cabinet door')):
                continue
            x1 = int(det.get('xmin', 0))
            y1 = int(det.get('ymin', 0))
            x2 = int(det.get('xmax', img_w - 1))
            y2 = int(det.get('ymax', img_h - 1))
            pad_x = max(8, int((x2 - x1) * 0.35))
            pad_y = max(8, int((y2 - y1) * 0.25))
            handle_rois.append(
                (
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(img_w, x2 + pad_x),
                    min(img_h, y2 + pad_y),
                )
            )
        return handle_rois or [(0, 0, img_w, img_h)]


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
