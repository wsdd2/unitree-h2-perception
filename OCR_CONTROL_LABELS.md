# Cabinet Control OCR Labels

The optional OCR layer keeps the visual detector class unchanged and adds:

```text
label_text
label_confidence
semantic_name      # e.g. white toggle switch/备用
control_id
```

It searches right/below/left label ROIs around buttons, switches, selectors,
meters, gauges, handles and the lock point. Each crop is tested in four
orientations. A label is published only after the configured number of
identical OCR observations.

## H2 dependency

Tesseract is used instead of PaddleOCR by default so the ROS Python environment
does not need another OpenCV/NumPy stack:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
```

Verify:

```bash
tesseract --version
tesseract --list-langs | grep -E 'chi_sim|eng'
```

## Configuration

Edit:

```text
src/yolo_trt_ros2/config/inspection_perception.yaml
```

Relevant parameters:

```yaml
control_ocr_enabled: true
control_ocr_backend: tesseract
control_ocr_language: chi_sim
control_ocr_dictionary_path: /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/control_labels_zh.txt
control_ocr_interval_frames: 6
control_ocr_min_confidence: 0.65
control_ocr_stable_frames: 3
control_ocr_max_controls_per_frame: 8
```

The dictionary format is:

```text
canonical|alias1|alias2
```

Add all site-specific labels to `config/control_labels_zh.txt`.

## Request by semantic identity

Observe:

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id observe-reserve-001 \
  --stage 10 \
  --stage-name inspect_inside_panel \
  --action observe_targets \
  --semantic-name "white toggle switch/备用"
```

Select one labeled control for a supported action:

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id press-stop-001 \
  --stage 12 \
  --stage-name press_inside_button \
  --action press \
  --class-name "red push button" \
  --active-class "red push button" \
  --semantic-name "red push button/停止" \
  --active-semantic "red push button/停止" \
  --lock-target
```

OCR identity alone never bypasses class-specific motion safety. Unsupported
rotary/rocker/toggle actions remain localization-only.
