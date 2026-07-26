# Static Class Names and Dynamic OCR Semantics

## Single class registry

The authoritative file is:

```text
src/yolo_trt_ros2/config/cabinet_controls_classes.txt
```

It contains two kinds of entries.

### YOLOE prompt classes

Plain lines are sent to YOLOE:

```text
black cabinet door handle
red push button
...
white adhesive cabinet wall hook
```

### Generated/normalized publish classes

Lines prefixed by `publish:` are valid final `Object2D.class_name` values but
are not sent to YOLOE:

```text
publish: lock point
publish: green work tag
publish: red hang cord
publish: work tag grasp point
publish: cabinet hang hook
```

These names are produced by OpenCV or by normalizing a longer YOLOE prompt.
Keeping them out of the prompt list prevents duplicate detections.

At startup the detector logs both the YOLOE prompt count and the full canonical
registry count.

## OCR does not create detector classes

OCR identity is a runtime property of one detected object:

```text
class_name       = white toggle switch
label_text       = 备用
semantic_name    = white toggle switch/备用
control_id       = white_toggle_switch/备用
label_confidence = 0.91
```

For repeated rotary controls, spatial relation is added before OCR text:

```text
spatial_relation = top_row | middle_row
label_tag_present = true | false
semantic_name = black rotary selector switch/top_row/with_tag
semantic_name = black rotary selector switch/middle_row/手动
```

`top_row` is first estimated from 2D alignment with the red/green push-button
row and then confirmed from the `torso_link` Z difference. If a physical label
plate is visible but OCR text is not stable, `with_tag` is retained.

`class_name` remains a static value from the registry. `semantic_name` is
dynamic because it depends on the physical text visible next to that instance.
Therefore semantic names are not enumerated as YOLOE classes.

If OCR is missing, uncertain or not yet stable:

```text
label_text       = ""
semantic_name    = ""
control_id       = ""
```

The visual class and 3D result remain available.

## Existing ROS topics

No OCR-specific node or topic is added. The fields travel through:

```text
/detector/objects
/detector/objects_3d
/detector/objects_ik_json
```

For `Object3DArray`, access:

```text
objects[].detection.class_name
objects[].detection.label_text
objects[].detection.semantic_name
objects[].detection.control_id
objects[].point_target
```

## InspectionCommand matching

Match every instance of one visual class:

```text
requested_class_names:
  - white toggle switch
```

Match one OCR-distinguished instance:

```text
requested_semantic_names:
  - white toggle switch/备用
active_target_semantic_name: white toggle switch/备用
```

Semantic matching never bypasses action safety. For example, a recognized
toggle label is still localization-only until a toggle-specific motion skill
is implemented.
