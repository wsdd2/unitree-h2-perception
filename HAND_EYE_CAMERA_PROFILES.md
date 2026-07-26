# H2 Hand-Eye Camera Profiles

All current perception outputs use `torso_link`.

## Right wrist D435: eye-in-hand

```text
serial: 346522074739
mode: eye-in-hand
calibration:
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/outputs/eye_in_hand_20260726_115420.json
calibration FK reference: torso_link
saved transform: camera -> right_wrist_yaw_link
runtime/output reference: torso_link
target hand: right_wrist_yaw_link
record: Hand_Eye_Calib/H2_EYE_IN_HAND_CALIBRATION_20260726.md
```

For eye-in-hand, `torso_link` is the calibration/runtime FK reference. The
saved hand-eye matrix is the rigid camera-to-physical-wrist transform:

```text
T_torso_camera(q) = T_torso_right_wrist_yaw_link(q) * T_wrist_camera
P_torso = T_torso_camera(q) * P_camera
```

Export variables:

```bash
source ~/MscapeTech/Foxy_ROS/scripts/export_right_wrist_eye_in_hand.sh
```

```bash
ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=${H2_PERCEPTION_CONFIG} \
  cam_serial:=${H2_WRIST_CAM_SERIAL} \
  handeye_mode:=${H2_HANDEYE_MODE} \
  handeye_npy_path:=${H2_HANDEYE_JSON} \
  handeye_target_frame:=${H2_HANDEYE_TARGET_FRAME} \
  base_link:=${H2_BASE_LINK} \
  hand_link:=${H2_HAND_LINK} \
  dex1_tip_from_wrist_xyz:="${H2_DEX1_TIP_FROM_WRIST_XYZ}" \
  apply_tip_compensation:=${H2_APPLY_TIP_COMPENSATION} \
  webUI:=true
```

## Torso/abdomen D435: eye-to-hand, right arm target

```text
serial: 347622074029
mode: eye-to-hand
calibration:
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_to_hand/outputs/eye_to_hand_20260724_101155.json
reference: torso_link
target hand: right_wrist_yaw_link
```

```bash
ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  cam_serial:=347622074029 \
  handeye_mode:=eye-to-hand \
  handeye_npy_path:=/home/unitree/MscapeTech/Hand_Eye_Calib/eye_to_hand/outputs/eye_to_hand_20260724_101155.json \
  handeye_target_frame:=torso_link \
  base_link:=torso_link \
  hand_link:=right_wrist_yaw_link \
  webUI:=true
```

## Torso/abdomen D435: eye-to-hand, left arm target

The same `T_torso_camera` calibration is valid. The arm used to move the
calibration board does not make the fixed camera extrinsic right-arm-specific.
Runtime URDF/FK computes `T_torso_left_wrist` from current left-arm joints.

```bash
ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  cam_serial:=347622074029 \
  handeye_mode:=eye-to-hand \
  handeye_npy_path:=/home/unitree/MscapeTech/Hand_Eye_Calib/eye_to_hand/outputs/eye_to_hand_20260724_101155.json \
  handeye_target_frame:=torso_link \
  base_link:=torso_link \
  hand_link:=left_wrist_yaw_link \
  webUI:=true
```

Do not use a constant right-wrist-to-left-wrist offset. That transform changes
with both arm configurations. Also configure a measured left-hand
`dex1_tip_from_wrist_xyz`; do not assume the right-hand lateral sign.

For eye-in-hand, the camera carrier link and target hand are currently the same
`hand_link`; therefore the wrist-camera profile is right-arm-only. Supporting
right-wrist camera guidance for the left arm requires separate
`camera_hand_link` and `target_hand_link` parameters.
