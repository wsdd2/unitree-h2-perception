# H2 right-wrist D435 eye-in-hand profile (2026-07-26)
# Source this file before launching perception:
#   source ~/MscapeTech/Foxy_ROS/scripts/export_right_wrist_eye_in_hand.sh

export H2_WRIST_CAM_SERIAL=346522074739
export H2_HANDEYE_MODE=eye-in-hand
export H2_HANDEYE_JSON=/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/outputs/eye_in_hand_20260726_115420.json
export H2_HANDEYE_NPY_DIR=/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/outputs/eye_in_hand_20260726_115420_npy
export H2_HANDEYE_TARGET_FRAME=torso_link
export H2_BASE_LINK=torso_link
export H2_HAND_LINK=right_wrist_yaw_link
export H2_CAM_NAME=right_wrist_d435
export H2_DEX1_TIP_FROM_WRIST_XYZ='[0.14, 0.01, 0.012]'
export H2_APPLY_TIP_COMPENSATION=false
export H2_PERCEPTION_CONFIG=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
