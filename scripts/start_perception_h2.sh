#!/usr/bin/env bash
# H2 inspection perception launcher (LF endings).
# Usage:
#   bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
#   DETECTOR_MODE=yoloe bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
#   DETECTOR_MODE=yoloseg bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
#   DETECTOR_MODE=hybrid bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
#   DETECTOR_MODE=seg_on_request bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
#   YOLO_SEG_OPENCV_COLOR=false bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
set -eo pipefail

cd ~/MscapeTech/Foxy_ROS
conda deactivate 2>/dev/null || true
deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

# ROS setup.bash references optional unset vars; keep nounset off.
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Refuse launch if another inspection-perception stack is already up
# (same Domain / camera). Override: ALLOW_MULTI_PERCEPTION=1
python3 - <<'PY'
import os
import sys
sys.path.insert(0, os.path.expanduser('~/MscapeTech/Foxy_ROS/src/yolo_trt_ros2'))
from yolo_trt_ros2.perception_singleton import (
    PerceptionAlreadyRunningError,
    assert_no_conflicting_perception,
)
try:
    assert_no_conflicting_perception(web_port=int(os.environ.get('WEB_PORT', '8081')))
except PerceptionAlreadyRunningError as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(2)
print(
    'Perception singleton check OK: ROS_DOMAIN_ID=%s LOCALHOST_ONLY=%s'
    % (os.environ.get('ROS_DOMAIN_ID', '0'), os.environ.get('ROS_LOCALHOST_ONLY', '<unset>'))
)
PY

# Default: both YOLOE + YOLOSeg every frame (hybrid).
# Optional override: yoloe | yoloseg | hybrid | seg_on_request
DETECTOR_MODE="${DETECTOR_MODE:-hybrid}"
# OpenCV HSV color prefixes for YOLOSeg push button / toggle. true|false
YOLO_SEG_OPENCV_COLOR="${YOLO_SEG_OPENCV_COLOR:-true}"

ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  confidence:=0.10 \
  cam_serial:=346522074739 \
  handeye_mode:=eye-in-hand \
  handeye_npy_path:=/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/outputs/eye_in_hand_20260726_115420.json \
  handeye_target_frame:=torso_link \
  base_link:=torso_link \
  hand_link:=right_wrist_yaw_link \
  webUI:=true \
  web_port:=8081 \
  automation:=true \
  automation_execute:=false \
  apply_tip_compensation:=false \
  dex1_tip_from_wrist_xyz:="[0.0, 0.0, 0.0]" \
  blue_point_target_world_offset_xyz:="[0.0, 0.0, 0.0]" \
  projected_world_offset_xyz:="[0.0, 0.0, 0.0]" \
  handeye_mount_offset_from_wrist_xyz:="[0.0, 0.0, 0.0]" \
  "detector_mode:=${DETECTOR_MODE}" \
  "yolo_seg_opencv_color:=${YOLO_SEG_OPENCV_COLOR}"
