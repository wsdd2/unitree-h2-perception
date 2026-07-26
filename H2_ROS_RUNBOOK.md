# H2 ROS Daily Runbook

This is the daily-use runbook for the H2 cabinet-inspection perception system.

Keep only one normal path:

```text
Single process: RealSense -> YOLOE -> 3D projection -> IK target -> ROS result topics
```

H2 constants:

```text
H2 IP: 192.168.25.189
Workspace: /home/unitree/MscapeTech/Foxy_ROS
ROS: Humble
Config: /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
Robot status input: /robot/inspection_status
Main perception output: /detector/objects_ik_json
Web dashboard: http://192.168.25.189:8080/
```

## 1. Sync From Local

Run on WSL:

```bash
rsync -av --progress \
  --exclude build \
  --exclude install \
  --exclude log \
  --exclude thirdparty \
  --exclude '*.[rR][aA][rR]' \
  --exclude '*.[pP][tT]' \
  --exclude '*.[pP][tT][hH]' \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  /mnt/e/MscapeTech/Foxy_ROS/ \
  unitree@192.168.25.189:/home/unitree/MscapeTech/Foxy_ROS/
```

This command does not use `--delete`, so private files on H2 are kept.

## 2. Build On H2

Run on H2:

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash

rm -rf build/detector_msgs build/yolo_trt_ros2 install/detector_msgs install/yolo_trt_ros2

colcon build --packages-select detector_msgs yolo_trt_ros2

source install/setup.bash
```

Quick checks:

```bash
ros2 interface show detector_msgs/msg/RobotInspectionStatus
ros2 pkg executables yolo_trt_ros2
```

## 3. Daily Launch

Run on H2:

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  webUI:=true
```

Important:

- Use `webUI:=false` when the browser preview is not needed. This skips the
  HTTP server, overlay drawing and JPEG encoding.
- The integrated path never publishes raw RGB, aligned depth, CameraInfo or
  `/detector/debug_image`; frames move in memory inside one process.
- Do not add spaces after `\`.
- Do not run this inside conda.
- Do not set `PYTHONPATH` manually for daily launch; the H2 Unitree SDK path is handled inside the node.

Target compensation:

- `dex1_tip_from_wrist_xyz: [0.14, 0.01, 0.012]` is the measured Dex1-1 fingertip position relative to `right_wrist_yaw_link`.
- The ROS `point_target` is the desired Dex1-1 fingertip contact point in the target/world frame.
- ROS converts the contact point to a physical `right_wrist_yaw_link` target; the current IK control frame is the wrist itself.
- Blue push points additionally apply `blue_point_target_world_offset_xyz: [0.0, 0.0, -0.004]`, so only the blue button target is moved 4 mm toward the ground.
- `fk_backend: urdf` and `lock_waist: false` are intentional: waist joints from `rt/lf/lowstate` must be used, otherwise targets drift badly after lower-body/waist motion.
- `handeye_mount_offset_from_wrist_xyz` and `xr_ee_offset_from_wrist_xyz` are both zero for the new wrist-link calibration; no virtual 5 cm `R_ee` offset is used.

Expected nodes:

```bash
ros2 node list
```

Expected:

```text
/direct_realsense
/yolo_detector
/coordinate_projector
/web_dashboard
```

`/web_dashboard` is present only with `webUI:=true`. All four logical ROS nodes
run in one OS process.

Direct executable equivalent:

```bash
ros2 run yolo_trt_ros2 integrated_perception_node --webUI --ros-args \
  --params-file /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
```

## 4. Health Check

Use another terminal:

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Check camera:

```bash
ros2 topic echo --once /camera/color/camera_info
ros2 topic hz /camera/color/image_raw
```

Check perception:

```bash
ros2 topic echo --once /detector/objects
ros2 topic echo --once /detector/objects_3d
ros2 topic echo --once /detector/objects_ik_json
```

Check current and target joints:

```bash
ros2 topic echo --once /detector/current_joint_state
ros2 topic echo --once /detector/target_joint_state
```

Open web dashboard:

```text
http://192.168.25.189:8080/
```

## 5. Robot Engineer Subscription

Recommended topic:

```text
/detector/objects_ik_json
```

Type:

```text
std_msgs/msg/String
```

It contains:

```text
objects[].class_name
objects[].object_id
objects[].point_target
objects[].ik.success
objects[].ik.joint_values_rad
robot_status
```

Button workflow:

```text
1. Move arm to initial pose.
2. Subscribe /detector/objects_ik_json.
3. Find object where class_name == "red sticker push point".
4. Read object.point_target as the button target in pelvis frame.
5. If object.ik.success is true, use object.ik.joint_values_rad directly.
```

Minimal Python-style pseudo code:

```python
def on_msg(msg):
    data = json.loads(msg.data)

    for obj in data["objects"]:
        if obj["class_name"] == "red sticker push point":
            button_xyz = obj["point_target"]  # pelvis frame
            ik = obj.get("ik") or {}

            if ik.get("success"):
                move_arm_by_joints(ik["joint_values_rad"])
            else:
                move_arm_to_xyz(button_xyz)
```

Strongly typed alternative:

```text
/detector/objects_3d
detector_msgs/msg/Object3DArray
```

Find:

```text
object.detection.class_name == "red sticker push point"
```

Read:

```text
object.point_target.x
object.point_target.y
object.point_target.z
```

## 6. Robot Engineer Status Publisher

Robot side publishes:

```text
/robot/inspection_status
detector_msgs/msg/RobotInspectionStatus
```

Stage convention:

```text
0 idle
1 move_to_handle_front
2 press_blue_point
3 grasp_or_pull_handle
4 door_opened
5 recover_or_abort
6 pick_work_tag          # target = work tag grasp point
7 hang_work_tag          # target = cabinet hang hook
```

Hang-tag classes on `/detector/objects` / `/detector/objects_3d`:

```text
green work tag
red hang cord
work tag grasp point
cabinet hang hook
```

Enable with `detect_work_tag: true` in `inspection_perception.yaml`.

YOLOE prompts in `cabinet_controls_classes.txt` cover the tag/cord/hook;
OpenCV still owns cord-apex / white-force geometry and refines the hook tip
inside the YOLOE box when the semantic hook is present.

Example:

```bash
ros2 topic pub --once /robot/inspection_status detector_msgs/msg/RobotInspectionStatus "{
  header: {frame_id: 'pelvis'},
  stage_id: 2,
  stage_name: 'press_blue_point',
  current_action: 'moving_to_blue_button',
  motion_active: true,
  progress: 0.45,
  has_error: false,
  error_code: '',
  error_message: '',
  emergency_stop: false,
  target_reachable: true,
  reachability_message: 'ik ok',
  target_id: '01_blue_push_point'
}"
```

This status is mirrored into:

```text
/detector/objects_ik_json -> robot_status
Web dashboard /api/state -> robot_status
```

## 7. Example Scripts

Examples are in:

```text
~/MscapeTech/Foxy_ROS/examples/
```

Robot subscribes perception:

```bash
python3 examples/robot_subscribe_perception_example.py
```

Robot publishes status:

```bash
python3 examples/robot_publish_status_example.py \
  --stage 2 \
  --action moving_to_blue_button \
  --active \
  --progress 0.45 \
  --reachable true \
  --target-id 01_blue_push_point
```

## 8. If Daily Launch Fails

Do not switch to another operating mode in daily use.

Collect these logs and send them to the vision side:

```bash
ros2 node list
ros2 topic list
ros2 topic echo --once /camera/color/camera_info
ros2 topic echo --once /detector/objects_ik_json
```

If a process dies, send the full traceback, especially the 20 lines before:

```text
process has died
```
