# H2 Cabinet Automation

The automation path is split into two processes so ROS/Python libraries never
share the LowCmd/Pinocchio process:

```text
/robot/inspection_command + /robot/inspection_status + /detector/objects_3d
  -> cabinet_automation_node
  -> localhost JSON socket
  -> h2_cabinet_motion_worker.py
  -> rt/lowcmd + rt/dex1/right/cmd
```

Only the worker may write `rt/lowcmd`. It holds
`/tmp/h2_cabinet_motion_worker.lock` to reject a second worker.

Both perception FK and worker IK use measured waist yaw/roll/pitch. The waist
remains owned by the lower-body controller; cabinet automation solves and
commands only the fourteen arm joints.

## Command and status roles

- `/robot/inspection_command` declares requested classes, the one active class,
  selection policy, stability requirement and action.
- `/robot/inspection_status` reports current robot state, reachability, motion,
  error and emergency-stop state. A fresh reachable status is required before
  dispatch.
- `move` with an active handle class moves to a point
  `handle_preapproach_m` before the handle.
- `press` presses only `active_target_class_name`.
- `grasp_rotate` / `door` executes the verified five-stage door sequence.
- `observe_targets` reports every requested class without moving.
- `abort`, status stage 5, `has_error`, or `emergency_stop` aborts motion.

The selector requires five consecutive observations by default, a maximum
per-axis standard deviation of 3 mm, and a target age below 0.4 seconds.
One status identity is dispatched at most once; change stage/action/target_id
to arm another command.

`lock point` and `black cabinet door handle` may be detected in the same frame,
but they are deliberately executed one at a time:

- Stage 2 + `current_action: lock_point` selects only `lock point`. The dry-run
  result exposes `source.force_point_target_m`.
- Stage 3 selects only the handle. The dry-run result exposes
  `source.handle_image_left_target_m`,
  `source.handle_image_right_target_m`, and
  `source.handle_center_target_m`.

The handle labels are image-left/image-right, not robot-world left/right.

## Inside-cabinet controls

After the door is open, `observe_targets` can request any configured YOLOE
class, including meters, push buttons, the rotary selector, rocker switches and
toggle switches. Observation returns a generic `point_target_m`; pressable
classes additionally expose `force_point_target_m`.

Only request classes that are expected to be visible in the current frame. The
observation command completes after every requested class is stable.

Example status:

```bash
python3 examples/robot_publish_status_example.py \
  --stage 10 --stage-name inspect_inside_panel \
  --action observe_inside_controls --reachable true --rate 5
```

Example command:

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id inspect-inside-001 \
  --stage 10 --stage-name inspect_inside_panel \
  --action observe_targets \
  --class-name "square analog ammeter" \
  --class-name "digital panel meter" \
  --class-name "red push button" \
  --class-name "black rotary selector switch"
```

Current physical skills:

- `press`: lock point and red/green/yellow/black push buttons.
- `grasp_rotate` / `door`: cabinet door handle only.
- Meters currently provide localization only; analog needle reading and digital
  OCR are not implemented.
- Rotary/rocker/toggle classes currently provide localization only. They need
  class-specific axis, direction, angle/state verification and must not reuse
  the door-opening sequence.

## 1. Start the exclusive motion worker

On H2:

```bash
cd ~/H2_joint_cartesian
export H2_IFACE=eth0
source scripts/setup_env.sh
python3 scripts/h2_cabinet_motion_worker.py eth0 \
  --arm right \
  --max-offset 0.30 \
  --max-rotation-deg 120 \
  --max-ik-error 0.002 \
  --max-ik-rotation-error-deg 2 \
  --max-ik-iterations 120 \
  --gripper-side auto \
  --move-duration 2.0
```

Do not run the keyboard LowCmd script or another arm controller concurrently.

## 2. Dry-run target selection

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  webUI:=true \
  automation:=true \
  automation_execute:=false
```

Inspect planned commands:

```bash
ros2 topic echo /robot/cabinet_action_result_json
```

Confirm both raw semantic outputs:

```bash
ros2 topic echo /detector/objects_ik_json
```

## 3. Enable physical execution

Only after dry-run XYZ, class and status mapping have been checked:

```bash
ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  webUI:=true \
  automation:=true \
  automation_execute:=true
```

## Example status inputs

Keep a status heartbeat running in one terminal, then publish a matching
inspection command from another terminal.

Observe lock and handle together without motion:

```bash
ros2 topic pub -r 5 /robot/inspection_status detector_msgs/msg/RobotInspectionStatus \
  "{header: {frame_id: pelvis}, stage_id: 1, stage_name: locate_lock_and_handle, current_action: observe_targets, motion_active: false, progress: 0.0, has_error: false, emergency_stop: false, target_reachable: true}"
```

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id observe-lock-handle-001 \
  --stage 1 \
  --stage-name locate_lock_and_handle \
  --action observe_targets \
  --class-name "lock point" \
  --class-name "black cabinet door handle" \
  --stable-frames 5 \
  --max-position-std-m 0.003
```

The result event is `inspection_observation` and contains both classes.

Press the green cabinet button:

```bash
ros2 topic pub -r 5 /robot/inspection_status detector_msgs/msg/RobotInspectionStatus \
  "{header: {frame_id: pelvis}, stage_id: 2, stage_name: press_cabinet_button, current_action: green_push_button, motion_active: false, progress: 0.0, has_error: false, emergency_stop: false, target_reachable: true, target_id: green_button_1}"
```

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id press-green-001 --stage 2 --stage-name press_cabinet_button \
  --action press --class-name "green push button" \
  --active-class "green push button" --target-id green_button_1 --lock-target
```

Test the lock force point:

```bash
ros2 topic pub -r 5 /robot/inspection_status detector_msgs/msg/RobotInspectionStatus \
  "{header: {frame_id: pelvis}, stage_id: 2, stage_name: press_lock_point, current_action: lock_point, motion_active: false, progress: 0.0, has_error: false, emergency_stop: false, target_reachable: true, target_id: lock_point_1}"
```

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id press-lock-001 --stage 2 --stage-name press_lock_point \
  --action press --class-name "lock point" \
  --active-class "lock point" --target-id lock_point_1 --lock-target
```

Grasp and open the detected cabinet handle:

```bash
ros2 topic pub -r 5 /robot/inspection_status detector_msgs/msg/RobotInspectionStatus \
  "{header: {frame_id: pelvis}, stage_id: 3, stage_name: grasp_or_pull_handle, current_action: open_cabinet_door, motion_active: false, progress: 0.0, has_error: false, emergency_stop: false, target_reachable: true, target_id: cabinet_handle_1}"
```

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id open-handle-001 --stage 3 --stage-name grasp_or_pull_handle \
  --action grasp_rotate --class-name "black cabinet door handle" \
  --active-class "black cabinet door handle" --target-id cabinet_handle_1 --lock-target
```

Abort:

```bash
ros2 topic pub -r 5 /robot/inspection_status detector_msgs/msg/RobotInspectionStatus \
  "{header: {frame_id: pelvis}, stage_id: 5, stage_name: recover_or_abort, current_action: abort, motion_active: false, has_error: true, error_code: operator_abort, error_message: operator_abort, emergency_stop: false, target_reachable: false}"
```

This is geometric position control, not force control. Keep the physical
emergency stop ready and use conservative approach distances. The worker aborts
on heartbeat loss, repeated IK failure, stale ROS target, error status, or
client disconnection.
