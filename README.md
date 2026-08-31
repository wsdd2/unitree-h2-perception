# H2 ROS Perception

这是 H2 ACT的 ROS2 感知工作区。当前日常使用链路是：

```text
单进程 RealSense -> YOLOE/蓝点检测 -> 深度投影 -> 手眼/FK -> Dex1 补偿目标
```

当前 H2 部署信息：

```text
H2 IP: 192.168.25.189
H2 workspace: /home/unitree/MscapeTech/Foxy_ROS
ROS: Humble
Main config: /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
Web dashboard: http://192.168.25.189:8080/
Main robot-facing output: /detector/objects_ik_json
```

## 包结构

```text
Foxy_ROS/
  src/
    detector_msgs/
      msg/
        Object2D.msg
        Object2DArray.msg
        Object3D.msg
        Object3DArray.msg
        RobotInspectionStatus.msg
    yolo_trt_ros2/
      config/
        inspection_perception.yaml
      launch/
        inspection_perception.launch.py
      yolo_trt_ros2/
        integrated_perception_node.py
        direct_realsense_node.py
        yolo_detector_node.py
        coordinate_projector_node.py
        web_dashboard_node.py
        backends/
          ultralytics_backend.py
```

## 日常启动

在 H2 上运行：

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

不需要网页时改为 `webUI:=false`。此时不会启动 HTTP 服务，也不会绘制、转换或编码预览图像。

也可直接运行集成入口；`--webUI` 是标准 `argparse action='store_true'` 参数：

```bash
ros2 run yolo_trt_ros2 integrated_perception_node --webUI --ros-args \
  --params-file /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
```

打开网页（仅 `webUI:=true`）：

```text
http://192.168.25.189:8080/
```

注意：

- 不要在 conda 环境里启动 ROS。
- 不要手动设置 Unitree SDK 的 `PYTHONPATH`；节点内部已经处理。
- `\` 后面不要有空格。
- 如果 RealSense 报 `Device or resource busy`，通常是已有 direct/ROS/直连脚本占用了相机。

## 同步和重建

从本机 WSL 同步整个 ROS 工作区到 H2：

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

H2 上重建：

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash

rm -rf build/detector_msgs install/detector_msgs build/yolo_trt_ros2 install/yolo_trt_ros2
colcon build --packages-select detector_msgs yolo_trt_ros2

source install/setup.bash
```

如果只改了 `yolo_trt_ros2` Python 代码或 YAML，通常只需要：

```bash
rm -rf build/yolo_trt_ros2 install/yolo_trt_ros2
colcon build --packages-select yolo_trt_ros2
```

如果改了 `detector_msgs/msg/*.msg`，必须同时重建 `detector_msgs` 和 `yolo_trt_ros2`。

## 主要输出 Topic

```text
/detector/objects
  detector_msgs/msg/Object2DArray
  2D 检测结果，包括蓝点、把手检测框，以及把手端点像素字段。

/detector/objects_3d
  detector_msgs/msg/Object3DArray
  3D 目标结果。point_target 已经对齐直连脚本的 copy target 语义。

/detector/target_point
  geometry_msgs/msg/PointStamped
  当前最佳目标点。

/detector/target_pose
  geometry_msgs/msg/PoseStamped
  当前最佳目标位姿。

/detector/target_joint_state
  sensor_msgs/msg/JointState
  ROS IK 给出的目标关节角建议。

/detector/current_joint_state
  sensor_msgs/msg/JointState
  从 Unitree lowstate 读取的当前关节角。

/detector/objects_ik_json
  std_msgs/msg/String
  推荐给机器人工程师订阅的主输出。

/robot/inspection_status
  detector_msgs/msg/RobotInspectionStatus
  机器人侧发布的状态，视觉侧订阅并合并到 JSON 和网页。
```

## 坐标和补偿语义

当前 ROS 输出已经和直连脚本对齐。

关键配置在 `inspection_perception.yaml`：

```yaml
coordinate_projector:
  ros__parameters:
    handeye_mode: eye-in-hand
    handeye_target_frame: torso_link
    base_link: torso_link
    hand_link: right_wrist_yaw_link
    fk_backend: xr_pinocchio
    lock_waist: false

    handeye_mount_offset_from_wrist_xyz: [0.0, 0.0, 0.0]
    xr_ee_offset_from_wrist_xyz: [0.0, 0.0, 0.0]
    dex1_tip_from_wrist_xyz: [0.14, 0.01, 0.012]
    blue_point_target_world_offset_xyz: [0.0, 0.0, -0.004]
```

含义：

- `dex1_tip_from_wrist_xyz` 是 Dex1-1 指尖/接触点相对 `right_wrist_yaw_link` 的实测偏移。
- 蓝点会额外加 `blue_point_target_world_offset_xyz`，当前是世界 Z 方向向下 4mm。
- `/detector/objects_3d[].point_target` 和网页复制点位是“机械臂可直接执行的目标点”，等价于直连脚本里的 `preferred_copy_target / ree_target_for_dex1_tip`。
- 新标定和FK直接使用物理 `right_wrist_yaw_link`，不再使用前移5cm的虚拟 `R_ee`。

## 任务字段

推荐机器人侧订阅：

```text
/detector/objects_ik_json
```

消息类型：

```text
std_msgs/msg/String
```

JSON 中常用字段：

```text
objects[].class_name
objects[].object_id
objects[].point_target
objects[].ik.success
objects[].ik.joint_values_rad
objects[].handle_grasp_endpoint_targets_m
objects[].handle_mid_right_air_target_m
robot_status
```

### Stage 2: 戳蓝点

找：

```text
class_name == "blue push point"
```

读：

```text
object.point_target
```

如果 `object.ik.success == true`，也可以直接用：

```text
object.ik.joint_values_rad
```

### Stage 3: 推/拉弹起把手

把手检测有两套输出：

```text
handle_grasp_endpoint_targets_m
```

把手末端夹持线两端，已经做 Dex1 补偿。网页上黄/红端点 marker 双击复制对应点。

```text
handle_mid_right_air_target_m
```

把手中段右侧空气中的目标点，默认从把手中段向图像右方向偏移 1cm，再做 Dex1 补偿。机器人可先到这个点，然后执行向左推/拉动作。

对应配置：

```yaml
handle_mid_right_offset_m: 0.01
```

深度 fallback 会在只检测到蓝点、没检测到小把手框时，根据蓝点左侧的深度前景估计把手端点和中段：

```yaml
handle_depth_grasp_fallback: true
handle_depth_search_left_px: 360
handle_depth_search_right_px: 18
handle_depth_search_y_px: 70
handle_depth_near_delta_m: 0.015
handle_depth_max_blue_distance_px: 180.0
handle_depth_sticky_px: 45.0
```

### Stage 6: 夹起工作吊牌

开启配置：

```yaml
yolo_detector:
  ros__parameters:
    detect_work_tag: true
```

YOLOE prompts（`cabinet_controls_classes.txt`）已加入吊牌/挂钩语义；与 OpenCV 融合策略：

```text
green work tag / cabinet hang hook  -> 优先 YOLOE，挂钩尖点用 OpenCV 精修
red hang cord / work tag grasp point -> 优先 OpenCV 几何（倒 V + 白色着力点）
任一侧缺失时由另一侧补齐
```

检测类：

```text
green work tag          # 吊牌本体
red hang cord           # 红色吊绳
work tag grasp point    # 红线上白色着力点 / 绳环顶点（夹取目标）
cabinet hang hook       # 柜面白色挂钩（挂靠目标）
```

运控发布：

```text
stage_id: 6
stage_name: 'pick_work_tag'
```

此时 `/detector/target_point` 与 `/detector/target_pose` 切换为 `work tag grasp point`。
也可直接在 `/detector/objects_3d` / `objects_ik_json` 中按 `class_name` 过滤。

### Stage 7: 挂到挂钩

```text
stage_id: 7
stage_name: 'hang_work_tag'
```

此时公共目标切换为 `cabinet hang hook` 的 `point_target`。

## 网页使用

```text
http://192.168.25.189:8080/
```

网页元素：

- 蓝点框：蓝色按钮目标。
- 黄/红小方块：把手端点，双击复制对应 `handle_grasp_endpoint_targets_m[i]`。
- 青色小方块：把手中段右侧空气点，双击复制 `handle_mid_right_air_target_m`。
- 双击目标框：优先复制 `handle_mid_right_air_target_m`；没有该字段时复制把手中心或普通目标点。

## 快速检查

另开一个 H2 终端：

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

检查节点：

```bash
ros2 node list
```

应看到：

```text
/direct_realsense
/yolo_detector
/coordinate_projector
/web_dashboard
```

检查相机：

```bash
ros2 topic echo --once /camera/color/camera_info
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
```

检查感知：

```bash
ros2 topic echo --once /detector/objects
ros2 topic echo --once /detector/objects_3d
ros2 topic echo --once /detector/objects_ik_json --field data
```

检查关节：

```bash
ros2 topic echo --once /detector/current_joint_state
ros2 topic echo --once /detector/target_joint_state
```

## 机器人状态输入

机器人侧可以发布：

```text
/robot/inspection_status
detector_msgs/msg/RobotInspectionStatus
```

Stage 约定：

```text
0 idle
1 move_to_handle_front
2 press_blue_point
3 push_or_pull_handle
4 door_opened
5 recover_or_abort
```

示例：

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
  target_id: '00_blue_push_point'
}"
```

该状态会出现在：

```text
/detector/objects_ik_json -> robot_status
http://192.168.25.189:8080/api/state -> robot_status
```

## 示例脚本

```text
Foxy_ROS/examples/
```

机器人订阅感知示例：

```bash
python3 examples/robot_subscribe_perception_example.py
```

机器人发布状态示例：

```bash
python3 examples/robot_publish_status_example.py \
  --stage 2 \
  --action moving_to_blue_button \
  --active \
  --progress 0.45 \
  --reachable true \
  --target-id 00_blue_push_point
```

## 常见问题

### Package not found

通常是没有 source 工作区：

```bash
source /opt/ros/humble/setup.bash
source ~/MscapeTech/Foxy_ROS/install/setup.bash
```

### 话题看不到

确认当前终端和 launch 终端一致：

```bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### RealSense busy

检查是否已有进程占用相机：

```bash
ps -eo pid,ppid,stat,cmd | grep -E 'direct_realsense|run_h2_handle_pose_direct|realsense' | grep -v grep
for d in /dev/video*; do [ -e "$d" ] && fuser -v "$d" 2>&1; done
```

### 修改 msg 后报字段不存在

重建两个包：

```bash
rm -rf build/detector_msgs install/detector_msgs build/yolo_trt_ros2 install/yolo_trt_ros2
colcon build --packages-select detector_msgs yolo_trt_ros2
source install/setup.bash
```
