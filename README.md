# Unitree H2 Perception

面向 [Unitree H2](https://www.unitree.com/) 的 ROS2 Humble 感知工作区。示例任务为电网电柜检修任务，包括：按钮、旋钮、拨动开关、门锁着力点、柜门把手、工作牌/挂钩、表计，以及 OCR 铭牌语义。

当前日常链路：

```text
腕部 RealSense D435
  → YOLO26-seg / YOLOE / OpenCV（锁点、吊牌、ArUco）
  → 深度反投影 + 手眼外参 + 右腕 FK
  → torso_link 三维目标
  → 运控 InspectionCommand 过滤与锁定
```

仓库地址：<https://github.com/wsdd2/unitree-h2-perception>

## 现场约定

日常在 H2 PC2 本机运行，不要跨机混用 Domain / 相机 / 外参。

```text
H2 用户:            unitree
H2 IP:              192.168.25.189（历史文档里也写作 192.168.1.3，同一台机器）
工作区:             /home/unitree/MscapeTech/Foxy_ROS
ROS:                Humble
ROS_DOMAIN_ID:      42
ROS_LOCALHOST_ONLY: 1          # 运控必须和感知在同一台 H2 上
DDS:                rmw_fastrtps_cpp
相机:               右腕 D435，序列号 346522074739
手眼:               eye-in-hand，标定文件 eye_in_hand_20260726_115420.json
参考系:             torso_link / right_wrist_yaw_link
主配置:             src/yolo_trt_ros2/config/inspection_perception.yaml
Web 预览:           http://192.168.25.189:8081/
运控请求:           /robot/inspection_command
运控状态:           /robot/inspection_status
主输出:             /detector/objects   (detector_msgs/Object3DArray)
```

指尖补偿由运控自己做。感知默认 `apply_tip_compensation:=false`，发布的是接触点在 `torso_link` 下的坐标，不再把 Dex1 偏移叠进目标。

## 包结构

```text
Foxy_ROS/
  scripts/start_perception_h2.sh          # 日常启动（含单例检查）
  src/
    detector_msgs/msg/
      Object2D.msg / Object2DArray.msg
      Object3D.msg / Object3DArray.msg
      InspectionCommand.msg
      RobotInspectionStatus.msg
    yolo_trt_ros2/
      config/inspection_perception.yaml
      config/cabinet_controls_classes.txt
      config/control_labels_zh.txt
      launch/inspection_perception.launch.py
      yolo_trt_ros2/
        integrated_perception_node.py     # 单进程入口：相机 + 检测 + 投影 + 可选 WebUI
        direct_realsense_node.py
        yolo_detector_node.py             # 2D：YOLOE / YOLOSeg / OpenCV / OCR / ArUco
        yolo_seg_priority.py              # 闭集分割优先与颜色前缀
        coordinate_projector_node.py      # 深度 + 手眼 + FK → 3D
        web_dashboard_node.py
        cabinet_automation_node.py        # 可选自动化桥，默认不真机动臂
        perception_singleton.py           # 禁止重复拉起感知栈
        aruco_detector.py
        control_label_ocr.py
        backends/ultralytics_backend.py
  examples/                               # 运控订阅 / 发命令 / C++ 对接说明
```

## 检测栈

默认 `detector_mode:=yoloseg`：闭集 YOLO26-seg 每帧推理，YOLOE 关闭。可用模式：

| 模式 | YOLOE | YOLOSeg | 用途 |
|------|-------|---------|------|
| `yoloseg` | 关 | 每帧 | 日常电柜作业（默认） |
| `yoloe` | 开 | 关 | 开放词探索、补类别 |
| `hybrid` | 开 | 每帧 | 对照调试 |
| `seg_on_request` | 开 | 仅当 `InspectionCommand.requested_class_names` 命中 SEG 类 | 省 GPU |

启动时可覆盖：

```bash
DETECTOR_MODE=hybrid bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
YOLO_SEG_OPENCV_COLOR=true bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
```

`yolo_seg_opencv_color:=true` 时，分割得到的按钮/拨动开关会加上 OpenCV HSV 颜色前缀（`red|green|yellow`）。默认关闭，避免联调时类名不稳定。

YOLOSeg 训练类与对外发布名：

```text
push button                  → push button（可选 red/green/yellow 前缀）
rotary selector switch       → black rotary selector switch
rotary multiple selector switch → black rotary multiple selector switch
toggle switch                → toggle switch（可选颜色前缀）
cabinet door handle          → black cabinet door handle
lock point                   → lock point
work tag                     → green work tag
hang cord                    → red hang cord
hook                         → cabinet hang hook
```

OpenCV 仍负责：

- `lock point`：柜锁上的红色着力贴纸（不是早期文档里的蓝点）
- 工作牌几何：绿牌、红绳、白色夹取点、柜面挂钩尖点
- ArUco：`aruco_tag_<ID>`

OCR 不改 `class_name`，只给外观相同的控件补实例语义，例如 `white toggle switch/备用`。权威类名表见 `src/yolo_trt_ros2/config/cabinet_controls_classes.txt` 和 [CLASS_NAMES_AND_OCR.md](CLASS_NAMES_AND_OCR.md)。

## 坐标语义

关键配置在 `inspection_perception.yaml` 的 `yoloseg_backend`（即坐标投影节点）：

```yaml
handeye_mode: eye-in-hand
handeye_target_frame: torso_link
base_link: torso_link
hand_link: right_wrist_yaw_link
fk_backend: xr_pinocchio
lock_waist: false
apply_tip_compensation: false
dex1_tip_from_wrist_xyz: [0.0, 0.0, 0.0]
```

含义：

- 标定和 FK 都使用物理 `right_wrist_yaw_link`，不再使用前移 5 cm 的虚拟 `R_ee`。
- `point_target` 是 `torso_link` 下可给运控执行的接触点；Dex1 指尖偏移留给运控。
- 深度必须与彩色图对齐；投影拒绝超过约 1.2 m 的背景点。
- 检测耗时可能超过 0.5 s，投影按彩色时间戳匹配历史深度，不用“最新一帧深度”。

## Topic 合同

### 运控 → 视觉

```text
/robot/inspection_command    detector_msgs/InspectionCommand
/robot/inspection_status     detector_msgs/RobotInspectionStatus
```

`InspectionCommand` 告诉视觉本阶段要找什么：

```text
command_id                 同一请求重复发布时必须保持不变
stage_id / stage_name      必须与 InspectionStatus 一致
requested_action           observe_targets / move / press / grasp_rotate / door / abort
requested_class_names      本阶段要回报的静态类
active_target_class_name   允许驱动动作的那一个；仅观察时留空
requested_semantic_names   OCR 语义，例如 white toggle switch/备用
selection_policy           highest_confidence / nearest / leftmost / rightmost / specified_target_id
lock_target                true 时锁定已选实例直到完成或 abort
min_confidence / required_stable_frames / max_position_std_m / timeout_sec
```

建议两个消息都按 5 Hz 持续发布。完整阶段与类名合同见 [ROBOT_CONTROL_STAGE_CLASS_CONTRACT_ZH.md](ROBOT_CONTROL_STAGE_CLASS_CONTRACT_ZH.md)。

### 视觉 → 运控

```text
/detector/objects            detector_msgs/Object3DArray     # 主输出
/detector/objects_3d         同上的兼容别名
/detector/target_point       geometry_msgs/PointStamped      # 当前锁定/最佳目标
/detector/target_pose        geometry_msgs/PoseStamped       # 默认关闭
/detector/objects_ik_json    std_msgs/String                 # 默认关闭
/detector/objects_2d         detector_msgs/Object2DArray     # 中间 2D，调试用
/detector/debug_image        sensor_msgs/Image               # 仅 webUI:=true
```

`Object3D` 里运控最常用的字段：

```text
detection.class_name
detection.confidence
detection.semantic_name / control_id / label_text
valid
target_frame                 通常 torso_link
point_target                 三维目标 (x, y, z)
```

`/detector/objects_ik_json` 仍可用于调试，但日常联调请订阅 `/detector/objects`。

## 日常启动

在 H2 上：

```bash
bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh
```

脚本会：退出 conda、清 `PYTHONPATH`/`LD_LIBRARY_PATH`、source Humble、设置 Domain 42 与 `ROS_LOCALHOST_ONLY=1`、检查没有第二套感知进程，然后 launch：

```text
webUI:=true
web_port:=8081
automation:=true
automation_execute:=false      # dry-run，不会写 rt/lowcmd
detector_mode:=yoloseg
cam_serial:=346522074739
```

不需要网页时：

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  webUI:=false
```

注意：

- 不要在 conda 环境里启动 ROS。
- 不要手动设置 Unitree SDK 的 `PYTHONPATH`；节点内部已经处理。
- 同一 Domain / 同一相机只允许一套感知。脚本默认拒绝重复启动，可用 `ALLOW_MULTI_PERCEPTION=1` 覆盖。
- RealSense 报 `Device or resource busy` 时，先查是否已有 direct/ROS/直连脚本占用相机。

网页（仅 `webUI:=true`）：

```text
http://192.168.25.189:8081/
```

## 同步和重建

从本机 WSL 同步到 H2：

```bash
rsync -av --progress \
  --exclude build \
  --exclude install \
  --exclude log \
  --exclude thirdparty \
  --exclude .git \
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

只改 `yolo_trt_ros2` 的 Python / YAML 时，通常只需要重建该包。改了 `detector_msgs/msg/*.msg` 必须两个包一起重建。

## 运控最小对接

感知启动后，运控做三件事：发命令、收回传、按类名过滤。

```bash
source /opt/ros/humble/setup.bash
source ~/MscapeTech/Foxy_ROS/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

python3 ~/MscapeTech/Foxy_ROS/examples/ops_request_class_3d.py
```

Python 模板会以 5 Hz 发布同一 `command_id` 的 `InspectionCommand`，并在 `/detector/objects` 回调里自动取出请求类的 `point_target`。C++ 侧可把同一轮封装成 `/h2_arm/vision_command` Trigger 服务，说明见 [examples/运控_C++_vision_command实现说明.md](examples/运控_C++_vision_command实现说明.md)。

当前已实现的物理技能：

```text
observe_targets     只观察并返回 3D
move                移到预接近点
press               仅 lock point 和四色按钮
grasp_rotate / door 仅 black cabinet door handle 开门序列
abort               取消当前自动化
```

旋钮、拨动开关、工作牌抓取/挂钩、表计读数目前只能 `observe_targets`，不要当成已经能自动操作。

电柜自动化节点默认 `automation_execute:=false`。真机动臂前必须先 dry-run 确认坐标，流程见 [电柜自动化_运控通信_测试流程.md](电柜自动化_运控通信_测试流程.md)。

## 快速检查

另开一个 H2 终端，环境变量必须与启动终端一致：

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 node list
ros2 topic echo --once /camera/color/camera_info
ros2 topic hz /camera/color/image_raw
ros2 topic echo --once /detector/objects
ros2 interface show detector_msgs/msg/InspectionCommand
```

`ros2 node list` 在集成进程下通常能看到：

```text
/direct_realsense
/yolo_detector
/coordinate_projector
/web_dashboard          # 仅 webUI:=true
/cabinet_automation     # 仅 automation:=true
```

## 示例脚本

```text
examples/ops_request_class_3d.py                      发命令并自动收回传 3D
examples/grab_class_3d_with_command.py                带稳定性判定的抓取类请求
examples/h2_arm_vision_command_bridge.py              运控桥示例
examples/vision_command_observe_lock_handle.yaml      lock + handle 观察请求
examples/robot_publish_inspection_command_example.py
examples/robot_publish_status_example.py
examples/robot_subscribe_perception_example.py
```

## 相关文档

| 文档 | 内容 |
|------|------|
| [ROBOT_CONTROL_STAGE_CLASS_CONTRACT_ZH.md](ROBOT_CONTROL_STAGE_CLASS_CONTRACT_ZH.md) | 运控阶段、类名、动作合同 |
| [CLASS_NAMES_AND_OCR.md](CLASS_NAMES_AND_OCR.md) | 静态类名与 OCR 语义 |
| [电柜自动化_运控通信_测试流程.md](电柜自动化_运控通信_测试流程.md) | H2 上通信 dry-run 与真机门槛 |
| [CABINET_AUTOMATION.md](CABINET_AUTOMATION.md) | 自动化桥与 worker 分工 |
| [HAND_EYE_CAMERA_PROFILES.md](HAND_EYE_CAMERA_PROFILES.md) | 相机序列号与外参 |
| [运控联调_视觉启动与排障.md](运控联调_视觉启动与排障.md) | 启动与排障 |

## 常见问题

### Package not found

```bash
source /opt/ros/humble/setup.bash
source ~/MscapeTech/Foxy_ROS/install/setup.bash
```

### 话题看不到

确认当前终端与 launch 终端一致：

```bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

运控如果跑在另一台机器上，会因为 `ROS_LOCALHOST_ONLY=1` 看不到话题。把运控放到同一台 H2，或双方明确改这个变量。

### RealSense busy / 重复感知

```bash
ps -eo pid,ppid,stat,cmd | grep -E 'integrated_perception|direct_realsense|realsense' | grep -v grep
for d in /dev/video*; do [ -e "$d" ] && fuser -v "$d" 2>&1; done
```

### 修改 msg 后报字段不存在

```bash
rm -rf build/detector_msgs install/detector_msgs build/yolo_trt_ros2 install/yolo_trt_ros2
colcon build --packages-select detector_msgs yolo_trt_ros2
source install/setup.bash
```

### 目标乱跳

细长目标（挂钩、吊绳）不要只信最高置信度。用 `InspectionCommand.lock_target`、`required_stable_frames` 和 `max_position_std_m` 锁定实例；门限能阻止乱发，不能保证第一锁一定正确。
