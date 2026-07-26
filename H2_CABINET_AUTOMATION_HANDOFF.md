# H2 神秘小说明

本文档用于 H2 右臂电柜操作联调，覆盖：

- `lock point` 着力点检测与自动戳按
- `black cabinet door handle` 左右夹持点、中心点输出
- Dex1 夹爪开合
- 把手抓取与五阶段开门动作

机器人发布的完整 Stage/Class 合同见：

```text
ROBOT_CONTROL_STAGE_CLASS_CONTRACT.md
```

## 1. 系统数据流

```text
/robot/inspection_command
/robot/inspection_status
        │
        ▼
cabinet_automation_node
        │  localhost JSON socket :8765
        ▼
h2_cabinet_motion_worker.py
        ├── rt/lowcmd
        └── rt/dex1/right/cmd
```

`h2_cabinet_motion_worker.py` 是唯一允许写入 `rt/lowcmd` 的进程。启动自动化前必须关闭原键盘上肢运控脚本和其他 LowCmd 发布者。

## 2. 前置条件

1. H2 网络接口为 `eth0`。
2. H2 机械结构已稳定支撑，急停可用，操作空间已清空。
3. Dex1-1 gripper server 已按现有运控流程启动。
4. 以下文件存在：

```text
/home/unitree/H2_joint_cartesian/scripts/h2_cabinet_motion_worker.py
/home/unitree/MscapeTech/Foxy_ROS/install/setup.bash
```

确认：

```bash
ls -lh /home/unitree/H2_joint_cartesian/scripts/h2_cabinet_motion_worker.py
ls -lh /home/unitree/MscapeTech/Foxy_ROS/install/setup.bash
```

## 3. 首次部署或代码更新

在本地 WSL 执行：

```bash
rsync -av --progress \
  /mnt/e/MscapeTech/H2_joint_cartesian_remote/scripts/h2_cabinet_motion_worker.py \
  /mnt/e/MscapeTech/H2_joint_cartesian_remote/scripts/h2_xr_official_ik_demo.py \
  unitree@192.168.25.189:/home/unitree/H2_joint_cartesian/scripts/
```

```bash
rsync -av --progress \
  --exclude build \
  --exclude install \
  --exclude log \
  --exclude thirdparty \
  --exclude __pycache__ \
  /mnt/e/MscapeTech/Foxy_ROS/ \
  unitree@192.168.25.189:/home/unitree/MscapeTech/Foxy_ROS/
```

在 H2 执行：

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash

rm -rf build/detector_msgs build/yolo_trt_ros2
rm -rf install/detector_msgs install/yolo_trt_ros2

colcon build --packages-select detector_msgs yolo_trt_ros2
source install/setup.bash

ros2 interface show detector_msgs/msg/InspectionCommand
```

## 4. 可选：释放高层 MotionSwitcher 模式

仅在 Worker 报告仍有高层运动模式占用时执行：

```bash
cd ~/H2_joint_cartesian
export H2_IFACE=eth0
source scripts/setup_env.sh
python3 scripts/h2_release_motion_mode.py eth0
```

按提示输入：

```text
RELEASE
```

释放模式前必须确认机器人已稳定支撑。

## 5. 终端 1：启动唯一 LowCmd Worker

必须在 H2 终端执行，不要在本地 WSL 直接执行。

```bash
ssh unitree@192.168.25.189
```

```bash
cd ~/H2_joint_cartesian

export H2_IFACE=eth0
source scripts/setup_env.sh

python3 scripts/h2_cabinet_motion_worker.py eth0 \
  --arm right \
  --reference-frame torso_link \
  --max-offset 0.30 \
  --max-rotation-deg 120 \
  --max-ik-error 0.002 \
  --max-ik-rotation-error-deg 2 \
  --max-ik-iterations 120 \
  --gripper-side auto \
  --gripper-command-seconds 0.8 \
  --move-duration 2.0
```

预期输出包含：

```text
worker_gripper_ready
cabinet_motion_worker_ready host=127.0.0.1 port=8765
```

如果只做视觉 Dry-run，可以暂时不启动 Worker。

## 6. 终端 2：启动视觉与自动化桥

### 6.1 首次必须使用 Dry-run

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
  webUI:=true \
  automation:=true \
  automation_execute:=false
```

### 6.2 确认 Dry-run 坐标正确后允许物理执行

停止上一个 launch，再执行：

```bash
ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  webUI:=true \
  automation:=true \
  automation_execute:=true
```

## 7. 其余 ROS 终端公共环境

终端 3、4、5 均先执行：

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

## 8. 终端 3：持续发布机器人当前状态

状态必须以 5 Hz 持续发布。切换任务前先按 `Ctrl+C` 停止旧状态发布器。

### 8.1 同时观察 Lock 和 Handle

```bash
python3 examples/robot_publish_status_example.py \
  --stage 1 \
  --stage-name locate_lock_and_handle \
  --action observe_targets \
  --reachable true \
  --rate 5
```

### 8.2 Lock 戳按状态

```bash
python3 examples/robot_publish_status_example.py \
  --stage 2 \
  --stage-name press_lock_point \
  --action press_lock_point \
  --reachable true \
  --target-id lock_point_1 \
  --rate 5
```

### 8.3 Handle 抓取开门状态

```bash
python3 examples/robot_publish_status_example.py \
  --stage 3 \
  --stage-name grasp_or_pull_handle \
  --action open_cabinet_door \
  --reachable true \
  --target-id cabinet_handle_1 \
  --rate 5
```

## 9. 终端 4：发布 InspectionCommand

命令必须持续发布。切换任务时先停止旧命令，并为新命令使用新的 `command-id`。

### 9.1 同时观察两个 Class，不执行动作

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

### 9.2 戳按 Lock point

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id press-lock-001 \
  --stage 2 \
  --stage-name press_lock_point \
  --action press \
  --class-name "lock point" \
  --active-class "lock point" \
  --target-id lock_point_1 \
  --lock-target
```

### 9.3 抓取并旋转/拉动 Handle

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id open-handle-001 \
  --stage 3 \
  --stage-name grasp_or_pull_handle \
  --action grasp_rotate \
  --class-name "black cabinet door handle" \
  --active-class "black cabinet door handle" \
  --target-id cabinet_handle_1 \
  --lock-target
```

## 10. 终端 5：监控结果

```bash
ros2 topic echo /robot/cabinet_action_result_json --field data
```

同时检查视觉原始几何：

```bash
ros2 topic echo /detector/objects_ik_json --field data
```

观察命令的 `inspection_observation` 结果应包含：

```text
lock point
  force_point_target_m

black cabinet door handle
  handle_image_left_target_m
  handle_image_right_target_m
  handle_center_target_m
  handle_width_m
```

Handle 左右以相机画面左右定义，不代表 `pelvis` 世界坐标系左右。实际抓取动作使用 `handle_center_target_m`，左右点用于确认夹持位置和宽度。

## 11. 开门后的柜内目标

视觉已支持柜内按钮、表计、旋钮和开关定位。示例：

```bash
python3 examples/robot_publish_status_example.py \
  --stage 10 --stage-name inspect_inside_panel \
  --action observe_inside_controls --reachable true --rate 5
```

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

当前能力边界：

- 红/绿/黄/黑按钮可以使用 `press` 技能。
- 表计目前只输出 `point_target_m`，尚未实现指针读数或数字 OCR。
- 旋钮、rocker、toggle 目前只定位，不执行物理动作。
- `grasp_rotate` 当前只允许电柜门把手，禁止将开门序列用于旋钮。

## 12. 自动安全门限

- 目标默认连续稳定 5 帧。
- 最大位置标准差默认 3 mm。
- 目标最大年龄默认 0.4 s。
- RobotInspectionStatus 最大间隔 3 s。
- Worker 心跳超时默认 2 s。
- 连续 IK 失败、ROS状态错误、急停、连接断开均中止动作。
- 相同 `command_id` 不会重复执行。
- 一个状态阶段只允许派发一次。
- FK/IK每周期使用实际 waist yaw/roll/pitch；腰部由下肢控制器驱动，
  自动化只求解并下发14个手臂关节。
- 默认禁止 `--max-offset 999` 和 `--max-rotation-deg 999`。

当前属于几何位置控制，不是力控。戳按与抓取阶段必须保留人工急停和现场安全监护。

## 13. 正常停止顺序

1. 停止终端 4 的 InspectionCommand 发布器。
2. 发布 `abort` 命令或停止当前任务状态。
3. 停止终端 3 的 RobotInspectionStatus 发布器。
4. `Ctrl+C` 停止视觉 launch。
5. 最后 `Ctrl+C` 停止 LowCmd Worker。

Abort 命令：

```bash
python3 examples/robot_publish_inspection_command_example.py \
  --command-id abort-001 \
  --stage 5 \
  --stage-name recover_or_abort \
  --action abort
```

## 14. 常见错误

### Worker 文件不存在

```text
python3: can't open file ... h2_cabinet_motion_worker.py
```

说明 Worker 未上传，回到第 3 节同步。

### 在本地 WSL 运行了 H2 命令

```text
cd: /home/wsdd2/H2_joint_cartesian: No such file or directory
```

先执行：

```bash
ssh unitree@192.168.25.189
```

### Worker 拒绝启动

```text
Another LowCmd worker owns /tmp/h2_cabinet_motion_worker.lock
```

说明已有 Worker 或旧键盘控制进程，先停止现有运控进程，不要删除锁文件后强行并发运行。

### 没有自动化结果

依次检查：

```bash
ros2 topic hz /robot/inspection_command
ros2 topic hz /robot/inspection_status
ros2 topic hz /detector/objects_3d
ros2 topic hz /detector/objects_ik_json
```

Command 与 Status 的 `stage_id` 必须一致，状态必须满足：

```text
has_error=false
emergency_stop=false
target_reachable=true
```
