# Unitree H2 部署与基准测试指令集

这份指令面向 Unitree H2 PC2 宿主机直接部署 `Foxy_ROS` 电柜巡检感知链路，不使用 Docker：

```text
RGB/Depth/CameraInfo
  -> yolo_detector_node
  -> /detector/objects
  -> coordinate_projector_node
  -> /detector/objects_3d, /detector/target_point, /detector/target_pose, /detector/target_joint_state
  -> web_dashboard_node
  -> http://<H2-IP>:8080/
```

H2 侧已有约定：

- DDS 网卡通常是 `eth0`，必要时改成实际网卡名。
- H2 机器人基座/身体坐标当前配置为 `torso_link`。
- 当前手眼与运控统一使用物理 `right_wrist_yaw_link`，不再使用前移 5 cm 的虚拟 `R_ee`。
- 如果是 `eye-to-hand` 相机，优先使用 `T_cam2base.npy` 直接输出世界/基座坐标。
- 当前正式配置使用 `eye-in-hand`：`/home/unitree/MscapeTech/Hand_Eye_Calib/outputs/eye_in_hand_20260630_150210.json`，节点会自动解析同名 `_npy/T_cam2hand.npy`，并通过 H2 当前关节 FK 输出 `torso_link` 坐标。

## 0. 变量约定

在 Windows / WSL / H2 上把下面变量替换成实际值：

```text
H2_HOST=unitree@<H2-PC2-IP>
H2_WS=/home/unitree/MscapeTech
FOXY_WS=/home/unitree/MscapeTech/Foxy_ROS
MODEL_DIR=/home/unitree/MscapeTech/models
ROS_DISTRO=humble
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
WEB_URL=http://<H2-PC2-IP>:8080/
```

如果你的 H2 ROS/DDS 环境使用 CycloneDDS：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## 1. 本机传输到 H2

Windows PowerShell：

```powershell
scp -r E:\MscapeTech\Foxy_ROS unitree@<H2-PC2-IP>:~/tmp/
scp E:\MscapeTech\handle_recognition\minimal_test\yoloe-11s-seg.pt unitree@<H2-PC2-IP>:~/tmp/
```

WSL：

```bash
scp -r /mnt/e/MscapeTech/Foxy_ROS unitree@<H2-PC2-IP>:~/tmp/
scp /mnt/e/MscapeTech/handle_recognition/minimal_test/yoloe-11s-seg.pt unitree@<H2-PC2-IP>:~/tmp/
```

H2 宿主机上整理目录：

```bash
ssh unitree@<H2-PC2-IP>
mkdir -p ~/MscapeTech
rm -rf ~/MscapeTech/Foxy_ROS
mv ~/tmp/Foxy_ROS ~/MscapeTech/Foxy_ROS
mkdir -p ~/MscapeTech/models
mv ~/tmp/yoloe-11s-seg.pt ~/MscapeTech/models/
```

后续所有命令都在 H2 宿主机上执行。

## 2. H2 宿主机环境检查

H2 PC2 当前使用 ROS2 Humble，通常对应系统 Python 3.10。ROS2 编译和运行不要在 `h1_arm` conda 环境里执行，否则 Humble 的 Python 包会被 conda Python 污染，常见报错是 `ModuleNotFoundError: No module named 'em'`。

```bash
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

安装基础依赖：

```bash
sudo apt update
sudo apt install -y \
  python3-empy \
  python3-opencv \
  python3-numpy \
  python3-colcon-common-extensions \
  ros-humble-cv-bridge \
  ros-humble-image-tools \
  ros-humble-rqt-image-view \
  ros-humble-rosidl-default-generators \
  ros-humble-rosidl-default-runtime \
  ros-humble-rmw-cyclonedds-cpp
```

检查 Python/ROS：

```bash
which python3
python3 --version
python3 -c "import em; import cv2; import numpy; import rclpy; import cv_bridge; print('numpy', numpy.__version__, numpy.__file__); print('cv2', cv2.__file__); print('ros humble python ok')"
```

如果要跑 `backend: yoloe`：

```bash
python3 -c "import ultralytics; print('ultralytics ok')"
```

如果没有 `ultralytics`，先用 `backend: mock` 完成 ROS 链路基准；YOLOE 依赖建议单独在 H2 的 Python 环境里安装和锁版本。

## 3. 编译 Foxy_ROS

```bash
cd ~/MscapeTech/Foxy_ROS
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

确认可执行入口：

```bash
ros2 pkg executables yolo_trt_ros2
ros2 interface show detector_msgs/msg/Object3D
```

## 4. 配置 H2 感知参数

编辑：

```bash
vi ~/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
```

推荐先用 mock：

```yaml
yolo_detector:
  ros__parameters:
    backend: mock
    publish_debug_image: true
```

切 YOLOE：

```yaml
yolo_detector:
  ros__parameters:
    backend: yoloe
    model_path: /home/unitree/MscapeTech/models/yoloe-11s-seg.pt
    mobileclip_path: /home/unitree/MscapeTech/models/mobileclip_blt.ts
    prompts: 'red button,green button,black knob,selector switch,rotary switch,toggle switch,control panel switch,cabinet door handle,indicator light,pilot light'
    conf_thres: 0.08
    iou_thres: 0.45
    imgsz: 640
```

如果相机是 `eye-to-hand`，配置 `T_cam2base.npy`：

```yaml
coordinate_projector:
  ros__parameters:
    handeye_npy_path: /home/unitree/MscapeTech/Hand_Eye_Calib/outputs/eye_to_hand_xxx_npy/T_cam2base.npy
    handeye_target_frame: pelvis
```

如果是当前 H2 的 `eye-in-hand` 标定，推荐配置如下：

```yaml
coordinate_projector:
  ros__parameters:
    handeye_mode: eye-in-hand
    handeye_npy_path: /home/unitree/MscapeTech/Hand_Eye_Calib/outputs/eye_in_hand_20260630_150210.json
    handeye_target_frame: torso_link
    urdf_path: /home/unitree/MscapeTech/unitree_ros/robots/h2_description/H2.urdf
    base_link: torso_link
    hand_link: right_wrist_yaw_link
    network_interface: eth0
    domain_id: 0
    lowstate_topic: rt/lowstate
    lock_waist: false
    h2_ee_offset_xyz: [0.0, 0.0, 0.0]
    publish_target_joint_state: true
    target_joint_state_topic: /detector/target_joint_state
    ik_target_link: right_wrist_yaw_link
    ik_active_joints: 'right_shoulder_pitch_joint,right_shoulder_roll_joint,right_shoulder_yaw_joint,right_elbow_joint,right_wrist_roll_joint,right_wrist_pitch_joint,right_wrist_yaw_joint'
    ik_end_effector_offset_xyz: [0.0, 0.0, 0.0]
```

其中 `/detector/target_joint_state` 是 IK 解算出的目标关节角建议值，只读发布，不会直接控制机器人运动。

网页端默认也在同一份配置里启用：

```yaml
web_dashboard:
  ros__parameters:
    web_host: '0.0.0.0'
    web_port: 8080
    debug_image_topic: /detector/debug_image
    objects_topic: /detector/objects
    objects_3d_topic: /detector/objects_3d
    target_pose_topic: /detector/target_pose
    target_joint_state_topic: /detector/target_joint_state
```

如果 `unitree_sdk2py` 只在 `h1_arm` conda 环境中能找到，不要直接在 `h1_arm` 里运行 ROS2 Humble。先用 conda 找 SDK 源码路径：

```bash
conda activate h1_arm
python3 - <<'PY'
import pathlib
import unitree_sdk2py
print(pathlib.Path(unitree_sdk2py.__file__).resolve().parents[1])
PY
conda deactivate
```

然后在 ROS2 Humble 终端中把上面打印出的目录加入 `PYTHONPATH`。常见路径示例：

```bash
export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH
python3 -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('unitree_sdk2py ok')"
```

只有这个检查通过后，`eye-in-hand` 的实时 FK/IK 才能从 `lowstate_topic` 读取关节并输出 `point_target` 和 `target_joint_state`。如果 `rt/lowstate` 收不到，可以尝试 `rt/lf/lowstate`。

## 5. 启动真实相机

如果使用 RealSense ROS2 wrapper，目标 topic 应对齐为：

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw
/camera/color/camera_info
```

检查：

```bash
ros2 topic list
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info
```

没有真实相机时，可先发布测试 RGB 图像做 mock 基准：

```bash
ros2 run image_tools cam2image --ros-args -r image:=/camera/color/image_raw
```

如果 `cam2image` 试图打开 `/dev/video0` 且失败，说明 H2 宿主机没有可用的 V4L2 相机设备；这不影响后面的纯 ROS mock 测试，可用 README 中的 `/tmp/pub_test_image.py`。

## 6. 启动感知链路

终端 A：

```bash
cd ~/MscapeTech/Foxy_ROS
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
```

启动后，在本地电脑浏览器打开：

```text
http://<H2-PC2-IP>:8080/
```

页面会显示实时 `/detector/debug_image` 视频、2D 框、类别、3D 目标点、目标位姿和 IK 关节角。也可以直接访问：

```text
http://<H2-PC2-IP>:8080/stream.mjpg
http://<H2-PC2-IP>:8080/api/state
```

终端 B 查看输出：

```bash
source /opt/ros/humble/setup.bash
source ~/MscapeTech/Foxy_ROS/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic echo /detector/objects
```

3D 坐标：

```bash
ros2 topic echo /detector/objects_3d
ros2 topic echo /detector/target_point
ros2 topic echo /detector/target_pose
ros2 topic echo /detector/target_joint_state
```

调试图：

```bash
rqt_image_view /detector/debug_image
```

## 7. 基准测试 1：ROS 链路与 topic 频率

启动 benchmark 节点：

```bash
source /opt/ros/humble/setup.bash
source ~/MscapeTech/Foxy_ROS/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 run yolo_trt_ros2 benchmark_topics_node --ros-args -p period_sec:=5.0
```

同时也可以用原生命令：

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /detector/objects
ros2 topic hz /detector/debug_image
ros2 topic hz /detector/objects_3d
ros2 topic hz /detector/target_point
ros2 topic hz /detector/target_joint_state
```

记录模板：

```text
camera/color hz:
detector/objects hz:
detector/debug_image hz:
detector/objects_3d hz:
target_point hz:
target_joint_state hz:
header age avg/min/max:
```

判定建议：

- mock 后端：`/detector/objects` 应接近输入图像频率。
- YOLOE 后端：以实际模型速度为准，先记录 `imgsz=640`，再尝试降低到 `512` 做对比。
- `/detector/objects_3d` 低于 `/detector/objects` 时，优先检查深度图和 `CameraInfo` 是否同步、深度是否有效。

## 8. 基准测试 2：端到端坐标输出

启动完整链路后，查看一个目标的字段：

```bash
timeout 5 ros2 topic echo /detector/objects_3d
timeout 5 ros2 topic echo /detector/target_point
timeout 5 ros2 topic echo /detector/target_joint_state
```

检查点：

```text
valid: true
depth_m: 合理，通常 0.2m 到 5.0m
source_frame: camera_color_optical_frame 或相机实际 frame
target_frame: torso_link/handeye_target_frame
point_camera: 单位 m
point_target: 单位 m
message: camera_frame / eye_in_hand_fk / handeye_npy / transformed
target_joint_state.name: IK 使用的右臂关节名
target_joint_state.position: IK 目标关节角，单位 rad
```

如果启用了当前 H2 的 `eye-in-hand` 配置，`message` 应为：

```text
eye_in_hand_fk
```

## 9. 基准测试 3：H2 FK 与手眼坐标一致性

只读 FK 对比，不发控制命令：

```bash
conda activate h1_arm
cd ~/MscapeTech/h2_handeye
python3 compare_h2_fk.py --iface eth0 --period 2.0 --count 5
```

如果要匹配 H2 键盘笛卡尔控制脚本的腰部锁定逻辑：

```bash
python3 compare_h2_fk.py --iface eth0 --period 2.0 --count 5 --lock-waist
```

记录模板：

```text
ours_xyz:
hw_xyz:
delta_mm:
norm_mm:
rot_deg:
lock_waist 是否显著改善:
```

建议阈值：

- FK 位置差：先以 `norm_mm < 5 mm` 作为工程基线。
- FK 姿态差：先以 `rot_deg < 1 deg` 作为工程基线。
- 若 `--lock-waist` 后误差明显变小，应统一感知链路和控制链路的腰部处理。

## 10. 基准测试 4：H2 运动前 dry-run

门面接近规划 dry-run：

```bash
cd ~/MscapeTech
python -m h2_pipeline.part1.run_live_realsense \
  --bbox 220 80 1050 690 \
  --iface eth0 \
  --stand-off 0.75
```

只检查 JSON，不移动机器人：

```text
success:
forward_m:
left_m:
yaw_rad:
stand_off_m:
```

确认方向和幅度正确、现场清空后，才允许执行：

```bash
python -m h2_pipeline.part1.run_live_realsense \
  --bbox 220 80 1050 690 \
  --iface eth0 \
  --stand-off 0.75 \
  --execute
```

## 11. 推荐跑测顺序

```text
1. colcon build 通过
2. mock + 假 RGB 图像，确认 /detector/objects
3. mock + 真实 RGB-D，确认 /detector/objects_3d
4. YOLOE + 真实 RGB-D，记录 hz 和 header age
5. 加 handeye_npy_path 或 TF，确认 target_frame、point_target 和 target_joint_state
6. H2 compare_h2_fk.py，只读验证 FK 误差
7. h2_pipeline part1 dry-run，只看 JSON
8. 现场清空后再执行任何 H2 运动命令
```

## 12. 常见问题

### topic 看不到

```bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic list
```

必要时改 CycloneDDS：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 网页打不开

先确认节点启动日志里有：

```text
Web dashboard started: http://<H2-IP>:8080/
```

再在 H2 上检查端口：

```bash
ss -ltnp | grep 8080
curl http://127.0.0.1:8080/api/state
```

如果 H2 本机能 curl，本地电脑打不开，优先检查两边是否在同一网段、是否能 ping 通 H2，以及现场网络是否拦截 8080 端口。端口冲突时把 `inspection_perception.yaml` 里的 `web_port` 改成 `8081` 后重新启动。

如果 SSH 能连但网页端口被网络限制，可以在本地电脑开隧道：

```bash
ssh -L 8080:127.0.0.1:8080 unitree@<H2-PC2-IP>
```

然后本地浏览器访问：

```text
http://127.0.0.1:8080/
```

### debug image 有，3D 没有

检查：

```bash
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info
ros2 topic echo /detector/objects
```

常见原因：

- 深度图没有对齐到 color。
- bbox 中心点处深度为空。
- `depth_scale` 不匹配，RealSense `16UC1` 通常是 `0.001` m/mm。

### YOLOE 启动失败

先切回 mock：

```yaml
backend: mock
```

确认 ROS 链路没问题后，再检查：

```bash
python3 -c "from ultralytics import YOLOE; print('YOLOE ok')"
ls -lh /home/unitree/MscapeTech/models/yoloe-11s-seg.pt
```

如果 prompt 模式报 MobileCLIP/BLT 权重损坏或下载失败，先把本地 `mobileclip_blt.ts` 上传到：

```text
/home/unitree/MscapeTech/models/mobileclip_blt.ts
```

配置中指定：

```yaml
mobileclip_path: /home/unitree/MscapeTech/models/mobileclip_blt.ts
```

然后重新 `colcon build --symlink-install`。该参数会让 YOLOE 文本 prompt 编码阶段直接读取指定文件，绕过 Ultralytics 自动下载的损坏缓存。

### cv2 / cv_bridge 报 NumPy 2.x 不兼容

如果启动时报：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

说明系统里的 `cv2` 或 `cv_bridge` 是按 NumPy 1.x 编译的，但当前 Python 优先加载了 pip/用户目录里的 NumPy 2.x。H2 Humble 感知链路建议使用系统 Python 和 NumPy 1.x。

先确认来源：

```bash
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash

which python3
python3 -c "import numpy; print(numpy.__version__, numpy.__file__)"
python3 -m pip show numpy opencv-python opencv-contrib-python opencv-python-headless
```

修复方式优先使用系统 apt 包，并移除 pip 版 OpenCV：

```bash
python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python3 -m pip install --user "numpy<2"

sudo apt install --reinstall -y \
  python3-numpy \
  python3-opencv \
  ros-humble-cv-bridge
```

重新检查：

```bash
python3 -c "import numpy, cv2; from cv_bridge import CvBridge; print('numpy', numpy.__version__, numpy.__file__); print('cv2', cv2.__file__); print('cv_bridge ok')"
```

如果 `numpy.__version__` 仍然是 `2.x`，说明还有更高优先级的 pip 包在覆盖系统包，需要继续清理当前 `python3 -m pip show numpy` 显示的位置。

### 手眼文件加载失败

`coordinate_projector_node` 当前直接支持：

```text
T_cam2base.npy
T_cam2world.npy
T_camera2base.npy
T_camera2world.npy
```

文件必须是 `4x4` 的 numpy 矩阵，且单位为米。
