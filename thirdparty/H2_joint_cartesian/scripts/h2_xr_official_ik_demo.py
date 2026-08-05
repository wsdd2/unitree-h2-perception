#!/usr/bin/env python3
"""H2 dual-arm Cartesian IK demo based on Unitree xr_teleoperate control.

脚本主链路：
1. 用 xr_teleoperate/assets/h2/H2.urdf 建 Pinocchio 模型。
2. 锁住腿、腰、头，只保留双臂 14 个关节做 IK。
3. 在腕部追加 Dex1-1 夹爪负载，用 Pinocchio RNEA 计算重力前馈 tauff。
4. 调用官方 H2_ArmController.ctrl_dual_arm(q_target, tauff_target) 周期下发。
5. 结束时同时打印 IK 误差、真实 FK 执行误差、每个关节跟踪误差。

注意：官方 H2_ArmIK 依赖 pinocchio.casadi；当前 pip Pinocchio 环境没有这个模块。
因此这里使用本地兼容的 Pinocchio 数值 IK，但控制下发仍走官方 H2_ArmController。
"""

import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VENV = Path(sys.prefix)
XR_ROOT = ROOT / "third_party/xr_teleoperate"
ROBOT_CONTROL = XR_ROOT / "teleop/robot_control"
CMEEL_PREFIX = VENV / "lib/python3.12/site-packages/cmeel.prefix"
CYCLONEDDS_LIB = ROOT / "cyclonedds/install/lib"


def relaunch_with_clean_pinocchio_env():
    """重新启动一次脚本，隔离 Pinocchio/ROS/其他工作区的动态库冲突。

    H2 项目里同时存在 ROS、Franka、Kinova、cmeel Pinocchio 等库路径。
    如果 LD_LIBRARY_PATH 混在一起，Pinocchio 可能出现 symbol lookup error。
    这里通过 os.execve 重启当前 Python，并只保留当前 demo 需要的库路径。
    """
    if os.environ.get("H2_XR_PINOCCHIO_ENV") == "clean":
        return

    env = os.environ.copy()
    env["H2_XR_PINOCCHIO_ENV"] = "clean"
    env.pop("PYTHONPATH", None)

    keep_libs = []
    for path in env.get("LD_LIBRARY_PATH", "").split(":"):
        if not path:
            continue
        if VENV.name != ".venv_xr" and (
            "/opt/ros/" in path or "/franka_ros2_ws/" in path or "/kinova_pro/" in path
        ):
            continue
        keep_libs.append(path)

    pin_libs = [
        str(CMEEL_PREFIX / "lib"),
        str(CMEEL_PREFIX / "lib64"),
        str(CYCLONEDDS_LIB),
    ]
    if VENV.name == ".venv_xr":
        pin_libs.extend([
            "/opt/ros/jazzy/lib/x86_64-linux-gnu",
            "/opt/ros/jazzy/lib",
        ])
    env["LD_LIBRARY_PATH"] = ":".join(pin_libs + keep_libs)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


relaunch_with_clean_pinocchio_env()

sys.path.insert(0, str(CMEEL_PREFIX / "lib/python3.12/site-packages"))
sys.path.insert(0, str(XR_ROOT))
sys.path.insert(0, str(ROBOT_CONTROL))

import numpy as np
import pinocchio as pin

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from robot_arm import H2_ArmController, H2_JointIndex


class H2CompatibleIK:
    """Pinocchio IK compatible with official H2_ArmController q/tau interface.

    输出:
    - q: 双臂 14 关节目标角，顺序与 H2_ArmController 的 q_target 一致。
    - tauff: 双臂 14 关节前馈力矩，直接传给官方 ctrl_dual_arm。

    这里没有控制腿/腰/头，所以先 buildReducedRobot 锁住非手臂关节。
    """

    def __init__(self, gripper_payload_mass=0.55, gripper_payload_com=(0.07, 0.0, 0.0)):
        urdf_path = XR_ROOT / "assets/h2/H2.urdf"
        model_dir = XR_ROOT / "assets/h2"
        robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), str(model_dir))
        joints_to_lock = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "head_pitch_joint",
            "head_yaw_joint",
        ]
        # reduced model 只保留双臂关节，IK 维度从全身缩到 14 维。
        self.robot = robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.zeros(robot.model.nq),
        )
        self.model = self.robot.model
        self.data = self.model.createData()
        self.gripper_payload_mass = float(gripper_payload_mass)
        self.gripper_payload_com = np.array(gripper_payload_com, dtype=float)
        if self.gripper_payload_mass > 0.0:
            # 把夹爪当作固定在左右 wrist_yaw_joint 上的刚体负载。
            # 这样 pin.rnea() 算出来的 tauff 会包含夹爪重力补偿。
            self._append_gripper_payload("left_wrist_yaw_joint")
            self._append_gripper_payload("right_wrist_yaw_joint")
        # 末端控制点直接使用真实 wrist_yaw_joint 原点，不再前移到虚拟 R_ee(+0.05)。
        self.model.addFrame(
            pin.Frame(
                "L_ee",
                self.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3.Identity(),
                pin.FrameType.OP_FRAME,
            )
        )
        self.model.addFrame(
            pin.Frame(
                "R_ee",
                self.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3.Identity(),
                pin.FrameType.OP_FRAME,
            )
        )
        self.l_frame = self.model.getFrameId("L_ee")
        self.r_frame = self.model.getFrameId("R_ee")
        self.data = self.model.createData()

    def _append_gripper_payload(self, wrist_joint_name):
        """Add a Dex1-1-like payload inertia to the wrist joint.

        这个函数不修改 URDF 文件，只修改当前进程里的 Pinocchio model。
        因此可以通过命令行快速试 mass/COM，不会污染原始模型。
        """
        wrist_joint_id = self.model.getJointId(wrist_joint_name)
        if wrist_joint_id == 0:
            raise RuntimeError(f"missing wrist joint in Pinocchio model: {wrist_joint_name}")

        # 静态重力前馈主要由 mass 和 COM 决定。
        # 惯量张量对当前 v=0/a=0 的重力补偿影响很小，但 RNEA 需要完整 Inertia。
        # 这里用 Dex1-1 尺寸做盒体近似：143mm x 78mm x 67mm。
        size = np.array([0.143, 0.078, 0.067], dtype=float)
        ixx = self.gripper_payload_mass / 12.0 * (size[1] ** 2 + size[2] ** 2)
        iyy = self.gripper_payload_mass / 12.0 * (size[0] ** 2 + size[2] ** 2)
        izz = self.gripper_payload_mass / 12.0 * (size[0] ** 2 + size[1] ** 2)
        inertia = pin.Inertia(
            self.gripper_payload_mass,
            self.gripper_payload_com,
            np.diag([ixx, iyy, izz]),
        )
        self.model.appendBodyToJoint(wrist_joint_id, inertia, pin.SE3.Identity())

    def solve_ik(self, left_target, right_target, current_q, current_dq=None, return_diagnostics=False):
        """Solve dual-arm SE3 IK and compute gravity feedforward.

        left_target/right_target:
            Pinocchio SE3，分别表示左右末端目标位姿。
        current_q:
            当前双臂 14 关节角，作为 IK 初值。用当前状态做初值可减少跳变。
        return_diagnostics:
            True 时返回 planned FK 和 IK 误差，便于区分 IK 误差与真实执行误差。
        """
        q = np.array(current_q, dtype=float).copy()
        damping = 1e-4
        step_scale = 0.35
        iterations = 0
        final_err = None

        for iteration in range(80):
            iterations = iteration + 1
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            # SE3 位姿误差，包含平移和旋转。这里目标姿态基本保持当前姿态，
            # 所以我们主要关心最终打印的 position error mm。
            l_err = pin.log(self.data.oMf[self.l_frame].actInv(left_target)).vector
            r_err = pin.log(self.data.oMf[self.r_frame].actInv(right_target)).vector
            err = np.concatenate([l_err, r_err])
            final_err = err.copy()
            if np.linalg.norm(err) < 2e-4:
                break

            jl = pin.computeFrameJacobian(self.model, self.data, q, self.l_frame, pin.ReferenceFrame.LOCAL)
            jr = pin.computeFrameJacobian(self.model, self.data, q, self.r_frame, pin.ReferenceFrame.LOCAL)
            jac = np.vstack([jl, jr])
            # 阻尼最小二乘 IK：
            # dq = J^T (J J^T + lambda I)^-1 error
            # 这种写法比直接伪逆更稳，不容易在奇异位形附近炸。
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(12), err)
            # 限制每次 IK 迭代关节步长，避免目标较远时解跳太快。
            dq = np.clip(dq, -0.04, 0.04)
            q = pin.integrate(self.model, q, step_scale * dq)
            q = np.minimum(np.maximum(q, self.model.lowerPositionLimit), self.model.upperPositionLimit)

        # 速度/加速度置零时，RNEA 主要输出当前姿态所需的重力补偿力矩。
        # 这就是发送给官方控制器的 tauff。
        v = np.zeros(self.model.nv) if current_dq is None else np.array(current_dq, dtype=float) * 0.0
        tauff = pin.rnea(self.model, self.data, q, v, np.zeros(self.model.nv))
        if return_diagnostics:
            planned_left, planned_right = current_ee_poses(self, q)
            left_pos_error = left_target.translation - planned_left.translation
            right_pos_error = right_target.translation - planned_right.translation
            diagnostics = {
                "iterations": iterations,
                "se3_error_norm": float(np.linalg.norm(final_err)) if final_err is not None else 0.0,
                "left_planned": planned_left,
                "right_planned": planned_right,
                "left_pos_error": left_pos_error,
                "right_pos_error": right_pos_error,
            }
            return q, tauff, diagnostics
        return q, tauff


def make_pose(xyz):
    """Create a pure-translation SE3 target with identity orientation."""
    return pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array(xyz, dtype=float))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Unitree xr_teleoperate H2_ArmController with compatible H2 IK + tauff."
    )
    parser.add_argument("iface", help="Network interface, e.g. enp4s0")
    parser.add_argument("--left", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--right", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--arm", choices=["left", "right", "both"], default="both", help="Which arm applies --delta.")
    parser.add_argument("--delta", nargs=3, type=float, default=[0.0, 0.0, 0.01], metavar=("DX", "DY", "DZ"))
    parser.add_argument("--steps", type=int, default=40, help="Number of small IK/control updates.")
    parser.add_argument("--dt", type=float, default=0.02, help="Seconds between IK updates.")
    parser.add_argument("--settle", type=float, default=2.0, help="Seconds to hold current official controller target first.")
    parser.add_argument("--final-hold", type=float, default=1.0, help="Seconds to keep final target before measuring FK error.")
    parser.add_argument("--tau-scale", type=float, default=1.0, help="Scale official IK tauff before sending.")
    parser.add_argument(
        "--gripper-payload-mass",
        type=float,
        default=0.55,
        help="Dex1-1 payload mass appended to each wrist for Pinocchio RNEA tauff. Use 0 to disable.",
    )
    parser.add_argument(
        "--gripper-payload-com",
        nargs=3,
        type=float,
        default=[0.07, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Dex1-1 payload COM in wrist_yaw_joint frame, meters.",
    )
    parser.add_argument(
        "--right-wrist-pitch-kp",
        type=float,
        default=None,
        help="Override official H2 right_wrist_pitch motor kp. Default keeps official kp=50.",
    )
    parser.add_argument(
        "--right-wrist-pitch-kd",
        type=float,
        default=None,
        help="Override official H2 right_wrist_pitch motor kd. Default keeps official kd=2.",
    )
    parser.add_argument(
        "--arm-kp-overrides",
        nargs=14,
        type=float,
        default=None,
        metavar=("KP"),
        help="Override H2 arm kp values in JOINT_NAMES order. Omit to keep official gains.",
    )
    parser.add_argument(
        "--arm-kd-overrides",
        nargs=14,
        type=float,
        default=None,
        metavar=("KD"),
        help="Override H2 arm kd values in JOINT_NAMES order. Omit to keep official gains.",
    )
    parser.add_argument("--motion-mode", action="store_true", default=True, help="Use rt/arm_sdk topic.")
    parser.add_argument("--debug-lowcmd", action="store_true", help="Use rt/lowcmd debug topic instead of rt/arm_sdk.")
    parser.add_argument("--dry-run", action="store_true", help="Solve IK only; do not initialize DDS or publish commands.")
    return parser.parse_args()


def current_ee_poses(ik, q):
    """Run FK for q and return current left/right end-effector SE3."""
    pin.forwardKinematics(ik.model, ik.data, q)
    pin.updateFramePlacements(ik.model, ik.data)
    return ik.data.oMf[ik.l_frame].copy(), ik.data.oMf[ik.r_frame].copy()


JOINT_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

# JOINT_NAMES/q_target 的 14 维顺序，与官方 H2_ArmController 枚举顺序对齐。
# 注意官方枚举里 left wrist yaw 写成了 kLeftWristyaw，小写 y。
JOINT_INDEX_BY_ARM_Q_INDEX = [
    H2_JointIndex.kLeftShoulderPitch,
    H2_JointIndex.kLeftShoulderRoll,
    H2_JointIndex.kLeftShoulderYaw,
    H2_JointIndex.kLeftElbow,
    H2_JointIndex.kLeftWristRoll,
    H2_JointIndex.kLeftWristPitch,
    H2_JointIndex.kLeftWristyaw,
    H2_JointIndex.kRightShoulderPitch,
    H2_JointIndex.kRightShoulderRoll,
    H2_JointIndex.kRightShoulderYaw,
    H2_JointIndex.kRightElbow,
    H2_JointIndex.kRightWristRoll,
    H2_JointIndex.kRightWristPitch,
    H2_JointIndex.kRightWristYaw,
]


def apply_arm_gain_overrides(arm, args):
    """Apply optional Kp/Kd overrides to official H2_ArmController message.

    不传 --arm-kp-overrides / --arm-kd-overrides 时，完全保留官方默认值：
    shoulder/elbow kp=140 kd=3, wrist kp=50 kd=2。

    传入 14 个数时，按 JOINT_NAMES 顺序覆盖。覆盖值直接写到
    arm.msg.motor_cmd[...]，随后官方发布线程会用这些 K/D 周期下发。
    """
    if args.arm_kp_overrides is not None:
        for index, kp in enumerate(args.arm_kp_overrides):
            arm.msg.motor_cmd[JOINT_INDEX_BY_ARM_Q_INDEX[index]].kp = float(kp)
    if args.arm_kd_overrides is not None:
        for index, kd in enumerate(args.arm_kd_overrides):
            arm.msg.motor_cmd[JOINT_INDEX_BY_ARM_Q_INDEX[index]].kd = float(kd)

    if args.right_wrist_pitch_kp is not None:
        arm.msg.motor_cmd[H2_JointIndex.kRightWristPitch].kp = float(args.right_wrist_pitch_kp)
    if args.right_wrist_pitch_kd is not None:
        arm.msg.motor_cmd[H2_JointIndex.kRightWristPitch].kd = float(args.right_wrist_pitch_kd)

    print("active_arm_gains_begin")
    for index, name in enumerate(JOINT_NAMES):
        joint = JOINT_INDEX_BY_ARM_Q_INDEX[index]
        print(
            "active_arm_gain "
            f"idx={index} "
            f"name={name} "
            f"motor={joint.value} "
            f"kp={arm.msg.motor_cmd[joint].kp} "
            f"kd={arm.msg.motor_cmd[joint].kd}"
        )
    print("active_arm_gains_end")


def main():
    args = parse_args()
    motion_mode = args.motion_mode and not args.debug_lowcmd

    print("WARNING: clear the workspace around H2 before running official xr_teleoperate H2 IK demo.")
    print("This uses official H2_ArmController.ctrl_dual_arm(sol_q, sol_tauff).")
    print("IK is a local Pinocchio-compatible solver because pip Pinocchio lacks official pinocchio.casadi.")
    input("Press Enter to continue...")

    old_cwd = Path.cwd()
    os.chdir(ROBOT_CONTROL)
    try:
        arm_ik = H2CompatibleIK(
            gripper_payload_mass=args.gripper_payload_mass,
            gripper_payload_com=args.gripper_payload_com,
        )
        print(
            "gripper_payload_model "
            f"mass_kg={args.gripper_payload_mass} "
            f"com_m={list(args.gripper_payload_com)}"
        )

        if args.dry_run:
            # dry-run 只检查 IK 和 tauff 计算，不初始化 DDS、不发机器人命令。
            q0 = np.zeros(14)
            dq0 = np.zeros(14)
            left_target = make_pose(args.left or [0.25, 0.25, 0.10])
            right_target = make_pose(args.right or [0.25, -0.25, 0.10])
            sol_q, sol_tauff = arm_ik.solve_ik(left_target, right_target, q0, dq0)
            print("dry_run_h2_compatible_ik")
            print(f"sol_q={np.array2string(sol_q, precision=6, suppress_small=False)}")
            print(f"sol_tauff={np.array2string(sol_tauff, precision=6, suppress_small=False)}")
            return

        ChannelFactoryInitialize(0, args.iface)
        # 官方控制器内部会启动订阅线程和 250Hz 发布线程。
        # motion_mode=True 时使用 rt/arm_sdk；debug_lowcmd 时使用 rt/lowcmd。
        arm = H2_ArmController(motion_mode=motion_mode, simulation_mode=False)
        apply_arm_gain_overrides(arm, args)

        print(f"official_h2_controller_ready motion_mode={motion_mode} tau_scale={args.tau_scale}")
        time.sleep(args.settle)

        current_q = arm.get_current_dual_arm_q()
        left_start, right_start = current_ee_poses(arm_ik, current_q)
        # 默认用“当前末端位置 + delta”做相对运动。
        # 如果传 --left/--right，则用显式位置覆盖起点。
        if args.left is not None:
            left_start = make_pose(args.left)
        if args.right is not None:
            right_start = make_pose(args.right)
        print(f"start_left_xyz={left_start.translation.tolist()}")
        print(f"start_right_xyz={right_start.translation.tolist()}")
        delta = np.array(args.delta, dtype=float)

        for step in range(args.steps + 1):
            # 把总 delta 分成 steps 份，每一步重新读取真实关节角并重新 IK。
            # 这比一次性求最终目标更稳，也更接近在线伺服。
            ratio = step / max(args.steps, 1)
            left_delta = ratio * delta if args.arm in ("left", "both") else np.zeros(3)
            right_delta = ratio * delta if args.arm in ("right", "both") else np.zeros(3)
            left_target = pin.SE3(left_start.rotation, left_start.translation + left_delta)
            right_target = pin.SE3(right_start.rotation, right_start.translation + right_delta)

            current_q = arm.get_current_dual_arm_q()
            current_dq = arm.get_current_dual_arm_dq()
            # 每个控制周期：当前关节角 -> IK -> q/tauff -> 官方控制器下发。
            ik_result = arm_ik.solve_ik(
                left_target,
                right_target,
                current_q,
                current_dq,
                return_diagnostics=(step == args.steps),
            )
            if step == args.steps:
                sol_q, sol_tauff, final_ik_diag = ik_result
            else:
                sol_q, sol_tauff = ik_result
            # 这里是核心下发接口：
            # q_target 进入官方位置 PD，sol_tauff 作为前馈力矩一起发送。
            arm.ctrl_dual_arm(sol_q, sol_tauff * args.tau_scale)
            if step % 10 == 0 or step == args.steps:
                print(
                    "official_h2_step "
                    f"step={step}/{args.steps} "
                    f"left_xyz={left_target.translation.tolist()} "
                    f"right_xyz={right_target.translation.tolist()} "
                    f"sol_q0={sol_q[0]:.6f} sol_q7={sol_q[7]:.6f} "
                    f"tau_norm={float(np.linalg.norm(sol_tauff)):.6f}"
                )
            time.sleep(args.dt)

        final_left_target = left_target.copy()
        final_right_target = right_target.copy()
        final_sol_q = sol_q.copy()
        # 给控制器一点时间保持最终目标，再读真实关节和 FK 评估误差。
        time.sleep(args.final_hold)
        actual_q = arm.get_current_dual_arm_q()
        actual_left, actual_right = current_ee_poses(arm_ik, actual_q)
        left_error = final_left_target.translation - actual_left.translation
        right_error = final_right_target.translation - actual_right.translation
        left_ik_error = final_ik_diag["left_pos_error"]
        right_ik_error = final_ik_diag["right_pos_error"]
        q_error = final_sol_q - actual_q
        print(
            "official_h2_final_ik "
            f"iterations={final_ik_diag['iterations']} "
            f"se3_error_norm={final_ik_diag['se3_error_norm']:.6g} "
            f"left_target_xyz={final_left_target.translation.tolist()} "
            f"left_planned_xyz={final_ik_diag['left_planned'].translation.tolist()} "
            f"left_ik_error_m={left_ik_error.tolist()} "
            f"left_ik_error_mm={float(np.linalg.norm(left_ik_error) * 1000.0):.3f} "
            f"right_target_xyz={final_right_target.translation.tolist()} "
            f"right_planned_xyz={final_ik_diag['right_planned'].translation.tolist()} "
            f"right_ik_error_m={right_ik_error.tolist()} "
            f"right_ik_error_mm={float(np.linalg.norm(right_ik_error) * 1000.0):.3f}"
        )
        # official_h2_final_ik: IK 本身误差。
        # 如果这里是 0.02mm，而 final_error 是 2mm，就说明误差来自执行层。
        print(
            "official_h2_final_error "
            f"left_target_xyz={final_left_target.translation.tolist()} "
            f"left_actual_xyz={actual_left.translation.tolist()} "
            f"left_error_m={left_error.tolist()} "
            f"left_error_mm={float(np.linalg.norm(left_error) * 1000.0):.3f} "
            f"right_target_xyz={final_right_target.translation.tolist()} "
            f"right_actual_xyz={actual_right.translation.tolist()} "
            f"right_error_m={right_error.tolist()} "
            f"right_error_mm={float(np.linalg.norm(right_error) * 1000.0):.3f}"
        )
        # official_h2_joint_error: 每个关节 target_q - actual_q。
        # 用它定位到底是哪几个关节没有跟上 IK 解出来的目标。
        print("official_h2_joint_error_begin")
        for index, name in enumerate(JOINT_NAMES):
            print(
                "official_h2_joint_error "
                f"idx={index} "
                f"name={name} "
                f"target_q={final_sol_q[index]:.6f} "
                f"actual_q={actual_q[index]:.6f} "
                f"error_rad={q_error[index]:.6f} "
                f"error_deg={np.degrees(q_error[index]):.6f}"
            )
        print("official_h2_joint_error_end")
        print("official_h2_demo_complete")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
