import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# =========================
# LQR增益（来自离线求解）
# =========================
# 状态顺序通常为：
# [pitch, pitch_dot, position, velocity_error]
#
# 注意：
# - K 的每一项对应一个状态权重
# - 这里 velocity_error 是速度环外环误差
LQR_K = [ -4.042462128338206,
          -0.015403934099089448,
           1.8722600175113512e-17,
           2.2360679774997942]

# 其他备选实验参数（已注释，不参与运行）
# LQR_K = [-30.14, -0.035, 0, 6.236]
# LQR_K = [-2.14, -0.035, 0, 2.236]

# =========================
# 物理参数
# =========================
# 轮半径（用于：角速度 ↔ 线速度转换）
WHEEL_RADIUS = 0.034

# 电机最大角速度限制（rad/s）
MAX_MOTOR_VEL = 500.0


def clamp(n, minn, maxn):
    """
    限幅函数：
    防止控制输出超过电机能力
    """
    return max(min(maxn, n), minn)


"""
========================================================
                基于LQR的平衡控制器
========================================================
控制结构：
    外环：速度误差（velocity error）
    内环：姿态稳定（pitch / pitch_dot）

输入：
    MuJoCo模型 + 数据

输出：
    左右轮速度控制（ctrl）
========================================================
"""
class RobotLqr:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        # =========================
        # 状态变量（控制器内部缓存）
        # =========================

        # 当前角速度（未直接使用，保留扩展）
        self.velocity_angular = 0.0

        # 期望线速度（控制目标）
        self.velocity_linear_set_point = 0.0

        # yaw控制输入（左右轮差动控制）
        self.yaw = 0

        # 滤波后的 pitch 角速度（降低噪声）
        self.pitch_dot_filtered = 0.0

        # 滤波后的轮速（用于速度估计）
        self.velocity_angular_filtered = 0.0

    # =====================================================
    # 设置目标线速度
    # =====================================================
    def set_velocity_linear_set_point(self, vel: float) -> None:
        """
        设置机器人期望前进速度（m/s）
        """
        self.velocity_linear_set_point = vel

    # =====================================================
    # 设置 yaw（转向控制）
    # =====================================================
    def set_yaw(self, yaw: float) -> None:
        """
        yaw 控制说明：
        - 左右轮速度差 = yaw
        - 正值：向一侧转弯
        """
        self.yaw = yaw

    # =====================================================
    # 获取 pitch（俯仰角）
    # =====================================================
    def get_pitch(self) -> float:
        """
        从 body 四元数计算 pitch（绕x轴）
        """
        quat = self.data.body("robot_body").xquat

        # 防止非法 quaternion
        if quat[0] == 0:
            return 0

        # MuJoCo quaternion: [w, x, y, z]
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])

        # 转欧拉角（xyz）
        angles = rotation.as_euler('xyz', degrees=False)

        # pitch
        return angles[0]

    # =====================================================
    # 获取 pitch 角速度
    # =====================================================
    def get_pitch_dot(self) -> float:
        """
        从 free joint 角速度提取 pitch_dot
        """
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        return angular[0]

    # =====================================================
    # 轮速估计（左右轮平均）
    # =====================================================
    def get_wheel_velocity(self) -> float:
        """
        说明：
        - 左右轮方向相反（结构安装导致）
        - 所以需要符号修正
        """
        vel_m_0 = self.data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = self.data.joint('torso_r_wheel').qvel[0]

        # 左轮方向取反 + 右轮方向直接用
        return (vel_m_0 * -1 + vel_m_1) / 2.0

    # =====================================================
    # LQR核心控制计算
    # =====================================================
    def calculate_lqr_velocity(self) -> float:
        """
        输出：
            目标角速度（rad/s）

        控制结构：
            pitch        -> 稳定性主项
            pitch_dot    -> 阻尼项
            velocity_err  -> 速度外环
        """

        # pitch 取负号：统一坐标方向
        pitch = -self.get_pitch()
        pitch_dot = self.get_pitch_dot()

        # =========================
        # 一阶低通滤波（减少噪声）
        # =========================
        self.pitch_dot_filtered = (
            self.pitch_dot_filtered * .975 +
            pitch_dot * .025
        )

        self.velocity_angular_filtered = (
            self.velocity_angular_filtered * .975 +
            self.get_wheel_velocity() * .025
        )

        # =========================
        # 速度误差（外环）
        # =========================
        velocity_linear_error = (
            self.velocity_linear_set_point -
            self.velocity_angular_filtered * WHEEL_RADIUS
        )

        # =========================
        # LQR组合控制律
        # =========================
        lqr_v = (
            LQR_K[0] * (0 - pitch) +
            LQR_K[1] * self.pitch_dot_filtered +
            LQR_K[2] * 0 +
            LQR_K[3] * velocity_linear_error
        )

        # 转换为轮角速度
        return -lqr_v / WHEEL_RADIUS

    # =====================================================
    # 更新电机控制
    # =====================================================
    def update_motor_speed(self) -> None:
        """
        将 LQR 输出转换为左右轮速度控制
        """

        # 控制量（rad/s）
        vel = self.calculate_lqr_velocity()

        # 限幅防止电机饱和
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        # 差速驱动：
        # 左轮 = -vel + yaw
        # 右轮 =  vel + yaw
        self.data.actuator('motor_l_wheel').ctrl = [-vel + self.yaw]
        self.data.actuator('motor_r_wheel').ctrl = [vel + self.yaw]

    # =====================================================
    # 复位函数（随机初始化姿态）
    # =====================================================
    def reset(self):
        """
        重置控制器状态 + 随机姿态初始化
        """

        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

        # 随机朝向（用于训练/测试稳定性）
        x_rot = (np.random.random() - 0.5) * 2 * math.pi
        y_rot = (np.random.random() - 0.5) * 0.4
        z_rot = (np.random.random() - 0.5) * 0.4

        euler_angles = [x_rot, y_rot, z_rot]

        rotation = Rotation.from_euler('xyz', euler_angles)

        # 写入 MuJoCo 状态
        self.data.qpos[3:7] = rotation.as_quat()

        # 电机清零
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]