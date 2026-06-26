import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.1275               # 轮子半径 (m)
MAX_MOTOR_VEL = 300.0               # 与 XML 的 ctrlrange 一致

# 平衡参数（角度/角速度 → 速度指令，与参考模型一致）
PITCH_KP = 3.0
PITCH_KD = 1.1

# 速度控制参数
SPEED_KP = 2.3
SPEED_KI = 0.35
SPEED_INT_LIMIT = 0.8

# 偏航控制参数
YAW_DEADZONE = 0.01
YAW_GAIN = 1.0

# ========== 极性开关 ==========
INVERT_BALANCE = False              # 如果小车向前倒，改为 True
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class SegwayPID:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.roll_dot_filtered = 0.0
        self.speed_error_integral = 0.0
        self.target_pitch_dynamic = 0.0

        # 模型关键 ID
        self.body_id = model.body('segway').id
        self.free_dofadr = model.jnt_dofadr[model.joint('segway_free').id]
        self.l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]

        self.step_cnt = 0
        self.last_pitch = 0.0
        self.last_roll = 0.0

    def set_velocity_linear_set_point(self, vel):
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw):
        if abs(yaw) < YAW_DEADZONE:
            self.yaw = 0.0
        else:
            self.yaw = yaw * YAW_GAIN

    # ---------- 俯仰角（前后倾斜） ----------
    def get_pitch(self) -> float:
        """前倾为正（rad）"""
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        angles = rotation.as_euler('xyz', degrees=False)
        return angles[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('segway_free').qvel[-3:]
        return angular[0]

    # ---------- 侧倾角（左右倾斜） ----------
    def get_roll(self) -> float:
        """右倾为正"""
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        angles = rotation.as_euler('xyz', degrees=False)
        return angles[1]              # 绕 y 轴的欧拉角（在 xyz 顺序下）

    def get_roll_dot(self) -> float:
        angular = self.data.joint('segway_free').qvel[-3:]
        return angular[1]

    # ---------- 轮速 ----------
    def get_wheel_velocity_avg(self) -> float:
        l = self.data.qvel[self.l_dof]
        r = self.data.qvel[self.r_dof]
        return (l + r) / 2.0

    # ---------- 控制核心 ----------
    def calculate_common_speed(self) -> float:
        """计算同向速度指令（rad/s）"""
        self.last_pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot()
        wheel_vel = self.get_wheel_velocity_avg()
        wheel_vel_linear = wheel_vel * WHEEL_RADIUS

        # 滤波
        self.pitch_dot_filtered = 0.975 * self.pitch_dot_filtered + 0.025 * pitch_dot
        self.velocity_angular_filtered = 0.975 * self.velocity_angular_filtered + 0.025 * wheel_vel

        # 速度误差：实际 - 目标
        vel_error = self.velocity_angular_filtered * WHEEL_RADIUS - self.velocity_linear_set_point
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -SPEED_INT_LIMIT, SPEED_INT_LIMIT)

        # 计算目标俯仰角
        target_pitch = 0.5 * (SPEED_KP * vel_error + SPEED_KI * self.speed_error_integral)
        self.target_pitch_dynamic = 0.9 * self.target_pitch_dynamic + 0.1 * target_pitch # 对目标俯仰角做低通滤波（使用成员变量）
        pitch_error = self.target_pitch_dynamic - self.last_pitch

        motor_vel = PITCH_KP * pitch_error - PITCH_KD * self.pitch_dot_filtered

        if INVERT_BALANCE:
            motor_vel = -motor_vel
        return motor_vel / WHEEL_RADIUS

    def update_motor_speed(self):
        vel_common = self.calculate_common_speed()
        vel_common = clamp(vel_common, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        # ---------- 偏航差速 ----------
        wheel_base = 0.25 * 2         # 轮距 (m)
        yaw_diff = self.yaw * wheel_base / (2 * WHEEL_RADIUS)
        diff = yaw_diff

        # 最终左右轮速度
        left_vel = vel_common - diff
        right_vel = vel_common + diff

        left_vel = clamp(left_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        right_vel = clamp(right_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        self.data.actuator('motor_l_wheel').ctrl[0] = left_vel
        self.data.actuator('motor_r_wheel').ctrl[0] = right_vel

        # 调试打印
        self.step_cnt += 1
        if self.step_cnt % 200 == 0:
            print(f"pitch={np.degrees(self.last_pitch):6.2f}°, roll={np.degrees(self.last_roll):5.2f}°, "
                  f"vel_common={vel_common:7.2f}, diff={diff:6.2f}, "
                  f"left={left_vel:7.2f}, right={right_vel:7.2f}")

    def reset(self):
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.roll_dot_filtered = 0.0
        self.speed_error_integral = 0.0
        # 完全直立
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]