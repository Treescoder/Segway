# segway_ball_pid.py —— 串级 PID 平衡控制器（修正初始化 + 方向可调 + 前馈）
import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.1275
MAX_MOTOR_VEL = 100.0          # 轮速限制 (rad/s)
WHEEL_BASE = 0.5               # 轮距 (m)

# ★★★ 方向常量 ★★★
WHEEL_DIRECTION = 1    # 正轮速对应前进：给左右轮正速度，若前进则 1，若后退则 -1
BALANCE_SIGN = -1      # 角度环输出符号，如果纠正方向反了就改这个
PITCH_SIGN = -1        # 使前倾为正（若前倾时 get_pitch() 为负，则设 -1）

# 前馈速度偏置 (rad/s)，正值表示向前，用于补偿倾斜引起的重力矩
FEEDFORWARD_VEL = 0.0

# ---------- PID 参数 ----------
PITCH_KP = 70.0
PITCH_KD = 50
PITCH_KI = 0.0

SPEED_KP = 0.1
SPEED_KI = 0.01
INTEGRAL_LIMIT = 0.5

YAW_TO_WHEEL_DIFF = WHEEL_BASE / (2 * WHEEL_RADIUS)

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
        self.speed_error_integral = 0.0
        self.body_id = model.body('segway').id

    def set_velocity_linear_set_point(self, vel):
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw):
        self.yaw = yaw

    def get_pitch(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('segway_free').qvel[-3:]
        return angular[0]

    def get_wheel_velocity_avg(self) -> float:
        l_vel = self.data.joint('torso_l_wheel').qvel[0]
        r_vel = self.data.joint('torso_r_wheel').qvel[0]
        return (l_vel + r_vel) / 2.0 * WHEEL_DIRECTION

    def calculate_motor_velocity(self) -> float:
        pitch = PITCH_SIGN * self.get_pitch()
        pitch_dot = self.get_pitch_dot()

        self.pitch_dot_filtered = 0.975 * self.pitch_dot_filtered + 0.025 * pitch_dot
        wheel_avg = self.get_wheel_velocity_avg()
        self.velocity_angular_filtered = 0.975 * self.velocity_angular_filtered + 0.025 * wheel_avg

        actual_linear_speed = self.velocity_angular_filtered * WHEEL_RADIUS
        vel_error = actual_linear_speed - self.velocity_linear_set_point
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        target_pitch = SPEED_KP * vel_error + SPEED_KI * self.speed_error_integral

        pitch_error = target_pitch - pitch
        motor_rad_s = PITCH_KP * pitch_error - PITCH_KD * self.pitch_dot_filtered

        motor_rad_s *= BALANCE_SIGN
        motor_vel = motor_rad_s / WHEEL_RADIUS
        motor_vel += FEEDFORWARD_VEL
        return motor_vel

    def update_motor_speed(self):
        vel = self.calculate_motor_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        left_vel = vel * WHEEL_DIRECTION
        right_vel = vel * WHEEL_DIRECTION

        yaw_diff = self.yaw * YAW_TO_WHEEL_DIFF
        left_vel -= yaw_diff
        right_vel += yaw_diff

        left_vel = clamp(left_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        right_vel = clamp(right_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        self.data.actuator('motor_l_wheel').ctrl = [left_vel]
        self.data.actuator('motor_r_wheel').ctrl = [right_vel]

    def reset(self):
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.speed_error_integral = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]