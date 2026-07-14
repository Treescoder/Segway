# 加权PID，平衡角更稳定，来回振荡更小，但是抗扰能力弱
# 仿真代码： simulate_segway_ball_and_human.py
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.1275
MAX_TORQUE = 800.0            # 保留足够的上限

# ---------- 角度环（绕 x 轴俯仰） ----------
PITCH_KP = 120.0              # 较低比例增益，避免振荡
PITCH_KD = 60.0               # 较高阻尼，抑制超调
PITCH_KI = 60.0               # 温和积分
PITCH_INT_LIMIT = 60.0        # 积分限幅

# ---------- 速度跟踪 ----------
SPEED_KP = 1.0
SPEED_KI = 0.1
SPEED_INT_LIMIT = 10.0

# ---------- 静态前馈 ----------
FEEDFORWARD_TORQUE = -20.0

# ---------- 偏航控制 ----------
YAW_DEADZONE = 0.01
YAW_GAIN = 1.0

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class SegwayPID:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0

        self.pitch_integral = 0.0
        self.speed_error_integral = 0.0
        self.filtered_wheel_vel = 0.0

        self.body_id = model.body('segway').id
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

    def get_pitch(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('segway_free').qvel[-3:]
        return angular[0]

    def get_roll(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[1]

    def get_wheel_velocity_avg(self) -> float:
        return (self.data.qvel[self.l_dof] + self.data.qvel[self.r_dof]) / 2.0

    def update_motor_torque(self):
        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot()
        roll = self.get_roll()
        wheel_vel = self.get_wheel_velocity_avg()
        self.last_pitch = pitch
        self.last_roll = roll

        # 低通滤波
        self.filtered_wheel_vel = 0.9 * self.filtered_wheel_vel + 0.1 * wheel_vel

        # ---- 角度环（目标竖直 0°） ----
        pitch_error = 0.0 - pitch
        self.pitch_integral += pitch_error * 0.005
        self.pitch_integral = clamp(self.pitch_integral, -PITCH_INT_LIMIT, PITCH_INT_LIMIT)
        torque_balance = PITCH_KP * pitch_error - PITCH_KD * pitch_dot + PITCH_KI * self.pitch_integral

        # ---- 速度环 ----
        vel_actual = self.filtered_wheel_vel * WHEEL_RADIUS
        vel_error = vel_actual - self.velocity_linear_set_point
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -SPEED_INT_LIMIT, SPEED_INT_LIMIT)
        speed_correction = (SPEED_KP * vel_error + SPEED_KI * self.speed_error_integral) * 15.0

        # 总力矩 = 反馈+ 前馈偏置
        total_torque = torque_balance + speed_correction + FEEDFORWARD_TORQUE

        # ---- 偏航差速 ----
        wheel_base = 0.25 * 2
        yaw_diff = self.yaw * wheel_base / (2 * WHEEL_RADIUS) * 5.0

        left_torque = clamp(total_torque - yaw_diff, -MAX_TORQUE, MAX_TORQUE)
        right_torque = clamp(total_torque + yaw_diff, -MAX_TORQUE, MAX_TORQUE)

        self.data.actuator('motor_l_wheel').ctrl[0] = left_torque
        self.data.actuator('motor_r_wheel').ctrl[0] = right_torque


    def reset(self):
        self.pitch_integral = 0.0
        self.speed_error_integral = 0.0
        self.filtered_wheel_vel = 0.0
        init_pitch = 0.0
        qx = np.sin(init_pitch / 2.0)
        qw = np.cos(init_pitch / 2.0)
        self.data.qpos[3:7] = [qw, qx, 0.0, 0.0]
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]