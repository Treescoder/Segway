# 球包挂在里面（并联控制，修正速度误差符号）
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.24
MAX_TORQUE = 800.0

# 平衡环
PITCH_KP = 150.0
PITCH_KD = 170.0
PITCH_KI = 6.0
PITCH_INT_LIMIT = 10.0

# 速度环
SPEED_KP = 6.0
SPEED_KI = 0.9
SPEED_INT_LIMIT = 20.0

FEEDFORWARD_TORQUE = 0.0

# 转向
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
        return rot.as_euler('xyz', degrees=False)[0]   # 正=前倾

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
        wheel_vel = self.get_wheel_velocity_avg()

        self.filtered_wheel_vel = 0.9 * self.filtered_wheel_vel + 0.1 * wheel_vel

        # 平衡环（保持原始正确极性）
        pitch_error = np.deg2rad(0) - pitch
        self.pitch_integral += pitch_error * 0.005
        self.pitch_integral = clamp(self.pitch_integral, -PITCH_INT_LIMIT, PITCH_INT_LIMIT)
        torque_balance = PITCH_KP * pitch_error - PITCH_KD * pitch_dot + PITCH_KI * self.pitch_integral

        # 速度环（修正符号：目标 - 实际）
        actual_speed = self.filtered_wheel_vel * WHEEL_RADIUS
        vel_error = self.velocity_linear_set_point - actual_speed
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -SPEED_INT_LIMIT, SPEED_INT_LIMIT)
        speed_correction = SPEED_KP * vel_error + SPEED_KI * self.speed_error_integral

        total_torque = torque_balance + speed_correction + FEEDFORWARD_TORQUE

        # 转弯
        wheel_base = 0.25 * 2   # 0.5 m
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