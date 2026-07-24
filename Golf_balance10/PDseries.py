import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.24
wheel_base = 0.25 * 2
MAX_TORQUE = 50.0

PITCH_KP = 150.0
PITCH_KD = 100.0
PITCH_KI = 0.0
PITCH_INT_LIMIT = 6.0

SPEED_KP = .6
SPEED_KD = .0
SPEED_KI = .0
SPEED_INT_LIMIT = 10.0

FEEDFORWARD_TORQUE = 0

YAW_DEADZONE = 0.01
YAW_GAIN = 1.0
YAW_KP = 200
YAW_KD = 150

DOB_GAIN = 0

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class SegwayPID:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0
        self.balance_pitch = 0.0

        self.pitch_integral = 0.0
        self.speed_error_integral = 0.0
        self.filtered_wheel_vel = 0.0

        self.body_id = model.body('segway').id
        self.l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]

    def set_velocity_linear_set_point(self, vel):
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw):
        # if abs(yaw) < YAW_DEADZONE:
        #     self.yaw = 0.0
        # else:
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

    def get_roll(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[1]

    def get_yaw(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler("xyz", degrees=False)[2]

    def get_wheel_velocity_avg(self) -> float:
        return (self.data.qvel[self.l_dof] + self.data.qvel[self.r_dof]) / 2.0

    def update_motor_torque(self):
        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot()
        roll = self.get_roll()
        wheel_vel = self.get_wheel_velocity_avg()
        current_yaw = self.get_yaw()

        self.filtered_wheel_vel = 0.9 * self.filtered_wheel_vel + 0.1 * wheel_vel

        actual_speed = self.filtered_wheel_vel * WHEEL_RADIUS

        # ============================
        # 更新balance pitch
        # ============================

        if (abs(actual_speed) < 0.5 and
                abs(self.velocity_linear_set_point) < 0.5 and
                abs(pitch_dot) < np.deg2rad(10)):
            ALPHA = 0.005
            self.balance_pitch += ALPHA * (pitch - self.balance_pitch)
        self.balance_pitch = clamp(self.balance_pitch,np.deg2rad(-5),np.deg2rad(30))

        vel_error = actual_speed - self.velocity_linear_set_point
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -SPEED_INT_LIMIT, SPEED_INT_LIMIT)
        speed_correction = SPEED_KP * vel_error + SPEED_KD * vel_error + SPEED_KI * self.speed_error_integral

        pitch_target = speed_correction + self.balance_pitch
        pitch_target = clamp(pitch_target, np.deg2rad(-30), np.deg2rad(30))

        # pitch_error = np.deg2rad(0) - pitch
        pitch_error = pitch_target - pitch
        self.pitch_integral += pitch_error * 0.005
        self.pitch_integral = clamp(self.pitch_integral, -PITCH_INT_LIMIT, PITCH_INT_LIMIT)
        torque_balance = PITCH_KP * pitch_error - PITCH_KD * pitch_dot + PITCH_KI * self.pitch_integral

        total_torque = torque_balance + FEEDFORWARD_TORQUE

        yaw_error = np.arctan2(
            np.sin(self.yaw - current_yaw),
            np.cos(self.yaw - current_yaw)
        )
        # yaw_diff = self.yaw * wheel_base / (2 * WHEEL_RADIUS) * 5.0
        yaw_diff = YAW_KP * yaw_error

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