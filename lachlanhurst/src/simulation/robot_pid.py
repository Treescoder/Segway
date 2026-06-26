import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.034
MAX_MOTOR_VEL = 200.0          # 与 XML ctrlrange 一致

# PID 参数（仅俯仰平衡）
PITCH_KP = 8.0              # 比例增益: 放大会让俯仰角在加减速的时候也保持小角度平衡，但是速度跟踪不上，始终有误差；现在这个值可以很好的跟踪速度，但是转弯转的太急可能会抖
PITCH_KD = 1.1              # 微分阻尼
SPEED_KP = 0.3              # 暂不加入速度控制
SPEED_KI = 0.01
INTEGRAL_LIMIT = 0.1

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class RobotPID:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.speed_error_integral = 0.0

        self.body_id = model.body('robot_body').id
        self.free_dofadr = model.jnt_dofadr[model.joint('robot_body_joint').id]

    def set_velocity_linear_set_point(self, vel):
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw):
        self.yaw = yaw   # 稍后可以用，但先设为0

    # ---------- 俯仰角获取----------
    def get_pitch(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        angles = rotation.as_euler('xyz', degrees=False)
        return angles[0]          # 绕 x 轴角度

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        return angular[0]         # 绕 x 轴角速度

    def get_wheel_velocity_avg(self) -> float:
        """两轮平均角速度（已考虑左轮反转）"""
        l_vel = self.data.joint('torso_l_wheel').qvel[0]
        r_vel = self.data.joint('torso_r_wheel').qvel[0]
        return (l_vel * -1 + r_vel) / 2.0    # 左轮取反

    def calculate_motor_velocity(self) -> float:
        pitch = -self.get_pitch()            # 与原 LQR 一致：前倾时 pitch 为正
        pitch_dot = self.get_pitch_dot()

        # 滤波器
        self.pitch_dot_filtered = 0.975 * self.pitch_dot_filtered + 0.025 * pitch_dot
        self.velocity_angular_filtered = 0.975 * self.velocity_angular_filtered + 0.025 * self.get_wheel_velocity_avg()

        # 速度误差
        vel_error = self.velocity_angular_filtered * WHEEL_RADIUS - self.velocity_linear_set_point
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        target_pitch = SPEED_KP * vel_error + SPEED_KI * self.speed_error_integral

        # 俯仰 PD 控制
        pitch_error = target_pitch - pitch
        motor_vel = PITCH_KP * pitch_error - PITCH_KD * self.pitch_dot_filtered
        return motor_vel / WHEEL_RADIUS      # 转换为轮子角速度

    def update_motor_speed(self):
        vel = self.calculate_motor_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        # 左右电机同向转动（考虑轴反向），无偏航差速
        left_vel = -vel
        right_vel = vel

        # 加上偏航（yaw）用于转向
        left_vel += self.yaw
        right_vel += self.yaw

        left_vel = clamp(left_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        right_vel = clamp(right_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        self.data.actuator('motor_l_wheel').ctrl = [left_vel]
        self.data.actuator('motor_r_wheel').ctrl = [right_vel]

    def reset(self):
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.speed_error_integral = 0.0
        # 完全直立
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]