import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.1275
MAX_TORQUE = 100.0

# 俯仰 LQR 增益（需根据实际微调）
K_PITCH = -5.0
K_PITCH_DOT = -1.0
K_VEL_ERR = 2.0
KI_SPEED = 0.3

# 侧倾速度阻尼系数
ROLL_DAMPING = 80.0        # Nm / (rad/s)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class RobotLqr:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.target_speed = 0.0
        self.target_yaw_rate = 0.0
        self.speed_error_integral = 0.0
        self.pitch_dot_filtered = 0.0
        self.wheel_vel_filtered = 0.0
        self.body_id = model.body('robot_body').id
        self.l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]
        # 自由关节的 dof 地址（0-2平移，3-5转动：roll,pitch,yaw）
        self.roll_dof_adr = model.jnt_dofadr[model.joint('robot_body_joint').id] + 3

    def set_velocity_linear_set_point(self, vel):
        self.target_speed = vel

    def set_yaw(self, yaw):
        self.target_yaw_rate = yaw

    def get_roll_dot(self):
        """获取侧倾角速度（绕x轴）"""
        cvel = self.data.cvel[self.body_id]
        return cvel[3]          # 机体坐标系下的 x 轴角速度

    def get_orientation(self):
        quat = self.data.xquat[self.body_id].copy()
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat = quat / norm
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=False)
        pitch = -euler[1]   # 前倾为正
        return pitch

    def get_pitch_dot(self):
        cvel = self.data.cvel[self.body_id]
        return -cvel[4]

    def get_wheel_vel(self):
        l_vel = self.data.qvel[self.l_dof]
        r_vel = self.data.qvel[self.r_dof]
        return (-l_vel + r_vel) / 2.0

    def update_states(self):
        pitch = self.get_orientation()
        pitch_dot_raw = self.get_pitch_dot()
        wheel_vel_raw = self.get_wheel_vel()
        self.pitch_dot_filtered = 0.985 * self.pitch_dot_filtered + 0.015 * pitch_dot_raw
        self.wheel_vel_filtered = 0.985 * self.wheel_vel_filtered + 0.015 * wheel_vel_raw
        return pitch, self.pitch_dot_filtered, self.wheel_vel_filtered

    def apply_roll_damping(self):
        roll_vel = self.get_roll_dot()
        damping_torque = -ROLL_DAMPING * roll_vel
        self.data.qfrc_applied[self.roll_dof_adr] = damping_torque

    def compute_control(self):
        pitch, pitch_dot, wheel_vel = self.update_states()
        current_speed = wheel_vel * WHEEL_RADIUS
        vel_error = self.target_speed - current_speed
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -3.0, 3.0)

        u_long = (K_PITCH * (0 - pitch) +
                  K_PITCH_DOT * pitch_dot +
                  K_VEL_ERR * (vel_error + KI_SPEED * self.speed_error_integral))
        u_long = clamp(u_long, -MAX_TORQUE, MAX_TORQUE)

        u_yaw = clamp(self.target_yaw_rate * 0.5, -5.0, 5.0)

        left_torque = u_long - u_yaw
        right_torque = u_long + u_yaw

        return clamp(left_torque, -MAX_TORQUE, MAX_TORQUE), clamp(right_torque, -MAX_TORQUE, MAX_TORQUE)

    def update_motor_torque(self):
        # 1. 施加侧倾阻尼
        self.apply_roll_damping()
        # 2. 更新电机力矩
        left, right = self.compute_control()
        self.data.actuator('motor_l_wheel').ctrl[0] = left
        self.data.actuator('motor_r_wheel').ctrl[0] = right

    def update_motor_speed(self):
        self.update_motor_torque()

    def reset(self):
        self.target_speed = 0.0
        self.target_yaw_rate = 0.0
        self.speed_error_integral = 0.0
        self.pitch_dot_filtered = 0.0
        self.wheel_vel_filtered = 0.0
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.actuator('motor_l_wheel').ctrl[0] = 0
        self.data.actuator('motor_r_wheel').ctrl[0] = 0