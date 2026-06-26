import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.0672 / 2          # 0.0336 m
MAX_MOTOR_VEL = 200.0

PITCH_KP = 5.0
PITCH_KD = 0.7
SPEED_KP = 0.2
SPEED_KI = 0.01
INTEGRAL_LIMIT = 0.1

INVERT_PITCH = False               # 如果前倾时 pitch 为负，改为 True
INVERT_MOTOR = False               # 如果电机方向反了，改为 True

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class GolfPID:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.speed_error_integral = 0.0

        self.body_id = model.body('golf_main').id
        self.free_dofadr = model.jnt_dofadr[model.joint('golf_free').id]
        self.l_dof = model.jnt_dofadr[model.joint('lunL').id]
        self.r_dof = model.jnt_dofadr[model.joint('lunR').id]

        self.step_cnt = 0
        self.last_pitch = 0.0       # 用于调试

    def set_velocity_linear_set_point(self, vel):
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw):
        self.yaw = yaw

    # ---------- 俯仰角 ----------
    def get_pitch(self) -> float:
        R = self.data.xmat[self.body_id].reshape(3, 3, order='F')
        forward = -R[:, 0]   # 你模型的真实车头方向（初始指向世界 -x）
        pitch = np.arctan2(-forward[2], np.sqrt(forward[0]**2 + forward[1]**2))
        return -pitch if INVERT_PITCH else pitch   # 前倾为正

    def get_pitch_dot(self) -> float:
        R = self.data.xmat[self.body_id].reshape(3, 3, order='F')
        omega_world = self.data.qvel[self.free_dofadr+3 : self.free_dofadr+6]
        omega_body = R.T @ omega_world
        return -omega_body[1] if INVERT_PITCH else omega_body[1]

    def get_wheel_velocity_avg(self) -> float:
        l = self.data.qvel[self.l_dof]
        r = self.data.qvel[self.r_dof]
        return (l + r) / 2.0

    # ---------- 控制 ----------
    def calculate_motor_velocity(self) -> float:
        self.last_pitch = self.get_pitch()          # 保存供打印
        pitch_dot = self.get_pitch_dot()
        wheel_vel = self.get_wheel_velocity_avg()
        wheel_vel_linear = wheel_vel * WHEEL_RADIUS

        self.pitch_dot_filtered = 0.975 * self.pitch_dot_filtered + 0.025 * pitch_dot
        self.velocity_angular_filtered = 0.975 * self.velocity_angular_filtered + 0.025 * wheel_vel

        vel_error = self.velocity_angular_filtered * WHEEL_RADIUS - self.velocity_linear_set_point
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        target_pitch = SPEED_KP * vel_error + SPEED_KI * self.speed_error_integral

        pitch_error = target_pitch - self.last_pitch
        motor_vel = PITCH_KP * pitch_error - PITCH_KD * self.pitch_dot_filtered
        return motor_vel / WHEEL_RADIUS

    def update_motor_speed(self):
        vel = self.calculate_motor_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        if INVERT_MOTOR:
            vel = -vel

        left_vel = vel
        right_vel = vel

        # 转弯
        wheel_base = 0.08048 * 2
        yaw_offset = self.yaw * wheel_base / (2 * WHEEL_RADIUS)
        left_vel -= yaw_offset
        right_vel += yaw_offset

        left_vel = clamp(left_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        right_vel = clamp(right_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        self.data.actuator('left_motor').ctrl[0] = left_vel
        self.data.actuator('right_motor').ctrl[0] = right_vel

        # 调试打印（每秒一次）
        self.step_cnt += 1
        if self.step_cnt % 200 == 0:
            print(f"pitch={np.degrees(self.last_pitch):6.2f}°, "
                  f"target_pitch={np.degrees(SPEED_KP * (self.velocity_angular_filtered * WHEEL_RADIUS - self.velocity_linear_set_point) + SPEED_KI * self.speed_error_integral):5.2f}°, "
                  f"vel={vel:7.2f}, left={left_vel:.2f}, right={right_vel:.2f}")

    def reset(self):
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.speed_error_integral = 0.0
        # 保持 XML 默认姿态（车头 -x），不手动设置四元数
        self.data.actuator('left_motor').ctrl = [0]
        self.data.actuator('right_motor').ctrl = [0]