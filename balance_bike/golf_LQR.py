import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_continuous_are

# ================== 物理参数 ==================
m_wheel = 0.035
r_wheel = 0.0672 / 2
I_wheel = 0.5 * m_wheel * r_wheel ** 2
M_body = 0.757 - 2 * m_wheel
l_cm = 0.5 * 0.0903
J_body = 0.0010866
g = 9.81

Q_eq = J_body * M_body + (J_body + M_body * l_cm ** 2) * (2 * m_wheel + 2 * I_wheel / r_wheel ** 2)
A23 = - (M_body ** 2 * l_cm ** 2 * g) / Q_eq
A43 = M_body * l_cm * g * (M_body + 2 * m_wheel + 2 * I_wheel / r_wheel ** 2) / Q_eq
B21 = (J_body + M_body * l_cm ** 2 + M_body * l_cm * r_wheel) / (Q_eq * r_wheel)
B41 = - (M_body * l_cm / r_wheel + M_body + 2 * m_wheel + 2 * I_wheel / r_wheel ** 2) / Q_eq

A_4 = np.array([
    [0, 1, 0, 0],
    [0, 0, A23, 0],
    [0, 0, 0, 1],
    [0, 0, A43, 0]
])
B_4 = np.array([[0], [B21], [0], [B41]])

Q_lqr_4 = np.diag([1.0, 1.0, 500.0, 1000.0])
R_lqr_4 = np.eye(1) * 0.3

P_4 = solve_continuous_are(A_4, B_4, Q_lqr_4, R_lqr_4)
K_4 = np.linalg.inv(R_lqr_4) @ B_4.T @ P_4
K_4 = K_4.flatten()
print(f"K_4 = {K_4}")

K_4_SCALE = 0.85
K_4_INTEGRAL_SCALE = 0.5

MAX_TORQUE = 20.0

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class GolfLQR:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.pitch = 0.0
        self.pitch_dot = 0.0

        self.target_speed = 0.0
        self.target_yaw_rate = 0.0

        self.speed_error_integral = 0.0
        self.Ki_speed = 2.7 * K_4_INTEGRAL_SCALE

        # 偏航参数（极简直接叠加）
        self.yaw_gain = 0.03                # Nm per rad/s
        self.yaw_torque_filtered = 0.0
        self.yaw_filter_coef = 0.98
        self.max_yaw_torque = 0.2

        self.pitch_dot_filtered = 0.0
        self.wheel_vel_filtered = 0.0

        self.body_id = self.model.body('golf_main').id
        self.joint_id = self.model.joint('golf_free').id
        self.dofadr = self.model.jnt_dofadr[self.joint_id]
        self.l_joint_id = self.model.joint('lunL').id
        self.r_joint_id = self.model.joint('lunR').id
        self.l_dof = self.model.jnt_dofadr[self.l_joint_id]
        self.r_dof = self.model.jnt_dofadr[self.r_joint_id]

        self.K_4_scaled = K_4 * K_4_SCALE # 最终使用的 LQR 增益

    def update_states(self):
        pos = self.data.xpos[self.body_id]
        self.x = pos[0]
        self.x_dot = self.data.qvel[self.dofadr]

        quat = self.data.xquat[self.body_id].copy()
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat = quat / norm
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=False)
        self.pitch = -euler[1]

        cvel = self.data.cvel[self.body_id]
        pitch_dot_raw = -cvel[4]
        self.pitch_dot_filtered = 0.985 * self.pitch_dot_filtered + 0.015 * pitch_dot_raw
        self.pitch_dot = self.pitch_dot_filtered

        l_vel = self.data.qvel[self.l_dof]
        r_vel = self.data.qvel[self.r_dof]
        wheel_vel_raw = (l_vel + r_vel) / 2.0
        self.wheel_vel_filtered = 0.985 * self.wheel_vel_filtered + 0.015 * wheel_vel_raw

    def compute_control(self):
        self.update_states()


        speed_error = self.x_dot - self.target_speed
        self.speed_error_integral += speed_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -1.5, 1.5)

        # LQR的状态误差向量构造
        state_err = np.array([
            0.0,
            speed_error + self.Ki_speed * self.speed_error_integral, #速度误差：速度+积分
            self.pitch,
            self.pitch_dot
        ])

        u_long = -np.dot(self.K_4_scaled, state_err) #u=-K*状态误差
        u_long = clamp(u_long, -MAX_TORQUE, MAX_TORQUE)

        # ---------- 偏航控制（无条件直接叠加） ----------
        u_yaw_raw = self.yaw_gain * self.target_yaw_rate
        self.yaw_torque_filtered = (self.yaw_filter_coef * self.yaw_torque_filtered +
                                    (1 - self.yaw_filter_coef) * u_yaw_raw)
        u_yaw = clamp(self.yaw_torque_filtered, -self.max_yaw_torque, self.max_yaw_torque)

        left_torque = u_long - u_yaw
        right_torque = u_long + u_yaw

        left_torque = clamp(left_torque, -MAX_TORQUE, MAX_TORQUE)
        right_torque = clamp(right_torque, -MAX_TORQUE, MAX_TORQUE)

        return left_torque, right_torque

    def update_motor_torque(self):
        left, right = self.compute_control()
        self.data.actuator('left_motor').ctrl[0] = left
        self.data.actuator('right_motor').ctrl[0] = right

    def update_motor_speed(self):
        self.update_motor_torque()

    def set_velocity_linear_set_point(self, vel):
        self.target_speed = vel

    def set_yaw(self, yaw_rate):
        self.target_yaw_rate = yaw_rate

    def reset(self):
        self.target_speed = 0.0
        self.target_yaw_rate = 0.0
        self.x = 0.0
        self.x_dot = 0.0
        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.speed_error_integral = 0.0
        self.pitch_dot_filtered = 0.0
        self.wheel_vel_filtered = 0.0
        self.yaw_torque_filtered = 0.0