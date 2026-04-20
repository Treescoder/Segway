import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_continuous_are

# ================== 物理参数 ==================
m_wheel = 0.035
r_wheel = 0.0672 / 2
I_wheel = 0.5 * m_wheel * r_wheel**2
M_body = 0.757 - 2*m_wheel
l_cm = 0.5 * 0.0903
J_body = 0.0010866
d_wheel = 0.1612
g = 9.81

Q_eq = J_body * M_body + (J_body + M_body * l_cm**2) * (2*m_wheel + 2*I_wheel/r_wheel**2)
A23 = - (M_body**2 * l_cm**2 * g) / Q_eq
A43 = M_body * l_cm * g * (M_body + 2*m_wheel + 2*I_wheel/r_wheel**2) / Q_eq
B21 = (J_body + M_body*l_cm**2 + M_body*l_cm*r_wheel) / (Q_eq * r_wheel)
B41 = - (M_body*l_cm/r_wheel + M_body + 2*m_wheel + 2*I_wheel/r_wheel**2) / Q_eq

A_4 = np.array([
    [0, 1, 0, 0],
    [0, 0, A23, 0],
    [0, 0, 0, 1],
    [0, 0, A43, 0]
])
B_4 = np.array([[0], [B21], [0], [B41]])

Q_lqr_4 = np.diag([1.0, 1.0, 300.0, 1000.0])
R_lqr_4 = np.eye(1) * 0.3

P_4 = solve_continuous_are(A_4, B_4, Q_lqr_4, R_lqr_4)
K_4 = np.linalg.inv(R_lqr_4) @ B_4.T @ P_4
K_4 = K_4.flatten()

MAX_TORQUE = 5.0

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class GolfLQR:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.x = 0.0
        self.x_dot = 0.0
        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.target_speed = 0.0

        self.body_id = self.model.body('golf_main').id
        self.joint_id = self.model.joint('golf_free').id
        self.dofadr = self.model.jnt_dofadr[self.joint_id]
        self.l_joint_id = self.model.joint('lunL').id
        self.r_joint_id = self.model.joint('lunR').id

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


        self.pitch = -euler[1]         # 取反以匹配 euler="0 3.14 0" 的坐标系
        cvel = self.data.cvel[self.body_id]
        self.pitch_dot = -cvel[4]

    def compute_control(self):
        self.update_states()
        state_err = np.array([
            self.x,
            self.x_dot - self.target_speed,
            self.pitch,
            self.pitch_dot
        ])
        u = -np.dot(K_4, state_err)
        torque = clamp(u, -MAX_TORQUE, MAX_TORQUE)
        return torque, torque

    def update_motor_torque(self):
        left, right = self.compute_control()
        self.data.actuator('left_motor').ctrl[0] = left
        self.data.actuator('right_motor').ctrl[0] = right

    def update_motor_speed(self):  # 兼容旧版本
        self.update_motor_torque()

    def set_velocity_linear_set_point(self, vel):
        self.target_speed = vel

    def set_yaw(self, yaw_rate):
        pass

    def reset(self):
        self.target_speed = 0.0
        self.x = 0.0
        self.x_dot = 0.0
        self.pitch = 0.0
        self.pitch_dot = 0.0