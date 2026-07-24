"""
LQI 平衡控制器 (内环两状态 + 倾角积分，外环速度 PI)
内环：u = Kθ*(θ_ref - θ) - Kθ̇*θ̇ + Ki*∫(θ_ref - θ)dt
外环：速度 PI → θ_ref = θ_eq + pitch_offset
"""
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

# ==================== 物理参数 ====================
R_WHEEL = 0.24
GRAVITY = 9.81
M_TOTAL = 61.8
M_WHEEL = 9.0
M_BODY = M_TOTAL - M_WHEEL

# ==================== 外环速度 PI ====================
KP_SPEED = 0.25          # 降低，避免过激进
KI_SPEED = 0.4           # 降低，避免积分饱和
SPEED_INTEGRAL_MAX = 5.0
MAX_PITCH_OFFSET = np.deg2rad(15)

# ==================== 转向 ====================
KP_YAW = 10.0
KD_YAW = 0.5
MAX_STEER_TORQUE = 15.0
MAX_TORQUE = 800.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class SegwayLQR:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # 获取质心参数，计算自然平衡倾角（后仰为正）
        self.L, self.y_off = self._compute_body_com()
        self.theta_eq = np.arctan2(self.y_off, self.L)
        print(f"质心高度 L={self.L:.3f}m, 偏移 y={self.y_off:.3f}m, 平衡倾角={np.rad2deg(self.theta_eq):.1f}°")

        # 求解内环 LQR 增益（俯仰角+速度积分）
        self.K_theta, self.K_thetadot, self.K_i = self._compute_lqi_gains()
        print(f"LQI 增益: Kθ={self.K_theta:.1f}, Kθ̇={self.K_thetadot:.1f}, Ki={self.K_i:.3f}")

        self.v_target = 0.0
        self.yaw_target = 0.0
        self.speed_integral = 0.0
        self.pitch_error_integral = 0.0
        self.body_id = model.body('segway').id
        self.l_wheel_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_wheel_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]

    def _compute_body_com(self):
        orig = self.model.body_mass.copy()
        lw = self.model.body('l_wheel').id
        rw = self.model.body('r_wheel').id
        self.model.body_mass[lw] = 0.0
        self.model.body_mass[rw] = 0.0
        mujoco.mj_forward(self.model, self.data)
        com = self.data.subtree_com[self.model.body('segway').id]
        self.model.body_mass[:] = orig
        return com[2], com[1]   # z, y

    def _compute_lqi_gains(self):
        """内环两状态模型 + 倾角积分，求解 LQR 增益"""
        import control
        L, y = self.L, self.y_off
        Mb = M_BODY
        g = GRAVITY
        theta_eq = self.theta_eq
        J_body = Mb * (L**2 + y**2)  # 车身绕轮轴的转动惯量近似
        K_g = Mb * g * (L * np.cos(theta_eq) - y * np.sin(theta_eq)) # 重力刚度系数（在 theta_eq 处线性化）
        # 两状态模型：x = [θ, θ̇]^T
        a21 = K_g / J_body
        b2  = 1.0   # 力矩对 θ̈ 的影响系数（归一化）
        A = np.array([[0, 1],
                      [a21, 0]])
        B = np.array([[0],
                      [b2]])

        # 增广积分状态
        A_aug = np.block([[A, np.zeros((2,1))],
                          [-np.array([[1,0]]), 0]])
        B_aug = np.block([[B],
                          [0]])

        # LQR 权重
        Q = np.diag([150.0, 20.0, 100.0])   # θ, θ̇, ∫e_θ
        R = np.array([[0.01]])

        K_aug, _, _ = control.lqr(A_aug, B_aug, Q, R)
        K_theta = K_aug[0, 0]
        K_thetadot = K_aug[0, 1]
        K_i = K_aug[0, 2]

        # 限幅，防止模型误差导致增益过激
        K_theta = clamp(K_theta, 50.0, 250.0)
        K_thetadot = clamp(K_thetadot, 50.0, 200.0)
        K_i = clamp(K_i, 1.0, 15.0)
        return K_theta, K_thetadot, K_i

    def set_velocity(self, v):   self.v_target = v
    def set_yaw(self, yaw):      self.yaw_target = yaw

    def update(self, dt=0.005):
        pitch     = self._get_pitch()        # 前倾为负，后仰为正
        pitch_dot = self._get_pitch_dot()
        phi_dot   = self._get_phi_dot()
        cur_speed = phi_dot * R_WHEEL

        # ---- 外环速度 PI：产生目标倾角偏移 ----
        v_err = cur_speed  - self.v_target
        self.speed_integral += v_err * dt
        self.speed_integral = clamp(self.speed_integral, -SPEED_INTEGRAL_MAX, SPEED_INTEGRAL_MAX)
        pitch_offset = (KP_SPEED * v_err + KI_SPEED * self.speed_integral)
        pitch_offset = clamp(pitch_offset, -MAX_PITCH_OFFSET, MAX_PITCH_OFFSET)

        pitch_ref = self.theta_eq + 1.1 * pitch_offset # 目标倾角 = 平衡倾角 + 速度偏移

        # ---- 内环 LQI：倾角误差 + 积分 ----
        pitch_error = pitch_ref - pitch
        self.pitch_error_integral += pitch_error * dt
        self.pitch_error_integral = clamp(self.pitch_error_integral, -5.0, 5.0)

        u = (self.K_theta * pitch_error
             - self.K_thetadot * pitch_dot
             + self.K_i * self.pitch_error_integral)

        # ---- 转向差分 ----
        steer = self._compute_steer_torque()
        steer = clamp(steer, -MAX_STEER_TORQUE, MAX_STEER_TORQUE)

        # ---- 力矩分配（除以 2） ----
        u = clamp(u, -MAX_TORQUE, MAX_TORQUE)
        left  = (u + steer) / 2
        right = (u - steer) / 2

        self.data.actuator('motor_l_wheel').ctrl = [left]
        self.data.actuator('motor_r_wheel').ctrl = [right]

    def reset(self):
        self.speed_integral = 0.0
        self.pitch_error_integral = 0.0
        self.v_target = 0.0
        self.yaw_target = 0.0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ---------- 传感器 ----------
    def _get_pitch(self):
        quat = self.data.xquat[self.body_id]
        if np.linalg.norm(quat) < 1e-9:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[0]   # 前倾为负

    def _get_pitch_dot(self):
        return self.data.joint('segway_free').qvel[-3]

    def _get_phi_dot(self):
        lv = self.data.qvel[self.l_wheel_dof]
        rv = self.data.qvel[self.r_wheel_dof]
        return (lv + rv) / 2.0

    def _compute_steer_torque(self):
        yaw = self.data.joint('segway_free').qpos[2]
        yaw_err = np.arctan2(np.sin(self.yaw_target - yaw),
                             np.cos(self.yaw_target - yaw))
        yaw_rate = self.data.joint('segway_free').qvel[2]
        steer = KP_YAW * yaw_err - KD_YAW * yaw_rate
        return clamp(steer, -MAX_STEER_TORQUE, MAX_STEER_TORQUE)