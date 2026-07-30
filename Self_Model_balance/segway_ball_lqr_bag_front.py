"""
LQI 平衡控制器 (内环两状态 + 倾角积分，外环速度 PI)
内环：u = Kθ*(θ_ref - θ) - Kθ̇*θ̇ + Ki*∫(θ_ref - θ)dt
外环：速度 PI → θ_ref = θ_eq + pitch_offset / g
支持球包质量动态调节（安全解析计算质心）
"""
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

R_WHEEL = 0.24
GRAVITY = 9.81
M_WHEEL = 9.0

KP_SPEED = 7.0
KI_SPEED = 2.3
SPEED_INTEGRAL_MAX = 2.7
MAX_ACCEL = 2.0

KP_YAW = 10.0
KD_YAW = 0.5
MAX_STEER_TORQUE = 15.0
MAX_TORQUE = 80.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class SegwayLQR:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.body_id = model.body('segway').id
        self.l_wheel_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_wheel_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]
        self.bag_body_id = model.body('golf_bag').id

        # 球包在父坐标系（handlebar）中的位置（需转换到 segway 坐标系）
        # handlebar 在 segway 下的 pos 为 (0,0,0.02)，球包相对 handlebar 为 (0,0.35,0.5)
        # 所以球包在 segway 坐标系的位置：
        self.bag_pos_in_segway = np.array([0.0, 0.35, 0.5 + 0.02])  # z = 0.52

        # 保存除球包和车轮外的固定车身总质量和质心（相对于 segway 原点）
        self.M_fixed, self.com_fixed = self._compute_fixed_body_com()

        # 初始球包质量（从模型读取）
        self.bag_mass = model.body_mass[self.bag_body_id]

        # 计算当前总质心与平衡倾角
        self.update_com_and_theta_eq()

        # LQR 增益
        self.K_theta, self.K_thetadot, self.K_i = self._compute_lqi_gains()
        print(f"LQI 增益: Kθ={self.K_theta:.1f}, Kθ̇={self.K_thetadot:.1f}, Ki={self.K_i:.3f}")

        self.v_target = 0.0
        self.yaw_target = 0.0
        self.speed_integral = 0.0
        self.pitch_ref_filtered = self.theta_eq
        self.pitch_error_integral = 0.0

    def _compute_fixed_body_com(self):
        """计算除球包和车轮外的固定部件总质量与质心（仅在初始化时调用一次）"""
        orig_mass = self.model.body_mass.copy()
        lw_id = self.model.body('l_wheel').id
        rw_id = self.model.body('r_wheel').id

        # 将球包和车轮质量临时置零
        self.model.body_mass[self.bag_body_id] = 0.0
        self.model.body_mass[lw_id] = 0.0
        self.model.body_mass[rw_id] = 0.0

        mujoco.mj_forward(self.model, self.data)
        com_fixed = self.data.subtree_com[self.model.body('segway').id].copy()

        # 固定部分总质量 = 总质量 - 球包 - 车轮
        total_fixed_mass = np.sum(orig_mass) - orig_mass[self.bag_body_id] - orig_mass[lw_id] - orig_mass[rw_id]

        # 恢复质量
        self.model.body_mass[:] = orig_mass
        return total_fixed_mass, com_fixed

    def update_com_and_theta_eq(self):
        """根据当前球包质量，解析计算整体车身质心与平衡倾角"""
        # 整体质心 = (M_fixed * com_fixed + M_bag * bag_pos) / (M_fixed + M_bag)
        M_bag = self.bag_mass
        M_body = self.M_fixed + M_bag
        if M_body > 0:
            com_body = (self.M_fixed * self.com_fixed + M_bag * self.bag_pos_in_segway) / M_body
        else:
            com_body = self.com_fixed
        self.L = com_body[2]      # z 分量（高度）
        self.y_off = com_body[1]  # y 分量（偏移）
        self.theta_eq = np.arctan2(self.y_off, self.L)
        # 更新滤波器状态
        self.pitch_ref_filtered = self.theta_eq

    def _compute_lqi_gains(self):
        """计算 LQR 增益（使用当前车身质量）"""
        import control
        L, y = self.L, self.y_off
        Mb = self.M_fixed + self.bag_mass   # 车身总质量（不含轮）
        g = GRAVITY
        theta_eq = self.theta_eq
        J_body = Mb * (L**2 + y**2)
        K_g = Mb * g * (L * np.cos(theta_eq) - y * np.sin(theta_eq))
        a21 = K_g / J_body
        b2 = 1.0
        A = np.array([[0, 1],
                      [a21, 0]])
        B = np.array([[0],
                      [b2]])
        A_aug = np.block([[A, np.zeros((2,1))],
                          [-np.array([[1,0]]), 0]])
        B_aug = np.block([[B],
                          [0]])
        Q = np.diag([150.0, 200.0, 30.0])
        R = np.array([[0.05]])
        K_aug, _, _ = control.lqr(A_aug, B_aug, Q, R)
        K_theta = clamp(K_aug[0,0], 50.0, 250.0)
        K_thetadot = clamp(K_aug[0,1], 50.0, 200.0)
        K_i = clamp(K_aug[0,2], 1.0, 15.0)
        return K_theta, K_thetadot, K_i

    def update_bag_mass(self, new_mass):
        """安全更新球包质量，避免仿真过程中调用 mj_forward"""
        if abs(new_mass - self.bag_mass) < 0.01:   # 防抖，避免微小变化频繁触发
            return
        self.bag_mass = new_mass
        self.model.body_mass[self.bag_body_id] = new_mass
        # 解析计算新质心
        self.update_com_and_theta_eq()
        # 重新计算 LQR 增益
        self.K_theta, self.K_thetadot, self.K_i = self._compute_lqi_gains()
        # 重置积分器
        self.speed_integral = 0.0
        self.pitch_error_integral = 0.0
        print(f"球包质量更新为 {new_mass:.2f} kg, 新平衡倾角={np.rad2deg(self.theta_eq):.1f}°")

    def set_velocity(self, v):   self.v_target = v
    def set_yaw(self, yaw):      self.yaw_target = yaw

    def update(self, dt=0.005):
        pitch     = self._get_pitch()
        pitch_dot = self._get_pitch_dot()
        phi_dot   = self._get_phi_dot()
        cur_speed = phi_dot * R_WHEEL

        # ---- 外环速度 PI：产生目标倾角偏移 ----
        v_err = cur_speed  - self.v_target
        self.speed_integral += v_err * dt
        self.speed_integral = clamp(self.speed_integral, -SPEED_INTEGRAL_MAX, SPEED_INTEGRAL_MAX)
        accel_cmd = KP_SPEED * v_err + KI_SPEED * self.speed_integral
        accel_cmd = clamp(accel_cmd, -MAX_ACCEL, MAX_ACCEL)

        pitch_offset = accel_cmd / GRAVITY
        raw_pitch_ref = self.theta_eq + 1.2 * pitch_offset
        alpha = 0.3
        self.pitch_ref_filtered = alpha * raw_pitch_ref + (1 - alpha) * self.pitch_ref_filtered
        pitch_ref = self.pitch_ref_filtered

        pitch_error = pitch_ref - pitch
        self.pitch_error_integral += pitch_error * dt
        self.pitch_error_integral = clamp(self.pitch_error_integral, -2.0, 2.0)

        u = (self.K_theta * pitch_error
             - self.K_thetadot * pitch_dot
             + self.K_i * self.pitch_error_integral)

        steer = self._compute_steer_torque()
        steer = clamp(steer, -MAX_STEER_TORQUE, MAX_STEER_TORQUE)

        u = clamp(u, -MAX_TORQUE, MAX_TORQUE)
        left  = (u + steer) / 2
        right = (u - steer) / 2

        self.data.actuator('motor_l_wheel').ctrl = [left]
        self.data.actuator('motor_r_wheel').ctrl = [right]

    def reset(self):
        self.speed_integral = 0.0
        self.pitch_error_integral = 0.0
        self.pitch_ref_filtered = self.theta_eq
        self.v_target = 0.0
        self.yaw_target = 0.0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _get_pitch(self):
        quat = self.data.xquat[self.body_id]
        if np.linalg.norm(quat) < 1e-9:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[0]

    def _get_pitch_dot(self):
        return self.data.joint('segway_free').qvel[-3]

    def _get_phi_dot(self):
        lv = self.data.qvel[self.l_wheel_dof]
        rv = self.data.qvel[self.r_wheel_dof]
        return (lv + rv) / 2.0

    def _compute_steer_torque(self):
        yaw = self.data.joint('segway_free').qpos[2]
        yaw_err = np.arctan2(np.sin(yaw - self.yaw_target),
                             np.cos(yaw - self.yaw_target))
        yaw_rate = self.data.joint('segway_free').qvel[2]
        steer = KP_YAW * yaw_err - KD_YAW * yaw_rate
        return clamp(steer, -MAX_STEER_TORQUE, MAX_STEER_TORQUE)