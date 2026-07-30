"""
LQR 最优平衡+速度控制器

基于线性化倒立摆模型计算最优增益：
  状态: x = [phi, phi_dot, v_err, integral_v_err]
  控制: u = -K @ x

  K 由 scipy.linalg.solve_continuous_are 在初始化时计算

LQR 的速度控制原理（与直接反馈的本质区别）：
  直接反馈: 超速→制动→车体前倾→平衡PD前向力矩→加速（正反馈）
  LQR: 超速→前向力矩→车体后仰→重力减速→自然收敛
  LQR 利用非最小相位特性，通过重力实现减速，不经过平衡PD的力矩抵消

符号约定：
  u > 0 → 车前进 → 车体后仰（pitch 增大）
  phi = pitch - theta_eq（偏离平衡角）
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_continuous_are
import mujoco

R_WHEEL = 0.24
GRAVITY = 9.81
WHEEL_MASS = 9.0      # 单轮质量（与 XML 一致）
MIN_MUJOCO_MASS = 1e-4

# LQR 权重矩阵
# Q = [phi, phi_dot, v_err, integral]
# 完整耦合模型下 LQR 自动协调"倾斜↔加减速"，无需人工限幅补丁
Q_WEIGHTS = [500.0, 100.0, 300.0, 50.0]
R_WEIGHT = 1.0

# 积分限幅（防止长时间大误差下积分饱和）
INTEGRAL_LIMIT = 1.5

# 速度通道限幅随球包质量在 [MIN, MAX] 内调度。当前几何为球包 y=0.35m，
# 收紧限幅可减少追速度时的俯仰；0/3/9 kg 均验证到 3.2 m/s。
MAX_V_ERR_MIN = 0.40
MAX_V_ERR_MAX = 0.50

# 偏航控制参数
KP_YAW = 100.0
KD_YAW = 12.0
MAX_STEER_TORQUE = 35.0  # 高速人体转弯需要更快的车体偏航响应
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
        self.segway_qpos_addr = model.jnt_qposadr[model.joint('segway_free').id]

        # 球包位置从模型读取（XML 中 golf_bag 是 handlebar 子 body，
        # 球包固定几何：golf_bag@handlebar pos="0 0.35 0.5"。
        # 必须读取真实几何，theta_eq 与 LQR 增益才能与模型自洽；
        # 任何硬编码力臂都会让 LQR 失稳。
        mujoco.mj_forward(self.model, self.data)
        self.bag_pos_in_segway = (self.data.xpos[self.bag_body_id]
                                  - self.data.xpos[self.body_id]).copy()
        self.M_fixed, self.com_fixed = self._compute_fixed_body_com()
        self.bag_mass = model.body_mass[self.bag_body_id]
        self.max_v_err = MAX_V_ERR_MAX
        self._bag_nominal_mass = self.bag_mass
        self._bag_nominal_inertia = model.body_inertia[self.bag_body_id].copy()
        self.update_com_and_theta_eq()

        # 计算 LQR 增益
        self._compute_lqr_gains()

        self.v_target = 0.0
        self.yaw_target = 0.0
        self._prev_yaw = 0.0
        self._yaw_rate = 0.0
        self._v_integral = 0.0
        self._v_filtered = 0.0  # 低通滤波后的车速（抑制打滑尖峰）

    def _compute_lqr_gains(self, verbose=True):
        """从完整耦合的轮式倒立摆线性化模型计算 LQR 增益

        动力学方程（phi > 0 = 车体后仰, u > 0 = 前向驱动力矩）:
          车体转动:   I_p*phi_ddot - m_b*L*v_dot = m_b*g*L*phi + u
          平动+轮惯量: -m_b*L*phi_ddot + M_e*v_dot = u/R

        其中:
          I_p = m_b*L^2       车体绕轮轴转动惯量（点质量近似）
          M_e = m_b + m_w + I_w/R^2   等效平动质量（含轮转动惯量折算）

        联立求解得耦合线性系统 —— 关键在于 A[2][0] = m_b^2*L^2*g/D > 0:
        倾角通过重力直接驱动加速度，LQR 由此知道"倾斜是加减速的手段"，
        会自动协调平衡与速度控制，无需人工干预。
        """
        m_b = self.M_fixed + self.bag_mass  # 车体质量（不含轮，不含 mocap 人体！）
        L = self.L                          # 质心到轮轴的摆长
        R = R_WHEEL
        lw_id = self.model.body('l_wheel').id
        rw_id = self.model.body('r_wheel').id
        m_w = self.model.body_mass[lw_id] + self.model.body_mass[rw_id]
        I_w = (self.model.body_inertia[lw_id][0]
               + self.model.body_inertia[rw_id][0])  # 绕轮轴的转动惯量

        I_p = self.I_p                      # 车体绕轮轴真实转动惯量（含平行轴）
        M_e = m_b + m_w + I_w / R**2        # 等效平动质量
        D = I_p * M_e - (m_b * L)**2        # 耦合行列式

        mgl = m_b * GRAVITY * L
        mL = m_b * L

        # x = [phi, phi_dot, v_err, integral]
        A = np.array([
            [0,              1, 0, 0],
            [M_e * mgl / D,  0, 0, 0],
            [mL * mgl / D,   0, 0, 0],   # 重力耦合: 后仰 → 前加速
            [0,              0, 1, 0]
        ])
        B = np.array([
            [0],
            [(M_e + mL / R) / D],        # 反作用力矩 + 惯性耦合，恒为正
            [(mL + I_p / R) / D],
            [0]
        ])

        Q = np.diag(Q_WEIGHTS)
        R_mat = np.array([[R_WEIGHT]])

        P = solve_continuous_are(A, B, Q, R_mat)
        self.K = np.linalg.inv(R_mat) @ B.T @ P  # 1x4 gain matrix

        if verbose:
            print(f"LQR 增益: K = [{self.K[0,0]:.1f}, {self.K[0,1]:.1f}, "
                  f"{self.K[0,2]:.1f}, {self.K[0,3]:.1f}]")
            print(f"  u = -K@x = {-self.K[0,0]:.1f}*phi + {-self.K[0,1]:.1f}*phi_dot "
                  f"+ {-self.K[0,2]:.1f}*v_err + {-self.K[0,3]:.1f}*integ")
            print(f"  系统参数: m_b={m_b:.1f}kg, L={L:.3f}m, I_p={I_p:.2f}, "
                  f"M_e={M_e:.1f}kg, D={D:.1f}")

    def _segway_subtree_ids(self):
        """segway 子树内所有 body id（不含 world 及子树外的 body，如 mocap 人体）"""
        ids = []
        for i in range(self.model.nbody):
            j = i
            while j != 0:
                if j == self.body_id:
                    ids.append(i)
                    break
                j = self.model.body_parentid[j]
        return ids

    def _compute_fixed_body_com(self):
        """计算固定车体（segway 子树中除轮子和球包外）的质量、质心和转动惯量

        关键修正：
        1. 只统计 segway 子树 —— 排除 mocap 人体（否则 80kg 人被误算进车体！）
        2. 质心转换为轮轴相对坐标（segway 原点在轮轴处），而非世界坐标
        3. 同时累计绕轮轴 x 轴的真实转动惯量（平行轴定理），
           替代严重低估的点质量近似
        """
        lw_id = self.model.body('l_wheel').id
        rw_id = self.model.body('r_wheel').id
        exclude = {lw_id, rw_id, self.bag_body_id}

        mujoco.mj_forward(self.model, self.data)
        axle_origin = self.data.xpos[self.body_id].copy()  # segway 原点 = 轮轴

        total_m = 0.0
        com_acc = np.zeros(3)
        I_fixed = 0.0
        for i in self._segway_subtree_ids():
            if i in exclude:
                continue
            m = self.model.body_mass[i]
            if m <= 0:
                continue
            rel = self.data.xipos[i] - axle_origin  # 该 body 质心相对轮轴
            total_m += m
            com_acc += m * rel
            # 绕轮轴 x 轴惯量: 自身 I_xx + 平行轴项 m*(dy^2+dz^2)
            I_fixed += self.model.body_inertia[i][0] + m * (rel[1]**2 + rel[2]**2)

        com_fixed = com_acc / total_m if total_m > 0 else np.zeros(3)
        self.I_fixed = I_fixed
        return total_m, com_fixed

    def update_com_and_theta_eq(self):
        M_bag = self.bag_mass
        M_body = self.M_fixed + M_bag
        if M_body > 0:
            com_body = (self.M_fixed * self.com_fixed + M_bag * self.bag_pos_in_segway) / M_body
        else:
            com_body = self.com_fixed
        self.y_off = com_body[1]
        # 摆长 = 质心到轮轴的距离（非质心高度！）
        self.L = np.sqrt(com_body[1]**2 + com_body[2]**2)
        self.theta_eq = np.arctan2(self.y_off, com_body[2])
        # 车体绕轮轴总转动惯量 = 固定部分 + 球包（平行轴）
        bp = self.bag_pos_in_segway
        self.I_p = self.I_fixed + M_bag * (bp[1]**2 + bp[2]**2)
        # 随球包质量调度速度通道限幅：0/3/9kg -> 0.40/0.50/0.50m/s。
        self.max_v_err = clamp(
            MAX_V_ERR_MIN + M_bag / 22.5,
            MAX_V_ERR_MIN,
            MAX_V_ERR_MAX)

    def update_bag_mass(self, new_mass):
        new_mass = float(np.clip(new_mass, 0.0, 9.0))
        is_preset = any(abs(new_mass - preset) < 1e-9 for preset in (0.0, 3.0, 9.0))
        if not is_preset and abs(new_mass - self.bag_mass) < 0.05:
            return
        self.bag_mass = new_mass
        # 未挂包时用极小正质量满足 MuJoCo 数值约束；LQR 仍按 0 kg 建模。
        physics_mass = max(new_mass, MIN_MUJOCO_MASS)
        inertia_scale = physics_mass / self._bag_nominal_mass
        self.model.body_mass[self.bag_body_id] = physics_mass
        self.model.body_inertia[self.bag_body_id] = (
            self._bag_nominal_inertia * inertia_scale)
        mujoco.mj_forward(self.model, self.data)
        self.update_com_and_theta_eq()
        self._compute_lqr_gains(verbose=False)
        self._v_integral = 0.0  # 重置速度积分，避免增益重算时残留导致猛冲

    def set_velocity(self, v):
        self.v_target = -v

    def set_yaw(self, yaw):
        self.yaw_target = yaw

    def update(self, dt=0.005):
        pitch = self._get_pitch()
        pitch_dot = self._get_pitch_dot()
        phi_dot = self._get_phi_dot()
        raw_speed = phi_dot * R_WHEEL
        # 一阶低通（tau≈50ms）：抑制转向打滑时轮速虚高对 LQR 的污染
        alpha = dt / (dt + 0.05)
        self._v_filtered += alpha * (raw_speed - self._v_filtered)
        cur_speed = self._v_filtered

        # 偏航角速度
        cur_yaw = self.get_yaw()
        yaw_diff = np.arctan2(np.sin(cur_yaw - self._prev_yaw),
                              np.cos(cur_yaw - self._prev_yaw))
        self._yaw_rate = yaw_diff / dt if dt > 0 else 0.0
        self._prev_yaw = cur_yaw

        # 状态向量
        phi = pitch - self.theta_eq
        v_err = cur_speed - self.v_target
        # 宽松安全限幅（正常工况不触发）
        v_err_clamped = clamp(v_err, -self.max_v_err, self.max_v_err)

        # 积分（带抗饱和）— 使用原始 v_err 积分，保证稳态精度
        # 预览包含积分项，防止总力矩饱和时积分继续增长
        u_preview = -(self.K[0,0]*phi + self.K[0,1]*pitch_dot
                      + self.K[0,2]*v_err_clamped + self.K[0,3]*self._v_integral)
        if abs(u_preview) < MAX_TORQUE * 0.9:
            self._v_integral += v_err * dt
            self._v_integral = clamp(self._v_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)

        # LQR 控制: u = -K @ x（速度项使用自适应限幅后的 v_err）
        u = -(self.K[0,0]*phi + self.K[0,1]*pitch_dot
              + self.K[0,2]*v_err_clamped + self.K[0,3]*self._v_integral)
        u = clamp(u, -MAX_TORQUE, MAX_TORQUE)
        self._last_u = u

        # 偏航控制
        steer = self._compute_steer_torque()
        steer = clamp(steer, -MAX_STEER_TORQUE, MAX_STEER_TORQUE)

        left = (u - steer) / 2
        right = (u + steer) / 2

        self.data.actuator('motor_l_wheel').ctrl = [left]
        self.data.actuator('motor_r_wheel').ctrl = [right]

    def reset(self):
        self.v_target = 0.0
        self.yaw_target = 0.0
        self._v_integral = 0.0
        self._v_filtered = 0.0
        mujoco.mj_resetData(self.model, self.data)
        half = self.theta_eq / 2.0
        self.data.qpos[self.segway_qpos_addr + 3] = np.cos(half)
        self.data.qpos[self.segway_qpos_addr + 4] = np.sin(half)
        mujoco.mj_forward(self.model, self.data)
        self._prev_yaw = self.get_yaw()
        self._yaw_rate = 0.0

    # ==================== 姿态提取 ====================

    def _get_rotation(self):
        quat = self.data.xquat[self.body_id]
        if np.linalg.norm(quat) < 1e-9:
            return None
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])

    def _get_pitch(self):
        rot = self._get_rotation()
        if rot is None:
            return 0.0
        fwd = rot.apply(np.array([0.0, 1.0, 0.0]))
        horizontal = np.sqrt(fwd[0]**2 + fwd[1]**2)
        return np.arctan2(fwd[2], horizontal)

    def _get_pitch_dot(self):
        return self.data.joint('segway_free').qvel[3]

    def get_yaw(self):
        rot = self._get_rotation()
        if rot is None:
            return 0.0
        body_plus_y = rot.apply(np.array([0.0, 1.0, 0.0]))
        return np.arctan2(-body_plus_y[0], body_plus_y[1])

    def get_roll(self):
        rot = self._get_rotation()
        if rot is None:
            return 0.0
        right = rot.apply(np.array([1.0, 0.0, 0.0]))
        horizontal = np.sqrt(right[0]**2 + right[1]**2)
        return np.arctan2(right[2], horizontal)

    def get_position(self):
        return self.data.xpos[self.body_id].copy()

    def get_speed(self):
        return -self._get_phi_dot() * R_WHEEL

    def _get_phi_dot(self):
        lv = self.data.qvel[self.l_wheel_dof]
        rv = self.data.qvel[self.r_wheel_dof]
        return (lv + rv) / 2.0

    def _compute_steer_torque(self):
        yaw = self.get_yaw()
        yaw_err = np.arctan2(np.sin(yaw - self.yaw_target),
                             np.cos(yaw - self.yaw_target))
        # get_yaw 的正方向按摄像头前向定义，与原轮差速符号相反。
        steer = -KP_YAW * yaw_err - KD_YAW * self._yaw_rate
        return clamp(steer, -MAX_STEER_TORQUE, MAX_STEER_TORQUE)
