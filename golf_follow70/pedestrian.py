# -*- coding: utf-8 -*-
"""
pedestrian.py — 行人系统统一模块
================================
把原自行车项目里分散在主函数中的行人相关内容全部收拢到这里：
  1. WalkingHumanTrajectory : 行人轨迹动力学（原 Human.py，原样迁移）
  2. Pedestrian             : 行人总管家
       - mocap 位置/朝向驱动（human_main）
       - 步态动画（6 个腿关节正弦摆动）
       - 跟随指令生成（距离 PID + 分段速度策略 → speed_cmd / target_heading）
       - 轨迹与距离数据记录 + matplotlib 绘图

主函数用法（每个控制周期，例如 5ms 调一次）：
    ped = Pedestrian(model, data, dt=0.005)
    ...
    ped.update()                                  # 行人前进一步 + 更新模型姿态
    v_cmd, heading, dist = ped.follow_command(car_xy)   # 生成跟随指令
    # heading 是世界系目标航向角；差速车 +y 朝前，故 yaw_cmd = heading - pi/2
"""

import math
import numpy as np


# ============================================================
# 1. 行人轨迹（原 Human.py，未改动逻辑）
# ============================================================
class WalkingHumanTrajectory:
    """
    行人轨迹 + 虚拟跟随点
    高曲率、长弯度、连续动力学
    """

    def __init__(self, dt):
        self.dt = dt

        # ========================= 状态 =========================
        self.x = 3.0
        self.y = 0.0
        self.psi = 0.0

        self.v = 0.0     # 从静止自然起步（现实行人不会瞬间达到步速；也让车能同步起步不被拉开）
        self.omega = 0.0

        # ==================== 速度参数（0~3 m/s 全速域覆盖，2026-07-29） ====================
        # 需求：实际控制中人的速度 0~3 m/s 全覆盖（原 1.0~1.8 漫步区间只测中速段）。
        self.v_min = 0.0
        self.v_max = 3.0
        self.a_max = 0.5

        # ==================== 角速度参数（自然漫步，平缓转向） ====================
        self.w_max = 0.35
        self.alpha_max = 0.6

        self.w_base_amp = 0.16
        self.w_base_freq = 0.012

        self.w_drift_amp = 0.10
        self.w_drift_freq = 0.035

        # ========================= 虚拟目标点 =========================
        self.follow_dist = 3.0
        self.lookahead_gain = 0.8

        self.time = 0.0

    def step(self):
        """推进一个时间步，返回 dict 供控制直接使用"""

        # ---------- 平滑速度（0~3 m/s 全速域缓慢扫掠：走停→漫步→快走→慢跑） ----------
        # 多频正弦叠加：主周期 80s 扫 0→3→0，副频加自然波动；加速度限幅保证平滑。
        v_target = (1.5 + 1.35 * np.sin(2 * np.pi * 0.0125 * self.time)
                    + 0.30 * np.sin(2 * np.pi * 0.027 * self.time + 0.5))
        v_target = float(np.clip(v_target, self.v_min, self.v_max))
        dv = np.clip(v_target - self.v, -self.a_max * self.dt, self.a_max * self.dt)
        self.v += dv
        self.v = np.clip(self.v, self.v_min, self.v_max)

        # ---------- 平缓转向（多频叠加的自然漫步，不再剧烈蛇形） ----------
        w_des = (
            self.w_base_amp * np.sin(2 * np.pi * self.w_base_freq * self.time)
            + self.w_drift_amp * np.sin(2 * np.pi * self.w_drift_freq * self.time + 1.3)
            + 0.06 * np.sin(2 * np.pi * 0.006 * self.time + 0.7)
        )
        dw = np.clip(w_des - self.omega, -self.alpha_max * self.dt, self.alpha_max * self.dt)
        self.omega += dw
        self.omega = np.clip(self.omega, -self.w_max, self.w_max)

        # ---------- 状态更新 ----------
        self.psi += self.omega * self.dt
        self.x += self.v * np.cos(self.psi) * self.dt
        self.y += self.v * np.sin(self.psi) * self.dt

        # ---------- 虚拟目标点（行人正后方） ----------
        d_ref = self.follow_dist + self.lookahead_gain * self.v
        ref_x = self.x - d_ref * np.cos(self.psi)
        ref_y = self.y - d_ref * np.sin(self.psi)

        self.time += self.dt

        return {
            "human_pos": np.array([self.x, self.y]),
            "human_heading": self.psi,
            "human_speed": self.v,
            "ref_pos": np.array([ref_x, ref_y]),
        }


# ============================================================
# 2. 行人总管家：模型驱动 + 步态 + 跟随指令
# ============================================================
class Pedestrian:
    # ---------------- 可调参数（跟随控制） ----------------
    KP = 1.0367       # 距离 PID（2026-07-28 丝滑重训：com=-0.12 固定，120s 验证
    KI = 0.0962       #   rmse=0.127m accel=0.062 pitch_std=0.81° min_dist=2.51m）
    KD = 0.0156
    D_ALPHA = 0.8     # 微分平滑系数
    INT_LIMIT = 5.0   # 积分限幅

    V_MAX = 2.217     # 跟随速度上限 m/s（略高于行人峰值步速即可；过大反而助长追赶急停）
    FF_GAIN = 0.9242  # 行人速度前馈增益（消除稳态滞后与追停振荡）

    # ============ 防撞安全（核心需求：距离永不低于目标的一半） ============
    # 返回的 v_cmd 为「带符号闭环速度」：
    #   > 0  → 朝行人方向接近（正常闭环）
    #   < 0  → 远离行人方向后退（拉开距离，仅安全越界时触发）
    D_SAFE_RATIO = 0.5   # 硬安全线 = 目标距离 × 该比例（默认一半，绝不能低于此）
    APPROACH_A = 1.0285  # 防过冲制动减速度 m/s²（限制远处「超出跟随速度的额外接近速度」，从源头消除起步猛加速→冲过头）
    REVERSE_BRAKE = 2.2  # 越界时强制后退拉开距离的最大速度 m/s²（哪怕牺牲 pitch 稳定也要保安全）

    # ---------------- 步进 / 步态动画参数 ----------------
    STEP_LENGTH = 0.6      # 单步长度 m（步频 = 速度 / (2·步长)，与真人对齐）
    HIP_AMP = 0.45         # 髋关节前后摆幅 rad（≈±26°）
    KNEE_AMP = 0.7         # 膝关节屈曲峰值 rad（摆动相屈曲）
    ANKLE_AMP = 0.22       # 踝关节微调 rad
    ARM_AMP = 0.45         # 手臂前后摆幅 rad（与对侧腿反相，真人协调）
    SHOULDER_BASE = 0.30   # 手臂自然下垂基准角 rad（略向后下垂）
    ELBOW_BEND = 0.30      # 肘部自然微屈基准 rad
    ELBOW_SWING = 0.18     # 摆臂时肘部额外屈伸 rad
    BOB_AMP = 0.025        # 行走竖直起伏 m（2×步频）

    BODY_HEIGHT = 1.30     # mocap 根节点 z 高度

    # 驱动关节：腿 6 + 手臂 4（每个控制步写入 qpos 并清零 qvel，避免能量注入）
    JOINT_NAMES = ["right_hip_y", "right_knee", "right_ankle_x",
                   "left_hip_y", "left_knee", "left_ankle_x",
                   "right_shoulder", "right_elbow",
                   "left_shoulder", "left_elbow"]

    def __init__(self, model, data, dt):
        import mujoco  # 局部引入，避免模块被非 mujoco 环境导入时报错
        self.model = model
        self.data = data
        self.dt = dt

        # ---- mocap 与关节地址 ----
        self.mocap_id = model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_main")]
        self.joint_qpos = {}
        self.joint_dof = {}
        for name in self.JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_qpos[name] = model.jnt_qposadr[jid]
            self.joint_dof[name] = model.jnt_dofadr[jid]

        self.reset()

    # ------------------------------------------------------------
    def reset(self):
        """重置轨迹、步态相位、PID 状态与日志，并立刻摆好行人初始位姿"""
        self.traj = WalkingHumanTrajectory(self.dt)
        self.phase = 0.0

        # PID 状态
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_deriv = 0.0
        self._last_heading = 0.0
        self._last_info = None

        # 日志
        self.log_car = []       # [(x, y)]
        self.log_human = []     # [(x, y)]
        self.log_dist = []
        self.log_d_des = []
        self.log_v_cmd = []
        self.log_v_human = []

        # 初始位姿写入 mocap（不推进时间）
        self._write_mocap(np.array([self.traj.x, self.traj.y]), self.traj.psi)

    # ------------------------------------------------------------
    def update(self):
        """
        行人前进一个控制周期：
        推进轨迹 → 步态相位累加（与速度匹配）→ 更新 mocap 位姿（含起伏）→ 更新肢体关节
        返回轨迹 info dict
        """
        info = self.traj.step()
        self._last_info = info

        # 步频与速度匹配：每秒完整步态周期数 = v / (2·步长)
        # （旧版 STEP_FREQ_GAIN=1.5 使步频快了约 1.5 倍，像小碎步狂奔，已修正）
        v = info["human_speed"]
        stride_freq = v / (2.0 * self.STEP_LENGTH)
        self.phase += 2.0 * np.pi * stride_freq * self.dt

        self._write_mocap(info["human_pos"], info["human_heading"])
        self._animate(self.phase)
        return info

    # ------------------------------------------------------------
    def follow_command(self, car_xy):
        """
        由行人当前状态与车的位置生成跟随指令。
        参数:
            car_xy : (2,) 车在世界系 XY 位置
        返回:
            v_cmd          : 带符号闭环速度 m/s
                             >0 朝行人接近（正常闭环）；<0 远离行人后退（仅安全越界触发）
            target_heading : 世界系目标航向角 rad（指向行人）
            dist           : 当前车-人距离 m
        注意: 差速车模型 +y 为前进方向。主函数按 CAR_FRONT_TOWARD_HUMAN 映射：
            True  : yaw_cmd = heading - pi/2,  speed_send =  v_cmd
            False : yaw_cmd = heading + pi/2,  speed_send = -v_cmd
        """
        if self._last_info is None:
            info = {"human_pos": np.array([self.traj.x, self.traj.y]),
                    "human_speed": self.traj.v}
        else:
            info = self._last_info

        hx, hy = info["human_pos"]
        dx = hx - car_xy[0]
        dy = hy - car_xy[1]
        dist = float(np.hypot(dx, dy))

        # ---- 目标航向：距离过近时保持上一次航向，避免 atan2 抖动 ----
        if dist > 0.3:
            target_heading = float(np.arctan2(dy, dx))
            self._last_heading = target_heading
        else:
            target_heading = self._last_heading

        # ---- 距离 PID → 带符号闭环速度 ----
        # 约定：err = dist - d_des；err>0 太远→要更快接近（v 增大）；
        #       err<0 太近→要减速/后退（v 减小甚至变负）
        d_des = self.traj.follow_dist
        D_SAFE = d_des * self.D_SAFE_RATIO          # 硬安全线（默认一半）
        err = dist - d_des
        self._integral = np.clip(self._integral + err * self.dt,
                                 -self.INT_LIMIT, self.INT_LIMIT)
        deriv = (self.D_ALPHA * self._prev_deriv +
                 (1 - self.D_ALPHA) * (err - self._prev_err) / self.dt)
        self._prev_err = err
        self._prev_deriv = deriv

        v_h = float(info["human_speed"])
        v_pace = self.FF_GAIN * v_h        # 保持队形所需的基础速度（≈人速）
        v_pid = self.KP * err + self.KI * self._integral + self.KD * deriv
        v_cmd = v_pace + v_pid             # 带符号：正=接近，负=远离

        # ---- (1) 防过冲制动曲线（平滑，连续） ----
        # 远处限制「超出 v_pace 的额外接近速度」为 sqrt(2·A·剩余距离)，
        # 保证在剩余距离内总能平滑停下，从源头消除「起步猛加速→冲过头→几乎撞上」。
        # 仅当距离 > 目标时生效；接近目标时该上限自然趋近 0，不拖慢正常队形保持。
        # 整个安全层都是距离的光滑函数，绝不在某点硬性阶跃，避免速度指令跳变
        # 激起平衡环抖动（否则会反过来破坏 pitch 稳定甚至导致失控）。
        if dist > d_des:
            excess = v_cmd - v_pace
            if excess > 0.0:
                cap = math.sqrt(2.0 * self.APPROACH_A * (dist - d_des))
                if excess > cap:
                    excess = cap
            v_cmd = v_pace + excess

        # ---- (2) 近距平滑限速（连续，无阶跃） ----
        # dist 落在 (D_SAFE, d_des) 时限制允许速度，但限速曲线向「人速 v_pace」收敛而
        # 不是直接向 0 收敛：dist=d_des 时为 V_MAX（不干预）→ 中段渐降到人速（跟着走，
        # 距离自然缓慢恢复）→ 贴近安全线才降到 0。
        # 若像旧版直接 V_MAX·factor² 向零压，距离只要低于 ~2.7m 允许速度就低于人速，
        # 车被迫掉队→又追上→再被压速，形成「一走一停」的锯齿极限环并带动 pitch 大幅摆动。
        if d_des > dist > D_SAFE:
            factor = (dist - D_SAFE) / (d_des - D_SAFE)      # 1@目标, 0@安全线
            v_allowed = factor * v_pace + factor * factor * (self.V_MAX - v_pace)
            if v_cmd > v_allowed:
                v_cmd = v_allowed

        # ---- (3) 硬安全约束（平滑→反向，最后兜底） ----
        # 距离低于「目标的一半」立即平滑地强制后退拉开距离（哪怕牺牲 pitch 稳定也要保安全，
        # 绝不允许人车相撞）。factor 在 D_SAFE 处连续衔接 (2)，不会跳变。
        if dist < D_SAFE:
            under = (D_SAFE - dist) / D_SAFE                  # 0..1
            v_open = -self.REVERSE_BRAKE * min(1.0, under * 1.5)
            if v_cmd > v_open:
                v_cmd = v_open

        v_cmd = float(np.clip(v_cmd, -self.REVERSE_BRAKE, self.V_MAX))

        # ---- 日志 ----
        self.log_car.append((float(car_xy[0]), float(car_xy[1])))
        self.log_human.append((float(hx), float(hy)))
        self.log_dist.append(dist)
        self.log_d_des.append(d_des)
        self.log_v_cmd.append(v_cmd)
        self.log_v_human.append(v_h)

        return v_cmd, target_heading, dist

    # ------------------------------------------------------------
    def _write_mocap(self, pos_xy, heading):
        """把行人位置与朝向写入 mocap（含行走竖直起伏）"""
        # 竖直起伏：2×步频，支撑中期（双腿承重）身体最高，摆动中期最低
        bob = -self.BOB_AMP * math.cos(2.0 * self.phase)
        self.data.mocap_pos[self.mocap_id] = np.array(
            [pos_xy[0], pos_xy[1], self.BODY_HEIGHT + bob])
        half = heading / 2.0
        self.data.mocap_quat[self.mocap_id] = np.array(
            [np.cos(half), 0.0, 0.0, math.sin(half)])

    # ------------------------------------------------------------
    def _animate(self, phase):
        """
        协调步态 + 摆臂动画（每控制步调用一次）。
          - 双腿反相（左 = 右 + π），膝关节在摆动相平滑屈曲（钟形，连续可导）
          - 双臂与对侧腿反相摆动（真人协调方式），肘部随摆臂微屈
          - 竖直起伏已在 _write_mocap 中叠加到 mocap 根
        注意：直接改写 qpos 的同时必须把对应 qvel 清零，
        否则积分器会把虚假速度能量注入被动关节链，导致仿真发散。
        """
        qp = self.joint_qpos
        s = np.sin(phase)          # 右腿 / 对侧(左)臂 参考相位
        c = np.cos(phase)

        # ---------- 右腿（hip=-A·s，与左腿反相） ----------
        self.data.qpos[qp["right_hip_y"]] = -self.HIP_AMP * s
        # 膝关节：摆动相（右腿向前摆）屈曲，钟形连续；相位落后 hip 约 π/2
        self.data.qpos[qp["right_knee"]] = -self.KNEE_AMP * (0.5 - 0.5 * np.cos(phase + np.pi / 2))
        self.data.qpos[qp["right_ankle_x"]] = self.ANKLE_AMP * (0.4 * s)

        # ---------- 左腿（hip=+A·s，右腿反相） ----------
        self.data.qpos[qp["left_hip_y"]] = self.HIP_AMP * s
        self.data.qpos[qp["left_knee"]] = -self.KNEE_AMP * (0.5 - 0.5 * np.cos(phase + np.pi / 2 + np.pi))
        self.data.qpos[qp["left_ankle_x"]] = -self.ANKLE_AMP * (0.4 * s)

        # ---------- 摆臂：与同侧腿反相（真人协调：右臂前摆时右腿在后） ----------
        # 右臂 = 基准 + 与右腿(-A·s)反相的摆幅(+A·s)；肘部随摆臂微屈
        self.data.qpos[qp["right_shoulder"]] = self.SHOULDER_BASE + self.ARM_AMP * s
        self.data.qpos[qp["right_elbow"]] = self.ELBOW_BEND + self.ELBOW_SWING * max(0.0, s)
        # 左臂：与右臂反相（= 与左腿反相）
        self.data.qpos[qp["left_shoulder"]] = self.SHOULDER_BASE - self.ARM_AMP * s
        self.data.qpos[qp["left_elbow"]] = self.ELBOW_BEND + self.ELBOW_SWING * max(0.0, -s)

        # ---------- 清零被驱动关节的速度，防止能量注入 ----------
        for dof in self.joint_dof.values():
            self.data.qvel[dof] = 0.0

    # ------------------------------------------------------------
    def plot(self):
        """仿真结束后绘制：轨迹对比 + 距离曲线（原绘图逻辑迁移）"""
        if not self.log_dist:
            print("[Pedestrian] 无记录数据，跳过绘图")
            return
        import matplotlib.pyplot as plt

        car = np.array(self.log_car)
        hum = np.array(self.log_human)

        plt.figure(figsize=(12, 6))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(car[:, 0], car[:, 1], 'g-', linewidth=2, label='Cart Trajectory')
        ax1.plot(hum[:, 0], hum[:, 1], 'b--', linewidth=2, label='Human Trajectory')
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.legend()
        ax1.set_title('Golf Cart vs Human Trajectory')

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(self.log_dist, 'b-', label='Actual Distance')
        ax2.plot(self.log_d_des, 'r--', label='Desired Distance')
        ax2.set_xlabel('Control Step')
        ax2.set_ylabel('Distance [m]')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Cart-Human Distance vs Desired')

        plt.tight_layout()
        plt.show()
