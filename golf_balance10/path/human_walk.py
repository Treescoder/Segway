import numpy as np

class WalkingHumanTrajectory:
    """
    行人轨迹 + 虚拟跟随点
    高曲率、长弯度、连续动力学
    """

    def __init__(self, dt):
        self.dt = dt

        # =========================
        # 状态
        # =========================
        self.x = 3.0
        self.y = 0.0
        self.psi = 0.0

        self.v = 1.8
        self.omega = 0.0

        # =========================
        # 速度参数
        # =========================
        self.v_min = 1.3
        self.v_max = 2.6
        self.a_max = 0.6

        # =========================
        # 角速度参数（高曲率）
        # =========================
        self.w_max = 0.55
        self.alpha_max = 1.2

        self.w_base_amp = 0.35
        self.w_base_freq = 0.03

        self.w_drift_amp = 0.15
        self.w_drift_freq = 0.008

        # =========================
        # 虚拟目标点
        # =========================
        self.follow_dist = 3.0
        self.lookahead_gain = 0.8

        self.time = 0.0

    def step(self):
        """
        推进一个时间步
        返回 dict，供控制直接使用
        """

        # ---------- 平滑速度 ----------
        v_target = 1.9 + 0.4 * np.sin(0.05 * self.time)
        dv = np.clip(
            v_target - self.v,
            -self.a_max * self.dt,
            self.a_max * self.dt
        )
        self.v += dv
        self.v = np.clip(self.v, self.v_min, self.v_max)

        # ---------- 强曲率角速度 ----------
        w_des = (
            self.w_base_amp * np.sin(2 * np.pi * self.w_base_freq * self.time)
            + self.w_drift_amp * np.sin(
                2 * np.pi * self.w_drift_freq * self.time + 1.3
            )
        )

        dw = np.clip(
            w_des - self.omega,
            -self.alpha_max * self.dt,
            self.alpha_max * self.dt
        )
        self.omega += dw
        self.omega = np.clip(self.omega, -self.w_max, self.w_max)

        # ---------- 状态更新 ----------
        self.psi += self.omega * self.dt
        self.x += self.v * np.cos(self.psi) * self.dt
        self.y += self.v * np.sin(self.psi) * self.dt

        # ---------- 虚拟目标点 ----------
        d_ref = self.follow_dist + self.lookahead_gain * self.v
        ref_x = self.x - d_ref * np.cos(self.psi)
        ref_y = self.y - d_ref * np.sin(self.psi)

        self.time += self.dt

        return {
            "human_pos": np.array([self.x, self.y]),
            "human_heading": self.psi,
            "human_speed": self.v,
            "ref_pos": np.array([ref_x, ref_y])
        }