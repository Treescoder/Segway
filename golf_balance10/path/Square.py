import numpy as np
from .base import PathFollowerBase


class SquareFollower(PathFollowerBase):
    def __init__(self, side=30.0, k_heading=0.6, k_cross=-0.15):
        self.side = side
        self.k_heading = k_heading
        self.k_cross = k_cross
        self.name = "Square"

        # 正方形四个顶点（逆时针）
        self.points = np.array([
            [0.0, 0.0],
            [side, 0.0],
            [side, side],
            [0.0, side],
            [0.0, 0.0]   # 闭合
        ])

        self.edge_idx = 0  # 当前边索引

    def update(self, pos, heading, dt):
        p0 = self.points[self.edge_idx]
        p1 = self.points[self.edge_idx + 1]

        # 当前边方向
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        t_hat = edge_vec / edge_len

        # 法向（左法向，逆时针）
        n_hat = np.array([-t_hat[1], t_hat[0]])

        # 位置相对向量
        r = pos[:2] - p0

        # 横向误差（signed）
        cross_error = np.dot(r, n_hat)

        # 切向进度
        s = np.dot(r, t_hat)

        # ===== 切边逻辑 =====
        if s > edge_len:
            self.edge_idx = (self.edge_idx + 1) % 4
            return 0.0  # 切边瞬间不打方向

        # ===== 航向控制 =====
        desired_heading = np.arctan2(t_hat[1], t_hat[0])
        heading_error = desired_heading - heading
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi

        delta_des = (
            self.k_heading * heading_error +
            self.k_cross * cross_error
        )

        return np.clip(delta_des, -np.radians(30), np.radians(30))

    def plot_desired(self, ax, traj_x=None):
        xs = self.points[:, 0]
        ys = self.points[:, 1]
        line1, = ax.plot(xs, ys, 'r-', linewidth=3, label=f"Desired {self.name}")
        return [line1]