import numpy as np
from scipy.interpolate import CubicSpline

# ===================== Composite Closed Trajectory =====================
class ClosedCompositeFollower:
    """
    沿设计好的闭合曲折轨迹跟踪（与示例平滑轨迹一致）
    """
    def __init__(self):
        # -----------------------------
        # 平滑关键点设计（首尾闭合）
        # -----------------------------
        self.key_points = np.array([
            [0, 0],        # 起点
            [0, 40],       # 圆弧上方
            [50, 50],      # 直线段过渡
            [90, 70],      # S 型开始
            [130, 40],     # S 型折返
            [180, 50],     # 第二条直线
            [220, 20],     # 下方圆弧
            [150, -20],    # 返回折返段
            [70, -10],     # S 型回到起点附近
            [0, 0]         # 闭合首尾
        ], dtype=float)
        self.name = "Composite"

        # -----------------------------
        # 累积距离
        # -----------------------------
        self.dist = np.zeros(len(self.key_points))
        for i in range(1, len(self.key_points)):
            self.dist[i] = self.dist[i-1] + np.linalg.norm(self.key_points[i]-self.key_points[i-1])

        # -----------------------------
        # 周期样条生成平滑轨迹
        # -----------------------------
        self.cs_x = CubicSpline(self.dist, self.key_points[:,0], bc_type='periodic')
        self.cs_y = CubicSpline(self.dist, self.key_points[:,1], bc_type='periodic')

        self.total_length = self.dist[-1]

        # 跟踪状态
        self.prev_pos = None
        self.curr_s = 0.0  # 当前沿轨迹的累计长度

        # 控制参数
        self.k_heading = 0.6
        self.k_lat = -0.05
        self.k_lat_i = -0.002
        self.lat_integral = 0.0

    def update(self, pos, heading, dt):
        # -----------------------------
        # 计算沿轨迹当前位置最近点（参数s）
        # -----------------------------
        if self.prev_pos is None:
            dx = 0.0
        else:
            dx = np.linalg.norm(pos - self.prev_pos)
        self.prev_pos = pos.copy()
        t_dense = np.linspace(0, self.total_length, 2000)
        traj_x_dense = self.cs_x(t_dense)
        traj_y_dense = self.cs_y(t_dense)
        diff = np.sqrt((pos[0] - traj_x_dense) ** 2 + (pos[1] - traj_y_dense) ** 2)
        s_mod = t_dense[np.argmin(diff)]

        # 轨迹位置与导数
        traj_x = self.cs_x(s_mod)
        traj_y = self.cs_y(s_mod)
        traj_dx = self.cs_x.derivative()(s_mod)
        traj_dy = self.cs_y.derivative()(s_mod)

        # 横向误差
        lat_error = (pos[0] - traj_x) * (-traj_dy) + (pos[1] - traj_y) * traj_dx
        lat_error /= np.sqrt(traj_dx**2 + traj_dy**2)
        self.lat_integral += lat_error * dt

        # 期望航向
        desired_heading = np.arctan2(traj_dy, traj_dx)
        heading_error = desired_heading - heading
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi
        # 控制量
        delta_des = self.k_heading * heading_error + self.k_lat * lat_error + self.k_lat_i * self.lat_integral
        return np.clip(delta_des, -np.radians(45), np.radians(45))

    def plot_desired(self, ax, traj_x=None):
        t_dense = np.linspace(0, self.total_length, 2000)
        traj_x = self.cs_x(t_dense)
        traj_y = self.cs_y(t_dense)
        # 关键点蓝色
        kp_line, = ax.plot(self.key_points[:,0], self.key_points[:,1], 'b.', markersize=10, label='Key Points')
        # 平滑期望轨迹红色
        traj_line, = ax.plot(traj_x, traj_y, 'r-', linewidth=3, label=f"Desired {self.name}")
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'{self.name} Trajectory (~{self.total_length:.1f} m)')
        return [kp_line, traj_line]