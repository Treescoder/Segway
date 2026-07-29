import numpy as np
from .base import PathFollowerBase

class LineFollower(PathFollowerBase):
    def __init__(self, y_ref, k_heading=0.6, k_cross=-0.15):
        self.y_ref = y_ref
        self.k_heading = k_heading
        self.k_cross = k_cross

    def update(self, pos, heading, dt):
        cross_error = pos[1] - self.y_ref
        desired_heading = 0.0
        heading_error = desired_heading - heading
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi
        delta_des = self.k_heading * heading_error + self.k_cross * cross_error
        return np.clip(delta_des, -np.radians(30), np.radians(30))

    def plot_desired(self, ax, traj_x=None):
        if traj_x is None:
            traj_x = np.linspace(0, 100, 2)
        y_vals = np.ones_like(traj_x) * self.y_ref