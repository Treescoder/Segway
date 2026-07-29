import numpy as np
from .base import PathFollowerBase

class CircleFollower(PathFollowerBase):
    def __init__(self, cx_offset=0.0, cy_offset=10.0, radius=10.0,
                 k_heading=0.6, k_radius=-0.2, k_radius_i=-0.01, init_pos=None):
        if init_pos is None:
            raise ValueError("init_pos must be provided")
        self.cx = init_pos[0] + cx_offset
        self.cy = init_pos[1] + cy_offset
        self.R = radius
        self.k_heading = k_heading
        self.k_radius = k_radius
        self.k_radius_i = k_radius_i
        self.radius_integral = 0.0

    def update(self, pos, heading, dt):
        dx = pos[0] - self.cx
        dy = pos[1] - self.cy
        distance = np.sqrt(dx**2 + dy**2)
        radius_error = distance - self.R
        self.radius_integral += radius_error * dt

        desired_heading = np.arctan2(dx, -dy)
        heading_error = desired_heading - heading
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi

        delta_des = self.k_heading * heading_error - self.k_radius * radius_error - self.k_radius_i * self.radius_integral
        return np.clip(delta_des, -np.radians(135), np.radians(135))

    def plot_desired(self, ax, traj_x=None):
        theta = np.linspace(0, 2*np.pi, 200)
        x = self.cx + self.R * np.sin(theta)
        y = self.cy - self.R * np.cos(theta)
        line1, = ax.plot(x, y, 'r--', linewidth=2, label='Desired Circle')
        return [line1]