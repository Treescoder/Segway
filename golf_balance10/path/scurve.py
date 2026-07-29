import numpy as np
from .base import PathFollowerBase

class SCurveFollower(PathFollowerBase):
    def __init__(self, x0, y0, length=60.0, amplitude=15.0,
                 k_heading=0.6, k_lat=-0.04, k_lat_i=-0.003):
        self.x0 = x0
        self.y0 = y0
        self.length = length
        self.amplitude = amplitude
        self.k_heading = k_heading
        self.k_lat = k_lat
        self.k_lat_i = k_lat_i
        self.lat_integral = 0.0
        self.name = "S-Curve"

    def update(self, pos, heading, dt):
        t = pos[0] - self.x0
        y_d = self.y0 + self.amplitude * np.sin(2*np.pi*t/self.length)
        lat_error = pos[1] - y_d
        self.lat_integral += lat_error * dt

        desired_heading = np.arctan2(self.amplitude * (2*np.pi/self.length) * np.cos(2*np.pi*t/self.length), 1.0)
        heading_error = desired_heading - heading
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi
        delta_des = self.k_heading * heading_error + self.k_lat * lat_error + self.k_lat_i * self.lat_integral
        return np.clip(delta_des, -np.radians(45), np.radians(45))

    def plot_desired(self, ax, traj_x=None):
        if traj_x is None:
            traj_x = np.linspace(0, 100, 300)
        x_vals = self.x0 + traj_x
        y_vals = self.y0 + self.amplitude * np.sin(2 * np.pi * traj_x / self.length)
        line1, = ax.plot(x_vals, y_vals, 'r-', linewidth=3, label=f"Desired {self.name}")
        return [line1]