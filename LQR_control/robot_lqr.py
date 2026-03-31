import mujoco

# obtained from running `calculate_lqr_gains.py`
LQR_K = [-2.1402165848237837, -0.03501370844016172, 5.9748026764525894e-18, 2.236067977499789]

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class RobotLqr:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

    def set_velocity_linear_set_point(self, vel: float) -> None:
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw: float) -> None:
        self.yaw = yaw

    def calculate_lqr_velocity(self, pitch: float, pitch_dot: float, wheel_velocity: float, wheel_radius: float) -> float:
        """
        pitch: 当前横滚角 (rad)
        pitch_dot: 横滚角速度 (rad/s)
        wheel_velocity: 轮子平均速度 (rad/s)
        wheel_radius: 轮半径 (m)
        """
        # apply filters
        self.pitch_dot_filtered = (self.pitch_dot_filtered * 0.975) + (pitch_dot * 0.025)
        self.velocity_angular_filtered = (self.velocity_angular_filtered * 0.975) + (wheel_velocity * 0.025)

        velocity_linear_error = self.velocity_linear_set_point - self.velocity_angular_filtered * wheel_radius

        lqr_v = LQR_K[0] * (-pitch) + LQR_K[1] * self.pitch_dot_filtered + LQR_K[2] * 0 + LQR_K[3] * velocity_linear_error
        return -lqr_v / wheel_radius

    def update_motor_speed(self, vel: float, max_motor_vel: float) -> None:
        vel = clamp(vel, -max_motor_vel, max_motor_vel)
        self.data.actuator('motor_l_wheel').ctrl = [-vel + self.yaw]
        self.data.actuator('motor_r_wheel').ctrl = [vel + self.yaw]

    def reset(self):
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        # 不再处理姿态和电机，主函数处理状态初始化
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]