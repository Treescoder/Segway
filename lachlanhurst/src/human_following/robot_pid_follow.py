import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.034
MAX_MOTOR_VEL = 200.0

# 俯仰平衡 PID
PITCH_KP = 3.0
PITCH_KD = 1.1

# 速度闭环 PI（内环）
SPEED_KP = 0.3
SPEED_KI = 0.01
INTEGRAL_LIMIT = 0.1

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class RobotPID:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.speed_error_integral = 0.0

        self.body_id = model.body('robot_body').id
        self.free_dofadr = model.jnt_dofadr[model.joint('robot_body_joint').id]

        # ===== 跟踪控制参数 =====
        self.desired_follow_dist = 0.3
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0

    def set_velocity_linear_set_point(self, vel):
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw):
        self.yaw = yaw

    def get_pitch(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        angles = rotation.as_euler('xyz', degrees=False)
        return angles[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        return angular[0]

    def get_wheel_velocity_avg(self) -> float:
        l_vel = self.data.joint('torso_l_wheel').qvel[0]
        r_vel = self.data.joint('torso_r_wheel').qvel[0]
        return (-l_vel + r_vel) / 2.0

    def calculate_motor_velocity(self) -> float:
        pitch = -self.get_pitch()
        pitch_dot = self.get_pitch_dot()

        self.pitch_dot_filtered = 0.975 * self.pitch_dot_filtered + 0.025 * pitch_dot
        self.velocity_angular_filtered = 0.975 * self.velocity_angular_filtered + 0.025 * self.get_wheel_velocity_avg()

        vel_error = self.velocity_angular_filtered * WHEEL_RADIUS - self.velocity_linear_set_point
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        target_pitch = SPEED_KP * vel_error + SPEED_KI * self.speed_error_integral

        pitch_error = target_pitch - pitch
        motor_vel = PITCH_KP * pitch_error - PITCH_KD * self.pitch_dot_filtered
        return motor_vel / WHEEL_RADIUS

    def update_motor_speed(self):
        vel = self.calculate_motor_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        left_vel = -vel
        right_vel = vel

        left_vel += self.yaw
        right_vel += self.yaw

        left_vel = clamp(left_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        right_vel = clamp(right_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        self.data.actuator('motor_l_wheel').ctrl = [left_vel]
        self.data.actuator('motor_r_wheel').ctrl = [right_vel]

    # ---------- 行人跟踪控制 ----------
    def compute_tracking_commands(self, human_x, human_y, human_heading, human_speed, human_turn, dt):
        """
        根据行人状态计算车的速度指令和转向指令，并更新内部目标值。
        返回 (v_cmd, yaw_cmd, dist_err, heading_err)
        """
        robot_pos = self.data.xpos[self.body_id]
        robot_mat = self.data.xmat[self.body_id].reshape(3, 3)

        # 机器人前进方向：局部 -y 轴
        forward_global = -robot_mat[:, 1]
        right_global = robot_mat[:, 0]
        robot_heading = math.atan2(forward_global[1], forward_global[0])

        dx = human_x - robot_pos[0]
        dy = human_y - robot_pos[1]
        dist = math.hypot(dx, dy)
        dist_err = dist - self.desired_follow_dist

        target_heading = math.atan2(dy, dx)
        heading_err = target_heading - robot_heading
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        # ---- 速度控制（外层位置PID + 动态限速） ----
        self.dist_err_sum += dist_err * dt
        self.dist_err_sum = max(-1.0, min(1.0, self.dist_err_sum))
        Kp_dist = 2.7
        Ki_dist = 0.03
        Kd_dist = 0.05
        v_adj = (Kp_dist * dist_err +
                 Ki_dist * self.dist_err_sum +
                 Kd_dist * (dist_err - self.last_dist_err) / dt)
        self.last_dist_err = dist_err

        abs_heading_err = abs(heading_err)
        if abs_heading_err < math.radians(20):
            max_v = 1.5
        elif abs_heading_err < math.radians(60):
            max_v = 0.8
        elif abs_heading_err < math.radians(120):
            max_v = 0.3
        else:
            max_v = 0.1
        v_cmd = max(0.0, min(max_v, v_adj))

        # ---- 转向控制（航向误差反馈 + 转向前馈） ----
        Kp_heading = -9.0
        turn_feedforward = -1.5 * human_turn
        raw_yaw = Kp_heading * heading_err + turn_feedforward
        alpha = 0.3
        self.yaw_filtered = alpha * raw_yaw + (1 - alpha) * self.yaw_filtered
        yaw_cmd = max(-1.5, min(1.5, self.yaw_filtered))

        # 更新内部目标，供 update_motor_speed 使用
        self.set_velocity_linear_set_point(v_cmd)
        self.set_yaw(yaw_cmd)

        return v_cmd, yaw_cmd, dist_err, heading_err

    def reset(self):
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.speed_error_integral = 0.0
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]