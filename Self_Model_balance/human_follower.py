import math

class HumanFollower:
    def __init__(self, desired_dist=1.5):
        self.desired_dist = desired_dist

    def compute(self, robot_pos, robot_mat, human_x, human_y, human_heading, human_speed, human_turn, dt):
        # 前进方向：局部 -y
        forward_global = -robot_mat[:, 1]
        robot_heading = math.atan2(forward_global[1], forward_global[0])

        dx = human_x - robot_pos[0]
        dy = human_y - robot_pos[1]
        dist = math.hypot(dx, dy)
        dist_err = dist - self.desired_dist   # 正=人远，需靠近

        target_heading = math.atan2(dy, dx)
        heading_err = target_heading - robot_heading
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        # 极简 P 控制：误差为正（人远）→ 负速度靠近；误差为负（人近）→ 正速度后退
        Kp = 0.3   # 极低增益，缓慢修正
        v_cmd = -Kp * dist_err
        max_v = 10 / 3.6
        v_cmd = max(-max_v, min(max_v, v_cmd))

        # 转向控制（保持柔和）
        Kp_heading = 3.0
        turn_feedforward = 0.5 * human_turn
        raw_yaw = Kp_heading * heading_err + turn_feedforward
        yaw_cmd = max(-1.5, min(1.5, raw_yaw))

        return v_cmd, yaw_cmd, dist_err, heading_err, max_v, v_cmd

    def reset(self):
        pass