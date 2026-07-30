"""面向实车/C++ 的极简人员跟随控制器。

外部人员跟踪只需提供 human_speed、human_yaw。
distance、bearing_error 来自车载测距/摄像头；
cart_speed 来自轮速编码器，cart_yaw 来自 IMU。
"""

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class TrackingController:
    """人速前馈 + 车速P反馈 + 小距离修正。"""

    DESIRED_DISTANCE = 2.5
    MIN_DISTANCE = 1.0
    MAX_DISTANCE = 3.0

    KP_DISTANCE = 0.28
    MAX_DISTANCE_SPEED = 0.60
    DISTANCE_SPEED_RATE = 0.50  # 距离补偿每秒最多变化 0.50 m/s
    KP_SPEED = 0.45
    MAX_SPEED_FEEDBACK = 0.35
    MAX_SPEED = 3.80

    YAW_FEEDFORWARD = 0.50
    MAX_YAW_OFFSET = math.radians(35.0)
    CAMERA_HALF_FOV = math.radians(55.0)

    def __init__(self, desired_distance=2.5):
        self.desired_distance = float(desired_distance)
        self.direct_distance = self.desired_distance
        self.camera_error = 0.0
        self.person_visible = True
        self.safety_active = False
        self._distance_speed = 0.0

    def reset(self):
        self.safety_active = False
        self._distance_speed = 0.0

    def update(self, distance, bearing_error,
               human_speed, human_yaw, cart_speed, cart_yaw, dt):
        # 1. 纵向：人速是主指令；编码器反馈用于消除实际车速滞后。
        distance_error = distance - self.desired_distance
        target_distance_speed = clamp(
            self.KP_DISTANCE * distance_error,
            -self.MAX_DISTANCE_SPEED,
            self.MAX_DISTANCE_SPEED)
        distance_step = self.DISTANCE_SPEED_RATE * dt
        self._distance_speed += clamp(
            target_distance_speed - self._distance_speed,
            -distance_step,
            distance_step)
        speed_feedback = clamp(
            self.KP_SPEED * (human_speed - cart_speed),
            -self.MAX_SPEED_FEEDBACK,
            self.MAX_SPEED_FEEDBACK)
        speed_cmd = human_speed + speed_feedback + self._distance_speed

        # 仅在真正接近 1 m 时柔和降速；正常转弯不触发急刹或倒退。
        self.safety_active = distance < 1.3
        if self.safety_active:
            safe_scale = clamp(
                (distance - self.MIN_DISTANCE) / 0.3, 0.0, 1.0)
            speed_cmd = min(speed_cmd, human_speed * safe_scale)
        speed_cmd = clamp(speed_cmd, 0.0, self.MAX_SPEED)

        # 2. 横向：摄像头视线反馈 + 人体 yaw 前馈。
        sight_yaw = wrap_angle(cart_yaw + bearing_error)
        yaw_offset = clamp(
            wrap_angle(human_yaw - sight_yaw),
            -self.MAX_YAW_OFFSET,
            self.MAX_YAW_OFFSET)
        yaw_cmd = wrap_angle(
            sight_yaw + self.YAW_FEEDFORWARD * yaw_offset)

        self.direct_distance = distance
        self.camera_error = bearing_error
        self.person_visible = abs(bearing_error) <= self.CAMERA_HALF_FOV
        return speed_cmd, yaw_cmd, distance
