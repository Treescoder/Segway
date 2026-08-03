"""可直接移植到 C++ 的人员跟随控制器。

输入:
    人体速度/yaw、人与车的距离/方位、车辆速度/yaw、控制周期 dt
输出:
    已做加减速限制的车辆速度和 yaw 参考值
"""

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def move_towards(value, target, max_step):
    return value + clamp(target - value, -max_step, max_step)


class TrackingController:
    """人速前馈 + 车速反馈 + 距离闭环 + 预测防碰撞。"""

    DESIRED_DISTANCE = 1.90
    MIN_DISTANCE = 1.0
    MAX_DISTANCE = 3.0

    KP_DISTANCE = 0.55
    MAX_DISTANCE_SPEED = 0.95
    KP_SPEED = 0.45
    MAX_SPEED_FEEDBACK = 0.45
    MAX_SPEED = 3.90

    ACCEL_LIMIT = 0.85
    CATCHUP_DISTANCE = 2.60
    CATCHUP_ACCEL_LIMIT = 1.25
    DECEL_LIMIT = 1.15
    LOST_TARGET_DECEL = 1.20

    # 制动预测: 可用距离 = 闭合速度*反应时间 + 闭合速度²/(2*减速度)
    SAFETY_MARGIN = 0.12
    REACTION_TIME = 0.15
    SAFE_DECEL = 0.90

    # 人接近画面边缘时先降速再转向，避免高速切弯时从人身边冲过。
    TURN_SLOW_START = math.radians(25.0)
    TURN_SLOW_END = math.radians(52.0)
    MIN_TURN_SPEED = 0.80

    YAW_FEEDFORWARD = 0.45
    MAX_YAW_OFFSET = math.radians(35.0)
    MAX_YAW_RATE = 3.0
    CAMERA_HALF_FOV = math.radians(55.0)

    def __init__(self, desired_distance=DESIRED_DISTANCE):
        self.desired_distance = float(desired_distance)
        self.speed_ref = 0.0
        self.yaw_ref = 0.0
        self.direct_distance = self.desired_distance
        self.camera_error = 0.0
        self.person_visible = True
        self.safety_active = False

    def reset(self, cart_speed=0.0, cart_yaw=0.0):
        self.speed_ref = max(0.0, float(cart_speed))
        self.yaw_ref = float(cart_yaw)
        self.safety_active = False

    def update(self, distance, bearing_error,
               human_speed, human_yaw, cart_speed, cart_yaw, dt,
               target_valid=True):
        """计算下一周期参考值；角度单位 rad，速度单位 m/s。"""
        dt = clamp(float(dt), 1e-4, 0.1)
        values = (
            distance, bearing_error, human_speed,
            human_yaw, cart_speed, cart_yaw,
        )
        target_valid = target_valid and all(math.isfinite(v) for v in values)
        if not target_valid:
            self.person_visible = False
            self.safety_active = True
            self.speed_ref = move_towards(
                self.speed_ref, 0.0, self.LOST_TARGET_DECEL * dt
            )
            return self.speed_ref, self.yaw_ref, self.direct_distance

        distance = max(0.0, float(distance))
        human_speed = clamp(float(human_speed), 0.0, self.MAX_SPEED)
        cart_speed = max(0.0, float(cart_speed))

        # 视线航向：从车辆指向人体。
        sight_yaw = wrap_angle(cart_yaw + bearing_error)

        # 纵向基础指令：人体速度是前馈，车速误差和距离误差做反馈。
        distance_speed = clamp(
            self.KP_DISTANCE * (distance - self.desired_distance),
            -self.MAX_DISTANCE_SPEED,
            self.MAX_DISTANCE_SPEED,
        )
        speed_feedback = clamp(
            self.KP_SPEED * (human_speed - cart_speed),
            -self.MAX_SPEED_FEEDBACK,
            self.MAX_SPEED_FEEDBACK,
        )
        speed_target = human_speed + speed_feedback + distance_speed

        turn_ratio = clamp(
            (abs(bearing_error) - self.TURN_SLOW_START)
            / (self.TURN_SLOW_END - self.TURN_SLOW_START),
            0.0,
            1.0,
        )
        turn_speed_limit = (
            self.MAX_SPEED * (1.0 - turn_ratio)
            + self.MIN_TURN_SPEED * turn_ratio
        )
        speed_target = min(speed_target, turn_speed_limit)

        # 预测安全速度。转弯时使用沿人车视线方向的速度，而不是只比较标量车速。
        human_radial = human_speed * math.cos(
            wrap_angle(human_yaw - sight_yaw)
        )
        cart_projection = max(
            0.2, math.cos(wrap_angle(cart_yaw - sight_yaw))
        )
        cart_radial = cart_speed * cart_projection
        closing_speed = max(0.0, cart_radial - human_radial)
        available = max(
            0.0, distance - self.MIN_DISTANCE - self.SAFETY_MARGIN
        )
        a = self.SAFE_DECEL
        allowed_closing = (
            -a * self.REACTION_TIME
            + math.sqrt(
                (a * self.REACTION_TIME) ** 2 + 2.0 * a * available
            )
        )
        safe_cart_speed = max(
            0.0, (human_radial + allowed_closing) / cart_projection
        )
        self.safety_active = (
            closing_speed > allowed_closing
            or distance < self.MIN_DISTANCE + self.SAFETY_MARGIN
        )
        speed_target = min(speed_target, safe_cart_speed)
        speed_target = clamp(speed_target, 0.0, self.MAX_SPEED)

        # 唯一的一层速度斜坡；主程序不得再次对跟踪速度做限速。
        if speed_target >= self.speed_ref:
            rate = (
                self.CATCHUP_ACCEL_LIMIT
                if distance > self.CATCHUP_DISTANCE
                else self.ACCEL_LIMIT
            )
        else:
            rate = self.DECEL_LIMIT
        self.speed_ref = move_towards(
            self.speed_ref, speed_target, rate * dt
        )

        # 摄像头视线反馈 + 人体 yaw 前馈，让车辆在人体转弯时提前转向。
        yaw_offset = clamp(
            wrap_angle(human_yaw - sight_yaw),
            -self.MAX_YAW_OFFSET,
            self.MAX_YAW_OFFSET,
        )
        yaw_target = wrap_angle(
            sight_yaw + self.YAW_FEEDFORWARD * yaw_offset
        )
        yaw_error = wrap_angle(yaw_target - self.yaw_ref)
        self.yaw_ref = wrap_angle(
            self.yaw_ref
            + clamp(
                yaw_error,
                -self.MAX_YAW_RATE * dt,
                self.MAX_YAW_RATE * dt,
            )
        )

        self.direct_distance = distance
        self.camera_error = bearing_error
        self.person_visible = abs(bearing_error) <= self.CAMERA_HALF_FOV
        return self.speed_ref, self.yaw_ref, distance
