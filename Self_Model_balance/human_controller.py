"""MuJoCo mocap 人体的随机步行/跑步控制。

坐标约定与球车一致：
  yaw = 0 时面向 -Y（摄像头一侧）
  x += speed * sin(yaw) * dt
  y -= speed * cos(yaw) * dt
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class HumanController:
    """仅用于 MuJoCo：生成连续变速、随机转弯的人体轨迹和肢体动画。"""

    MAX_SPEED = 3.20          # 11.52 km/h，跑步段会明确超过 3 m/s
    MIN_SPEED = 0.0
    MAX_ACCEL = 0.50          # 人和车使用接近的速度斜率，避免追赶冲击
    MAX_DECEL = 0.60
    MAX_YAW_RATE = 0.42       # rad/s，约 24 deg/s

    SPEED_CHANGE_MIN = 2.0
    SPEED_CHANGE_MAX = 4.5
    YAW_CHANGE_MIN = 1.6
    YAW_CHANGE_MAX = 3.4

    BOUNDARY_RADIUS = 200.0

    # 每个周期都安排一次足够长的跑步段，保证实际速度超过 3 m/s。
    SPRINT_START = 3.0
    SPRINT_DURATION = 9.0
    SPRINT_INTERVAL = 22.0

    def __init__(self, model, data, start_x=0.0, start_y=-2.5):
        self.model = model
        self.data = data

        body_id = model.body("person").id
        self.mocap_id = model.body_mocapid[body_id]

        self.x = start_x
        self.y = start_y
        self.yaw = 0.0
        self.speed = 0.0
        self.motion_speed = 0.0
        self.accel = 0.0

        self.target_speed = 1.8
        self.target_yaw = 0.0

        self._speed_timer = 0.0
        self._speed_interval = 2.0
        self._yaw_timer = 0.0
        self._yaw_interval = 2.0
        self._walk_phase = 0.0
        self._elapsed = 0.0

        actuator_names = (
            "human_l_shoulder_act",
            "human_r_shoulder_act",
            "human_l_elbow_act",
            "human_r_elbow_act",
            "human_l_hip_act",
            "human_r_hip_act",
        )
        self._joint_actuators = {
            name: model.actuator(name).id
            for name in actuator_names
            if mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            ) >= 0
        }

        self._update_mocap()

    def reset(self, x=0.0, y=-2.5, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = 0.0
        self.motion_speed = 0.0
        self.accel = 0.0
        self.target_speed = 1.8
        self.target_yaw = yaw
        self._speed_timer = 0.0
        self._speed_interval = 2.0
        self._yaw_timer = 0.0
        self._yaw_interval = 2.0
        self._walk_phase = 0.0
        self._elapsed = 0.0
        self._update_mocap()
        self._update_joint_animation()

    def update(self, dt):
        self._elapsed += dt
        cycle_time = self._elapsed % self.SPRINT_INTERVAL
        sprinting = (
            self.SPRINT_START
            <= cycle_time
            < self.SPRINT_START + self.SPRINT_DURATION
        )

        # 速度并非匀速：跑步段达到 3.2 m/s，其余时间按偏高速分布变化。
        self._speed_timer += dt
        if sprinting:
            self.target_speed = self.MAX_SPEED
        elif self._speed_timer >= self._speed_interval:
            self._speed_timer = 0.0
            self._speed_interval = np.random.uniform(
                self.SPEED_CHANGE_MIN, self.SPEED_CHANGE_MAX
            )
            mode = np.random.random()
            if mode < 0.05:
                self.target_speed = np.random.uniform(0.8, 1.2)
            elif mode < 0.25:
                self.target_speed = np.random.uniform(1.3, 1.9)
            elif mode < 0.60:
                self.target_speed = np.random.uniform(1.9, 2.4)
            else:
                self.target_speed = np.random.uniform(2.5, self.MAX_SPEED)

        # 每次都选择 22°～58° 的明确左转或右转，避免随机到近似直行。
        self._yaw_timer += dt
        if self._yaw_timer >= self._yaw_interval:
            self._yaw_timer = 0.0
            self._yaw_interval = np.random.uniform(
                self.YAW_CHANGE_MIN, self.YAW_CHANGE_MAX
            )
            turn_sign = -1.0 if np.random.random() < 0.5 else 1.0
            turn_magnitude = np.random.uniform(
                np.deg2rad(22.0), np.deg2rad(58.0)
            )
            self.target_yaw += turn_sign * turn_magnitude

        # 离开大场地中心太远时返回中心。
        if np.hypot(self.x, self.y) > self.BOUNDARY_RADIUS:
            self.target_yaw = np.arctan2(-self.x, self.y)

        speed_error = self.target_speed - self.speed
        rate = self.MAX_ACCEL if speed_error > 0.0 else self.MAX_DECEL
        speed_step = rate * dt
        speed_change = np.clip(speed_error, -speed_step, speed_step)
        self.accel = speed_change / dt if dt > 0.0 else 0.0
        self.speed = np.clip(
            self.speed + speed_change, self.MIN_SPEED, self.MAX_SPEED
        )

        # 两个低幅、不同频率的步幅分量，消除机械匀速感。
        gait_variation = (
            0.055 * np.sin(self._walk_phase * 0.73)
            + 0.025 * np.sin(self._walk_phase * 1.91)
        )
        self.motion_speed = np.clip(
            self.speed + gait_variation, self.MIN_SPEED, self.MAX_SPEED
        )

        yaw_error = np.arctan2(
            np.sin(self.target_yaw - self.yaw),
            np.cos(self.target_yaw - self.yaw),
        )
        speed_ratio = self.motion_speed / self.MAX_SPEED
        # 高速时曲率稍小，仍至少约 15.6 deg/s，既明显转弯又不瞬移。
        yaw_rate_limit = self.MAX_YAW_RATE * (1.0 - 0.35 * speed_ratio)
        yaw_step = yaw_rate_limit * dt
        self.yaw += np.clip(yaw_error, -yaw_step, yaw_step)
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))

        self.x += self.motion_speed * np.sin(self.yaw) * dt
        self.y -= self.motion_speed * np.cos(self.yaw) * dt
        self._walk_phase += self.motion_speed * dt * 2.8

        self._update_mocap()
        self._update_joint_animation()

    def _update_joint_animation(self):
        if not self._joint_actuators:
            return

        activity = min(self.motion_speed / 0.45, 1.0)
        speed_ratio = self.motion_speed / self.MAX_SPEED
        swing = np.sin(self._walk_phase)

        # 手臂摆幅维持原效果；腿仅由髋部小幅抬起，膝盖固定不弯。
        arm_amplitude = activity * (0.35 + 0.45 * speed_ratio)
        leg_amplitude = activity * (0.16 + 0.16 * speed_ratio)
        targets = {
            "human_l_hip_act": leg_amplitude * swing,
            "human_r_hip_act": -leg_amplitude * swing,
            "human_l_shoulder_act": -0.75 * arm_amplitude * swing,
            "human_r_shoulder_act": 0.75 * arm_amplitude * swing,
            "human_l_elbow_act": activity * (
                0.30 + 0.20 * max(0.0, swing)
            ),
            "human_r_elbow_act": activity * (
                0.30 + 0.20 * max(0.0, -swing)
            ),
        }
        for name, value in targets.items():
            actuator_id = self._joint_actuators.get(name)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = value

    def _update_mocap(self):
        bob = (
            0.025
            * abs(np.sin(self._walk_phase))
            * min(self.motion_speed / 1.0, 1.0)
        )
        lean = np.clip(-self.accel * 0.04, -0.08, 0.08)
        self.data.mocap_pos[self.mocap_id] = [self.x, self.y, bob]

        rotation = (
            Rotation.from_euler("z", self.yaw + np.pi)
            * Rotation.from_euler("x", lean)
        )
        quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
        self.data.mocap_quat[self.mocap_id] = [
            quaternion[3],
            quaternion[0],
            quaternion[1],
            quaternion[2],
        ]

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            # 跟踪器使用平滑基准速度；motion_speed 只负责视觉上的步态起伏。
            "speed": self.speed,
        }
