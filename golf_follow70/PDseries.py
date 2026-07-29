# -*- coding: utf-8 -*-
"""
SegwayPID 平衡控制（串级：速度环 → 俯仰环 + 偏航差速环）
与原版唯一差别：reset() 中 free joint 的 qpos 地址改为动态获取。
原来硬编码 qpos[3:7]，在场景 include 了 human.xml 之后行人关节会排在前面，
硬编码地址会写坏行人关节，因此必须动态定位 segway_free。
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.24
wheel_base = 0.25 * 2
MAX_TORQUE = 50.0

PITCH_KP = 216.08
PITCH_KD = 71.60
PITCH_KI = 31.78
PITCH_INT_LIMIT = 6.0

SPEED_KP = 0.288
SPEED_KD = 0.0255
SPEED_KI = 0.0
SPEED_INT_LIMIT = 10.0
# ★ 串级外环（速度环）输出限幅：限制其对 pitch_target 的最大贡献（rad）。
#   没有这层限幅时，KP 项（0.21×2m/s≈24°）+ KI 积分（0.08×10=46°）叠加在
#   轻载 -33° 大平衡角上会把 pitch_target 顶死到 ±45° 总钳位 → 车身近乎横躺、
#   球包蹭地，只能龟速爬行（轻包 3m/s 跟随发散的根因）。
SPEED_CORR_LIMIT = np.deg2rad(14)

FEEDFORWARD_TORQUE = 0

YAW_DEADZONE = 0.01
YAW_GAIN = 1.0
YAW_KP = 295.82
YAW_KD = 168.79

DOB_GAIN = 0

# ★ 平衡角自适应学习率（原硬编码 0.005，现改为可调，并改为「只要俯仰角速度小就持续学习」）
BALANCE_ALPHA = 0.06
BALANCE_PITCH_DOT_LIMIT = np.deg2rad(10)  # 俯仰角速度低于此值即视为准静态，更新平衡基准

# ★ 平衡角自适应「锚」（可配置）：
#   - 初值 BALANCE_PITCH_INIT：起步时的平衡角猜测。若结构上已知平衡角（如 com_y=0 时
#     恒为 +28.7°，与球包质量无关），预置初值可消除起步「爬角期」被行人拉开的问题。
#   - 钳位 [BALANCE_PITCH_MIN, BALANCE_PITCH_MAX]：自适应律本质是正反馈漂移，窗口就是锚，
#     窗口越窄越稳。默认值 = 原 -0.12 后置重心配置（初值 0，窗 [-5°, 30°]），行为不变。
BALANCE_PITCH_INIT = 0.0
BALANCE_PITCH_MIN = np.deg2rad(-5)
BALANCE_PITCH_MAX = np.deg2rad(30)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class SegwayPID:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0
        self.balance_pitch = BALANCE_PITCH_INIT

        self.pitch_integral = 0.0
        self.speed_error_integral = 0.0
        self.filtered_wheel_vel = 0.0
        self.prev_vel_error = 0.0

        self.body_id = model.body('segway').id
        self.l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]
        # ★ 修复：动态获取 free joint 的 qpos 起始地址（include human.xml 后不再是 0）
        self.free_qposadr = model.jnt_qposadr[model.joint('segway_free').id]

    def set_velocity_linear_set_point(self, vel):
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw):
        self.yaw = yaw

    def get_pitch(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('segway_free').qvel[-3:]
        return angular[0]

    def get_roll(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler('xyz', degrees=False)[1]

    def get_yaw(self) -> float:
        quat = self.data.xquat[self.body_id]
        if quat[0] == 0:
            return 0.0
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler("xyz", degrees=False)[2]

    def get_wheel_velocity_avg(self) -> float:
        return (self.data.qvel[self.l_dof] + self.data.qvel[self.r_dof]) / 2.0

    def update_motor_torque(self):
        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot()
        roll = self.get_roll()
        wheel_vel = self.get_wheel_velocity_avg()
        current_yaw = self.get_yaw()

        self.filtered_wheel_vel = 0.9 * self.filtered_wheel_vel + 0.1 * wheel_vel

        actual_speed = self.filtered_wheel_vel * WHEEL_RADIUS

        # ============================
        # 更新 balance pitch（★ 改为：只要俯仰角速度足够小即持续学习，
        #   不再要求近乎静止，这样持续跟随时也能把基准贴合实际平衡点，
        #   俯仰环只需要修正动态偏差，pitch 波动更小）
        # ============================
        if abs(pitch_dot) < BALANCE_PITCH_DOT_LIMIT:
            self.balance_pitch += BALANCE_ALPHA * (pitch - self.balance_pitch)
        # ★ 注意：此自适应律（balance_pitch 追踪实际 pitch）本质是正反馈漂移，
        #   钳位窗口就是它的「锚」——实测把窗放宽到 ±40° 会让全负载段（含满包）漂移发散。
        #   负载变化引起的平衡角偏移由「俯仰积分 + 放宽的 pitch_target 限幅」承担，勿放宽窗口。
        self.balance_pitch = clamp(self.balance_pitch, BALANCE_PITCH_MIN, BALANCE_PITCH_MAX)

        vel_error = actual_speed - self.velocity_linear_set_point
        # ★ 速度误差微分（修正原 bug：原代码第二项是 SPEED_KD*vel_error 而非微分）
        vel_dot = (vel_error - self.prev_vel_error) / 0.005
        self.prev_vel_error = vel_error
        self.speed_error_integral += vel_error * 0.005
        self.speed_error_integral = clamp(self.speed_error_integral, -SPEED_INT_LIMIT, SPEED_INT_LIMIT)
        speed_correction = (SPEED_KP * vel_error
                            + SPEED_KI * self.speed_error_integral
                            + SPEED_KD * vel_dot)
        # ★ 外环输出限幅 + 抗积分饱和（back-calculation）：
        #   输出被钳住时把积分回退到「刚好饱和」的水平，避免积分继续堆积
        #   导致松开后大幅过冲（轻包 5m/s 冲过头的根因之一）。
        corr_clamped = clamp(speed_correction, -SPEED_CORR_LIMIT, SPEED_CORR_LIMIT)
        if SPEED_KI > 0.0 and corr_clamped != speed_correction:
            self.speed_error_integral -= (speed_correction - corr_clamped) / SPEED_KI
            self.speed_error_integral = clamp(self.speed_error_integral,
                                              -SPEED_INT_LIMIT, SPEED_INT_LIMIT)
        speed_correction = corr_clamped

        pitch_target = speed_correction + self.balance_pitch
        # ★ ±45°：给大平衡角（轻载 -33°）之上留出速度环修正空间（旧 ±30° 会钳死）
        pitch_target = clamp(pitch_target, np.deg2rad(-45), np.deg2rad(45))

        pitch_error = pitch_target - pitch
        self.pitch_integral += pitch_error * 0.005
        self.pitch_integral = clamp(self.pitch_integral, -PITCH_INT_LIMIT, PITCH_INT_LIMIT)
        torque_balance = PITCH_KP * pitch_error - PITCH_KD * pitch_dot + PITCH_KI * self.pitch_integral

        total_torque = torque_balance + FEEDFORWARD_TORQUE

        yaw_error = np.arctan2(
            np.sin(self.yaw - current_yaw),
            np.cos(self.yaw - current_yaw)
        )
        yaw_diff = YAW_KP * yaw_error

        left_torque = clamp(total_torque - yaw_diff, -MAX_TORQUE, MAX_TORQUE)
        right_torque = clamp(total_torque + yaw_diff, -MAX_TORQUE, MAX_TORQUE)

        self.data.actuator('motor_l_wheel').ctrl[0] = left_torque
        self.data.actuator('motor_r_wheel').ctrl[0] = right_torque

    def reset(self):
        self.balance_pitch = BALANCE_PITCH_INIT
        self.pitch_integral = 0.0
        self.speed_error_integral = 0.0
        self.filtered_wheel_vel = 0.0
        self.prev_vel_error = 0.0
        init_pitch = 0.0
        qx = np.sin(init_pitch / 2.0)
        qw = np.cos(init_pitch / 2.0)
        # ★ 修复：使用动态地址而非硬编码 qpos[3:7]
        adr = self.free_qposadr
        self.data.qpos[adr + 3:adr + 7] = [qw, qx, 0.0, 0.0]
        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]
