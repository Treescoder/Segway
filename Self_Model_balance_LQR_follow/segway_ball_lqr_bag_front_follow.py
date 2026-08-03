"""无称重传感器的平衡车控制器。

控制器运行时只使用：俯仰角/角速度、轮速、yaw和上一周期力矩。
球包质量只用于MuJoCo修改被控对象，不进入控制算法。

状态：[phi, pitch_rate, target_speed-cart_speed, speed_integral]
控制：u = -K @ x

K使用4.5 kg中间载荷模型离线计算。未知平衡角由扰动观测器在线估计。
"""

import numpy as np
import mujoco


R_WHEEL = 0.24
MIN_MUJOCO_MASS = 1e-4

# 4.5 kg中间载荷的离线LQR增益，嵌入式可直接固化。
FIXED_LQR_GAIN = np.array([[
    171.1217, 41.6678, -25.8755, -7.0711
]])

INTEGRAL_LIMIT = 1.5
MAX_ACCEL_V_ERR = 0.90
MAX_BRAKE_V_ERR = 0.90
MAX_TORQUE = 80.0

# 名义4.5 kg模型：pitch_ddot = A*(pitch-theta_eq) + B*u。
NOMINAL_A_PITCH = 22.6899
NOMINAL_B_PITCH = 0.38239
PITCH_ACCEL_FILTER_TAU = 0.05
THETA_OBSERVER_TAU = 0.10
THETA_ADAPT_GAIN = 0.01
THETA_ADAPT_ERROR_LIMIT = 0.60
THETA_EQ_LIMIT = np.deg2rad(15.0)

KP_YAW = 120.0
KD_YAW = 15.0
MAX_STEER_TORQUE = 45.0


def clamp(value, low, high):
    return max(low, min(high, value))


class SegwayLQR:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.body_id = model.body("segway").id
        self.bag_body_id = model.body("golf_bag").id
        self.bag_geom_id = model.geom("golf_bag_geom").id
        self.l_wheel_dof = model.jnt_dofadr[
            model.joint("torso_l_wheel").id
        ]
        self.r_wheel_dof = model.jnt_dofadr[
            model.joint("torso_r_wheel").id
        ]
        free_joint_id = model.joint("segway_free").id
        self.segway_qpos_addr = model.jnt_qposadr[free_joint_id]
        self.segway_dof_addr = model.jnt_dofadr[free_joint_id]

        # 以下数据仅服务于MuJoCo质量滑块，控制公式不读取。
        self._sim_bag_mass = float(model.body_mass[self.bag_body_id])
        self._bag_nominal_mass = self._sim_bag_mass
        self._bag_nominal_inertia = model.body_inertia[
            self.bag_body_id
        ].copy()
        self._bag_nominal_contype = int(
            model.geom_contype[self.bag_geom_id]
        )
        self._bag_nominal_conaffinity = int(
            model.geom_conaffinity[self.bag_geom_id]
        )
        self._bag_nominal_rgba = model.geom_rgba[
            self.bag_geom_id
        ].copy()

        self.K = FIXED_LQR_GAIN.copy()
        self.v_target = 0.0
        self.yaw_target = 0.0
        self.theta_eq = 0.0  # 观测器估计值，不是质量计算值
        self._v_integral = 0.0
        self._v_filtered = 0.0
        self._prev_yaw = 0.0
        self._yaw_rate = 0.0
        self._prev_pitch_dot = 0.0
        self._pitch_accel_filtered = 0.0
        self._last_u = 0.0

    def set_simulated_bag_mass(self, new_mass):
        """仅改变MuJoCo物理质量，模拟球杆取放。

        此数值不参与theta_eq、K或任何控制量计算。
        """
        new_mass = float(np.clip(new_mass, 0.0, 9.0))
        is_preset = any(
            abs(new_mass - value) < 1e-9 for value in (0.0, 3.0, 9.0)
        )
        if (
            not is_preset
            and abs(new_mass - self._sim_bag_mass) < 0.05
        ):
            return

        self._sim_bag_mass = new_mass
        physics_mass = max(new_mass, MIN_MUJOCO_MASS)
        inertia_scale = physics_mass / self._bag_nominal_mass
        self.model.body_mass[self.bag_body_id] = physics_mass
        self.model.body_inertia[self.bag_body_id] = (
            self._bag_nominal_inertia * inertia_scale
        )

        bag_present = new_mass >= 0.1
        self.model.geom_contype[self.bag_geom_id] = (
            self._bag_nominal_contype if bag_present else 0
        )
        self.model.geom_conaffinity[self.bag_geom_id] = (
            self._bag_nominal_conaffinity if bag_present else 0
        )
        self.model.geom_rgba[self.bag_geom_id] = self._bag_nominal_rgba
        self.model.geom_rgba[self.bag_geom_id, 3] = (
            self._bag_nominal_rgba[3] if bag_present else 0.0
        )
        mujoco.mj_forward(self.model, self.data)

    def set_velocity(self, velocity):
        self.v_target = max(0.0, float(velocity))

    def set_yaw(self, yaw):
        self.yaw_target = float(yaw)

    def update(self, dt=0.005):
        dt = clamp(float(dt), 1e-4, 0.1)
        pitch = self.get_pitch()
        pitch_dot = self._get_pitch_dot()

        # 编码器车速低通，抑制转向打滑尖峰。
        speed_alpha = dt / (dt + 0.05)
        self._v_filtered += speed_alpha * (
            self.get_speed() - self._v_filtered
        )
        speed_error = self.v_target - self._v_filtered

        # yaw角速度。
        current_yaw = self.get_yaw()
        yaw_error_step = np.arctan2(
            np.sin(current_yaw - self._prev_yaw),
            np.cos(current_yaw - self._prev_yaw),
        )
        self._yaw_rate = yaw_error_step / dt
        self._prev_yaw = current_yaw

        # 扰动观测：用IMU角加速度和上一周期力矩反推平衡角。
        pitch_accel = (pitch_dot - self._prev_pitch_dot) / dt
        self._prev_pitch_dot = pitch_dot
        accel_alpha = dt / (dt + PITCH_ACCEL_FILTER_TAU)
        self._pitch_accel_filtered += accel_alpha * (
            pitch_accel - self._pitch_accel_filtered
        )
        theta_observed = pitch - (
            self._pitch_accel_filtered
            - NOMINAL_B_PITCH * self._last_u
        ) / NOMINAL_A_PITCH
        theta_observed = clamp(
            theta_observed, -THETA_EQ_LIMIT, THETA_EQ_LIMIT
        )
        observer_alpha = dt / (dt + THETA_OBSERVER_TAU)
        self.theta_eq += observer_alpha * (
            theta_observed - self.theta_eq
        )

        # 车速误差仅作慢速零偏修正，不用人体或球包质量。
        adapt_error = clamp(
            speed_error,
            -THETA_ADAPT_ERROR_LIMIT,
            THETA_ADAPT_ERROR_LIMIT,
        )
        self.theta_eq = clamp(
            self.theta_eq + THETA_ADAPT_GAIN * adapt_error * dt,
            -THETA_EQ_LIMIT,
            THETA_EQ_LIMIT,
        )

        phi = pitch - self.theta_eq
        speed_error_limited = clamp(
            speed_error, -MAX_BRAKE_V_ERR, MAX_ACCEL_V_ERR
        )

        # 积分抗饱和。
        u_preview = -(
            self.K[0, 0] * phi
            + self.K[0, 1] * pitch_dot
            + self.K[0, 2] * speed_error_limited
            + self.K[0, 3] * self._v_integral
        )
        if abs(u_preview) < MAX_TORQUE * 0.9:
            self._v_integral = clamp(
                self._v_integral + speed_error * dt,
                -INTEGRAL_LIMIT,
                INTEGRAL_LIMIT,
            )

        u = -(
            self.K[0, 0] * phi
            + self.K[0, 1] * pitch_dot
            + self.K[0, 2] * speed_error_limited
            + self.K[0, 3] * self._v_integral
        )
        u = clamp(u, -MAX_TORQUE, MAX_TORQUE)
        self._last_u = u

        steer = clamp(
            self._compute_steer_torque(),
            -MAX_STEER_TORQUE,
            MAX_STEER_TORQUE,
        )
        left_torque = (u - steer) / 2.0
        right_torque = (u + steer) / 2.0
        self.data.actuator("motor_l_wheel").ctrl = [left_torque]
        self.data.actuator("motor_r_wheel").ctrl = [right_torque]

    def reset(self):
        self.v_target = 0.0
        self.yaw_target = 0.0
        self.theta_eq = 0.0
        self._v_integral = 0.0
        self._v_filtered = 0.0
        self._prev_pitch_dot = 0.0
        self._pitch_accel_filtered = 0.0
        self._last_u = 0.0

        mujoco.mj_resetData(self.model, self.data)
        # 不使用质量预置姿态，从竖直0°启动。
        self.data.qpos[self.segway_qpos_addr + 3] = 1.0
        self.data.qpos[self.segway_qpos_addr + 4] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._prev_yaw = self.get_yaw()
        self._yaw_rate = 0.0

    def get_pitch(self):
        body_rotation = self.data.xmat[self.body_id].reshape(3, 3)
        forward = body_rotation[:, 1]
        return np.arctan2(
            forward[2], np.hypot(forward[0], forward[1])
        )

    def get_pitch_error(self):
        return self.get_pitch() - self.theta_eq

    def _get_pitch_dot(self):
        # MuJoCo free joint第一个转动速度是本模型轮轴方向。
        return float(self.data.qvel[self.segway_dof_addr + 3])

    def get_yaw(self):
        body_rotation = self.data.xmat[self.body_id].reshape(3, 3)
        forward = body_rotation[:, 1]
        return np.arctan2(-forward[0], forward[1])

    def get_roll(self):
        body_rotation = self.data.xmat[self.body_id].reshape(3, 3)
        right = body_rotation[:, 0]
        return np.arctan2(right[2], np.hypot(right[0], right[1]))

    def get_position(self):
        return self.data.xpos[self.body_id].copy()

    def get_speed(self):
        left_rate = self.data.qvel[self.l_wheel_dof]
        right_rate = self.data.qvel[self.r_wheel_dof]
        return -(left_rate + right_rate) * 0.5 * R_WHEEL

    def _compute_steer_torque(self):
        yaw_error = np.arctan2(
            np.sin(self.get_yaw() - self.yaw_target),
            np.cos(self.get_yaw() - self.yaw_target),
        )
        return -KP_YAW * yaw_error - KD_YAW * self._yaw_rate
