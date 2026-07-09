import pathlib
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

def clamp(x, xmin, xmax):
    return max(min(xmax, x), xmin)

def main():
    # ================== 模型加载 ==================
    model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent / "xml/scene.xml"))
    data = mujoco.MjData(model)

    # ================== 仿真参数 ==================
    model.opt.timestep = 0.001
    CONTROL_FREQ = 200          # Hz
    T_CTL = 1.0 / CONTROL_FREQ
    next_ctrl_time = 0.0
    PRINT_PERIOD = 0.5          # s

    # ================== ID获取 ==================
    body_id = model.body("robot_body").id
    body_joint = model.joint("robot_body_joint").id
    body_dof = model.jnt_dofadr[body_joint]
    motor_l = model.actuator("motor_l_wheel").id
    motor_r = model.actuator("motor_r_wheel").id
    l_joint = model.joint("torso_l_wheel").id
    l_dof = model.jnt_dofadr[l_joint]
    r_joint = model.joint("torso_r_wheel").id
    r_dof = model.jnt_dofadr[r_joint]

    # ================== 控制参数 ==================
    target_speed = 0.1
    target_yaw = 0.0
    pitch_kp = 10.0
    pitch_kd = 0.01
    speed_kp = 2.0
    speed_ki = 0.01
    WHEEL_RADIUS = 0.034
    MAX_MOTOR_TORQUE = 2.0
    INTEGRAL_LIMIT = 1.0

    # ================== PID变量 ==================
    pitch_dot_filtered = 0.0
    velocity_filtered = 0.0
    speed_error_integral = 0.0
    target_pitch = 0.0

    # ================== 时间变量 ==================
    real_time_start = time.monotonic()
    last_print_time = 0.0
    next_speed_time = 0.0
    SPEED_FREQ = 50  # 速度环50Hz
    T_SPEED = 1.0 / SPEED_FREQ

    # ================== Viewer ==================
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.5
        viewer.cam.elevation = -45
        viewer.cam.azimuth = 120

        while viewer.is_running():

            # ---------- 实时同步 ----------
            if data.time >= (time.monotonic() - real_time_start):
                time.sleep(1e-5)
                continue

            # ---------- 控制 ----------
            if data.time >= next_ctrl_time:
                # ================== 获取状态 ==================
                quat = data.xquat[body_id]
                R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])

                # 前倾为正
                pitch = -R.as_euler("xyz", degrees=False)[0]
                pitch_dot = data.qvel[body_dof + 3]
                forward_speed = data.qvel[1]

                l_vel = data.qvel[l_dof]
                r_vel = data.qvel[r_dof]

                wheel_vel = (-l_vel + r_vel) * 0.5

                # ================== 状态滤波 ==================
                pitch_dot_filtered = (0.975 * pitch_dot_filtered + 0.025 * pitch_dot)
                velocity_filtered = (0.975 * velocity_filtered + 0.025 * wheel_vel)

                # ================== 外环（速度 PI）==================
                vel_error = - (target_speed - velocity_filtered * WHEEL_RADIUS)
                speed_error_integral += vel_error * T_CTL
                speed_error_integral = clamp(speed_error_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                target_pitch = (speed_kp * vel_error + speed_ki * speed_error_integral)

                MAX_PITCH = np.deg2rad(5)
                target_pitch = clamp(target_pitch, -MAX_PITCH, MAX_PITCH)

                # ================== 内环（倾角 PD）==================
                pitch_error = target_pitch - pitch
                torque = (pitch_kp * pitch_error - pitch_kd * pitch_dot_filtered)
                torque = clamp(torque, -MAX_MOTOR_TORQUE, MAX_MOTOR_TORQUE)

                # ================== 电机输出 ==================
                left_torque = -torque + target_yaw
                right_torque = torque + target_yaw

                left_torque = clamp(left_torque, -MAX_MOTOR_TORQUE, MAX_MOTOR_TORQUE)
                right_torque = clamp(right_torque, -MAX_MOTOR_TORQUE, MAX_MOTOR_TORQUE)

                data.ctrl[motor_l] = left_torque
                data.ctrl[motor_r] = right_torque

                next_ctrl_time += T_CTL

            # ---------- 仿真 ----------
            mujoco.mj_step(model, data)

            # ---------- 相机跟随 ----------
            viewer.cam.lookat[:] = data.xpos[body_id]

            # ---------- 调试输出 ----------
            if data.time - last_print_time >= PRINT_PERIOD:
                last_print_time = data.time

                pitch_deg = pitch * 57.2958
                target_pitch_deg = target_pitch * 57.2958
                pitch_dot_deg = pitch_dot * 57.2958

                print(
                    f"[t={data.time:5.2f}s] "
                    f"Pitch={pitch_deg:6.2f}° "
                    f"PitchDot={pitch_dot_deg:7.2f}°/s "
                    f"Target_Pitch={target_pitch_deg:6.2f}° | "
                    f"Speed={wheel_vel * WHEEL_RADIUS:5.2f}/{target_speed:5.2f} m/s "
                    f"Err={vel_error:6.2f} "
                    f"LTorque={left_torque:5.2f} "
                    f"RTorque={right_torque:5.2f}"
                )

            # ---------- Viewer ----------
            viewer.sync()


if __name__ == "__main__":
    main()