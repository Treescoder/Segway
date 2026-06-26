import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import time
import matplotlib.pyplot as plt

def main():
    model = mujoco.MjModel.from_xml_path(r"xml\scene.xml")
    data = mujoco.MjData(model)

    # 仿真参数
    model.opt.timestep = 0.001
    ctl_interval = 10
    T_ctl = ctl_interval * model.opt.timestep
    next_ctrl_time = 0.0
    start_time = time.time()
    SIM_DURATION = 60.0            # 仿真时长，建议覆盖多个周期

    # ID 获取
    motor_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_l_wheel")
    motor_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_r_wheel")
    robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_body")

    # 模型参数
    WHEEL_RADIUS = 0.1275
    MAX_TORQUE = 30.0
    MAX_STEER_TORQUE = 5.0

    # 目标速度 (m/s)
    target_vel = 1.0

    # ----- 平衡与速度控制器参数 (保持不变) -----
    vel_kp = 0.5
    vel_ki = 0.001
    vel_limit = 0.15
    vel_error_prev = 0.0
    vel_output = 0.0

    ang_kp = 50.0
    ang_kd = 0.01
    max_pitch_rate = 0.07

    # 状态变量
    prev_pitch = 0.0
    pitch_dot_filtered = 0.0
    alpha = 0.1
    prev_target_pitch = 0.0

    # ----- 周期性曲线转向控制参数 (正弦期望航向角) -----
    yaw_amplitude = 0.7        # 航向角幅度 (rad)，约28.6度
    yaw_frequency = 0.07        # 频率 (Hz)，即每秒摆动次数
    # 期望航向角: desired_yaw = amplitude * sin(2*pi*frequency*t)

    # 转向 PD 控制器参数
    steer_kp = 2.0             # 比例增益
    steer_kd = 0.2             # 微分增益
    prev_yaw_error = 0.0

    # ----- 数据记录 -----
    record_time = []
    record_x = []
    record_y = []
    record_yaw = []
    record_desired_yaw = []
    record_pitch = []
    record_target_pitch = []
    record_vel = []
    record_torque = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = viewer.cam
        cam.distance = 5.0
        cam.elevation = -15
        cam.azimuth = 15

        while viewer.is_running():
            sim_time = time.time() - start_time
            if sim_time > SIM_DURATION:
                break

            if data.time >= next_ctrl_time:
                # ---------- 状态读取 ----------
                quat = data.body("robot_body").xquat
                R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                euler = R.as_euler('xyz', degrees=False)
                pitch = euler[0]
                yaw   = euler[2]

                pitch_dot = (pitch - prev_pitch) / T_ctl
                prev_pitch = pitch
                pitch_dot_filtered = alpha * pitch_dot + (1 - alpha) * pitch_dot_filtered

                ang_vel_l = data.joint("torso_l_wheel").qvel[0]
                ang_vel_r = data.joint("torso_r_wheel").qvel[0]
                vel_l = ang_vel_l * WHEEL_RADIUS
                vel_r = ang_vel_r * WHEEL_RADIUS
                linear_vel = (-vel_l + vel_r) / 2.0

                # ---------- 速度环 ----------
                speed_error = target_vel - linear_vel
                delta = vel_kp * (speed_error - vel_error_prev) + vel_ki * speed_error
                vel_output += delta
                vel_output = np.clip(vel_output, -vel_limit, vel_limit)
                vel_error_prev = speed_error
                target_pitch_raw = vel_output

                delta_pitch = target_pitch_raw - prev_target_pitch
                max_delta = max_pitch_rate * T_ctl
                if abs(delta_pitch) > max_delta:
                    target_pitch = prev_target_pitch + np.sign(delta_pitch) * max_delta
                else:
                    target_pitch = target_pitch_raw
                prev_target_pitch = target_pitch

                # ---------- 角度环：同向驱动力矩 ----------
                balance_torque = ang_kp * (pitch - target_pitch) + ang_kd * pitch_dot_filtered
                balance_torque = np.clip(balance_torque, -MAX_TORQUE, MAX_TORQUE)

                # ---------- 周期性期望航向角生成 (正弦) ----------
                t = sim_time
                desired_yaw = yaw_amplitude * np.sin(2 * np.pi * yaw_frequency * t)

                # 航向角误差，归一化到 [-pi, pi]
                yaw_error = desired_yaw - yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                # PD 控制输出转向扭矩
                steer_torque = steer_kp * yaw_error + steer_kd * (yaw_error - prev_yaw_error) / T_ctl
                prev_yaw_error = yaw_error
                steer_torque = np.clip(steer_torque, -MAX_STEER_TORQUE, MAX_STEER_TORQUE)

                # ---------- 控制分配 ----------
                left_cmd_raw  = balance_torque + steer_torque
                right_cmd_raw = balance_torque - steer_torque
                data.ctrl[motor_l] = -left_cmd_raw
                data.ctrl[motor_r] =  right_cmd_raw

                # ---------- 记录数据 ----------
                pos_x = data.qpos[0]
                pos_y = data.qpos[1]
                record_time.append(sim_time)
                record_x.append(pos_x)
                record_y.append(pos_y)
                record_yaw.append(yaw)
                record_desired_yaw.append(desired_yaw)
                record_pitch.append(pitch)
                record_target_pitch.append(target_pitch)
                record_vel.append(linear_vel)
                record_torque.append(balance_torque)

                # ---------- 打印信息 ----------
                if data.time % 0.2 < T_ctl:
                    print(f"[t={sim_time:5.2f}s] "
                          f"pitch={np.degrees(pitch):5.2f}° "
                          f"vel={linear_vel:5.2f} m/s "
                          f"x={pos_x:6.2f} m, y={pos_y:6.2f} m "
                          f"des_yaw={np.degrees(desired_yaw):5.1f}° "
                          f"yaw_err={np.degrees(yaw_error):5.1f}° "
                          f"balance_torque={balance_torque:5.2f} Nm  "
                          f"left_cmd_raw={left_cmd_raw:5.2f} right_cmd_raw={right_cmd_raw:5.2f}")

                next_ctrl_time += T_ctl

            cam.lookat[:] = data.xpos[robot_body] + 0.03
            mujoco.mj_step(model, data)
            viewer.sync()

    # 仿真结束，绘制曲线
    plot_results(record_time, record_x, record_y, record_yaw, record_desired_yaw,
                 record_pitch, record_target_pitch, record_vel, record_torque)

def plot_results(t, x, y, yaw, desired_yaw, pitch, target_pitch, vel, torque):
    plt.figure(figsize=(12, 10))

    # 轨迹图
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', linewidth=1.5)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Vehicle Trajectory (Periodic Curve)')
    plt.axis('equal')
    plt.grid(True)

    # 速度曲线
    plt.subplot(2, 2, 2)
    plt.plot(t, vel, 'g-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.grid(True)

    # 俯仰角与期望俯仰角
    plt.subplot(2, 2, 3)
    plt.plot(t, np.rad2deg(pitch), 'r-', label='Actual Pitch', linewidth=1.5)
    plt.plot(t, np.rad2deg(target_pitch), 'b--', label='Target Pitch', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (deg)')
    plt.title('Pitch Angle Control')
    plt.legend()
    plt.grid(True)

    # 航向角跟踪（实际 vs 期望）
    plt.subplot(2, 2, 4)
    plt.plot(t, np.rad2deg(yaw), 'm-', label='Actual Yaw', linewidth=1.5)
    plt.plot(t, np.rad2deg(desired_yaw), 'c--', label='Desired Yaw', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle (deg)')
    plt.title('Yaw Angle Tracking (Periodic)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()