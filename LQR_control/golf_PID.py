import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import time
import math

def main():
    model = mujoco.MjModel.from_xml_path(r"xml\scene.xml")
    data = mujoco.MjData(model)

    # 仿真参数
    model.opt.timestep = 0.001
    ctl_interval = 10                  # 请尝试 50，但建议先保持 1 调好参数
    T_ctl = ctl_interval * model.opt.timestep
    next_ctrl_time = 0.0
    start_time = time.time()
    SIM_DURATION = 500.0

    # ID 获取
    motor_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_l_wheel")
    motor_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_r_wheel")
    robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_body")

    # 模型参数
    WHEEL_RADIUS = 0.1275
    MAX_TORQUE = 20.0

    # 控制参数
    target_vel = 1.5

    # 速度环（增量式 PI）
    vel_kp = 0.25
    vel_ki = 0.001
    vel_limit = 0.15                    # 期望倾角限幅（rad）
    vel_error_prev = 0.0
    vel_output = 0.0                    # 上次输出（期望倾角）

    # 角度环（位置式 PD）
    ang_kp = 20
    ang_kd = 1.5

    # 期望倾角变化率限制
    max_pitch_rate = 0.08
    prev_target_pitch = 0.0

    # 状态变量
    prev_pitch = 0.0
    pitch_dot_filtered = 0.0
    alpha = 0.1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = viewer.cam
        cam.distance = 2.2
        cam.elevation = -30
        cam.azimuth = 60

        while viewer.is_running():
            sim_time = time.time() - start_time
            if sim_time > SIM_DURATION:
                break

            if data.time >= next_ctrl_time:
                # ---- 状态读取 ----
                quat = data.body("robot_body").xquat
                R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                pitch = R.as_euler('xyz', degrees=False)[0]
                pitch_dot = (pitch - prev_pitch) / T_ctl
                prev_pitch = pitch

                ang_vel_l = data.joint("torso_l_wheel").qvel[0]
                ang_vel_r = data.joint("torso_r_wheel").qvel[0]
                vel_l = ang_vel_l * WHEEL_RADIUS
                vel_r = ang_vel_r * WHEEL_RADIUS
                linear_vel = (-vel_l + vel_r) / 2.0

                pitch_dot_filtered = alpha * pitch_dot + (1 - alpha) * pitch_dot_filtered

                # ---- 速度环（增量式 PI） ----
                speed_error = target_vel - linear_vel
                delta = vel_kp * (speed_error - vel_error_prev) + vel_ki * speed_error
                vel_output += delta
                vel_output = np.clip(vel_output, -vel_limit, vel_limit)
                vel_error_prev = speed_error
                target_pitch_raw = vel_output

                # 期望倾角变化率限制
                delta_pitch = target_pitch_raw - prev_target_pitch
                max_delta = max_pitch_rate * T_ctl
                if abs(delta_pitch) > max_delta:
                    target_pitch = prev_target_pitch + np.sign(delta_pitch) * max_delta
                else:
                    target_pitch = target_pitch_raw
                prev_target_pitch = target_pitch

                # ---- 角度环（位置式 PD） ----
                torque = ang_kp * (pitch - target_pitch) + ang_kd * pitch_dot_filtered
                torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)

                # ---- 执行 ----
                data.ctrl[motor_l] = -torque
                data.ctrl[motor_r] = torque

                # ---- 打印 ----
                if data.time % 0.2 < T_ctl:
                    print(f"[t={sim_time:5.2f}s] "
                          f"pitch={np.degrees(pitch):5.2f}° "
                          f"pitch_dot={np.degrees(pitch_dot):5.2f}°/s "
                          f"vel_l={vel_l:5.2f} vel_r={vel_r:5.2f} "
                          f"vel={linear_vel:5.2f} m/s "
                          f"target_pitch={np.degrees(target_pitch):5.2f}° "
                          f"torque={torque:5.2f}")

                next_ctrl_time += T_ctl
                if data.time % 1 < T_ctl:
                    print(f"pos_x={data.qpos[0]:.3f} m, vel={linear_vel:.2f} m/s")

            cam.lookat[:] = data.xpos[robot_body] + 0.03
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()