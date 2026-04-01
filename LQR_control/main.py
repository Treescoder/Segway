import mujoco
import mujoco.viewer
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    model = mujoco.MjModel.from_xml_path(r"xml\scene.xml")
    data = mujoco.MjData(model)

    # ================== 仿真 & 控制时间设置 ==================
    model.opt.timestep = 0.001          # 仿真步长 dt = 2 ms
    ctl_interval = 1                    # 每 25 个 step 更新一次控制
    T_ctl = ctl_interval * model.opt.timestep
    next_ctrl_time = 0.0

    # ================== ID 获取 ==================
    motor_l_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_l_wheel")
    motor_r_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_r_wheel")
    robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_body")

    # ================== 仿真参数 ==================
    WHEEL_RADIUS = 0.034
    MAX_TORQUE = 20.0 # Nm
    start_time = time.time()
    SIM_DURATION = 50.0     # 仿真总时长（秒）
    kp, kd, kv = 200.0, 5.0, 0.3
    velocity_angular_filtered = pitch_dot_filtered = prev_pitch = 0.0

    # ================== 数据记录 ==================
    log_time = []
    log_pitch = []
    log_pitch_dot = []
    log_lwheel_vel = []
    log_rwheel_vel = []

    # ================== 打开 MuJoCo 被动可视化窗口 ==================
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
                # ================== 读取状态（用于打印） ==================
                quat = data.body("robot_body").xquat
                R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                pitch = R.as_euler('xyz', degrees=False)[0]
                pitch_dot = (pitch - prev_pitch) / T_ctl
                prev_pitch = pitch

                vel_l = data.joint("torso_l_wheel").qvel[0] * WHEEL_RADIUS # 这个求的是角速度
                vel_r = data.joint("torso_r_wheel").qvel[0] * WHEEL_RADIUS
                vel = (vel_l * -1 + vel_r) / 2.0
                #
                pitch_dot_filtered = (pitch_dot_filtered * .975) + (pitch_dot * .025)
                # velocity_angular_filtered = (velocity_angular_filtered * .975) + (vel * .025)
                # velocity_linear_error = 0.5 - velocity_angular_filtered * WHEEL_RADIUS

                torque = - kp * pitch - kd * pitch_dot_filtered + kv * (0 - vel)
                torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)

                data.ctrl[motor_l_wheel] = torque
                data.ctrl[motor_r_wheel] = -torque

                log_time.append(data.time)
                log_pitch.append(pitch)
                log_pitch_dot.append(pitch_dot)
                log_lwheel_vel.append(vel_l)
                log_rwheel_vel.append(vel_r)

                # ------------------ 实时打印 ------------------
                if data.time % 5 < T_ctl:
                    print(f"[t={sim_time:5.2f}s] "
                          f"[pitch={np.degrees(pitch):.2f}] "
                          f"[pitch_dot={np.degrees(pitch_dot):.2f}] "
                          f"[vel_l={vel_l:.2f}] "
                          f"[vel_r={vel_r:.2f}]"
                          f"[torque={torque:.2f}]")

                next_ctrl_time += T_ctl

            # 摄像机跟随车体
            cam.lookat[:] = data.xpos[robot_body] + 0.03

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()