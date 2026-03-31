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
    MAX_MOTOR_VEL = 500.0  # rad/s
    start_time = time.time()
    SIM_DURATION = 50.0     # 仿真总时长（秒）

    # ================== 数据记录 ==================
    log_time = []
    log_pitch = []
    log_pitch_dot = []
    log_lwheel_vel = []
    log_rwheel_vel = []

    # PD 控制增益
    Kp = 10.0   # P 增益
    Kd = 10.0    # D 增益

    # ================== 打开 MuJoCo 被动可视化窗口 ==================
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = viewer.cam
        cam.distance = 1.5
        cam.elevation = -20
        cam.azimuth = 45

        while viewer.is_running():
            sim_time = time.time() - start_time
            if sim_time > SIM_DURATION:
                break

            if data.time >= next_ctrl_time:
                # ------------------ 获取姿态 ------------------
                quat = data.body("robot_body").xquat
                # 防止四元数为 0
                if quat[0] == 0:
                    pitch = 0.0
                else:
                    R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                    pitch = R.as_euler('xyz', degrees=False)[0]  # x 轴为 pitch

                pitch_dot = data.joint("robot_body_joint").qvel[0]  # 近似 pitch_dot

                # ------------------ PD 控制 ------------------
                control_signal = - Kp * pitch - Kd * pitch_dot

                # 限制轮子速度
                control_signal = np.clip(control_signal, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

                # 将控制信号赋给左右轮子
                data.ctrl[motor_l_wheel] = control_signal
                data.ctrl[motor_r_wheel] = -control_signal

                # ------------------ 记录数据 ------------------
                vel_l = data.joint("torso_l_wheel").qvel[0]
                vel_r = data.joint("torso_r_wheel").qvel[0]

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
                          f"[vel_r={vel_r:.2f}]")

                next_ctrl_time += T_ctl

            # 摄像机跟随车体
            cam.lookat[:] = data.xpos[robot_body] + 0.03

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()