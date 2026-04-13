import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def main():
    model = mujoco.MjModel.from_xml_path(r"xml\scene.xml")
    data = mujoco.MjData(model)

    # ================== 仿真 & 控制时间设置 ==================
    dt = model.opt.timestep = 0.001
    ctl_interval = 20
    T_ctl = ctl_interval * model.opt.timestep
    next_ctrl_time = 0.0

    # ================== ID 获取 ==================
    lunL_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
    lunR_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
    golf_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "golf_main")
    lunL_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lunL")
    lunL_dof_idx = model.jnt_dofadr[lunL_joint_id]
    lunR_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lunR")
    lunR_dof_idx = model.jnt_dofadr[lunR_joint_id]

    # ================== 仿真参数 ==================
    WHEEL_RADIUS = 0.0336
    MAX_TORQUE = 20.0
    start_time = time.time()
    SIM_DURATION = 1000.0
    target_vel = 1.5  # 先用低速，稳定后可提高

    # ---------- 控制器参数 ----------
    kp = 15.0
    kd = 0.01
    kv = 20.0          # 比例增益适中
    ki = 3.0           # 积分增益较小，依赖前馈
    pitch_limit = 0.05  # 限制最大倾角，防止过冲
    # 前馈力矩，根据稳态观测设定，这里给一个合理估计值（若不准可微调）
    feedforward_torque = 1.2

    # 状态变量
    vel_filtered = 0.0
    pitch_dot_filtered = 0.0
    prev_pitch = 0.0
    vel_int = 0.0
    target_pitch = 0.0

    # ================== 数据记录 ==================
    log_time, log_pitch, log_pitch_dot = [], [], []
    log_lwheel_vel, log_rwheel_vel, log_torque = [], [], []
    log_target_pitch = []

    # ================== 打开 MuJoCo 可视化窗口 ==================
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = viewer.cam
        cam.distance, cam.elevation, cam.azimuth = 0.5, -30, 40

        while viewer.is_running():
            if time.time() - start_time > SIM_DURATION:
                break

            if data.time >= next_ctrl_time:
                # ---------- 状态读取 ----------
                quat = data.body("golf_main").xquat
                R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                pitch = R.as_euler('xyz', degrees=False)[1]
                pitch_dot = (pitch - prev_pitch) / T_ctl
                prev_pitch = pitch

                vel_l = data.qvel[lunL_dof_idx] * WHEEL_RADIUS
                vel_r = data.qvel[lunR_dof_idx] * WHEEL_RADIUS
                vel = (vel_l - vel_r) / 2.0

                # 滤波
                pitch_dot_filtered = 0.9 * pitch_dot_filtered + 0.1 * pitch_dot
                vel_filtered = 0.8 * vel_filtered + 0.2 * vel  # 稍微加快速度反馈

                # ---------- 外环（速度 PI → 倾角） ----------
                vel_err = target_vel - vel_filtered
                # 动态目标倾角，并加入随速度误差减小而衰减的因子（防止超调）
                target_pitch = kv * vel_err + ki * vel_int
                # 当速度接近目标时，进一步限制目标倾角
                max_pitch = pitch_limit * min(1.0, abs(vel_err) * 5.0)
                target_pitch = np.clip(target_pitch, -max_pitch, max_pitch)

                # ---------- 内环（倾角 PD → 力矩） ----------
                torque = kp * (target_pitch - pitch) + kd * pitch_dot_filtered
                torque += feedforward_torque
                torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)

                data.ctrl[lunL_motor] = -torque
                data.ctrl[lunR_motor] = torque

                # ---------- 记录数据 ----------
                log_time.append(data.time)
                log_pitch.append(pitch)
                log_pitch_dot.append(pitch_dot_filtered)
                log_lwheel_vel.append(vel_l)
                log_rwheel_vel.append(vel_r)
                log_torque.append(torque)
                log_target_pitch.append(target_pitch)

                if data.time % 0.25 < T_ctl:
                    print(f"[t={data.time:5.2f}s] "
                          f"target_pitch={target_pitch:6.3f} | "
                          f"pitch={pitch:6.3f} | "
                          f"vel_f={vel_filtered:5.2f} / vel={vel:5.2f} | "
                          f"torque={torque:6.2f} | "
                          f"int={vel_int:6.3f}")

                next_ctrl_time += T_ctl

            cam.lookat[:] = data.xpos[golf_body]
            mujoco.mj_step(model, data)
            viewer.sync()

    # ================== 绘图 ==================
    log_vel = [(vl - vr) / 2.0 for vl, vr in zip(log_lwheel_vel, log_rwheel_vel)]

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(log_time, np.degrees(log_pitch), 'b-', linewidth=1.5)
    plt.plot(log_time, np.degrees(log_target_pitch), 'r--', linewidth=1.0, label='target pitch')
    plt.ylabel('Pitch (deg)')
    plt.grid(True)
    plt.legend()
    plt.title('Simulation Results (Stable with Feedforward)')

    plt.subplot(3, 1, 2)
    plt.plot(log_time, log_vel, 'r-', linewidth=1.5)
    plt.axhline(y=target_vel, color='k', linestyle='--', label=f'Target = {target_vel} m/s')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(log_time, log_torque, 'g-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()