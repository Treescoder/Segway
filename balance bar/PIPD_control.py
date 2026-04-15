import mujoco
import mujoco.viewer
import time
import numpy as np
from windows import move_window_to_second_screen
from scipy.spatial.transform import Rotation

def main():
    model = mujoco.MjModel.from_xml_path(r"xml\scene.xml")
    data = mujoco.MjData(model)

    # ================== 仿真 & 控制时间设置 ==================
    model.opt.timestep = 0.001          # 仿真步长 dt = 2 ms
    ctl_interval = 50                    # 每 25 个 step 更新一次控制
    T_ctl = ctl_interval * model.opt.timestep
    next_ctrl_time = 0.0

    # ================== ID 获取 ==================
    motor_l_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_l_wheel")
    motor_r_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_r_wheel")
    segway = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "segway")
    lwheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "torso_l_wheel")
    lwheel_dof_idx = model.jnt_dofadr[lwheel_joint_id]
    rwheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "torso_r_wheel")
    rwheel_dof_idx = model.jnt_dofadr[rwheel_joint_id]

    # ================== 仿真参数 ==================
    WHEEL_RADIUS = 0.1275
    MAX_TORQUE = 20.0 # Nm
    start_time = time.time()
    SIM_DURATION = 50.0     # 仿真总时长（秒）
    target_vel = 0.1 # 目标速度
    kp, kd, kv, ki = 5.0, 1.0, 20, 1.0
    vel_filtered = pitch_dot_filtered = prev_pitch = step_count = 0.0
    pitch_limit, vel_int = 0.05, 0.0
    x0 = data.qpos[0]
    v_integral = 0.0
    prev_time = data.time

    # ================== 数据记录 ==================
    log_time, log_pitch, log_pitch_dot, log_lwheel_vel, log_rwheel_vel, log_error = [], [], [], [], [], []

    # ================== 打开 MuJoCo 被动可视化窗口 ==================
    with (mujoco.viewer.launch_passive(model, data) as viewer):
        move_window_to_second_screen(10, 20, 1900, 900)
        cam, cam.distance, cam.elevation, cam.azimuth = viewer.cam, 2.0, -60, 120

        while viewer.is_running():
            if time.time() - start_time > SIM_DURATION: break

            if data.time >= next_ctrl_time:
                # ================== 读取状态（用于打印） ==================
                quat = data.body("segway").xquat
                R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                pitch = R.as_euler('xyz', degrees=False)[0]
                pitch_dot = (pitch - prev_pitch) / T_ctl
                prev_pitch = pitch
                vel_l = data.qvel[lwheel_dof_idx] * WHEEL_RADIUS # 这个求的是角速度
                vel_r = data.qvel[rwheel_dof_idx] * WHEEL_RADIUS
                vel = (vel_l + vel_r) / 2.0
                v_world = data.qvel[0:3]
                v_forward = v_world[1]  # 假设x方向前进
                # pitch_dot = data.qvel[3]
                # print(f"real={v_forward:.3f}, wheel_est={vel:.3f}") # 轮速和车速有一些偏差
                # v_integral += v_forward * dt
                # x_real = data.qpos[1] - x0
                # error = v_integral - x_real
                # log_error.append(error)
                # if int(current_time) % 1 == 0:
                #     print(f"x_real={x_real:.4f}, int_v={v_integral:.4f}, err={error:.6f}")
                # print(f"sim_time={data.time:.2f}")
                pitch_dot_filtered = (pitch_dot_filtered * .9) + (pitch_dot * .1)
                vel_filtered = (vel_filtered * .975) + (v_forward * .025)

                # ===== 外环（速度 PI → 倾角）=====
                vel_err = target_vel - v_forward
                vel_int += vel_err * T_ctl
                target_pitch = 0.0
                target_pitch += kv * vel_err + ki * vel_int
                target_pitch = np.clip(target_pitch, -pitch_limit, pitch_limit)

                # ===== 内环（倾角 PD → 力矩）=====
                torque = kp * (pitch - target_pitch) + kd * pitch_dot_filtered
                torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)

                data.ctrl[motor_l_wheel] = - torque
                data.ctrl[motor_r_wheel] = - torque

                log_time.append(data.time)
                log_pitch.append(pitch)
                log_pitch_dot.append(pitch_dot)
                log_lwheel_vel.append(vel_l)
                log_rwheel_vel.append(vel_r)

                # ------------------ 实时打印 ------------------
                if data.time % 0.25 < T_ctl:
                    print(f"[t={data.time:5.2f}s] "
                          f"[pitch={pitch:5.2f}] "
                          f"[pitch_dot={pitch_dot:5.2f}] "
                          f"[vel_l={vel_l:5.2f}/vel={vel:5.2f} "
                          f"[vel_r={vel_r:5.2f}]"
                          f"[torque={torque:5.2f}]")

                next_ctrl_time += T_ctl

            cam.lookat[:] = data.xpos[segway]
            mujoco.mj_step(model, data)
            step_count += 1
            if step_count % 15 == 0:
                time.sleep(0.01)  # 固定帧率 ~100Hz
                viewer.sync()

if __name__ == "__main__":
    main()