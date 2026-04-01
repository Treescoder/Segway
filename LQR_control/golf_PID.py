import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import time

def read_imu(data):
    quat = data.body("robot_body").xquat
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pitch = rot.as_euler('xyz', degrees=False)[0]
    rate = data.joint("robot_body_joint").qvel[0]
    return pitch, rate

def main():
    model = mujoco.MjModel.from_xml_path('xml/scene.xml')
    data = mujoco.MjData(model)
    l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_l_wheel")
    r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_r_wheel")

    # 控制器参数（关键：比例必须足够大，微分提供阻尼）
    kp = 35.0       # 比例增益，提供恢复力矩
    kd = 5.0        # 微分增益，抑制振荡
    max_torque = 8.0   # 力矩限幅，防止动作过激

    # 可选：施加一个初始扰动，观察响应（取消注释以测试）
    # data.qpos[3] = 0.1   # 自由关节的第4个关节（倾角），索引需根据实际模型确认

    filtered_rate = 0.0
    alpha = 0.1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            dt = model.opt.timestep

            pitch, rate_raw = read_imu(data)
            filtered_rate = alpha * rate_raw + (1 - alpha) * filtered_rate

            # 控制律：正倾角 → 正力矩（车轮向前），负倾角 → 负力矩（车轮向后）
            torque = kp * pitch + kd * filtered_rate
            torque = np.clip(torque, -max_torque, max_torque)

            # 左轮关节轴方向为负，指令取反
            data.ctrl[l_id] = -torque
            data.ctrl[r_id] = torque

            if data.time % 0.2 < dt:
                vel_l = data.joint("torso_l_wheel").qvel[0]
                vel_r = data.joint("torso_r_wheel").qvel[0]
                linear_vel = (-vel_l + vel_r) / 2.0 * 0.034
                print(f"t={data.time:.2f} 倾角={np.degrees(pitch):5.2f}° 力矩={torque:5.2f} "
                      f"线速度={linear_vel:5.2f} m/s")

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()