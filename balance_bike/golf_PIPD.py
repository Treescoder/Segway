import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtWidgets import QApplication
import sys

from golf_ui import ControlPanel

# ================== 常量 ==================
XML_PATH = r"xml/scene.xml"
WHEEL_RADIUS = 0.0672 / 2
MAX_TORQUE = 20.0

MODEL_TIMESTEP = 0.001
CTRL_STEPS = 20                     # 20 * 0.001 = 0.02s
T_CTRL = CTRL_STEPS * MODEL_TIMESTEP

# ========== 串级 PID 参数 ==========
# 内环：角度 PD（在您稳定纯PD基础上略微提升刚度）
pitch_Kp = 35.0                     # 角度比例
pitch_Kd = 25.0                     # 角度微分
pitch_base = 0.1                    # 直立目标倾角

# 外环：速度 PI（开始用较小增益，逐渐增加）
speed_Kp = 2.5                      # 速度比例
speed_Ki = 1.1                      # 速度积分
max_speed_integral = 20.0            # 积分限幅
speed_to_pitch_gain = 0.02         # 1 m/s 偏差 ≈ 2.86° 倾角
MAX_PITCH_TARGET = np.radians(10)   # 期望倾角限幅 ±10°

# 力矩平滑（防止高频抖动）
torque_smoothing = 0.8

# 俯仰角速度低通滤波
FILTER_COEF = 0.98

def clamp(n, mn, mx):
    return max(min(mx, n), mn)

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    panel = ControlPanel()
    panel.show()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = MODEL_TIMESTEP

    body_id = model.body('golf_main').id
    l_dof = model.jnt_dofadr[model.joint('lunL').id]
    r_dof = model.jnt_dofadr[model.joint('lunR').id]
    left_actuator = model.actuator('left_motor').id
    right_actuator = model.actuator('right_motor').id

    # 内部状态
    speed_integral = 0.0
    pitch_dot_filtered = 0.0
    pitch_target_dynamic = pitch_base

    torque_last = 0.0
    next_ctrl_time = 0.0

    print("PIPD 平衡车控制器 (串级：速度PI → 角度PD)")
    print(f"内环角度: Kp={pitch_Kp}, Kd={pitch_Kd}")
    print(f"外环速度: Kp={speed_Kp}, Ki={speed_Ki}, 倾角增益={speed_to_pitch_gain}, 限幅±{np.degrees(MAX_PITCH_TARGET):.0f}°")

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = viewer.cam
        cam.distance = model.stat.extent * 2
        cam.elevation = -25
        cam.azimuth = 45

        while viewer.is_running():
            app.processEvents()
            target_speed = panel.target_speed

            if data.time >= next_ctrl_time:
                # ---- 状态获取 ----
                # 轮速
                l_ang = data.qvel[l_dof]
                r_ang = data.qvel[r_dof]
                avg_speed = (l_ang + r_ang) / 2.0 * WHEEL_RADIUS

                # 俯仰角
                quat = data.xquat[body_id].copy()
                norm = np.linalg.norm(quat)
                if norm < 1e-6:
                    quat = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    quat = quat / norm
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                euler = rot.as_euler('xyz', degrees=False)
                pitch = -euler[1]

                cvel = data.cvel[body_id]
                pitch_dot_raw = -cvel[4]
                pitch_dot_filtered = (FILTER_COEF * pitch_dot_filtered +
                                      (1 - FILTER_COEF) * pitch_dot_raw)
                pitch_dot = pitch_dot_filtered

                # ---- 外环：速度 PI → 期望倾角 ----
                speed_error = avg_speed - target_speed
                speed_integral += speed_error * T_CTRL
                speed_integral = clamp(speed_integral, -max_speed_integral, max_speed_integral)
                speed_pid = speed_Kp * speed_error + speed_Ki * speed_integral

                # 期望倾角 = 基础倾角 - 增益 * speed_pid（速度偏快→后仰减速）
                pitch_target = pitch_base - speed_to_pitch_gain * speed_pid
                pitch_target = clamp(pitch_target, -MAX_PITCH_TARGET, MAX_PITCH_TARGET)
                pitch_target_dynamic = 0.9 * pitch_target_dynamic + 0.1 * pitch_target

                # ---- 内环：角度 PD ----
                pitch_error = pitch - pitch_target
                torque_raw = pitch_Kp * pitch_error + pitch_Kd * pitch_dot
                # 力矩平滑
                torque_smoothed = torque_smoothing * torque_raw + (1 - torque_smoothing) * torque_last
                torque_last = torque_smoothed
                torque = clamp(torque_smoothed, -MAX_TORQUE, MAX_TORQUE)

                # 左右轮（暂无偏航）
                data.ctrl[left_actuator] = torque
                data.ctrl[right_actuator] = torque

                # 每 0.5 秒打印
                if data.time % 0.5 < T_CTRL:
                    print(f"[t={data.time:5.2f}s] "
                          f"轮速={avg_speed:5.2f}/{target_speed:5.2f} | "
                          f"仰角={np.degrees(pitch):5.1f}° → {np.degrees(pitch_target):5.1f}° | "
                          f"力矩={torque:7.2f} Nm | "
                          f"积分={speed_integral:5.2f}")

                next_ctrl_time += T_CTRL

            cam.lookat[:] = data.xpos[body_id]
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()