# -*- coding: utf-8 -*-
"""
main_follow.py — 双轮差速高尔夫球车 · 行人跟随仿真主程序（跟随专用版）
指令链路（未改动）：
  行人 Pedestrian.follow_command() → speed_cmd/yaw_cmd → 距离自适应斜坡 → SegwayPID 串级
窗口：MuJoCo 自带 viewer（launch_passive），无需 PySide6。

本版相对上一版的变更：
  - 仅保留跟随模式（去掉 manual / M 键切换）；
  - 去掉 yaw、speed 控制滑块（cmd_yaw/cmd_speed 已从 XML 删除），跟随速度由指令链直接给出；
  - 复位改用 MuJoCo 自带 reset：把『初始姿态/偏航』烘焙进 model.qpos0，主循环检测 data.time
    归零即完整复位（物理 + 控制器 + 行人 + 指令时基），保证回到仿真开始状态；R 键已移除；
  - 终端输出：去掉 FOLLOW/MANUAL 模式标识与 v_cmd，pitch 旁新增球包质量；
  - 相机环绕减速为 viewer.cam.azimuth += 0.005。
球包质量仍可用右侧面板 cmd_bag 滑块实时调整（set_bag_mass 同步平衡角锚与跟随增益）。
"""
import time
import pathlib

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

import PDseries as P
from PDseries import SegwayPID
from pedestrian import Pedestrian
from config import (FOLLOW_PROFILE, RAMP_GENTLE, RAMP_SAFE, GATE_BRAKE, YAW_ACC,
                    equilibrium_pitch_deg, apply_follow_gains, apply_profile)

CAR_FRONT_TOWARD_HUMAN = False   # False=车头背向、球包拖后（默认）
CTRL_DT = 0.005                  # 控制周期 5ms（与 SegwayPID 内部积分步长一致）
SPEED_CAP = 5.0                  # 仅防数值发散的安全上限（非功能限幅，正常跟随远小于此）


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def init_yaw():
    """车头朝向 / 背向行人对应的初始绕 z 偏航角"""
    return -np.pi / 2 if CAR_FRONT_TOWARD_HUMAN else np.pi / 2


def _debug_info(data, body_id, l_dof, r_dof, last_dist, yaw_ref, speed_ref, bag_kg):
    """每 0.5s 打印俯仰/偏航/速度/球包质量/控制量（FOLLOW 模式标识与 v_cmd 已去掉）"""
    quat = data.xquat[body_id]
    euler = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=True)
    pitch, yaw = euler[0], euler[2]
    actual_speed = (data.qvel[l_dof] + data.qvel[r_dof]) / 2 * 0.24
    l_ctrl, r_ctrl = data.ctrl[0], data.ctrl[1]
    print(f"[t={data.time:6.2f}s] dist={last_dist:5.2f} m | "
          f"pitch={pitch:6.2f}° bag={bag_kg:5.2f}kg | "
          f"yaw={yaw:7.2f}° yaw_ref={np.degrees(yaw_ref):7.2f}° | "
          f"v={actual_speed:5.2f} v_ref={speed_ref:5.2f} m/s | "
          f"ctrl=({l_ctrl:6.2f},{r_ctrl:6.2f})")


def main():
    xml = pathlib.Path(__file__).parent / 'xml/scene_follow.xml'
    model = mujoco.MjModel.from_xml_path(str(xml))
    apply_profile(model)                       # COMY0：底盘质心归零→+29° 平衡角（覆盖模型）
    data = mujoco.MjData(model)
    bag_idx = model.actuator('cmd_bag').id      # 右侧面板球包质量滑块（零增益虚拟执行器）
    robot = SegwayPID(model, data)
    ped = Pedestrian(model, data, dt=CTRL_DT)
    body_id = model.body('segway').id
    l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
    r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]
    free_qposadr = model.jnt_qposadr[model.joint('segway_free').id]
    human_mocap_id = model.body_mocapid[model.body('human_main').id]

    bag_kg = float(model.body_mass[model.body('golf_bag').id])  # 初始球包质量（=XML 当前值）
    data.ctrl[bag_idx] = bag_kg

    # 把『初始姿态』烘焙进 model.qpos0：MuJoCo 自带 reset(mj_resetData) 据此回到开始状态
    theta0 = init_yaw()
    model.qpos0[free_qposadr:free_qposadr + 3] = [0.0, 0.0, 0.24]
    model.qpos0[free_qposadr + 3:free_qposadr + 7] = [np.cos(theta0 / 2), 0.0, 0.0, np.sin(theta0 / 2)]

    speed_cmd = speed_ref = yaw_cmd = yaw_ref = 0.0
    last_dist = 0.0
    last_print = 0.0

    def set_bag_mass(mass_kg):
        """实时改球包质量并同步解析平衡角锚（COMY0 档）；下限夹到 1e-3 避免零质量"""
        nonlocal bag_kg
        bag_kg = clamp(mass_kg, 0.0, 10.0)
        bm = max(bag_kg, 1e-3)
        bid = model.body('golf_bag').id
        model.body_mass[bid] = bm
        model.body_inertia[bid] = np.array([1.10, 1.10, 0.10]) * max(bm / 10.0, 0.01)
        if FOLLOW_PROFILE == 'COMY0':
            eq = equilibrium_pitch_deg(model)
            P.BALANCE_PITCH_INIT = np.deg2rad(eq)
            P.BALANCE_PITCH_MIN = np.deg2rad(eq - 0.75)
            P.BALANCE_PITCH_MAX = np.deg2rad(eq + 0.75)   # 窄窗(±0.75°)锚死自适应漂移
            apply_follow_gains(bm)                        # 跟随环增益随质量同步调度
            robot.balance_pitch = P.BALANCE_PITCH_INIT    # 立刻归位，避免慢速重学掉队
            robot.pitch_integral = 0.0
            print(f"[BagMass] {bag_kg:.2f}kg → 锚(解析平衡角)={eq:+.2f}°  窗=±0.75°")

    def reset_state():
        """完整复位到『仿真开始』状态（MuJoCo 自带 reset 也走这里）"""
        mujoco.mj_resetData(model, data)        # qpos←qpos0(含初始偏航)/qvel0/ctrl0/time0
        robot.reset(); ped.reset()
        data.ctrl[bag_idx] = bag_kg             # 球包滑块被 reset 清零，恢复当前质量
        theta = init_yaw()
        data.qpos[free_qposadr + 3:free_qposadr + 7] = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
        data.qpos[free_qposadr:free_qposadr + 3] = [0.0, 0.0, 0.24]
        mujoco.mj_forward(model, data)

    # 启动即处于『开始状态』
    reset_state()
    yaw_cmd = yaw_ref = init_yaw(); robot.set_yaw(yaw_ref)
    speed_cmd = speed_ref = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 6.0; viewer.cam.elevation = -25; viewer.cam.azimuth = 45
        start_ns = time.monotonic_ns()
        next_ctrl = 0.0
        prev_time = 0.0

        while viewer.is_running():
            real_t = (time.monotonic_ns() - start_ns) / 1e9
            # 检测 MuJoCo 自带 reset：mj_resetData 把 data.time 归零 → 完整复位 + 墙钟时基归零，
            # 避免 reset 后 data.time 远小于 real_t 而疯狂追帧
            if data.time < prev_time - 1e-6:
                reset_state()
                yaw_cmd = yaw_ref = init_yaw(); robot.set_yaw(yaw_ref)
                speed_cmd = speed_ref = 0.0
                next_ctrl = 0.0; last_print = 0.0
                start_ns = time.monotonic_ns()
            prev_time = data.time

            if data.time < real_t:
                if next_ctrl > data.time + CTRL_DT:
                    next_ctrl = 0.0   # Reset 后 data.time 归零，同步控制时基
                if data.time >= next_ctrl:
                    next_ctrl = data.time + CTRL_DT
                    if abs(data.ctrl[bag_idx] - bag_kg) > 0.02:
                        set_bag_mass(data.ctrl[bag_idx])   # 球包质量滑块实时调度
                    # ===== 行人跟随指令生成（仅跟随模式） =====
                    ped.update()
                    car_xy = data.xpos[body_id][:2].copy()
                    v_cmd, target_heading, dist = ped.follow_command(car_xy)
                    if CAR_FRONT_TOWARD_HUMAN:   # 朝向/速度按约定映射（翻转180°）
                        yaw_cmd = wrap_angle(target_heading - np.pi / 2)
                        speed_cmd = v_cmd
                    else:
                        yaw_cmd = wrap_angle(target_heading + np.pi / 2)
                        speed_cmd = -v_cmd
                    last_dist = dist
                    speed_cmd = clamp(speed_cmd, -SPEED_CAP, SPEED_CAP)   # 仅防发散，不截断跟随
                    # ===== 斜坡限幅 + 串级 PID（原链路，未改动） =====
                    d = last_dist if last_dist > 0.0 else 3.0   # 距离自适应斜坡（连续无阶跃）
                    err = speed_cmd - speed_ref
                    if err > 0:   # 减速/后退（安全方向）：速率随距离连续升向 RAMP_SAFE
                        urg = clamp((GATE_BRAKE - d) / (GATE_BRAKE - 1.5), 0.0, 1.0)
                        rate = RAMP_GENTLE + (RAMP_SAFE - RAMP_GENTLE) * urg
                    else:         # 加速接近：恒用温和 RAMP_GENTLE → 速度平滑、pitch 无明显摆动
                        rate = RAMP_GENTLE
                    speed_ref += clamp(err, -rate * CTRL_DT, rate * CTRL_DT)
                    step = YAW_ACC * CTRL_DT   # 最大偏航角速度 rad/s（全质量段联训最优）
                    yaw_ref = wrap_angle(yaw_ref + clamp(wrap_angle(yaw_cmd - yaw_ref), -step, step))
                    robot.set_velocity_linear_set_point(speed_ref)
                    robot.set_yaw(yaw_ref)
                    robot.update_motor_torque()
                mujoco.mj_step(model, data)
                if data.time - last_print >= 0.5:
                    last_print = data.time
                    _debug_info(data, body_id, l_dof, r_dof, last_dist, yaw_ref, speed_ref, bag_kg)
            else:
                time.sleep(1e-5)
            # 相机跟随车-人中点，距离随两者间距自适应；缓慢环绕
            cart = data.xpos[body_id]; human = data.mocap_pos[human_mocap_id]
            mid = 0.5 * (cart + human); mid[2] = 0.6
            viewer.cam.lookat[:] = mid
            viewer.cam.distance = 4.0 + 0.4 * np.linalg.norm(cart[:2] - human[:2])
            viewer.cam.elevation = clamp(viewer.cam.elevation, -90, -10)
            viewer.cam.azimuth += 0.005
            viewer.sync()
    ped.plot()   # 关闭窗口后绘制轨迹 & 距离曲线


if __name__ == "__main__":
    main()
