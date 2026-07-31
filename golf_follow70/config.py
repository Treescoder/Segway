# -*- coding: utf-8 -*-
"""config.py — 构型档案 / 增益调度 / 平衡角解析（主函数 from config import 调用）"""
import numpy as np
import mujoco
from pedestrian import Pedestrian
import PDseries as P

FOLLOW_PROFILE = 'COMY0'   # 底盘 COM 由 XML 定义（inertial pos），不再运行时覆盖
BAG_MASS = None            # None=用 XML 默认(10kg)；COMY0 档支持 1.5~10kg

# 距离自适应斜坡（全质量联训最优，连续无阶跃）
RAMP_GENTLE, RAMP_SAFE, GATE_BRAKE, YAW_ACC = 1.2044, 4.0, 2.7694, 0.6245

# SegwayPID 增益（params_fullmass2 六质量点联训）
PITCH_KP, PITCH_KD, PITCH_KI = 270.95, 47.47, 154.03
SPEED_KP, SPEED_KI, SPEED_KD = 0.2133, 0.08, 0.00877
YAW_KP, YAW_KD = 120.0, 126.21
BALANCE_ALPHA = 0.004444

# 跟随环基准（轻包1.5kg锚 ↔ 重包10kg锚；apply_follow_gains 按质量线性插值）
G = dict(KP=(0.30, 0.9095), KI=(0.03, 0.2184), KD=0.0411, FF=1.0,
         VMAX=(3.15, 3.5), APPROACH_A=(0.06, 0.6214),
         D_SAFE=(0.63, 0.50), REVERSE_BRAKE=3.5)


def equilibrium_pitch_deg(model):
    """解析摆体(chassis/handlebar/bag_mount/golf_bag)合成质心相对轮轴的平衡俯仰角(°)。
    投影到车体本体『侧向(Y)-竖直(Z)』平面，消除偏航依赖：在 launch_passive 已烘焙的
    yaw=±90° 初始姿态下也能算出与 yaw=0 一致的正确平衡角；旧公式只取 world Y-Z 平面，
    在偏航姿态下返回 0 → 平衡锚被清零 → 车翻倒跟丢。"""
    d = mujoco.MjData(model); mujoco.mj_forward(model, d)
    M, com = 0.0, np.zeros(3)
    for nm in ('chassis', 'handlebar', 'bag_mount', 'golf_bag'):
        b = model.body(nm).id; m = model.body_mass[b]
        com += m * d.xipos[b]; M += m
    com /= M
    axle = 0.5 * (d.xpos[model.body('l_wheel').id] + d.xpos[model.body('r_wheel').id])
    seg = model.body('segway').id
    R = d.xmat[seg].reshape(3, 3)
    off = com - axle
    yc = float(np.dot(off, R[:, 1]))   # 车体侧向分量
    zc = float(np.dot(off, R[:, 2]))   # 车体竖直分量
    return float(np.degrees(np.arctan2(yc, zc)))


def _sched(bag_kg, light, heavy, lo=1.5, hi=10.0):
    """按质量在 light(1.5kg)/heavy(10kg) 锚点间线性插值（超界夹紧）"""
    a = max(0.0, min(1.0, (float(bag_kg) - lo) / (hi - lo)))
    return light + (heavy - light) * a


def apply_follow_gains(bag_kg):
    """跟随环增益随质量调度（0~3m/s 全速域 × 1.5~10kg 无头台联合验证，COMY0 档）"""
    Pedestrian.KP = _sched(bag_kg, *G['KP'])
    Pedestrian.KI = _sched(bag_kg, *G['KI'])
    Pedestrian.KD = G['KD']; Pedestrian.FF_GAIN = G['FF']
    Pedestrian.V_MAX = _sched(bag_kg, *G['VMAX'])
    Pedestrian.APPROACH_A = _sched(bag_kg, *G['APPROACH_A'])
    Pedestrian.D_SAFE_RATIO = _sched(bag_kg, *G['D_SAFE'])
    Pedestrian.REVERSE_BRAKE = G['REVERSE_BRAKE']


def apply_profile(model):
    """按构型档案覆盖平衡角锚与增益（COMY0 档）"""
    if BAG_MASS is not None:   # 指定球包质量时改写模型质量/惯量
        bid = model.body('golf_bag').id; bm = max(float(BAG_MASS), 0.01)
        model.body_mass[bid] = bm
        model.body_inertia[bid] = np.array([1.10, 1.10, 0.10]) * max(bm / 10.0, 0.01)
    if FOLLOW_PROFILE != 'COMY0':
        return
    cid = model.body('chassis').id; model.body_ipos[cid, 1] = 0.0  # COMY0：底盘质心归零 → 平衡角+29°
    P.SPEED_KP, P.SPEED_KI, P.SPEED_KD = SPEED_KP, SPEED_KI, SPEED_KD
    eq = equilibrium_pitch_deg(model)
    P.BALANCE_PITCH_INIT = np.deg2rad(eq)
    P.BALANCE_PITCH_MIN = np.deg2rad(eq - 0.75); P.BALANCE_PITCH_MAX = np.deg2rad(eq + 0.75)  # 窄窗(±0.75°)锚死自适应漂移
    P.PITCH_KP, P.PITCH_KD, P.PITCH_KI = PITCH_KP, PITCH_KD, PITCH_KI
    P.YAW_KP, P.YAW_KD = YAW_KP, YAW_KD; P.BALANCE_ALPHA = BALANCE_ALPHA
    print(f"[profile] COMY0: bag={model.body_mass[model.body('golf_bag').id]:.2f}kg "
          f"锚(解析平衡角)={eq:+.2f}°  窗=±0.75°")
    apply_follow_gains(model.body_mass[model.body('golf_bag').id])
