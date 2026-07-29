# -*- coding: utf-8 -*-
"""
train_follow.py — 高尔夫球车跟随参数「训练」脚本
================================================
用闭环 MuJoCo 仿真评测一组控制参数，并以「距离误差 + 速度平滑度 + 俯仰波动」
的加权和作为目标函数，做分阶段随机搜索，自动找到平衡 / 速度 / 跟踪 / 底盘重心
的最优组合。

评测的闭环完全复刻 main_follow.py（CAR_FRONT_TOWARD_HUMAN=False 翻转模式）：
  行人轨迹(Pedestrian) → 速度/航向指令 → 斜坡限幅 → SegwayPID 串级控制 → 力矩

用法:
    python train_follow.py
结果写入 params_best.json，并打印 基线 vs 最优 的对比指标。
"""
import json
import time
import pathlib

import numpy as np
import mujoco

from PDseries import SegwayPID
from pedestrian import Pedestrian

CTRL_DT = 0.005
XML = str(pathlib.Path(__file__).parent.joinpath('xml/scene_follow.xml'))


# ============================================================
# 默认参数（调参起点，均为物理量级合理的初值）
# ============================================================
DEFAULTS = {
    # —— 俯仰平衡环（com_y=-0.12 已验证最优组，冻结） ——
    'pitch_kp': 216.08, 'pitch_kd': 71.60, 'pitch_ki': 31.78,
    'speed_kp': 0.288, 'speed_ki': 0.0, 'speed_kd': 0.0255,
    'yaw_kp': 295.82, 'yaw_kd': 168.79,
    'bal_alpha': 0.06,
    # —— 底盘重心（用户确认固定 -0.12；com_z 不再动） ——
    'com_y': -0.12,
    'com_z': 0.0,
    # —— 跟随距离环（当前 pedestrian 默认，作为搜索起点） ——
    'fp_kp': 0.443, 'fp_ki': 0.222, 'fp_kd': 0.0528,
    'ff_gain': 1.061, 'v_max': 2.755,
    # —— 距离自适应斜坡（连续三段：正常温和 / 近线快刹 / 拉开快追） ——
    'ramp_gentle': 0.5,   # 正常跟随时的加减速率（丝滑核心）
    'ramp_catch': 0.5,    # 追赶加速速率（实测必须≈gentle：加快追赶会激起 -9° 倾角车的极限环）
    'gate_far': 0.3,      # dist 超过 (3.0+gate_far) 后开始提升追赶速率
    'gate_brake': 6.0,    # 刹车速率提升门（越大→刹车整体越快；6.0≈全程较快，实测最稳）
    'approach_a': 1.6,    # 接近制动曲线减速度（越小越早收速度）
    'yaw_acc': 0.706,
}

# 跟踪 + 斜坡相关参数搜索范围（平衡增益与重心已冻结为 -0.12 最优组，不再搜索）
RANGE_FOLLOW = {
    'fp_kp': [0.25, 1.20], 'fp_ki': [0.0, 0.35], 'fp_kd': [0.0, 0.50],
    'ff_gain': [0.85, 1.15], 'v_max': [2.2, 3.0],
    'ramp_gentle': [0.30, 0.90], 'ramp_catch': [0.30, 0.90],
    'gate_far': [0.15, 1.00], 'gate_brake': [2.40, 6.00],
    'approach_a': [0.60, 2.00], 'yaw_acc': [0.30, 1.00],
}


def apply_params(p):
    """把参数写进 PDseries 模块全局量与 Pedestrian 类属性"""
    import PDseries as P
    P.PITCH_KP = p['pitch_kp']
    P.PITCH_KD = p['pitch_kd']
    P.PITCH_KI = p['pitch_ki']
    P.SPEED_KP = p['speed_kp']
    P.SPEED_KI = p['speed_ki']
    P.SPEED_KD = p['speed_kd']
    P.YAW_KP = p['yaw_kp']
    P.YAW_KD = p['yaw_kd']
    P.BALANCE_ALPHA = p['bal_alpha']
    Pedestrian.KP = p['fp_kp']
    Pedestrian.KI = p['fp_ki']
    Pedestrian.KD = p['fp_kd']
    Pedestrian.FF_GAIN = p['ff_gain']
    Pedestrian.V_MAX = p['v_max']
    Pedestrian.APPROACH_A = p.get('approach_a', Pedestrian.APPROACH_A)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def run_once(p, T):
    """跑 T 秒闭环仿真，返回指标 dict（含 cost）"""
    model = mujoco.MjModel.from_xml_path(XML)
    # 底盘重心微调（y=后移量，z=整体重心高度，z 越负重心越低 → 稳定俯仰角越大）
    cid = model.body('chassis').id
    model.body_ipos[cid, 1] = p['com_y']
    model.body_ipos[cid, 2] = p.get('com_z', 0.0)
    # 球包质量变化（客户装/卸球包，0~10kg；惯量按质量等比缩放，0 时给极小值防奇异）
    if 'bag_mass' in p:
        bid = model.body('golf_bag').id
        bm = max(float(p['bag_mass']), 0.01)
        model.body_mass[bid] = bm
        model.body_inertia[bid] = np.array([1.10, 1.10, 0.10]) * max(bm / 10.0, 0.01)
    data = mujoco.MjData(model)
    steps_per_ctrl = int(round(CTRL_DT / model.opt.timestep))

    robot = SegwayPID(model, data)
    ped = Pedestrian(model, data, dt=CTRL_DT)
    body_id = model.body('segway').id
    free_adr = model.jnt_qposadr[model.joint('segway_free').id]
    l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
    r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]

    init_yaw = np.pi / 2  # CAR_FRONT_TOWARD_HUMAN=False

    def setpose(th):
        data.qpos[free_adr + 3:free_adr + 7] = [np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)]

    setpose(init_yaw)
    mujoco.mj_forward(model, data)
    robot.reset()
    setpose(init_yaw)
    robot.set_yaw(init_yaw)

    speed_ref = 0.0
    yaw_ref = init_yaw
    yaw_acc = p.get('yaw_acc', 0.706)
    ramp_gentle = p.get('ramp_gentle', 0.5)
    ramp_catch = max(p.get('ramp_catch', 1.2), ramp_gentle)
    gate_far = p.get('gate_far', 0.3)
    gate_brake = max(p.get('gate_brake', 2.9), 1.6)

    n = int(T / CTRL_DT)
    rec_t, rec_dist, rec_vcmd, rec_actual, rec_pitch = [], [], [], [], []
    fell = False
    for i in range(n):
        ped.update()
        car = data.xpos[body_id][:2].copy()
        v_cmd, heading, dist = ped.follow_command(car)
        yaw_cmd = wrap(heading + np.pi / 2)
        speed_send = -v_cmd
        # 与 main_follow 一致的「距离自适应斜坡」（三段连续，无阶跃）：
        #   - 正常跟随（dist≈3m）：加减速都用温和速率 ramp_gentle → 丝滑、pitch 无明显摆动
        #   - 逼近安全线（dist<2.4m）：减速/后退速率随距离连续升到 2.5 → 守住 1.5m 硬线
        #   - 被拉开（dist>3+gate_far，含起步）：加速速率连续升到 ramp_catch → 人一走车立刻跟上
        e = speed_send - speed_ref
        if e > 0:   # 减速 / 后退方向（安全方向）
            urg = min(max((gate_brake - dist) / (gate_brake - 1.5), 0.0), 1.0)
            rate = ramp_gentle + (2.5 - ramp_gentle) * urg
        else:       # 加速接近方向
            far = min(max((dist - 3.0 - gate_far) / 1.5, 0.0), 1.0)
            rate = ramp_gentle + (ramp_catch - ramp_gentle) * far
        e = min(max(e, -rate * CTRL_DT), rate * CTRL_DT)
        speed_ref += e
        ey = np.clip(wrap(yaw_cmd - yaw_ref), -yaw_acc * CTRL_DT, yaw_acc * CTRL_DT)
        yaw_ref = wrap(yaw_ref + ey)
        robot.set_velocity_linear_set_point(speed_ref)
        robot.set_yaw(yaw_ref)
        robot.update_motor_torque()
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)
        pitch = robot.get_pitch()
        actual = (data.qvel[l_dof] + data.qvel[r_dof]) / 2.0 * 0.24
        rec_t.append(data.time)
        rec_dist.append(dist)
        rec_vcmd.append(v_cmd)
        rec_actual.append(actual)
        rec_pitch.append(pitch)
        if abs(pitch) > np.deg2rad(50):
            fell = True
            break

    rec = np.array([rec_t, rec_dist, rec_vcmd, rec_actual, rec_pitch]).T
    if len(rec) < 50 or fell or not np.all(np.isfinite(rec[:, 3])):
        return {'cost': 1e4, 'fell': fell, 'n': len(rec), 'min_dist': 0.0}

    t = rec[:, 0]
    dist = rec[:, 1]
    actual = rec[:, 3]
    pitch = rec[:, 4]
    mask = t > 2.0   # 距离统计：只跳过 2s 平衡建立期，起步追赶段计入（用户要求起步跟得上）
    pmask = t > 5.0  # pitch 统计：跳过起步倾角建立暂态
    dm = dist[mask] - 3.0
    dist_rmse = float(np.sqrt(np.mean(dm ** 2)))
    dist_bias = float(np.mean(dm))
    accel = np.abs(np.diff(actual)) / CTRL_DT
    mean_abs_accel = float(np.mean(accel))
    pitch_deg = np.degrees(pitch[pmask])
    pitch_std = float(np.std(pitch_deg))
    pitch_mean = float(np.mean(pitch_deg))
    max_abs_pitch = float(np.max(np.abs(np.degrees(pitch[pmask]))))
    min_dist = float(np.min(dist))            # 全程最近距离（防撞硬约束验证用）
    early = dist[t < 20.0]
    start_peak = float(np.max(early)) if len(early) else 3.0   # 起步被拉开的峰值距离
    safe_line = 3.0 * 0.5                     # 目标距离的一半

    # ---- 目标函数：越小越好。本轮核心诉求 = 丝滑（加减速幅度小 + pitch 摆动不可见）----
    cost = (8.0 * dist_rmse
            + 5.0 * abs(dist_bias)
            + 25.0 * mean_abs_accel                 # 加减速平滑：权重大幅加重
            + 3.0 * pitch_std
            + 4.0 * max(0.0, pitch_std - 3.0)       # pitch 摆动明显（>3°）重罚
            + 2.0 * max(0.0, max_abs_pitch - 20.0)
            + 3.0 * max(0.0, start_peak - 4.0)      # 起步跟不上（被拉开超 1m）惩罚
            # 安全裕度（硬指标）：1.5m 硬线绝不允许突破
            + 15.0 * max(0.0, 2.0 - min_dist)
            + 120.0 * max(0.0, 1.7 - min_dist)
            + 1000.0 * max(0.0, 1.5 - min_dist)
            + 5000.0 * fell)

    return {
        'cost': cost, 'fell': fell, 'n': len(rec),
        'dist_rmse': dist_rmse, 'dist_bias': dist_bias,
        'mean_abs_accel': mean_abs_accel,
        'pitch_std': pitch_std, 'pitch_mean': pitch_mean,
        'max_abs_pitch': max_abs_pitch, 'min_dist': min_dist,
        'start_peak': start_peak, 'safe_line': safe_line,
    }


def search(ranges, fixed, n, T, best=None, seed=0):
    """在 ranges 内随机搜索 n 次，fixed 为冻结参数；best 为当前全局最优"""
    rng = np.random.default_rng(seed)
    if best is None:
        best = {'cost': 1e9, 'params': dict(DEFAULTS)}
    for k in range(n):
        p = dict(DEFAULTS)
        p.update(fixed)
        for key, (lo, hi) in ranges.items():
            p[key] = float(lo + (hi - lo) * rng.random())
        apply_params(p)
        m = run_once(p, T)
        if m['cost'] < best['cost']:
            best = {'cost': m['cost'], 'params': p, 'metrics': m}
            print(f"  [new best #{k}] cost={m['cost']:.3f} "
                  f"dist_rmse={m['dist_rmse']:.3f} accel={m['mean_abs_accel']:.3f} "
                  f"pitch_std={m['pitch_std']:.2f}° maxpitch={m['max_abs_pitch']:.1f}° "
                  f"min_dist={m['min_dist']:.2f}m start_peak={m.get('start_peak', 0):.2f}m")
        elif (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{n} evals, current best cost={best['cost']:.3f}")
    return best


def local_search(center, ranges, n, T, seed=7):
    """以 center 为锚点、在范围 30% 宽度内做局部细搜"""
    rng = np.random.default_rng(seed)
    best = {'cost': 1e9, 'params': dict(center)}
    for k in range(n):
        p = dict(center)
        for key, (lo, hi) in ranges.items():
            span = 0.3 * (hi - lo)
            p[key] = float(np.clip(center[key] + span * (rng.random() * 2 - 1), lo, hi))
        apply_params(p)
        m = run_once(p, T)
        if m['cost'] < best['cost']:
            best = {'cost': m['cost'], 'params': p, 'metrics': m}
            print(f"  [local best #{k}] cost={best['cost']:.3f} "
                  f"dist_rmse={best['metrics']['dist_rmse']:.3f} "
                  f"accel={best['metrics']['mean_abs_accel']:.3f} "
                  f"pitch_std={best['metrics']['pitch_std']:.2f}°")
    return best


def evaluate_baseline(T=30):
    """用当前代码默认参数跑一次，作为对照"""
    apply_params(DEFAULTS)
    return run_once(DEFAULTS, T)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("基线（当前代码默认参数）30s 评测 ...")
    base = evaluate_baseline(30)
    print(f"  基线 cost={base['cost']:.3f}  dist_rmse={base.get('dist_rmse',0):.3f}"
          f"  accel={base.get('mean_abs_accel',0):.3f}"
          f"  pitch_std={base.get('pitch_std',0):.2f}°"
          f"  maxpitch={base.get('max_abs_pitch',0):.1f}°"
          f"  min_dist={base.get('min_dist',0):.2f}m"
          f"  com_y={DEFAULTS['com_y']}")

    # 先把当前默认（丝滑基线）作为候选之一，保证结果绝不劣于现状
    apply_params(DEFAULTS)
    m0 = run_once(DEFAULTS, 60)
    best = {'cost': m0['cost'], 'params': dict(DEFAULTS), 'metrics': m0}
    print(f"  默认参数 60s: cost={m0['cost']:.3f} rmse={m0.get('dist_rmse', 0):.3f}"
          f" accel={m0.get('mean_abs_accel', 0):.3f} pitch_std={m0.get('pitch_std', 0):.2f}°"
          f" start_peak={m0.get('start_peak', 0):.2f}m min_dist={m0.get('min_dist', 0):.2f}m")

    # ---- 阶段 A：跟踪 + 斜坡随机搜索（平衡增益与重心冻结为 -0.12 最优组） ----
    print("\n[Stage A] 跟踪 + 斜坡参数随机搜索 (80 evals, 60s) ...")
    best = search(RANGE_FOLLOW, {}, 80, 60, best=best, seed=11)
    print(f"  Stage A best cost={best['cost']:.3f}")

    # ---- 阶段 B：局部细搜 ----
    print("\n[Stage B] 局部细搜 (50 evals, 60s) ...")
    bestB = local_search(best['params'], RANGE_FOLLOW, 50, 60, seed=12)
    if bestB['cost'] < best['cost']:
        best = bestB
    print(f"  Stage B best cost={best['cost']:.3f}")

    # ---- 最终验证 120s（长时域，确认不翻车/不撞/平滑） ----
    apply_params(best['params'])
    final = run_once(best['params'], 120)
    print("\n[Final] 120s 验证:")
    print(f"  cost={final['cost']:.3f}  dist_rmse={final['dist_rmse']:.3f}"
          f"  dist_bias={final['dist_bias']:+.3f}  accel={final['mean_abs_accel']:.3f}"
          f"  pitch_std={final['pitch_std']:.2f}°  pitch_mean={final['pitch_mean']:+.2f}°"
          f"  maxpitch={final['max_abs_pitch']:.1f}°  min_dist={final['min_dist']:.2f}m"
          f"  start_peak={final['start_peak']:.2f}m  (安全线={final['safe_line']:.2f}m)")

    # 保存
    out = {
        'best_params': best['params'],
        'best_cost': best['cost'],
        'final_metrics': {k: final[k] for k in
                          ['cost', 'dist_rmse', 'dist_bias', 'mean_abs_accel',
                           'pitch_std', 'pitch_mean', 'max_abs_pitch', 'min_dist',
                           'start_peak', 'safe_line', 'fell']},
        'baseline_metrics': {k: base.get(k, None) for k in
                             ['cost', 'dist_rmse', 'dist_bias', 'mean_abs_accel',
                              'pitch_std', 'pitch_mean', 'max_abs_pitch', 'min_dist',
                              'start_peak', 'safe_line', 'fell']},
    }
    with open(pathlib.Path(__file__).parent.joinpath('params_best.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成，耗时 {time.time() - t0:.1f}s，结果已写入 params_best.json")
    print("最优参数:")
    for k, v in best['params'].items():
        print(f"  {k} = {v}")
