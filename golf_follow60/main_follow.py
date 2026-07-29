# -*- coding: utf-8 -*-
"""
main_follow.py — 双轮差速高尔夫球车 · 行人跟随仿真主程序
=========================================================
在原 Qt 手动遥控框架基础上接入行人跟随：
  - 行人系统全部封装在 pedestrian.Pedestrian 中（轨迹/模型/步态/跟随指令/绘图）
  - 指令链路：Pedestrian.follow_command() → speed_cmd / yaw_cmd
              → 原有斜坡限幅器 → SegwayPID 串级控制（完全复用，未改动）
  - 保留手动模式：Follow / Manual 按钮切换，Manual 时滑条生效
  - 关闭窗口后自动弹出 轨迹对比 + 距离曲线 图

坐标约定：
  车体 body +y 为「车头」端（把手 + 球包所在侧）。
  跟随时目标航向 target_heading 指向行人，按 CAR_FRONT_TOWARD_HUMAN 选择：
    True  : 车头朝向行人（接近时车头先到）  -> yaw_cmd = heading - pi/2, 速度正向
    False : 车头背向行人、球包拖在后面（即「反方向行走为正」）
            -> yaw_cmd = heading + pi/2, 速度取反
  初始化车体绕 z 旋转 ±90°，使车头朝向 / 背向行人起始方向。
"""

from collections import deque
import time
import pathlib

import mujoco
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QLabel, QSizePolicy
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from scipy.spatial.transform import Rotation

from PDseries import SegwayPID
import PDseries as P   # 需实时修改 BALANCE_PITCH 锚（随球包质量调度）
from pedestrian import Pedestrian

# 车体朝向约定（差速车 body +y 为「车头」：把手与球包所在端）
#   True  : 车头朝向行人（接近时车头先到）
#   False : 车头背向行人、球包拖在后面（即「反方向行走为正」，默认）
CAR_FRONT_TOWARD_HUMAN = False

# ==================== 构型档案（一行切换） ====================
#   'COM012' : 底盘重心 -0.12（现状默认）。平衡角约 -9°，需满包 10kg 配平，
#              球包 <6kg 时平衡角剧烈变化（0kg 无平衡点）。
#   'COMY0'  : 底盘重心 0（不做后置配平）。整车重心方向恒指向球包（前上方），
#              锚 = 解析平衡角 ±0.75° 随球包质量自动调度（0~10kg 任意质量正确）。
#              params_fullmass2 六质量点(1.5~10kg)联训参数，120s 验证：
#              1.5kg σ3.7°/rmse0.44 → 3.1kg σ1.9°/rmse0.17 → 10kg σ0.57°/rmse0.08，
#              accel 全段 0.03~0.05、min_dist≥2.2m、不翻车。
#              ⚠ 物理边界：<1.5kg 摆长退化（0kg 质心几乎在轮轴上，无有效平衡点），
#              1kg 勉强可用（σ5°），0kg 不可用——实车轻载时应拒绝跟随或加配重。
FOLLOW_PROFILE = 'COMY0'
BAG_MASS = None   # None=用 XML 默认(10kg)；COMY0 档支持 1.5~10（kg）


def equilibrium_pitch_deg(model):
    """解析计算摆体(除车轮)的平衡俯仰角（°）：
    直立姿态下摆体合成质心相对轮轴中心的 (前向y, 垂直z) → θ_eq = atan2(y, z)。
    对任意球包质量 / 重心配置自动正确（实车对应：按载荷标定或用轴距力传感估计）。"""
    data_tmp = mujoco.MjData(model)
    mujoco.mj_forward(model, data_tmp)
    M, com = 0.0, np.zeros(3)
    for nm in ('chassis', 'handlebar', 'bag_mount', 'golf_bag'):
        b = model.body(nm).id
        m = model.body_mass[b]
        com += m * data_tmp.xipos[b]
        M += m
    com /= M
    axle = 0.5 * (data_tmp.xpos[model.body('l_wheel').id]
                  + data_tmp.xpos[model.body('r_wheel').id])
    rel = com - axle
    return float(np.degrees(np.arctan2(rel[1], rel[2])))


def apply_profile(model):
    """按构型档案覆盖 底盘重心 / 平衡角锚 / 控制增益（COM012 档全部保持文件默认，不动）"""
    import PDseries as P
    from pedestrian import Pedestrian as Ped
    if BAG_MASS is not None:
        bid = model.body('golf_bag').id
        bm = max(float(BAG_MASS), 0.01)
        model.body_mass[bid] = bm
        model.body_inertia[bid] = np.array([1.10, 1.10, 0.10]) * max(bm / 10.0, 0.01)
    if FOLLOW_PROFILE != 'COMY0':
        return
    # --- 底盘重心归零（XML 里是 -0.12，运行时覆盖） ---
    cid = model.body('chassis').id
    model.body_ipos[cid, 1] = 0.0
    # --- 平衡角锚 = 解析平衡角 ± 0.75° 死锁窗（随球包质量自动调度，2026-07-28） ---
    #   固定 28.7° 锚只对 ≥3kg 成立；轻包时平衡角快速下滑（2kg:27.6°, 1.5kg:27.0°,
    #   1kg:25.9°, 0kg:1.9°），错锚会导致轻包段失稳甚至撞人。现按当前模型现场解析
    #   计算，0~10kg 任意质量锚都正确。窗必须保持窄（±0.75°）：宽窗会让自适应正
    #   反馈漂移发散（历史教训）。
    eq_deg = equilibrium_pitch_deg(model)
    P.BALANCE_PITCH_INIT = np.deg2rad(eq_deg)
    P.BALANCE_PITCH_MIN = np.deg2rad(eq_deg - 0.75)
    P.BALANCE_PITCH_MAX = np.deg2rad(eq_deg + 0.75)
    print(f"[profile] COMY0: bag={model.body_mass[model.body('golf_bag').id]:.2f}kg "
          f"锚(解析平衡角)={eq_deg:+.2f}°  窗=±0.75°")
    # --- 平衡/偏航增益（params_fullmass2.json，1.5~10kg 六质量点联合训练 2026-07-28） ---
    P.PITCH_KP, P.PITCH_KD, P.PITCH_KI = 270.95, 47.47, 154.03
    P.SPEED_KP, P.SPEED_KI, P.SPEED_KD = 0.2133, 0.0, 0.00877
    P.YAW_KP, P.YAW_KD = 120.0, 126.21
    P.BALANCE_ALPHA = 0.004444
    # --- 跟随环 ---
    Ped.KP, Ped.KI, Ped.KD = 0.9095, 0.2184, 0.0411
    Ped.FF_GAIN, Ped.V_MAX, Ped.APPROACH_A = 1.20, 2.3284, 0.6214
    # --- 斜坡 ---
    UpdateSimThread.RAMP_GENTLE = 1.2044
    UpdateSimThread.GATE_BRAKE = 2.7694

def init_yaw():
    """车头朝向 / 背向行人对应的初始绕 z 偏航角"""
    return -np.pi / 2 if CAR_FRONT_TOWARD_HUMAN else np.pi / 2

CTRL_DT = 0.005  # 控制周期 5ms（与 SegwayPID 内部积分步长一致）

format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2, 0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


class Viewport(QOpenGLWindow):
    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn):
        super().__init__()
        self.model, self.data, self.cam, self.opt, self.scn = model, data, cam, opt, scn
        self.width = self.height = 0
        self.scale = 1.0
        self.__last_pos = None
        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.body_id = model.body('segway').id
        # 行人 mocap id：相机看车-人中点
        self.human_mocap_id = model.body_mocapid[model.body('human_main').id]

    def mousePressEvent(self, e):
        self.__last_pos = e.position()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.RightButton:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif e.buttons() & Qt.MouseButton.LeftButton:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif e.buttons() & Qt.MouseButton.MiddleButton:
            act = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        p = e.position()
        dx, dy = p.x() - self.__last_pos.x(), p.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, act, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = p

    def wheelEvent(self, e):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0, -0.0005 * e.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width, self.height = w, h

    def setScreenScale(self, f):
        self.scale = f

    def paintGL(self):
        # ---- 相机跟随车-人中点，距离随两者间距自适应 ----
        cart_pos = self.data.xpos[self.body_id]
        human_pos = self.data.mocap_pos[self.human_mocap_id]
        mid = 0.5 * (cart_pos + human_pos)
        mid[2] = 0.6  # 视线高度
        self.cam.lookat = mid.copy()
        d = np.linalg.norm(cart_pos[:2] - human_pos[:2])
        self.cam.distance = 4.0 + 0.4 * d
        self.cam.azimuth += 0.05  # 缓慢环绕
        self.cam.elevation = clamp(self.cam.elevation, -90, -10)

        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        screen = self.screen()
        if screen is not None:
            self.scale = screen.devicePixelRatio()
        vp = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(vp, self.scn, self.con)
        self.runtime.append(time.time() - t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):
    def __init__(self, model, data, parent=None):
        super().__init__(parent)
        self.model, self.data = model, data
        self.running = True
        self.robot = SegwayPID(model, data)
        # ★ 行人系统（轨迹 + 模型驱动 + 跟随指令）
        self.ped = Pedestrian(model, data, dt=CTRL_DT)
        self.follow_enabled = True   # True: 行人跟随 / False: 手动滑条

        self.speed_cmd = 0.0   # 指令源（跟随器或 UI）
        self.speed_ref = 0.0   # 斜坡限幅后 → 送 PID
        self.yaw_cmd = 0.0
        self.yaw_ref = 0.0
        self.body_id = model.body('segway').id
        self.l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]
        self.free_qposadr = model.jnt_qposadr[model.joint('segway_free').id]
        self.reset()
        self.last_print_time = 0.0
        self._last_dist = 0.0

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def run(self):
        # ★ 控制触发改为「仿真时间」对齐（关键抖动修复，2026-07-28）：
        #   旧逻辑用墙钟 time.monotonic_ns() 判断是否到 5ms 控制周期，而 Windows 的
        #   sleep/线程调度粒度实际 1~15ms → 控制周期在 5~20ms 间随机抖动。实验证明
        #   周期抖动下 pitch σ 随球包质量增大而恶化（8.95kg: 0.8°→8.2°），这正是
        #   「满包时 live 一抖一抖、无头评测却最平滑」的根源。
        #   现在每 CTRL_DT 仿真时间必触发一次控制（与 train_follow 无头评测严格一致），
        #   墙钟只负责整体不超实时，调度延迟不再改变控制时序。
        next_ctrl_time = 0.0
        while self.running:
            if self.data.time < self.real_time / 1e9:
                if next_ctrl_time > self.data.time + CTRL_DT:
                    next_ctrl_time = 0.0   # Reset 后 data.time 归零，同步控制时基
                if self.data.time >= next_ctrl_time:
                    next_ctrl_time = self.data.time + CTRL_DT

                    # ========== 行人跟随指令生成 ==========
                    if self.follow_enabled:
                        self.ped.update()  # 行人走一步 + 更新模型姿态
                        car_xy = self.data.xpos[self.body_id][:2].copy()
                        v_cmd, target_heading, dist = self.ped.follow_command(car_xy)
                        # 朝向 / 速度按 CAR_FRONT_TOWARD_HUMAN 映射（翻转 180° + 速度取反）
                        if CAR_FRONT_TOWARD_HUMAN:
                            self.yaw_cmd = wrap_angle(target_heading - np.pi / 2)
                            self.speed_cmd = v_cmd
                        else:
                            self.yaw_cmd = wrap_angle(target_heading + np.pi / 2)
                            self.speed_cmd = -v_cmd
                        self._last_dist = dist

                    # ========== 斜坡限幅 + 串级 PID（原链路，未改动） ==========
                    self.update_speed_ref()
                    self.update_yaw_ref()
                    self.robot.set_velocity_linear_set_point(self.speed_ref)
                    self.robot.set_yaw(self.yaw_ref)
                    self.robot.update_motor_torque()

                mujoco.mj_step(self.model, self.data)
                if self.data.time - self.last_print_time >= 0.5:
                    self.last_print_time = self.data.time
                    self._print_debug_info()
            else:
                time.sleep(0.00001)

    def _print_debug_info(self):
        try:
            quat = self.data.xquat[self.body_id]
            rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            euler = rot.as_euler('xyz', degrees=True)
            pitch, yaw = euler[0], euler[2]
            l_vel, r_vel = self.data.qvel[self.l_dof], self.data.qvel[self.r_dof]
            actual_speed = (l_vel + r_vel) / 2 * 0.24
            l_ctrl, r_ctrl = self.data.ctrl[0], self.data.ctrl[1]
            mode = "FOLLOW" if self.follow_enabled else "MANUAL"
            print(f"[t={self.data.time:6.2f}s|{mode}] "
                  f"dist={self._last_dist:5.2f} m | "
                  f"pitch={pitch:6.2f}° | "
                  f"yaw={yaw:7.2f}° yaw_ref={np.degrees(self.yaw_ref):7.2f}° | "
                  f"v={actual_speed:5.2f} v_cmd={self.speed_cmd:5.2f} v_ref={self.speed_ref:5.2f} m/s | "
                  f"ctrl=({l_ctrl:6.2f},{r_ctrl:6.2f})")
        except Exception as e:
            print(f"Debug print error: {e}")

    def stop(self):
        self.running = False
        self.wait()

    def _set_init_pose(self):
        """车体绕 z 旋转 ±90°：车头朝向 / 背向行人起始方向"""
        adr = self.free_qposadr
        theta = init_yaw()
        self.data.qpos[adr + 3:adr + 7] = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()
        self.ped.reset()          # 行人回到 (3, 0)，PID/日志清零
        self._set_init_pose()     # 车头朝向行人
        self.last_print_time = 0.0
        self.speed_cmd = 0
        self.speed_ref = 0
        # yaw 初值与初始姿态一致，避免起步扭转
        self.yaw_cmd = init_yaw()
        self.yaw_ref = init_yaw()
        self.robot.set_yaw(self.yaw_ref)

    def set_speed(self, s):
        if not self.follow_enabled:
            self.speed_cmd = s

    def set_yaw(self, y):
        if not self.follow_enabled:
            self.yaw_cmd = y

    # ---- 距离自适应斜坡参数（2026-07-28 训练最优，与 train_follow.py 评测一致） ----
    RAMP_GENTLE = 0.7864  # 正常跟随的加减速率 m/s²（丝滑核心，加速方向恒用此值）
    RAMP_SAFE = 2.5       # 逼近安全线时刹车/后退的最大速率 m/s²（守住 1.5m 硬线）
    GATE_BRAKE = 3.1667   # dist 低于此值后，减速速率由 GENTLE 连续升向 RAMP_SAFE（1.5m 处满速率）

    def update_speed_ref(self):
        # ★ 距离自适应斜坡（连续，无阶跃）：
        #   - 加速接近方向：恒用温和速率 RAMP_GENTLE → 速度变化平滑、pitch 无明显摆动
        #     （实测不能加快：-9° 前倾配置下快追→急刹会激起平衡环极限环甚至冲撞）
        #   - 减速/后退方向（安全方向）：速率随距离连续增大，逼近 1.5m 硬线时达 RAMP_SAFE，
        #     确保永不突破「目标距离一半」的安全要求
        dist = self._last_dist if self._last_dist > 0.0 else 3.0
        error = self.speed_cmd - self.speed_ref
        if error > 0:   # 减速 / 后退方向（安全方向）
            urg = clamp((self.GATE_BRAKE - dist) / (self.GATE_BRAKE - 1.5), 0.0, 1.0)
            rate = self.RAMP_GENTLE + (self.RAMP_SAFE - self.RAMP_GENTLE) * urg
        else:           # 加速接近方向
            rate = self.RAMP_GENTLE
        error = clamp(error, -rate * CTRL_DT, rate * CTRL_DT)
        self.speed_ref += error

    def update_yaw_ref(self):
        ACC = 0.6245  # 最大偏航角速度 rad/s（params_fullmass2 全质量段联训最优）
        STEP = ACC * CTRL_DT
        error = wrap_angle(self.yaw_cmd - self.yaw_ref)
        error = clamp(error, -STEP, STEP)
        self.yaw_ref = wrap_angle(self.yaw_ref + error)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        xml_path = pathlib.Path(__file__).parent.joinpath('xml/scene_follow.xml')
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        apply_profile(self.model)   # ★ 按构型档案覆盖重心/锚/增益（COM012 档为空操作）
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0, 0, 0])
        self.cam.distance = 6.0
        self.cam.elevation = -25
        self.cam.azimuth = 45
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, 10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout()
        top = QHBoxLayout()

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_sim)
        top.addWidget(reset_btn)

        # Follow / Manual 切换
        self.mode_btn = QPushButton("Mode: FOLLOW")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setChecked(True)
        self.mode_btn.clicked.connect(self.toggle_mode)
        top.addWidget(self.mode_btn)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(3)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        # Speed 滑块（Manual 模式用）
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(int(-15 / 3.6 * 1000))
        self.speed_slider.setMaximum(int(15 / 3.6 * 1000))
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self._on_speed_slider)
        self.speed_value_label = QLabel("0.0 km/h")
        self.speed_value_label.setMinimumWidth(72)
        speed_layout.addWidget(QLabel("Speed"))
        speed_layout.addWidget(self.speed_slider, 1)
        speed_layout.addWidget(self.speed_value_label)
        # Yaw 滑块（Manual 模式用）
        yaw_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(int(-np.deg2rad(180) * 1000))
        self.yaw_slider.setMaximum(int(np.deg2rad(180) * 1000))
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self._on_yaw_slider)
        self.yaw_value_label = QLabel("0.0°")
        self.yaw_value_label.setMinimumWidth(60)
        yaw_layout.addWidget(QLabel("Yaw"))
        yaw_layout.addWidget(self.yaw_slider, 1)
        yaw_layout.addWidget(self.yaw_value_label)
        ctrl_layout.addLayout(speed_layout)
        ctrl_layout.addLayout(yaw_layout)

        # ---- 球包质量滑块（FOLLOW / Manual 模式均可随时修改，0~10kg，下方再加一行） ----
        bag_layout = QHBoxLayout()
        self.bag_slider = QSlider(Qt.Horizontal)
        self.bag_slider.setMinimum(0)
        self.bag_slider.setMaximum(1000)   # 0..1000 → 0.00..10.00 kg（0.01kg 步进）
        bid0 = self.model.body('golf_bag').id
        init_mass = float(self.model.body_mass[bid0])
        self.bag_slider.setValue(int(round(init_mass / 10.0 * 1000)))
        self.bag_slider.valueChanged.connect(self._on_bag_slider)
        self.bag_value_label = QLabel(f"{init_mass:.2f} kg")
        self.bag_value_label.setMinimumWidth(62)
        bag_layout.addWidget(QLabel("Bag kg"))
        bag_layout.addWidget(self.bag_slider, 1)
        bag_layout.addWidget(self.bag_value_label)
        ctrl_layout.addLayout(bag_layout)

        top.addLayout(ctrl_layout)
        top.setContentsMargins(8, 0, 8, 0)
        layout.addLayout(top)
        # ★ 仿真画面容器设为可扩展并占满剩余空间，避免顶栏控制条挤压仿真窗口
        container = QWidget.createWindowContainer(self.viewport)
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(container, stretch=1)
        layout.setContentsMargins(0, 4, 0, 0)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        # ---- 适配显示器：自动按主屏可用区域（避开任务栏）设置窗口尺寸并居中 ----
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            sg = screen.availableGeometry()
            win_w = int(sg.width() * 0.94)
            win_h = int(sg.height() * 0.92)
            win_x = int(sg.x() + (sg.width() - win_w) / 2)
            win_y = int(sg.y() + (sg.height() - win_h) / 2)
            self.setGeometry(win_x, win_y, win_w, win_h)
        else:
            self.resize(1650, 850)

        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps):
        mode = "FOLLOW" if self.th.follow_enabled else "MANUAL"
        self.statusBar().showMessage(
            f"[{mode}]  Avg runtime: {fps:.0e}s  Sim time: {self.data.time:.0f}s")

    def toggle_mode(self):
        self.th.follow_enabled = self.mode_btn.isChecked()
        if self.th.follow_enabled:
            self.mode_btn.setText("Mode: FOLLOW")
        else:
            self.mode_btn.setText("Mode: MANUAL")
            # 切手动时以当前状态为起点，避免跳变
            self.speed_slider.setValue(0)
            self.th.speed_cmd = 0.0
            self.th.yaw_cmd = self.th.yaw_ref

    def reset_sim(self):
        self.speed_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.speed_value_label.setText("0.0 km/h")
        self.yaw_value_label.setText("0.0°")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()

    # ---- 球包质量实时滑块（0~10kg，仿真运行中随时可改） ----
    def set_bag_mass(self, mass_kg):
        """实时修改球包质量并同步更新解析平衡角锚（COMY0 档）。
        mass_kg 为滑条显示值；模型质量下限夹到 1e-3 避免零质量数值问题。"""
        bid = self.model.body('golf_bag').id
        bm = max(float(mass_kg), 0.0)
        model_mass = max(bm, 1e-3)
        self.model.body_mass[bid] = model_mass
        self.model.body_inertia[bid] = np.array([1.10, 1.10, 0.10]) * max(model_mass / 10.0, 0.01)
        if FOLLOW_PROFILE == 'COMY0':
            eq_deg = equilibrium_pitch_deg(self.model)
            P.BALANCE_PITCH_INIT = np.deg2rad(eq_deg)
            P.BALANCE_PITCH_MIN = np.deg2rad(eq_deg - 0.75)
            P.BALANCE_PITCH_MAX = np.deg2rad(eq_deg + 0.75)
            print(f"[BagMass] {bm:.2f}kg → 锚(解析平衡角)={eq_deg:+.2f}°  窗=±0.75°")

    def _on_speed_slider(self, v):
        self.speed_value_label.setText(f"{(v / 1000) * 3.6:.1f} km/h")
        self.th.set_speed(v / 1000)

    def _on_yaw_slider(self, v):
        self.yaw_value_label.setText(f"{-v / 1000 * 180.0 / np.pi:.1f}°")
        self.th.set_yaw(-v / 1000)

    def _on_bag_slider(self, v):
        mass = v / 100.0   # 0..1000 → 0.00..10.00 kg
        self.bag_value_label.setText(f"{mass:.2f} kg")
        self.set_bag_mass(mass)


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
    # 关闭窗口后绘制轨迹 & 距离曲线
    w.th.ped.plot()
