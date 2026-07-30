# -*- coding: utf-8 -*-
"""
main_follow.py — 双轮差速高尔夫球车 · 行人跟随仿真主程序
在原 Qt 手动遥控框架基础上接入行人跟随：
  - 行人系统全部封装在 pedestrian.Pedestrian 中（轨迹/模型/步态/跟随指令/绘图）
  - 指令链路：Pedestrian.follow_command() → speed_cmd / yaw_cmd
              → 原有斜坡限幅器 → SegwayPID 串级控制（完全复用，未改动）
  - 保留手动模式：Follow / Manual 按钮切换，Manual 时滑条生效
  - 关闭窗口后自动弹出 轨迹对比 + 距离曲线 图
构型/增益/平衡角解析统一放 config.py（主函数 import 调用）。
坐标约定：车体 body +y 为「车头」端（把手 + 球包所在侧）。
  CAR_FRONT_TOWARD_HUMAN True : 车头朝向行人（接近时车头先到）  -> yaw_cmd = heading - pi/2, 速度正向
  CAR_FRONT_TOWARD_HUMAN False: 车头背向行人、球包拖后（默认） -> yaw_cmd = heading + pi/2, 速度取反
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
import PDseries as P   # 需实时修改 BALANCE_PITCH 锚（set_bag_mass 中直接用）
from pedestrian import Pedestrian
# 构型档案 / 增益调度 / 平衡角解析（详见 config.py）
from config import (FOLLOW_PROFILE, RAMP_GENTLE, RAMP_SAFE, GATE_BRAKE, YAW_ACC,
                    equilibrium_pitch_deg, apply_follow_gains, apply_profile)

CAR_FRONT_TOWARD_HUMAN = False   # True=车头朝向行人；False=车头背向、球包拖后（默认）


def _setup_opengl_format():
    """配置默认 OpenGL 表面格式（深度/模板/多重采样）"""
    f = QSurfaceFormat()
    f.setDepthBufferSize(24); f.setStencilBufferSize(8); f.setSamples(4)
    f.setSwapInterval(1); f.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    f.setVersion(2, 0); f.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    f.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(f)


_setup_opengl_format()


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def init_yaw():
    """车头朝向 / 背向行人对应的初始绕 z 偏航角"""
    return -np.pi / 2 if CAR_FRONT_TOWARD_HUMAN else np.pi / 2


CTRL_DT = 0.005   # 控制周期 5ms（与 SegwayPID 内部积分步长一致）


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
        self.human_mocap_id = model.body_mocapid[model.body('human_main').id]   # 相机看车-人中点

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
        cart_pos = self.data.xpos[self.body_id]
        human_pos = self.data.mocap_pos[self.human_mocap_id]
        mid = 0.5 * (cart_pos + human_pos); mid[2] = 0.6   # 视线高度，相机看车-人中点
        self.cam.lookat = mid.copy()
        d = np.linalg.norm(cart_pos[:2] - human_pos[:2])
        self.cam.distance = 4.0 + 0.4 * d   # 距离随两者间距自适应
        self.cam.azimuth += 0.05   # 缓慢环绕
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
        self.ped = Pedestrian(model, data, dt=CTRL_DT)   # 行人系统（轨迹 + 模型驱动 + 跟随指令）
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
        # 控制触发改为「仿真时间」对齐（关键抖动修复）：旧逻辑用墙钟判断是否到 5ms 控制周期，
        # 而 Windows sleep/线程调度粒度 1~15ms → 周期在 5~20ms 间抖动，pitch σ 随质量恶化
        # （8.95kg: 0.8°→8.2°），即「满包 live 一抖一抖、无头评测却最平滑」根源。
        # 现每 CTRL_DT 仿真时间必触发一次控制（与无头评测一致），墙钟只负责不超实时。
        next_ctrl_time = 0.0
        while self.running:
            if self.data.time < self.real_time / 1e9:
                if next_ctrl_time > self.data.time + CTRL_DT:
                    next_ctrl_time = 0.0   # Reset 后 data.time 归零，同步控制时基
                if self.data.time >= next_ctrl_time:
                    next_ctrl_time = self.data.time + CTRL_DT
                    # ===== 行人跟随指令生成 =====
                    if self.follow_enabled:
                        self.ped.update()   # 行人走一步 + 更新模型姿态
                        car_xy = self.data.xpos[self.body_id][:2].copy()
                        v_cmd, target_heading, dist = self.ped.follow_command(car_xy)
                        if CAR_FRONT_TOWARD_HUMAN:   # 朝向/速度按约定映射（翻转 180° + 速度取反）
                            self.yaw_cmd = wrap_angle(target_heading - np.pi / 2)
                            self.speed_cmd = v_cmd
                        else:
                            self.yaw_cmd = wrap_angle(target_heading + np.pi / 2)
                            self.speed_cmd = -v_cmd
                        self._last_dist = dist
                    # ===== 斜坡限幅 + 串级 PID（原链路，未改动） =====
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
        self.yaw_cmd = init_yaw()   # yaw 初值与初始姿态一致，避免起步扭转
        self.yaw_ref = init_yaw()
        self.robot.set_yaw(self.yaw_ref)

    def set_speed(self, s):
        if not self.follow_enabled:
            self.speed_cmd = s

    def set_yaw(self, y):
        if not self.follow_enabled:
            self.yaw_cmd = y

    def update_speed_ref(self):
        dist = self._last_dist if self._last_dist > 0.0 else 3.0   # 距离自适应斜坡（连续无阶跃）：逼近安全线时减速速率连续升向 RAMP_SAFE
        error = self.speed_cmd - self.speed_ref
        if error > 0:   # 减速/后退（安全方向）：速率随距离连续增大，逼近 1.5m 硬线时达 RAMP_SAFE
            urg = clamp((GATE_BRAKE - dist) / (GATE_BRAKE - 1.5), 0.0, 1.0)
            rate = RAMP_GENTLE + (RAMP_SAFE - RAMP_GENTLE) * urg
        else:           # 加速接近方向：恒用温和速率 RAMP_GENTLE → 速度平滑、pitch 无明显摆动
            rate = RAMP_GENTLE
        error = clamp(error, -rate * CTRL_DT, rate * CTRL_DT)
        self.speed_ref += error

    def update_yaw_ref(self):
        STEP = YAW_ACC * CTRL_DT   # 最大偏航角速度 rad/s（全质量段联训最优）
        error = wrap_angle(self.yaw_cmd - self.yaw_ref)
        error = clamp(error, -STEP, STEP)
        self.yaw_ref = wrap_angle(self.yaw_ref + error)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        xml = pathlib.Path(__file__).parent / 'xml/scene_follow.xml'
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        apply_profile(self.model)   # 按构型档案覆盖平衡角锚/增益
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0, 0, 0]); self.cam.distance = 6.0
        self.cam.elevation = -25; self.cam.azimuth = 45
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, 10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout(); top = QHBoxLayout()
        reset_btn = QPushButton("Reset"); reset_btn.clicked.connect(self.reset_sim); top.addWidget(reset_btn)
        self.mode_btn = QPushButton("Mode: FOLLOW"); self.mode_btn.setCheckable(True); self.mode_btn.setChecked(True)
        self.mode_btn.clicked.connect(self.toggle_mode); top.addWidget(self.mode_btn)

        ctrl = QVBoxLayout(); ctrl.setSpacing(3); ctrl.setContentsMargins(0, 0, 0, 0)
        self.speed_slider, self.speed_value_label, srow = self._make_slider("Speed", int(-15 / 3.6 * 1000), int(15 / 3.6 * 1000), 0, self._on_speed_slider)
        self.yaw_slider, self.yaw_value_label, yrow = self._make_slider("Yaw", int(-np.deg2rad(180) * 1000), int(np.deg2rad(180) * 1000), 0, self._on_yaw_slider)
        bid0 = self.model.body('golf_bag').id; init_mass = float(self.model.body_mass[bid0])
        self.bag_slider, self.bag_value_label, brow = self._make_slider("Bag kg", 0, 1000, int(round(init_mass / 10.0 * 1000)), self._on_bag_slider)
        self.bag_value_label.setText(f"{init_mass:.2f} kg")
        ctrl.addLayout(srow); ctrl.addLayout(yrow); ctrl.addLayout(brow)
        top.addLayout(ctrl); top.setContentsMargins(8, 0, 8, 0); layout.addLayout(top)

        container = QWidget.createWindowContainer(self.viewport)   # 仿真画面容器占满剩余空间
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(container, stretch=1); layout.setContentsMargins(0, 4, 0, 0)
        w = QWidget(); w.setLayout(layout); self.setCentralWidget(w)

        self._center_on_screen()   # 适配主屏可用区域并居中（避开任务栏）
        self.th = UpdateSimThread(self.model, self.data, self); self.th.start()

    def _make_slider(self, name, lo, hi, init, slot):
        """构造 标签+滑条+数值 一行控件并连接回调（Speed/Yaw/Bag 共用）"""
        sl = QSlider(Qt.Horizontal); sl.setMinimum(lo); sl.setMaximum(hi); sl.setValue(init)
        sl.valueChanged.connect(slot)
        vl = QLabel(""); vl.setMinimumWidth(72)
        row = QHBoxLayout(); row.addWidget(QLabel(name)); row.addWidget(sl, 1); row.addWidget(vl)
        return sl, vl, row

    def _center_on_screen(self):
        """按主屏可用区域(避开任务栏)设置窗口尺寸并居中"""
        s = QGuiApplication.primaryScreen()
        if s is None:
            self.resize(1650, 850); return
        sg = s.availableGeometry()
        w, h = int(sg.width() * 0.94), int(sg.height() * 0.92)
        self.setGeometry(int(sg.x() + (sg.width() - w) / 2), int(sg.y() + (sg.height() - h) / 2), w, h)

    @Slot(float)
    def show_runtime(self, fps):
        mode = "FOLLOW" if self.th.follow_enabled else "MANUAL"
        self.statusBar().showMessage(f"[{mode}]  Avg runtime: {fps:.0e}s  Sim time: {self.data.time:.0f}s")

    def toggle_mode(self):
        self.th.follow_enabled = self.mode_btn.isChecked()
        if self.th.follow_enabled:
            self.mode_btn.setText("Mode: FOLLOW")
        else:
            self.mode_btn.setText("Mode: MANUAL")   # 切手动以当前状态为起点，避免跳变
            self.speed_slider.setValue(0); self.th.speed_cmd = 0.0
            self.th.yaw_cmd = self.th.yaw_ref

    def reset_sim(self):
        self.speed_slider.setValue(0); self.yaw_slider.setValue(0)
        self.speed_value_label.setText("0.0 km/h"); self.yaw_value_label.setText("0.0°")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()

    def set_bag_mass(self, mass_kg):
        """实时修改球包质量并同步更新解析平衡角锚（COMY0 档）；质量下限夹到 1e-3 避免零质量数值问题。"""
        bid = self.model.body('golf_bag').id
        bm = max(float(mass_kg), 0.0); model_mass = max(bm, 1e-3)
        self.model.body_mass[bid] = model_mass
        self.model.body_inertia[bid] = np.array([1.10, 1.10, 0.10]) * max(model_mass / 10.0, 0.01)
        if FOLLOW_PROFILE == 'COMY0':
            eq_deg = equilibrium_pitch_deg(self.model)
            P.BALANCE_PITCH_INIT = np.deg2rad(eq_deg)
            P.BALANCE_PITCH_MIN = np.deg2rad(eq_deg - 0.75)
            P.BALANCE_PITCH_MAX = np.deg2rad(eq_deg + 0.75)   # 窄窗锚死自适应漂移
            apply_follow_gains(model_mass)   # 跟随环增益随质量同步调度（轻包物理刹车弱→余量/安全线收紧）
            print(f"[BagMass] {bm:.2f}kg → 锚(解析平衡角)={eq_deg:+.2f}°  窗=±0.75°  跟随环已按质量调度")

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
    w.th.ped.plot()   # 关闭窗口后绘制轨迹 & 距离曲线
