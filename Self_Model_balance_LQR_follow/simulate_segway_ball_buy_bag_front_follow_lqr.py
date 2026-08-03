"""
高尔夫平衡球车 —— 人类跟踪仿真平台

功能：
  - MuJoCo 物理仿真 + PySide6 OpenGL 渲染
  - LQI 平衡/速度一体化控制 + 偏航 PD
  - 人类随机行走（mocap 体，随机速度/转弯，有限加速度）
  - 跟踪控制（目标 1.90m，速度/yaw 前馈）
  - 手动/跟踪模式切换
  - 跌倒检测
  - 球包质量动态调节

操作：
  - 鼠标左键拖动：旋转视角
  - 鼠标右键拖动：平移视角
  - 鼠标滚轮：缩放
  - Tracking 按钮：切换跟踪/手动模式
  - Reset 按钮：重置仿真
  - Reset Human 按钮：重置人类位置
"""
import time
import pathlib
import mujoco
import numpy as np
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QLabel
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import QSurfaceFormat, QShortcut, QKeySequence

try:
    from .segway_ball_lqr_bag_front_follow import SegwayLQR
    from .human_controller import HumanController
    from .tracking_controller import TrackingController
except ImportError:  # 允许在 PyCharm 中直接运行当前文件
    from segway_ball_lqr_bag_front_follow import SegwayLQR
    from human_controller import HumanController
    from tracking_controller import TrackingController


CONTROL_DT = 0.005  # 200 Hz；嵌入式版本也应使用固定控制周期

# ============ OpenGL 格式 ============
fmt = QSurfaceFormat()
fmt.setDepthBufferSize(24)
fmt.setStencilBufferSize(8)
fmt.setSamples(4)
fmt.setSwapInterval(1)
fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
fmt.setVersion(2, 0)
fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(fmt)


# ============ 3D 视口 ============
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
        self.person_body_id = model.body('person').id

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
        mujoco.mjv_moveCamera(self.model, act, dx / max(self.height, 1),
                              dy / max(self.height, 1), self.scn, self.cam)
        self.__last_pos = p

    def wheelEvent(self, e):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0, -0.0005 * e.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width, self.height = w, h

    def paintGL(self):
        body_pos = self.data.xpos[self.body_id]
        person_pos = self.data.xpos[self.person_body_id]
        # 观察视角跟随人车中点，长距离行驶也不会跑出画面。
        self.cam.lookat = 0.5 * (body_pos + person_pos)
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None,
                               self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        screen = self.screen()
        if screen is not None:
            self.scale = screen.devicePixelRatio()
        vp = mujoco.MjrRect(0, 0, int(self.width * self.scale),
                            int(self.height * self.scale))
        mujoco.mjr_render(vp, self.scn, self.con)
        self.runtime.append(time.time() - t)
        self.updateRuntime.emit(np.average(self.runtime))


# ============ 仿真线程 ============
class UpdateSimThread(QThread):
    def __init__(self, model, data, parent=None):
        super().__init__(parent)
        self.model, self.data = model, data
        self.running = True
        mujoco.mj_forward(model, data)

        # 固定LQR增益 + IMU/力矩扰动观测器；控制器不读球包质量。
        self.robot = SegwayLQR(model, data)
        self.human = HumanController(model, data, start_x=0.0, start_y=-1.90)
        self.tracker = TrackingController()

        # 模式
        self.tracking_mode = False
        self.fallen = False

        # 速度/偏航
        self.speed_cmd = 0.0
        self.speed_ref = 0.0
        self.yaw_cmd = 0.0
        self.yaw_ref = 0.0

        # 球包质量只是MuJoCo仿真工况，不传入控制算法。
        self.default_bag_mass = 9.0
        self.bag_mass = self.default_bag_mass
        self.target_bag_mass = self.default_bag_mass
        self.bag_mass_rate = 1.0  # kg/s，模拟逐支取放球杆，避免质心阶跃

        # 显示用
        self.current_distance = 0.0
        self.human_speed = 0.0
        self.cart_speed = 0.0
        self.camera_visible = True
        self.camera_error_deg = 0.0

        self.reset()
        self.last_print_time = 0.0

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def run(self):
        while self.running:
            if self.data.time < self.real_time / 1e9:
                # 控制周期必须绑定 MuJoCo 仿真时间。旧版用墙钟计时，界面一慢
                # 就丢控制周期，导致显示2.78m/s的人实际只移动约0.9m/s。
                if self.data.time + 1e-12 >= self.next_control_time:
                    self.next_control_time += CONTROL_DT

                    if self.tracking_mode:
                        # 更新人类运动
                        self.human.update(CONTROL_DT)
                        # 仿真适配层：实车中 distance/bearing_error 必须来自传感器。
                        h = self.human.get_state()
                        cart_pos = self.robot.get_position()
                        cart_yaw = self.robot.get_yaw()
                        dx = h['x'] - cart_pos[0]
                        dy = h['y'] - cart_pos[1]
                        distance = np.hypot(dx, dy)
                        sight_yaw = np.arctan2(dx, -dy)
                        bearing_error = np.arctan2(
                            np.sin(sight_yaw - cart_yaw),
                            np.cos(sight_yaw - cart_yaw))
                        target_speed, target_yaw, dist = self.tracker.update(
                            distance, bearing_error,
                            h['speed'], h['yaw'],
                            self.robot.get_speed(), cart_yaw, CONTROL_DT,
                            target_valid=(
                                abs(bearing_error)
                                <= self.tracker.CAMERA_HALF_FOV
                            ))
                        # 跟踪器已经完成速度/yaw斜坡，避免主程序重复限速。
                        self.speed_cmd = self.speed_ref = target_speed
                        self.yaw_cmd = self.yaw_ref = target_yaw
                        self.current_distance = dist
                        self.human_speed = h['speed']
                        self.camera_visible = self.tracker.person_visible
                        self.camera_error_deg = np.rad2deg(
                            abs(self.tracker.camera_error))
                    else:
                        self.current_distance = 0.0
                        self.human_speed = 0.0

                    # 手动模式仍由主程序平滑；跟踪模式由可移植跟踪器完成。
                    if not self.tracking_mode:
                        self.update_speed_ref()
                        self.update_yaw_ref()
                    self.update_bag_mass()

                    # 跌倒检测
                    self._check_fallen()

                    # 控制器更新
                    if not self.fallen:
                        self.robot.set_velocity(self.speed_ref)
                        self.robot.set_yaw(self.yaw_ref)
                        self.robot.update(CONTROL_DT)
                    else:
                        self.data.actuator('motor_l_wheel').ctrl = [0.0]
                        self.data.actuator('motor_r_wheel').ctrl = [0.0]

                mujoco.mj_step(self.model, self.data)

                # 更新显示用数据
                self.cart_speed = self.robot.get_speed()

                if self.data.time - self.last_print_time >= 0.5:
                    self.last_print_time = self.data.time
                    self._print_debug_info()
            else:
                time.sleep(0.00001)

    def _check_fallen(self):
        """相对在线估计平衡角判断跌倒，并同时检查横滚。"""
        if self.fallen:
            return
        pitch = self.robot.get_pitch()
        pitch_error = np.arctan2(
            np.sin(pitch - self.robot.theta_eq),
            np.cos(pitch - self.robot.theta_eq))
        roll = self.robot.get_roll()
        if abs(pitch_error) > np.deg2rad(35.0) or abs(roll) > np.deg2rad(30.0):
            if not self.fallen:
                self.fallen = True
                print(f"[WARNING] 球车跌倒! pitch={np.rad2deg(pitch):.1f} deg, "
                      f"eq={np.rad2deg(self.robot.theta_eq):.1f} deg, "
                      f"roll={np.rad2deg(roll):.1f} deg")

    def _print_debug_info(self):
        pitch_deg = np.rad2deg(self.robot.get_pitch())
        yaw_deg = np.rad2deg(self.robot.get_yaw())
        cart_pos = self.robot.get_position()
        h = self.human.get_state()

        if self.tracking_mode:
            print(f"[t={self.data.time:.1f}s] TRACKING | "
                  f"pitch={pitch_deg:6.1f} "
                  f"eq_hat={np.rad2deg(self.robot.theta_eq):5.1f} "
                  f"yaw={yaw_deg:6.1f} | "
                  f"cart=({cart_pos[0]:5.1f},{cart_pos[1]:5.1f}) v={self.cart_speed:.2f} | "
                  f"human=({h['x']:5.1f},{h['y']:5.1f}) v={h['speed']:.2f} | "
                  f"dist={self.current_distance:.2f}m | "
                  f"camera={'OK' if self.camera_visible else 'LOST'} "
                  f"err={self.camera_error_deg:.1f}deg"
                  f"{' FALLEN!' if self.fallen else ''}")
        else:
            print(f"[t={self.data.time:.1f}s] MANUAL | "
                  f"pitch={pitch_deg:6.1f} "
                  f"eq_hat={np.rad2deg(self.robot.theta_eq):5.1f} "
                  f"yaw={yaw_deg:6.1f} | "
                  f"v={self.cart_speed:.2f} target={self.speed_ref:.2f}"
                  f"{' FALLEN!' if self.fallen else ''}")

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.next_control_time = 0.0

        # 球包质量
        self.bag_mass = self.default_bag_mass
        self.target_bag_mass = self.default_bag_mass
        self.robot.set_simulated_bag_mass(self.bag_mass)

        # 重置LQR及未知平衡角观测器。
        self.robot.reset()

        # 重置人类
        self.human.reset()

        # 重置跟踪控制器
        self.tracker.reset(0.0, self.robot.get_yaw())

        # 重置状态
        self.fallen = False
        self.last_print_time = 0.0
        self.speed_cmd = 0.0
        self.speed_ref = 0.0
        self.yaw_cmd = 0.0
        self.yaw_ref = 0.0
        self.current_distance = 0.0
        self.human_speed = 0.0
        self.camera_visible = True
        self.camera_error_deg = 0.0

    def set_speed(self, s):
        self.speed_cmd = s

    def set_yaw(self, y):
        self.yaw_cmd = y

    def set_bag_mass(self, mass):
        self.target_bag_mass = float(np.clip(mass, 0.0, 9.0))

    def update_bag_mass(self):
        """在仿真线程内渐变质量，避免跨线程改模型和瞬时质心阶跃。"""
        max_step = self.bag_mass_rate * CONTROL_DT
        delta = np.clip(
            self.target_bag_mass - self.bag_mass, -max_step, max_step)
        if abs(delta) > 1e-9:
            self.bag_mass += delta
            self.robot.set_simulated_bag_mass(self.bag_mass)

    def set_tracking(self, enabled):
        self.tracking_mode = enabled
        if enabled:
            self.tracker.reset(
                self.robot.get_speed(), self.robot.get_yaw()
            )
            self.speed_ref = self.tracker.speed_ref
            self.yaw_ref = self.tracker.yaw_ref
            print("=== 跟踪模式启动 ===")
        else:
            self.speed_cmd = 0.0
            self.yaw_cmd = self.robot.get_yaw()
            print("=== 手动模式 ===")

    def reset_human(self):
        """仅重置人类位置"""
        self.human.reset()
        self.tracker.reset(
            self.robot.get_speed(), self.robot.get_yaw()
        )
        print("=== 人类位置已重置 ===")

    def update_speed_ref(self):
        # 起步保持柔和；制动必须明显更快，安全保护区内进一步加快。
        if self.speed_cmd >= self.speed_ref:
            rate = 0.60
        else:
            rate = 0.85
        STEP = rate * CONTROL_DT
        error = self.speed_cmd - self.speed_ref
        if error > STEP:
            error = STEP
        elif error < -STEP:
            error = -STEP
        self.speed_ref += error

    def update_yaw_ref(self):
        ACC = 3.0  # rad/s
        STEP = ACC * CONTROL_DT
        error = np.arctan2(
            np.sin(self.yaw_cmd - self.yaw_ref),
            np.cos(self.yaw_cmd - self.yaw_ref))
        if error > STEP:
            error = STEP
        elif error < -STEP:
            error = -STEP
        self.yaw_ref += error


# ============ 主窗口 ============
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Golf Balance Cart - Human Tracking")

        xml_path = pathlib.Path(__file__).parent.joinpath('xml/scene.xml')
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # 相机
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0, -2, 0.5])
        self.cam.distance = 10.5
        self.cam.elevation = -25
        self.cam.azimuth = 45

        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, 10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.updateRuntime.connect(self.show_runtime)

        # ---- 布局 ----
        layout = QVBoxLayout()
        top = QHBoxLayout()

        # 按钮
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(80)
        reset_btn.clicked.connect(self.reset_sim)

        self.tracking_btn = QPushButton("Tracking: OFF")
        self.tracking_btn.setCheckable(True)
        self.tracking_btn.setFixedWidth(140)
        self.tracking_btn.toggled.connect(self.toggle_tracking)

        human_reset_btn = QPushButton("Reset Human")
        human_reset_btn.setFixedWidth(100)
        human_reset_btn.clicked.connect(self.reset_human)

        self.camera_view_btn = QPushButton("Camera View")
        self.camera_view_btn.setCheckable(True)
        self.camera_view_btn.setFixedWidth(110)
        self.camera_view_btn.toggled.connect(self.toggle_camera_view)

        top.addWidget(reset_btn)
        top.addWidget(self.tracking_btn)
        top.addWidget(human_reset_btn)
        top.addWidget(self.camera_view_btn)

        # 滑块
        ctrl_layout = QVBoxLayout()

        # 速度
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(-15 / 3.6 * 1000)
        self.speed_slider.setMaximum(15 / 3.6 * 1000)
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(lambda v: self.th.set_speed(v / 1000))
        speed_layout.addWidget(QLabel("Speed"))
        speed_layout.addWidget(self.speed_slider)

        # 偏航
        yaw_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-np.deg2rad(180) * 1000)
        self.yaw_slider.setMaximum(np.deg2rad(180) * 1000)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(lambda v: self.th.set_yaw(-v / 1000))
        yaw_layout.addWidget(QLabel("Yaw"))
        yaw_layout.addWidget(self.yaw_slider)

        # 球包质量
        bag_mass_layout = QHBoxLayout()
        self.bag_mass_slider = QSlider(Qt.Horizontal)
        self.bag_mass_slider.setMinimum(0)
        self.bag_mass_slider.setMaximum(900)
        self.bag_mass_slider.setSingleStep(10)
        self.bag_mass_slider.setValue(900)
        self.bag_mass_slider.valueChanged.connect(self.on_bag_mass_changed)
        self.bag_mass_label = QLabel("Sim Bag: 9.0 kg")
        self.bag_mass_label.setMinimumWidth(70)
        bag_mass_layout.addWidget(self.bag_mass_label)
        bag_mass_layout.addWidget(self.bag_mass_slider)
        for label, value in (("No bag", 0), ("Empty 3kg", 300), ("Full 9kg", 900)):
            btn = QPushButton(label)
            btn.setFixedWidth(82)
            btn.clicked.connect(
                lambda checked=False, v=value: self.bag_mass_slider.setValue(v))
            bag_mass_layout.addWidget(btn)

        ctrl_layout.addLayout(speed_layout)
        ctrl_layout.addLayout(yaw_layout)
        ctrl_layout.addLayout(bag_mass_layout)

        top.addLayout(ctrl_layout)
        top.setContentsMargins(8, 0, 8, 0)
        layout.addLayout(top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0, 4, 0, 0)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(1650, 850)

        # 仿真线程
        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.set_bag_mass(9.0)
        self.th.start()

        # 快捷键
        QShortcut(QKeySequence("T"), self, activated=self._shortcut_tracking)
        QShortcut(QKeySequence("R"), self, activated=self.reset_sim)
        QShortcut(QKeySequence("H"), self, activated=self.reset_human)
        QShortcut(QKeySequence("C"), self, activated=self.camera_view_btn.toggle)

    def _shortcut_tracking(self):
        self.tracking_btn.toggle()

    def toggle_tracking(self, checked):
        self.th.set_tracking(checked)
        self.tracking_btn.setText("Tracking: ON" if checked else "Tracking: OFF")
        self.speed_slider.setEnabled(not checked)
        self.yaw_slider.setEnabled(not checked)

    def on_bag_mass_changed(self, val):
        mass = val / 100.0
        self.bag_mass_label.setText(f"Sim Bag: {mass:.1f} kg")
        self.th.set_bag_mass(mass)

    def reset_human(self):
        self.th.reset_human()

    def toggle_camera_view(self, checked):
        if checked:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.model.camera("person_tracking_camera").id
            self.camera_view_btn.setText("Free View")
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.fixedcamid = -1
            self.camera_view_btn.setText("Camera View")

    @Slot(float)
    def show_runtime(self, fps):
        mode = "TRACKING" if self.th.tracking_mode else "MANUAL"
        fallen = " [FALLEN! Press Reset]" if self.th.fallen else ""
        if self.th.tracking_mode:
            dist_str = f"  Dist: {self.th.current_distance:.2f}m"
            spd_str = f"  Human: {self.th.human_speed:.2f}m/s  Cart: {self.th.cart_speed:.2f}m/s"
            camera_str = (
                f"  Camera: {'VISIBLE' if self.th.camera_visible else 'LOST'}"
                f" ({self.th.camera_error_deg:.1f}deg)")
        else:
            dist_str = ""
            spd_str = f"  Cart: {self.th.cart_speed:.2f}m/s"
            camera_str = ""
        self.statusBar().showMessage(
            f"[{mode}]{fallen}  Avg: {fps:.1e}s  Sim: {self.data.time:.0f}s"
            f"{dist_str}{spd_str}{camera_str}")

    def reset_sim(self):
        self.speed_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.bag_mass_slider.setValue(900)
        self.bag_mass_label.setText("Sim Bag: 9.0 kg")
        self.tracking_btn.setChecked(False)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()

    def closeEvent(self, event):
        self.th.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
