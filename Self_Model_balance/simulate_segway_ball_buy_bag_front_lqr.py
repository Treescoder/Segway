from collections import deque
import time
import mujoco
import numpy as np
import pathlib
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QLabel
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from segway_ball_lqr_bag_front import SegwayLQR
from scipy.spatial.transform import Rotation

format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2,0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)

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

    def mousePressEvent(self, e): self.__last_pos = e.position()
    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.RightButton: act = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif e.buttons() & Qt.MouseButton.LeftButton: act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif e.buttons() & Qt.MouseButton.MiddleButton: act = mujoco.mjtMouse.mjMOUSE_ZOOM
        else: return
        p = e.position()
        dx, dy = p.x()-self.__last_pos.x(), p.y()-self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, act, dx/self.height, dy/self.height, self.scn, self.cam)
        self.__last_pos = p
    def wheelEvent(self, e):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005*e.angleDelta().y(), self.scn, self.cam)
    def initializeGL(self): self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
    def resizeGL(self, w, h): self.width, self.height = w, h
    def setScreenScale(self, f): self.scale = f
    def paintGL(self):
        body_pos = self.data.xpos[self.body_id]
        self.cam.lookat = body_pos.copy()
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        screen = self.screen()
        if screen is not None:
            self.scale = screen.devicePixelRatio()
        vp = mujoco.MjrRect(0,0,int(self.width*self.scale),int(self.height*self.scale))
        mujoco.mjr_render(vp, self.scn, self.con)
        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))

class UpdateSimThread(QThread):
    def __init__(self, model, data, parent=None):
        super().__init__(parent)
        self.model, self.data = model, data
        self.running = True
        self.robot = SegwayLQR(model, data)
        self.speed_cmd = 0.0
        self.speed_ref = 0.0
        self.yaw_cmd = 0.0
        self.yaw_ref = 0.0
        self.body_id = model.body('segway').id
        self.l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]
        self.reset()
        self.last_print_time = 0.0

    @property
    def real_time(self): return time.monotonic_ns() - self.real_time_start

    def run(self):
        while self.running:
            if self.data.time < self.real_time/1e9:
                if (time.monotonic_ns()-self.last_robot_update)/1e9 >= 0.005:
                    self.last_robot_update = time.monotonic_ns()
                    self.update_speed_ref()
                    self.update_yaw_ref()
                    self.robot.set_velocity(self.speed_ref)
                    self.robot.set_yaw(self.yaw_ref)
                    self.robot.update()
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
            pitch, roll, yaw = euler[0], euler[1], euler[2]
            l_vel, r_vel = self.data.qvel[self.l_dof], self.data.qvel[self.r_dof]
            actual_speed = (l_vel+r_vel) / 2 * 0.24
            l_ctrl, r_ctrl = self.data.ctrl[0], self.data.ctrl[1]
            print(f"[t={self.data.time:.2f}s] pitch={pitch:6.2f}° roll={roll:6.2f}° yaw={yaw:6.2f}° | "
                  f"actual_speed={actual_speed:6.3f} m/s | target_speed={self.speed_ref:5.2f} m/s | "
                  f"L_vel={l_vel:6.2f} R_vel={r_vel:6.2f} | ctrl=({l_ctrl:6.2f},{r_ctrl:6.2f})")
        except Exception as e:
            print(f"Debug print error: {e}")

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()
        self.last_print_time = 0.0
        self.speed_cmd = 0
        self.speed_ref = 0
        self.yaw_cmd = 0
        self.yaw_ref = 0

    def set_speed(self, s): self.speed_cmd = s
    def set_yaw(self, y): self.yaw_cmd = y

    def update_speed_ref(self):
        ACC = 0.8
        STEP = ACC * 0.005
        error = self.speed_cmd - self.speed_ref
        if error > STEP:
            error = STEP
        elif error < -STEP:
            error = -STEP
        self.speed_ref += error

    def update_yaw_ref(self):
        ACC = 0.698
        STEP = ACC * 0.005
        error = np.arctan2(
            np.sin(self.yaw_cmd - self.yaw_ref),
            np.cos(self.yaw_cmd - self.yaw_ref)
        )
        if error > STEP:
            error = STEP
        elif error < -STEP:
            error = -STEP
        self.yaw_ref += error

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        xml_path = pathlib.Path(__file__).parent.joinpath('xml/scene.xml')
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0,0,0])
        self.cam.distance = self.model.stat.extent * 15
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
        ctrl_layout = QVBoxLayout()
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(-15/3.6 * 1000)
        self.speed_slider.setMaximum(15/3.6 * 1000)
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(lambda v: self.th.set_speed(v/1000))
        speed_layout.addWidget(QLabel("Speed"))
        speed_layout.addWidget(self.speed_slider)
        yaw_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-np.deg2rad(180)*1000)
        self.yaw_slider.setMaximum(np.deg2rad(180)*1000)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(lambda v: self.th.set_yaw(-v/1000))
        yaw_layout.addWidget(QLabel("Yaw"))
        yaw_layout.addWidget(self.yaw_slider)
        ctrl_layout.addLayout(speed_layout)
        ctrl_layout.addLayout(yaw_layout)
        top.addLayout(ctrl_layout)
        top.setContentsMargins(8,0,8,0)
        layout.addLayout(top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,4,0,0)
        w = QWidget(); w.setLayout(layout); self.setCentralWidget(w)
        self.resize(1650,850)
        self.move(1950, 20)
        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps):
        self.statusBar().showMessage(f"Avg runtime: {fps:.0e}s  Sim time: {self.data.time:.0f}s")

    def reset_sim(self):
        self.speed_slider.setValue(0)
        self.yaw_slider.setValue(0)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()

if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()