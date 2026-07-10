# sim/runs_qt.py
import sys
import time
import math
import numpy as np
import mujoco
from collections import deque
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import QGuiApplication, QSurfaceFormat

# ---------- OpenGL 格式：不用改 ----------
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
        self.width, self.height, self.scale = 0, 0, 1.0
        self.__last_pos = None
        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.body_id = model.body('robot_body').id

    def mousePressEvent(self, e): self.__last_pos = e.position()
    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.RightButton: act = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif e.buttons() & Qt.MouseButton.LeftButton: act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif e.buttons() & Qt.MouseButton.MiddleButton: act = mujoco.mjtMouse.mjMOUSE_ZOOM
        else: return
        p = e.position()
        dx = p.x() - self.__last_pos.x()
        dy = p.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, act, dx/self.height, dy/self.height, self.scn, self.cam)
        self.__last_pos = p
    def wheelEvent(self, e):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005*e.angleDelta().y(), self.scn, self.cam)
    def initializeGL(self): self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
    def resizeGL(self, w, h): self.width, self.height = w, h
    def setScreenScale(self, f): self.scale = f
    def paintGL(self):
        rp = self.data.xpos[self.body_id]
        hp = self.data.mocap_pos[self.model.body_mocapid[self.model.body('human').id]]
        mid = (rp + hp) / 2
        self.cam.lookat = mid.copy()
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        vp = mujoco.MjrRect(0,0,int(self.width*self.scale),int(self.height*self.scale))
        mujoco.mjr_render(vp, self.scn, self.con)
        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))

# ---------- 注意：SimThread 需要接收 control_callback，但为了保持原样，我们传入 control_callback ----------
class SimThread(QThread):
    def __init__(self, model, data, control_callback):
        super().__init__()
        self.model, self.data = model, data
        self.control_callback = control_callback
        self.running = True
        self.real_start = time.monotonic_ns()
        self.last_update = time.monotonic_ns()

    def run(self): # 设置控制频率200Hz, 5ms
        while self.running:
            if self.data.time < (time.monotonic_ns() - self.real_start) / 1e9:
                if (time.monotonic_ns() - self.last_update) / 1e9 >= 1/200:
                    self.last_update = time.monotonic_ns()
                    dt = 1/200 # 控制周期
                    self.control_callback(self.model, self.data, dt)
                mujoco.mj_step(self.model, self.data)
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

class Window(QMainWindow): # Mujoco相机设置
    def __init__(self, model, data, control_callback):
        super().__init__()
        self.model = model
        self.data = data
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0.,0.,0.])
        self.cam.distance = model.stat.extent*2
        self.cam.elevation = -25
        self.cam.azimuth = 45
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(model, 10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True

        self.viewport = Viewport(model, data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout()
        top = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_sim)
        top.addWidget(reset_btn)
        top.addStretch()
        layout.addLayout(top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,4,0,0)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(800,600)

        self.th = SimThread(model, data, control_callback)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps):
        self.statusBar().showMessage(f"Avg runtime: {fps:.0e}s  Sim time: {self.data.time:.0f}s")

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.real_start = time.monotonic_ns()
        self.th.last_update = time.monotonic_ns()

def run_experiments(model_path, control_callback, init_callback=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if init_callback is not None:
        init_callback(model, data)
    mujoco.mj_forward(model, data)

    window = Window(model, data, control_callback)
    window.show()
    app.exec()
    window.th.stop()