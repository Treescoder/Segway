from collections import deque
import time
import mujoco
import numpy as np
import pathlib
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QGroupBox, QHBoxLayout, QSlider, QLabel
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import QGuiApplication, QSurfaceFormat

from segway_pid import SegwayPID

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

    def __init__(self, model, data, cam, opt, scn) -> None:
        super().__init__()
        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scn = scn
        self.width = 0
        self.height = 0
        self.scale = 1.0
        self.__last_pos = None
        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1/60*1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.body_id = model.body('segway').id

    def mousePressEvent(self, event):
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, action, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005 * event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h

    def setScreenScale(self, scaleFactor: float) -> None:
        self.scale = scaleFactor

    def paintGL(self) -> None:
        body_pos = self.data.xpos[self.body_id]
        self.cam.lookat = body_pos.copy()
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(viewport, self.scn, self.con)
        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.running = True
        self.robot = SegwayPID(model, data)
        self.speed = 0.0
        self.yaw = 0.0
        self.body_id = self.model.body('segway').id
        self.l_joint_id = self.model.joint('torso_l_wheel').id
        self.r_joint_id = self.model.joint('torso_r_wheel').id
        self.reset()
        self.last_print_time = 0.0

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def run(self) -> None:
        while self.running:
            if self.data.time < self.real_time / 1_000_000_000:
                if (time.monotonic_ns() - self.last_robot_update) / 1_000_000_000 >= (1/200):
                    self.last_robot_update = time.monotonic_ns()
                    self.robot.set_velocity_linear_set_point(self.speed)
                    self.robot.set_yaw(self.yaw)
                    self.robot.update_motor_speed()    # 力矩控制
                mujoco.mj_step(self.model, self.data)
                current_time = self.data.time
                if current_time - self.last_print_time >= 0.5:
                    self.last_print_time = current_time
                    self._print_debug_info()
            else:
                time.sleep(0.00001)

    def _print_debug_info(self):
        try:
            quat = self.data.xquat[self.body_id]
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            euler = r.as_euler('zyx', degrees=True)
            yaw, pitch, roll = euler

            l_vel = self.data.qvel[self.l_joint_id]
            r_vel = self.data.qvel[self.r_joint_id]
            l_torque = self.data.actuator_force[0]
            r_torque = self.data.actuator_force[1]
            l_ctrl = self.data.ctrl[0]
            r_ctrl = self.data.ctrl[1]

            WHEEL_RADIUS = 0.1275
            actual_speed = (l_vel + r_vel) / 2.0 * WHEEL_RADIUS

            print(f"[t={self.data.time:.2f}s] "
                  f"pitch={pitch:6.2f}° roll={roll:6.2f}° yaw={yaw:6.2f}° | "
                  f"actual_speed={actual_speed:6.3f} m/s | target_speed={self.speed:5.2f} m/s | "
                  f"L: vel={l_vel:6.2f} rad/s, torque={l_torque:6.2f}, ctrl={l_ctrl:6.2f} | "
                  f"R: vel={r_vel:6.2f} rad/s, torque={r_torque:6.2f}, ctrl={r_ctrl:6.2f}")
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

    def set_speed(self, speed: float) -> None:
        self.speed = speed

    def set_yaw(self, yaw: float) -> None:
        self.yaw = yaw


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        xml_path = pathlib.Path(__file__).parent.joinpath('xml\scene.xml')   # 你的新模型 XML
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout()
        layout_top = QHBoxLayout()
        layout_top.setSpacing(8)
        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(90)
        reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        reset_button.clicked.connect(self.reset_simulation)
        layout_top.addWidget(reset_button)
        layout_robot_controls = QVBoxLayout()
        layout_robot_controls.setContentsMargins(0,0,0,0)
        layout_robot_controls.addWidget(self.create_top())
        layout_top.addLayout(layout_robot_controls)
        layout_top.setContentsMargins(8,0,8,0)
        layout.addLayout(layout_top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,4,0,0)
        layout.setStretch(1,1)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(800, 600)
        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps: float):
        self.statusBar().showMessage(
            f"Average runtime: {fps:.0e}s\t"
            f"Simulation time: {self.data.time:.0f}s"
        )

    def create_top(self):
        layout = QVBoxLayout()
        label_width = 60
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(-4 * 1000)
        self.speed_slider.setMaximum(4 * 1000)
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        speed_label = QLabel("Speed")
        speed_label.setFixedWidth(label_width)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)

        yaw_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-10 * 1000)
        self.yaw_slider.setMaximum(10 * 1000)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self._yaw_changed)
        yaw_label = QLabel("Yaw")
        yaw_label.setFixedWidth(label_width)
        yaw_layout.addWidget(yaw_label)
        yaw_layout.addWidget(self.yaw_slider)

        layout.addLayout(speed_layout)
        layout.addLayout(yaw_layout)
        w = QGroupBox("Segway Control")
        w.setLayout(layout)
        return w

    def _speed_changed(self, value: int) -> None:
        speed = value / 1000
        self.th.set_speed(speed)

    def _yaw_changed(self, value: int) -> None:
        yaw = value / 1000
        self.th.set_yaw(yaw)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([0.0, 0.0, 0.0])
        cam.distance = self.model.stat.extent * 10
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
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