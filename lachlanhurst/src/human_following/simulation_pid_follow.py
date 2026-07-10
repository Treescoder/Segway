from collections import deque
import time
import math
import mujoco
import numpy as np
import pathlib
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QGroupBox, QHBoxLayout, QSlider, QLabel
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import (
    QGuiApplication, QSurfaceFormat
)

# 根据实际路径修改导入
from lachlanhurst.src.simulation.robot_pid import RobotPID

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
        self.body_id = model.body('robot_body').id

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
        robot_pos = self.data.xpos[self.body_id]
        human_pos = self.data.mocap_pos[self.model.body_mocapid[self.model.body('human').id]]
        mid_pos = human_pos # (robot_pos + human_pos) / 2.0
        self.cam.lookat = mid_pos.copy()

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

        self.robot = RobotPID(model, data)

        # 行人控制参数（滑块）
        self.human_speed = 0.0      # m/s
        self.human_turn = 0.0       # rad/s（暂不使用）

        # 行人状态（初始位置在车正前方 -y 方向 0.3m）
        self.human_x = 0.0
        self.human_y = -0.3
        self.human_heading = 0.0

        # 跟踪控制器 PID 参数
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.desired_follow_dist = 0.3

        # 获取关键 ID
        self.body_id = self.model.body('robot_body').id
        self.human_mocap_id = self.model.body_mocapid[self.model.body('human').id]

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
                    dt = 1.0 / 200.0

                    # 1. 行人运动学更新（仅直线移动）
                    # self.human_heading  #保持不变（0），不转弯
                    self.human_y -= self.human_speed * dt   # 沿 -y 前进

                    actual_human_speed = self.human_speed

                    # 设置 mocap 位置（z=0 使模型脚底接地）
                    self.data.mocap_pos[self.human_mocap_id] = [self.human_x, self.human_y, 0.0]
                    # 无旋转，保持面向 -y（默认朝向）
                    self.data.mocap_quat[self.human_mocap_id] = [1, 0, 0, 0]

                    # 2. 跟踪控制器（纯位置 PID，无转向）
                    robot_pos = self.data.xpos[self.body_id]
                    robot_mat = self.data.xmat[self.body_id].reshape(3, 3)
                    # 前进方向为局部 -y 轴
                    forward_global = -robot_mat[:, 1]

                    dx = self.human_x - robot_pos[0]
                    dy = self.human_y - robot_pos[1]
                    dist = math.hypot(dx, dy)
                    dist_err = dist - self.desired_follow_dist

                    # 投影到前进方向（标量距离误差）
                    forward_err = dx * forward_global[0] + dy * forward_global[1]

                    # PID 计算速度指令（只前进，不后退）
                    self.dist_err_sum += dist_err * dt
                    self.dist_err_sum = max(-1.0, min(1.0, self.dist_err_sum))  # 减小积分限幅
                    Kp = 2.7
                    Ki = 0.03
                    Kd = 0.05
                    v_adj = (Kp * dist_err +
                             Ki * self.dist_err_sum +
                             Kd * (dist_err - self.last_dist_err) / dt)
                    self.last_dist_err = dist_err
                    v_cmd = max(0.0, min(1.5, v_adj))   # 限制速度范围

                    # 转向控制暂时为 0，避免振荡
                    yaw_cmd = 0.0

                    self.robot.set_velocity_linear_set_point(v_cmd)
                    self.robot.set_yaw(yaw_cmd)
                    self.robot.update_motor_speed()

                # 仿真步进
                mujoco.mj_step(self.model, self.data)

                # 调试打印
                current_time = self.data.time
                if current_time - self.last_print_time >= 0.5:
                    self.last_print_time = current_time
                    robot_pos = self.data.xpos[self.body_id]
                    l_vel = self.data.joint('torso_l_wheel').qvel[0]
                    r_vel = self.data.joint('torso_r_wheel').qvel[0]
                    wheel_avg = (-1 * l_vel + r_vel) / 2.0
                    actual_car_speed = wheel_avg * 0.034 * 2   # 正值表示向前（-y方向），负值向后

                    dx = self.human_x - robot_pos[0]
                    dy = self.human_y - robot_pos[1]
                    actual_dist = math.hypot(dx, dy)
                    print(f"[t={self.data.time:.2f}s] "
                          f"target_human_speed={self.human_speed:.2f} m/s "
                          f"actual_human_speed={actual_human_speed:.2f} m/s | "
                          f"car_speed={actual_car_speed:.2f} m/s | "
                          f"dist={actual_dist:.2f}m (desired {self.desired_follow_dist}m) "
                          f"fwd_err={forward_err:.2f} v_cmd={v_cmd:.2f} yaw_cmd={yaw_cmd:.2f}")
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()
        self.human_x = 0.0
        self.human_y = -0.3
        self.human_heading = 0.0
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.last_print_time = 0.0

    def set_human_speed(self, speed: float) -> None:
        self.human_speed = speed

    def set_human_turn(self, turn: float) -> None:
        self.human_turn = turn   # 保留接口，暂不实现


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath('sim\scene.xml')))
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
        label_width = 70

        speed_layout = QHBoxLayout()
        self.human_speed_slider = QSlider(Qt.Horizontal)
        self.human_speed_slider.setMinimum(0)
        self.human_speed_slider.setMaximum(3000)     # 0 ~ 2.0 m/s
        self.human_speed_slider.setValue(0)
        self.human_speed_slider.valueChanged.connect(self._human_speed_changed)
        speed_label = QLabel("Human Speed")
        speed_label.setFixedWidth(label_width)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.human_speed_slider)

        # 转向滑块暂时保留但不接线（或可隐藏）
        turn_layout = QHBoxLayout()
        self.human_turn_slider = QSlider(Qt.Horizontal)
        self.human_turn_slider.setMinimum(-2000)
        self.human_turn_slider.setMaximum(2000)
        self.human_turn_slider.setValue(0)
        turn_label = QLabel("Human Turn")
        turn_label.setFixedWidth(label_width)
        turn_layout.addWidget(turn_label)
        turn_layout.addWidget(self.human_turn_slider)

        layout.addLayout(speed_layout)
        # layout.addLayout(turn_layout)

        w = QGroupBox("Human Control")
        w.setLayout(layout)
        return w

    def _human_speed_changed(self, value: int) -> None:
        speed = value / 1000
        self.th.set_human_speed(speed)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([0.0, 0.0, 0.0])
        cam.distance = self.model.stat.extent * 2
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
        self.human_speed_slider.setValue(0)
        # self.human_turn_slider.setValue(0)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()