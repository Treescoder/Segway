# 行人轨迹手动调节
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

from lachlanhurst.src.simulation.robot_pid import RobotPID

# ---------- OpenGL 设置 ----------
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
    """3D 视图，摄像机跟随人车中间点"""
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
        mid_pos = (robot_pos + human_pos) / 2.0
        self.cam.lookat = mid_pos.copy()

        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(viewport, self.scn, self.con)
        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


# ---------- 行人跟踪控制器 ----------
class HumanFollower:
    """计算速度与转向指令，不修改底层平衡控制"""
    def __init__(self, desired_dist=0.3):
        self.desired_dist = desired_dist
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0

    def compute(self, robot_pos, robot_mat, human_x, human_y, human_heading, human_speed, human_turn, dt):
        # 机器人前进方向 (局部 -y)
        forward_global = -robot_mat[:, 1]
        right_global = robot_mat[:, 0]
        robot_heading = math.atan2(forward_global[1], forward_global[0])

        dx = human_x - robot_pos[0]
        dy = human_y - robot_pos[1]
        dist = math.hypot(dx, dy)
        dist_err = dist - self.desired_dist

        target_heading = math.atan2(dy, dx)
        heading_err = target_heading - robot_heading
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        # ---------- 速度控制（纯距离 PID，无前馈） ----------
        self.dist_err_sum += dist_err * dt
        self.dist_err_sum = max(-1.0, min(1.0, self.dist_err_sum))
        Kp, Ki, Kd = 2.7, 0.03, 0.05
        v_adj = (Kp * dist_err + Ki * self.dist_err_sum + Kd * (dist_err - self.last_dist_err) / dt)
        self.last_dist_err = dist_err

        # 动态限速：航向误差大时减速
        abs_heading = abs(heading_err)
        if abs_heading < math.radians(20):
            max_v = 1.5
        elif abs_heading < math.radians(60):
            max_v = 0.8
        elif abs_heading < math.radians(120):
            max_v = 0.3
        else:
            max_v = 0.1
        v_cmd = max(0.0, min(max_v, v_adj))

        # ---------- 转向控制 ----------
        Kp_heading = -9.0
        turn_feedforward = -1.5 * human_turn
        raw_yaw = Kp_heading * heading_err + turn_feedforward
        alpha = 0.3
        self.yaw_filtered = alpha * raw_yaw + (1 - alpha) * self.yaw_filtered
        yaw_cmd = max(-1.5, min(1.5, self.yaw_filtered))

        return v_cmd, yaw_cmd, dist_err, heading_err, max_v

    def reset(self):
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0


# ---------- 仿真主线程 ----------
class UpdateSimThread(QThread):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.running = True

        self.robot = RobotPID(model, data)
        self.follower = HumanFollower(desired_dist=0.3)

        # 行人状态
        self.human_speed = 0.0
        self.human_turn = 0.0
        self.human_x = 0.0
        self.human_y = -0.3
        self.human_heading = 0.0

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

                    # 更新行人位置与朝向
                    self.human_heading += self.human_turn * dt
                    self.human_x += self.human_speed * math.sin(self.human_heading) * dt
                    self.human_y -= self.human_speed * math.cos(self.human_heading) * dt

                    self.data.mocap_pos[self.human_mocap_id] = [self.human_x, self.human_y, 0.0]
                    qw = math.cos(self.human_heading / 2)
                    qz = math.sin(self.human_heading / 2)
                    self.data.mocap_quat[self.human_mocap_id] = [qw, 0, 0, qz]

                    # 计算跟踪指令
                    robot_pos = self.data.xpos[self.body_id]
                    robot_mat = self.data.xmat[self.body_id].reshape(3, 3)
                    v_cmd, yaw_cmd, dist_err, heading_err, max_v = self.follower.compute(
                        robot_pos, robot_mat,
                        self.human_x, self.human_y,
                        self.human_heading, self.human_speed, self.human_turn, dt)

                    # 发送给底层平衡控制器
                    self.robot.set_velocity_linear_set_point(v_cmd)
                    self.robot.set_yaw(yaw_cmd)
                    self.robot.update_motor_speed()

                mujoco.mj_step(self.model, self.data)

                # 定期打印状态
                current_time = self.data.time
                if current_time - self.last_print_time >= 0.5:
                    self.last_print_time = current_time
                    robot_pos = self.data.xpos[self.body_id]
                    robot_mat = self.data.xmat[self.body_id].reshape(3, 3)
                    robot_forward = -robot_mat[:, 1]
                    robot_heading_raw = math.atan2(robot_forward[1], robot_forward[0])

                    # 归一化人的角度到 [-180, 180]
                    human_deg = math.degrees(self.human_heading) % 360.0
                    if human_deg > 180.0:
                        human_deg -= 360.0

                    # 归一化车的角度到 [-180, 180]，并转换基准
                    robot_deg = math.degrees(robot_heading_raw) + 90.0
                    robot_deg = (robot_deg + 180.0) % 360.0 - 180.0

                    # 实际车速（m/s）
                    l_vel = self.data.joint('torso_l_wheel').qvel[0]
                    r_vel = self.data.joint('torso_r_wheel').qvel[0]
                    wheel_avg = (-1 * l_vel + r_vel) / 2.0
                    actual_car_speed = wheel_avg * 0.034

                    print(f"[t={self.data.time:.1f}s] "
                          f"速度(人/车): {self.human_speed:.2f}/{actual_car_speed:.2f} m/s | "
                          f"角度(人/车): {human_deg:.1f}°/{robot_deg:.1f}° | "
                          f"距离误差: {dist_err:.2f} m | 航向误差: {math.degrees(heading_err):.1f}° | "
                          f"限速: {max_v:.1f} m/s")
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()
        self.follower.reset()
        self.human_x = 0.0
        self.human_y = -0.3
        self.human_heading = 0.0
        self.last_print_time = 0.0

    def set_human_speed(self, speed: float) -> None:
        self.human_speed = speed

    def set_human_turn(self, turn: float) -> None:
        self.human_turn = turn


# ---------- 主窗口 ----------
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
        layout_top.addWidget(self.create_control_panel())
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
            f"Average runtime: {fps:.0e}s\tSimulation time: {self.data.time:.0f}s")

    def create_control_panel(self):
        layout = QVBoxLayout()

        speed_layout = QHBoxLayout()
        self.human_speed_slider = QSlider(Qt.Horizontal)
        self.human_speed_slider.setMinimum(0)
        self.human_speed_slider.setMaximum(3000)
        self.human_speed_slider.setValue(0)
        self.human_speed_slider.valueChanged.connect(
            lambda v: self.th.set_human_speed(v / 1000))
        speed_layout.addWidget(QLabel("Human Speed"))
        speed_layout.addWidget(self.human_speed_slider)

        turn_layout = QHBoxLayout()
        self.human_turn_slider = QSlider(Qt.Horizontal)
        self.human_turn_slider.setMinimum(-2000)
        self.human_turn_slider.setMaximum(2000)
        self.human_turn_slider.setValue(0)
        self.human_turn_slider.valueChanged.connect(
            lambda v: self.th.set_human_turn(v / 1000))
        turn_layout.addWidget(QLabel("Human Turn"))
        turn_layout.addWidget(self.human_turn_slider)

        layout.addLayout(speed_layout)
        layout.addLayout(turn_layout)

        w = QGroupBox("Human Control")
        w.setLayout(layout)
        return w

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
        self.human_turn_slider.setValue(0)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()