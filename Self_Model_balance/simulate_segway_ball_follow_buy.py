# 行人轨迹手动调节（修正航向与速度方向）
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
from segway_ball_follow_pid_buy import SegwayPID

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


class Viewport(QOpenGLWindow): #"""3D 视图，摄像机跟随人车中间点"""
    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn) -> None:
        super().__init__()
        self.model, self.data, self.cam, self.opt, self.scn = model, data, cam, opt, scn
        self.width = self.height = 0
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
        mujoco.mjv_moveCamera(self.model, action, dx/self.height, dy/self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005*event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width, self.height = w, h

    def setScreenScale(self, f):
        self.scale = f

    def paintGL(self):
        robot_pos = self.data.xpos[self.body_id]
        human_pos = self.data.mocap_pos[self.model.body_mocapid[self.model.body('human').id]]
        self.cam.lookat = ((robot_pos + human_pos) / 2.0).copy()
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        vp = mujoco.MjrRect(0,0, int(self.width*self.scale), int(self.height*self.scale))
        mujoco.mjr_render(vp, self.scn, self.con)
        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))

# ---------- 行人跟踪控制器 ----------
class HumanFollower:
    def __init__(self, desired_dist= 0.3):
        self.desired_dist = desired_dist
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0

    def compute(self, robot_pos, robot_mat, human_x, human_y, human_heading, human_speed, human_turn, dt):
        forward_global = -robot_mat[:, 1] # 车辆前进方向-y
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
        Kp, Ki, Kd = 200.7, 0.03, 0.05
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
        self.yaw_filtered = alpha*raw_yaw + (1-alpha)*self.yaw_filtered
        yaw_cmd = max(-1.5, min(1.5, self.yaw_filtered))

        return v_cmd, yaw_cmd, dist_err, heading_err, max_v

    def reset(self):
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0

# ---------- 仿真主线程 ----------
class UpdateSimThread(QThread):
    def __init__(self, model, data, parent=None):
        super().__init__(parent)
        self.model, self.data = model, data
        self.running = True
        self.robot = SegwayPID(model, data)
        self.follower = HumanFollower(desired_dist=0.3)

        self.human_speed = 0.0
        self.human_turn = 0.0
        self.human_x = 0.0
        self.human_y = 3.0
        self.human_heading = math.pi

        self.body_id = model.body('segway').id
        self.human_mocap_id = model.body_mocapid[model.body('human').id]
        self.reset()
        self.last_print_time = 0.0

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def run(self):
        while self.running:
            if self.data.time < self.real_time / 1_000_000_000:
                if (time.monotonic_ns() - self.last_robot_update) / 1e9 >= 0.005:
                    self.last_robot_update = time.monotonic_ns()
                    dt = 1.0 / 200.0
                    # 更新行人位置与朝向
                    self.human_heading += self.human_turn * dt
                    self.human_x += self.human_speed * math.sin(self.human_heading) * dt
                    self.human_y -= self.human_speed * math.cos(self.human_heading) * dt

                    self.data.mocap_pos[self.human_mocap_id] = [self.human_x, self.human_y, 0.0]
                    qw = math.cos(self.human_heading/2)
                    qz = math.sin(self.human_heading/2)
                    self.data.mocap_quat[self.human_mocap_id] = [qw, 0, 0, qz]
                    # 计算跟踪指令
                    robot_pos = self.data.xpos[self.body_id]
                    robot_mat = self.data.xmat[self.body_id].reshape(3,3)
                    v_cmd, yaw_cmd, dist_err, heading_err, max_v = self.follower.compute(
                        robot_pos, robot_mat, self.human_x, self.human_y,
                        self.human_heading, self.human_speed, self.human_turn, dt)
                    # 发送给底层平衡控制器
                    self.robot.set_velocity_linear_set_point(v_cmd)
                    self.robot.set_yaw(yaw_cmd)
                    self.robot.update_motor_torque()

                mujoco.mj_step(self.model, self.data)

                # 定期打印状态
                current_time = self.data.time
                if current_time - self.last_print_time >= 0.5:
                    self.last_print_time = current_time
                    robot_pos = self.data.xpos[self.body_id]
                    robot_mat = self.data.xmat[self.body_id].reshape(3,3)
                    forward_global = -robot_mat[:, 1]
                    robot_heading_raw = math.atan2(forward_global[1], forward_global[0])
                    human_deg = math.degrees(self.human_heading) % 360 # 归一化人的角度到 [-180, 180]
                    if human_deg > 180: human_deg -= 360
                    robot_deg = math.degrees(robot_heading_raw) - 90 # 归一化车的角度到 [-180, 180]，并转换基准
                    robot_deg = (robot_deg + 180) % 360 - 180   # 映射到[-180,180]

                    l_vel = self.data.joint('torso_l_wheel').qvel[0]
                    r_vel = self.data.joint('torso_r_wheel').qvel[0]
                    actual_speed = (l_vel + r_vel) / 2.0 * 0.24

                    print(f"[t={self.data.time:.1f}s] "
                          f"速度(人/车): {self.human_speed:.2f}/{actual_speed:.2f} m/s | "
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
        self.human_y = 3.0
        self.human_heading = math.pi
        self.last_print_time = 0.0

    def set_human_speed(self, speed):
        self.human_speed = speed   # 不再反号

    def set_human_turn(self, turn):
        self.human_turn = turn

# ---------- 主窗口 ----------
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(
            str(pathlib.Path(__file__).parent.joinpath('xml/scene.xml')))
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
        top = QHBoxLayout(); top.setSpacing(8)
        reset_btn = QPushButton("Reset"); reset_btn.clicked.connect(self.reset_simulation)
        top.addWidget(reset_btn)
        top.addWidget(self.create_control_panel())
        top.setContentsMargins(8,0,8,0)
        layout.addLayout(top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,4,0,0); layout.setStretch(1,1)
        w = QWidget(); w.setLayout(layout); self.setCentralWidget(w)
        self.resize(800, 600)
        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps):
        self.statusBar().showMessage(f"Avg runtime: {fps:.0e}s  Sim time: {self.data.time:.0f}s")

    def create_control_panel(self):
        layout = QVBoxLayout()
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0); self.speed_slider.setMaximum(3000); self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(lambda v: self.th.set_human_speed(v/1000))
        speed_layout.addWidget(QLabel("Human Speed")); speed_layout.addWidget(self.speed_slider)
        turn_layout = QHBoxLayout()
        self.turn_slider = QSlider(Qt.Horizontal)
        self.turn_slider.setMinimum(-2000); self.turn_slider.setMaximum(2000); self.turn_slider.setValue(0)
        self.turn_slider.valueChanged.connect(lambda v: self.th.set_human_turn(v/1000))
        turn_layout.addWidget(QLabel("Human Turn")); turn_layout.addWidget(self.turn_slider)
        layout.addLayout(speed_layout); layout.addLayout(turn_layout)
        w = QGroupBox("Human Control"); w.setLayout(layout)
        return w

    def create_free_camera(self):
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = np.zeros(3); cam.distance = self.model.stat.extent * 20
        cam.elevation = -25; cam.azimuth = 45
        return cam

    def reset_simulation(self):
        self.speed_slider.setValue(0); self.turn_slider.setValue(0)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()

if __name__ == "__main__":
    app = QApplication()
    w = Window(); w.show()
    app.exec()
    w.th.stop()