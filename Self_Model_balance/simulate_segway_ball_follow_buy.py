import math
import time
import mujoco
import numpy as np
import pathlib
from collections import deque
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QGroupBox, QHBoxLayout, QSlider, QLabel
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from segway_ball_follow_pid_buy import SegwayPID, WHEEL_RADIUS

# ========== 跟踪参数 ==========
DESIRED_DIST = 3.0
DIST_KP = 50.0
DIST_KI = 0.03
DIST_KD = 0.05
DIST_INT_LIMIT = 1.0
HEADING_KP = -15.0
TURN_FF = -1.6
YAW_ALPHA = 0.3
MAX_YAW = 1.5

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ---------- OpenGL 格式 ----------
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
        self.human_mocap_id = model.body_mocapid[model.body('human').id]

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
        robot_pos = self.data.xpos[self.body_id]
        human_pos = self.data.mocap_pos[self.human_mocap_id]
        mid = (robot_pos + human_pos) / 2.0
        self.cam.lookat = mid.copy()
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        vp = mujoco.MjrRect(0,0,int(self.width*self.scale),int(self.height*self.scale))
        mujoco.mjr_render(vp, self.scn, self.con)
        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


# ---------- 行人跟踪控制器 ----------
class HumanFollower:
    def __init__(self, desired_dist=DESIRED_DIST):
        self.desired_dist = desired_dist
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0

    def compute(self, robot_pos, robot_mat, human_x, human_y, human_heading, human_speed, human_turn, dt):
        forward_global = -robot_mat[:, 1]   # 指向 -X 世界方向
        robot_heading = math.atan2(forward_global[1], forward_global[0])

        dx = human_x - robot_pos[0]
        dy = human_y - robot_pos[1]
        dist = math.hypot(dx, dy)
        dist_err = dist - self.desired_dist

        target_heading = math.atan2(dy, dx)
        heading_err = target_heading - robot_heading
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        # 速度 PID
        self.dist_err_sum += dist_err * dt
        self.dist_err_sum = clamp(self.dist_err_sum, -DIST_INT_LIMIT, DIST_INT_LIMIT)
        v_adj = (DIST_KP * dist_err +
                 DIST_KI * self.dist_err_sum +
                 DIST_KD * (dist_err - self.last_dist_err) / dt)
        self.last_dist_err = dist_err

        # 动态限速
        ah = abs(heading_err)
        if ah < math.radians(10):
            maxv = 1.5
        elif ah < math.radians(45):
            maxv = 0.8
        elif ah < math.radians(100):
            maxv = 0.3
        else:
            maxv = 0.1
        v_cmd = clamp(v_adj, 0.0, maxv)
        if abs(human_speed) < 0.01 and abs(dist_err) < 0.1:
            v_cmd = 0.0
            self.dist_err_sum = 0.0
            self.last_dist_err = dist_err

        # 偏航控制
        raw_yaw = HEADING_KP * heading_err + TURN_FF * human_turn
        self.yaw_filtered = YAW_ALPHA * raw_yaw + (1 - YAW_ALPHA) * self.yaw_filtered
        yaw_cmd = clamp(self.yaw_filtered, -MAX_YAW, MAX_YAW)

        return v_cmd, yaw_cmd, dist_err, heading_err

    def reset(self):
        self.dist_err_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_filtered = 0.0


# ---------- 仿真线程 ----------
class UpdateSimThread(QThread):
    def __init__(self, model, data, parent=None):
        super().__init__(parent)
        self.model, self.data = model, data
        self.running = True
        self.robot = SegwayPID(model, data)
        self.follower = HumanFollower()

        # 行人状态（由滑块控制）
        self.human_speed = 0.0
        self.human_turn = 0.0
        self.human_x = 0.0
        self.human_y = 0.0                  # 将在 reset 中设置
        self.human_heading = 0.0

        self.body_id = model.body('segway').id
        self.human_mocap_id = model.body_mocapid[model.body('human').id]
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
                    dt = 0.005

                    self.human_heading += self.human_turn * dt
                    self.human_x += self.human_speed * math.sin(self.human_heading) * dt
                    self.human_y -= self.human_speed * math.cos(self.human_heading) * dt

                    self.data.mocap_pos[self.human_mocap_id] = [self.human_x, self.human_y, 0.0]
                    qw = math.cos(self.human_heading / 2)
                    qz = math.sin(self.human_heading / 2)
                    self.data.mocap_quat[self.human_mocap_id] = [qw, 0, 0, qz]

                    robot_pos = self.data.xpos[self.body_id]
                    robot_mat = self.data.xmat[self.body_id].reshape(3, 3)
                    v_cmd, yaw_cmd, dist_err, hdg_err = self.follower.compute(
                        robot_pos, robot_mat,
                        self.human_x, self.human_y,
                        self.human_heading, self.human_speed, self.human_turn, dt)

                    self.robot.set_velocity_linear_set_point(v_cmd)
                    self.robot.set_yaw(yaw_cmd)
                    self.robot.update_motor_torque()

                mujoco.mj_step(self.model, self.data)
                if self.data.time - self.last_print_time >= 0.5:
                    self.last_print_time = self.data.time
                    self._print_debug_info()
            else:
                time.sleep(0.00001)

    def _print_debug_info(self):
        try:
            robot_pos = self.data.xpos[self.body_id]
            robot_mat = self.data.xmat[self.body_id].reshape(3, 3)
            fwd = -robot_mat[:, 0]          # 与跟踪算法一致
            robot_heading = math.atan2(fwd[1], fwd[0])

            dx = self.human_x - robot_pos[0]
            dy = self.human_y - robot_pos[1]
            dist_err = math.hypot(dx, dy) - DESIRED_DIST

            human_deg = math.degrees(self.human_heading) % 360
            human_deg = (human_deg + 180) % 360 - 180
            robot_deg = math.degrees(robot_heading)

            l_vel = self.data.qvel[self.l_dof]
            r_vel = self.data.qvel[self.r_dof]
            actual_speed = (l_vel + r_vel) / 2.0 * WHEEL_RADIUS

            print(f"[t={self.data.time:.1f}s] "
                  f"人({self.human_x:.2f},{self.human_y:.2f}) "
                  f"车({robot_pos[0]:.2f},{robot_pos[1]:.2f}) "
                  f"人角={human_deg:.1f}° 车角={robot_deg:.1f}° "
                  f"车速={actual_speed:.2f} m/s 距离误差={dist_err:.2f} m")
        except Exception as e:
            print(f"Debug print error: {e}")

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()            # ★ 仅调用平衡控制器复位，不旋转车体
        self.follower.reset()

        self.human_x = 0.0
        self.human_y = 5.0
        self.human_heading = math.pi   # 朝向 -X（180°）
        self.last_print_time = 0.0

        self.data.mocap_pos[self.human_mocap_id] = [self.human_x, self.human_y, 0.0]
        qw = math.cos(self.human_heading / 2)
        qz = math.sin(self.human_heading / 2)
        self.data.mocap_quat[self.human_mocap_id] = [qw, 0, 0, qz]

    def set_human_speed(self, s): self.human_speed = s
    def set_human_turn(self, t): self.human_turn = t


# ---------- 主窗口 ----------
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
        # 行人速度滑块
        speed_layout = QHBoxLayout()
        self.human_speed_slider = QSlider(Qt.Horizontal)
        self.human_speed_slider.setMinimum(0)
        self.human_speed_slider.setMaximum(3000)
        self.human_speed_slider.setValue(0)
        self.human_speed_slider.valueChanged.connect(lambda v: self.th.set_human_speed(v / 1000))
        speed_layout.addWidget(QLabel("Human Speed"))
        speed_layout.addWidget(self.human_speed_slider)
        # 行人转向滑块
        turn_layout = QHBoxLayout()
        self.human_turn_slider = QSlider(Qt.Horizontal)
        self.human_turn_slider.setMinimum(-2000)
        self.human_turn_slider.setMaximum(2000)
        self.human_turn_slider.setValue(0)
        self.human_turn_slider.valueChanged.connect(lambda v: self.th.set_human_turn(v / 1000))
        turn_layout.addWidget(QLabel("Human Turn"))
        turn_layout.addWidget(self.human_turn_slider)
        ctrl_layout.addLayout(speed_layout)
        ctrl_layout.addLayout(turn_layout)
        top.addLayout(ctrl_layout)
        top.setContentsMargins(8,0,8,0)
        layout.addLayout(top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,4,0,0)
        w = QWidget(); w.setLayout(layout); self.setCentralWidget(w)
        self.resize(800,600)
        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps):
        self.statusBar().showMessage(f"Avg runtime: {fps:.0e}s  Sim time: {self.data.time:.0f}s")

    def reset_sim(self):
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