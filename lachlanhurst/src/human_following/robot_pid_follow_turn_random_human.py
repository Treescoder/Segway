# 行人随机生成轨迹，但是仿真界面和控制算法未优化，比较混乱
# 跟随+仿真该代码；主控制：robot_pid_follow
from collections import deque
import time
import mujoco
import numpy as np
import math
import pathlib
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QLabel
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtGui import QGuiApplication, QSurfaceFormat

from lachlanhurst.src.simulation.robot_pid import RobotPID # 导入平衡车核心控制算法robot_pid
from sim.random_human import RandomHumanTrajectory # 导入随机行人轨迹random_human

# ---------- 跟踪控制参数 ----------
DESIRED_DIST = 0.3
DIST_KP = 3.0
DIST_KI = 0.03
DIST_KD = 0.05
HEADING_KP = -9.0
TURN_FF = -1.5
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

class SimThread(QThread):
    def __init__(self, model, data):
        super().__init__()
        self.model, self.data = model, data
        self.running = True
        self.robot = RobotPID(model, data)

        # 行人轨迹生成器
        self.human = RandomHumanTrajectory(start_x=0.0, start_y=-0.3)
        self.human_mocap_id = model.body_mocapid[model.body('human').id]

        # 跟踪器状态
        self.dist_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_f = 0.0
        self.last_print_time = 0.0

        self.reset()

    def reset(self):
        self.real_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()
        self.human = RandomHumanTrajectory(0.0, -0.3)
        self.dist_sum = 0.0
        self.last_dist_err = 0.0
        self.yaw_f = 0.0
        self.last_print_time = 0.0

    def run(self):
        while self.running:
            # 实时同步
            if self.data.time < (time.monotonic_ns() - self.real_start) / 1e9:
                if (time.monotonic_ns() - self.last_robot_update) / 1e9 >= 1/200:
                    self.last_robot_update = time.monotonic_ns()
                    dt = 1/200

                    # 更新行人运动学
                    self.human.update(dt)
                    self.data.mocap_pos[self.human_mocap_id] = [self.human.x, self.human.y, 0.0]
                    qw = math.cos(self.human.heading/2)
                    qz = math.sin(self.human.heading/2)
                    self.data.mocap_quat[self.human_mocap_id] = [qw,0,0,qz]

                    # 跟踪计算
                    rpos = self.data.xpos[self.robot.body_id]
                    rmat = self.data.xmat[self.robot.body_id].reshape(3,3)
                    fwd = -rmat[:,1]
                    r_hdg = math.atan2(fwd[1], fwd[0])

                    dx = self.human.x - rpos[0]
                    dy = self.human.y - rpos[1]
                    dist = math.hypot(dx, dy)
                    dist_err = dist - DESIRED_DIST

                    target_hdg = math.atan2(dy, dx)
                    hdg_err = target_hdg - r_hdg
                    hdg_err = math.atan2(math.sin(hdg_err), math.cos(hdg_err))

                    # 速度 PID
                    self.dist_sum += dist_err * dt
                    self.dist_sum = clamp(self.dist_sum, -1.0, 1.0)
                    v_adj = (DIST_KP*dist_err + DIST_KI*self.dist_sum +
                             DIST_KD*(dist_err - self.last_dist_err)/dt)
                    self.last_dist_err = dist_err

                    # 动态限速
                    ah = abs(hdg_err)
                    if ah < math.radians(20): maxv = 1.5
                    elif ah < math.radians(60): maxv = 0.8
                    elif ah < math.radians(120): maxv = 0.3
                    else: maxv = 0.1
                    v_cmd = clamp(v_adj, 0.0, maxv)

                    # 转向
                    raw_yaw = HEADING_KP*hdg_err + TURN_FF*self.human.turn_rate
                    self.yaw_f = YAW_ALPHA*raw_yaw + (1-YAW_ALPHA)*self.yaw_f
                    yaw_cmd = clamp(self.yaw_f, -MAX_YAW, MAX_YAW)

                    self.robot.set_velocity_linear_set_point(v_cmd)
                    self.robot.set_yaw(yaw_cmd)
                    self.robot.update_motor_speed()

                mujoco.mj_step(self.model, self.data)

                # 打印
                if self.data.time - self.last_print_time >= 0.5:
                    self.last_print_time = self.data.time
                    rpos = self.data.xpos[self.robot.body_id]
                    rmat = self.data.xmat[self.robot.body_id].reshape(3,3)
                    fwd = -rmat[:,1]
                    r_hdg_raw = math.atan2(fwd[1], fwd[0])
                    h_deg = math.degrees(self.human.heading) % 360
                    if h_deg > 180: h_deg -= 360
                    r_deg = math.degrees(r_hdg_raw) + 90
                    r_deg = (r_deg + 180) % 360 - 180

                    lv = self.data.joint('torso_l_wheel').qvel[0]
                    rv = self.data.joint('torso_r_wheel').qvel[0]
                    spd = ((-lv+rv)/2)*0.034

                    print(f"[t={self.data.time:.1f}s] "
                          f"人速={self.human.speed:.2f} 车速={spd:.2f} m/s | "
                          f"人角度={h_deg:.1f}° 车角度={r_deg:.1f}° | "
                          f"距离误差={dist_err:.2f}m 航向误差={math.degrees(hdg_err):.1f}°")
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        xml_path = str(pathlib.Path(__file__).parent.joinpath('sim\scene.xml'))
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0.,0.,0.])
        self.cam.distance = self.model.stat.extent*2
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
        top.addStretch()
        layout.addLayout(top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,4,0,0)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(800,600)

        self.th = SimThread(self.model, self.data)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps):
        self.statusBar().showMessage(f"Avg runtime: {fps:.0e}s  Sim time: {self.data.time:.0f}s")

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()

if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()