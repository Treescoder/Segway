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
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from segway_ball_pid_bag_front import SegwayPID      # 力矩环平衡控制器
from human_follower import HumanFollower             # 跟踪控制器
from scipy.spatial.transform import Rotation

WHEEL_RADIUS = 0.24

# ================= OpenGL 设置 =================
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

# ================= 仿真线程 =================
class SimThread(QThread):
    def __init__(self, model, data):
        super().__init__()
        self.model, self.data = model, data
        self.running = True
        self.robot = SegwayPID(model, data)
        self.follower = HumanFollower(desired_dist=1.5)

        self.human_speed_cmd = 0.0
        self.human_turn_target = 0.0
        self.human_speed = 0.0
        self.human_turn = 0.0
        self.human_x = 0.0
        self.human_y = -1.5
        self.human_heading = 0.0

        self.speed_ref = 0.0
        self.yaw_ref = 0.0
        self.last_max_v = 0.0
        self.last_yaw_cmd = 0.0

        self.body_id = model.body('segway').id
        self.human_mocap_id = model.body_mocapid[model.body('human').id]
        self.l_dof = model.jnt_dofadr[model.joint('torso_l_wheel').id]
        self.r_dof = model.jnt_dofadr[model.joint('torso_r_wheel').id]
        self.stable_duration = 0.5
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

                    # ---- 行人运动学 ----
                    if self.data.time < self.stable_duration:
                        self.human_speed = 0.0
                        self.human_turn = 0.0
                    else:
                        self.human_speed = self._smooth_ramp(self.human_speed, self.human_speed_cmd, 10.0)

                        heading_err_human = math.atan2(math.sin(self.human_turn_target - self.human_heading),
                                                       math.cos(self.human_turn_target - self.human_heading))
                        max_turn_rate = 2.5
                        Kp_human = 6.0
                        desired_turn = max(-max_turn_rate, min(max_turn_rate, Kp_human * heading_err_human))
                        self.human_turn = self._smooth_ramp(self.human_turn, desired_turn, 3.0)

                    self.human_heading += self.human_turn * dt
                    self.human_x += self.human_speed * math.sin(self.human_heading) * dt
                    self.human_y -= self.human_speed * math.cos(self.human_heading) * dt
                    self.data.mocap_pos[self.human_mocap_id] = [self.human_x, self.human_y, 0.3]
                    qw = math.cos(self.human_heading/2)
                    qz = math.sin(self.human_heading/2)
                    self.data.mocap_quat[self.human_mocap_id] = [qw, 0, 0, qz]

                    # ---- 车跟踪控制 ----
                    if self.data.time < self.stable_duration:
                        self.speed_ref = 0.0
                        self.yaw_ref = 0.0
                    else:
                        robot_pos = self.data.xpos[self.body_id]
                        robot_mat = self.data.xmat[self.body_id].reshape(3,3)
                        v_cmd, yaw_cmd, dist_err, heading_err, max_v, v_adj = self.follower.compute(
                            robot_pos, robot_mat, self.human_x, self.human_y, self.human_heading, self.human_speed, self.human_turn, dt)
                        self.last_max_v = max_v
                        self.last_yaw_cmd = yaw_cmd
                        self.v_adj = v_adj
                        self.speed_ref = self._smooth_ramp(self.speed_ref, v_cmd, 1.5)
                        self.yaw_ref = self._smooth_ramp_angle(self.yaw_ref, yaw_cmd, 4.0)

                    self.robot.set_velocity_linear_set_point(self.speed_ref)
                    self.robot.set_yaw(self.yaw_ref)
                    self.robot.update_motor_torque()

                mujoco.mj_step(self.model, self.data)

                if self.data.time - self.last_print_time >= 0.5:
                    self.last_print_time = self.data.time
                    self._print_debug_info()
            else:
                time.sleep(0.00001)

    def _smooth_ramp(self, cur, tar, acc):
        err = tar - cur
        step = acc * 0.005
        if err > step: return cur + step
        elif err < -step: return cur - step
        else: return tar

    def _smooth_ramp_angle(self, cur, tar, acc):
        err = math.atan2(math.sin(tar-cur), math.cos(tar-cur))
        step = acc * 0.005
        if err > step: return cur + step
        elif err < -step: return cur - step
        else: return cur + err

    def _print_debug_info(self):
        try:
            robot_pos = self.data.xpos[self.body_id]
            robot_mat = self.data.xmat[self.body_id].reshape(3,3)
            fwd = -robot_mat[:,1]
            robot_heading_raw = math.atan2(fwd[1], fwd[0])

            human_deg = math.degrees(self.human_heading) % 360
            if human_deg > 180: human_deg -= 360
            target_deg = math.degrees(self.human_turn_target)
            robot_deg = math.degrees(robot_heading_raw) + 90
            robot_deg = (robot_deg + 180) % 360 - 180

            lv = self.data.qvel[self.l_dof]
            rv = self.data.qvel[self.r_dof]
            car_speed = (lv + rv) / 2 * WHEEL_RADIUS

            dx = self.human_x - robot_pos[0]
            dy = self.human_y - robot_pos[1]
            dist = math.hypot(dx, dy)
            dist_err = dist - self.follower.desired_dist
            target_heading = math.atan2(dy, dx)
            heading_err = target_heading - robot_heading_raw
            heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

            print(f"[t={self.data.time:.1f}s] "
                  f"速度(人/车): {self.human_speed:.2f}/{car_speed:.2f} m/s (ref={self.speed_ref:.2f}) | "
                  f"人({self.human_x:.2f},{self.human_y:.2f}) hdg={human_deg:.1f}° target={target_deg:.1f}° | "
                  f"车({robot_pos[0]:.2f},{robot_pos[1]:.2f}) | "
                  f"距离误差: {dist_err:.2f}m 航向误差: {math.degrees(heading_err):.1f}° | "
                  f"max_v={self.last_max_v:.1f} yaw_cmd={self.last_yaw_cmd:.2f}")
        except Exception as e:
            print(f"Debug print error: {e}")

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()
        self.follower.reset()
        self.human_x = 0.0
        self.human_y = -1.5
        self.human_heading = 0.0
        self.human_speed = 0.0
        self.human_turn = 0.0
        self.human_speed_cmd = 0.0
        self.human_turn_target = 0.0
        self.speed_ref = 0.0
        self.yaw_ref = 0.0
        self.last_print_time = 0.0

    def set_human_speed(self, speed):
        self.human_speed_cmd = speed

    def set_human_turn_target(self, target_rad):
        self.human_turn_target = target_rad

# ================= 主窗口 =================
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        xml_path = pathlib.Path(__file__).parent.joinpath('xml/scene.xml')
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = 0.001
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
        self.human_speed_slider = QSlider(Qt.Horizontal)
        self.human_speed_slider.setMinimum(0)
        self.human_speed_slider.setMaximum(4.16 * 1000)
        self.human_speed_slider.setValue(0)
        self.human_speed_slider.valueChanged.connect(lambda v: self.th.set_human_speed(v/1000))
        speed_layout.addWidget(QLabel("Human Speed"))
        speed_layout.addWidget(self.human_speed_slider)

        turn_layout = QHBoxLayout()
        self.human_turn_slider = QSlider(Qt.Horizontal)
        self.human_turn_slider.setMinimum(-1000)
        self.human_turn_slider.setMaximum(1000)
        self.human_turn_slider.setValue(0)
        self.human_turn_slider.valueChanged.connect(
            lambda v: self.th.set_human_turn_target(math.radians(v / 1000 * 90)))
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
        self.resize(1650,850)
        self.move(1950, 20)

        self.th = SimThread(self.model, self.data)
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