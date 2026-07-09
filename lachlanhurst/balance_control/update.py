import time                                              # 时间模块
import mujoco                                            # MuJoCo物理引擎
from scipy.spatial.transform import Rotation as R        # 四元数转欧拉角
from PySide6.QtCore import QThread                       # Qt线程
from PySide6.QtGui import QSurfaceFormat                 # OpenGL显示格式
from lachlanhurst.balance_control.robot_pid import RobotPID  # 平衡车PID控制器
format = QSurfaceFormat()                                # 创建OpenGL格式
format.setDepthBufferSize(24)                            # 设置24位深度缓冲
format.setStencilBufferSize(8)                           # 设置8位模板缓冲
format.setSamples(4)                                     # 开启4倍MSAA抗锯齿
format.setSwapInterval(1)                                # 开启垂直同步
format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)      # 使用双缓冲
format.setVersion(2, 0)                                  # 使用OpenGL 2.0
format.setRenderableType(QSurfaceFormat.OpenGL)          # 使用OpenGL渲染
format.setProfile(QSurfaceFormat.CompatibilityProfile)   # 使用兼容模式
QSurfaceFormat.setDefaultFormat(format)                  # 设置全局默认OpenGL格式

class UpdateSimThread(QThread):
    """MuJoCo仿真线程"""

    WHEEL_RADIUS = 0.034                                 # 车轮半径(m)
    CONTROL_FREQ = 200                                   # 控制频率(Hz)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None):
        super().__init__(parent)                         # 初始化线程
        self.model = model                               # MuJoCo模型
        self.data = data                                 # MuJoCo数据
        self.running = True                              # 线程运行标志
        self.robot = RobotPID(model, data)               # 控制器对象
        self.speed = 0.0                                 # 目标速度(m/s)
        self.yaw = 0.0                                   # 目标转向
        self.body_id = model.body("robot_body").id       # 车体Body编号
        self.l_joint_id = model.joint("torso_l_wheel").id# 左轮关节编号
        self.r_joint_id = model.joint("torso_r_wheel").id# 右轮关节编号
        self.reset()                                     # 初始化计时器

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start    # 返回真实运行时间(ns)

    def run(self):
        while self.running:                              # 主循环

            if self.data.time >= self.real_time / 1e9:   # 仿真超过真实时间
                time.sleep(1e-5)                         # 防止CPU空转
                continue

            if (time.monotonic_ns() - self.last_robot_update) / 1e9 >= 1 / self.CONTROL_FREQ:
                self.last_robot_update = time.monotonic_ns() # 更新时间戳

                self.robot.set_velocity_linear_set_point(self.speed)  # 设置目标速度
                self.robot.set_yaw(self.yaw)                          # 设置目标转向
                self.robot.update_motor_torque()                      # 更新控制器输出

            mujoco.mj_step(self.model, self.data)        # 仿真一步

            if self.data.time - self.last_print_time >= 0.5:  # 每0.5秒打印一次
                self.last_print_time = self.data.time
                self._print_debug_info()

    def _print_debug_info(self):
        try:
            quat = self.data.xquat[self.body_id]         # 读取车体四元数

            yaw, roll, pitch = R.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]).as_euler("zyx", degrees=True)          # 转欧拉角

            l_vel = self.data.qvel[self.l_joint_id]      # 左轮速度
            r_vel = self.data.qvel[self.r_joint_id]      # 右轮速度

            speed = (-l_vel + r_vel) * 0.5 * self.WHEEL_RADIUS   # 实际线速度

            print(
                f"[t={self.data.time:.2f}s] "
                f"pitch={pitch:6.2f}° "
                f"roll={roll:6.2f}° "
                f"yaw={yaw:6.2f}° | "
                f"actual_speed={speed:6.2f} m/s | "
                f"target_speed={self.speed:5.2f} m/s | "
                f"L:{l_vel:6.2f} rad/s ctrl={self.data.ctrl[0]:6.2f} | "
                f"R:{r_vel:6.2f} rad/s ctrl={self.data.ctrl[1]:6.2f}"
            )                                            # 输出调试信息

        except Exception as e:
            print(f"Debug print error: {e}")             # 输出异常信息

    def stop(self):
        self.running = False                             # 停止线程
        self.wait()                                      # 等待线程结束

    def reset(self):
        self.real_time_start = time.monotonic_ns()       # 记录开始时间
        self.last_robot_update = self.real_time_start    # 初始化控制更新时间
        self.last_print_time = 0.0                       # 初始化打印时间
        self.robot.reset()                               # 重置控制器

    def set_speed(self, speed: float):
        self.speed = speed                               # 设置目标速度

    def set_yaw(self, yaw: float):
        self.yaw = yaw                                   # 设置目标转向

    def set_pitch_kp(self, kp):
        self.robot.set_pitch_kp(kp)                      # 设置俯仰KP

    def set_pitch_kd(self, kd):
        self.robot.set_pitch_kd(kd)                      # 设置俯仰KD

    def set_speed_kp(self, kp):
        self.robot.speed_kp = kp                         # 设置速度KP

    def set_speed_ki(self, ki):
        self.robot.speed_ki = ki                         # 设置速度KI