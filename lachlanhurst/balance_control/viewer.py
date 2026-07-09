from collections import deque                          # 双端队列
import time                                            # 时间模块
import mujoco                                          # MuJoCo物理引擎
import numpy as np                                     # 数值计算
from PySide6.QtCore import QTimer, Qt, Signal          # Qt定时器、事件、信号
from PySide6.QtGui import QSurfaceFormat               # OpenGL显示格式
from PySide6.QtOpenGL import QOpenGLWindow             # OpenGL渲染窗口
format = QSurfaceFormat()                              # 创建OpenGL格式
format.setDepthBufferSize(24)                          # 设置24位深度缓冲
format.setStencilBufferSize(8)                         # 设置8位模板缓冲
format.setSamples(4)                                   # 开启4倍MSAA抗锯齿
format.setSwapInterval(1)                              # 开启垂直同步
format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)    # 使用双缓冲
format.setVersion(2, 0)                   # 使用OpenGL 2.0
format.setRenderableType(QSurfaceFormat.OpenGL)        # 使用OpenGL渲染
format.setProfile(QSurfaceFormat.CompatibilityProfile) # 使用兼容模式
QSurfaceFormat.setDefaultFormat(format)                # 设置全局默认OpenGL格式

class Viewport(QOpenGLWindow):
    updateRuntime = Signal(float)                         # 渲染耗时信号

    def __init__(self, model, data, cam, opt, scn):
        super().__init__()                                # 初始化窗口

        self.model = model                                # MuJoCo模型
        self.data = data                                  # MuJoCo数据
        self.cam = cam                                    # 相机对象
        self.opt = opt                                    # 渲染选项
        self.scn = scn                                    # 场景对象

        self.width = 0                                    # 窗口宽度
        self.height = 0                                   # 窗口高度
        self.scale = 1.0                                  # DPI缩放比例
        self.__last_pos = None                            # 上次鼠标位置

        self.runtime = deque(maxlen=1000)                 # 最近1000帧耗时

        self.timer = QTimer()                             # 刷新定时器
        self.timer.setInterval(1000 / 60)                 # 60 FPS刷新
        self.timer.timeout.connect(self.update)           # 定时刷新窗口
        self.timer.start()                                # 启动定时器

        self.body_id = model.body("robot_body").id        # 机器人Body编号

    def mousePressEvent(self, event):
        self.__last_pos = event.position()  # 记录鼠标位置

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:  # 右键平移
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:  # 左键旋转
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:  # 中键缩放
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return  # 无按键退出

        pos = event.position()  # 当前鼠标位置
        dx = pos.x() - self.__last_pos.x()  # 水平位移
        dy = pos.y() - self.__last_pos.y()  # 垂直位移

        mujoco.mjv_moveCamera(  # 移动相机
            self.model,
            action,
            dx / self.height,
            dy / self.height,
            self.scn,
            self.cam,
        )

        self.__last_pos = pos  # 更新鼠标位置

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.0005 * event.angleDelta().y(),
            self.scn,
            self.cam
        )

    def initializeGL(self):
        self.con = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_100
        )

    def resizeGL(self, w, h):
        self.width = w                              # 保存新的窗口宽度
        self.height = h                             # 保存新的窗口高度

    def setScreenScale(self, scaleFactor: float) -> None:
        self.scale = scaleFactor

    def paintGL(self) -> None:
        body_pos = self.data.xpos[self.body_id]

        self.cam.lookat = body_pos.copy()

        t = time.time()

        mujoco.mjv_updateScene(
            self.model,                              # 模型
            self.data,                               # 当前仿真状态
            self.opt,                                # 渲染选项
            None,                                    # perturb（交互扰动），这里不用
            self.cam,                                # 当前相机
            mujoco.mjtCatBit.mjCAT_ALL,              # 渲染所有对象
            self.scn                                 # 输出到场景对象
        )

        viewport = mujoco.MjrRect(
            0,
            0,
            int(self.width * self.scale),            # 实际像素宽度
            int(self.height * self.scale)            # 实际像素高度
        )

        mujoco.mjr_render(
            viewport,
            self.scn,
            self.con
        )

        # 保存这一帧渲染耗时
        self.runtime.append(time.time() - t)

        # 计算最近1000帧平均耗时
        self.updateRuntime.emit(np.average(self.runtime))