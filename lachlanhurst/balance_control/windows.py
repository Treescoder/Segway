import pathlib
import mujoco
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSlider, QVBoxLayout, QWidget
)
from lachlanhurst.balance_control.update import UpdateSimThread
from lachlanhurst.balance_control.viewer import Viewport
format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2, 0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()                                                                                # 初始化父类

        self.model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent / "xml/scene.xml"))  # 加载MuJoCo模型
        self.data = mujoco.MjData(self.model)                                                              # 创建仿真数据
        self.cam = self.create_free_camera()                                                               # 创建自由相机
        self.opt = mujoco.MjvOption()                                                                      # 创建渲染选项
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)                                        # 创建渲染场景

        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)                     # 创建OpenGL窗口
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())        # 设置DPI缩放
        self.viewport.updateRuntime.connect(self.show_runtime)                                             # 连接FPS显示信号

        self.move(200, 50)                                                                                # 设置窗口位置
        self.resize(800, 600)                                                                               # 设置窗口大小

        layout = QVBoxLayout()                                                                            # 主布局
        top = QHBoxLayout()                                                                               # 顶部布局
        top.setSpacing(8)                                                                                 # 控件间距
        top.setContentsMargins(8, 0, 8, 0)                                                                # 顶部边距

        reset = QPushButton("Reset")                                                                      # Reset按钮
        reset.setMinimumWidth(90)                                                                         # 按钮宽度
        reset.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)                                     # 按钮尺寸策略
        reset.clicked.connect(self.reset_simulation)                                                      # 绑定复位函数
        top.addWidget(reset)                                                                              # 添加按钮

        top.addWidget(self.create_top())                                                                  # 添加控制面板

        layout.addLayout(top)                                                                             # 添加顶部布局
        layout.addWidget(QWidget.createWindowContainer(self.viewport))                                    # 嵌入OpenGL窗口
        layout.setContentsMargins(0, 4, 0, 0)                                                             # 主布局边距
        layout.setStretch(1, 1)                                                                           # OpenGL区域拉伸

        widget = QWidget()                                                                                # 创建中央Widget
        widget.setLayout(layout)                                                                          # 设置主布局
        self.setCentralWidget(widget)                                                                     # 设置中央窗口

        self.th = UpdateSimThread(self.model, self.data, self)                                            # 创建仿真线程
        self.th.start()                                                                                   # 启动仿真线程

    @Slot(float)
    def show_runtime(self, fps: float):
        self.statusBar().showMessage(
            f"Average runtime: {fps:.0e}s\t"
            f"Simulation time: {self.data.time:.0f}s"
        )

    def add_slider(self, layout, name, minimum, maximum, value, slot):  # 创建滑块
        row = QHBoxLayout()  # 水平布局
        label = QLabel(name)
        label.setFixedWidth(60)  # 参数名称
        slider = QSlider(Qt.Horizontal)  # 滑块
        slider.setRange(int(minimum * 1000), int(maximum * 1000))  # 范围
        slider.setValue(int(value * 1000))  # 默认值
        slider.valueChanged.connect(slot)  # 信号
        text = QLabel(f"{value:.3f}");
        text.setFixedWidth(60)  # 数值显示
        row.addWidget(label);
        row.addWidget(slider);
        row.addWidget(text)  # 添加控件
        layout.addLayout(row)  # 添加布局
        return slider, text  # 返回对象

    def create_top(self):  # 控制面板
        layout = QVBoxLayout()  # 主布局

        self.speed_slider, self.speed_value = self.add_slider(layout, "Speed", -4, 4, 0, self._speed_changed)  # 速度
        self.yaw_slider, self.yaw_value = self.add_slider(layout, "Yaw", -10, 10, 0, self._yaw_changed)  # 转向
        self.kp_slider, self.kp_value = self.add_slider(layout, "kp", 0, 100, 10, self._kp_changed)  # Pitch KP
        self.kd_slider, self.kd_value = self.add_slider(layout, "kd", 0, 100, 0.01, self._kd_changed)  # Pitch KD
        self.kv_slider, self.kv_value = self.add_slider(layout, "kv", 0, 100, 1, self._kv_changed)  # Speed KP
        self.ki_slider, self.ki_value = self.add_slider(layout, "ki", 0, 100, 0.01, self._ki_changed)  # Speed KI

        box = QGroupBox("Robot Control")  # 分组框
        box.setLayout(layout)  # 设置布局
        return box  # 返回

    def update_slider(self, value, label, setter):  # 更新参数
        value /= 1000  # 转换单位
        label.setText(f"{value:.3f}")  # 更新显示
        setter(value)  # 写入控制线程

    def _speed_changed(self, value): self.update_slider(value, self.speed_value, self.th.set_speed)  # Speed
    def _yaw_changed(self, value): self.update_slider(value, self.yaw_value, self.th.set_yaw)  # Yaw
    def _kp_changed(self, value): self.update_slider(value, self.kp_value, self.th.set_pitch_kp)  # Pitch KP
    def _kd_changed(self, value): self.update_slider(value, self.kd_value, self.th.set_pitch_kd)  # Pitch KD
    def _kv_changed(self, value): self.update_slider(value, self.kv_value, self.th.set_speed_kp)  # Speed KP
    def _ki_changed(self, value): self.update_slider(value, self.ki_value, self.th.set_speed_ki)  # Speed KI

    def create_free_camera(self):  # 创建自由相机
        cam = mujoco.MjvCamera()  # 相机对象
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # 自由相机
        cam.fixedcamid = -1  # 不使用固定相机
        cam.lookat = np.zeros(3)  # 观察中心
        cam.distance = self.model.stat.extent * 0.7  # 相机距离
        cam.elevation, cam.azimuth = -30, 45  # 俯仰/方位角
        return cam  # 返回相机

    def reset_simulation(self):  # 重置仿真
        self.speed_slider.setValue(0)
        self.yaw_slider.setValue(0)  # 控制量归零
        self.kp_slider.setValue(10000)
        self.kd_slider.setValue(10)  # Pitch参数
        self.kv_slider.setValue(1000)
        self.ki_slider.setValue(10)  # Speed参数
        mujoco.mj_resetData(self.model, self.data)  # 重置状态
        mujoco.mj_forward(self.model, self.data)  # 更新运动学
        self.th.reset()  # 重置控制线程