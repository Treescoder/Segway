from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PySide6.QtCore import Qt

class ControlPanel(QWidget):
    """简单的滑块面板，用于设置目标速度和偏航率。
       主程序直接读取 self.target_speed 和 self.target_yaw_rate 即可。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.target_speed = 0.0
        self.target_yaw_rate = 0.0
        self.setWindowTitle("Golf PIPD Control")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # 速度滑块
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed (m/s)")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(-4000)   # -4.0
        self.speed_slider.setMaximum(4000)    #  4.0
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self._on_speed)
        self.speed_value_label = QLabel("0.00")
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)

        # 偏航滑块
        yaw_layout = QHBoxLayout()
        yaw_label = QLabel("Yaw Rate (rad/s)")
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-10000)    # -10.0
        self.yaw_slider.setMaximum(10000)     #  10.0
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self._on_yaw)
        self.yaw_value_label = QLabel("0.00")
        yaw_layout.addWidget(yaw_label)
        yaw_layout.addWidget(self.yaw_slider)
        yaw_layout.addWidget(self.yaw_value_label)

        layout.addLayout(speed_layout)
        layout.addLayout(yaw_layout)
        self.setLayout(layout)

    def _on_speed(self, value):
        self.target_speed = value / 1000.0
        self.speed_value_label.setText(f"{self.target_speed:.2f}")

    def _on_yaw(self, value):
        self.target_yaw_rate = value / 1000.0
        self.yaw_value_label.setText(f"{self.target_yaw_rate:.2f}")