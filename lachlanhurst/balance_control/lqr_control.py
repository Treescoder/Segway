from PySide6.QtGui import QSurfaceFormat             # OpenGL显示格式
from PySide6.QtWidgets import QApplication           # Qt应用程序
from lachlanhurst.balance_control.windows import Window  # 主窗口
format = QSurfaceFormat()                            # 创建OpenGL格式
format.setDepthBufferSize(24)                        # 设置24位深度缓冲
format.setStencilBufferSize(8)                       # 设置8位模板缓冲
format.setSamples(4)                                 # 开启4倍MSAA抗锯齿
format.setSwapInterval(1)                            # 开启垂直同步
format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)  # 使用双缓冲
format.setVersion(2, 0)                              # 使用OpenGL 2.0
format.setRenderableType(QSurfaceFormat.OpenGL)      # 使用OpenGL渲染
format.setProfile(QSurfaceFormat.CompatibilityProfile)  # 使用兼容模式
QSurfaceFormat.setDefaultFormat(format)              # 设置全局默认OpenGL格式



if __name__ == "__main__":
    # 创建Qt应用
    app = QApplication()
    # 创建主窗口
    w = Window()
    # 显示窗口
    w.show()
    # Qt事件循环
    # 程序会一直停留在这里
    app.exec()
    # 用户关闭窗口后停止仿真线程
    w.th.stop()