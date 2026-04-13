import time
import mujoco
import mujoco.viewer
import numpy as np
import pathlib

from robot_lqr import RobotLqr


def main():
    # ====== 加载模型 ======
    model_path = pathlib.Path(__file__).parent / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # ====== 初始化控制器 ======
    robot = RobotLqr(model, data)

    # 控制输入（替代UI滑块）
    speed = 0.01
    yaw = 0.0

    # 时间控制（模拟你原来的 real_time 机制）
    real_time_start = time.monotonic()
    last_control_time = time.monotonic()

    # ====== 启动viewer ======
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            sim_time = data.time
            real_time = time.monotonic() - real_time_start

            # 防止仿真跑太快（和你原代码一致）
            if sim_time < real_time:

                # ====== 200Hz 控制 ======
                if time.monotonic() - last_control_time >= 1/200:
                    last_control_time = time.monotonic()

                    # 设置输入
                    robot.set_velocity_linear_set_point(speed)
                    robot.set_yaw(yaw)

                    # LQR 控制
                    robot.update_motor_speed()

                # ====== 物理步进 ======
                mujoco.mj_step(model, data)

            else:
                time.sleep(0.00001)

            viewer.sync()


if __name__ == "__main__":
    main()