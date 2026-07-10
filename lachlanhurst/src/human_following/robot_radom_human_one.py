# 跟踪和控制的核心代码均已包,优化版行人轨迹随机生成，控制频率和Mujoco相机参数在runs_qt中，行人轨迹在random_human中
import math
import pathlib # 文件路径
from scipy.spatial.transform import Rotation
from sim.runs_qt import run_experiments
from sim.random_human import RandomHumanTrajectory
import numpy as np

# ========== 全局参数 ==========
WHEEL_RADIUS = 0.034
MAX_MOTOR_VEL = 200.0
# 控制平衡参数
PITCH_KP = 6.0 # 平衡角度PD控制
PITCH_KD = 1.1
SPEED_KP = 0.5 # 平衡速度PI控制
SPEED_KI = 0.07
INTEGRAL_LIMIT = 0.1
# 控制跟踪参数
DESIRED_DIST = 0.3 # 跟踪目标误差
DIST_KP = 4.0 # 跟踪控制PID: 生成目标速度调整值
DIST_KI = 0.05
DIST_KD = 0.05
HEADING_KP = -15.0 # 跟踪速度转向P控制
TURN_FF = -1.6 # 行人转向的前馈控制系数
YAW_ALPHA = 0.3 # 滤波系数
MAX_YAW = 1.5 # 限制最大转向率

def clamp(val, lo, hi): # 限制控制变量在[lo,hi]之间：vel, yaw
    return max(lo, min(hi, val))

def main():
    # ---------- 平衡控制状态变量 ----------
    velocity_linear_set_point = 0.0 # 小车期望的线速度：由跟踪环输出
    yaw_cmd_internal = 0.0 # 小车期望的转向速度
    pitch_dot_filtered = velocity_angular_filtered = 0.0 # 滤波后俯仰角角速度和车轮角速度
    speed_error_integral = 0.0
    # ---------- 跟踪控制状态变量 ----------
    dist_sum = 0.0 # 距离误差积分项
    last_dist_err = 0.0 # 上一次距离误差
    yaw_f = 0.0 # 滤波后偏航角
    last_print_time = 0.0 # 控制输出频率
    human = None   # 稍后初始化
    prev_human_pos = (0.0, 0.0)  # 用于计算实际人速

    def init_callback(model, data): # 初始化外部（main中）函数 nonlocal
        nonlocal velocity_linear_set_point, yaw_cmd_internal, pitch_dot_filtered, velocity_angular_filtered, speed_error_integral
        nonlocal human, dist_sum, last_dist_err, yaw_f, last_print_time
        prev_human_pos = (0.0, -0.3)
        # 重置所有状态
        velocity_linear_set_point = yaw_cmd_internal = pitch_dot_filtered = velocity_angular_filtered = speed_error_integral = 0.0
        dist_sum = last_dist_err = yaw_f = 0.0
        last_print_time = 0.0
        human = RandomHumanTrajectory(start_x=0.0, start_y=-0.3) # 创建行人位置
        # 平衡车初始位姿
        data.qpos[0:2] = [0.0, 0.0] # 小车世界坐标系下x,y,z轴方向位置
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # w=1, x=0, y=0, z=0 表示无旋转（直立）
        data.qvel[:] = 0.0 # 初始化平衡车速度
        data.actuator('motor_l_wheel').ctrl = [0]
        data.actuator('motor_r_wheel').ctrl = [0]

    def control_callback(model, data, dt): # 每隔0.005s进行控制
        nonlocal velocity_linear_set_point, yaw_cmd_internal, pitch_dot_filtered, velocity_angular_filtered, speed_error_integral
        nonlocal dist_sum, last_dist_err, yaw_f, last_print_time, human, prev_human_pos

        # ---------- 1. 更新行人状态 ----------
        human.update(dt) # RandomHumanTrajectory 类的实例：(human.x, human.y) 和 human.heading （弧度）
        human_mocap_id = model.body_mocapid[model.body('human').id]
        data.mocap_pos[human_mocap_id] = [human.x, human.y, 0.0]
        qw = math.cos(human.heading / 2) # w = cos(θ/2)
        qz = math.sin(human.heading / 2) # w = cos(θ/2)
        data.mocap_quat[human_mocap_id] = [qw, 0, 0, qz] # 计算行人的朝向heading

        # ---------- 2. 机器人状态 ----------
        robot_body_id = model.body('robot_body').id
        free_joint_id = model.joint('robot_body_joint').id # 获取自由关节的 qvel 起始索引
        vel_start = model.jnt_dofadr[free_joint_id]  # 自由关节有 6 个自由度：3 线速度 + 3 角速度
        linear_vel_body = data.qvel[vel_start:vel_start + 3]  # 世界坐标系下的线速度 (vx, vy, vz)
        rpos = data.xpos[robot_body_id] # 平衡车的三维向量x,y,z
        rmat = data.xmat[robot_body_id].reshape(3, 3) # 平衡车的旋转矩阵
        fwd = -rmat[:, 1] # 平衡车前进方向，沿-y轴
        r_hdg = math.atan2(fwd[1], fwd[0]) # 平衡车航向角arctan(y,x)

        # ---------- 3. 跟踪误差 ----------
        dx = human.x - rpos[0]
        dy = human.y - rpos[1]
        dist = math.hypot(dx, dy) # 行人到平衡车的直线距离
        dist_err = dist - DESIRED_DIST # 距离误差
        target_hdg = math.atan2(dy, dx) # 平衡车的目标航向角
        hdg_err = target_hdg - r_hdg
        hdg_err = math.atan2(math.sin(hdg_err), math.cos(hdg_err)) # 角度误差归一化[-Π，Π]

        # ---------- 4. 由跟踪误差计算平衡车目标速度 ----------
        dist_sum += dist_err * dt
        dist_sum = clamp(dist_sum, -1.0, 1.0) # 积分项累加
        v_adj = (DIST_KP * dist_err +
                 DIST_KI * dist_sum +
                 DIST_KD * (dist_err - last_dist_err) / dt) # 计算平衡车的目标速度
        last_dist_err = dist_err

        ah = abs(hdg_err) # 动态限速：根据航向误差调整最大目标速度，航向角过大，应先转向，不应快速前进
        if ah < math.radians(10):
            maxv = 1.5
        elif ah < math.radians(45):
            maxv = 0.8
        elif ah < math.radians(100):
            maxv = 0.3
        else:
            maxv = 0.1
        v_cmd = clamp(v_adj, 0.0, maxv)
        if abs(human.speed) < 0.01: # 保证人不动，车不动
            v_cmd = 0.0

        # ---------- 5. 由行人航向角和误差计算目标转向 ----------
        raw_yaw = HEADING_KP * hdg_err + TURN_FF * human.turn_rate # 人的航向角前馈+P控制
        yaw_f = YAW_ALPHA * raw_yaw + (1 - YAW_ALPHA) * yaw_f # 转向滤波
        yaw_cmd = clamp(yaw_f, -MAX_YAW, MAX_YAW)

        # ---------- 6. 平衡控制 ----------
        velocity_linear_set_point = v_cmd
        yaw_cmd_internal = yaw_cmd
        body_id = model.body('robot_body').id
        quat = data.xquat[body_id]
        if quat[0] == 0:
            pitch = 0.0
        else:
            rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]) # 调整顺序[x,y,z,w]
            pitch = rotation.as_euler('xyz', degrees=False)[0]
        pitch = -pitch   # 前倾为正
        pitch_dot = data.joint('robot_body_joint').qvel[-3:][0]
        pitch_dot_filtered = 0.975 * pitch_dot_filtered + 0.025 * pitch_dot # 俯仰角速度滤波，俯仰更平滑

        l_vel = data.joint('torso_l_wheel').qvel[0] # 左轮速
        r_vel = data.joint('torso_r_wheel').qvel[0] # 右轮速
        wheel_avg = (l_vel * -1 + r_vel) / 2.0
        velocity_angular_filtered = 0.975 * velocity_angular_filtered + 0.025 * wheel_avg
        actual_speed = velocity_angular_filtered * WHEEL_RADIUS # 平衡车前进速度
        vel_error = actual_speed - velocity_linear_set_point
        speed_error_integral += vel_error * dt
        speed_error_integral = clamp(speed_error_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        target_pitch = SPEED_KP * vel_error + SPEED_KI * speed_error_integral # 串级PID

        pitch_error = target_pitch - pitch
        motor_vel = PITCH_KP * pitch_error - PITCH_KD * pitch_dot_filtered
        motor_vel = motor_vel / WHEEL_RADIUS
        motor_vel = clamp(motor_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        left_vel = -motor_vel + yaw_cmd_internal # 平衡轮速+差速转向
        right_vel = motor_vel + yaw_cmd_internal
        left_vel = clamp(left_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        right_vel = clamp(right_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        data.actuator('motor_l_wheel').ctrl = [left_vel]
        data.actuator('motor_r_wheel').ctrl = [right_vel]

        # ---------- 7. 打印 ----------
        if data.time - last_print_time >= 0.5:
            last_print_time = data.time
            actual_speed_print = ((-l_vel + r_vel) / 2.0) * WHEEL_RADIUS
            # actual_forward_speed = np.dot(linear_vel_body,fwd)
            dx_human = human.x - prev_human_pos[0]
            dy_human = human.y - prev_human_pos[1]
            actual_human_speed = math.hypot(dx_human, dy_human) / 0.5
            prev_human_pos = (human.x, human.y)

            h_deg = math.degrees(human.heading) % 360
            h_deg = (h_deg + 180) % 360 - 180
            r_hdg_deg = math.degrees(r_hdg) + 90
            r_hdg_deg = (r_hdg_deg + 180) % 360 - 180
            print(f"[t={data.time:.1f}s] "
                  f"人位置=({human.x:.2f},{human.y:.2f}) 人速={actual_human_speed:.2f} 前向速度={actual_speed_print:.2f} | "
                  f"人角度={h_deg:.1f}° 车角度={r_hdg_deg:.1f}° | "
                  f"距离误差={dist_err:.2f}m 航向误差={math.degrees(hdg_err):.1f}°")

    # ---------- 启动仿真（使用稳定的 Qt 仿真模块） ----------
    xml_path = str(pathlib.Path(__file__).parent.joinpath('sim\scene.xml'))
    run_experiments(xml_path, control_callback, init_callback)

if __name__ == "__main__":
    main()