import math
import random

class RandomHumanTrajectory:
    def __init__(self, start_x=0.0, start_y=-0.3, start_heading=0.0):
        self.x = start_x
        self.y = start_y
        self.heading = start_heading
        self.speed = 0.0
        self.turn_rate = 0.0
        self.target_speed = random.uniform(1.5, 3.0)
        self.target_turn = random.uniform(-0.2, 0.2)
        self.time_since_change = 0.0
        self.change_interval = random.uniform(3.0, 6.0)   # 保持3~6秒

    def update(self, dt):
        self.time_since_change += dt
        if self.time_since_change >= self.change_interval:
            self.time_since_change = 0.0
            self.target_speed = random.uniform(1.5, 3.0)   # 快速走
            self.target_turn = random.uniform(-1.5, 1.5)   # 小转向
            self.change_interval = random.uniform(3.0, 6.0)

        # 快速跟踪目标（几乎瞬间达到）
        speed_diff = self.target_speed - self.speed
        turn_diff = self.target_turn - self.turn_rate

        # 每步调整20%的差距（约0.5秒达到目标）
        self.speed += speed_diff * 0.2
        self.turn_rate += turn_diff * 0.2

        # 限制范围
        self.speed = max(0.0, min(3.5, self.speed))
        self.turn_rate = max(-0.5, min(0.5, self.turn_rate))

        self.heading += self.turn_rate * dt
        self.x += self.speed * math.sin(self.heading) * dt
        self.y -= self.speed * math.cos(self.heading) * dt