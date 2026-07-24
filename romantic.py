import random
from math import sin, cos, pi
import tkinter as tk
from tkinter import Toplevel, Label

# 屏幕尺寸
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# 爱心参数 - 进一步增大爱心尺寸
HEART_SCALE = 25
HEART_OFFSET_Y = -200  # 爱心整体上移200像素

# 关心的话语列表 - 减少不同消息数量但允许重复
BASE_MESSAGES = [
    "别熬夜💤", "我想你❤️", "多喝水💧", "按时吃饭🍚",
    "记得添衣🌤️", "注意休息💼", "要开心😊", "你最重要⭐", "爱你哦💖",
    "照顾好自己🤗", "梦里见🌙", "等你回来🏠", "注意安全🛡️", "别太累💕",
    "记得想我📱", "别皱眉🙏", "么么哒😘", "早点休息🌙", "好想你💕"
]

# 卡片数量 - 大幅增加
NUM_CARDS = 100

# 好看的颜色列表 - 精心挑选的柔和、明亮颜色
BEAUTIFUL_COLORS = [
    "#FFB6C1",  # 浅粉红
    "#FFA07A",  # 浅鲑肉色
    "#FFD700",  # 金黄色
    "#98FB98",  # 苍绿色
    "#87CEEB",  # 天蓝色
    "#DDA0DD",  # 梅红色
    "#F0E68C",  # 卡其色
    "#FFA500",  # 橙色
    "#FF69B4",  # 热粉红
    "#00CED1",  # 深 turquoise
    "#FF6347",  # 番茄红
    "#7B68EE",  # 中紫色
    "#32CD32",  # 酸橙绿
    "#FF1493",  # 深粉红
    "#00BFFF",  # 深天蓝
    "#ADFF2F",  # 绿黄色
    "#FF4500",  # 橙红色
    "#DA70D6",  # 兰花紫
    "#20B2AA",  # 浅海洋绿
    "#FF8C00",  # 暗橙色
]

# 生成卡片内容，允许重复
MESSAGES = []
for i in range(NUM_CARDS):
    MESSAGES.append(random.choice(BASE_MESSAGES))


def heart_function(t, scale=HEART_SCALE):
    """计算爱心形状的坐标"""
    x = 16 * (sin(t) ** 3)
    y = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t)
    # 缩放并居中，同时上移
    x = x * scale + SCREEN_WIDTH / 2
    y = -y * scale + SCREEN_HEIGHT / 2 + HEART_OFFSET_Y  # 负号是为了翻转y轴，加上偏移量上移
    return int(x), int(y)


class ExtraLargeHeart:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("给亲爱的你")
        self.root.geometry("400x150+100+100")
        self.root.configure(bg="#FFE4E1")
        # 添加标题
        title_label = Label(
            self.root,
            text="给最爱的你\n关心话语将自动组成超大爱心",
            font=("微软雅黑", 14),
            fg="#FF1493",
            bg="#FFE4E1",
            justify="center"
        )
        title_label.pack(expand=True, fill="both", pady=10)
        # 进度标签
        self.progress_label = Label(
            self.root,
            text="准备开始...",
            font=("微软雅黑", 11),
            fg="#FF69B4",
            bg="#FFE4E1"
        )
        self.progress_label.pack(pady=5)
        # 存储所有标签窗口
        self.windows = []
        self.current_index = 0
        self.is_scattered = False  # 标记卡片是否已散开
        self.scatter_animation_running = False  # 标记散开动画是否正在运行
        self.final_positions = {}  # 存储最终位置
        # 计算爱心轮廓上的点
        self.heart_points = []
        num_points = NUM_CARDS
        for i in range(num_points):
            t = 2 * pi * i / num_points
            x, y = heart_function(t)
            self.heart_points.append((x, y))

    def start_animation(self):
        """开始自动动画"""
        self.progress_label.config(text="动画开始...")
        self.root.after(150, self.show_next_message)  # 0.15秒后开始

    def show_next_message(self):
        """显示下一个消息"""
        if self.current_index >= len(MESSAGES):
            # 所有消息显示完毕
            self.progress_label.config(text=f"完成! 共{len(MESSAGES)}条关心话语组成了爱心")
            return
        # 更新进度
        self.progress_label.config(text=f"显示第 {self.current_index + 1}/{len(MESSAGES)} 条")
        # 创建新窗口 - 保持小尺寸
        message = MESSAGES[self.current_index]
        x, y = self.heart_points[self.current_index]
        window = Toplevel(self.root)
        window.title(f"关心 #{self.current_index + 1}")
        # 使用更大的窗口宽度，高度不变
        window_width = 200  # 进一步增加宽度
        window_height = 70
        window.geometry(f"{window_width}x{window_height}+{x - window_width // 2}+{y - window_height // 2}")

        # 为每个卡片随机选择一个好看的颜色
        card_color = random.choice(BEAUTIFUL_COLORS)
        window.configure(bg=card_color)
        # 添加消息标签 - 使用更大的字体
        label = Label(
            window,
            text=message,
            font=("微软雅黑", 13, "bold"),
            fg="#FFFFFF",  # 白色文字，与彩色背景形成对比
            bg=card_color,  # 与窗口背景一致
            wraplength=180,  # 进一步增加换行宽度以适应更宽的卡片
            justify="center"
        )
        label.pack(expand=True, fill="both", padx=5, pady=3)
        # 添加序号标签 - 使用更大的字体
        index_label = Label(
            window,
            text=f"{self.current_index + 1}",
            font=("微软雅黑", 9, "bold"),
            fg="#FFFFFF",  # 白色文字
            bg=card_color  # 与窗口背景一致
        )
        index_label.pack(side="bottom", pady=1)

        # 绑定点击事件，点击后所有卡片散开
        window.bind("<Button-1>", lambda event: self.scatter_cards())
        label.bind("<Button-1>", lambda event: self.scatter_cards())
        index_label.bind("<Button-1>", lambda event: self.scatter_cards())
        self.windows.append(window)
        self.current_index += 1
        # 设置下一个窗口的出现时间 - 缩短间隔
        if self.current_index < len(MESSAGES):
            delay = 80  # 0.08秒间隔，极速动画
            self.root.after(delay, self.show_next_message)
        else:
            # 所有窗口创建完毕
            self.root.after(500, self.finalize_animation)

    def scatter_cards(self):
        """点击卡片后，所有卡片随机散开的动画效果"""
        if self.is_scattered:
            return  # 如果已经散开，不再重复执行

        self.is_scattered = True
        self.scatter_animation_running = True  # 标记动画开始
        self.progress_label.config(text="卡片正在散开...")

        # 为每个窗口生成均匀分布的目标位置
        scatter_targets = []
        valid_windows = []

        # 将屏幕分成网格区域，确保均匀分布
        grid_cols = 10  # 横向分10格
        grid_rows = 8  # 纵向分8格
        cell_width = SCREEN_WIDTH / grid_cols
        cell_height = SCREEN_HEIGHT / grid_rows

        # 创建所有可能的网格位置
        grid_positions = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                # 计算网格中心位置，加上随机偏移
                center_x = col * cell_width + cell_width / 2
                center_y = row * cell_height + cell_height / 2

                # 在网格内随机偏移（±30像素）
                offset_x = random.randint(-30, 30)
                offset_y = random.randint(-30, 30)

                final_x = int(center_x + offset_x - 100)  # 减去窗口宽度的一半（200/2）
                final_y = int(center_y + offset_y - 35)  # 减去窗口高度的一半（70/2）

                # 确保不超出边界
                final_x = max(0, min(final_x, SCREEN_WIDTH - 200))
                final_y = max(0, min(final_y, SCREEN_HEIGHT - 70))

                grid_positions.append((final_x, final_y))

        # 打乱网格位置顺序，让分布更随机
        random.shuffle(grid_positions)

        for idx, window in enumerate(self.windows):
            try:
                # 检查窗口是否还存在
                if not window.winfo_exists():
                    continue

                # 从网格位置中选择一个目标位置
                if idx < len(grid_positions):
                    target_x, target_y = grid_positions[idx]
                else:
                    # 如果窗口数量超过网格数，使用随机位置
                    target_x = random.randint(0, SCREEN_WIDTH - 200)
                    target_y = random.randint(0, SCREEN_HEIGHT - 70)

                valid_windows.append(window)
                scatter_targets.append((target_x, target_y))
            except:
                continue

        # 更新windows列表为有效窗口
        self.windows = valid_windows

        # 开始动画移动 - 极速散开效果
        if valid_windows:
            self.animate_scatter(0, scatter_targets, steps=5, delay=8)

    def animate_scatter(self, step, targets, steps=5, delay=8):
        """执行散开动画，快速移动卡片到目标位置 - 极速有冲击力版本"""
        # 检查动画是否应该停止
        if not self.scatter_animation_running or step >= steps or not self.windows:
            # 动画完成，将所有卡片固定在最终位置
            for i, window in enumerate(self.windows):
                if window.winfo_exists():
                    try:
                        target_x, target_y = targets[i]
                        window.geometry(f"+{target_x}+{target_y}")
                        window.resizable(False, False)
                        window._position_locked = True
                        self.final_positions[window] = (target_x, target_y)
                        window.update_idletasks()
                    except:
                        pass
            self.scatter_animation_running = False
            self.progress_label.config(text="💥 卡片已散开！点击重新开始按钮可重置")
            self.root.after(100, self.monitor_positions)
            return

        progress = step / steps

        # 更强烈的缓动函数（ease-out-back 增强版）
        ease_progress = 1 + 5.0 * (progress - 1) ** 3 + 3.0 * (progress - 1) ** 2

        for i, window in enumerate(self.windows):
            if not window.winfo_exists():
                continue

            if hasattr(window, '_position_locked') and window._position_locked:
                continue

            try:
                if step == 0:
                    if not hasattr(window, '_start_x'):
                        window._start_x = window.winfo_x()
                        window._start_y = window.winfo_y()

                    start_x = window._start_x
                    start_y = window._start_y
                    target_x, target_y = targets[i]
                else:
                    target_x, target_y = targets[i]

                    if not hasattr(window, '_start_x'):
                        window._start_x = window.winfo_x()
                        window._start_y = window.winfo_y()

                    start_x = window._start_x
                    start_y = window._start_y

                new_x = int(start_x + (target_x - start_x) * ease_progress)
                new_y = int(start_y + (target_y - start_y) * ease_progress)

                window.geometry(f"+{new_x}+{new_y}")
                window.update_idletasks()
            except:
                pass

        if self.scatter_animation_running:
            self.root.after(delay, lambda: self.animate_scatter(step + 1, targets, steps, delay))

    def monitor_positions(self):
        """监控卡片位置，确保它们保持在最终位置"""
        if not self.is_scattered or not self.final_positions:
            return

        for window, (target_x, target_y) in list(self.final_positions.items()):
            if window.winfo_exists():
                try:
                    current_x = window.winfo_x()
                    current_y = window.winfo_y()

                    # 如果位置偏离超过1像素，重新设置
                    if abs(current_x - target_x) > 1 or abs(current_y - target_y) > 1:
                        window.geometry(f"+{target_x}+{target_y}")
                        window.update_idletasks()
                except:
                    # 如果窗口不存在，从监控列表中移除
                    if window in self.final_positions:
                        del self.final_positions[window]

        # 继续监控（每500毫秒检查一次）
        if self.final_positions:
            self.root.after(500, self.monitor_positions)

    def finalize_animation(self):
        """动画完成后的处理"""
        self.progress_label.config(text=f"完成! 共{len(MESSAGES)}条关心话语组成了爱心\n点击任意卡片可看到惊喜效果")
        # 添加重新开始按钮
        restart_button = tk.Button(
            self.root,
            text="重新开始动画",
            command=self.restart_animation,
            bg="#FF69B4",
            fg="white",
            font=("微软雅黑", 10),
            padx=15,
            pady=5
        )
        restart_button.pack(pady=10)
        # 添加关闭所有窗口按钮
        close_all_button = tk.Button(
            self.root,
            text="关闭所有关心窗口",
            command=self.close_all_windows,
            bg="#FF69B4",
            fg="white",
            font=("微软雅黑", 10),
            padx=15,
            pady=5
        )
        close_all_button.pack(pady=5)

    def close_all_windows(self):
        """关闭所有关心窗口"""
        for window in self.windows:
            window.destroy()
        self.windows = []

    def restart_animation(self):
        """重新开始动画"""
        # 停止任何正在运行的散开动画
        self.scatter_animation_running = False

        # 关闭所有窗口
        self.close_all_windows()
        # 重置状态
        self.current_index = 0
        self.is_scattered = False  # 重置散开状态
        # 重新生成消息（随机选择）
        global MESSAGES
        MESSAGES = []
        for i in range(NUM_CARDS):
            MESSAGES.append(random.choice(BASE_MESSAGES))
        # 移除按钮
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button) and widget.cget("text") in ["重新开始动画", "关闭所有关心窗口"]:
                widget.destroy()
        # 重新开始动画
        self.start_animation()

    def run(self):
        """运行应用"""
        # 延迟一下开始动画，让用户看到初始界面
        self.root.after(1200, self.start_animation)
        self.root.mainloop()


if __name__ == "__main__":
    app = ExtraLargeHeart()
    app.run()
