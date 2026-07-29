import tkinter as tk

def choose_path_gui():
    """
    弹窗选择轨迹类型
    返回 'circle', 'line', 'scurve', 'composite'
    """
    choice = {'value': None}  # 用字典存选择结果

    def select_circle():
        choice['value'] = 'circle'
        root.destroy()
    def select_line():
        choice['value'] = 'line'
        root.destroy()
    def select_scurve():
        choice['value'] = 'scurve'
        root.destroy()
    def select_composite():
        choice['value'] = 'composite'
        root.destroy()

    root = tk.Tk()
    root.title("选择路径类型")
    # 设置窗口始终置顶
    root.attributes("-topmost", True)

    # 计算窗口居中
    win_width = 500
    win_height = 180
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_pos = (screen_width - win_width) // 2
    y_pos = (screen_height - win_height) // 2
    root.geometry(f"{win_width}x{win_height}+{x_pos}+{y_pos}")
    root.resizable(False, False)

    # 标签
    label = tk.Label(root, text="请选择轨迹类型",
                     font=("Arial", 16, "bold"),
                     fg="black")  # 黑色字体
    label.pack(pady=20)

    # 按钮容器 Frame
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    # 按钮样式统一
    btn_style = {"font": ("Arial", 11), "width": 10, "height": 1,
                 "bg": "#006400", "fg": "white"}  # 深绿色按钮, 白字

    tk.Button(btn_frame, text="圆轨迹", **btn_style, command=select_circle).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="直线轨迹", **btn_style, command=select_line).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="S型轨迹", **btn_style, command=select_scurve).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="组合轨迹", **btn_style, command=select_composite).pack(side=tk.LEFT, padx=5)

    root.mainloop()

    if choice['value'] is None:
        raise ValueError("用户未选择轨迹")
    return choice['value']