import ctypes
import time

def move_window_to_second_screen(
        offset_x=100,
        offset_y=10,
        width=1000,
        height=900):
    """
    将当前前台窗口移动到扩展屏（假设在右侧）
    """

    user32 = ctypes.windll.user32

    # 主屏宽度
    primary_w = user32.GetSystemMetrics(0)

    # 当前前台窗口句柄
    hwnd = user32.FindWindowW(None, "MuJoCo : balance_bike scene")


    # 如果没找到，稍微等一下（窗口刚创建时）
    retry = 0
    while hwnd == 0 and retry < 10:
        time.sleep(0.1)
        hwnd = user32.FindWindowW(None, "MuJoCo : segway scene")

        retry += 1

    if hwnd == 0:
        print("❌ 没找到 MuJoCo 窗口")
        return

    # 目标位置（右侧扩展屏）
    x = offset_x
    y = offset_y

    # 0x0004 = SWP_NOZORDER（不改变层级）
    user32.SetWindowPos(hwnd, 0, x, y, width, height, 0x0004)

# get windows name
# def list_all_windows():
#     user32 = ctypes.windll.user32
#     EnumWindows = user32.EnumWindows
#     EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
#
#     def callback(hwnd, lParam):
#         length = user32.GetWindowTextLengthW(hwnd)
#         buff = ctypes.create_unicode_buffer(length + 1)
#         user32.GetWindowTextW(hwnd, buff, length + 1)
#         print(hwnd, buff.value)
#         return True
#
#     EnumWindows(EnumWindowsProc(callback), 0)
#
# list_all_windows()