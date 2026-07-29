from .path_selector import choose_path_gui
from .circle import CircleFollower
from .Square import SquareFollower
from .scurve import SCurveFollower
from .composite import ClosedCompositeFollower

def create_path(init_pos):
    t = choose_path_gui()
    if t == "circle":
        return CircleFollower(cx_offset=0, cy_offset=10, radius=10, init_pos=init_pos)
    if t == "line":
        return SquareFollower()
    if t == "scurve":
        return SCurveFollower(init_pos[0], init_pos[1])
    if t == "composite":
        return ClosedCompositeFollower()
    raise ValueError("未知路径")