import random
from typing import Any, Dict

label_color_map: Dict[str, Any] = dict()


def get_color(label, color_map):
    color = color_map.get(label) if color_map else None
    color = color if color else color_map.get(label.lower()) if color_map and isinstance(label, str) else None
    color = label_color_map.get(label) if not color else color
    if color is None:
        color = [random.randint(0, 255) for _ in range(3)]
        label_color_map[label] = color
    return color


def to_hex(color):
    return "#%02x%02x%02x" % tuple(color) if color else "#000000"


def to_rgb(color):
    return "rgb(" + ",".join([str(x) for x in color]) + ")" if color else "rgb(0,0,0)"
