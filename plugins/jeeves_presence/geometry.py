"""Where a computer_use action lands on screen, from the elements of the latest capture.

cua-driver reports element frames as global desktop points (top-left origin), so an element
action lands at the centre of its frame. Coordinate actions are relative to the captured window's
screenshot; those are forwarded with the capture's app, window title and image size and the app
resolves them against the live window frame.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

POINTER_ACTIONS = {"click", "double_click", "right_click", "middle_click", "drag", "scroll", "set_value"}
KEY_ACTIONS = {"type", "key"}

_ELEMENT_LINE = re.compile(r"#(\d+) .*? @ \((-?\d+), (-?\d+), (\d+), (\d+)\)")
_HEADER = re.compile(r"capture mode=\w+ (\d+)x(\d+)(?: app=(.+?))?(?: window='(.*)')?$", re.MULTILINE)


@dataclass
class CaptureState:
    elements: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    app: str = ""
    window_title: str = ""
    width: int = 0
    height: int = 0

    def center(self, index: Any) -> tuple[int, int] | None:
        try:
            x, y, w, h = self.elements[int(index)]
        except (KeyError, TypeError, ValueError):
            return None
        if w <= 0 and h <= 0 and x == 0 and y == 0:
            return None
        return x + w // 2, y + h // 2

    def window_center(self) -> tuple[int, int] | None:
        """Centre of all known element frames — a stand-in for the window when nothing is targeted."""
        frames = [b for b in self.elements.values() if any(b)]
        if not frames:
            return None
        left = min(b[0] for b in frames)
        top = min(b[1] for b in frames)
        right = max(b[0] + b[2] for b in frames)
        bottom = max(b[1] + b[3] for b in frames)
        return (left + right) // 2, (top + bottom) // 2


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        if isinstance(result.get("text_summary"), str):
            return result["text_summary"]
        for part in result.get("content") or ():
            if isinstance(part, Mapping) and part.get("type") == "text":
                return str(part.get("text") or "")
    return ""


def update_from_result(state: CaptureState, result: Any) -> bool:
    """Refresh *state* from a computer_use result that carries a capture. True when it did."""
    text = _result_text(result)
    if not text:
        return False
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, Mapping) and isinstance(data.get("elements"), list):
        elements: dict[int, tuple[int, int, int, int]] = {}
        for e in data["elements"]:
            bounds = e.get("bounds") if isinstance(e, Mapping) else None
            if isinstance(bounds, list) and len(bounds) == 4:
                elements[int(e["index"])] = tuple(int(v) for v in bounds)  # type: ignore[assignment]
        state.elements = elements
        state.app = str(data.get("app") or "")
        state.window_title = str(data.get("window_title") or "")
        state.width, state.height = int(data.get("width") or 0), int(data.get("height") or 0)
        return True
    header = _HEADER.search(text)
    if not header:
        return False
    state.width, state.height = int(header.group(1)), int(header.group(2))
    state.app, state.window_title = header.group(3) or "", header.group(4) or ""
    state.elements = {int(m.group(1)): tuple(int(m.group(i)) for i in range(2, 6))  # type: ignore[misc]
                      for m in _ELEMENT_LINE.finditer(text)}
    return True


def _coordinate(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def plan_pointer(state: CaptureState, args: Mapping[str, Any]) -> dict[str, Any] | None:
    """The cursor event for a pointer action: an absolute ``point`` when an element resolves,
    else a window-relative ``window_point`` for the app to resolve. None when nothing is known."""
    action = str(args.get("action") or "")
    if action not in POINTER_ACTIONS:
        return None
    window = {"app": state.app, "window_title": state.window_title,
              "image_width": state.width, "image_height": state.height}

    def target(element_key: str, coord_key: str) -> dict[str, Any] | None:
        point = state.center(args.get(element_key)) if args.get(element_key) is not None else None
        if point:
            return {"point": point}
        coord = _coordinate(args.get(coord_key))
        if coord:
            return {"window_point": coord, **window}
        return None

    if action == "drag":
        start, end = target("from_element", "from_coordinate"), target("to_element", "to_coordinate")
        if not (start and end):
            return None
        return {"action": "drag", "from": start, "to": end}
    hit = target("element", "coordinate")
    if hit is None and action == "scroll":
        center = state.window_center()
        hit = {"point": center} if center else None
    if hit is None:
        return None
    button = str(args.get("button") or ("right" if action == "right_click" else "left"))
    return {"action": action, "button": button, **hit,
            **({"direction": args.get("direction")} if action == "scroll" else {})}
