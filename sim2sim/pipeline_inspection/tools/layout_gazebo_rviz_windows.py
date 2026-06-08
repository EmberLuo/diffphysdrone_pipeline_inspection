#!/usr/bin/env python3
"""Arrange Gazebo and RViz side by side on the recording display."""

from __future__ import annotations

import argparse
import time

from Xlib import X, display


def walk_windows(root):
    children = root.query_tree().children
    for child in children:
        yield child
        yield from walk_windows(child)


def window_name(disp, window) -> str:
    for attr in ("WM_NAME", "_NET_WM_NAME"):
        value = window.get_full_property(disp.intern_atom(attr), X.AnyPropertyType)
        if value is None:
            continue
        raw = value.value
        if isinstance(raw, bytes):
            return raw.decode(errors="ignore")
        if isinstance(raw, str):
            return raw
        try:
            return bytes(raw).decode(errors="ignore")
        except Exception:
            continue
    return ""


def window_class(window) -> str:
    prop = window.get_wm_class()
    if not prop:
        return ""
    return " ".join(prop)


def find_window(disp, root, needles: tuple[str, ...]):
    for window in walk_windows(root):
        name = window_name(disp, window)
        if not name or "selection owner" in name.lower() or "client leader" in name.lower():
            continue
        try:
            geom = window.get_geometry()
        except Exception:
            continue
        if geom.width < 200 or geom.height < 200:
            continue
        text = f"{name} {window_class(window)}".lower()
        if all(needle.lower() in text for needle in needles):
            return window
    return None


def configure(window, x: int, y: int, width: int, height: int) -> None:
    window.configure(x=x, y=y, width=width, height=height, border_width=0)
    window.raise_window()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display", default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    disp = display.Display(args.display)
    root = disp.screen().root
    split = args.width // 2
    deadline = time.time() + args.timeout
    gazebo = None
    rviz = None

    while time.time() < deadline:
        gazebo = gazebo or find_window(disp, root, ("gazebo",))
        rviz = rviz or find_window(disp, root, ("rviz",))
        if gazebo is not None and rviz is not None:
            break
        time.sleep(0.5)

    if gazebo is not None:
        configure(gazebo, 0, 0, split, args.height)
    if rviz is not None:
        configure(rviz, split, 0, args.width - split, args.height)
    disp.sync()

    missing = []
    if gazebo is None:
        missing.append("Gazebo")
    if rviz is None:
        missing.append("RViz")
    if missing:
        raise SystemExit(f"Could not find windows: {', '.join(missing)}")


if __name__ == "__main__":
    main()
