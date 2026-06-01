"""Shared Matplotlib style helpers for thesis figures."""

from __future__ import annotations

from matplotlib import font_manager
import matplotlib.pyplot as plt


CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]


def setup_chinese_matplotlib() -> str | None:
    """Configure Matplotlib to render Chinese text without garbled glyphs."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in CHINESE_FONT_CANDIDATES if name in available_fonts), None)
    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = CHINESE_FONT_CANDIDATES + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 8
    return selected
