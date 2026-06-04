"""Shared Matplotlib style helpers for thesis figures."""

from __future__ import annotations

from matplotlib import font_manager
import matplotlib.pyplot as plt


CHINESE_FONT_CANDIDATES = [
    "SimSun",
    "Songti SC",
    "STSong",
    "FZSongS-Extended",
    "FZShuSong-Z01",
    "Source Han Serif CN",
    "Noto Serif CJK SC",
    "Noto Serif CJK JP",
]

ENGLISH_FONT_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Liberation Serif",
    "Nimbus Roman",
]


def setup_chinese_matplotlib() -> str | None:
    """Configure thesis figures with Song-style Chinese and Times-style Latin."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in CHINESE_FONT_CANDIDATES if name in available_fonts), None)
    latin = next((name for name in ENGLISH_FONT_CANDIDATES if name in available_fonts), None)

    font_family = []
    if latin:
        font_family.append(latin)
    if selected:
        font_family.append(selected)
    font_family.extend(["DejaVu Serif", "DejaVu Sans"])

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.serif"] = font_family
    plt.rcParams["font.sans-serif"] = font_family
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 8
    return selected
