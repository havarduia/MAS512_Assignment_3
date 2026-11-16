"""Shared chart style for the Q2 plots and GIFs: colours, matplotlib defaults, and the map drawing.

Colours follow a fixed categorical order (test 1 = blue, 2 = orange, 3 = aqua, 4 = yellow,
5 = magenta) and a separate status palette (reached / collision / timeout / fell).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEST_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # tests 1-5
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
OUTCOME_COLOR = dict(success=GOOD, collision=CRITICAL, timeout=WARNING, fell=SERIOUS)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})


def draw_layout(ax, layout, target, arena):
    """Arena walls, obstacles (grey; coloured decoys gold) and the red goal, seen from above.
    layout: [(obstacle dict with kind / half / round, centre xy, yaw), ...]"""
    ax.add_patch(Rectangle((-arena, -arena), 2 * arena, 2 * arena, fill=False, ec=INK2, lw=2.5))
    for ob, xy, yaw in layout:
        hx, hy = ob["half"][:2]
        color = {"box": "#9a9890", "pillar": "#8a7e70", "wall": "#b8b4a8", "decoy": "#c9a227"}[ob["kind"]]
        if ob["round"]:
            ax.add_patch(Circle(xy, hx, color=color, lw=0))
        else:
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            pts = (R @ np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]).T).T + xy
            ax.add_patch(Polygon(pts, color=color, lw=0))
    ax.add_patch(Circle(target, 0.18, color=CRITICAL, lw=0, zorder=4))


def map_axes(ax, layout, goal, arena, reach):
    """Top-down map on ax: layout, goal with its dashed reach circle, no ticks."""
    draw_layout(ax, layout, goal, arena)
    ax.add_patch(Circle(goal, reach, fill=False, ec=CRITICAL, ls="--", lw=1))
    ax.set_xlim(-arena - 0.2, arena + 0.2); ax.set_ylim(-arena - 0.2, arena + 0.2)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
