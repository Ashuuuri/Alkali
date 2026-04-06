"""
Visualize nfchain CU allocation across four workload targets.

Upper half: pipeline flow diagram (stages with worker counts)
Lower half: island grid (48 MEs across 4 islands, coloured by stage)

Usage:
    python3 experiments/plot_cu_allocation.py
Output:
    experiments/cu_allocation.png
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------

STAGE_NAMES = [
    "parse +\nfirewall",
    "flow\ntracker",
    "meta +\nLB lookup",
    "LB update\n+ send",
]

CONFIGS = [
    {
        "title": "default\n(target=100)",
        "cu_total": 13,
        "pool": 48,
        "stages": [4, 3, 3, 3],
    },
    {
        "title": "8 Mpps\n(target=100)",
        "cu_total": 13,
        "pool": 48,
        "stages": [4, 3, 3, 3],
    },
    {
        "title": "20 Mpps\n(target=40)",
        "cu_total": 29,
        "pool": 48,
        "stages": [6, 4, 8, 11],
    },
    {
        "title": "40 Mpps\n(target=20)",
        "cu_total": 48,
        "pool": 48,
        "stages": [12, 11, 16, 9],
    },
]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

STAGE_COLORS = [
    "#A8C4E0",  # stage 1 – blue
    "#A8D5BA",  # stage 2 – green
    "#F2D98B",  # stage 3 – yellow
    "#E8A87C",  # stage 4 – orange
    "#C4A8D5",  # stage 5 – purple (spare)
]
IDLE_COLOR  = "#E8E8E8"
ARROW_COLOR = "#555555"
BOX_EDGE    = "#333333"
GRID_EDGE   = "#888888"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ISLANDS     = 4
MES_PER_ISL = 12   # 4 × 12 = 48

def stage_color(idx):
    """0-based stage index → colour string."""
    return STAGE_COLORS[idx] if idx < len(STAGE_COLORS) else STAGE_COLORS[-1]


def build_cu_grid(stages):
    """
    Return (4, 12) int array where value is 1-based stage index (0 = idle).
    CUs are assigned sequentially: cu0 → Island 0 col 0, ..., cu11 → Island 0 col 11,
    cu12 → Island 1 col 0, etc.
    """
    grid = np.zeros((ISLANDS, MES_PER_ISL), dtype=int)
    cu = 0
    for s_idx, count in enumerate(stages):
        for _ in range(count):
            if cu >= ISLANDS * MES_PER_ISL:
                break
            island = cu // MES_PER_ISL
            col    = cu %  MES_PER_ISL
            grid[island, col] = s_idx + 1  # 1-based
            cu += 1
    return grid


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def draw_pipeline(ax, stages, names, colors):
    """Draw rounded-rect stage boxes with arrows on *ax*."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(stages)
    margin  = 0.04
    gap     = 0.02          # gap between box and arrow
    box_h   = 0.52
    box_y   = (1 - box_h) / 2

    total_w   = 1 - 2 * margin
    # arrow takes a fixed fraction of each slot's width
    arrow_frac = 0.14 if n > 3 else 0.12
    slot_w = total_w / (n + (n - 1) * arrow_frac / (1 - arrow_frac * (n - 1) / n))
    # simpler: equal slots for boxes + arrows
    # divide total_w into n boxes and (n-1) arrows
    arrow_w = total_w * 0.06
    box_w   = (total_w - (n - 1) * arrow_w) / n

    for i in range(n):
        x = margin + i * (box_w + arrow_w)

        # box
        fc = colors[i]
        box = FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=BOX_EDGE, linewidth=1.2,
            transform=ax.transData, clip_on=False,
        )
        ax.add_patch(box)

        # stage label
        ax.text(
            x + box_w / 2, box_y + box_h * 0.62,
            names[i],
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#222222",
            transform=ax.transData,
        )
        # worker count
        ax.text(
            x + box_w / 2, box_y + box_h * 0.22,
            f"×{stages[i]}",
            ha="center", va="center",
            fontsize=9, color="#444444",
            transform=ax.transData,
        )

        # arrow to next box
        if i < n - 1:
            ax_start = x + box_w + gap
            ax_end   = x + box_w + arrow_w - gap
            ax.annotate(
                "",
                xy=(ax_end, 0.5), xytext=(ax_start, 0.5),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ARROW_COLOR,
                    lw=1.5,
                    mutation_scale=12,
                ),
            )


def draw_island_grid(ax, grid, stage_count):
    """Draw the 4×12 ME island grid on *ax*."""
    ax.set_xlim(-0.5, MES_PER_ISL - 0.5)
    ax.set_ylim(-0.5, ISLANDS - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    cell_size = 1.0   # in data coords

    for isl in range(ISLANDS):
        for col in range(MES_PER_ISL):
            s = grid[isl, col]
            fc = stage_color(s - 1) if s > 0 else IDLE_COLOR
            rect = plt.Rectangle(
                (col - 0.5, isl - 0.5), cell_size, cell_size,
                facecolor=fc, edgecolor=GRID_EDGE, linewidth=0.6,
            )
            ax.add_patch(rect)

            # ME index inside cell (small text)
            cu_idx = isl * MES_PER_ISL + col
            ax.text(
                col, isl, str(cu_idx),
                ha="center", va="center",
                fontsize=5.5, color="#444444",
            )

        # island label on the left
        ax.text(
            -0.75, isl,
            f"Isl {isl}",
            ha="right", va="center",
            fontsize=7, color="#333333",
        )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def main():
    n_cols = len(CONFIGS)

    fig = plt.figure(figsize=(16, 8))

    # title row height : pipeline row height : grid row height
    height_ratios = [0.10, 0.40, 0.38, 0.12]
    gs = fig.add_gridspec(
        4, n_cols,
        height_ratios=height_ratios,
        hspace=0.08, wspace=0.10,
        left=0.04, right=0.97,
        top=0.93, bottom=0.04,
    )

    for col_idx, cfg in enumerate(CONFIGS):
        stages      = cfg["stages"]
        n_stages    = len(stages)
        cu_total    = cfg["cu_total"]
        pool        = cfg["pool"]
        colors      = [stage_color(i) for i in range(n_stages)]
        grid        = build_cu_grid(stages)

        # ── Row 0: column title ──────────────────────────────────────────
        ax_title = fig.add_subplot(gs[0, col_idx])
        ax_title.axis("off")
        ax_title.text(
            0.5, 0.5,
            cfg["title"],
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="#111111",
            transform=ax_title.transAxes,
        )
        ax_title.text(
            0.5, -0.05,
            f"CU used: {cu_total} / {pool}",
            ha="center", va="top",
            fontsize=9, color="#555555",
            transform=ax_title.transAxes,
        )

        # ── Row 1: pipeline flow ─────────────────────────────────────────
        ax_pipe = fig.add_subplot(gs[1, col_idx])
        draw_pipeline(ax_pipe, stages, STAGE_NAMES, colors)

        # ── Row 2: island grid ───────────────────────────────────────────
        ax_grid = fig.add_subplot(gs[2, col_idx])
        draw_island_grid(ax_grid, grid, n_stages)

        # thin divider line between pipeline and grid
        line_y = gs[1, col_idx].get_position(fig).y0 - 0.005
        fig.add_artist(
            matplotlib.lines.Line2D(
                [gs[1, col_idx].get_position(fig).x0,
                 gs[1, col_idx].get_position(fig).x1],
                [line_y, line_y],
                transform=fig.transFigure,
                color="#CCCCCC", linewidth=0.8,
            )
        )

    # ── Row 3: shared legend ─────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis("off")

    handles = []
    for i, name in enumerate(STAGE_NAMES):
        label = f"Stage {i+1}: {name.replace(chr(10), ' ')}"
        handles.append(
            mpatches.Patch(facecolor=stage_color(i), edgecolor=BOX_EDGE,
                           linewidth=0.8, label=label)
        )
    handles.append(
        mpatches.Patch(facecolor=IDLE_COLOR, edgecolor=GRID_EDGE,
                       linewidth=0.8, label="Idle ME")
    )
    ax_leg.legend(
        handles=handles,
        loc="center", ncol=len(handles),
        fontsize=9, frameon=True,
        framealpha=0.9, edgecolor="#CCCCCC",
        handlelength=1.4, handleheight=1.0,
        columnspacing=1.5,
    )

    # ── Super-title ──────────────────────────────────────────────────────
    fig.suptitle(
        "nfchain — Netronome ME allocation by workload target",
        fontsize=13, fontweight="bold", y=0.98, color="#111111",
    )

    out_path = os.path.join(os.path.dirname(__file__), "cu_allocation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
