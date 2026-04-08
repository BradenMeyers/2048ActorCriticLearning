"""
generate_findings_plots.py — generate summary charts for the Key Findings section.

Outputs (saved to results/findings/):
    performance_comparison.png  — bar chart of mean eval scores by agent/mode
    training_efficiency.png     — A2C vs MCTS-AC score over wall-clock time
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = "results/findings"
os.makedirs(OUT_DIR, exist_ok=True)

A2C_LOG  = "pretrained/a2c/a2c_log.csv"
MCTS_LOG = "pretrained/mcts/mcts_log.csv"

# ── style ─────────────────────────────────────────────────────────────────────
BLUE   = "#4C8BF5"
ORANGE = "#F5A623"
GREEN  = "#27AE60"
GRAY   = "#95A5A6"
RED    = "#E74C3C"
PURPLE = "#8E44AD"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "grid.linewidth":   0.8,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})


# ── 1. Performance comparison ─────────────────────────────────────────────────
def plot_performance():
    agents = [
        "Random\nbaseline",
        "Uniform MCTS\n(200 sims)",
        "MCTS-AC\n(greedy)",
        "MCTS-AC\n(MCTS search)",
        "A2C\n(greedy)",
    ]
    means = [1053, 5000, 2413, 5324, 6950]
    colors = [GRAY, GREEN, ORANGE, PURPLE, BLUE]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(agents, means, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=0.5)

    # value labels on bars
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{val:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Mean Score (100 games)")
    ax.set_title("Agent Performance Comparison")
    ax.set_ylim(0, max(means) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "performance_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── 2. Training efficiency ─────────────────────────────────────────────────────
def plot_training_efficiency():
    a2c  = pd.read_csv(A2C_LOG)
    mcts = pd.read_csv(MCTS_LOG)

    # Convert elapsed seconds to hours
    a2c["hours"]  = a2c["elapsed_s"]  / 3600
    mcts["hours"] = mcts["elapsed_s"] / 3600

    # Smooth with rolling window
    a2c["smooth"]  = a2c["avg_score"].rolling(20, min_periods=1).mean()
    mcts["smooth"] = mcts["avg_score"].rolling(5,  min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: score vs episodes ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(a2c["episode"],  a2c["smooth"],  color=BLUE,   lw=2,   label="A2C (110k episodes)")
    ax.plot(mcts["episode"], mcts["smooth"], color=ORANGE, lw=2,   label="MCTS-AC (520 episodes)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Avg Score (smoothed)")
    ax.set_title("Score vs Training Episodes")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: score vs wall-clock time ──────────────────────────────────────
    ax = axes[1]
    ax.plot(a2c["hours"],  a2c["smooth"],  color=BLUE,   lw=2,   label="A2C")
    ax.plot(mcts["hours"], mcts["smooth"], color=ORANGE, lw=2,   label="MCTS-AC")
    ax.set_xlabel("Wall-clock Time (hours)")
    ax.set_ylabel("Avg Score (smoothed)")
    ax.set_title("Score vs Training Time")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Training Efficiency: A2C vs MCTS-AC", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "training_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_performance()
    plot_training_efficiency()
    print("Done.")
