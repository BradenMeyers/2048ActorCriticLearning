"""
Plot training metrics from an log CSV.

Usage:
    python plot_log.py                          # plots a2c_log.csv
    python plot_log.py a2c_log_1.csv            # plots a specific file
    python plot_log.py a2c_log.csv a2c_log_1.csv  # overlays multiple runs
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_FILES = sys.argv[1:] if len(sys.argv) > 1 else ["a2c_log.csv"]

dfs = []
for path in LOG_FILES:
    df = pd.read_csv(path)
    df["_label"] = path
    dfs.append(df)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training Metrics", fontsize=14)

METRICS = [
    ("avg_score",    "Avg Score",       axes[0, 0]),
    ("avg_max_tile", "Avg Max Tile",    axes[0, 1]),
    ("actor_loss",       "Actor Loss",      axes[1, 0]),
    ("critic_loss",      "Critic Loss",         axes[1, 1]),
]

for df in dfs:
    label = df["_label"].iloc[0]
    for col, title, ax in METRICS:
        ax.plot(df["episode"], df[col], label=label, linewidth=1)

for col, title, ax in METRICS:
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    if len(dfs) > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training.png", dpi=150)
print("Saved training.png")
plt.show()
