"""
sensitivity_analysis.py — UniformMCTS parameter sensitivity analysis
=====================================================================
Varies one parameter at a time from a base config, runs all evaluations
in parallel threads, and prints a sorted results table.

Usage
-----
    python sensitivity_analysis.py
    python sensitivity_analysis.py --games 50   # more games, slower
    python sensitivity_analysis.py --sims 400   # more sims per move
"""

import argparse
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from game import Game2048, Move
from MCTS import UniformMCTS


# ─────────────────────────────────────── runner ─────────────────────────────

def run_config(
    label:            str,
    n_games:          int,
    n_simulations:    int,
    seed:             int,
    c:                float,
    rollout_depth:    int,
    gamma:            float,
    terminal_penalty: float,
    reuse_tree:       bool,
) -> dict:
    """Run n_games with the given config. Returns a result dict."""
    random.seed(seed)
    np.random.seed(seed)

    mcts = UniformMCTS(
        c=c,
        n_simulations=n_simulations,
        gamma=gamma,
        rollout_depth=rollout_depth,
        terminal_penalty=terminal_penalty,
    )

    scores, max_tiles = [], []
    t0 = time.time()

    for i in range(n_games):
        game = Game2048(seed=seed + i)
        mcts.reset_tree()

        while not game.is_over:
            if not reuse_tree:
                mcts.reset_tree()
            action = mcts.best_action(game)
            game.step(Move(action))

        scores.append(game.score)
        max_tiles.append(game.max_tile)

    elapsed = time.time() - t0
    tile_dist = Counter(max_tiles)

    return {
        "label":           label,
        "mean_score":      float(np.mean(scores)),
        "median_score":    float(np.median(scores)),
        "max_score":       int(np.max(scores)),
        "win_rate":        sum(t >= 2048 for t in max_tiles) / n_games * 100,
        "max_512_plus":    sum(t >= 512  for t in max_tiles) / n_games * 100,
        "tile_dist":       dict(tile_dist),
        "elapsed":         elapsed,
        # params for reference
        "c":               c,
        "rollout_depth":   rollout_depth,
        "gamma":           gamma,
        "terminal_penalty": terminal_penalty,
        "reuse_tree":      reuse_tree,
    }


# ─────────────────────────────────────── sweep config ───────────────────────

def build_configs(n_games: int, n_simulations: int, seed: int) -> list[dict]:
    """
    One-at-a-time sensitivity sweep around the base config.
    The base config is included once (marked with *).
    """
    base = dict(
        n_games=n_games,
        n_simulations=n_simulations,
        seed=seed,
        c=160.0,
        rollout_depth=10,
        gamma=0.99,
        terminal_penalty=-10.0,
        reuse_tree=False,
    )

    configs = []

    def add(label, **overrides):
        cfg = {**base, **overrides, "label": label}
        configs.append(cfg)

    # ── baseline (random moves) ──────────────────────────────────────────────
    add("random baseline", n_simulations=0)  # handled specially below

    # ── base config ──────────────────────────────────────────────────────────
    add("BASE *")

    # ── c: exploration constant ───────────────────────────────────────────────
    for c in [5, 20, 40, 80, 320, 640, 1280]:
        add(f"c={c}", c=c)

    # ── rollout_depth: leaf evaluation depth ──────────────────────────────────
    for d in [1, 3, 5, 20, 50, 100]:
        add(f"rollout={d}", rollout_depth=d)

    # ── gamma: discount factor ────────────────────────────────────────────────
    for g in [0.5, 0.8, 0.9, 0.95, 1.0]:
        add(f"gamma={g}", gamma=g)

    # ── terminal_penalty: reward at game-over ─────────────────────────────────
    for tp in [-100, -50, -20, -5, -1, 0]:
        add(f"penalty={tp}", terminal_penalty=tp)

    # ── reuse_tree ────────────────────────────────────────────────────────────
    add("reuse_tree=True",  reuse_tree=True)

    return configs


def run_random_baseline(n_games: int, seed: int) -> dict:
    """Pure random moves baseline."""
    random.seed(seed)
    np.random.seed(seed)
    scores, max_tiles = [], []
    t0 = time.time()
    for i in range(n_games):
        game = Game2048(seed=seed + i)
        while not game.is_over:
            game.step(random.choice(game.available_moves()))
        scores.append(game.score)
        max_tiles.append(game.max_tile)
    return {
        "label":            "random baseline",
        "mean_score":       float(np.mean(scores)),
        "median_score":     float(np.median(scores)),
        "max_score":        int(np.max(scores)),
        "win_rate":         0.0,
        "max_512_plus":     sum(t >= 512 for t in max_tiles) / n_games * 100,
        "tile_dist":        dict(Counter(max_tiles)),
        "elapsed":          time.time() - t0,
        "c": "-", "rollout_depth": "-", "gamma": "-",
        "terminal_penalty": "-", "reuse_tree": "-",
    }


# ─────────────────────────────────────── display ────────────────────────────

def _bar(pct: float, width: int = 20) -> str:
    return "█" * int(pct / 100 * width)


def print_table(results: list[dict]) -> None:
    results_sorted = sorted(results, key=lambda r: r["mean_score"], reverse=True)

    # header
    col = "{:<22}  {:>8}  {:>8}  {:>8}  {:>7}  {:>8}  {:>7}"
    print("\n" + "=" * 80)
    print(col.format("Config", "Mean", "Median", "Max", "Win%", "≥512%", "Time(s)"))
    print("=" * 80)

    baseline = next((r for r in results_sorted if r["label"] == "random baseline"), None)

    for r in results_sorted:
        delta = ""
        if baseline and r["label"] != "random baseline":
            d = r["mean_score"] - baseline["mean_score"]
            delta = f"  ({'+' if d >= 0 else ''}{d:.0f})"
        print(col.format(
            r["label"][:22],
            f"{r['mean_score']:.0f}{delta}" if not delta else f"{r['mean_score']:.0f}",
            f"{r['median_score']:.0f}",
            f"{r['max_score']}",
            f"{r['win_rate']:.1f}%",
            f"{r['max_512_plus']:.1f}%",
            f"{r['elapsed']:.1f}",
        ))
        if delta:
            print(f"  {'':22}  vs baseline: {delta.strip()}")

    print("=" * 80)

    # tile distribution for top 5
    print("\nTop 5 configs — max tile distribution:")
    for r in results_sorted[:5]:
        dist = r["tile_dist"]
        tiles_str = "  ".join(
            f"{t}:{dist[t]}" for t in sorted(dist, reverse=True) if dist[t] > 0
        )
        print(f"  {r['label']:<22}  {tiles_str}")


def print_sensitivity_summary(results: list[dict]) -> None:
    """Print the delta vs base for each parameter group."""
    base = next((r for r in results if r["label"] == "BASE *"), None)
    if not base:
        return

    groups = {
        "c":               [r for r in results if r["label"].startswith("c=")],
        "rollout_depth":   [r for r in results if r["label"].startswith("rollout=")],
        "gamma":           [r for r in results if r["label"].startswith("gamma=")],
        "terminal_penalty":[r for r in results if r["label"].startswith("penalty=")],
        "reuse_tree":      [r for r in results if r["label"].startswith("reuse_tree=")],
    }

    print("\nSensitivity summary (delta mean score vs BASE):")
    print(f"  BASE mean score: {base['mean_score']:.0f}\n")

    for param, group in groups.items():
        if not group:
            continue
        best = max(group, key=lambda r: r["mean_score"])
        worst = min(group, key=lambda r: r["mean_score"])
        spread = best["mean_score"] - worst["mean_score"]
        print(f"  {param:<20}  spread={spread:>6.0f}  "
              f"best={best['label']} ({best['mean_score']:.0f}, "
              f"{best['mean_score']-base['mean_score']:+.0f})  "
              f"worst={worst['label']} ({worst['mean_score']:.0f}, "
              f"{worst['mean_score']-base['mean_score']:+.0f})")


# ─────────────────────────────────────── main ───────────────────────────────

def main():
    p = argparse.ArgumentParser(description="UniformMCTS sensitivity analysis")
    p.add_argument("--games", type=int,   default=20,  help="games per config")
    p.add_argument("--sims",  type=int,   default=200, help="MCTS simulations per move")
    p.add_argument("--seed",  type=int,   default=42,  help="base RNG seed")
    p.add_argument("--workers", type=int, default=8,   help="parallel threads")
    args = p.parse_args()

    configs = build_configs(args.games, args.sims, args.seed)

    # Separate baseline (no MCTS object needed)
    baseline_cfg  = next(c for c in configs if c["label"] == "random baseline")
    mcts_configs  = [c for c in configs if c["label"] != "random baseline"]

    total = len(configs)
    print(f"Running {total} configs × {args.games} games × {args.sims} sims "
          f"in up to {args.workers} threads...")
    print(f"Seed: {args.seed}  (all configs use the same seed for fair comparison)\n")

    results = []
    completed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        # Submit baseline separately
        futures = {ex.submit(run_random_baseline, baseline_cfg["n_games"], baseline_cfg["seed"]): "random baseline"}

        # Submit all MCTS configs
        for cfg in mcts_configs:
            label = cfg.pop("label")
            fut   = ex.submit(run_config, label, **cfg)
            futures[fut] = label

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            completed += 1
            elapsed = time.time() - t_start
            print(f"  [{completed:>2}/{total}] {result['label']:<22}  "
                  f"mean={result['mean_score']:>7.0f}  "
                  f"elapsed={elapsed:.1f}s")

    print_table(results)
    print_sensitivity_summary(results)


if __name__ == "__main__":
    main()
