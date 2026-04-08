"""
evaluate.py — shared evaluation logic for 2048 agents
======================================================
Extracted from train_a2c.py, train_mcts.py, and mcts_uniform.py.

Usage
-----
    from evaluate import evaluate_agent, evaluate_checkpoint, print_results

    # Evaluate any callable agent
    evaluate_agent(select_action=my_agent, label="My Agent", n_games=100)

    # Evaluate from a saved checkpoint (loads weights automatically)
    from networks import CNNActorCritic
    evaluate_checkpoint("a2c_checkpoint.pt", CNNActorCritic, device, label="A2C", n_games=100)
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Callable

import numpy as np
import torch

from runners.game import Game2048, Move
from runners.utils import action_mask


def print_results(
    label:     str,
    scores:    list[int],
    max_tiles: list[int],
    duration:  float | None = None,
) -> None:
    """Print a formatted summary of evaluation results."""
    n_games  = len(scores)
    tile_dist = Counter(max_tiles)
    print(f"\n{'='*50}")
    print(f"  {label} — {n_games} games")
    print(f"{'='*50}")
    print(f"  Mean score      : {np.mean(scores):>10.1f}")
    print(f"  Median score    : {np.median(scores):>10.1f}")
    print(f"  Max score       : {np.max(scores):>10}")
    print(f"  Win rate (≥2048): {sum(t >= 2048 for t in max_tiles) / n_games * 100:>7.1f}%")
    if duration is not None:
        print(f"  Duration        : {duration:.2f}s")
    print(f"\n  Max tile distribution:")
    for tile in sorted(tile_dist):
        pct = tile_dist[tile] / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"    {tile:>5}: {bar:<40} {pct:.1f}%")


def evaluate_agent(
    select_action: Callable[[Game2048], int],
    label:   str = "Agent",
    n_games: int = 100,
) -> dict:
    """
    Run n_games with select_action and print results.

    Parameters
    ----------
    select_action : callable
        Any function that takes a Game2048 and returns an action index (0–3).
    label : str
        Name shown in the results header.
    n_games : int
        Number of games to evaluate.

    Returns
    -------
    dict with keys: scores, max_tiles, win_rate, mean_score
    """
    scores, max_tiles = [], []
    t0 = time.time()

    for i in range(n_games):
        game = Game2048(seed=i)
        while not game.is_over:
            action = select_action(game)
            game.step(Move(action))
        scores.append(game.score)
        max_tiles.append(game.max_tile)

    print_results(label, scores, max_tiles, duration=time.time() - t0)
    return {
        "scores":     scores,
        "max_tiles":  max_tiles,
        "win_rate":   sum(t >= 2048 for t in max_tiles) / n_games,
        "mean_score": float(np.mean(scores)),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    net_class,
    device,
    label:   str = "Agent",
    n_games: int = 100,
) -> dict:
    """
    Load a checkpoint, build a greedy selector, and call evaluate_agent.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file saved by train_a2c or train_mcts.
    net_class : type
        The network class to instantiate (e.g. CNNActorCritic or LinearActorCritic).
    device : torch.device
    label : str
    n_games : int
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Train first.")
        return {}

    net = net_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    def select_action(game: Game2048) -> int:
        with torch.no_grad():
            state  = net.board_to_tensor(game.board).to(device)
            mask   = action_mask(game).to(device)
            policy, _ = net(state, mask)
            return int(policy.argmax().item())

    return evaluate_agent(select_action, label=label, n_games=n_games)
