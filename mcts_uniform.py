"""
Uniform-policy MCTS for 2048 — diagnostic tool
================================================
Tests that the tree search and reward signal are working correctly,
with no neural network involved.

Prior:           uniform over legal moves
Leaf evaluation: random rollout for `rollout_depth` steps

Usage
-----
    python mcts_uniform.py                        # 20 games, 200 sims, 50-step rollout
    python mcts_uniform.py --games 50 --sims 500
    python mcts_uniform.py --display 2            # watch it play (requires pygame)

Alternatively, use the unified entry point:
    python main.py --mode uniform_mcts --sims 200 --games 20
"""

import argparse
import random
import time

import numpy as np

from game import Game2048, Move
from MCTS import UniformMCTS
from evaluate import print_results
from display import display_agent


# ──────────────────────────────────────────── play ──────────────────────────

def play_games(
    n_games:        int   = 20,
    n_simulations:  int   = 200,
    rollout_depth:  int   = 50,
    c:              float = 160,
    gamma:          float = 0.99,
    clear_tree:     bool  = False,
    seed:           int   = 0,
):
    """
    Play n_games with the uniform MCTS agent and print stats.

    clear_tree=True  — clear the tree between moves within a game (faster, less accurate)
    clear_tree=False — keep the tree between moves (expensive but fully correct)
    """
    random.seed(seed)
    np.random.seed(seed)

    mcts = UniformMCTS(
        c=c,
        n_simulations=n_simulations,
        gamma=gamma,
        rollout_depth=rollout_depth,
    )

    scores, max_tiles = [], []
    t0 = time.time()

    for i in range(n_games):
        game = Game2048(seed=seed + i)
        mcts.reset_tree()

        while not game.is_over:
            if clear_tree:
                mcts.reset_tree()
            action = mcts.best_action(game)
            game.step(Move(action))

        scores.append(game.score)
        max_tiles.append(game.max_tile)

        if (i + 1) % max(1, n_games // 10) == 0:
            print(f"  game {i+1:>4}/{n_games} | score {game.score:>7} | max tile {game.max_tile}")

    print_results("Uniform MCTS", scores, max_tiles, duration=time.time() - t0)


def play_random(n_games: int = 100, seed: int = 0):
    """Baseline: pure random moves."""
    random.seed(seed)
    scores, max_tiles = [], []
    t0 = time.time()
    for i in range(n_games):
        game = Game2048(seed=seed + i)
        while not game.is_over:
            moves = game.available_moves()
            if not moves:
                break
            game.step(random.choice(moves))
        scores.append(game.score)
        max_tiles.append(game.max_tile)
    print_results("Random baseline", scores, max_tiles, duration=time.time() - t0)


# ──────────────────────────────────────────── CLI ───────────────────────────

def main():
    p = argparse.ArgumentParser(description="Uniform-policy MCTS for 2048")
    p.add_argument("--games",    type=int,   default=20,  help="number of games to play")
    p.add_argument("--sims",     type=int,   default=200, help="MCTS simulations per move (0 = random)")
    p.add_argument("--rollout",  type=int,   default=10,  help="random rollout depth for leaf eval")
    p.add_argument("--c",        type=float, default=160, help="PUCT exploration constant")
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--clear-tree",    action="store_true", help="clear tree between moves in a game")
    p.add_argument("--baseline", action="store_true",  help="also run random baseline")
    p.add_argument("--display",  type=int,   default=-1, help="watch agent play at DISPLAY fps")
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()

    if args.display != -1:
        mcts = UniformMCTS(
            n_simulations=args.sims,
            rollout_depth=args.rollout,
            c=args.c,
            gamma=args.gamma,
        )
        display_agent(
            select_action=mcts.best_action,
            caption="2048 — Uniform MCTS",
            n_games=args.games,
            speed=args.display,
        )
        return

    if args.sims == 0:
        play_random(n_games=args.games, seed=args.seed)
        return

    print(f"Uniform MCTS | sims={args.sims} rollout={args.rollout} c={args.c} clear_tree={args.clear_tree}")
    play_games(
        n_games=args.games,
        n_simulations=args.sims,
        rollout_depth=args.rollout,
        c=args.c,
        gamma=args.gamma,
        clear_tree=args.clear_tree,
        seed=args.seed,
    )

    if args.baseline:
        print("\nRunning random baseline for comparison...")
        play_random(n_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
