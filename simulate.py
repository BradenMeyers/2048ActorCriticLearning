"""
2048 Simulation Harness
=======================
Run many games headlessly, collect stats, plug in your own agent.

Quick start
-----------
    python simulate.py                    # 100 random games, summary table
    python simulate.py --n 500 --agent expectimax --depth 3
    python simulate.py --csv results.csv

Extend by subclassing BaseAgent.
"""

import argparse
import csv
import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from game import Game2048, Move


# ================================================================= agents

class BaseAgent(ABC):
    """Subclass this to plug in your own policy."""

    @abstractmethod
    def select_move(self, game: Game2048) -> Move:
        ...


class RandomAgent(BaseAgent):
    """Uniformly random legal move."""
    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def select_move(self, game: Game2048) -> Move:
        moves = game.available_moves()
        return self._rng.choice(moves) if moves else Move.UP


class GreedyAgent(BaseAgent):
    """Pick the move that maximises immediate score."""
    def select_move(self, game: Game2048) -> Move:
        best_move, best_score = None, -1
        for m in Move:
            clone = Game2048.from_board(game.board, game.score)
            moved, reward = clone.step(m)
            if moved and reward > best_score:
                best_score = reward
                best_move  = m
        return best_move if best_move is not None else Move.UP


class ExpectimaxAgent(BaseAgent):
    """
    Expectimax search with a heuristic evaluation function.
    Heuristic: empty cells + monotonicity + smoothness (classic combo).
    """

    def __init__(self, depth: int = 3):
        self.depth = depth

    def select_move(self, game: Game2048) -> Move:
        best_move, best_val = None, float("-inf")
        for m in Move:
            clone = Game2048.from_board(game.board, game.score)
            moved, _ = clone.step(m)
            if not moved:
                continue
            val = self._expect(clone, self.depth - 1)
            if val > best_val:
                best_val = val
                best_move = m
        return best_move if best_move is not None else Move.UP

    def _max_node(self, game: Game2048, depth: int) -> float:
        if depth == 0 or game.is_over:
            return _heuristic(game.board)
        best = float("-inf")
        for m in Move:
            clone = Game2048.from_board(game.board, game.score)
            moved, _ = clone.step(m)
            if moved:
                best = max(best, self._expect(clone, depth - 1))
        return best if best > float("-inf") else _heuristic(game.board)

    def _expect(self, game: Game2048, depth: int) -> float:
        if depth == 0 or game.is_over:
            return _heuristic(game.board)
        empty = game.empty_cells
        if not empty:
            return _heuristic(game.board)
        total = 0.0
        for (r, c) in empty:
            for val, prob in ((2, 0.9), (4, 0.1)):
                clone       = Game2048.from_board(game.board, game.score)
                clone.board[r, c] = val
                total      += prob * self._max_node(clone, depth - 1)
        return total / len(empty)


def _heuristic(board: np.ndarray) -> float:
    """Weighted sum of empty count, monotonicity, and smoothness."""
    empty        = float((board == 0).sum())
    mono         = _monotonicity(board)
    smooth       = _smoothness(board)
    max_corner   = _max_corner_bonus(board)
    return 2.7 * np.log(empty + 1) + 1.0 * mono + 0.1 * smooth + 1.0 * max_corner


def _monotonicity(board: np.ndarray) -> float:
    score = 0.0
    for row in board:
        nz = row[row > 0]
        if len(nz) > 1:
            diffs = np.diff(np.log2(nz.astype(float)))
            score += max(float((-diffs[diffs < 0]).sum()),
                         float(( diffs[diffs > 0]).sum()))
    for col in board.T:
        nz = col[col > 0]
        if len(nz) > 1:
            diffs = np.diff(np.log2(nz.astype(float)))
            score += max(float((-diffs[diffs < 0]).sum()),
                         float(( diffs[diffs > 0]).sum()))
    return -score


def _smoothness(board: np.ndarray) -> float:
    log_b = np.where(board > 0, np.log2(board.astype(float) + 1e-9), 0)
    smooth  = -np.sum(np.abs(np.diff(log_b, axis=1)))
    smooth -= np.sum(np.abs(np.diff(log_b, axis=0)))
    return float(smooth)


def _max_corner_bonus(board: np.ndarray) -> float:
    corners = [board[0, 0], board[0, -1], board[-1, 0], board[-1, -1]]
    if board.max() == max(corners):
        return float(np.log2(board.max() + 1))
    return 0.0


AGENTS = {
    "random":      lambda **kw: RandomAgent(seed=kw.get("seed")),
    "greedy":      lambda **kw: GreedyAgent(),
    "expectimax":  lambda **kw: ExpectimaxAgent(depth=kw.get("depth", 3)),
}


# ================================================================= runner

@dataclass
class GameResult:
    game_id:    int
    score:      int
    max_tile:   int
    n_moves:    int
    won:        bool
    duration_s: float


@dataclass
class SimStats:
    n_games:       int
    agent:         str
    mean_score:    float
    median_score:  float
    max_score:     int
    win_rate:      float
    tile_counts:   dict = field(default_factory=dict)
    total_time_s:  float = 0.0
    results:       list  = field(default_factory=list)

    def print_summary(self):
        print(f"\n{'='*52}")
        print(f"  Agent: {self.agent}   Games: {self.n_games}")
        print(f"{'='*52}")
        print(f"  Mean score    : {self.mean_score:>10.1f}")
        print(f"  Median score  : {self.median_score:>10.1f}")
        print(f"  Max score     : {self.max_score:>10}")
        print(f"  Win rate (≥2048): {self.win_rate*100:>7.1f}%")
        print(f"  Total time    : {self.total_time_s:>10.2f}s")
        print(f"  Time/game     : {self.total_time_s/self.n_games*1000:>9.1f}ms")
        print(f"\n  Max tile distribution:")
        for tile in sorted(self.tile_counts):
            bar = "█" * int(self.tile_counts[tile] / self.n_games * 40)
            pct = self.tile_counts[tile] / self.n_games * 100
            print(f"    {tile:>5}: {bar:<40} {pct:.1f}%")
        print()


def run_simulation(
    n_games:    int = 100,
    agent_name: str = "random",
    depth:      int = 3,
    seed:       Optional[int] = None,
    verbose:    bool = False,
) -> SimStats:
    """
    Run n_games complete games with the specified agent.

    Parameters
    ----------
    n_games    : number of games
    agent_name : one of 'random', 'greedy', 'expectimax'
    depth      : search depth (expectimax only)
    seed       : base RNG seed; each game gets seed+i for reproducibility
    verbose    : print progress

    Returns
    -------
    SimStats object with all results
    """
    agent   = AGENTS[agent_name](depth=depth, seed=seed)
    results = []

    t_total = time.perf_counter()
    for i in range(n_games):
        game_seed = (seed + i) if seed is not None else None
        game      = Game2048(seed=game_seed)
        n_moves   = 0
        t0        = time.perf_counter()

        while not game.is_over:
            move = agent.select_move(game)
            game.step(move)
            n_moves += 1

        elapsed = time.perf_counter() - t0
        result  = GameResult(
            game_id    = i,
            score      = game.score,
            max_tile   = game.max_tile,
            n_moves    = n_moves,
            won        = game.won(),
            duration_s = elapsed,
        )
        results.append(result)

        if verbose and (i + 1) % max(1, n_games // 10) == 0:
            print(f"  [{i+1}/{n_games}] score={game.score} max={game.max_tile}")

    scores = [r.score    for r in results]
    tiles  = [r.max_tile for r in results]

    stats = SimStats(
        n_games      = n_games,
        agent        = agent_name,
        mean_score   = float(np.mean(scores)),
        median_score = float(np.median(scores)),
        max_score    = int(np.max(scores)),
        win_rate     = sum(r.won for r in results) / n_games,
        tile_counts  = dict(Counter(tiles)),
        total_time_s = time.perf_counter() - t_total,
        results      = results,
    )
    return stats


def save_csv(stats: SimStats, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(stats.results[0]).keys())
        writer.writeheader()
        for r in stats.results:
            writer.writerow(asdict(r))
    print(f"Saved {len(stats.results)} rows → {path}")


# ================================================================= CLI

def main():
    p = argparse.ArgumentParser(description="2048 simulation harness")
    p.add_argument("--n",      type=int, default=100,     help="number of games")
    p.add_argument("--agent",  default="random",          choices=list(AGENTS),)
    p.add_argument("--depth",  type=int, default=3,       help="expectimax depth")
    p.add_argument("--seed",   type=int, default=None,    help="base RNG seed")
    p.add_argument("--csv",    default=None,              help="save results to CSV")
    p.add_argument("--verbose",action="store_true")
    args = p.parse_args()

    print(f"Running {args.n} games with agent='{args.agent}' ...")
    stats = run_simulation(
        n_games    = args.n,
        agent_name = args.agent,
        depth      = args.depth,
        seed       = args.seed,
        verbose    = args.verbose,
    )
    stats.print_summary()

    if args.csv:
        save_csv(stats, args.csv)


if __name__ == "__main__":
    main()
