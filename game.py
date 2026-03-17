"""
2048 Game Engine
================
Pure backend implementation with no GUI dependencies.
Designed for simulation, AI research, and headless batch runs.
"""

import random
import numpy as np
from copy import deepcopy
from enum import IntEnum
from typing import Optional


class Move(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


class Game2048:
    """
    Self-contained 2048 game engine.

    The board is a 4x4 numpy array of integers.
    Empty cells are represented as 0.

    Typical usage
    -------------
    game = Game2048()
    while not game.is_over():
        move = your_policy(game)        # Move.UP / DOWN / LEFT / RIGHT
        moved, reward = game.step(move)
    print(game.score)

    Simulation usage
    ----------------
    game = Game2048(seed=42)            # reproducible
    state = game.board.copy()           # snapshot
    game2 = Game2048.from_board(state)  # clone
    """

    SIZE = 4

    # ------------------------------------------------------------------ init

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> "Game2048":
        """Reset to a fresh game and return self (for chaining)."""
        self.board   = np.zeros((self.SIZE, self.SIZE), dtype=np.int64)
        self.score   = 0
        self._over   = False
        self._add_random_tile()
        self._add_random_tile()
        return self

    @classmethod
    def from_board(cls, board: np.ndarray, score: int = 0,
                   seed: Optional[int] = None) -> "Game2048":
        """Create a game from an existing board state (e.g. for tree search)."""
        g = cls(seed=seed)
        g.board = board.copy().astype(np.int64)
        g.score = score
        g._over = not g._has_moves()
        return g

    # ----------------------------------------------------------------- state

    @property
    def is_over(self) -> bool:
        return self._over

    @property
    def max_tile(self) -> int:
        return int(self.board.max())

    @property
    def empty_cells(self) -> list[tuple[int, int]]:
        rows, cols = np.where(self.board == 0)
        return list(zip(rows.tolist(), cols.tolist()))

    @property
    def n_empty(self) -> int:
        return int((self.board == 0).sum())

    def won(self) -> bool:
        return bool((self.board >= 2048).any())

    # ------------------------------------------------------------------ step

    def step(self, move: Move) -> tuple[bool, int]:
        """
        Apply a move.

        Returns
        -------
        moved  : bool  – False if the board didn't change (illegal move)
        reward : int   – points earned this step (sum of merged tiles)
        """
        if self._over:
            return False, 0

        before = self.board.copy()
        reward = self._apply_move(move)
        moved  = not np.array_equal(self.board, before)

        if moved:
            self._add_random_tile()
            if not self._has_moves():
                self._over = True

        return moved, reward

    def available_moves(self) -> list[Move]:
        """Return list of moves that would actually change the board."""
        moves = []
        for m in Move:
            test = Game2048.from_board(self.board, self.score)
            moved, _ = test.step(m)
            if moved:
                moves.append(m)
        return moves

    # ------------------------------------------------------------- internals

    def _add_random_tile(self):
        empty = self.empty_cells
        if not empty:
            return
        r, c = self._rng.choice(empty)
        self.board[r, c] = 4 if self._rng.random() < 0.1 else 2

    def _apply_move(self, move: Move) -> int:
        """Rotate board so every move reduces to a LEFT merge."""
        rotations = {Move.LEFT: 0, Move.RIGHT: 2,
                     Move.UP:   1, Move.DOWN:  3}
        k = rotations[move]
        self.board = np.rot90(self.board, k)
        reward = self._merge_left()
        self.board = np.rot90(self.board, -k % 4)
        return reward

    def _merge_left(self) -> int:
        """Slide and merge all rows to the left. Returns score delta."""
        reward = 0
        for r in range(self.SIZE):
            row          = self.board[r]
            new_row, pts = _slide_and_merge(row)
            self.board[r] = new_row
            reward       += pts
        self.score += reward
        return reward

    def _has_moves(self) -> bool:
        if self.n_empty > 0:
            return True
        # Check for adjacent equal tiles
        if np.any(self.board[:, :-1] == self.board[:, 1:]):
            return True
        if np.any(self.board[:-1, :] == self.board[1:, :]):
            return True
        return False

    # ------------------------------------------------------- representation

    def __repr__(self) -> str:
        lines = [f"Score: {self.score}  Max: {self.max_tile}"]
        sep   = "+" + "+".join(["------"] * self.SIZE) + "+"
        lines.append(sep)
        for row in self.board:
            cells = "|".join(f"{v:^6}" if v else f"{'':^6}" for v in row)
            lines.append(f"|{cells}|")
            lines.append(sep)
        return "\n".join(lines)


# ------------------------------------------------------------------ helpers

def _slide_and_merge(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide nonzero values left, merge equal adjacent pairs, return (new_row, pts)."""
    tiles  = row[row != 0].tolist()
    merged = []
    pts    = 0
    i      = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            val   = tiles[i] * 2
            pts  += val
            merged.append(val)
            i    += 2
        else:
            merged.append(tiles[i])
            i    += 1
    result          = np.zeros(len(row), dtype=np.int64)
    result[:len(merged)] = merged
    return result, pts
