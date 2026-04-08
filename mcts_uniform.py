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

Compare against random baseline with --sims 0:
    python mcts_uniform.py --sims 0 --games 100
"""

import argparse
import math
import random
import time
from collections import Counter, deque

import numpy as np

from game import Game2048, Move

N_ACTIONS = 4


# ──────────────────────────────────────────── reward ────────────────────────

def compute_reward(moved: bool, merge_reward: int, game: Game2048) -> float:
    if not moved:
        return -1.0
    # merge = math.sqrt(merge_reward + 1)
    # empty_bonus = 0.2 * game.n_empty
    # return merge + empty_bonus
    return merge_reward


# ──────────────────────────────────────────── MCTS node ─────────────────────

class Node:
    """One node in the search tree. Prior is always uniform over legal moves."""

    def __init__(self, mask: np.ndarray):
        n_legal = mask.sum()
        self.mask  = mask
        self.prior = np.where(mask, 1.0 / max(n_legal, 1), 0.0)  # uniform over legal moves
        self._N    = np.zeros(N_ACTIONS)    # Visit count for each action
        self._W    = np.zeros(N_ACTIONS)    # Total value for each action

    @property
    def total_visits(self) -> int:
        return int(self._N.sum())

    @property
    def Q(self) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self._N > 0, self._W / self._N, 0.0)

    def select_action(self, c: float) -> int:
        """PUCT with prior."""
        # Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration = c * self.prior * np.sqrt(self.total_visits + 1) / (1 + self._N)
        puct = np.where(self.mask, self.Q + exploration, -np.inf)
        
        # Selects randomly among ties
        max_val = np.max(puct)
        candidates = np.where(puct == max_val)[0]
        return int(np.random.choice(candidates))
        # return int(np.argmax(puct)) # Selects first index
    
    # def select_action(self, c: float) -> int:
    #     """PUCT without prior."""
    #     # Equation 9.1 Q(s,a) + c * sqrt(N(s) / N(s,a))
    #     # If N(s,a) = 0, treat Q(s,a) as 0 and exploration as infinite (always select unvisited actions first)
    #     exploration = np.where(self._N > 0, c * np.sqrt(self.total_visits / self._N), np.inf)
    #     puct = np.where(self.mask, self.Q + exploration, -np.inf)
    #     return int(np.argmax(puct))

    def update(self, action: int, value: float):
        self._N[action] += 1
        self._W[action] += value

    def best_action(self) -> int:
        """Most-visited legal action."""
        # TODO Why is it most visited rathan than highest? 
        masked_N = np.where(self.mask, self._N, -1)
        return int(np.argmax(masked_N))
    
    def sampled_action(self) -> int:
        """Sample an action according to visit counts."""
        # Mask invalid actions
        valid_N = np.where(self.mask, self._N, 0.0)

        # Handle edge case: no visits yet
        if valid_N.sum() == 0:
            valid_actions = np.where(self.mask)[0]
            return int(np.random.choice(valid_actions))

        # Convert visit counts → probabilities
        probs = valid_N / valid_N.sum()

        # Sample action
        action = np.random.choice(len(probs), p=probs)
        return int(action)


# ──────────────────────────────────────────── MCTS ──────────────────────────

class UniformMCTS:
    """
    Pure MCTS — no neural network, no learned policy.

    Selection:  PUCT with uniform prior
    Expansion:  on first visit to a state
    Evaluation: random rollout for `rollout_depth` steps
    Backprop:   discounted return along the path
    """

    def __init__(
        self,
        c:               float = 1.5,
        n_simulations:   int   = 200,
        gamma:           float = 0.99,
        rollout_depth:   int   = 50,
        terminal_penalty: float = 0.0,
    ):
        self.c                = c
        self.n_simulations    = n_simulations
        self.gamma            = gamma
        self.rollout_depth    = rollout_depth
        self.terminal_penalty = terminal_penalty
        self.tree: dict[tuple, Node] = {}

    def reset_tree(self):
        self.tree = {}

    def _key(self, board: np.ndarray) -> tuple:
        # TODO Hash the board for better performace?
        return tuple(board.flatten().tolist())

    def _rollout(self, game: Game2048) -> float:
        """Play randomly for rollout_depth steps; return discounted reward sum."""
        # TODO do i need to deep copy the game here? 
        g = Game2048.from_board(game.board, game.score)
        G = 0.0
        discount = 1.0  # What is discount? Related to how much future rewards are counted in this states 
        for _ in range(self.rollout_depth):
            if g.is_over:
                G += discount * self.terminal_penalty
                break
            moves = g.available_moves()
            if not moves: # Do i need this check since g.is_over should cover it?
                break
            move = random.choice(moves)
            moved, merge_reward = g.step(move)
            G += discount * compute_reward(moved, merge_reward, g)
            discount *= self.gamma
        return G

    def _expand(self, game: Game2048) -> float:
        """
        Add node to tree; 
        return rollout value estimate.
        """
        mask = np.array([m in game.available_moves() for m in Move], dtype=bool)
        key  = self._key(game.board)
        self.tree[key] = Node(mask=mask)
        return self._rollout(game)

    def _simulate(self, root_game: Game2048) -> None:
        '''
        Run one MCTS simulation from the root game state, updating the tree.
        '''
        g    = Game2048.from_board(root_game.board, root_game.score)
        path = []  # (key, action, reward)

        # SELECTION + EXPANSION
        while True:
            key = self._key(g.board)

            if g.is_over:
                value = self.terminal_penalty
                break

            if key not in self.tree:
                value = self._expand(g)
                break

            node   = self.tree[key]
            action = node.select_action(self.c)
            move   = Move(action)
            moved, merge_reward = g.step(move)
            reward = compute_reward(moved, merge_reward, g)
            path.append((key, action, reward))

            if not moved:
                value = reward
                break

        # BACKPROPAGATION
        G = value
        for key, action, reward in reversed(path):
            G = reward + self.gamma * G
            self.tree[key].update(action, G)

    def best_action(self, game: Game2048) -> int:
        """Run simulations and return the most-visited action."""
        root_key = self._key(game.board)

        if root_key not in self.tree:
            mask = np.array([m in game.available_moves() for m in Move], dtype=bool)
            self.tree[root_key] = Node(mask=mask)

        for _ in range(self.n_simulations):
            self._simulate(game)

        return self.tree[root_key].sampled_action()


# ──────────────────────────────────────────── play ──────────────────────────

def play_games(
    n_games:        int   = 20,
    n_simulations:  int   = 200,
    rollout_depth:  int   = 50,
    c:              float = 1.5,
    gamma:          float = 0.99,
    reuse_tree:     bool  = False,
    seed:           int   = 0,
):
    """
    Play n_games with the uniform MCTS agent and print stats.

    reuse_tree=True  — keep the tree between moves within a game (faster, less accurate)
    reuse_tree=False — reset tree each move (expensive but fully correct)
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
            if not reuse_tree:
                mcts.reset_tree()
            action = mcts.best_action(game)
            game.step(Move(action))

        scores.append(game.score)
        max_tiles.append(game.max_tile)

        # Log progress every 10% of games
        if (i + 1) % max(1, n_games // 10) == 0:
            print(f"  game {i+1:>4}/{n_games} | score {game.score:>7} | max tile {game.max_tile}")

    elapsed = time.time() - t0
    _print_results(n_games, scores, max_tiles, elapsed)


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
    elapsed = time.time() - t0
    _print_results(n_games, scores, max_tiles, elapsed, label="Random baseline")


def _print_results(n_games, scores, max_tiles, elapsed, label="Uniform MCTS"):
    tile_dist = Counter(max_tiles)
    print(f"\n{'='*50}")
    print(f"  {label} — {n_games} games")
    print(f"{'='*50}")
    print(f"  Mean score      : {np.mean(scores):>10.1f}")
    print(f"  Median score    : {np.median(scores):>10.1f}")
    print(f"  Max score       : {np.max(scores):>10}")
    print(f"  Win rate (≥2048): {sum(t >= 2048 for t in max_tiles)/n_games*100:>7.1f}%")
    print(f"  Duration        : {elapsed:.1f}s")
    print(f"\n  Max tile distribution:")
    for tile in sorted(tile_dist):
        pct = tile_dist[tile] / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"    {tile:>5}: {bar:<40} {pct:.1f}%")


def display(
    n_games:       int   = 1,
    speed:         int   = 4,
    n_simulations: int   = 200,
    rollout_depth: int   = 50,
):
    import pygame

    mcts = UniformMCTS(n_simulations=n_simulations, rollout_depth=rollout_depth)

    pygame.init()
    screen = pygame.display.set_mode((400, 450))
    pygame.display.set_caption("2048 — Uniform MCTS")
    font_t = pygame.font.SysFont(None, 40)
    font_s = pygame.font.SysFont(None, 28)
    clk    = pygame.time.Clock()

    TILE_COLORS = {
        0: (205,193,180), 2: (238,228,218), 4: (237,224,200),
        8: (242,177,121), 16: (245,149,99),  32: (246,124,95),
        64: (246,94,59),  128: (237,207,114), 256: (237,204,97),
        512: (237,200,80), 1024: (237,197,63), 2048: (237,194,46),
    }

    for game_i in range(n_games):
        game = Game2048()
        mcts.reset_tree()

        while not game.is_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = mcts.best_action(game)
            game.step(Move(action))

            screen.fill((187, 173, 160))
            for r in range(4):
                for c in range(4):
                    val   = int(game.board[r, c])
                    color = TILE_COLORS.get(val, (60, 58, 50))
                    rect  = pygame.Rect(c*97+8, r*97+58, 90, 90)
                    pygame.draw.rect(screen, color, rect, border_radius=6)
                    if val > 0:
                        txt = font_t.render(str(val), True, (119,110,101))
                        screen.blit(txt, txt.get_rect(center=rect.center))

            score_txt = font_s.render(
                f"Score: {game.score}   Max: {game.max_tile}", True, (255,255,255))
            screen.blit(score_txt, (10, 15))
            pygame.display.flip()
            if speed > 0:
                clk.tick(speed)

        print(f"Game {game_i+1}: score={game.score}  max_tile={game.max_tile}")

    pygame.quit()


# ──────────────────────────────────────────── CLI ───────────────────────────

def main():
    p = argparse.ArgumentParser(description="Uniform-policy MCTS for 2048")
    p.add_argument("--games",    type=int,   default=20,  help="number of games to play")
    p.add_argument("--sims",     type=int,   default=200, help="MCTS simulations per move (0 = random)")
    p.add_argument("--rollout",  type=int,   default=10,  help="random rollout depth for leaf eval")
    p.add_argument("--c",        type=float, default=160, help="PUCT exploration constant")
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--reuse",    action="store_false", help="reuse tree between moves in a game")
    p.add_argument("--baseline", action="store_true",     help="also run random baseline")
    p.add_argument("--display",  type=int,   default=-1,  help="watch agent play at DISPLAY fps")
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()

    if args.display != -1:
        display(n_simulations=args.sims, rollout_depth=args.rollout, speed=args.display)
        return

    if args.sims == 0:
        play_random(n_games=args.games, seed=args.seed)
        return

    print(f"Uniform MCTS | sims={args.sims} rollout={args.rollout} c={args.c} reuse_tree={args.reuse}")
    play_games(
        n_games=args.games,
        n_simulations=args.sims,
        rollout_depth=args.rollout,
        c=args.c,
        gamma=args.gamma,
        reuse_tree=args.reuse,
        seed=args.seed,
    )

    if args.baseline:
        print("\nRunning random baseline for comparison...")
        play_random(n_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
