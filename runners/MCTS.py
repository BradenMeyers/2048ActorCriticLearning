"""
MCTS.py — Monte Carlo Tree Search for 2048
===========================================

Class hierarchy
---------------
    BaseMCTS              — shared tree, board key, reset_tree, abstract _expand/_simulate
    ├── MCTS              — neural-network guided (AlphaZero-style)
    └── UniformMCTS       — pure MCTS with uniform prior and random rollout (diagnostic)

Node
----
    Single Node class — prior is set by the caller (network output or uniform).
    Supports get_policy(eta), best_action(), sampled_action(), and debug_str().
"""

import random
from abc import ABC, abstractmethod

import numpy as np
import torch

from runners.game import Game2048, Move
from runners.utils import N_ACTIONS, ACTION_NAMES
from runners.utils import sqrt_reward, action_mask


# ─────────────────────────────────────── helpers ────────────────────────────

def _board_str(board: np.ndarray) -> str:
    """Return a compact string representation of the 4×4 board for debug output."""
    rows = []
    for r in range(4):
        rows.append("  " + " ".join(f"{int(board[r, c]):>5}" for c in range(4)))
    return "\n".join(rows)


# ─────────────────────────────────────── node ───────────────────────────────

class Node:
    """
    Tree node for MCTS.

    The caller sets the prior:
      - network policy output  → used by MCTS
      - uniform over legal moves → used by UniformMCTS

    original_prior stores the clean prior so Dirichlet noise can be re-applied
    on each call without compounding (MCTS only).
    """

    def __init__(self, mask: np.ndarray, prior: np.ndarray = None):
        self.mask           = mask
        if prior is None:
            n_legal = mask.sum()
            prior   = np.where(mask, 1.0 / max(n_legal, 1), 0.0)
        self.prior          = prior
        self.original_prior = prior.copy()
        self._N             = np.zeros(N_ACTIONS)
        self._W             = np.zeros(N_ACTIONS)

    @property
    def total_visits(self) -> int:
        return int(self._N.sum())

    @property
    def Q(self) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self._N > 0, self._W / self._N, 0.0)

    def update(self, action: int, value: float):
        self._N[action] += 1
        self._W[action] += value

    def select_action(self, c: float) -> int:
        """PUCT: Q + c * P * sqrt(N_parent) / (1 + N_child). Breaks ties randomly."""
        exploration = c * self.prior * np.sqrt(self.total_visits + 1) / (1 + self._N)
        puct = np.where(self.mask, self.Q + exploration, -np.inf)
        max_val = np.max(puct)
        candidates = np.where(puct == max_val)[0]
        return int(np.random.choice(candidates))

    def get_policy(self, eta: float) -> np.ndarray:
        """Visit-count distribution used to train the actor head.

        eta=1   → proportional to visit counts
        eta=inf → one-hot on most-visited action
        """
        if eta == float('inf'):
            policy = np.zeros(N_ACTIONS, dtype=np.float32)
            policy[np.argmax(self._N)] = 1.0
            return policy
        counts = self._N ** eta
        total = counts.sum()
        if total == 0:
            policy = np.where(self.mask, 1.0 / self.mask.sum(), 0.0)
            return policy / policy.sum()
        return counts / total

    def best_action(self) -> int:
        """Most-visited legal action."""
        return int(np.argmax(np.where(self.mask, self._N, -1)))

    def sampled_action(self) -> int:
        """Sample an action proportional to visit counts."""
        valid_N = np.where(self.mask, self._N, 0.0)
        if valid_N.sum() == 0:
            return int(np.random.choice(np.where(self.mask)[0]))
        probs = valid_N / valid_N.sum()
        return int(np.random.choice(len(probs), p=probs))

    def debug_str(self, chosen_action: int = -1) -> str:
        total = max(self.total_visits, 1)
        lines = []
        for a in range(N_ACTIONS):
            marker = " <--" if a == chosen_action else ""
            legal  = "" if self.mask[a] else " [illegal]"
            lines.append(
                f"  {ACTION_NAMES[a]}  N={int(self._N[a]):>4} ({int(self._N[a]/total*100):>2}%)"
                f"  Q={self.Q[a]:>7.2f}"
                f"  P={self.prior[a]:.2f}"
                f"{legal}{marker}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────── base class ─────────────────────────

class BaseMCTS(ABC):
    """
    Abstract base for MCTS variants.

    Subclasses must implement:
        _expand(game)   → float   — add a leaf node, return its value estimate
        _simulate(game) → None    — one full select/expand/backprop pass
        best_action(game) → int   — run simulations, return the chosen action

    Shared:
        _board_key(board) — hashable board state
        reset_tree()      — clear the tree dict
    """

    def __init__(
        self,
        n_simulations:    int   = 100,
        c:                float = 25.0,
        gamma:            float = 0.99,
        terminal_penalty: float = 0.0,
        dir_alpha:        float = 0.3,
        dir_epsilon:      float = 0.0,   # 0 = no noise; MCTS overrides to 0.25
    ):
        self.n_simulations    = n_simulations
        self.c                = c
        self.gamma            = gamma
        self.terminal_penalty = terminal_penalty
        self.dir_alpha        = dir_alpha
        self.dir_epsilon      = dir_epsilon
        self.tree: dict       = {}

    def _board_key(self, board: np.ndarray) -> tuple:
        """Hashable representation of a board state."""
        return tuple(board.flatten().tolist())

    def reset_tree(self) -> None:
        """Clear the search tree. Call once at the start of each game/episode."""
        self.tree = {}

    @abstractmethod
    def _get_prior_and_value(self, game: Game2048) -> tuple[np.ndarray, float, np.ndarray]:
        """Return (prior, value, mask) for the given game state."""
        ...

    def _expand(self, game: Game2048) -> float:
        """Add a Node to the tree; return the leaf value estimate."""
        key = self._board_key(game.board)
        prior, value, mask = self._get_prior_and_value(game)
        self.tree[key] = Node(prior=prior, mask=mask)
        return value

    def _simulate(self, game: Game2048) -> None:
        """One full simulation: select → expand → evaluate → backprop."""
        game = Game2048.from_board(game.board, game.score)
        path = []

        while True:
            key = self._board_key(game.board)

            if game.is_over:
                value = self.terminal_penalty
                break

            if key not in self.tree:
                value = self._expand(game)
                break

            node   = self.tree[key]
            action = node.select_action(self.c)
            moved, merge_reward = game.step(Move(action))
            reward = sqrt_reward(moved, merge_reward, game)
            path.append((key, action, reward))

            if not moved:
                value = 0.0
                print("Warning: MCTS selected an illegal move. Ending simulation.")
                break

        G = value
        for key, action, reward in reversed(path):
            G = reward + self.gamma * G
            self.tree[key].update(action, G)

    def get_policy(self, game: Game2048, eta: float = 1.0,
                   add_noise: bool = False, debug: bool = False) -> np.ndarray:
        """
        Run n_simulations and return a policy distribution over actions.

        eta=1   → proportional to visit counts  (use during training)
        eta=inf → one-hot on most-visited action (use during evaluation)
        add_noise=True → inject Dirichlet noise into the root prior; only has
                         effect when dir_epsilon > 0 (i.e. MCTS, not UniformMCTS).

        The tree carries visit counts across calls within the same episode
        (warm-start). Call reset_tree() between episodes.
        """
        root_key = self._board_key(game.board)

        if root_key not in self.tree:
            self._expand(game)

        if add_noise and self.dir_epsilon > 0 and root_key in self.tree:
            root  = self.tree[root_key]
            noise = np.random.dirichlet([self.dir_alpha] * N_ACTIONS)
            root.prior = (1 - self.dir_epsilon) * root.original_prior + self.dir_epsilon * noise

        for _ in range(self.n_simulations):
            self._simulate(game)

        if root_key not in self.tree:
            mask   = np.array([m in game.available_moves() for m in Move])
            policy = mask.astype(float)
            return policy / policy.sum()

        policy = self.tree[root_key].get_policy(eta)

        if debug:
            chosen = int(np.argmax(policy))
            print(f"\n[MCTS] score={game.score}  sims={self.n_simulations}  tree_nodes={len(self.tree)}")
            print(_board_str(game.board))
            print(self.tree[root_key].debug_str(chosen))

        return policy

    def best_action(self, game: Game2048, eta: float = float("inf")) -> int:
        """
        Run simulations and return the best action.

        eta=inf → argmax of visit counts (deterministic, default)
        eta=1   → sample proportional to visit counts (stochastic)
        """
        policy = self.get_policy(game, eta=eta)
        if eta == float("inf"):
            return int(np.argmax(policy))
        return int(np.random.choice(len(policy), p=policy))


# ─────────────────────────────────────── MCTS ────────────────────────────────

class MCTS(BaseMCTS):
    """
    Monte Carlo Tree Search guided by a trained ActorCritic network.

    The network serves two roles:
        - Actor (policy head): prior π(a|s) biases which branches to explore
        - Critic (value head):  V(s) evaluates leaf nodes without rollouts

    2048 is stochastic — after each move a random tile spawns. We handle
    this by sampling one tile placement per simulation (rather than averaging
    over all placements). This introduces some variance but keeps the tree
    simple and is standard practice for stochastic MCTS.
    """

    def __init__(
        self,
        net,
        device,
        c:                float = 25.0,
        n_simulations:    int   = 100,
        gamma:            float = 0.99,
        empty_threshold:  int   = 6,
        dir_alpha:        float = 0.3,
        dir_epsilon:      float = 0.25,
        terminal_penalty: float = 0.0,
    ):
        super().__init__(n_simulations=n_simulations, c=c, gamma=gamma,
                         terminal_penalty=terminal_penalty,
                         dir_alpha=dir_alpha, dir_epsilon=dir_epsilon)
        self.net             = net
        self.device          = device
        self.empty_threshold = empty_threshold

    def _get_prior_and_value(self, game: Game2048):
        """Forward pass → (prior, value, mask)."""
        state = self.net.board_to_tensor(game.board).to(self.device)
        mask  = action_mask(game).to(self.device)
        with torch.no_grad():
            policy, value = self.net(state, mask)
        return policy.cpu().numpy(), value.cpu().item(), mask.cpu().numpy()

    def search(self, game: Game2048) -> int:
        """
        Return the best action, with a greedy fallback on open boards.

        When there are many empty cells, tile spawns make tree hits rare so
        MCTS adds little over a single network pass. Above empty_threshold,
        return the greedy argmax directly.
        """
        if game.n_empty > self.empty_threshold:
            state = self.net.board_to_tensor(game.board).to(self.device)
            mask  = action_mask(game).to(self.device)
            with torch.no_grad():
                policy, _ = self.net(state, mask)
            return int(policy.argmax().item())

        return self.best_action(game)


# ─────────────────────────────────────── UniformMCTS ────────────────────────

class UniformMCTS(BaseMCTS):
    """
    Pure MCTS — no neural network, no learned policy.

    Selection:  PUCT with uniform prior  (UniformNode)
    Expansion:  on first visit to a state
    Evaluation: random rollout for rollout_depth steps
    Backprop:   discounted return along the path

    This uses raw merge reward (not the shaped rewards in utils) intentionally —
    the goal is to test whether the tree search itself works, independent of
    reward engineering.
    """

    def __init__(
        self,
        c:                float = 160.0,
        n_simulations:    int   = 200,
        gamma:            float = 0.99,
        rollout_depth:    int   = 50,
        terminal_penalty: float = -10.0,
        dir_alpha:        float = 0.3,
        dir_epsilon:      float = 0.0,   # 0 = no noise; MCTS overrides to 0.25
    ):
        super().__init__(n_simulations=n_simulations, c=c, gamma=gamma, terminal_penalty=terminal_penalty, dir_alpha=dir_alpha, dir_epsilon=dir_epsilon)
        self.rollout_depth    = rollout_depth


    def _rollout(self, game: Game2048) -> float:
        """Play randomly for rollout_depth steps; return discounted reward sum."""
        g        = Game2048.from_board(game.board, game.score)
        G        = 0.0
        discount = 1.0
        for _ in range(self.rollout_depth):
            if g.is_over:
                G += discount * self.terminal_penalty
                break
            moves = g.available_moves()
            if not moves:
                break
            moved, merge_reward = g.step(random.choice(moves))
            G        += discount * (float(merge_reward) if moved else -1.0)
            discount *= self.gamma
        return G

    def _get_prior_and_value(self, game: Game2048) -> tuple[np.ndarray, float, np.ndarray]:
        """Uniform prior + rollout value estimate."""
        mask = np.array([m in game.available_moves() for m in Move], dtype=bool)
        return None, self._rollout(game), mask

    def best_action(self, game: Game2048, eta: float = 1.0) -> int:
        """Sample proportional to visit counts by default (eta=1), not argmax."""
        return super().best_action(game, eta=eta)
