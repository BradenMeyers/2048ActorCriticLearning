'''
2048 Actor-Critic Utils
'''
import math
from runners.game import Game2048
import torch
import torch.nn as nn

# TODO: Should look at gamma because we care about future rewards a lot
# but we need it to make it so we don't lose the game. 
# Do we need a negative reward for losing?

N_ACTIONS = 4   # UP DOWN LEFT RIGHT
ACTION_NAMES = ["UP", "DN ", "LT ", "RT "]
N_TILE_LEVELS = 16          # levels 0..15
STATE_DIM     = 4 * 4 * N_TILE_LEVELS   # 256
# ================================================================= reward
# TODO could add reward for max tile


def circ_reward(moved: bool, merge_reward: int, game: Game2048,
                   episode: int = 1, total_episodes: int = 1) -> float:
    """
    Curriculum reward — shifts from survival-focused to score-focused over time.

    Early training:  empty_coef=0.8, merge_coef=0.3  → learn to keep board open
    Late training:   empty_coef=0.1, merge_coef=1.0  → learn to build big tiles

    Why curriculum?
        Board management (not losing) and tile building (scoring big) are
        somewhat separate skills. Learning survival first gives the agent the
        foundation to then exploit open space for large merges.

    Why not just empty cells?
        Merging reduces tile count → fewer empty cells → negative reward.
        The agent would learn to avoid merging. Empty cell bonus only works
        correctly as a secondary signal on top of a merge reward.
    """
    if not moved:
        return -1.0

    progress   = episode / max(total_episodes, 1)
    merge_coef = 0.3 * (1 - progress) + 1.0 * progress   # 0.3 → 1.0
    empty_coef = 0.8 * (1 - progress) + 0.1 * progress   # 0.8 → 0.1

    log_merge      = math.log2(merge_reward + 1)
    empty_bonus    = empty_coef * game.n_empty
    max_tile_bonus = 0.1 * math.log2(game.max_tile + 1)  # reward maintaining high tiles

    return merge_coef * log_merge + empty_bonus + max_tile_bonus

def sqrt_reward(moved: bool, merge_reward: int, game: Game2048, empty_bonus: float = 0.2) -> float:
    """
    # TODO could look at reward shaping for improving the credit assignment. 
    Shaped reward function.

    Why empty cell bonus?
        Most steps have merge_reward=0 (no merge happened).  Without a per-step
        signal the agent gets almost no feedback and learning is very slow.
        Rewarding open board space gives a gradient every single step.
    """
    if not moved:
        return -1.0     # small penalty for illegal move (shouldn't happen with masking)

    merge = math.sqrt(merge_reward + 1)
    empty_bonus = empty_bonus * game.n_empty # Should i reward this more?
    
    return (merge + empty_bonus) 

def log_reward(moved: bool, merge_reward: int, game: Game2048, empty_bonus: float = 0.2) -> float:
    if not moved:
        return -1.0     # small penalty for illegal move (shouldn't happen with masking)

    log_merge = math.log2(merge_reward + 1)
    empty_bonus = empty_bonus * game.n_empty # Should i reward this more? 
    return (log_merge + empty_bonus) 

# TODO: upgrade to GAE (Generalized Advantage Estimation) for lower variance.
def compute_returns(rewards: list, last_value: float, gamma: float) -> list:
    """
    Reward-to-go: G_t = r_t + y*r_{t+1} + γ²*r_{t+2} + ...

    Built backwards from episode end. last_value bootstraps from the critic
    if the episode hit max_steps (otherwise 0.0 — game ended naturally).

    This is what the critic is trained to predict.
    """
    returns = []
    G = last_value
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

# Other Utils

class RunningNormalizer:
    """
    Maintains an exponential moving average of return mean and variance,
    then normalizes returns by those running stats.

    Why not per-batch normalization?
        Per-batch stats shift every update, so the critic's output scale
        changes every gradient step. When MCTS queries the critic for a leaf
        value, it gets a number whose meaning depends on the last batch seen —
        not the absolute quality of the board state.

    Why not raw returns?
        Returns from sqrt_reward accumulate to ~100-500 over a full game.
        MSE on raw values that large causes exploding gradients early in training.

    This gives you the best of both: stable scale, consistent meaning.
    """
    def __init__(self, momentum: float = 0.99):
        self.mean = 0.0
        self.var  = 1.0
        self.momentum = momentum
        self._initialized = False

    def update(self, x: torch.Tensor):
        """Update running stats with a new batch of returns."""
        batch_mean = x.mean().item()
        batch_var  = x.var().item() if x.numel() > 1 else 1.0
        if not self._initialized:
            self.mean = batch_mean
            self.var  = max(batch_var, 1e-4)
            self._initialized = True
        else:
            m = self.momentum
            self.mean = m * self.mean + (1 - m) * batch_mean
            self.var  = m * self.var  + (1 - m) * max(batch_var, 1e-4)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using running stats (not the current batch)."""
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)


def action_mask(game: Game2048) -> torch.Tensor:
    """
    Returns a (4,) boolean tensor — True for legal moves.
    Used to zero out illegal action logits before softmax.

    Due to problems with making moves that dont change the board
    """
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool)
    for m in game.available_moves():
        mask[int(m)] = True
    return mask
