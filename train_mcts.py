"""
AC MCTS ( MCTS with Actor-Critic) for 2048
==
Trains a neural network to play 2048 using the MCTS with Actor-Critic algorithm.

Usage
-----
    pip install torch numpy
    python train_mcts.py                        # train with defaults
    python train_mcts.py --episodes 5000        # more training
    python train_mcts.py --eval                 # evaluate a saved checkpoint
    python train_mcts.py --display 2            # watch agent play at 2 fps

Files produced
--------------
    mcts_checkpoint.pt   - saved model weights
    mcts_log.csv         - per-episode training stats

Architecture
------------
    State:  log2 encoding → (1, 4, 4) spatial tensor
    Trunk:  CNN — Conv2d(1→64, 2x2) → Conv2d(64→128, 2x2) → Linear(256)
    Heads:  Actor → 4 action logits | Critic → 1 scalar value
"""

import argparse
import csv
import math
import os
import time
from collections import deque, Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game import Game2048, Move


# ===== constants

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4   # UP DOWN LEFT RIGHT

class MCTSNode:
    # Represents a node in the MCTS tree.
    def __init__(self, prior: np.ndarray, mask: np.ndarray):
        self.prior = prior          # Prior policy from the actor head (4,)
        self.mask  = mask           # Legal move mask (4,)
        self._N = np.zeros(N_ACTIONS)              # Visit count
        self._W   = np.zeros(N_ACTIONS)          # Total value of all visits (for computing Q)
        # self.children    = [None] * N_ACTIONS  # Child nodes for each action

    @property
    def total_visits(self) -> int:
        return int(self._N.sum())

    @property
    def N(self) -> np.ndarray:
        return self._N
    
    @property
    def W(self) -> np.ndarray:
        return self._W
    
    @property
    def Q(self) -> np.ndarray:
        # Q = W / N, but handle division by zero for unvisited actions
        with np.errstate(divide='ignore', invalid='ignore'):
            Q = np.where(self._N > 0, self._W / self._N, 0.0)
        return Q
    
    def update(self, action: int, value: float):
        """
        Update visit count and total value for the given action.
        Called during backpropagation after a simulation.

        value is the return from the simulation (e.g. final score or critic value).
        We want to maximize this, so we add it to W.
        """
        self._N[action] += 1  # Increment visit count for this action
        self._W[action] += value # Add the simulation return to total value for this action


    def select_action(self, c: float) -> int:
        """
        Select action using PUCT formula: Q + c * P * sqrt(N_parent) / (1 + N_child)

        Q = W / N for each action (0 if N=0)
        P = prior policy from actor head
        N_parent = total visits to this node
        N_child = visits to each child action

        c controls exploration vs exploitation:
            c=0 → greedy (always pick highest Q)
            c→∞ → pure exploration (pick according to prior P)

        """
        total_visits = self.total_visits
        exploration = c * self.prior * np.sqrt(total_visits + 1) / (1 + self.N)

        puct = np.where(self.mask, self.Q + exploration, -np.inf)  # Mask illegal moves with -inf

        return int(np.argmax(puct))  # Return the action index with highest PUCT score
    
    def get_policy(self, eta: float) -> np.ndarray:
        """
        Convert visit counts to a policy distribution for training the actor head.

        eta = 1, porportion to visit counts (π(a) = N(a) / N_total)
        eta → 0, becomes one-hot on most visited action (π(a) = 1 if N(a) = max else 0)
        """
        if eta == float('inf'):
            # Deterministic: one-hot on most visited action
            policy = np.zeros(N_ACTIONS, dtype=np.float32)
            best_action = np.argmax(self.N)
            policy[best_action] = 1.0
            return policy
        
        counts = self.N ** eta
        total_counts = counts.sum()
        if total_counts == 0:
            # No visits, return uniform distribution over legal moves
            policy = np.where(self.mask, 1.0 / self.mask.sum(), 0.0)
            return policy / policy.sum()  # Normalize to sum to 1
        return counts / total_counts  # Normalize to get probabilities
    


class MCTS:
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
        net,                          # trained ActorCritic
        device,                       # torch device
        c:              float = 1.5,  # exploration constant in PUCT
        n_simulations:  int   = 100,  # simulations per move
        gamma:          float = 0.99, # discount factor for backprop returns
        empty_threshold: int  = 6,    # use greedy above this many empty cells
    ):
        self.net              = net
        self.device           = device
        self.c                = c
        self.n_simulations    = n_simulations
        self.gamma            = gamma
        self.empty_threshold  = empty_threshold
 
        # Tree: maps board_key → MCTSNode
        # Board key is a tuple of 16 ints (flattened board)
        self.tree: dict[tuple, MCTSNode] = {}
 
    def _board_key(self, board: np.ndarray) -> tuple:
        # So we can find similar board states. 
        """Hashable representation of a board state."""
        return tuple(board.flatten().tolist())
 
    def _get_prior_and_value(self, game: Game2048):
        """
        Run one forward pass of the network to get:
            prior : (4,) policy distribution πθ(a|s)
            value : scalar V(s)
            mask  : (4,) bool — legal moves
        """
 
        state = board_to_tensor(game.board).to(self.device)
        mask  = action_mask(game).to(self.device)
 
        with torch.no_grad():
            policy, value = self.net(state, mask)
 
        prior = policy.cpu().numpy()
        val   = value.cpu().item()
        msk   = mask.cpu().numpy()
        return prior, val, msk
 
    def _expand(self, game: Game2048) -> float:
        """
        Expand a new leaf node:
            1. Run network to get prior and value
            2. Add node to tree
            3. Return value estimate for backprop
 
        Returns the critic's value estimate V(s).
        """
        key = self._board_key(game.board)
        prior, value, mask = self._get_prior_and_value(game)
        self.tree[key] = MCTSNode(prior=prior, mask=mask)
        return value
 
    def _simulate(self, root_game: Game2048) -> None:
        """
        Run one full simulation: select → expand → evaluate → backprop.

        We work on clones of the game state so the real game is untouched.
        """
        game = Game2048.from_board(root_game.board, root_game.score)
        path = []   # list of (board_key, action, shaped_reward) triples for backprop

        # ---- SELECTION: walk tree using PUCT until we hit an unexpanded node ----
        while True:
            key = self._board_key(game.board)

            if game.is_over:
                # Terminal node — value is 0 (no future reward)
                value = 0.0
                break

            if key not in self.tree:
                # Unexpanded leaf — expand and get value from critic
                value = self._expand(game)
                break

            # Node exists — select action using PUCT
            node   = self.tree[key]
            action = node.select_action(self.c)

            # Take action in cloned game, record shaped reward for this step
            move = Move(action)
            moved, merge_reward = game.step(move)
            reward = compute_reward(moved, merge_reward, game)
            path.append((key, action, reward))

            if not moved:
                # This shouldn't happen with a good mask but handle gracefully
                value = 0.0
                print("Warning: MCTS selected an illegal move. Ending simulation.")
                break

        # ---- BACKPROPAGATION: accumulate discounted returns along the path ----
        # G starts at the leaf critic estimate (or 0 for terminal/error),
        # then each step folds in its own reward: G = r + γ·G
        G = value
        for key, action, reward in reversed(path):
            G = reward + self.gamma * G
            self.tree[key].update(action, G)
    
    def get_policy(self, game: Game2048, eta: float = 1.0) -> np.ndarray:
        """
        Run search and return the full πMCTS distribution.
        Used during AlphaZero-style training to get policy targets.
 
        eta=1   → proportional to visit counts (exploration during training)
        eta=inf → greedy (best play at eval time)
        """
        self.tree = {}
        for _ in range(self.n_simulations):
            self._simulate(game)
 
        root_key = self._board_key(game.board)
        if root_key not in self.tree:
            # Fallback: uniform over legal moves
            mask = np.array([m in game.available_moves() for m in Move])
            policy = mask.astype(float)
            return policy / policy.sum()
 
        return self.tree[root_key].get_policy(eta)

    def search(self, game: Game2048) -> int:
        """
        Return the best action for the current position.

        When the board has more than empty_threshold open cells, the many
        possible tile spawns dilute the tree — different simulations almost
        never revisit the same state, so MCTS adds little over a single
        network pass.  In that regime we just return the greedy argmax.
        Once the board tightens up (few empty cells → fewer spawn outcomes →
        higher chance of tree hits), we run the full MCTS search.
        """
        if game.n_empty > self.empty_threshold:
            # Greedy: one forward pass, no tree search
            state = board_to_tensor(game.board).to(self.device)
            mask  = action_mask(game).to(self.device)
            with torch.no_grad():
                policy, _ = self.net(state, mask)
            return int(policy.argmax().item())

        policy = self.get_policy(game, eta=float("inf"))
        return int(np.argmax(policy))

# = state encoder

def board_to_tensor(board: np.ndarray) -> torch.Tensor:
    """
    Convert 4x4 numpy board to a (1, 4, 4) float tensor using log2 encoding.

    Each cell → log2(value) / 15.0, normalised to [0, 1].
    Empty cells → 0.0.

    Why log2 instead of one-hot?
        One-hot treats each tile level as an independent category — the network
        has no idea 256 and 512 are related. Log2 encoding gives tiles ordinal
        meaning: equal adjacent cells are always exactly 1/15 apart in value,
        so the CNN can learn the merge condition (two equal neighbours) as a
        single consistent filter regardless of which tile level it is.

    Shape: (1, 4, 4) = (channels, H, W).
        forward() adds the batch dim → (1, 1, 4, 4) before Conv2d.
    """
    state = np.zeros((1, 4, 4), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            state[0, r, c] = math.log2(val) / 15.0 if val > 0 else 0.0
    return torch.tensor(state, dtype=torch.float32)


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


# = network
class ActorCritic(nn.Module):
    """
    CNN shared-trunk actor-critic.

    Why CNN?
        The merge condition — two equal adjacent tiles — is a 2x1 spatial
        pattern. A Conv2d filter can represent this directly. A flat linear
        network has to learn the same pattern implicitly across all 256 inputs,
        which takes much longer and has to relearn it for every tile level.

    Architecture:
        Input (1, 4, 4)
          → Conv2d(1→64,   kernel=2) → ReLU   # local 2x2 patterns
          → Conv2d(64→128, kernel=2) → ReLU   # higher-level structure
          → Flatten → Linear(512→256) → ReLU
               ↓
        ┌──────┴──────┐
        ↓             ↓
     Actor head   Critic head
     Linear(4)    Linear(1)
     softmax      raw scalar
     → π(a|s)     → V(s)
    """

    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2),    # (1,4,4) → (64,3,3)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),  # (64,3,3) → (128,2,2)
            nn.ReLU(),
            nn.Flatten(),                        # → (512,)
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
        )

        self.actor_head  = nn.Linear(256, N_ACTIONS)
        self.critic_head = nn.Linear(256, 1)


# class ActorCritic(nn.Module):
#     """
#     Shared-trunk network with two heads:
#         - Actor head  → policy π(a|s), probability over 4 moves
#         - Critic head → value V(s), expected future return

#     Architecture:
#         Input (256) → Linear(512) → ReLU → Linear(512) → ReLU
#                                                 ↓
#                               ┌─────────────────┤
#                               ↓                 ↓
#                          Actor head        Critic head
#                          Linear(4)         Linear(1)
#                          (logits)          (scalar)
#     """

#     def __init__(self, state_dim: int = STATE_DIM, hidden: int = 512):
#         super().__init__()

#         # Shared trunk — learns board representations useful to both heads
#         self.trunk = nn.Sequential(
#             nn.Linear(state_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#         )

#         # Actor head — outputs raw logits (NOT probabilities yet)
#         self.actor_head = nn.Linear(hidden, N_ACTIONS)

#         # Critic head — outputs a single scalar value estimate
#         self.critic_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters
        ----------
        state : (1, 4, 4) from board_to_tensor — (channels, H, W)
                Conv2d needs (batch, C, H, W) = (1, 1, 4, 4), added here.
        mask  : (4,) bool — True = legal move

        Returns
        -------
        policy : (4,)  probability over actions  (batch dim squeezed)
        value  : (1,)  scalar state value        (batch dim squeezed)
        """
        # (1, 4, 4) → (1, 1, 4, 4): add batch dimension for Conv2d
        if state.dim() == 3:
            state = state.unsqueeze(0)

        features = self.trunk(state)          # → (1, 256)
        logits   = self.actor_head(features)  # → (1, 4)

        # Mask illegal moves — unsqueeze mask to broadcast over batch dim
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

        policy = F.softmax(logits, dim=-1).squeeze(0)  # → (4,)
        value  = self.critic_head(features).squeeze(0) # → (1,)

        return policy, value


# ================================================================= reward

# def compute_reward(moved: bool, merge_reward: int, game: Game2048,
#                    episode: int = 1, total_episodes: int = 1) -> float:
#     """
#     Curriculum reward — shifts from survival-focused to score-focused over time.

#     Early training:  empty_coef=0.8, merge_coef=0.3  → learn to keep board open
#     Late training:   empty_coef=0.1, merge_coef=1.0  → learn to build big tiles

#     Why curriculum?
#         Board management (not losing) and tile building (scoring big) are
#         somewhat separate skills. Learning survival first gives the agent the
#         foundation to then exploit open space for large merges.

#     Why not just empty cells?
#         Merging reduces tile count → fewer empty cells → negative reward.
#         The agent would learn to avoid merging. Empty cell bonus only works
#         correctly as a secondary signal on top of a merge reward.
#     """
#     if not moved:
#         return -1.0

#     progress   = episode / max(total_episodes, 1)
#     merge_coef = 0.3 * (1 - progress) + 1.0 * progress   # 0.3 → 1.0
#     empty_coef = 0.8 * (1 - progress) + 0.1 * progress   # 0.8 → 0.1

#     log_merge      = math.log2(merge_reward + 1)
#     empty_bonus    = empty_coef * game.n_empty
#     max_tile_bonus = 0.1 * math.log2(game.max_tile + 1)  # reward maintaining high tiles

#     return merge_coef * log_merge + empty_bonus + max_tile_bonus


def compute_reward(moved: bool, merge_reward: int, game: Game2048) -> float:
    """
    # TODO could look at reward shaping for improving the credit assignment. 
    Shaped reward function.

    Components:
        1. log2(merge_value + 1)  — credit for merges, log-scaled
        2. 0.1 * n_empty          — small bonus for keeping board open

    Why log scaling?
        Raw merge values (8, 16, 512, 2048, ...) span 3 orders of magnitude.
        Without scaling the network chases big merges early and ignores board
        structure. Log scale compresses this so all merges matter roughly equally.

    Why empty cell bonus?
        Most steps have merge_reward=0 (no merge happened).  Without a per-step
        signal the agent gets almost no feedback and learning is very slow.
        Rewarding open board space gives a gradient every single step.
    """
    if not moved:
        return -1.0     # small penalty for illegal move (shouldn't happen with masking)

    log_merge = math.log2(merge_reward + 1)
    empty_bonus = 0.6 * game.n_empty # Should i reward this more? 
    # TODO could add reward here for max tile
    return (log_merge + empty_bonus) * 0.1 # scale down to keep rewards in a reasonable range


# = training loop

def compute_returns(rewards: list, last_value: float, gamma: float) -> list:
    """
    Reward-to-go: G_t = r_t + y*r_{t+1} + γ²*r_{t+2} + ...

    Built backwards from episode end. last_value bootstraps from the critic
    if the episode hit max_steps (otherwise 0.0 — game ended naturally).

    This is what the critic is trained to predict.
    TODO: upgrade to GAE (Generalized Advantage Estimation) for lower variance.
    """
    returns = []
    G = last_value
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def train(
    n_episodes:      int   = 2000,
    gamma:           float = 0.99,
    lr:              float = 3e-4,
    entropy_coef:    float = 0.01,
    value_coef:      float = 0.5,
    max_steps:       int   = 5000,
    checkpoint_path: str   = "mcts_checkpoint.pt",
    log_path:        str   = "mcts_log.csv",
    log_every:       int   = 50,
):
    print(f"Training on: {DEVICE}")
    print(f"Episodes: {n_episodes}  γ={gamma}  lr={lr}  entropy_coef={entropy_coef}\n")

    net       = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Resume from checkpoint if one exists
    start_episode = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"] + 1
        print(f"Resumed from checkpoint — continuing from episode {start_episode}\n")
    else:
        print("No checkpoint found — starting fresh\n")

    total_episodes = start_episode + n_episodes  # for curriculum progress

    recent_scores    = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    log_rows         = []
    t_start          = time.time()

    for episode in range(start_episode, start_episode + n_episodes):

        game = Game2048()
        states, actions, rewards, values, log_probs, masks = [], [], [], [], [], []

        for _ in range(max_steps):
            if game.is_over:
                break

            state_tensor = board_to_tensor(game.board).to(DEVICE)
            mask_tensor  = action_mask(game).to(DEVICE)

            policy, value = net(state_tensor, mask_tensor)

            # Sample action from policy distribution
            dist   = torch.distributions.Categorical(policy)
            action = dist.sample()

            # Take the action in the game
            move           = Move(action.item())
            moved, raw_rew = game.step(move)        # Move and get score delta (merge reward)
            reward         = compute_reward(moved, raw_rew, game) # Reward for this transition. 
            # reward         = compute_reward(moved, raw_rew, game, episode, total_episodes) # Reward with ciriculum learing

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value.squeeze()) # why do we have to squeeze here?
            log_probs.append(dist.log_prob(action))
            masks.append(mask_tensor)

        # Bootstrap if episode hit max_steps without finishing
        last_value = 0.0
        if not game.is_over and len(states) > 0:
            print("Episode hit max steps without ending. Bootstrapping last value from critic.")
            print("Warning this will cause issues in the training because we wont be learning how to play with higher tiles.")
            with torch.no_grad():
                last_state = board_to_tensor(game.board).to(DEVICE)
                last_mask  = action_mask(game).to(DEVICE)
                _, last_v  = net(last_state, last_mask)
                last_value = last_v.item()

        returns   = compute_returns(rewards, last_value, gamma) # Reward credit assignment to previous states
        returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        values_t  = torch.stack(values)
        log_probs_t = torch.stack(log_probs)

        # Advantage = how much better was this outcome vs what the critic expected?
        # TODO what does detach mean?
        # Detach values so critic gradient doesn't flow through the advantage
        advantages = returns_t - values_t.detach()

        # Normalize advantages ONLY — not returns.
        # Normalizing returns destroys the scale the critic needs to learn V(s).
        # Normalizing advantages just stabilizes the actor gradient.
        if len(advantages) > 1:
            # returns_t   = (returns_t   - returns_t.mean())   / (returns_t.std()   + 1e-8)
            advantages  = (advantages  - advantages.mean())  / (advantages.std()  + 1e-8)

        # Actor loss: push up probability of actions with positive advantage
        actor_loss = -(log_probs_t * advantages).mean()

        # Critic loss: make value predictions track actual returns
        critic_loss = F.mse_loss(values_t, returns_t)

        # Entropy: -E[log π(a)] from sampled log_probs
        # Prevents policy collapsing to always picking one move
        entropy = -log_probs_t.mean()

        # Combined loss
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        # ---- update network ----
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()

        recent_scores.append(game.score)
        recent_max_tiles.append(game.max_tile)

        if episode % log_every == 0:
            avg_score    = np.mean(recent_scores)
            avg_max_tile = np.mean(recent_max_tiles)
            elapsed      = time.time() - t_start
            print(
                f"Ep {episode:>5} | "
                f"AvgScore {avg_score:>8.0f} | "
                f"AvgMaxTile {avg_max_tile:>6.0f} | "
                f"Loss {loss.item():>7.4f} | "
                f"Entropy {entropy.item():>5.3f} | "
                f"Time {elapsed:>6.0f}s"
            )
            log_rows.append({
                "episode":      episode,
                "avg_score":    round(avg_score, 1),
                "avg_max_tile": round(avg_max_tile, 1),
                "loss":         round(loss.item(), 5),
                "actor_loss":   round(actor_loss.item(), 5),
                "critic_loss":  round(critic_loss.item(), 5),
                "entropy":      round(entropy.item(), 5),
                "elapsed_s":    round(elapsed, 1),
            })

    # save checkpoint
    torch.save({
        "model_state_dict":     net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode":              start_episode + n_episodes - 1,
        "config": {
            "gamma":        gamma,
            "lr":           lr,
            "entropy_coef": entropy_coef,
            "value_coef":   value_coef,
        }
    }, checkpoint_path)
    print(f"\nModel saved → {checkpoint_path}")

    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Log saved   → {log_path}")

    return net


# = evaluation

def evaluate(checkpoint_path: str = "mcts_checkpoint.pt", n_games: int = 100):
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Train first.")
        return

    net = ActorCritic().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    print(f"Evaluating {n_games} games with trained A2C agent...")
    scores, max_tiles = [], []

    with torch.no_grad():
        for i in range(n_games):
            game = Game2048(seed=i)
            while not game.is_over:
                state  = board_to_tensor(game.board).to(DEVICE)
                mask   = action_mask(game).to(DEVICE)
                policy, _ = net(state, mask)
                # At eval time: pick the highest-probability legal move (greedy)
                # TODO here look at mcts for improving the move selection.
                action = policy.argmax().item() # Greedy action for now. Need to add mcts
                game.step(Move(action))
            scores.append(game.score)
            max_tiles.append(game.max_tile)

    _print_results("A2C Greedy", n_games, scores, max_tiles)


def evaluate_mcts(
    checkpoint_path: str = "mcts_checkpoint.pt",
    n_games:         int = 100,
    n_simulations:   int = 100,
    c:               float = 1.5,
):
    """
    MCTS eval — uses tree search guided by the trained network.
 
    At each move, runs n_simulations of MCTS using:
        - Policy head as prior to bias exploration (PUCT formula)
        - Value head to evaluate leaf nodes (no random rollouts needed)
 
    Then picks the most visited action (greedy over visit counts).
 
    This is stronger than greedy eval because MCTS looks ahead through
    many simulated futures rather than just one network forward pass.
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Train first.")
        return
 
    net = ActorCritic().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    # TODO tune n simulations based on time
    mcts = MCTS(net=net, device=DEVICE, c=c, n_simulations=n_simulations)
 
    print(f"Evaluating {n_games} games with MCTS ({n_simulations} sims/move)...")
    scores, max_tiles = [], []
 
    for i in range(n_games):
        game = Game2048(seed=i)
        while not game.is_over:
            action = mcts.search(game)
            game.step(Move(action))
 
        scores.append(game.score)
        max_tiles.append(game.max_tile)
 
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_games}] score={game.score} max={game.max_tile}")
 
    _print_results(f"A2C + MCTS ({n_simulations} sims)", n_games, scores, max_tiles)


def evaluate_uniform(n_games: int = 100):
    """
    Baseline: uniform random policy over legal moves only.
    """
    import random
    scores, max_tiles = [], []
    for i in range(n_games):
        game = Game2048(seed=i)
        while not game.is_over:
            game.step(random.choice(game.available_moves()))
        scores.append(game.score)
        max_tiles.append(game.max_tile)

    _print_results("Uniform Random", n_games, scores, max_tiles)

def display_agent(checkpoint_path: str = "mcts_checkpoint.pt",
                  n_games: int = 1, speed: int = 2, type: str = "greedy"):
    import pygame

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint at {checkpoint_path}")
        return

    net = ActorCritic().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    pygame.init()
    screen = pygame.display.set_mode((400, 450))
    pygame.display.set_caption("2048 — A2C Agent")
    font_t = pygame.font.SysFont(None, 40)
    font_s = pygame.font.SysFont(None, 28)
    clk    = pygame.time.Clock()

    TILE_COLORS = {
        0: (205,193,180), 2: (238,228,218), 4: (237,224,200),
        8: (242,177,121), 16: (245,149,99),  32: (246,124,95),
        64: (246,94,59),  128: (237,207,114), 256: (237,204,97),
        512: (237,200,80), 1024: (237,197,63), 2048: (237,194,46),
    }
    mcts = MCTS(net=net, device=DEVICE, c=1.5, n_simulations=100)

    for game_i in range(n_games):
        game = Game2048()
        while not game.is_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            with torch.no_grad():
                state  = board_to_tensor(game.board).to(DEVICE)
                mask   = action_mask(game).to(DEVICE)
                policy, _ = net(state, mask)
                if type == "greedy":
                    action = policy.argmax().item() # Greedy action
                elif type == "mcts":
                    action = mcts.search(game)
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
            clk.tick(speed)

        print(f"Game {game_i+1}: score={game.score}  max_tile={game.max_tile}")

    pygame.quit()

def _print_results(label: str, n_games: int, scores: list, max_tiles: list):
    """Shared result printer for all eval modes."""
    from collections import Counter
    tile_dist = Counter(max_tiles)
    print(f"\n{'='*45}")
    print(f"  {label} — {n_games} games")
    print(f"{'='*45}")
    print(f"  Mean score      : {np.mean(scores):>10.1f}")
    print(f"  Median score    : {np.median(scores):>10.1f}")
    print(f"  Max score       : {np.max(scores):>10}")
    print(f"  Win rate (≥2048): {sum(t >= 2048 for t in max_tiles)/n_games*100:>7.1f}%")
    print(f"\n  Max tile distribution:")
    for tile in sorted(tile_dist):
        pct = tile_dist[tile] / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"    {tile:>5}: {bar:<40} {pct:.1f}%")

# ===== CLI

# TODO: Should look at gamma because we care about future rewards a lot
# but we need it to make it so we don't lose the game. 
# Do we need a negative reward for losing?

def main():
    p = argparse.ArgumentParser(description="A2C trainer for 2048")
    p.add_argument("--episodes",     type=int,   default=2000)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef",   type=float, default=0.5)
    p.add_argument("--log-every",    type=int,   default=50)
    p.add_argument("--checkpoint",   default="mcts_checkpoint.pt")
    p.add_argument("--eval",         type=int, default=-1, help="num of games to evaluate")
    p.add_argument("--type",         type=str, default="greedy")
    p.add_argument("--display",      type=int,   default=-1,
                   help="watch agent play at DISPLAY fps (e.g. --display 4)")
    args = p.parse_args()

    if args.eval >= 0:
        if args.type.lower() == "greedy":
            evaluate(args.checkpoint, args.eval)
        elif args.type.lower() == "uniform":
            evaluate_uniform(args.eval)
        elif args.type.lower() == "mcts":
            evaluate_mcts(args.checkpoint, args.eval)
    elif args.display != -1:
        display_agent(args.checkpoint, speed=args.display, type=args.type)
    else:
        train(
            n_episodes      = args.episodes,
            gamma           = args.gamma,
            lr              = args.lr,
            entropy_coef    = args.entropy_coef,
            value_coef      = args.value_coef,
            log_every       = args.log_every,
            checkpoint_path = args.checkpoint,
        )


if __name__ == "__main__":
    main()