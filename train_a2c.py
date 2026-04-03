"""
A2C (Advantage Actor-Critic) for 2048
==
Trains a neural network to play 2048 using the Advantage Actor-Critic algorithm.

Usage
-----
    pip install torch numpy
    python train_a2c.py                        # train with defaults
    python train_a2c.py --episodes 5000        # more training
    python train_a2c.py --eval                 # evaluate a saved checkpoint

Files produced
--------------
    a2c_checkpoint.pt   – saved model weights
    a2c_log.csv         – per-episode training stats
"""

import argparse
import csv
import math
import os
import time
from collections import deque
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game import Game2048, Move


# ===== constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State: 4x4 board, each cell one-hot over 16 tile levels (0, 2, 4, ..., 32768)
# Level 0 = empty, level k = tile value 2^k.  Max realistic tile is ~2^15.
N_TILE_LEVELS = 16          # levels 0..15
STATE_DIM     = 4 * 4 * N_TILE_LEVELS   # 256
N_ACTIONS     = 4           # UP DOWN LEFT RIGHT

# = state encoder

def board_to_tensor(board: np.ndarray) -> torch.Tensor:
    """
    Convert a 4x4 numpy board to a (256,) one-hot float tensor.

    Each cell becomes a one-hot vector of length 16:
        empty  → index 0
        tile 2 → index 1
        tile 4 → index 2
        ...
        tile 2^15 → index 15
    """
    state = np.zeros((4, 4, N_TILE_LEVELS), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            idx = int(math.log2(val)) if val > 0 else 0
            if idx >= N_TILE_LEVELS:
                print(f"Warning: tile value {val} exceeds max level {N_TILE_LEVELS-1}. Clamping.")
                idx = min(idx, N_TILE_LEVELS - 1)   # safety clamp
            state[r, c, idx] = 1.0
    return torch.tensor(state.flatten(), dtype=torch.float32)


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
    Shared-trunk network with two heads:
        - Actor head  → policy π(a|s), probability over 4 moves
        - Critic head → value V(s), expected future return

    Architecture:
        Input (256) → Linear(512) → ReLU → Linear(512) → ReLU
                                                ↓
                              ┌─────────────────┤
                              ↓                 ↓
                         Actor head        Critic head
                         Linear(4)         Linear(1)
                         (logits)          (scalar)
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 512):
        super().__init__()

        # Shared trunk — learns board representations useful to both heads
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head — outputs raw logits (NOT probabilities yet)
        self.actor_head = nn.Linear(hidden, N_ACTIONS)

        # Critic head — outputs a single scalar value estimate
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters
        ----------
        state : (batch, 256) or (256,) float tensor
        mask  : (batch, 4) or (4,) bool tensor — True = legal move
                If provided, illegal moves get -inf logits before softmax.

        Returns
        -------
        policy : (batch, 4) action probability distribution
        value  : (batch, 1) state value estimate
        """
        features = self.trunk(state)

        logits = self.actor_head(features)

        # Mask illegal moves — set their logits to -inf so softmax → 0
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        policy = F.softmax(logits, dim=-1)
        value  = self.critic_head(features)

        return policy, value


# = reward

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
    empty_bonus = 0.1 * game.n_empty
    return log_merge + empty_bonus


# = training loop

def compute_returns(rewards: list, last_value: float, gamma: float) -> list:
    """
    Compute discounted returns G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

    We work backwards from the end of the episode.
    last_value is the critic's estimate of V(s_T) — the bootstrapped value
    of the state we stopped collecting at (0.0 if episode ended naturally).

    This is the target that the critic will be trained to predict.
    """
    returns = []
    G = last_value
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train(
    n_episodes:      int   = 2000,
    gamma:           float = 0.99,
    lr:              float = 1e-3,
    entropy_coef:    float = 0.01,
    value_coef:      float = 0.5,
    max_steps:       int   = 5000,
    checkpoint_path: str   = "a2c_checkpoint.pt",
    log_path:        str   = "a2c_log.csv",
    log_every:       int   = 50,
):
    """
    Main A2C training loop.

    Key hyperparameters
    -------------------
    gamma        : discount factor — how much future rewards matter (0.99 = a lot)
    lr           : learning rate for Adam optimizer
    entropy_coef : weight on entropy bonus — prevents policy from collapsing to
                   always picking one move.  Higher = more exploration.
    value_coef   : weight on critic loss relative to actor loss
    max_steps    : safety cap on steps per episode
    """
    print(f"Training on: {DEVICE}")
    print(f"Episodes: {n_episodes}  γ={gamma}  lr={lr}  entropy_coef={entropy_coef}\n")

    net       = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Rolling window for tracking progress
    recent_scores    = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)

    log_rows = []
    t_start  = time.time()

    for episode in range(1, n_episodes + 1):

        game = Game2048()

        # ---- collect one full episode of experience ----
        states, actions, rewards, values, log_probs, masks = [], [], [], [], [], []

        for _ in range(max_steps):
            if game.is_over:
                break

            state_tensor = board_to_tensor(game.board).to(DEVICE)
            mask_tensor  = action_mask(game).to(DEVICE)

            # WHAT IS MY INITIAL policy. Is it just uniform?

            # Forward pass — get policy and value for current state
            policy, value = net(state_tensor, mask_tensor)

            # Sample action from policy distribution
            dist   = torch.distributions.Categorical(policy)
            action = dist.sample()

            # Take the action in the game
            move           = Move(action.item())
            moved, raw_rew = game.step(move)
            reward         = compute_reward(moved, raw_rew, game)

            # Store everything we'll need for the update
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value.squeeze())
            log_probs.append(dist.log_prob(action))
            masks.append(mask_tensor)

        # ---- compute returns and advantages ----

        # Bootstrap value from last state (0 if game ended, V(s_T) if we hit max_steps)
        last_value = 0.0
        if not game.is_over and len(states) > 0:
            with torch.no_grad():
                last_mask = action_mask(game).to(DEVICE)
                last_state = board_to_tensor(game.board).to(DEVICE)
                _, last_v  = net(last_state, last_mask)
                last_value = last_v.item()

        returns   = compute_returns(rewards, last_value, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        
        # normalize returns to stabilize critic learning
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        values_t  = torch.stack(values)
        log_probs_t = torch.stack(log_probs)

        # Advantage = how much better was this outcome vs what the critic expected?
        # Detach values so critic gradient doesn't flow through the advantage
        advantages = returns_t - values_t.detach()

        # Normalize advantages — helps training stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- compute losses ----

        # Actor loss: negative log-prob weighted by advantage
        # If advantage > 0 (better than expected), increase probability of this action
        # If advantage < 0 (worse than expected), decrease probability
        actor_loss = -(log_probs_t * advantages).mean()

        # Critic loss: MSE between predicted value and actual return
        # Train the critic to predict returns accurately
        critic_loss = F.mse_loss(values_t, returns_t)

        # Entropy bonus: encourages exploration by penalizing overconfident policies
        # Without this the policy often collapses to always picking one move
        policy_stack = torch.stack([
            net(s, m)[0] for s, m in zip(states, masks)
        ])
        entropy = -(policy_stack * torch.log(policy_stack + 1e-8)).sum(dim=-1).mean()

        # Combined loss
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        # ---- update network ----
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()

        # ---- logging ----
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

    # ---- save checkpoint ----
    torch.save({
        "model_state_dict":     net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode":              n_episodes,
        "config": {
            "gamma":        gamma,
            "lr":           lr,
            "entropy_coef": entropy_coef,
            "value_coef":   value_coef,
        }
    }, checkpoint_path)
    print(f"\nModel saved → {checkpoint_path}")

    # ---- save log ----
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Log saved   → {log_path}")

    return net


# = evaluation

def evaluate(checkpoint_path: str = "a2c_checkpoint.pt", n_games: int = 100):
    """
    Load a saved checkpoint and run n_games to measure performance.
    Compares against the random agent as a baseline.
    """
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
                action = policy.argmax().item()
                game.step(Move(action))

            scores.append(game.score)
            max_tiles.append(game.max_tile)

    from collections import Counter
    tile_dist = Counter(max_tiles)

    print(f"\n{'='*45}")
    print(f"  A2C Agent — {n_games} games")
    print(f"{'='*45}")
    print(f"  Mean score   : {np.mean(scores):>10.1f}")
    print(f"  Median score : {np.median(scores):>10.1f}")
    print(f"  Max score    : {np.max(scores):>10}")
    print(f"  Win rate (≥2048): {sum(t >= 2048 for t in max_tiles)/n_games*100:.1f}%")
    print(f"\n  Max tile distribution:")
    for tile in sorted(tile_dist):
        pct = tile_dist[tile] / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"    {tile:>5}: {bar:<40} {pct:.1f}%")


def display_agent(net: ActorCritic, n_games: int = 5):
    """
    Display the agent playing n_games with pygame.
    """
    import pygame
    from game import Game2048

    pygame.init()
    size = (400, 400)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()

    for i in range(n_games):
        game = Game2048(seed=i)
        while not game.is_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state  = board_to_tensor(game.board).to(DEVICE)
            mask   = action_mask(game).to(DEVICE)
            policy, _ = net(state, mask)
            action = policy.argmax().item()
            game.step(Move(action))

            # Render the game board
            screen.fill((187, 173, 160))  # background color
            for r in range(4):
                for c in range(4):
                    val = game.board[r, c]
                    color = (205, 193, 180) if val == 0 else (238, 228, 218)
                    rect = pygame.Rect(c*100+5, r*100+5, 90, 90)
                    pygame.draw.rect(screen, color, rect)
                    if val > 0:
                        font = pygame.font.SysFont(None, 40)
                        text = font.render(str(val), True, (119, 110, 101))
                        text_rect = text.get_rect(center=rect.center)
                        screen.blit(text, text_rect)

            pygame.display.flip()
            clock.tick(10)  # slow down for visibility

    pygame.quit()

# ===== CLI

def main():
    p = argparse.ArgumentParser(description="A2C trainer for 2048")
    p.add_argument("--episodes",     type=int,   default=2000)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef",   type=float, default=0.5)
    p.add_argument("--log-every",    type=int,   default=50)
    p.add_argument("--checkpoint",   default="a2c_checkpoint.pt")
    p.add_argument("--display",      action="store_true")
    p.add_argument("--eval",         action="store_true",
                   help="evaluate a saved checkpoint instead of training")
    p.add_argument("--eval-games",   type=int, default=100)
    args = p.parse_args()

    if args.eval:
        evaluate(args.checkpoint, args.eval_games)
    elif args.display:
        net = ActorCritic().to(DEVICE)
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        net.load_state_dict(checkpoint["model_state_dict"])
        display_agent(net, n_games=1)
    else:
        train(
            n_episodes   = args.episodes,
            gamma        = args.gamma,
            lr           = args.lr,
            entropy_coef = args.entropy_coef,
            value_coef   = args.value_coef,
            log_every    = args.log_every,
            checkpoint_path = args.checkpoint,
        )


if __name__ == "__main__":
    main()