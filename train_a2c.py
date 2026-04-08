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
    python train_a2c.py --display 2            # watch agent play at 2 fps

Files produced
--------------
    a2c_checkpoint.pt   - saved model weights
    a2c_log.csv         - per-episode training stats

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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game import Game2048, Move
from utils import circ_reward, sqrt_reward, log_reward, compute_returns, action_mask
from networks import CNNActorCritic as ActorCritic

# ===== constants

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# = training loop

def train(
    n_episodes:      int   = 2000,
    gamma:           float = 0.99,
    lr:              float = 3e-4,
    entropy_coef:    float = 0.01,
    value_coef:      float = 0.5,
    max_steps:       int   = 5000,
    checkpoint_path: str   = "a2c_checkpoint.pt",
    log_path:        str   = "a2c_log.csv",
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

            state_tensor = net.board_to_tensor(game.board).to(DEVICE)
            mask_tensor  = action_mask(game).to(DEVICE)

            policy, value = net(state_tensor, mask_tensor)

            # Sample action from policy distribution
            dist   = torch.distributions.Categorical(policy)
            action = dist.sample()

            # Take the action in the game
            move           = Move(action.item())
            moved, raw_rew = game.step(move)        # Move and get score delta (merge reward)
            reward         = log_reward(moved, raw_rew, game, empty_bonus=0.2) * 0.1 # Reward for this transition. 

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
                last_state = net.board_to_tensor(game.board).to(DEVICE)
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
        entropy = -log_probs_t.mean() # TODO check this
        # TODO HERE
        # entropy = dist.entropy().mean()  # exact H(π) = -Σ π(a) log π(a)

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

def evaluate(checkpoint_path: str = "a2c_checkpoint.pt", n_games: int = 100):
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
                state  = net.board_to_tensor(game.board).to(DEVICE)
                mask   = action_mask(game).to(DEVICE)
                policy, _ = net(state, mask)
                # At eval time: pick the highest-probability legal move (greedy)
                # TODO here look at mcts for improving the move selection.
                action = policy.argmax().item() # Greedy action for now. Need to add mcts
                game.step(Move(action))
            scores.append(game.score)
            max_tiles.append(game.max_tile)

    tile_dist = Counter(max_tiles)
    print(f"\n{'='*45}")
    print(f"  A2C Agent — {n_games} games")
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

    tile_dist = Counter(max_tiles)
    print(f"\n{'='*45}")
    print(f"  Uniform Random — {n_games} games")
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


def display_agent(checkpoint_path: str = "a2c_checkpoint.pt",
                  n_games: int = 1, speed: int = 2):
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

    for game_i in range(n_games):
        game = Game2048()
        while not game.is_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            with torch.no_grad():
                state  = net.board_to_tensor(game.board).to(DEVICE)
                mask   = action_mask(game).to(DEVICE)
                policy, _ = net(state, mask)
                action = policy.argmax().item() # Greedy action
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
    p.add_argument("--checkpoint",   default="a2c_checkpoint.pt")
    p.add_argument("--eval",         action="store_true")
    p.add_argument("--eval-uniform", action="store_true")
    p.add_argument("--eval-games",   type=int,   default=100)
    p.add_argument("--display",      type=int,   default=-1,
                   help="watch agent play at DISPLAY fps (e.g. --display 4)")
    args = p.parse_args()

    if args.eval:
        evaluate(args.checkpoint, args.eval_games)
    elif args.eval_uniform:
        evaluate_uniform(args.eval_games)
    elif args.display != -1:
        display_agent(args.checkpoint, speed=args.display)
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