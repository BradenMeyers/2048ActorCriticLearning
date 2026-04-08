"""
A2C (Advantage Actor-Critic) for 2048
==
Trains a neural network to play 2048 using the Advantage Actor-Critic algorithm.

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

import csv
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from runners.game import Game2048, Move
from runners.utils import circ_reward, sqrt_reward, log_reward, compute_returns, action_mask
from runners.networks import LinearActorCritic as ActorCritic

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