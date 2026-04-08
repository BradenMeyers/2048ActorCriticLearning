"""
AC MCTS (MCTS with Actor-Critic) for 2048
==
Trains a neural network to play 2048 using the MCTS with Actor-Critic algorithm.

Files produced
--------------
    mcts_checkpoint.pt   - saved model weights
    mcts_log.csv         - per-episode training stats

Architecture
------------
    State:  one-hot encoding → (256,) flat tensor
    Trunk:  Linear(256→512) → ReLU → Linear(512→512) → ReLU
    Heads:  Actor → 4 action logits | Critic → 1 scalar value
"""

import csv
import os
import time
from collections import deque
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game import Game2048, Move
import time
from utils import STATE_DIM, compute_returns, circ_reward, sqrt_reward, log_reward, action_mask, N_ACTIONS, ACTION_NAMES, RunningNormalizer
from networks import LinearActorCritic as ActorCritic
from MCTS import MCTS

# ===== constants

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# = training loop

def train_mcts(
    n_episodes:       int   = 2000,
    gamma:            float = 0.99,
    lr:               float = 3e-4,
    value_coef:       float = 0.5,
    max_steps:        int   = 5000,
    n_simulations:    int   = 50,
    collect_every:    int   = 10,   # episodes to collect before each training phase
    n_grad_steps:     int   = 3,    # gradient updates per collection round
    minibatch_size:   int   = 512,  # steps sampled per gradient update
    terminal_penalty: float = 0.0,  # reward added at game-over in training and MCTS
    save_dir:         str   = "",   # directory to write checkpoint and log into
    checkpoint_path:  str   = "mcts_checkpoint.pt",
    log_path:         str   = "mcts_log.csv",
    log_every:        int   = 50,
    debug_every:      int   = 0,   # print MCTS search stats every N moves (0 = off)
):
    """
    AlphaZero-style training loop with replay buffer and a frozen behavior network.

    Each round:
      1. COLLECT: play collect_every episodes using target_net (frozen) inside
         MCTS. Store (state, mask, pi_mcts, return) for every step.
      2. TRAIN: sample random minibatches from the buffer and do n_grad_steps
         gradient updates on net (the live training network).
      3. SYNC: copy net → target_net so the next round collects with the
         freshly trained weights.

    Why a separate target_net?
        If we train net while also using it for MCTS, the data distribution
        shifts under every gradient step. Keeping target_net frozen for a full
        collection round stabilises the training targets.

    Actor loss: cross-entropy(pi_mcts, pi_net) = -sum(pi_mcts * log(pi_net))
    Critic loss: MSE(V(s), G_t)
    """
    # Resolve output paths — prefix with save_dir if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, os.path.basename(checkpoint_path))
        log_path        = os.path.join(save_dir, os.path.basename(log_path))

    print(f"Training MCTS on: {DEVICE}")
    print(f"Episodes: {n_episodes}  gamma={gamma}  lr={lr}  sims={n_simulations}")
    print(f"collect_every={collect_every}  n_grad_steps={n_grad_steps}  minibatch={minibatch_size}")
    print(f"checkpoint -> {checkpoint_path}\n")

    net        = ActorCritic().to(DEVICE)
    optimizer  = optim.Adam(net.parameters(), lr=lr)
    ret_normalizer = RunningNormalizer(momentum=0.99)

    start_episode = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"] + 1
        print(f"Resumed from checkpoint -- continuing from episode {start_episode}\n")
    else:
        print("No checkpoint found -- starting fresh\n")

    # Frozen behavior network used by MCTS during data collection.
    # Only synced from net at the end of each round.
    target_net = copy.deepcopy(net)
    target_net.eval()
    mcts = MCTS(net=target_net, device=DEVICE, c=45.0, n_simulations=n_simulations,
                gamma=gamma, terminal_penalty=terminal_penalty)

    recent_scores    = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    log_rows         = []
    t_start          = time.time()

    end_episode = start_episode + n_episodes
    for round_start in range(start_episode, end_episode, collect_every):
        round_end = min(round_start + collect_every, end_episode)

        # ---- 1. COLLECT --------------------------------------------------------
        # Play collect_every episodes and accumulate (state, mask, pi, return).
        buffer = []

        for episode in range(round_start, round_end):
            game = Game2048()
            mcts.reset_tree()  # fresh tree per episode; reused across moves within it
            episode_data = []  # (state_tensor, mask_tensor, pi_target, reward)

            move_count = 0
            for _ in range(max_steps):
                if game.is_over:
                    break

                state_tensor = net.board_to_tensor(game.board).to(DEVICE)
                mask_tensor  = action_mask(game).to(DEVICE)

                # MCTS on the frozen target_net — no gradient tracking needed
                should_debug = debug_every > 0 and move_count % debug_every == 0
                pi     = mcts.get_policy(game, eta=1.0, add_noise=True, debug=should_debug)
                target = torch.tensor(pi, dtype=torch.float32)  # keep on CPU until batched

                action = int(torch.multinomial(target, 1).item())
                move_count += 1
                moved, merge_reward = game.step(Move(action))
                reward = sqrt_reward(moved, merge_reward, game)
                

                episode_data.append((state_tensor.cpu(), mask_tensor.cpu(), target, reward))

            if not episode_data:
                continue

            # Add terminal penalty to the last step if the game ended naturally
            if game.is_over and terminal_penalty != 0.0:
                s, m, pi, r = episode_data[-1]
                episode_data[-1] = (s, m, pi, r + terminal_penalty)

            # Bootstrap critic value if the episode hit max_steps
            last_value = 0.0
            if not game.is_over:
                with torch.no_grad():
                    last_state = net.board_to_tensor(game.board).to(DEVICE)
                    last_mask  = action_mask(game).to(DEVICE)
                    _, last_v  = net(last_state, last_mask)
                    last_value = last_v.item()

            rewards = [r for *_, r in episode_data]
            returns = compute_returns(rewards, last_value, gamma)

            for i, (state, mask, pi, _) in enumerate(episode_data):
                buffer.append((state, mask, pi, returns[i]))

            recent_scores.append(game.score)
            recent_max_tiles.append(game.max_tile)

        if not buffer:
            continue

        # ---- 2. TRAIN ----------------------------------------------------------
        # Multiple minibatch gradient steps on the collected data.
        actor_loss_sum = critic_loss_sum = 0.0

        for _ in range(n_grad_steps):
            batch = random.sample(buffer, min(minibatch_size, len(buffer)))

            states_t  = torch.stack([s for s, m, pi, r in batch]).to(DEVICE)   # (B, STATE_DIM)
            masks_t   = torch.stack([m for s, m, pi, r in batch]).to(DEVICE)   # (B, 4)
            pis_t     = torch.stack([pi for s, m, pi, r in batch]).to(DEVICE)  # (B, 4)
            returns_t = torch.tensor(
                [r for s, m, pi, r in batch], dtype=torch.float32, device=DEVICE
            )                                                                    # (B,)

            policy_out, value_out = net(states_t, masks_t)   # (B,4), (B,1)

            log_pi      = torch.log(policy_out + 1e-8)        # (B, 4)
            actor_loss  = -(pis_t * log_pi).sum(dim=-1).mean()
            # Normalize returns for critic training only
            # Keep raw returns for MCTS backprop
            ret_normalizer.update(returns_t)
            critic_loss = F.mse_loss(value_out.squeeze(-1), ret_normalizer.normalize(returns_t))

            loss = actor_loss + value_coef * critic_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            actor_loss_sum  += actor_loss.item()
            critic_loss_sum += critic_loss.item()

        # ---- 3. SYNC -----------------------------------------------------------
        # Copy trained weights into the frozen behavior network for next round.
        target_net.load_state_dict(net.state_dict())

        # ---- LOGGING -----------------------------------------------------------
        last_episode = round_end - 1
        if last_episode % log_every < collect_every or last_episode + 1 >= end_episode:
            avg_score    = np.mean(recent_scores)
            avg_max_tile = np.mean(recent_max_tiles)
            elapsed      = time.time() - t_start
            avg_actor    = actor_loss_sum  / n_grad_steps
            avg_critic   = critic_loss_sum / n_grad_steps
            print(
                f"Ep {last_episode:>5} | "
                f"AvgScore {avg_score:>8.0f} | "
                f"AvgMaxTile {avg_max_tile:>6.0f} | "
                f"ActorLoss {avg_actor:>7.4f} | "
                f"CriticLoss {avg_critic:>7.4f} | "
                f"Time {elapsed:>6.0f}s"
            )
            log_rows.append({
                "episode":      last_episode,
                "avg_score":    round(avg_score, 1),
                "avg_max_tile": round(avg_max_tile, 1),
                "actor_loss":   round(avg_actor, 5),
                "critic_loss":  round(avg_critic, 5),
                "elapsed_s":    round(elapsed, 1),
            })

        # ---- SAVE (every round) ------------------------------------------------
        torch.save({
            "model_state_dict":     net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode":              last_episode,
            "config": {
                "gamma":          gamma,
                "lr":             lr,
                "value_coef":     value_coef,
                "n_simulations":  n_simulations,
                "collect_every":  collect_every,
                "n_grad_steps":   n_grad_steps,
                "minibatch_size": minibatch_size,
            }
        }, checkpoint_path)

        if log_rows:
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
                writer.writeheader()
                writer.writerows(log_rows)

    print(f"\nModel saved -> {checkpoint_path}")
    print(f"Log saved   -> {log_path}")
    return net