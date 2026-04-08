"""
main.py — unified entry point for the 2048 Actor-Critic project
===============================================================

Modes
-----
    train_a2c     Train the CNN A2C agent
    train_mcts    Train the MCTS + LinearActorCritic agent (AlphaZero-style)
    uniform_mcts  Run the diagnostic uniform-policy MCTS (no network)
    evaluate      Evaluate a saved checkpoint
    simulate      Headless simulation with classical agents (random/greedy/expectimax)
    display       Watch an agent play with pygame
    gui           Interactive pygame game (human play)
    terminal      Interactive curses game (human play)

Examples
--------
    python main.py --mode train_a2c
    python main.py --mode train_mcts --episodes 5000 --n-simulations 100
    python main.py --mode evaluate --agent a2c --checkpoint a2c_checkpoint.pt
    python main.py --mode evaluate --agent mcts --eval-type mcts --eval-sims 200
    python main.py --mode display --agent mcts --checkpoint mcts_checkpoint.pt --speed 4
    python main.py --mode uniform_mcts --sims 200 --games 20 --baseline
    python main.py --mode simulate --sim-agent expectimax --depth 3 --n 500
    python main.py --mode gui
    python main.py --mode terminal
"""

import argparse

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2048 Actor-Critic — unified entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["train_a2c", "train_mcts", "uniform_mcts",
                 "evaluate", "display", "simulate", "gui", "terminal"],
    )

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",  default=None,   help="path to .pt checkpoint file")
    p.add_argument("--episodes",    type=int,   default=2000)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--eval-games",  type=int,   default=100)
    p.add_argument("--speed",       type=int,   default=4,  help="display fps")
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--n-games",     type=int,   default=1,  help="games to display")

    # ── A2C training ──────────────────────────────────────────────────────────
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef",   type=float, default=0.5)
    p.add_argument("--log-every",    type=int,   default=50)

    # ── MCTS training ─────────────────────────────────────────────────────────
    p.add_argument("--n-simulations",    type=int,   default=50)
    p.add_argument("--collect-every",    type=int,   default=10)
    p.add_argument("--n-grad-steps",     type=int,   default=3)
    p.add_argument("--minibatch",        type=int,   default=512)
    p.add_argument("--terminal-penalty", type=float, default=0.0)
    p.add_argument("--debug-every",      type=int,   default=0)
    p.add_argument("--dir",              type=str,   default="")

    # ── eval / display ────────────────────────────────────────────────────────
    p.add_argument("--agent",        default="a2c",
                   choices=["a2c", "mcts", "uniform"])
    p.add_argument("--eval-type",    default="greedy",
                   choices=["greedy", "mcts", "random"],
                   help="move selection strategy during evaluation")
    p.add_argument("--eval-c",       type=float, default=25.0,  help="MCTS c for eval")
    p.add_argument("--eval-sims",    type=int,   default=100,   help="MCTS sims for eval")
    p.add_argument("--display-type", default="greedy",
                   choices=["greedy", "mcts"],
                   help="move selection strategy during display")

    # ── uniform MCTS ──────────────────────────────────────────────────────────
    p.add_argument("--games",    type=int,   default=20)
    p.add_argument("--sims",     type=int,   default=200)
    p.add_argument("--rollout",  type=int,   default=10)
    p.add_argument("--c",        type=float, default=160.0)
    p.add_argument("--reuse",    action="store_true", help="reuse tree between moves")
    p.add_argument("--baseline", action="store_true", help="also run random baseline")

    # ── simulate ──────────────────────────────────────────────────────────────
    p.add_argument("--sim-agent", default="random",
                   choices=["random", "greedy", "expectimax"])
    p.add_argument("--n",      type=int,  default=100)
    p.add_argument("--depth",  type=int,  default=3)
    p.add_argument("--csv",    type=str,  default=None)
    p.add_argument("--verbose",action="store_true")

    return p


def _load_net(net_class, checkpoint_path: str):
    """Load a network from a checkpoint. Returns (net, checkpoint_dict)."""
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Train first.")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    net = net_class().to(DEVICE)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net


def main():
    args = _build_parser().parse_args()

    # ── train_a2c ─────────────────────────────────────────────────────────────
    if args.mode == "train_a2c":
        from train_a2c import train
        ckpt = args.checkpoint or "a2c_checkpoint.pt"
        train(
            n_episodes      = args.episodes,
            gamma           = args.gamma,
            lr              = args.lr,
            entropy_coef    = args.entropy_coef,
            value_coef      = args.value_coef,
            log_every       = args.log_every,
            checkpoint_path = ckpt,
        )

    # ── train_mcts ────────────────────────────────────────────────────────────
    elif args.mode == "train_mcts":
        from train_mcts import train_mcts
        ckpt = args.checkpoint or "mcts_checkpoint.pt"
        train_mcts(
            n_episodes        = args.episodes,
            gamma             = args.gamma,
            lr                = args.lr,
            value_coef        = args.value_coef,
            n_simulations     = args.n_simulations,
            collect_every     = args.collect_every,
            n_grad_steps      = args.n_grad_steps,
            minibatch_size    = args.minibatch,
            terminal_penalty  = args.terminal_penalty,
            save_dir          = args.dir,
            log_every         = args.log_every,
            debug_every       = args.debug_every,
            checkpoint_path   = ckpt,
        )

    # ── evaluate ──────────────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        from evaluate import evaluate_agent, evaluate_checkpoint
        from utils import action_mask

        if args.agent == "a2c":
            from networks import CNNActorCritic
            ckpt = args.checkpoint or "a2c_checkpoint.pt"
            evaluate_checkpoint(
                checkpoint_path = ckpt,
                net_class       = CNNActorCritic,
                device          = DEVICE,
                label           = f"A2C ({args.eval_type})",
                n_games         = args.eval_games,
            )

        elif args.agent == "mcts":
            from networks import LinearActorCritic
            from MCTS import MCTS
            ckpt = args.checkpoint or "mcts_checkpoint.pt"
            net  = _load_net(LinearActorCritic, ckpt)

            if args.eval_type == "mcts":
                mcts_eval = MCTS(net=net, device=DEVICE, c=args.eval_c,
                                 n_simulations=args.eval_sims)
                def select_action(game):
                    return mcts_eval.search(game)
            elif args.eval_type == "random":
                import random as _random
                def select_action(game):
                    return int(_random.choice(game.available_moves()))
            else:
                def select_action(game):
                    with torch.no_grad():
                        state  = net.board_to_tensor(game.board).to(DEVICE)
                        mask   = action_mask(game).to(DEVICE)
                        policy, _ = net(state, mask)
                        return int(policy.argmax().item())

            evaluate_agent(select_action,
                           label=f"MCTS-AC ({args.eval_type})",
                           n_games=args.eval_games)

        elif args.agent == "uniform":
            from mcts_uniform import play_games, play_random
            if args.sims == 0:
                play_random(n_games=args.eval_games, seed=args.seed or 42)
            else:
                play_games(n_games=args.eval_games, n_simulations=args.sims,
                           rollout_depth=args.rollout, c=args.c, gamma=args.gamma,
                           reuse_tree=args.reuse, seed=args.seed or 42)

    # ── display ───────────────────────────────────────────────────────────────
    elif args.mode == "display":
        from display import display_agent
        from utils import action_mask

        if args.agent == "uniform":
            from MCTS import UniformMCTS
            mcts_u = UniformMCTS(n_simulations=args.sims, rollout_depth=args.rollout,
                                 c=args.c, gamma=args.gamma)
            display_agent(
                select_action = mcts_u.best_action,
                caption       = "2048 — Uniform MCTS",
                n_games       = args.n_games,
                speed         = args.speed,
            )

        elif args.agent == "a2c":
            from networks import CNNActorCritic
            ckpt = args.checkpoint or "a2c_checkpoint.pt"
            net  = _load_net(CNNActorCritic, ckpt)

            def select_action(game):
                with torch.no_grad():
                    state  = net.board_to_tensor(game.board).to(DEVICE)
                    mask   = action_mask(game).to(DEVICE)
                    policy, _ = net(state, mask)
                    return int(policy.argmax().item())

            display_agent(select_action, caption="2048 — A2C Agent",
                          n_games=args.n_games, speed=args.speed)

        elif args.agent == "mcts":
            from networks import LinearActorCritic
            from MCTS import MCTS
            ckpt = args.checkpoint or "mcts_checkpoint.pt"
            net  = _load_net(LinearActorCritic, ckpt)

            if args.display_type == "mcts":
                mcts_d = MCTS(net=net, device=DEVICE, c=args.c,
                              n_simulations=args.n_simulations, empty_threshold=10)
                def select_action(game):
                    return mcts_d.search(game)
            else:
                def select_action(game):
                    with torch.no_grad():
                        state  = net.board_to_tensor(game.board).to(DEVICE)
                        mask   = action_mask(game).to(DEVICE)
                        policy, _ = net(state, mask)
                        return int(policy.argmax().item())

            display_agent(select_action, caption="2048 — MCTS Agent",
                          n_games=args.n_games, speed=args.speed)

    # ── uniform_mcts ──────────────────────────────────────────────────────────
    elif args.mode == "uniform_mcts":
        from mcts_uniform import play_games, play_random
        if args.sims == 0:
            play_random(n_games=args.games, seed=args.seed or 42)
        else:
            print(f"Uniform MCTS | sims={args.sims} rollout={args.rollout} c={args.c} reuse={args.reuse}")
            play_games(
                n_games       = args.games,
                n_simulations = args.sims,
                rollout_depth = args.rollout,
                c             = args.c,
                gamma         = args.gamma,
                reuse_tree    = args.reuse,
                seed          = args.seed or 42,
            )
            if args.baseline:
                print("\nRunning random baseline for comparison...")
                play_random(n_games=args.games, seed=args.seed or 42)

    # ── simulate ──────────────────────────────────────────────────────────────
    elif args.mode == "simulate":
        from simulate import run_simulation, save_csv
        print(f"Running {args.n} games with agent='{args.sim_agent}' ...")
        stats = run_simulation(
            n_games    = args.n,
            agent_name = args.sim_agent,
            depth      = args.depth,
            seed       = args.seed,
            verbose    = args.verbose,
        )
        stats.print_summary()
        if args.csv:
            save_csv(stats, args.csv)

    # ── gui ───────────────────────────────────────────────────────────────────
    elif args.mode == "gui":
        from gui import run_gui
        run_gui(seed=args.seed)

    # ── terminal ──────────────────────────────────────────────────────────────
    elif args.mode == "terminal":
        from terminal_ui import run_terminal
        run_terminal(seed=args.seed)


if __name__ == "__main__":
    main()
