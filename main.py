"""
main.py — unified entry point for the 2048 Actor-Critic project
===============================================================

Modes
-----
    train_a2c     Train the CNN A2C agent
    train_mcts    Train the MCTS + LinearActorCritic agent (AlphaZero-style)
    evaluate      Evaluate an agent (a2c / mcts / uniform / baseline)
    display       Watch an agent play with pygame
    gui           Interactive pygame game (human play)
    terminal      Interactive curses game (human play)

Examples
--------
    python main.py --mode train_a2c
    python main.py --mode train_mcts --episodes 5000 --n-simulations 100
    python main.py --mode evaluate --agent a2c --checkpoint a2c_checkpoint.pt
    python main.py --mode evaluate --agent mcts --eval-type mcts --eval-sims 200
    python main.py --mode evaluate --agent uniform --sims 200 --games 20
    python main.py --mode evaluate --agent baseline --games 20
    python main.py --mode display --agent mcts --checkpoint mcts_checkpoint.pt --speed 4
    python main.py --mode gui
    python main.py --mode terminal
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import argparse

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_A2C_CHECKPOINT = "pretrained/a2c/a2c_checkpoint.pt"
DEFAULT_MCTS_CHECKPOINT = "pretrained/mcts/mcts_checkpoint.pt"

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2048 Actor-Critic — unified entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["train_a2c", "train_mcts", "evaluate", "display", "gui", "terminal"],
    )

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",  default=None,   help="path to .pt checkpoint file")
    p.add_argument("--episodes",    type=int,   default=2000)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--games",       type=int,   default=100)
    p.add_argument("--speed",       type=int,   default=0,  help="display fps")
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--n-games",     type=int,   default=1,  help="games to display")
    p.add_argument("--c",            type=float, default=80.0, help="MCTS exploration constant")

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
                   choices=["a2c", "mcts", "uniform", "baseline"])
    p.add_argument("--eval-type",    default="greedy",
                   choices=["greedy", "mcts", "random"],
                   help="move selection strategy during evaluation")
    p.add_argument("--eval-sims",    type=int,   default=100,   help="MCTS sims for eval")
    p.add_argument("--display-type", default="greedy",
                   choices=["greedy", "mcts"],
                   help="move selection strategy during display")

    # ── uniform MCTS ──────────────────────────────────────────────────────────
    p.add_argument("--sims",         type=int,   default=200)
    p.add_argument("--rollout",      type=int,   default=10)
    p.add_argument("--clear-tree",   action="store_true", help="clear tree between moves")

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
        ckpt = args.checkpoint or DEFAULT_A2C_CHECKPOINT
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
        ckpt = args.checkpoint or DEFAULT_MCTS_CHECKPOINT
        # TODO add C here
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
        from runners.evaluate import evaluate_agent, evaluate_checkpoint
        from runners.utils import action_mask

        if args.agent == "a2c":
            from runners.networks import LinearActorCritic
            ckpt = args.checkpoint or DEFAULT_A2C_CHECKPOINT
            evaluate_checkpoint(
                checkpoint_path = ckpt,
                net_class       = LinearActorCritic,
                device          = DEVICE,
                label           = f"A2C ({args.eval_type})",
                n_games         = args.games,
            )

        elif args.agent == "mcts":
            from runners.networks import LinearActorCritic
            from runners.MCTS import MCTS
            ckpt = args.checkpoint or DEFAULT_MCTS_CHECKPOINT
            net  = _load_net(LinearActorCritic, ckpt)

            if args.eval_type == "mcts":
                mcts_eval = MCTS(net=net, device=DEVICE, c=args.c,
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
                           n_games=args.games)

        elif args.agent == "uniform":
            from mcts_uniform import play_games
            play_games(n_games=args.games, n_simulations=args.sims,
                       rollout_depth=args.rollout, c=args.c, gamma=args.gamma,
                       clear_tree=args.clear_tree, seed=args.seed or 42)

        elif args.agent == "baseline":
            from mcts_uniform import play_random
            play_random(n_games=args.games, seed=args.seed or 42)

    # ── display ───────────────────────────────────────────────────────────────
    elif args.mode == "display":
        from runners.display import display_agent
        from runners.utils import action_mask

        if args.agent == "uniform":
            from runners.MCTS import UniformMCTS
            mcts_u = UniformMCTS(n_simulations=args.sims, rollout_depth=args.rollout,
                                 c=args.c, gamma=args.gamma)
            display_agent(
                select_action = mcts_u.best_action,
                caption       = "2048 — Uniform MCTS",
                n_games       = args.n_games,
                speed         = args.speed,
            )

        elif args.agent == "a2c":
            from runners.networks import LinearActorCritic
            ckpt = args.checkpoint or DEFAULT_A2C_CHECKPOINT
            net  = _load_net(LinearActorCritic, ckpt)

            def select_action(game):
                with torch.no_grad():
                    state  = net.board_to_tensor(game.board).to(DEVICE)
                    mask   = action_mask(game).to(DEVICE)
                    policy, _ = net(state, mask)
                    return int(policy.argmax().item())

            display_agent(select_action, caption="2048 — A2C Agent",
                          n_games=args.n_games, speed=args.speed)

        elif args.agent == "mcts":
            from runners.networks import LinearActorCritic
            from runners.MCTS import MCTS
            ckpt = args.checkpoint or DEFAULT_MCTS_CHECKPOINT
            net  = _load_net(LinearActorCritic, ckpt)

            if args.display_type == "mcts":
                mcts_d = MCTS(net=net, device=DEVICE, c=args.c,
                              n_simulations=args.n_simulations, empty_threshold=10)
                print(f"Displaying MCTS Agent | sims={args.n_simulations} rollout={args.rollout} c={args.c}")
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

    # ── gui ───────────────────────────────────────────────────────────────────
    elif args.mode == "gui":
        from runners.gui import run_gui
        run_gui(seed=args.seed)

    # ── terminal ──────────────────────────────────────────────────────────────
    elif args.mode == "terminal":
        from runners.terminal_ui import run_terminal
        run_terminal(seed=args.seed)


if __name__ == "__main__":
    main()
