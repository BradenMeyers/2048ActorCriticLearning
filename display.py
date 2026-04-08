"""
display.py — shared pygame rendering for 2048 agents
=====================================================
Extracted from train_a2c.py, train_mcts.py, and mcts_uniform.py.

Usage
-----
    from display import display_agent

    # Any callable that maps Game2048 → int action index
    display_agent(select_action=my_agent, caption="My Agent", n_games=3, speed=4)
"""

from __future__ import annotations

from typing import Callable

from game import Game2048, Move


TILE_COLORS: dict[int, tuple] = {
    0:    (205, 193, 180),
    2:    (238, 228, 218),
    4:    (237, 224, 200),
    8:    (242, 177, 121),
    16:   (245, 149,  99),
    32:   (246, 124,  95),
    64:   (246,  94,  59),
    128:  (237, 207, 114),
    256:  (237, 204,  97),
    512:  (237, 200,  80),
    1024: (237, 197,  63),
    2048: (237, 194,  46),
}


def draw_board(screen, game: Game2048, font_large, font_small) -> None:
    """Render the current board state onto a pygame surface."""
    import pygame

    screen.fill((187, 173, 160))
    for r in range(4):
        for c in range(4):
            val   = int(game.board[r, c])
            color = TILE_COLORS.get(val, (60, 58, 50))
            rect  = pygame.Rect(c * 97 + 8, r * 97 + 58, 90, 90)
            pygame.draw.rect(screen, color, rect, border_radius=6)
            if val > 0:
                txt = font_large.render(str(val), True, (119, 110, 101))
                screen.blit(txt, txt.get_rect(center=rect.center))

    score_txt = font_small.render(
        f"Score: {game.score}   Max: {game.max_tile}", True, (255, 255, 255)
    )
    screen.blit(score_txt, (10, 15))


def display_agent(
    select_action: Callable[[Game2048], int],
    caption: str = "2048 — Agent",
    n_games: int = 1,
    speed:   int = 4,
) -> None:
    """
    Watch an agent play 2048 in a pygame window.

    Parameters
    ----------
    select_action : callable
        Any function that takes a Game2048 and returns an action index (0–3).
        The caller is responsible for building this — display.py has no knowledge
        of neural networks or MCTS.
    caption : str
        Window title.
    n_games : int
        Number of games to play before closing.
    speed : int
        Frames per second cap. Set to 0 for uncapped.
    """
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((400, 450))
    pygame.display.set_caption(caption)
    font_large = pygame.font.SysFont(None, 40)
    font_small = pygame.font.SysFont(None, 28)
    clk = pygame.time.Clock()

    for game_i in range(n_games):
        game = Game2048()

        while not game.is_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = select_action(game)
            game.step(Move(action))

            draw_board(screen, game, font_large, font_small)
            pygame.display.flip()
            if speed > 0:
                clk.tick(speed)

        print(f"Game {game_i + 1}: score={game.score}  max_tile={game.max_tile}")

    pygame.quit()
