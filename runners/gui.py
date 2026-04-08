"""
Pygame GUI for 2048
===================
Entirely optional — the rest of the project works without it.
Install:  pip install pygame
Run:      python gui.py
"""

try:
    import pygame
except ImportError:
    raise SystemExit("pygame is not installed. Run:  pip install pygame")

import sys
from runners.game import Game2048, Move

# ----------------------------------------------------------------- palette

BG_COLOR      = (18,  18,  18)
GRID_COLOR    = (30,  30,  30)
EMPTY_COLOR   = (40,  40,  40)
TEXT_DARK     = (30,  30,  30)
TEXT_LIGHT    = (245, 245, 235)

TILE_COLORS = {
    0:    (40,  40,  40),
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
SUPER_TILE_COLOR = (60, 58, 50)   # for tiles > 2048

FONT_TILE  = None
FONT_SCORE = None
FONT_MSG   = None

CELL    = 110
PADDING = 14
MARGIN  = 20
GRID_SZ = CELL * 4 + PADDING * 5
HEADER  = 120
WIN_W   = GRID_SZ + MARGIN * 2
WIN_H   = GRID_SZ + HEADER + MARGIN * 2

KEY_MAP = {
    pygame.K_UP:    Move.UP,
    pygame.K_DOWN:  Move.DOWN,
    pygame.K_LEFT:  Move.LEFT,
    pygame.K_RIGHT: Move.RIGHT,
    pygame.K_w:     Move.UP,
    pygame.K_s:     Move.DOWN,
    pygame.K_a:     Move.LEFT,
    pygame.K_d:     Move.RIGHT,
}


def _tile_color(value: int):
    return TILE_COLORS.get(value, SUPER_TILE_COLOR)


def _text_color(value: int):
    return TEXT_DARK if value in (0, 2, 4) else TEXT_LIGHT


def _draw_rounded_rect(surface, color, rect, radius=10):
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def _draw_board(surface, game: Game2048):
    surface.fill(BG_COLOR)

    # --- header
    title = FONT_SCORE.render("2048", True, TEXT_LIGHT)
    surface.blit(title, (MARGIN, MARGIN))

    score_text = FONT_SCORE.render(f"Score: {game.score}", True, (187, 173, 160))
    surface.blit(score_text, (MARGIN, MARGIN + 44))

    # --- grid background
    grid_rect = pygame.Rect(MARGIN, HEADER + MARGIN, GRID_SZ, GRID_SZ)
    _draw_rounded_rect(surface, GRID_COLOR, grid_rect, radius=8)

    # --- cells
    for row in range(4):
        for col in range(4):
            x = MARGIN + PADDING + col * (CELL + PADDING)
            y = HEADER + MARGIN + PADDING + row * (CELL + PADDING)
            val = int(game.board[row, col])

            cell_rect = pygame.Rect(x, y, CELL, CELL)
            _draw_rounded_rect(surface, _tile_color(val), cell_rect, radius=6)

            if val:
                text = str(val)
                fs   = 36 if val < 1000 else 28 if val < 10000 else 22
                font = pygame.font.SysFont("Arial Rounded MT Bold", fs, bold=True)
                surf = font.render(text, True, _text_color(val))
                tx   = x + (CELL - surf.get_width())  // 2
                ty   = y + (CELL - surf.get_height()) // 2
                surface.blit(surf, (tx, ty))


def _draw_overlay(surface, message: str, sub: str = ""):
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((18, 18, 18, 180))
    surface.blit(overlay, (0, 0))

    msg_surf = FONT_MSG.render(message, True, TEXT_LIGHT)
    surface.blit(msg_surf, ((WIN_W - msg_surf.get_width()) // 2,
                             WIN_H // 2 - 40))
    if sub:
        sub_surf = FONT_SCORE.render(sub, True, (187, 173, 160))
        surface.blit(sub_surf, ((WIN_W - sub_surf.get_width()) // 2,
                                 WIN_H // 2 + 20))


def run_gui(seed=None):
    global FONT_TILE, FONT_SCORE, FONT_MSG

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("2048")
    clock  = pygame.time.Clock()

    FONT_TILE  = pygame.font.SysFont("Arial Rounded MT Bold", 40, bold=True)
    FONT_SCORE = pygame.font.SysFont("Arial", 28, bold=True)
    FONT_MSG   = pygame.font.SysFont("Arial", 48, bold=True)

    game     = Game2048(seed=seed)
    won_once = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game     = Game2048(seed=seed)
                    won_once = False
                    continue

                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

                move = KEY_MAP.get(event.key)
                if move is not None and not game.is_over:
                    game.step(move)
                    if game.won() and not won_once:
                        won_once = True

        _draw_board(screen, game)

        if game.is_over:
            _draw_overlay(screen, "Game Over",
                          f"Score: {game.score}  |  R to restart")
        elif won_once and game.won():
            pass   # let them keep playing; won_once prevents the popup loop

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_gui(seed)
