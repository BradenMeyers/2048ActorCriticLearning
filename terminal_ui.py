"""
Terminal renderer for 2048
==========================
Optional visual layer using only the standard library (curses).
Run standalone:  python terminal_ui.py
"""

import curses
import sys
from game import Game2048, Move

# Map arrow / wasd keys to Move enum
_KEY_MAP = {
    curses.KEY_UP:    Move.UP,
    curses.KEY_DOWN:  Move.DOWN,
    curses.KEY_LEFT:  Move.LEFT,
    curses.KEY_RIGHT: Move.RIGHT,
    ord('w'): Move.UP,
    ord('s'): Move.DOWN,
    ord('a'): Move.LEFT,
    ord('d'): Move.RIGHT,
}

# Colors per tile value (curses pair index : fg, bg)
_TILE_COLORS = {
    0:    (curses.COLOR_BLACK,   curses.COLOR_BLACK),
    2:    (curses.COLOR_BLACK,   250),   # light grey
    4:    (curses.COLOR_BLACK,   223),   # warm beige
    8:    (curses.COLOR_WHITE,   208),   # orange
    16:   (curses.COLOR_WHITE,   202),   # dark orange
    32:   (curses.COLOR_WHITE,   196),   # red
    64:   (curses.COLOR_WHITE,   124),   # dark red
    128:  (curses.COLOR_BLACK,   226),   # yellow
    256:  (curses.COLOR_BLACK,   220),
    512:  (curses.COLOR_BLACK,   214),
    1024: (curses.COLOR_WHITE,   34),    # green
    2048: (curses.COLOR_WHITE,   226),   # gold
}
_DEFAULT_COLOR = (curses.COLOR_WHITE, 57)  # purple for big tiles

_CELL_W = 8
_CELL_H = 3


def _color_pair_for(value: int, pairs: dict) -> int:
    key = value if value in _TILE_COLORS else "default"
    return curses.color_pair(pairs[key])


def _draw_board(stdscr, game: Game2048, pairs: dict, msg: str = ""):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    board_h = _CELL_H * 4 + 1
    board_w = _CELL_W * 4 + 1
    origin_y = max(0, (h - board_h - 4) // 2)
    origin_x = max(0, (w - board_w) // 2)

    # Header
    header = "  2 0 4 8  "
    stdscr.addstr(origin_y, origin_x, header, curses.A_BOLD)
    origin_y += 1
    score_line = f"Score: {game.score:<10}  Best Tile: {game.max_tile}"
    stdscr.addstr(origin_y, origin_x, score_line)
    origin_y += 2

    # Grid
    for row in range(4):
        for line in range(_CELL_H):
            y = origin_y + row * _CELL_H + line
            for col in range(4):
                x = origin_x + col * _CELL_W
                val = int(game.board[row, col])
                attr = _color_pair_for(val, pairs)
                if line == 0 or line == _CELL_H - 1:
                    cell_str = "+" + "-" * (_CELL_W - 2) + "+"[:-1] if col < 3 else "+" + "-" * (_CELL_W - 2) + "+"
                    stdscr.addstr(y, x, cell_str, attr)
                else:
                    text = str(val) if val else ""
                    cell_str = f"|{text:^{_CELL_W - 2}}"
                    if col == 3:
                        cell_str += "|"
                    stdscr.addstr(y, x, cell_str, attr)

    footer_y = origin_y + board_h + 1
    controls = "Arrow keys / WASD to move  |  R to restart  |  Q to quit"
    stdscr.addstr(footer_y, origin_x, controls)
    if msg:
        stdscr.addstr(footer_y + 1, origin_x, msg, curses.A_BOLD)

    stdscr.refresh()


def _setup_colors() -> dict:
    curses.start_color()
    curses.use_default_colors()
    pairs = {}
    idx = 1
    for key, (fg, bg) in _TILE_COLORS.items():
        curses.init_pair(idx, fg, bg)
        pairs[key] = idx
        idx += 1
    curses.init_pair(idx, *_DEFAULT_COLOR)
    pairs["default"] = idx
    return pairs


def run_terminal(seed=None):
    def _main(stdscr):
        curses.curs_set(0)
        pairs = _setup_colors()
        game  = Game2048(seed=seed)
        msg   = ""

        while True:
            _draw_board(stdscr, game, pairs, msg)
            key = stdscr.getch()

            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                game = Game2048(seed=seed)
                msg  = ""
                continue

            move = _KEY_MAP.get(key)
            if move is None:
                continue

            if game.is_over:
                msg = "Game over! Press R to restart."
                continue

            moved, _ = game.step(move)
            if not moved:
                msg = "Invalid move."
            elif game.won() and "won" not in msg:
                msg = "You hit 2048! Keep going or Q to quit."
            elif game.is_over:
                msg = f"Game over! Final score: {game.score}"
            else:
                msg = ""

    curses.wrapper(_main)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_terminal(seed)
