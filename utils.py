import datetime
from pathlib import Path
from typing import NamedTuple, List, Literal, Optional, Union
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from pgx import State
from pgx._src.visualizer import global_config, Visualizer
import svgwrite
import svgwrite.container
import rich.progress as rp

def save_svg_animation(
    states: State,
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
    frame_duration_seconds: Optional[float] = None,
) -> None:
    assert not states.env_id[0].startswith(
        "minatar"
    ), "MinAtar does not support svg animation."
    assert str(filename).endswith(".svg")
    v = Visualizer(color_theme=color_theme, scale=scale)

    if frame_duration_seconds is None:
        frame_duration_seconds = global_config.frame_duration_seconds

    frame_groups = []
    dwg = None
    for i in range(states.current_player.shape[0]):
        state = jax.tree_util.tree_map(lambda x: x[i], states)
        dwg = v.get_dwg(states=state)
        assert (
            len(
                [
                    e
                    for e in dwg.elements
                    if type(e) is svgwrite.container.Group
                ]
            )
            == 1
        ), "Drawing must contain only one group"
        group: svgwrite.container.Group = dwg.elements[-1]
        group["id"] = f"_fr{i:x}"  # hex frame number
        group["class"] = "frame"
        frame_groups.append(group)
        if state.terminated.all():
            break

    assert dwg is not None
    del dwg.elements[-1]
    total_seconds = frame_duration_seconds * len(frame_groups)

    style = f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite;}}"
    style += f"@keyframes _k{{0%,{100/len(frame_groups)}%{{visibility:visible}}{100/len(frame_groups) * 1.000001}%,100%{{visibility:hidden}}}}"

    for i, group in enumerate(frame_groups):
        dwg.add(group)
        style += (
            f"#{group['id']}{{animation-delay:{i * frame_duration_seconds}s}}"
        )
    dwg.defs.add(svgwrite.container.Style(content=style))
    dwg.saveas(filename)


def elo_from_results(results, base=1000, max_delta=1000):
    return (
          base
        - np.clip(
              400
            * np.log(2 / np.clip(results+1, 1e-100, 2-1e-100) - 1)
            / np.log(10),
            -max_delta,
            +max_delta
        )
    )


def move_pgn(
    board: List[List[str]],
    move: int,
    i: int,
    brackets: bool=False,
    gardner: bool=False,
    pgc: ModuleType | None=None
) -> str:
    assert(pgc is not None)
    moves_from, moves_plane = (move // 49, move % 49) if gardner else (move // 73, move % 73)
    moves_to = pgc.TO_MAP[moves_from, moves_plane]
    moves_underpromotion = moves_plane // 3 if moves_plane < 9 else -1
    size = 5 if gardner else 8
    def square2cart(square):
        row, col = square % size, square // size
        if i % 2 == 1:
            row = size - 1 - row
        return row, col
    def square2str(square):
        row, col = square2cart(square)
        return chr(ord('a') + col) + str(row + 1)
    from_row, from_col = square2cart(moves_from)
    to_row, to_col = square2cart(moves_to)
    piece = board[from_row][from_col]
    promotion = piece == 'P' and to_row == (size-1 if i % 2 == 0 else 0)
    new_piece = piece
    if promotion:
        new_piece = "QRBN"[1+moves_underpromotion]
    if piece == 'K' and abs(from_col - to_col) == 2: # Castling
        assert from_row == to_row
        board[to_row][(from_col + to_col) // 2] = 'R'
        if to_col > from_col:
            rook = 7
        else:
            rook = 0
        assert board[to_row][rook] == 'R'
        board[to_row][rook] = ' '
    board[to_row][to_col] = new_piece
    board[from_row][from_col] = ' '
    return (
          ("" if i % 2 == 1 else str(i // 2 + 1) + ". ")
        + ("(" if brackets else "")
        + ("" if piece in " P" else piece)
        + square2str(moves_from)
        + square2str(moves_to)
        + ("" if not promotion else "=" + "QRBN"[1+moves_underpromotion])
        + (")" if brackets else "")
    )


def to_pgn(
    moves,
    round: str='?',
    player0: str='?',
    player1: str='?',
    result: str | int='?',
    gardner: bool=False,
    pgc: ModuleType | None=None
) -> str:
    first_player = moves[0][3]
    white, black = player0, player1
    if first_player == 1:
        white, black = black, white
        if isinstance(result, int):
            result = -result
    if isinstance(result, int):
        result = "1/2-1/2" if result == 0 else "1-0" if result == 1 else "0-1"
    board = [
        ['R','N','B','Q','K'],
        ['P','P','P','P','P'],
        [' ',' ',' ',' ',' '],
        ['P','P','P','P','P'],
        ['R','N','B','Q','K']
    ]
    if not gardner:
        board = [
            ['R','N','B','Q','K', 'B', 'N', 'R'],
            ['P','P','P','P','P', 'P', 'P', 'P'],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            ['P','P','P','P','P', 'P', 'P', 'P'],
            ['R','N','B','Q','K', 'B', 'N', 'R']
        ]
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if gardner: # Hack to have gardner on 8x8 board
        fen = "8/8/8/rnbqk3/ppppp3/8/PPPPP3/RNBQK3 w - - 0 1"
    else:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    game = (
       f'[Event "MCTX Training"]\n'
       f'[Site "aku.ren"]\n'
       f'[Date "{date}"]\n'
       f'[Round "{round}"]\n'
       f'[White "{white}"]\n'
       f'[Black "{black}"]\n'
       f'[Result "{result}"]\n'
       f'[FEN "{fen}"]\n'
   ) + ' '.join([
       move_pgn(
           board,
           m,
           i,
           1-l,
           gardner=gardner,
           pgc=pgc
        )
       for i, (m, t, l, _) in enumerate(moves)
       if t == 0
    ])
    return game


class Sample(NamedTuple):
    board: jnp.ndarray
    obs: jnp.ndarray
    # board_or_obs: jnp.ndarray
    lam: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray
