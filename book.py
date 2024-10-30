from typing import Dict, List, NamedTuple

from itertools import zip_longest
import argparse
import os
import pickle
import rich.progress as rp
import sys

from utils_progress import ProgressEMA

class Style():
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
# ]]]]

parser = argparse.ArgumentParser(
    prog='OpeningBook',
    description='Generate opening book from a .pgn database',
)
parser.add_argument('database', type=str)
parser.add_argument('-o', '--opening', type=str)
parser.add_argument('-p', '--player', type=str)
parser.add_argument('-b', '--black', action=argparse.BooleanOptionalAction)
parser.add_argument('-r', '--refresh', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

# games = []
headers = ""
outcome = None

class Leaf(NamedTuple):
    outcome: int
    headers: str
    moves: List
    cursor: int

    @property
    def res(self):
        tmp = [0, 0, 0]
        tmp[self.outcome] = 1
        return tmp

    @property
    def cnt(self):
        return 1

    def __str__(self) -> str:
        r = Style.BLUE + self.headers + Style.GREEN + ' '.join(
            map(' '.join, zip_longest(
                map(lambda i: str(i+1) + '.', range((len(self.moves)+1)//2)),
                self.moves[::2],
                self.moves[1::2],
            fillvalue=''))
        ) + Style.RESET
        return r

class Node(NamedTuple):
    res: List[int]
    moves: Dict
    cnt: int=0

tree = Node([0]*3, {})

def parse_game(
    node: Node,
    i_move: int,
    moves: List[str],
    outcome: int,
    headers: str
):
    if i_move == len(moves):
        return 0, 0
    cnt, cnt_uniq = 1, 0
    move = moves[i_move]
    if move not in node.moves:
        node.moves[move] = Leaf(outcome, headers, moves, i_move)
        cnt_uniq = len(moves) - i_move
        return cnt_uniq, cnt_uniq
    if isinstance(node.moves[move], Leaf): # Expand:
        leaf = node.moves[move]
        node.moves[move] = Node([0]*3, {}, cnt=1)
        node.moves[move].res[leaf.outcome] = 1
        parse_game(
            node.moves[move],
            leaf.cursor+1,
            leaf.moves,
            leaf.outcome,
            leaf.headers
        )

    a, b = parse_game(
        node.moves[move],
        i_move+1,
        moves,
        outcome,
        headers
    )
    cnt += a
    cnt_uniq += b
    node.moves[move] = node.moves[move]._replace(cnt=node.moves[move].cnt + 1)
    node.moves[move].res[outcome] += 1
    return cnt, cnt_uniq

def parse_file(file):
    global headers, tree, outcome, games
    cnt, cnt_uniq = 0, 0
    valid = args.player is None
    with open(file, 'r') as f:
        for line in f:
            if line[0] != '1':
                headers += line
                if 'Result' in line:
                    if '1-0' in line:
                        outcome = 0
                    elif '1/2-1/2' in line:
                        outcome = 1
                    else:
                        outcome = 2
                elif args.player is not None:
                    if (args.black and 'Black' in line) \
                            or (not args.black and 'White' in line):
                        valid = args.player in line
                continue
            # games.append(line.strip())
            if valid:
                moves = [
                    move
                    for i, move in enumerate(line.strip().split())
                    if i % 3 != 0
                ]
                assert outcome is not None
                a, b = parse_game(tree, 0, moves, outcome, headers)
                cnt += a
                cnt_uniq += b
            headers = ""
            valid = args.player is None
    return cnt, cnt_uniq

sys.setrecursionlimit(10000)
if not os.path.isfile("book_cache.pkl"):
    cache = {}
else:
    with open("book_cache.pkl", "rb") as file:
        cache = pickle.load(file)

config = (args.database, args.player, args.black)
if args.refresh or config not in cache:
    print("Loading Database...")
    while len(cache) > (10 if config in cache else 9):
        print(len(cache))
        cache.pop(next(iter(cache)))
    with ProgressEMA(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[logs]}"),
        speed_estimate_period=2
    ) as progress:
        if os.path.isdir(args.database):
            files = os.listdir(args.database)
            task = progress.add_task(
                "[cyan]Processing pgns files...",
                total=len(files),
                logs='...',
            )
            cnt, cnt_uniq = 0, 0
            for file in files:
                if os.path.isfile(os.path.join(args.database, file)):
                    a, b = parse_file(os.path.join(args.database, file))
                    cnt += a
                    cnt_uniq += b
                progress.update(
                    task,
                    advance=1,
                    logs=f'{cnt} positions processed, {cnt_uniq} unique positions found',
                )
        elif os.path.isfile(args.database):
            a, b = parse_file(args.database)
            print(f"Parsed {a} positions, {b} unique positions found")
        else:
            folder = '/'.join(args.database.split('/')[:-1])
            prefix = args.database.split('/')[-1]
            if os.path.isdir(folder):
                cnt, cnt_uniq = 0, 0
                files = []
                task = progress.add_task(
                    "[red]Filtering files...",
                    total=len(os.listdir(folder)),
                    logs='...',
                )
                for file in os.listdir(folder):
                    if os.path.isfile(os.path.join(folder, file)) \
                            and file.startswith(prefix):
                        files.append(file)
                    progress.update(
                        task,
                        advance=1,
                        logs=f'{len(files)} pgns found',
                    )
                task = progress.add_task(
                    "[cyan]Processing pgns files...",
                    total=len(files),
                    logs='...',
                )
                for file in files:
                    a, b = parse_file(os.path.join(folder, file))
                    cnt += a
                    cnt_uniq += b
                    progress.update(
                        task,
                        advance=1,
                        logs=f'{cnt} positions processed, {cnt_uniq} unique positions found',
                    )
    cache[config] = tree
    with open("book_cache.pkl", "wb") as file:
        pickle.dump(cache, file, protocol=pickle.HIGHEST_PROTOCOL)
else:
    tree = cache[config]


node = tree
moves = [] if args.opening is None else args.opening.split()
for move in moves:
    try:
        node = node.moves[move]
    except KeyError:
        print(f"Move '{move}' invalid or not found")
        sys.exit(1)

def format_res(res):
    t = sum(res)
    return f"{res[0]/t:7.2%} {res[1]/t:7.2%} {res[2]/t:7.2%}"

if isinstance(node, Leaf):
    print(node)
else:
    cnt = 0
    tres = [0] * 3
    moves = list(node.moves.keys())
    moves.sort(key=lambda x: node.moves[x].cnt, reverse=True)
    print(f"{Style.GREEN}Moves:    {Style.RED}cnt {Style.BLUE}   Win    Draw    Loss{Style.RESET}")
    print("-" * 37)
    for move in moves:
        print(f"{Style.GREEN}{move:>5}: {Style.RED}{node.moves[move].cnt:6} {Style.BLUE}{format_res(node.moves[move].res)}{Style.RESET}")
        cnt += node.moves[move].cnt
        res = node.moves[move].res
        tres[0] += res[0]
        tres[1] += res[1]
        tres[2] += res[2]
    print("-" * 37)
    print(f"{Style.GREEN}Total: {Style.RED}{cnt:6} {Style.BLUE}{format_res(tres)}{Style.RESET}")
