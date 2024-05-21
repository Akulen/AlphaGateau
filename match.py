import argparse
import os
import json

import numpy as np
import jax
import flax.linen as nn
import pgx
import pgx.chess as pgc
import rich.progress as rp
from sklearn.linear_model import LinearRegression
from rich.pretty import pprint

from models import load_model
import mcts
from utils import to_pgn
from ranking import compute_elo


devices = jax.local_devices()
num_devices = len(devices)

games_dir = f"./tournaments/ranking matches"
os.makedirs(games_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        prog='MatchMaker',
        description='Run a match between two models',
    )
    parser.add_argument('model1')
    parser.add_argument('iteration1', type=int)
    parser.add_argument('model2')
    parser.add_argument('iteration2', type=int)
    parser.add_argument('-n', '--n-games', type=int, default=64)
    parser.add_argument('-m', '--max-plies', type=int, default=256)
    parser.add_argument('-s', '--n-sims', type=int, default=128)
    parser.add_argument('-r', '--seed', type=int, default=42)

    args = parser.parse_args()

    assert(os.path.isfile(f"./models/{args.model1}/{args.iteration1:06}.ckpt"))
    assert(os.path.isfile(f"./models/{args.model2}/{args.iteration2:06}.ckpt"))

    env_id = "chess"
    env = pgx.make(env_id)
    model1, model_params1 = load_model(
        env,
        f"./models/{args.model1}/{args.iteration1:06}.ckpt",
        f"{args.model1}-{args.iteration1:03}"
    )
    model2, model_params2 = load_model(
        env,
        f"./models/{args.model2}/{args.iteration2:06}.ckpt",
        f"{args.model2}-{args.iteration2:03}"
    )

    with open("rankings.json", "r") as file:
        data = json.load(file)
    with open("rankings.backup.json", "w") as file:
        json.dump(data, file)

    R, games = mcts.full_pit(
        env,
        model1,
        jax.device_put_replicated(model_params1, devices),
        model2,
        jax.device_put_replicated(model_params2, devices),
        jax.random.PRNGKey(args.seed),
        n_games=args.n_games,
        max_plies=args.max_plies,
        n_sim=args.n_sims,
        n_devices=num_devices
    )
    wins, draws, losses = map(
        lambda r: ((R == r).sum()).item(),
        [1, 0, -1]
    )
    model1_name = f'{args.model1}/{args.iteration1:03}'
    model2_name = f'{args.model2}/{args.iteration2:03}'
    if model1_name not in data['results']:
        data['results'][model1_name] = {}
    if model2_name not in data['results'][model1_name]:
        data['results'][model1_name][model2_name] = [0, 0, 0]
    if model2_name not in data['results']:
        data['results'][model2_name] = {}
    if model1_name not in data['results'][model2_name]:
        data['results'][model2_name][model1_name] = [0, 0, 0]
    data['results'][model1_name][model2_name][0] += wins
    data['results'][model1_name][model2_name][1] += draws
    data['results'][model1_name][model2_name][2] += losses
    data['results'][model2_name][model1_name][0] += losses
    data['results'][model2_name][model1_name][1] += draws
    data['results'][model2_name][model1_name][2] += wins

    print(f'{model1_name}[{data["elo"][model1_name]:4}]'
          f' vs '
          f'[{data["elo"][model2_name]:4}]{model2_name}: '
          f'{wins:2}/{draws:2}/{losses:2}')

    data['elo'] = compute_elo(data['results'])

    count = [128] * 3
    with open(os.path.join(
        games_dir,
        f"{model1.id} vs {model2.id}.pgn"
    ), "w") as f:
        for r, g in zip(R, games):
            r_i = int(np.round(r))
            if count[r_i+1] > 0:
                count[r_i+1] -= 1
                print(to_pgn(
                    g,
                    round=f"Ranking Match {model1_name} vs {model2_name}",
                    player0=model1.id,
                    player1=model2.id,
                    result=r_i,
                    pgc=pgc
                ), file=f)

    pprint(data['elo'])
    with open("rankings.json", "w") as file:
        json.dump(data, file)

if __name__ == "__main__":
    main()
