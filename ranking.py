import os
import json
import pickle
from typing import Type

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


devices = jax.local_devices()
num_devices = len(devices)

games_dir = f"./tournaments/ranking matches"
os.makedirs(games_dir, exist_ok=True)

def match_two_hot(i1: int, i2: int, n: int) -> np.ndarray:
    x = np.zeros(n)
    x[i1] = 1
    x[i2] = -1
    return x

def score_from_results(results):
    return (results[0] - results[2]) / (sum(results) + 1)

# Data should be a symmetric 2d dictionary
def compute_elo(data):
    players = list(data.keys())
    player_id = {
        players[i]: i
        for i in range(len(players))
    }

    matches = [
        (player1, player2)
        for player1 in players
        for player2 in data[player1].keys()
        if player1 < player2
    ]

    # pprint(outcomes)
    # for model1 in models.keys():
    #     for model2 in models.keys():
    #         print(f"{outcomes[model1].get(model2, 0): 6.3f}", end=" ")
    #     print()

    # model_keys = list(models.keys())
    # matches = [
    #     (i1, i2)
    #     for i1, model1 in enumerate(model_keys)
    #     for i2, model2 in enumerate(model_keys)
    #     if str(model1) < str(model2)
    # ]
    X = np.array([
        match_two_hot(player_id[player1], player_id[player2], len(players))
        for player1, player2 in matches
    ] + [
        [1] * len(players)
    ])
    y = np.concatenate([
        -(400 / np.log(10)) * np.log(1 / np.clip(np.array([
            (score_from_results(data[player1][player2]) + 1) / 2
            for player1, player2 in matches
        ]), 1e-10, 1-1e-10) - 1),
        [1000 * len(players)] # We set the average elo to 1000
    ])
    w = np.concatenate([
        [sum(data[player1][player2]) for player1, player2 in matches],
        [1]
    ])
    reg = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=w)
    # print(model_keys)
    elo = (reg.coef_+0.5).astype(int)
    return dict(sorted({
        player: int(elo[i])
        for i, player in enumerate(players)
    }.items()))

def evaluate_model(model_name, it, data, models, models_params, env, n_matches=5):
    # We first assume the model is average
    full_model_name = f'{model_name}/{it:03}'
    if full_model_name not in data['elo']:
        data['elo'][full_model_name] = 1000

    used = {full_model_name}
    rng_key = jax.random.PRNGKey(42)
    opponents = []
    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[logs]}"),
        speed_estimate_period=1000
    ) as progress:
        task = progress.add_task(
            "[green]Estimating Elo",
            total=n_matches,
            logs='...'
        )
        for _ in range(n_matches):
            progress.update(
                task,
                logs=f"Current Estimate: {data['elo'][full_model_name]:4}, Looking for opponent...",
            )
            # We look for the model that hasn't been chosen yet with the closest
            # elo
            best_candidate, distance = None, np.inf
            for candidate in data['elo'].keys():
                if candidate in used:
                    continue
                cand_dist = abs(
                    data['elo'][candidate] - data['elo'][full_model_name]
                )
                if cand_dist < distance:
                    best_candidate = candidate
                    distance = cand_dist
            assert best_candidate is not None
            used.add(best_candidate)
            progress.update(
                task,
                logs=f"Current Estimate: {data['elo'][full_model_name]:4}, Playing against [{data['elo'][best_candidate]:4}]{best_candidate}",
            )
            opponent = best_candidate
            opponents.append(opponent)
            if opponent not in models:
                models[opponent], models_params[opponent] = load_model(
                    env,
                    f"./models/{best_candidate.split('/')[0]}/000{best_candidate.split('/')[1]}.ckpt",
                    f"{best_candidate.split('/')[0]}-{best_candidate.split('/')[1]}"
                )

            rng_key, subkey = jax.random.split(rng_key)
            R, games = mcts.full_pit(
                env,
                models[full_model_name],
                jax.device_put_replicated(models_params[full_model_name], devices),
                models[opponent],
                jax.device_put_replicated(models_params[opponent], devices),
                subkey,
                n_games=64,
                max_plies=256,
                n_devices=num_devices
            )
            wins, draws, losses = map(
                lambda r: ((R == r).sum()).item(),
                [1, 0, -1]
            )
            if full_model_name not in data['results']:
                data['results'][full_model_name] = {}
            if opponent not in data['results'][full_model_name]:
                data['results'][full_model_name][opponent] = [0, 0, 0]
            if opponent not in data['results']:
                data['results'][opponent] = {}
            if full_model_name not in data['results'][opponent]:
                data['results'][opponent][full_model_name] = [0, 0, 0]
            data['results'][full_model_name][opponent][0] += wins
            data['results'][full_model_name][opponent][1] += draws
            data['results'][full_model_name][opponent][2] += losses
            data['results'][opponent][full_model_name][0] += losses
            data['results'][opponent][full_model_name][1] += draws
            data['results'][opponent][full_model_name][2] += wins

            print(f'{full_model_name}[{data["elo"][full_model_name]:4}] vs [{data["elo"][opponent]:4}]{opponent}: {wins:2}/{draws:2}/{losses:2}')

            data['elo'] = compute_elo(data['results'])

            count = [128] * 3
            with open(os.path.join(
                games_dir,
                f"{models[full_model_name].id} vs {models[opponent].id}.pgn"
            ), "a") as f:
                for r, g in zip(R, games):
                    r_i = int(np.round(r))
                    if count[r_i+1] > 0:
                        count[r_i+1] -= 1
                        print(to_pgn(
                            g,
                            round=f"Ranking Match {full_model_name} vs {opponent}",
                            player0=models[full_model_name].id,
                            player1=models[opponent].id,
                            result=r_i,
                            pgc=pgc
                        ), file=f)
            elos = '|'.join([f'{data["elo"][opp]:4}' for opp in opponents])
            progress.update(
                task,
                advance=1,
                logs=f"Final Elo for {full_model_name}: {data['elo'][full_model_name]:4} [{elos}]",
            )

    return data, models, models_params



def main():
    env_id = "chess"
    env = pgx.make(env_id)

    with open("rankings.json", "r") as file:
        data = json.load(file)
    with open("rankings.backup.json", "w") as file:
        json.dump(data, file)

    models = {}
    models_params = {}

    model_name = "chess_2024-05-20:14h19"
    # model_name = "chess_2024-03-25:18h42"
    for it in range(1, 34, 2):
        models[f'{model_name}/{it:03}'], models_params[f'{model_name}/{it:03}'] = load_model(
            env,
            f"./models/{model_name}/{it:06}.ckpt",
            f'{model_name}-{it:03}'
        )

        data, models, models_params = evaluate_model(
            model_name, it, data, models, models_params, env
        )

    data['elo'] = compute_elo(data['results'])
    pprint(data['elo'])
    with open("rankings.json", "w") as file:
        json.dump(data, file)

if __name__ == "__main__":
    main()
