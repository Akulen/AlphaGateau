import os
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

from models import ModelManager, EdgeNet, EdgeNet2
import mcts
from utils import to_pgn


devices = jax.local_devices()
num_devices = len(devices)


def match_two_hot(i1: int, i2: int, n: int) -> np.ndarray:
    x = np.zeros(n)
    x[i1] = 1
    x[i2] = -1
    return x

def load_model(
    env: pgx.Env,
    file_name: str,
    model_name: str,
    net: Type[nn.Module]=EdgeNet
):
    with open(file_name, "rb") as f:
        dic = pickle.load(f)
        model = ModelManager(
            id=model_name,
            model=net(
                n_actions=env.num_actions,
                inner_size=dic['config']['inner_size'],
                n_gnn_layers=dic['config'].get('n_gnn_layers', 5),
                dot_v2=dic['config']['dotv2'],
                use_embedding=dic['config']['use_embedding'],
                attention_pooling=dic['config']['attention_pooling'],
                mix_edge_node=dic['config'].get('mix_edge_node', False),
                add_features=dic['config'].get('add_features', True),
            ),
            use_embedding=dic['config']['use_embedding'],
            use_graph=dic['config']['use_gnn'],
            new_graph=dic['config'].get('new_graph', False),
        )
        model_params = {
            'params': dic['params'],
            'batch_stats': dic['batch_stats']
        }
    return model, model_params

def main():
    env_id = "chess"
    env = pgx.make(env_id)

    run1 = "chess_2024-03-25:18h42"
    run2 = "chess_2024-04-06:22h14"

    games_dir = f"./tournaments/{run1} vs {run2}"
    # games_dir = f"./tournaments/{run1}"
    os.makedirs(games_dir, exist_ok=True)

    iterations = list(range(59, 100, 10))
    # iterations = list(range(50, 401, 50))
    # iterations = list(range(300, 401, 5))
    # iterations = [375, 395]
    # iterations = [50]
    models = {}
    models_params = {}
    for it in iterations:
        models[it], models_params[it] = load_model(
            env,
            f"./models/{run1}/{it:06}.ckpt",
            f"EdgeNet2-{run1}-{it:03}",
            EdgeNet2
        )

    iterations = list(range(1, 20, 2))
    for it in iterations:
        models[f"test{it:03d}"], models_params[f"test{it:03d}"] = load_model(
            env,
            f"./models/{run2}/{it:06}.ckpt",
            f"EdgeNet2-{run2}-{it:03}",
            EdgeNet2
        )

    matches = [
        (model1, model2)
        for model1 in models.keys()
        for model2 in models.keys()
        if str(model1) < str(model2)
    ]

    rng_key = jax.random.PRNGKey(42)

    outcomes = {
        model: {}
        for model in models.keys()
    }

    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[logs]}"),
        speed_estimate_period=1000
    ) as progress:
        task = progress.add_task(
            "[green]Running Tournament",
            total=len(matches),
            logs='...'
        )
        for model1, model2 in matches:
            progress.update(
                task,
                logs=f"{models[model1].id} vs {models[model2].id}",
            )
            rng_key, subkey = jax.random.split(rng_key)
            R, games = mcts.full_pit(
                env,
                models[model1],
                jax.device_put_replicated(models_params[model1], devices),
                models[model2],
                jax.device_put_replicated(models_params[model2], devices),
                subkey,
                n_games=64,
                max_plies=256,
                n_devices=num_devices
            )
            wins, draws, losses = map(
                lambda r: ((R == r).sum()).item(),
                [1, 0, -1]
            )
            outcomes[model1][model2] = R.mean().item()
            outcomes[model2][model1] = -R.mean().item()

            count = [128] * 3
            with open(os.path.join(
                games_dir,
                f"{models[model1].id} vs {models[model2].id}.pgn"
            ), "w") as f:
                for r, g in zip(R, games):
                    r_i = int(np.round(r))
                    if count[r_i+1] > 0:
                        count[r_i+1] -= 1
                        print(to_pgn(
                            g,
                            round=f"Tournament for {run1} vs {run2}",
                            player0=models[model1].id,
                            player1=models[model2].id,
                            result=r_i,
                            pgc=pgc
                        ), file=f)
            progress.update(
                task,
                advance=1,
            )
            print(f"{models[model1].id} vs {models[model2].id}, {wins: 3}/{draws: 3}/{losses: 3}")

    # elo = {
    #     model: np.random.randint(0, 2000)
    #     for model in models.keys()
    # }
    # outcomes = {
    #     model1: {
    #         model2: 2 / (1 + 10 ** ((elo[model2] - elo[model1]) / 400 + np.random.normal())) - 1
    #         for model2 in models.keys()
    #     }
    #     for model1 in models.keys()
    # }

    pprint(outcomes)
    for model1 in models.keys():
        for model2 in models.keys():
            print(f"{outcomes[model1].get(model2, 0): 6.3f}", end=" ")
        print()

    model_keys = list(models.keys())
    matches = [
        (i1, i2)
        for i1, model1 in enumerate(model_keys)
        for i2, model2 in enumerate(model_keys)
        if str(model1) < str(model2)
    ]
    X = np.array([
        match_two_hot(i1, i2, len(model_keys))
        for i1, i2 in matches
    ] + [
        [1] * len(model_keys)
    ])
    y = np.concatenate([
        -(400 / np.log(10)) * np.log(1 / np.clip(np.array([
            (outcomes[model_keys[i1]][model_keys[i2]] + 1) / 2
            for i1, i2 in matches
        ]), 1e-10, 1-1e-10) - 1),
        [1000 * len(model_keys)]
    ])
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    print(model_keys)
    pprint((reg.coef_+0.5).astype(int))

if __name__ == "__main__":
    main()
