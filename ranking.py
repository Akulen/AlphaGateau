import os
import json

import numpy as np
import jax
import pgx
import pgx.chess as pgc
import rich.progress as rp
from rich.pretty import pprint

from models import load_model
import mcts
from utils import to_pgn, Sample
from utils_progress import ProgressEMA, TimeRemainingColumn
from elo import compute_elo

import argparse

parser = argparse.ArgumentParser(
    prog='ModelRatings',
    description='Evaluate the Elo ratings of a range of iterations of a model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('model', type=str)
parser.add_argument(
    '-s',
    '--start',
    type=int,
    default=1,
    help='Start of iteration range, included'
)
parser.add_argument(
    '-e',
    '--end',
    type=int,
    default=100,
    help='End of iteration range, excluded'
)
parser.add_argument(
    '-k',
    '--step',
    type=int,
    default=2,
    help='Step size of iteration range'
)
parser.add_argument(
    '-m',
    '--n_matches',
    type=int,
    default=5,
    help='Number of opponents to play against for evaluation'
)
parser.add_argument(
    '-n',
    '--n_games',
    type=int,
    default=64,
    help='Number of games to play against each opponent'
)
parser.add_argument(
    '-i',
    '--n_sim',
    type=int,
    default=128,
    help='Number of MCTS simulations'
)
parser.add_argument(
    '-a',
    '--elo_average',
    type=int,
    default=1000,
    help='Average Elo of all models'
)
parser.add_argument(
    '-g',
    '--elo_guess',
    type=int,
    default=None,
    help='Guess of the initial Elo rating of the first iteration'
)

devices = jax.local_devices()
num_devices = len(devices)

games_dir = f"./tournaments/ranking matches"
os.makedirs(games_dir, exist_ok=True)


def evaluate_model(
    model_name,
    it,
    data,
    models,
    models_params,
    env,
    progress,
    n_matches=5,
    n_games=64,
    n_sim=128,
    elo_guess=1000,
    elo_average=1000,
    super_task=None
):
    # We first assume the model is average
    full_model_name = f'{model_name}/{it:03}'
    if full_model_name not in data['elo']:
        data['elo'][full_model_name] = elo_guess
        data['std'][full_model_name] = 499.5

    used = {full_model_name}
    rng_key = jax.random.PRNGKey(42)
    opponents = []
    task = progress.add_task(
        "[green]Estimating Elo",
        total=n_matches,
        logs='...'
    )
    for _ in range(n_matches):
        progress.update(
            task,
            logs=f"Current Estimate: {data['elo'][full_model_name]:4}±{int(2*data['std'][full_model_name]):3}, Looking for opponent...",
        )
        # We look for the model that hasn't been chosen yet with the closest
        # elo
        best_candidate, distance, acc = None, np.inf, 0
        for candidate in data['elo'].keys():
            if candidate in used:
                continue
            cand_dist = abs(
                data['elo'][candidate] - data['elo'][full_model_name]
            )
            if cand_dist < distance \
                    or (cand_dist == distance and acc < data['std'][candidate]):
                best_candidate = candidate
                distance = cand_dist
                acc = data['std'][candidate]
        assert best_candidate is not None
        used.add(best_candidate)
        progress.update(
            task,
            logs=f"Current Estimate: {data['elo'][full_model_name]:4}±{int(2*data['std'][full_model_name]):3}, Playing against [{data['elo'][best_candidate]:4}]{best_candidate}",
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
            n_games=n_games,
            max_plies=256,
            n_sim=n_sim,
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

        print(f'{full_model_name}[{data["elo"][full_model_name]:4}±{int(2*data["std"][full_model_name]):3}] vs [{data["elo"][opponent]:4}±{int(2*data["std"][opponent]):3}]{opponent}: {wins:2}/{draws:2}/{losses:2}')

        data['elo'], data['std'] = compute_elo(
            data['results'],
            wls=True,
            average=elo_average
        )

        count = [n_games] * 3
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
            logs=f"Final Elo for {full_model_name}: {data['elo'][full_model_name]:4}±{int(2*data['std'][full_model_name]):3} [{elos}]",
        )
        if super_task is not None:
            progress.advance(super_task, 1/n_matches)

    return data, models, models_params, (task, full_model_name, opponents)



def main():
    args = parser.parse_args()

    env_id = "chess"
    env = pgx.make(env_id)

    with open("rankings.json", "r") as file:
        data = json.load(file)
    with open("rankings.backup.json", "w") as file:
        json.dump(data, file)

    models = {}
    models_params = {}

    start, end, step = args.start, args.end, args.step
    X = list(range(start, end, step))
    model_name = args.model
    prev_elo = args.elo_guess or args.elo_average
    if f"{model_name}/{start-step:03}" in data['elo']:
        prev_elo = data['elo'][f"{model_name}/{start-step:03}"]
    update_data = []
    with ProgressEMA(
        rp.TextColumn("[progress.description]{task.description}"),
        rp.BarColumn(),
        rp.TaskProgressColumn(),
        TimeRemainingColumn(exponential_moving_average=True),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[logs]}"),
        refresh_per_second=1
    ) as progress:
        task = progress.add_task(
            "[red]Evaluating Model",
            total=len(X),
            logs=model_name
        )
        for i, it in enumerate(X):
            progress.update(
                task,
                logs=f"{model_name}/{it:03}"
            )
            models[f'{model_name}/{it:03}'], models_params[f'{model_name}/{it:03}'] = load_model(
                env,
                f"./models/{model_name}/{it:06}.ckpt",
                f'{model_name}-{it:03}'
            )

            data, models, models_params, task_data = evaluate_model(
                model_name,
                it,
                data,
                models,
                models_params,
                env,
                progress,
                n_matches=args.n_matches,
                n_games=args.n_games,
                n_sim=args.n_sim,
                elo_guess=prev_elo,
                elo_average=args.elo_average,
                super_task=task,
            )
            update_data.append(task_data)
            prev_elo = data['elo'][f'{model_name}/{it:03}']
            progress.update(task, completed=i+1)

            for task_upd, full_model_name, opponents in update_data:
                elos = '|'.join([f'{data["elo"][opp]:4}' for opp in opponents])
                progress.update(
                    task_upd,
                    logs=f"Final Elo for {full_model_name}: {data['elo'][full_model_name]:4}±{int(2*data['std'][full_model_name]):3} [{elos}]",
                )

    data['elo'], data['std'] = compute_elo(
        data['results'],
        wls=True,
        average=args.elo_average
    )
    with open("rankings.json", "w") as file:
        json.dump(data, file)
    pprint({f"{model_name}/{it:03}": data['elo'][f'{model_name}/{it:03}'] for it in X})

if __name__ == "__main__":
    main()
