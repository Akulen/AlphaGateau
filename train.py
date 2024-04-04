import faulthandler
faulthandler.enable()

import time
from functools import partial
import os
import pickle
from typing import NamedTuple, Tuple

from aim import Run
import chex
import humanhash
import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import optax
import pgx
import rich.progress as rp
from rich.pretty import pprint

from models import ModelManager, AZNet, EdgeNet, EdgeNet2
import mcts
from utils import elo_from_results, to_pgn

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

devices = jax.local_devices()
num_devices = len(devices) # in {1, 6, 8}
assert 24 % num_devices == 0

def reduce_multiple(n, m):
    return int(n / m) * m

config = {
    'gardner': True,
    'use_gnn': True,
    'n_gnn_layers': 6, # 5,
    'new_graph': True,
    'mix_edge_node': True,
    'add_features': True,
    'use_embedding': True,
    'attention_pooling': True,
    'inner_size': 128,
    'n_iter': 100,
    'eval_interval': 5,
    'eval_batch_size': 128, #, # 1024,
    'selfplay_batch_size': 1024, # 256, # 1024,
    'training_batch_size': 2**7, # 4 096
    'window_size': 2_000_000,
    'n_training_pass': 1,
    'max_num_steps': 256,
    'num_simulations': 128,
    'learning_rate': 0.001,
    'dotv2': True,
}
if config['use_gnn'] == False:
    config['use_embedding'] = False
if config['gardner']:
    import pgx.gardner_chess as pgc
else:
    import pgx.chess as pgc
    config['max_num_steps'] = 512

config['eval_batch_size'] = reduce_multiple(
    config['eval_batch_size'],
    num_devices
)
config['selfplay_batch_size'] = reduce_multiple(
    config['selfplay_batch_size'],
    (num_devices * config['training_batch_size']) // config['max_num_steps']
)
config['window_size'] = reduce_multiple(
    config['window_size'],
    config['training_batch_size'] * num_devices
)

def resume_task(progress: rp.Progress, task_id: rp.TaskID) -> None:
    with progress._lock:
        task = progress._tasks[task_id]
        if task.start_time is None:
            progress.start_task(task_id)
        elif task.stop_time is not None:
            current_time = progress.get_time()
            task.start_time = task.start_time - task.stop_time + current_time
            task.stop_time = None

if config['gardner']:
    class BaseLineModel:
        def __init__(self, forward):
            self.forward = forward

        def init(self, key, x):
            pass

        def apply(self, params, x, mutable, training):
            return self.forward.apply(
                params['params'],
                params['batch_stats'],
                x,
                is_eval=True
            )


    def pgx_make_baseline_model(
        model_id: pgx.BaselineModelId,
        download_dir: str="baselines"
    ):
        import haiku as hk
        from pgx._src.baseline import _load_baseline_model, _create_az_model_v0

        model_args, model_params, model_state = _load_baseline_model(
            model_id, download_dir
        )

        def forward_fn(x, is_eval=False):
            net = _create_az_model_v0(**model_args)
            policy_out, value_out = net(
                x,
                is_training=not is_eval,
                test_local_stats=False
            )
            return policy_out, value_out

        forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
        baseline_params = {
            'params': model_params,
            'batch_stats': model_state
        }
        return (
            BaseLineModel(forward),
            baseline_params
        )

    env_id = 'gardner_chess'
    env = pgx.make(env_id)
    model_id = 'gardner_chess_v0'
    baseline_model, baseline_params = pgx_make_baseline_model(model_id)
    baseline_name = "Baseline_pgx"
    baseline_model = ModelManager(
        id=baseline_name,
        model=baseline_model, # type: ignore
        use_embedding=False,
        use_graph=False,
        new_graph=False,
    )
    baseline_params = jax.tree_map(
        partial(jax.device_put_replicated, devices=devices),
        baseline_params
    )
else:
    env_id = 'chess'
    env = pgx.make(env_id)
    # with open("./models/chess_2024-02-05:14h08/000390.ckpt", "rb") as f:
    it = 50
    baseline_name = f"chess_2024-02-05:14h08/{it:06}"
    with open(f"./models/{baseline_name}.ckpt", "rb") as f:
        # dic = {
        #     "config": config,
        #     "rng_key": rng_key,
        #     "params": jax.device_get(params_0),
        #     "batch_stats": jax.device_get(batch_stats_0),
        #     "opt_state": jax.device_get(opt_state_0),
        #     "iteration": iteration,
        #     "frames": frames,
        #     "hours": hours,
        #     "pgx.__version__": pgx.__version__,
        #     "env_id": env.id,
        #     "env_version": env.version,
        #     "R": R,
        # }
        dic = pickle.load(f)
        baseline_model = ModelManager(
            id=f"Baseline_EdgeNet2_{baseline_name}",
            model=EdgeNet(
                n_actions=env.num_actions,
                inner_size=dic['config']['inner_size'],
                dot_v2=dic['config']['dotv2'],
                use_embedding=dic['config']['use_embedding'],
                attention_pooling=dic['config']['attention_pooling'],
            ),
            use_embedding=dic['config']['use_embedding'],
            use_graph=dic['config']['use_gnn'],
            new_graph=dic['config'].get('new_graph', False),
        )
        baseline_params, baseline_batch_stats = jax.device_put_replicated(
            (dic['params'], dic['batch_stats']),
            devices
        )
        baseline_params = {
            'params': baseline_params,
            'batch_stats': baseline_batch_stats
        }

init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

optimizer = optax.adam(learning_rate=config['learning_rate'])


@partial(jax.pmap, static_broadcasted_argnums=[1, 3, 4, 5, 6])
def selfplay(
    rng_key: chex.PRNGKey,
    model: ModelManager,
    params: chex.ArrayTree,
    env: pgx.Env,
    n_games: int=128,
    max_plies: int=512,
    n_sim: int=128
) -> mcts.PlyOutput:
    batch_size = n_games // num_devices

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, max_plies)
    _, data = jax.lax.scan(
        partial(
            mcts.play_ply_datagen,
            model=model,
            params=params,
            env=env,
            n_sim=n_sim
        ),
        state,
        key_seq
    )

    return data


class Sample(NamedTuple):
    board: jnp.ndarray
    obs: jnp.ndarray
    # board_or_obs: jnp.ndarray
    lam: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: mcts.PlyOutput) -> Sample:
    batch_size = config['selfplay_batch_size'] // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config['max_num_steps'] - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config['max_num_steps']),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        # board_or_obs=data.board if config['use_embedding'] and config['use_gnn'] else data.obs,
        board=data.board,
        # must be float because of total_move_count and no_progress_count
        # could be jnp.int4 without them
        obs=data.obs,
        lam=data.lam,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(
    params: chex.ArrayTree,
    batch_stats: chex.ArrayTree,
    samples: Sample,
    model: ModelManager
) -> Tuple[jnp.ndarray, Tuple[chex.ArrayTree, jnp.ndarray, jnp.ndarray]]:
    (logits, value), batch_stats = model(
        model.format_data(
            board=samples.board,
            observation=samples.obs,
            legal_action_mask=samples.lam
        ),
        legal_action_mask=samples.lam,
        params={'params': params, 'batch_stats': batch_stats},
        training=True
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    policy_loss_norm = optax.kl_divergence(
        jnp.log(jax.nn.softmax(logits)+1e-40),
        samples.policy_tgt
    )
    policy_loss_norm = jnp.mean(policy_loss_norm)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    # mask if the episode is truncated
    value_loss = jnp.mean(value_loss * samples.mask)
    value_loss = jnp.sqrt(value_loss)

    return policy_loss + value_loss, (batch_stats, policy_loss_norm, value_loss)


@partial(jax.pmap, axis_name="i", static_broadcasted_argnums=[4])
def train(
    params: chex.ArrayTree,
    batch_stats: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    data: Sample,
    model: ModelManager
) -> Tuple[
    chex.ArrayTree,
    chex.ArrayTree,
    chex.ArrayTree,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray
]:
    grads, (batch_stats, policy_loss, value_loss) = jax.grad(
        partial(loss_fn, model=model),
        has_aux=True
    )(params, batch_stats, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    max_grad = jax.tree_util.tree_reduce(
        lambda x, g: jnp.max(jnp.array([x, jnp.abs(g).max()])),
        grads,
        initializer=jnp.finfo(jnp.float32).min
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, batch_stats, opt_state, policy_loss, value_loss, max_grad


def training(
    sample_window: Sample,
    model: ModelManager,
    params: chex.ArrayTree,
    batch_stats: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    n_steps: int=2,
    batch_size: int=2**12
) -> Tuple[
    Tuple[Sample, chex.ArrayTree, chex.ArrayTree, chex.ArrayTree],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
]:
    n_data = sample_window.obs.shape[0]
    num_updates = (
           n_data
        // num_devices
        // batch_size
    )

    policy_losses = []
    value_losses = []
    max_grads = []
    for _step in range(n_steps):
        sub_key, rng_key = jax.random.split(rng_key)
        ixs = jax.device_get(jax.random.permutation(
            sub_key,
            jnp.arange(n_data)
        ))
        sample_window = jax.tree_map(lambda x: x[ixs], sample_window) # shuffle
        minibatches = jax.tree_map(
            lambda x: x.reshape(
                (
                    num_updates,
                    num_devices,
                    batch_size
                ) + x.shape[1:]
            ),
            sample_window
        )
        for i_batch in range(num_updates):
            batch = jax.tree_map(lambda x: x[i_batch], minibatches)
            params, batch_stats, opt_state, policy_loss, value_loss, max_grad =\
                train(params, batch_stats, opt_state, batch, model)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            max_grads.append(max_grad)

    policy_losses = jnp.concatenate(policy_losses)
    value_losses = jnp.concatenate(value_losses)
    max_grads = jnp.concatenate(max_grads)

    return (
        (sample_window, params, batch_stats, opt_state),
        (policy_losses, value_losses, max_grads)
    )


def evaluate(
    rng_key: chex.PRNGKey,
    model: ModelManager,
    params: chex.ArrayTree,
    n_games: int=128,
    max_plies: int=512,
    n_sim: int=128,
    save_n_games: int=0,
    games_file: str | None=None,
    round_name: str="Evaluation",
) -> Tuple[jnp.ndarray, float, float, float, float]:
    R, games = mcts.full_pit(
        env,
        model, params,
        baseline_model, baseline_params,
        rng_key,
        n_games=n_games,
        max_plies=max_plies,
        n_sim=n_sim,
        n_devices=num_devices
    )

    count = [save_n_games] * 3
    if games_file is not None:
        with open(games_file, "w") as f:
            for r, g in zip(R, games):
                r_i = int(np.round(r))
                if count[r_i+1] > 0:
                    count[r_i+1] -= 1
                    print(to_pgn(
                        g,
                        round=round_name,
                        player0=model.id,
                        player1=baseline_model.id,
                        result=r_i,
                        gardner=config['gardner'],
                        pgc=pgc
                    ), file=f)

    avg_R = R.mean().item()
    win_rate, draw_rate, lose_rate = map(
        lambda r: ((R == r).sum() / R.size).item(),
        [1, 0, -1]
    )
    # opt_state = jax.device_put_replicated(opt_state0, devices)
    return R, avg_R, win_rate, draw_rate, lose_rate


def main():
    debug = False

    run = None
    run_name = "EdgeNet" if config['use_gnn'] else "AZNet"
    if config['new_graph']:
        run_name += "2"
    if not debug:
        run = Run(
            repo='aim://localhost:53800',
            experiment='mctx_dev_'
                + ('gardner' if config['gardner'] else 'chess'),
            capture_terminal_logs=False,
            system_tracking_interval=None,
        )
        run.name = humanhash.humanize(run.hash, words=3)
        run_name += " " + run.name

    if config['use_gnn']:
        if config['new_graph']:
            model = EdgeNet2(
                n_actions=env.num_actions,
                n_gnn_layers=config['n_gnn_layers'],
                inner_size=config['inner_size'],
                attention_pooling=config['attention_pooling'],
                mix_edge_node=config['mix_edge_node'],
                add_features=config['add_features'],
            )
        else:
            model = EdgeNet(
                n_actions=env.num_actions,
                inner_size=config['inner_size'],
                dot_v2=config['dotv2'],
                use_embedding=config['use_embedding'],
                attention_pooling=config['attention_pooling'],
            )
    else:
        model = AZNet(
            num_actions=env.num_actions,
            num_channels=config['inner_size'],
            num_blocks=6,
        )
    model = ModelManager(
        id=f"{run_name}",
        model=model,
        use_embedding=config['use_embedding'],
        use_graph=config['use_gnn'],
        new_graph=config['new_graph'],
    )

    dummy_state = init_fn(jax.random.split(jax.random.PRNGKey(0), 2))
    x = model.format_data(state=dummy_state)

    variables = model.init(jax.random.PRNGKey(0), x)
    params, batch_stats = variables['params'], variables['batch_stats']
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    config['param_count'] = param_count
    config['baseline'] = baseline_name

    if run is not None:
        run["hparams"] = config

    opt_state0 = optimizer.init(params=params)
    params, batch_stats, opt_state = jax.device_put_replicated(
        (params, batch_stats, opt_state0),
        devices
    )

    now = time.strftime("%Y-%m-%d:%Hh%M")
    models_dir = os.path.join("models", f"{env_id}_{now}")
    os.makedirs(models_dir, exist_ok=True)
    games_dir = os.path.join("games", f"{env_id}_{now}")
    os.makedirs(games_dir, exist_ok=True)

    frames = 0
    hours = {
        'eval': 0.0,
        'selfplay': 0.0,
        'train': 0.0,
    }
    sample_window = None

    rng_key = jax.random.PRNGKey(42)
    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[logs]}"),
        speed_estimate_period=10000
    ) as progress:
        task_gen = progress.add_task(
            "[cyan]Generating",
            total=config['n_iter'],
            logs='...',
            start=False
        )
        task_train = progress.add_task(
            "[red]Training",
            total=config['n_iter'],
            logs='...',
            start=False
        )
        task_eval = progress.add_task(
            "[green]Evaluating",
            total=(config['n_iter'] + config['eval_interval'] - 1)
                // config['eval_interval'],
            logs='...',
            start=False
        )
        for iteration in range(config['n_iter']):
            # Selfplay
            resume_task(progress, task_gen)
            st = time.time()

            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            data: mcts.PlyOutput = selfplay(
                keys,
                model,
                {'params': params, 'batch_stats': batch_stats},
                env,
                config['selfplay_batch_size'],
                config['max_num_steps'],
                config['num_simulations']
            )

            samples: Sample = compute_loss_input(data)
            # (#devices, batch, max_num_steps, ...)
            samples = jax.device_get(samples)

            frames += (
                  samples.obs.shape[0]
                * samples.obs.shape[1]
                * samples.obs.shape[2]
            )
            samples = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, *x.shape[3:])),
                samples
            )

            et = time.time()
            hours['selfplay'] += (et - st) / 3600
            progress.update(
                task_gen,
                advance=1,
                logs=f'{frames} frames'
            )
            progress.stop_task(task_gen)

            if sample_window is None:
                sample_window = samples
            else:
                sample_window = jax.tree_util.tree_map(
                    lambda x, y:
                        np.concatenate(
                            [x, y],
                            axis=0
                        )[:config['window_size']],
                    samples,
                    sample_window
                )

            # Training
            resume_task(progress, task_train)
            st = time.time()

            rng_key, subkey = jax.random.split(rng_key)

            data_tuple, losses = training(
                sample_window,
                model,
                params,
                batch_stats,
                opt_state,
                subkey,
                n_steps=config['n_training_pass'],
                batch_size=config['training_batch_size']
            )
            sample_window, params, batch_stats, opt_state = data_tuple
            policy_losses, value_losses, max_grad = losses
            policy_loss = policy_losses.mean().item()
            value_loss = value_losses.mean().item()
            loss = policy_loss + value_loss

            if run is not None:
                run.track(
                    {
                        "loss": loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "max_grad": max_grad.max().item(),
                    },
                    context={
                        'subset': 'train',
                    },
                    epoch=iteration
                )
            et = time.time()
            hours['train'] += (et - st) / 3600
            progress.update(
                task_train,
                advance=1,
                logs=(
                    f'loss (policy + value): '
                    f'{loss:.5f} ({policy_loss:.5f} + {value_loss:.5f})'
                )
            )
            progress.stop_task(task_train)

            st = time.time()
            if iteration % config['eval_interval'] == config['eval_interval']-1:
                resume_task(progress, task_eval)
                # Evaluation
                rng_key, subkey = jax.random.split(rng_key)
                R, avg_R, win_rate, draw_rate, lose_rate = evaluate(
                    subkey,
                    model,
                    {'params': params, 'batch_stats': batch_stats},
                    n_games=config['eval_batch_size'],
                    max_plies=config['max_num_steps'],
                    n_sim=config['num_simulations'],
                    save_n_games=config['eval_batch_size'],
                    games_file=os.path.join(
                        games_dir,
                        f'{iteration:06d}.pgn'
                    ),
                    round_name=f'{iteration}'
                )
                if run is not None:
                    run.track(
                        {
                            "elo_rating": elo_from_results(
                                avg_R,
                                base=1000,
                                max_delta=1000
                            ),
                            "avg_R": avg_R,
                            "win_rate": win_rate,
                            "draw_rate": draw_rate,
                            "lose_rate": lose_rate,
                        },
                        context={
                            'subset': 'eval',
                            'tag': 'eval/vs_baseline',
                        },
                        step=iteration,
                        epoch=iteration
                    )
                progress.update(
                    task_eval,
                    advance=1,
                    logs=(
                        f'win/lose rate (mean R): '
                        f'{win_rate:.3f} / {lose_rate:.3f} ({avg_R:.3f})'
                    )
                )
                progress.stop_task(task_eval)

                # Store checkpoints
                params_0, batch_stats_0, opt_state_0 = jax.tree_util.tree_map(
                    lambda x: x[0],
                    (params, batch_stats, opt_state)
                )
                with open(os.path.join(
                    models_dir,
                    f"{iteration:06d}.ckpt"
                ), "wb") as f:
                    dic = {
                        "config": config,
                        "rng_key": rng_key,
                        "params": jax.device_get(params_0),
                        "batch_stats": jax.device_get(batch_stats_0),
                        "opt_state": jax.device_get(opt_state_0),
                        "iteration": iteration,
                        "frames": frames,
                        "hours": hours,
                        "pgx.__version__": pgx.__version__,
                        "env_id": env.id,
                        "env_version": env.version,
                        "R": R,
                    }
                    pickle.dump(dic, f)
            et = time.time()
            hours['eval'] += (et - st) / 3600

            # Update logs

            if run is not None:
                for tp in hours.keys():
                    run.track(
                        {
                            "hours": hours[tp],
                        },
                        context={
                            'subset': tp,
                        },
                        epoch=iteration
                    )
                run.track(
                    {
                        "hours": sum(hours.values()),
                    },
                    context={
                        'subset': "total",
                    },
                    epoch=iteration
                )
                run.track(
                    {
                        "frames": frames,
                    },
                    epoch=iteration
                )





# pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=0.5)

if __name__ == "__main__":
    main()
