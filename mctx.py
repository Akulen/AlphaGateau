from aim import Run
from functools import partial
import humanhash
import jax
import jax.numpy as jnp
# import mctx
import optax
import pgx
import rich.progress as rp

from jpyger import state_to_graph
from models import EdgeNet

devices = jax.local_devices()
num_devices = len(devices)

config = {
    'n_iter': 10,
    'eval_interval': 1,
    'selfplay_batch_size': 12, # 1024,
}

env_id = 'gardner_chess'
model_id = 'gardner_chess_v0'
env = pgx.make(env_id)
baseline = pgx.make_baseline_model(model_id)

init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

state_to_graph = jax.jit(partial(state_to_graph, n_actions=env.num_actions))

@jax.pmap
def evaluate(rng_key, params):
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config['selfplay_batch_size'] // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = init_fn(keys)

    def body_fn(val):
        key, state, R = val
        graphs = state_to_graph(state)
        my_logits, _ = model.apply(
            {'params': params},
            graphs,
            training=False
        )
        my_logits = my_logits.reshape((batch_size, -1))
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        masked_logits: pgx.Array = jnp.where(state.legal_action_mask, logits, -jnp.inf)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, masked_logits, axis=-1)
        # action = jnp.argmax(logits, axis=-1)
        state = step_fn(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (key, state, jnp.zeros(batch_size))
    )
    return R


if __name__ == "__main__":
    run = Run(
        repo='aim://localhost:53800',
        experiment='mctx_dev',
    )
    run["hparams"] = config

    model = EdgeNet()
    optimizer = optax.adam(learning_rate=0.001)

    dummy_state = init_fn(jax.random.split(jax.random.PRNGKey(0), 2))
    variables = model.init(jax.random.PRNGKey(0), state_to_graph(dummy_state))
    params = variables["params"]
    # print(params['Dense_0']['kernel'].shape) # type: ignore
    # batch_stats = variables["batch_stats"]
    # print(batch_stats)

    opt_state = optimizer.init(params=params)

    params, opt_state = jax.device_put_replicated((params, opt_state), devices)

    batch_size = 2
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    state = init_fn(keys)
    # print(state_to_graph(state))

    rng_key = jax.random.PRNGKey(42)
    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("win/lose rate: {task.fields[win]} / {task.fields[lose]}"),
    ) as progress:
        task = progress.add_task("[cyan]Iteration", total=config['n_iter'], win='N/A', lose='N/A')
        for iteration in range(config['n_iter']):
            if iteration % config['eval_interval'] == 0:
                # Evaluation
                rng_key, subkey = jax.random.split(rng_key)
                keys = jax.random.split(subkey, num_devices)
                R = evaluate(keys, params)
                print(R)
                avg_R = R.mean().item()
                win_rate, draw_rate, lose_rate = map(
                    lambda r: ((R == r).sum() / R.size).item(),
                    [1, 0, -1]
                )
                run.track({
                    "eval/vs_baseline/avg_R": avg_R,
                    "eval/vs_baseline/win_rate": win_rate,
                    "eval/vs_baseline/draw_rate": draw_rate,
                    "eval/vs_baseline/lose_rate": lose_rate,
                }, epoch=iteration)
                progress.update(
                    task,
                    win=f"{win_rate:.3f}",
                    lose=f"{lose_rate:.3f}",
                )

                # # Store checkpoints
                # model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
                # with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                #     dic = {
                #         "config": config,
                #         "rng_key": rng_key,
                #         "model": jax.device_get(model_0),
                #         "opt_state": jax.device_get(opt_state_0),
                #         "iteration": iteration,
                #         "frames": frames,
                #         "hours": hours,
                #         "pgx.__version__": pgx.__version__,
                #         "env_id": env.id,
                #         "env_version": env.version,
                #     }
                #     pickle.dump(dic, f)
            progress.update(task, advance=1)

        # pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=0.5)

