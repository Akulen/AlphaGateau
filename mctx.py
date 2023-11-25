from functools import partial
import jax
import jax.numpy as jnp
# import mctx
import optax
import pgx

from jpyger import state_to_graph
from models import EdgeNet

devices = jax.local_devices()
num_devices = len(devices)

config = {
    'n_iter': 2,
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
    for iteration in range(config['n_iter']):
        if iteration % config['eval_interval'] == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, params)
            print(R)
            # pgx.save_svg_animation(states[:n_iter+1], f"gardner_test.svg", frame_duration_seconds=0.5)
            import sys
            sys.exit()
            raise NotImplementedError
            # log.update(
            #     {
            #         f"eval/vs_baseline/avg_R": R.mean().item(),
            #         f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
            #         f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
            #         f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
            #     }
            # )

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
        raise NotImplementedError
    # states = []
    # batch_size = 2
    # keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    # state = init_fn(keys)
    # print(state_to_graph(state))

    # states.append(state)
    # while not (state.terminated | state.truncated).all():
    #     logits, value = baseline(state.observation)
    #     action = logits.argmax(-1)
    #     state = step_fn(state, action)
    #     states.append(state)

    # pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=0.5)

