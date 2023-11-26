from functools import partial
from typing import cast, NamedTuple, Tuple

from aim import Run
import chex
import jax
import jax.numpy as jnp
from jpyger import state_to_graph
import mctx
from mctx._src.base import RecurrentFnOutput
import optax
import pgx
from pgx.experimental import auto_reset
import rich.progress as rp

from models import EdgeNet

devices = jax.local_devices()
num_devices = len(devices)

config = {
    'n_iter': 10, # 100,
    'eval_interval': 1, # 5,
    'selfplay_batch_size': 12, # 1024,
    'training_batch_size': 12, # 1024,
    'max_num_steps': 256,
    'num_simulations': 32,
}

env_id = 'gardner_chess'
model_id = 'gardner_chess_v0'
env = pgx.make(env_id)
baseline = pgx.make_baseline_model(model_id)

init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

state_to_graph = jax.jit(partial(state_to_graph, n_actions=env.num_actions))

model = EdgeNet()
optimizer = optax.adam(learning_rate=0.001)

@jax.pmap
def evaluate(rng_key: jnp.ndarray, params) -> jnp.ndarray:
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config['selfplay_batch_size'] // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = init_fn(keys)

    def body_fn(val):
        key, state, R = val
        graphs = state_to_graph(state.observation, state.legal_action_mask)
        my_logits, _ = model.apply(
            {'params': params},
            graphs,
            training=False
        )
        my_logits = my_logits.reshape((batch_size, -1))
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        masked_logits: pgx.Array = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
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


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    lam: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


def recurrent_fn(
    params: chex.ArrayTree,
    rng_key: jnp.ndarray,
    action: chex.Array,
    state: pgx.State
) -> Tuple[RecurrentFnOutput, pgx.State]:
    # model: params
    # state: embedding
    del rng_key

    current_player = state.current_player
    state = step_fn(state, action)

    graphs = state_to_graph(state.observation, state.legal_action_mask)
    r_tuple: Tuple[jnp.ndarray, jnp.ndarray] = model.apply(
        {'params': params}, # type: ignore
        graphs,
        training=False
    )
    logits, value = r_tuple
    value = jnp.reshape(value, (-1,))
    logits = logits.reshape((value.shape[-1], -1))
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput( # type: ignore
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


@jax.pmap
def selfplay(rng_key: jnp.ndarray, params) -> SelfplayOutput:
    batch_size = config['selfplay_batch_size'] // num_devices

    def step_fn(state: pgx.State, key) -> Tuple[pgx.State, SelfplayOutput]:
        key1, key2 = jax.random.split(key)

        graphs = state_to_graph(state.observation, state.legal_action_mask)
        logits, value = model.apply(
            {'params': params},
            graphs,
            training=False
        )
        logits = logits.reshape((batch_size, -1))
        value = value.reshape((batch_size,)) # type: ignore
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state) # type: ignore
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config['num_simulations'],
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=state.observation,
            lam=state.legal_action_mask,
            action_weights=cast(jnp.ndarray, policy_output.action_weights),
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = init_fn(keys)
    key_seq = jax.random.split(rng_key, config['max_num_steps'])
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    lam: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
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
        obs=data.obs,
        lam=data.lam,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(params, samples: Sample):
    graphs = state_to_graph(samples.obs, samples.lam)
    r_tuple: Tuple[jnp.ndarray, jnp.ndarray] = model.apply(
        {'params': params}, # type: ignore
        graphs,
        training=True
    )
    logits, value = r_tuple
    value = jnp.reshape(value, (-1,))
    logits = logits.reshape((value.shape[-1], -1))

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(params, opt_state, data: Sample):
    grads, (policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        params, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, policy_loss, value_loss


def main():
    debug = False
    if not debug:
        run = Run(
            repo='aim://localhost:53800',
            experiment='mctx_dev',
        )
        run["hparams"] = config

    dummy_state = init_fn(jax.random.split(jax.random.PRNGKey(0), 2))
    variables = model.init(jax.random.PRNGKey(0), state_to_graph(dummy_state.observation, dummy_state.legal_action_mask))
    params = variables["params"]
    # print(params['Dense_0']['kernel'].shape) # type: ignore
    # batch_stats = variables["batch_stats"]
    # print(batch_stats)

    opt_state = optimizer.init(params=params)

    params, opt_state = jax.device_put_replicated((params, opt_state), devices)

    # batch_size = 2
    # keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    # state = init_fn(keys)
    # print(state_to_graph(state.observation, state.legal_action_mask))

    frames = 0
    rng_key = jax.random.PRNGKey(42)
    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("win/lose rate: {task.fields[win]} / {task.fields[lose]}"),
    ) as progress:
        task = progress.add_task(
            "[cyan]Iteration",
            total=config['n_iter'],
            win='N/A',
            lose='N/A'
        )
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
                if not debug:
                    run.track({ # type: ignore
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

            # Selfplay
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            data: SelfplayOutput = selfplay(keys, params)
            samples: Sample = compute_loss_input(data)

            # Shuffle samples and make minibatches
            samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
            frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
            samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
            rng_key, subkey = jax.random.split(rng_key)
            ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
            samples = jax.tree_map(lambda x: x[ixs], samples)  # shuffle
            num_updates = samples.obs.shape[0] // config['training_batch_size']
            minibatches = jax.tree_map(
                lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
            )

            # Training
            policy_losses, value_losses = [], []
            for i in range(num_updates):
                minibatch: Sample = jax.tree_map(lambda x: x[i], minibatches)
                params, opt_state, policy_loss, value_loss = train(params, opt_state, minibatch)
                policy_losses.append(policy_loss.mean().item())
                value_losses.append(value_loss.mean().item())
            policy_loss = sum(policy_losses) / len(policy_losses)
            value_loss = sum(value_losses) / len(value_losses)

            # Update logs

            if not debug:
                run.track({ # type: ignore
                    "train/policy_loss": policy_loss,
                    "train/value_loss": value_loss,
                    "frames": frames,
                }, epoch=iteration)
            progress.update(task, advance=1)





    # pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=0.5)

if __name__ == "__main__":
    main()
