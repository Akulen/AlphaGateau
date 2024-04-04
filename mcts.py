from functools import partial
from typing import cast, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from models import ModelManager

config = {}
num_devices = 8

def recurrent_fn(
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    action: chex.Array,
    state: pgx.State,
    env: pgx.Env,
    model: ModelManager,
) -> Tuple[mctx.RecurrentFnOutput, pgx.State]:
    del rng_key

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    logits, value = model(
        model.format_data(state=state),
        legal_action_mask=state.legal_action_mask,
        params=params
    )

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

def play_ply(
    val: Tuple[pgx.State, jnp.ndarray],
    key: chex.PRNGKey,
    model: ModelManager,
    params: chex.ArrayTree,
    env: pgx.Env,
    n_sim: int=128
) -> Tuple[Tuple[pgx.State, jnp.ndarray], jnp.ndarray]:
    state0, R = val
    batch_size = state0.observation.shape[0]

    logits, value = model(
        model.format_data(state=state0),
        legal_action_mask=state0.legal_action_mask,
        params=params
    )

    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state0) # type: ignore
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=key,
        root=root,
        recurrent_fn=partial(recurrent_fn, env=env, model=model),
        num_simulations=n_sim,
        invalid_actions=~state0.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    action1 = policy_output.action

    state1 = jax.vmap(env.step)(state0, action1)
    R = R + state1.rewards[jnp.arange(batch_size), 0]

    x = jnp.stack([
        action1,
        state0.terminated,
        state0.legal_action_mask[jnp.arange(batch_size),action1],
        state0.current_player
    ])
    return (state1, R), x

class PlyOutput(NamedTuple):
    board: jnp.ndarray
    obs: jnp.ndarray
    lam: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray

def play_ply_datagen(
    state: pgx.State,
    key: chex.PRNGKey,
    model: ModelManager,
    params: chex.ArrayTree,
    env: pgx.Env,
    n_sim: int=128
) -> Tuple[pgx.State, PlyOutput]:
    batch_size = state.observation.shape[0]
    key1, key2 = jax.random.split(key)

    logits, value = model(
        model.format_data(state=state),
        legal_action_mask=state.legal_action_mask,
        params=params
    )

    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state) # type: ignore
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=key1,
        root=root,
        recurrent_fn=partial(recurrent_fn, env=env, model=model),
        num_simulations=n_sim,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    actor = state.current_player
    keys = jax.random.split(key2, batch_size)
    new_state = jax.vmap(auto_reset(env.step, env.init))(
        state,
        policy_output.action, # should sample and not take best if first 20 moves
        keys
    )
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(new_state.terminated, 0.0, discount)
    return new_state, PlyOutput(
        board=state._board, # type: ignore
        obs=state.observation,
        lam=state.legal_action_mask,
        action_weights=cast(jnp.ndarray, policy_output.action_weights),
        reward=new_state.rewards[jnp.arange(new_state.rewards.shape[0]), actor],
        terminated=new_state.terminated,
        discount=discount,
    )


def play_move(
    val: Tuple[pgx.State, jnp.ndarray],
    key: chex.PRNGKey,
    model1: ModelManager,
    params1: chex.ArrayTree,
    model2: ModelManager,
    params2: chex.ArrayTree,
    env: pgx.Env,
    n_sim: int=128
) -> Tuple[Tuple[pgx.State, jnp.ndarray], jnp.ndarray]:
    state0, R = val

    key1, key2 = jax.random.split(key)

    (state1, R), r1 = play_ply((state0, R), key1, model1, params1, env, n_sim)
    (state2, R), r2 = play_ply((state1, R), key2, model2, params2, env, n_sim)

    return (state2, R), jnp.stack([r1, r2])

def init_with_player(key: chex.PRNGKey, player: jnp.int32, env: pgx.Env):
    state: pgx.State = env.init(key)
    state = state.replace(current_player=player) # type: ignore
    observation = env.observe(state, state.current_player)
    return state.replace(observation=observation) # type: ignore

def pit(
    env: pgx.Env,
    model1: ModelManager,
    params1: chex.ArrayTree,
    model2: ModelManager,
    params2: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    n_games: int=128,
    max_plies: int=512,
    n_sim: int=128,
    reverse_player: bool=False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, n_games)
    states = jax.vmap(partial(init_with_player, env=env))(
        keys,
        (jnp.ones if reverse_player else jnp.zeros)(n_games, dtype=jnp.int32)
    )
    if reverse_player:
        model1, model2 = model2, model1
        params1, params2 = params2, params1

    keys = jax.random.split(key, max_plies // 2)
    (_, R), actions = jax.lax.scan(
        partial(
            play_move,
            env=env,
            n_sim=n_sim,
            model1=model1,
            params1=params1,
            model2=model2,
            params2=params2
        ),
        (states, jnp.zeros(n_games)),
        keys
    )
    return R, actions

_pit_fn = jax.pmap(
    pit,
    static_broadcasted_argnums=[0, 1, 3, 6, 7, 8, 9]
)
def full_pit(
    env: pgx.Env,
    model1: ModelManager,
    params1: chex.ArrayTree,
    model2: ModelManager,
    params2: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    n_games: int=128,
    max_plies: int=512,
    n_sim: int=128,
    n_devices: int=1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    Rs, games = dict(), dict()
    for reverse_player in [False, True]:
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, n_devices)
        Rs[reverse_player], games[reverse_player] = _pit_fn(
            env,
            model1, params1,
            model2, params2,
            keys,
            n_games // (2 * n_devices),
            max_plies,
            n_sim,
            reverse_player
        )
    R = jnp.concatenate([Rs[False], Rs[True]]).reshape((-1,))
    # (#devices, moves, half-moves, data, batch)
    games = jnp.concatenate([games[False], games[True]])
    games = games.transpose([0, 4, 1, 2, 3]) \
                 .reshape((
                     games.shape[0] * games.shape[4], # Games
                     games.shape[1] * games.shape[2], # Moves
                     # Data: (action, terminated?, legal?, player)
                     games.shape[3]
                 ))
    return R, games
