import datetime
from functools import partial
import os
import pickle
import time
from typing import cast, NamedTuple, Tuple

from aim import Run
import chex
import humanhash
import jax
import jax.numpy as jnp
import jax.profiler
from jpyger import state_to_graph
import mctx
from mctx._src.base import RecurrentFnOutput
import numpy as np
import optax
import pgx
from pgx.experimental import auto_reset
import rich.progress as rp
from rich.pretty import pprint

from models import AZNet, EdgeNet

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

devices = jax.local_devices()
num_devices = len(devices) # in {1, 6, 8}
assert 24 % num_devices == 0

def reduce_multiple(n, m):
    return int(n / m) * m

config = {
    'gardner': False,
    'use_gnn': True,
    'use_embedding': True,
    'attention_pooling': True,
    'inner_size': 128,
    'n_iter': 401,
    'eval_interval': 5,
    'selfplay_batch_size': 256, # 1024,
    'training_batch_size': 4096,
    'window_size': 400_000,
    'n_training_pass': 2,
    'max_num_steps': 256, # should be prime with num_devices
    'num_simulations': 128, # 32,
    'learning_rate': 0.001,
    'dotv2': True,
}
if config['gardner']:
    import pgx.gardner_chess as pgc
else:
    import pgx.chess as pgc
    config['max_num_steps'] = 512
# config['training_batch_size'] = reduce_multiple(config['training_batch_size'], config['selfplay_batch_size'])
config['selfplay_batch_size'] = reduce_multiple(config['selfplay_batch_size'], (num_devices * config['training_batch_size']) // config['max_num_steps'])
config['window_size'] = reduce_multiple(config['window_size'], config['training_batch_size'] * num_devices)

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
    env_id = 'gardner_chess'
    env = pgx.make(env_id)
    model_id = 'gardner_chess_v0'
    baseline = pgx.make_baseline_model(model_id)
else:
    env_id = 'chess'
    env = pgx.make(env_id)
    # baseline = lambda obs: (jnp.ones((obs.shape[0], env.num_actions)), jnp.zeros((obs.shape[0],)))
    with open("models/chess_2024-01-09:16h56/000400.ckpt", "rb") as f:
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
        params = dic['params']
        batch_stats = dic['batch_stats']
        config_baseline = dic['config']
    baseline_model = EdgeNet(
        n_actions=env.num_actions,
        inner_size=config_baseline['inner_size'],
        dot_v2=config_baseline['dotv2'],
        use_embedding=config_baseline['use_embedding'],
        attention_pooling=config_baseline['attention_pooling'],
    )
    def baseline(x):
        r_tuple, _ = baseline_model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x,
            mutable=['batch_stats'],
            training=False
        )
        logits, value = r_tuple
        value = jnp.reshape(value, (-1,))
        logits = logits.reshape((value.shape[-1], -1))
        return logits, value

init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

state_to_graph = jax.jit(partial(state_to_graph, use_embedding=config['use_embedding']))

model = EdgeNet(
    n_actions=env.num_actions,
    inner_size=config['inner_size'],
    dot_v2=config['dotv2'],
    use_embedding=config['use_embedding'],
    attention_pooling=config['attention_pooling'],
) if config['use_gnn'] else AZNet(
    num_actions=env.num_actions,
    num_channels=config['inner_size'],
    num_blocks=6,
)
optimizer = optax.adam(learning_rate=config['learning_rate'])

def recurrent_fn(
    params: chex.ArrayTree,
    rng_key: jnp.ndarray,
    action: chex.Array,
    state: pgx.State,
    use_baseline: bool=False
) -> Tuple[RecurrentFnOutput, pgx.State]:
    # model: params
    # state: embedding
    del rng_key

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    if use_baseline:
        if config['gardner']:
            x = state.observation
        else:
            x = state_to_graph(state._board, state.observation, state.legal_action_mask)
        logits, value = baseline(x)
    else:
        if config['use_gnn']:
            x = state_to_graph(state._board, state.observation, state.legal_action_mask)
        else:
            x = state.observation
        r_tuple, _ = model.apply(
            params, # type: ignore
            x,
            mutable=['batch_stats'],
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


class SelfplayOutput(NamedTuple):
    board: jnp.ndarray
    obs: jnp.ndarray
    lam: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(rng_key: jnp.ndarray, params, batch_stats) -> SelfplayOutput:
    batch_size = config['selfplay_batch_size'] // num_devices

    def body_fn(state: pgx.State, key) -> Tuple[pgx.State, SelfplayOutput]:
        key1, key2 = jax.random.split(key)
        board, observation, lam = state._board, state.observation, state.legal_action_mask # type: ignore

        if config['use_gnn']:
            x = state_to_graph(board, observation, lam)
        else:
            x = observation
        (logits, value), _ = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x,
            mutable=['batch_stats'],
            training=False
        )
        logits = logits.reshape((batch_size, -1))
        value = value.reshape((batch_size,)) # type: ignore
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state) # type: ignore

        policy_output = mctx.gumbel_muzero_policy(
            params={'params': params, 'batch_stats': batch_stats},
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
            board=board,
            obs=observation,
            lam=lam,
            action_weights=cast(jnp.ndarray, policy_output.action_weights),
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config['max_num_steps'])
    _, data = jax.lax.scan(body_fn, state, key_seq)

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
        # board_or_obs=data.board if config['use_embedding'] and config['use_gnn'] else data.obs,
        board=data.board,
        obs=data.obs,
        lam=data.lam,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(params, batch_stats, samples: Sample):
    if config['use_gnn']:
        x = state_to_graph(samples.board, samples.obs, samples.lam)
    else:
        x = samples.obs
    (logits, value), batch_stats = model.apply(
        {'params': params, 'batch_stats': batch_stats}, # type: ignore
        x,
        mutable=['batch_stats'],
        training=True
    )
    batch_stats = batch_stats['batch_stats']
    value = jnp.reshape(value, (-1,))
    logits = logits.reshape((value.shape[-1], -1))

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    policy_loss_norm = optax.kl_divergence(jnp.log(jax.nn.softmax(logits)+1e-40), samples.policy_tgt)
    policy_loss_norm = jnp.mean(policy_loss_norm)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (batch_stats, policy_loss_norm, value_loss)


def train(params, batch_stats, opt_state, data: Sample):
    grads, (batch_stats, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        params, batch_stats, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    max_grad = jax.tree_util.tree_reduce(
        lambda x, g: jnp.max(jnp.array([x, jnp.abs(g).max()])),
        grads,
        initializer=jnp.finfo(jnp.float32).min
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, batch_stats, opt_state, policy_loss, value_loss, max_grad


@partial(jax.pmap, axis_name="i")
def training(sample_window, params, batch_stats, opt_state, keys):
    num_updates = sample_window.obs.shape[0] // config['training_batch_size']
    # num_updates = (samples.obs.shape[0] + config['training_batch_size'] - 1) // config['training_batch_size']

    def train_step(val, key):
        sample_window, params, batch_stats, opt_state = val
        ixs = jax.random.permutation(key, jnp.arange(sample_window.obs.shape[0]))
        sample_window = jax.tree_map(lambda x: x[ixs], sample_window)  # shuffle
        minibatches = jax.tree_map(
            lambda x: x.reshape((num_updates, config['training_batch_size']) + x.shape[1:]),
            sample_window
        )

        def train_batch(val_batch, batch):
            params, batch_stats, opt_state, policy_loss, value_loss, max_grad = train(*val_batch, batch)
            return (params, batch_stats, opt_state), (policy_loss, value_loss, max_grad)

        (params, batch_stats, opt_state), (policy_losses, value_losses, max_grad) = jax.lax.scan(
            train_batch,
            (params, batch_stats, opt_state),
            minibatches
        )
        return (sample_window, params, batch_stats, opt_state), (policy_losses, value_losses, max_grad)

    # Training step
    keys = jax.random.split(keys, config['n_training_pass'])
    return jax.lax.scan(train_step, (sample_window, params, batch_stats, opt_state), keys)


@jax.pmap
def evaluate(rng_key: jnp.ndarray, params, batch_stats) -> jnp.ndarray:
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config['selfplay_batch_size'] // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = init_fn(keys)

    def body_fn(val):
        key, state, R = val
        if config['use_gnn']:
            x = state_to_graph(state._board, state.observation, state.legal_action_mask)
        else:
            x = state.observation
        (my_logits, _), _ = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x,
            mutable=['batch_stats'],
            training=False
        )
        my_logits = my_logits.reshape((batch_size, -1))
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        # masked_logits: pgx.Array = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        key, subkey = jax.random.split(key)
        # action = jax.random.categorical(subkey, masked_logits, axis=-1)
        action = jax.random.categorical(subkey, logits, axis=-1) # type: ignore
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


@jax.jit
@jax.vmap
def init_with_player(key: jnp.ndarray, player: jnp.int32):
    state: pgx.State = env.init(key)
    state = state.replace(current_player=player) # type: ignore
    observation = env.observe(state, state.current_player)
    return state.replace(observation=observation) # type: ignore


@partial(jax.pmap, static_broadcasted_argnums=[3])
def evaluate_mcts(rng_key: jnp.ndarray, params, batch_stats, my_player: bool=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = config['selfplay_batch_size'] // num_devices // 2

    key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    states = init_with_player(keys, (jnp.zeros if my_player else jnp.ones)(batch_size, dtype=jnp.int32))

    def predict_root(state: pgx.State, use_baseline: bool=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if use_baseline:
            if config['gardner']:
                x = state.observation
            else:
                x = state_to_graph(state._board, state.observation, state.legal_action_mask)
            logits, value = baseline(x)
        else:
            if config['use_gnn']:
                x = state_to_graph(state._board, state.observation, state.legal_action_mask) # type: ignore
            else:
                x = state.observation
            (logits, value), _ = model.apply(
                {'params': params, 'batch_stats': batch_stats},
                x,
                mutable=['batch_stats'],
                training=False
            )
            logits = logits.reshape((batch_size, -1))
            value = value.reshape((batch_size,)) # type: ignore

        return logits, value

    def get_action(
        state: pgx.State,
        logits: jnp.ndarray,
        value: jnp.ndarray,
        keys: jnp.ndarray,
        use_baseline: bool=False
    ) -> chex.Array:
        # if use_baseline:
        #     keys, subkey = jax.random.split(keys)
        #     action = jax.random.categorical(subkey, logits, axis=-1)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state) # type: ignore
        policy_output = mctx.gumbel_muzero_policy(
            params={} if use_baseline else {'params': params, 'batch_stats': batch_stats},
            rng_key=keys,
            root=root,
            recurrent_fn=partial(recurrent_fn, use_baseline=use_baseline),
            num_simulations=config['num_simulations'],
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action

    # def body_fn(val: Tuple[jnp.ndarray, pgx.State, jnp.int32]) -> Tuple[jnp.ndarray, pgx.State, jnp.int32]:
    #     key, state, R = val
    #     key, keys1 = jax.random.split(key)
    #     key, keys2 = jax.random.split(key)

    #     # First Player
    #     logits, value = predict_root(state, use_baseline=not my_player)
    #     action = get_action(state, logits, value, keys1, use_baseline=not my_player)

    #     state = step_fn(state, action)
    #     R = R + state.rewards[jnp.arange(batch_size), int(not my_player)]

    #     # Ugly hack to alternate players
    #     # Second Player
    #     logits, value = predict_root(state, use_baseline=my_player)
    #     action = get_action(state, logits, value, keys2, use_baseline=my_player)

    #     state = step_fn(state, action)
    #     R = R + state.rewards[jnp.arange(batch_size), int(not my_player)]

    #     return (key, state, R)
    def body_fn(
        val: Tuple[pgx.State, jnp.int32],
        key: jnp.ndarray
    ) -> Tuple[Tuple[pgx.State, jnp.int32], jnp.ndarray]:
        state0, R = val
        keys1, keys2 = jax.random.split(key)

        # First Player
        logits, value = predict_root(state0, use_baseline=not my_player)
        action1 = get_action(state0, logits, value, keys1, use_baseline=not my_player)

        state1 = step_fn(state0, action1)
        R = R + state1.rewards[jnp.arange(batch_size), 0]

        # Ugly hack to alternate players
        # Second Player
        logits, value = predict_root(state1, use_baseline=my_player)
        action2 = get_action(state1, logits, value, keys2, use_baseline=my_player)

        state2 = step_fn(state1, action2)
        R = R + state2.rewards[jnp.arange(batch_size), 0]

        x = jnp.stack([
            jnp.stack([action1, state0.terminated, state0.legal_action_mask[jnp.arange(batch_size),action1], state0.current_player]),
            jnp.stack([action2, state1.terminated, state1.legal_action_mask[jnp.arange(batch_size),action2], state1.current_player])
        ])
        return (state2, R), x

    # _, _, R = jax.lax.while_loop(
    #     lambda x: ~(x[1].terminated.all()),
    #     body_fn,
    #     (key, state, jnp.zeros(batch_size))
    # )
    keys = jax.random.split(key, config['max_num_steps'] // 2)
    (_, R), actions = jax.lax.scan(
        body_fn,
        (states, jnp.zeros(batch_size)),
        keys
    )
    # states = jax.tree_util.tree_map(
    #     # we need to add a leading useless dimension as jnp.append is actually jnp.concatenate
    #     lambda x, y: jnp.append(x, y[None], axis=0),
    #     states,
    #     final_state
    # )
    return R, actions


def elo_from_results(results, base=1000, max_delta=1000):
    return base - np.clip(400 * np.log(2 / np.clip(results+1, 1e-100, 2-1e-100) - 1) / np.log(10), -max_delta, max_delta)


def move_pgn(board, move, i, brackets=False):
    moves_from, moves_plane = (move // 49, move % 49) if config['gardner'] else (move // 73, move % 73)
    moves_to = pgc.TO_MAP[moves_from, moves_plane]
    moves_underpromotion = moves_plane // 3 if moves_plane < 9 else -1
    size = 5 if config['gardner'] else 8
    def square2cart(square):
        row, col = square % size, square // size
        if i % 2 == 1:
            row = size - 1 - row
        return row, col
    def square2str(square):
        row, col = square2cart(square)
        return chr(ord('a') + col) + str(row + 1)
    from_row, from_col = square2cart(moves_from)
    to_row, to_col = square2cart(moves_to)
    piece = board[from_row][from_col]
    promotion = piece == 'P' and to_row == (size-1 if i % 2 == 0 else 0)
    new_piece = piece
    if promotion:
        new_piece = "QRBN"[1+moves_underpromotion]
    if piece == 'K' and abs(from_col - to_col) == 2: # Castling
        assert from_row == to_row
        board[to_row][(from_col + to_col) // 2] = 'R'
        if to_col > from_col:
            rook = 7
        else:
            rook = 0
        assert board[to_row][rook] == 'R'
        board[to_row][rook] = ' '
    board[to_row][to_col] = new_piece
    board[from_row][from_col] = ' '
    return (
          ("" if i % 2 == 1 else str(i // 2 + 1) + ". ")
        + ("(" if brackets else "")
        + ("" if piece in " P" else piece)
        + square2str(moves_from)
        + square2str(moves_to)
        + ("" if not promotion else "=" + "QRBN"[1+moves_underpromotion])
        + (")" if brackets else "")
    )


def to_pgn(moves, round:str|int='?', player0='?', player1='?', result:str|int='?'):
    first_player = moves[0][3]
    white, black = player0, player1
    if first_player == 1:
        white, black = black, white
        if isinstance(result, int):
            result = -result
    if isinstance(result, int):
        result = "1/2-1/2" if result == 0 else "1-0" if result == 1 else "0-1"
    board = [
        ['R','N','B','Q','K'],
        ['P','P','P','P','P'],
        [' ',' ',' ',' ',' '],
        ['P','P','P','P','P'],
        ['R','N','B','Q','K']
    ]
    if not config['gardner']:
        board = [
            ['R','N','B','Q','K', 'B', 'N', 'R'],
            ['P','P','P','P','P', 'P', 'P', 'P'],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            [' ',' ',' ',' ',' ', ' ', ' ', ' '],
            ['P','P','P','P','P', 'P', 'P', 'P'],
            ['R','N','B','Q','K', 'B', 'N', 'R']
        ]
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fen = "8/8/8/rnbqk3/ppppp3/8/PPPPP3/RNBQK3 w - - 0 1" if config['gardner'] else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # Hack to have gardner on 8x8 board
    game = (
       f'[Event "MCTX Training"]\n'
       f'[Site "aku.ren"]\n'
       f'[Date "{date}"]\n'
       f'[Round "{round}"]\n'
       f'[White "{white}"]\n'
       f'[Black "{black}"]\n'
       f'[Result "{result}"]\n'
       f'[FEN "{fen}"]\n'
   ) + ' '.join([move_pgn(board, m, i, 1-l) for i, (m, t, l, _) in enumerate(moves) if t == 0])
    return game


def main():
    debug = False

    dummy_state = init_fn(jax.random.split(jax.random.PRNGKey(0), 2))
    if config['use_gnn']:
        x = state_to_graph(dummy_state._board, dummy_state.observation, dummy_state.legal_action_mask)
    else:
        x = dummy_state.observation
    variables = model.init(jax.random.PRNGKey(0), x)
    params, batch_stats = variables['params'], variables['batch_stats']
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    config['param_count'] = param_count

    if not debug:
        run = Run(
            repo='aim://localhost:53800',
            experiment='mctx_dev_' + ('gardner' if config['gardner'] else 'chess'),
            capture_terminal_logs=False,
        )
        run.name = humanhash.humanize(run.hash, words=3)
        run["hparams"] = config

    opt_state0 = optimizer.init(params=params)
    (params, batch_stats, opt_state) = jax.device_put_replicated((params, batch_stats, opt_state0), devices)

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
    prev_loss = None
    old_params, old_batch_stats, old_opt_state = None, None, None 
    sample_window = None

    rng_key = jax.random.PRNGKey(42)
    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[logs]}"),
        speed_estimate_period=1000
    ) as progress:
        task_eval = progress.add_task(
            "[green]Evaluating",
            total=(config['n_iter'] + config['eval_interval'] - 1) // config['eval_interval'],
            logs='...',
            start=False
        )
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
        for iteration in range(config['n_iter']):
            st = time.time()
            if iteration % config['eval_interval'] == 0:
                resume_task(progress, task_eval)
                # Evaluation
                rng_key, subkey1 = jax.random.split(rng_key)
                rng_key, subkey2 = jax.random.split(rng_key)
                keys1 = jax.random.split(subkey1, num_devices)
                keys2 = jax.random.split(subkey2, num_devices)
                R1, games1 = evaluate_mcts(keys1, params, batch_stats, True)
                R2, games2 = evaluate_mcts(keys2, params, batch_stats, False)
                R = jnp.concatenate([R1, R2]).reshape((-1,))
                games = jnp.concatenate([games1, games2]) # (#devices, moves, half-moves, data, batch)
                games = games.transpose([0, 4, 1, 2, 3]) \
                             .reshape((games.shape[0] * games.shape[4], games.shape[1] * games.shape[2], games.shape[3]))
                count = [5] * 3
                with open(os.path.join(games_dir, f"{iteration:06d}.pgn"), "w") as f:
                    for r, g in zip(R, games):
                        r_i = int(np.round(r))
                        if count[r_i+1] > 0:
                            count[r_i+1] -= 1
                            print(to_pgn(
                                g,
                                round=iteration,
                                player0=("GNN" if config['use_gnn'] else "CNN")+f"{iteration:03d}",
                                player1="pgx baseline",
                                result=r_i
                            ), file=f)

                # print(R)
                avg_R = R.mean().item()
                if prev_loss is not None and prev_loss - 1 > avg_R:
                    params, batch_stats, opt_state = old_params, old_batch_stats, old_opt_state
                else:
                    prev_loss = avg_R
                    old_params, old_batch_stats, old_opt_state = params, batch_stats, opt_state
                win_rate, draw_rate, lose_rate = map(
                    lambda r: ((R == r).sum() / R.size).item(),
                    [1, 0, -1]
                )
                opt_state = jax.device_put_replicated(opt_state0, devices)
                if not debug:
                    run.track( # type: ignore
                        {
                            "elo_rating": elo_from_results(avg_R, base=1000, max_delta=1000),
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
                    logs=f'win/lose rate (mean R): {win_rate:.3f} / {lose_rate:.3f} ({avg_R:.3f})'
                )
                progress.stop_task(task_eval)

                # Store checkpoints
                params_0, batch_stats_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (params, batch_stats, opt_state))
                with open(os.path.join(models_dir, f"{iteration:06d}.ckpt"), "wb") as f:
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

            # Selfplay
            resume_task(progress, task_gen)
            st = time.time()
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            data: SelfplayOutput = selfplay(keys, params, batch_stats)
            samples: Sample = compute_loss_input(data)
            # jax.profiler.save_device_memory_profile(f"memory{iteration:03d}-selfplay.prof")
            samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
            frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
            samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
            et = time.time()
            hours['selfplay'] += (et - st) / 3600
            progress.update(
                task_gen,
                advance=1,
                logs=f'{frames} frames'
            )
            progress.stop_task(task_gen)
            # jax.profiler.save_device_memory_profile(f"memory{iteration:03d}-deviceget.prof")

            if sample_window is None:
                sample_window = samples
            else:
                sample_window = jax.tree_util.tree_map(
                    lambda x, y: np.concatenate([x, y], axis=0)[:config['window_size']],
                    samples, sample_window
                )
            # jax.profiler.save_device_memory_profile(f"memory{iteration:03d}-window.prof")

            # Training
            resume_task(progress, task_train)
            st = time.time()

            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)

            sample_window = jax.tree_util.tree_map(
                lambda x: x.reshape((num_devices, -1, *x.shape[1:]), order='F'),
                sample_window
            )
            (sample_window, params, batch_stats, opt_state), (policy_losses, value_losses, max_grad) = \
                training(sample_window, params, batch_stats, opt_state, keys)
            sample_window = jax.device_get(sample_window)
            sample_window = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, *x.shape[2:])),
                sample_window
            )
            policy_loss = policy_losses.mean().item()
            value_loss = value_losses.mean().item()
            loss = policy_loss + value_loss
            if not debug:
                run.track( # type: ignore
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

            # Update logs

            if not debug:
                for tp in hours.keys():
                    run.track( # type: ignore
                        {
                            "hours": hours[tp],
                        },
                        context={
                            'subset': tp,
                        },
                        epoch=iteration
                    )
                run.track( # type: ignore
                    {
                        "frames": frames,
                    },
                    epoch=iteration
                )
            progress.update(
                task_train,
                advance=1,
                logs=f'loss (policy + value): {loss:.5f} ({policy_loss:.5f} + {value_loss:.5f})'
            )
            progress.stop_task(task_train)





    # pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=0.5)

if __name__ == "__main__":
    main()
