import jax
jax.config.update('jax_platform_name', 'cpu')
# before execute any computation / allocation
print(jax.numpy.ones(3).device()) # TFRT_CPU_0
import flax.linen as nn
import jax.numpy as jnp
import jraph
import jraph._src.utils as jutils
# import mctx
import optax
import pgx
import pgx.gardner_chess as pgc
from typing import Callable, NamedTuple, Optional
from jraph import NodeFeatures, AggregateEdgesToNodesFn
from functools import partial

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

class MultiGraphsTuple(NamedTuple):
    # Original fields
    nodes: Optional[jraph.ArrayTree]
    edges: Optional[jraph.ArrayTree]
    receivers: Optional[jnp.ndarray]
    senders: Optional[jnp.ndarray]
    globals: Optional[jraph.ArrayTree]
    n_node: jnp.ndarray
    n_edge: jnp.ndarray

    # Additional fields
    n_edge_grid: Optional[jnp.ndarray]
    grid_receivers: Optional[jnp.ndarray] = None
    grid_senders: Optional[jnp.ndarray] = None
    attacks_receivers: Optional[jnp.ndarray] = None
    attacks_senders: Optional[jnp.ndarray] = None
    defends_receivers: Optional[jnp.ndarray] = None
    defends_senders: Optional[jnp.ndarray] = None

def GraphConvolution(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    aggregate_nodes_fn: AggregateEdgesToNodesFn = jutils.segment_sum,
    add_self_edges: bool = False,
    symmetric_normalization: bool = True):
    """Returns a method that applies a Graph Convolution layer.

    Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,

    NOTE: This implementation does not add an activation after aggregation.
    If you are stacking layers, you may want to add an activation between
    each layer.

    Args:
        update_node_fn: function used to update the nodes. In the paper a single
            layer MLP is used.
        aggregate_nodes_fn: function used to aggregates the sender nodes.
        add_self_edges: whether to add self edges to nodes in the graph as in the
            paper definition of GCN. Defaults to False.
        symmetric_normalization: whether to use symmetric normalization. Defaults
            to True. Note that to replicate the fomula of the linked paper, the
            adjacency matrix must be symmetric. If the adjacency matrix is not
            symmetric the data is prenormalised by the sender degree matrix and post
            normalised by the receiver degree matrix.

    Returns:
        A method that applies a Graph Convolution layer.
    """
    def _ApplyGCN(graph):
        """Applies a Graph Convolution layer."""
        nodes, _, receivers, senders, _, _, _, _, _, _, _, _, _, _ = graph

        # First pass nodes through the node updater.
        nodes = update_node_fn(nodes)
        # Equivalent to jnp.sum(n_node), but jittable
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        if add_self_edges:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            # In principle, a `GraphsTuple` should partition by n_edge, but in
            # this case it is not required since a GCN is agnostic to whether
            # the `GraphsTuple` is a batch of graphs or a single large graph.
            conv_receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
            conv_senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
        else:
            conv_senders = senders
            conv_receivers = receivers

        # pylint: disable=g-long-lambda
        if symmetric_normalization:
            # Calculate the normalization values.
            count_edges = lambda x: jutils.segment_sum(
                jnp.ones_like(conv_senders), x, total_num_nodes
            )
            sender_degree = count_edges(conv_senders)
            receiver_degree = count_edges(conv_receivers)

            # Pre normalize by sqrt sender degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = jax.tree_util.tree_map(
                lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
                nodes,
            )
            # Aggregate the pre normalized nodes.
            nodes = jax.tree_util.tree_map(
                lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers, total_num_nodes),
                nodes
            )
            # Post normalize by sqrt receiver degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = jax.tree_util.tree_map(
                lambda x: (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]),
                nodes,
            )
        else:
            nodes = jax.tree_util.tree_map(
                lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers, total_num_nodes),
                nodes
            )
        # pylint: enable=g-long-lambda
        return graph._replace(nodes=nodes)

    return _ApplyGCN

class EGNN(nn.Module):
    out_dim: int = 128
    @nn.compact
    def __call__(self, graph):
        x = jnp.concatenate([
            GraphConvolution(nn.Dense(self.out_dim), add_self_edges=True)(graph).nodes,
            GraphConvolution(nn.Dense(self.out_dim), add_self_edges=True)(graph._replace(
                senders=graph.grid_senders,
                receivers=graph.grid_receivers
            )).nodes
        ], axis=-1)
        x = jax.nn.relu(nn.Dense(self.out_dim)(x))
        return graph._replace(nodes=x)

class NodeDot(nn.Module):
    @nn.compact
    def __call__(self, x, senders, receivers):
        return jnp.sum(x[senders] * x[receivers], axis=-1)

class EdgeNet(nn.Module):
    inner_size: int = 128
    n_gnn_layers: int = 7
    n_eval_layers: int = 5

    @nn.compact
    def __call__(self, graphs, training=False):
        graphs = graphs._replace(nodes=nn.Embed(num_embeddings=13, features=self.inner_size)(graphs.nodes))

        for _ in range(self.n_gnn_layers):
            graphs = EGNN(out_dim=self.inner_size)(graphs)

        logits = nn.Dense(self.inner_size)(graphs.nodes)
        logits = NodeDot()(logits, graphs.senders, graphs.receivers)

        n_partitions = len(graphs.n_node)
        segment_ids = jnp.repeat(
            jnp.arange(n_partitions),
            graphs.n_node,
            axis=0,
            total_repeat_length=graphs.nodes.shape[0]
        )
        node_mask = jnp.where(
            jnp.arange(graphs.nodes.shape[0]) % graphs.n_node[0] == 0,
            jnp.int32(0),
            jnp.int32(1)
        )
        x = graphs.nodes * jnp.tile(node_mask, (self.inner_size, 1)).transpose()
        v = jraph.segment_sum(x, segment_ids, graphs.n_node.shape[0])
        v /= jnp.tile(graphs.n_node - 1, (self.inner_size, 1)).transpose()
        for _ in range(self.n_eval_layers):
            v = nn.relu(nn.Dense(self.inner_size)(v))
        v = nn.tanh(nn.Dense(1)(v))

        return logits, v

        # x, edge_index, attacks_edge_index, defends_edge_index, grid_edge_index =\
        #     data.x, data.edge_index, data.attacks_edge_index, data.defends_edge_index, data.grid_edge_index

        # x = self.piece_emb(x)

        # for move_gnn, grid_gnn, attacks_gnn, defends_gnn, reduce in zip(self.gnn_moves, self.gnn_grid, self.gnn_attacks, self.gnn_defends, self.reduce):
        #     x1 = move_gnn(x, edge_index)
        #     x2 = grid_gnn(x, grid_edge_index)
        #     x3 = attacks_gnn(x, attacks_edge_index)
        #     x4 = defends_gnn(x, defends_edge_index)
        #     x = torch.cat((x1, x2, x3, x4), dim=1)
        #     x = reduce(x)
        #     x = F.relu(x)

        # x_eval = self.pool(x, batch=data.batch)
        # for eval_layer in self.eval_layers:
        #     x_eval = F.relu(x_eval)
        #     x_eval = eval_layer(x_eval)
        # x_eval = F.tanh(x_eval)

        # x_policy = self.policy_output(x)
        # U, V = edge_index
        # policy = (x_policy[U] * x_policy[V]).sum(dim=-1)

        # return self.evaluation_output(x_eval).squeeze(dim=-1), policy
        # raise NotImplementedError

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

@jax.jit
def _state_nodes(state):
    batch_size = state.observation.shape[0]
    node_types_grid = (state.observation[:,::-1,:,:12] * jnp.arange(1, 13)).sum(axis=-1).astype(jnp.int32)
    node_types = node_types_grid.reshape(
        node_types_grid.shape[:-2] + (node_types_grid.shape[-2] * node_types_grid.shape[-1],),
    order='F')
    n_nodes = jnp.full(batch_size, node_types.shape[-1]+1)
    node_types = jnp.concatenate((
        -jnp.ones((batch_size, 1), dtype=jnp.int32), # First one is a dummy node
        node_types,
    ), axis=-1)
    node_types = node_types.reshape((-1,))
    return node_types, n_nodes

@jax.jit
def action_to_edge(actions, action_mask):
    moves_from, moves_plane = actions // 49, actions % 49
    moves_to = jnp.where(
        action_mask, pgc.TO_MAP[moves_from, moves_plane]+1, jnp.int32(0)
    )
    moves_from = jnp.where(
        action_mask, moves_from+1, jnp.int32(0)
    )
    moves_underpromotion = jnp.where(
        action_mask & (moves_plane < 9), jnp.int32(moves_plane // 3), jnp.int32(-1)
    )
    return moves_from, moves_to, moves_underpromotion

@jax.jit
def _state_edges_moves(state, n_nodes):
    batch_size = state.observation.shape[0]
    n_edges = env.num_actions
    edge_mask = state.legal_action_mask
    moves_from, moves_to, moves_underpromotion = action_to_edge(
        jnp.tile(jnp.arange(n_edges), (batch_size, 1)),
        edge_mask
    )

    batch_mask = jnp.tile(jnp.arange(batch_size) * n_nodes, (n_edges, 1)).transpose()
    moves_from = (moves_from + batch_mask).reshape(-1)
    moves_to = (moves_to + batch_mask).reshape(-1)
    moves_underpromotion = moves_underpromotion.reshape(-1)

    n_edges = jnp.full(batch_size, n_edges)

    return n_edges, moves_from.astype(jnp.int32), moves_to.astype(jnp.int32), moves_underpromotion

@jax.jit
def _state_edges_grid(state, n_nodes):
    batch_size = state.observation.shape[0]
    n_row, n_col = state.observation.shape[-3:-1]

    grid = jnp.arange(1, 1 + n_row * n_col).reshape((n_row, n_col), order='F')
    edges = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            from_cells = grid[max(0, -i):min(n_row, n_row-i), max(0, -j):min(n_col, n_col-j)]
            to_cells = grid[max(0, i):min(n_row, n_row+i), max(0, j):min(n_col, n_col+j)]
            edges.append(jnp.stack((from_cells, to_cells)).reshape((2, -1)))
    edges = jnp.concatenate(edges, axis=1)
    n_edges = edges.shape[1]

    batch_mask = jnp.arange(batch_size).repeat(n_edges)
    edges = jnp.tile(edges, batch_size)
    moves_from = edges[0] + batch_mask * n_nodes
    moves_to = edges[1] + batch_mask * n_nodes

    n_edges = jnp.full(batch_size, n_edges)

    return n_edges, moves_from, moves_to

@jax.jit
def state_to_graph(state):
    node_types, n_nodes = _state_nodes(state)
    n_edges, moves_from, moves_to, moves_underpromotion = _state_edges_moves(state, n_nodes[0])
    n_edges_grid, grid_moves_from, grid_moves_to = _state_edges_grid(state, n_nodes[0])

    return MultiGraphsTuple(
        n_node=n_nodes,
        nodes=node_types,
        n_edge=n_edges,
        senders=moves_from,
        receivers=moves_to,
        edges=moves_underpromotion, # .reshape(moves_underpromotion.shape + (1,)),
        n_edge_grid=n_edges_grid,
        grid_senders=grid_moves_from,
        grid_receivers=grid_moves_to,
        globals=None
    )

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
    # print(state_to_graph(state))7350

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
    # print(state_to_graph(state, batch_size))

    # states.append(state)
    # while not (state.terminated | state.truncated).all():
    #     logits, value = baseline(state.observation)
    #     action = logits.argmax(-1)
    #     state = step_fn(state, action)
    #     states.append(state)

    # pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=0.5)

