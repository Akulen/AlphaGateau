import jax
import jax.numpy as jnp
import jraph
from jraph import NodeFeatures, AggregateEdgesToNodesFn
import jraph._src.utils as jutils
from typing import Callable, NamedTuple, Optional

# TODO: use something more generic
import pgx.gardner_chess as pgc

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
    aggregate_nodes_fn: AggregateEdgesToNodesFn = jutils.segment_sum, # type: ignore
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

def _state_edges_moves(state, n_nodes, n_actions):
    batch_size = state.observation.shape[0]
    edge_mask = state.legal_action_mask
    moves_from, moves_to, moves_underpromotion = action_to_edge(
        jnp.tile(jnp.arange(n_actions), (batch_size, 1)),
        edge_mask
    )

    batch_mask = jnp.tile(jnp.arange(batch_size) * n_nodes, (n_actions, 1)).transpose()
    moves_from = (moves_from + batch_mask).reshape(-1)
    moves_to = (moves_to + batch_mask).reshape(-1)
    moves_underpromotion = moves_underpromotion.reshape(-1)

    n_edges = jnp.full(batch_size, n_actions)

    return n_edges, moves_from.astype(jnp.int32), moves_to.astype(jnp.int32), moves_underpromotion

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

def state_to_graph(state, n_actions):
    node_types, n_nodes = _state_nodes(state)
    n_edges, moves_from, moves_to, moves_underpromotion = _state_edges_moves(state, n_nodes[0], n_actions)
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
