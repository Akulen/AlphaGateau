from functools import partial
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Union, Callable, Tuple
import jax
import jax.numpy as jnp
import jraph
import pgx.chess as pgc
import pgx.gardner_chess as pgg

ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# Extended version of jraph.GraphsTuple to include additional fields
class MultiGraphsTuple(NamedTuple):
    # Original fields
    nodes: Optional[ArrayTree]
    edges_actions: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    receivers: Optional[jnp.ndarray]
    senders: Optional[jnp.ndarray]
    globals: Optional[ArrayTree]
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
    n_edge_active: Optional[jnp.ndarray] = None
    active_senders: Optional[jnp.ndarray] = None
    active_receivers: Optional[jnp.ndarray] = None
    n_edge_passive: Optional[jnp.ndarray] = None
    passive_senders: Optional[jnp.ndarray] = None
    passive_receivers: Optional[jnp.ndarray] = None

def _state_nodes(state_obs, use_embedding=True):
    batch_size = state_obs.shape[0]
    if use_embedding:
        node_types_grid = (state_obs[:,::-1,:,:12] * jnp.arange(1, 13)).sum(axis=-1).astype(jnp.int32)[:,:,:,None]
    else:
        node_types_grid = state_obs # (batch, col, row, inner_dim)
    node_types = node_types_grid.reshape(
        (*node_types_grid.shape[:-3], node_types_grid.shape[-3] * node_types_grid.shape[-2], -1),
    order='F')
    n_nodes = jnp.full(batch_size, node_types.shape[-2]+1)
    node_types = jnp.concatenate((
        -jnp.ones((batch_size, 1, node_types.shape[-1]), dtype=jnp.int32), # First one is a dummy node
        node_types,
    ), axis=-2)
    node_types = node_types.reshape((-1, node_types.shape[-1]))
    return node_types, n_nodes

def action_to_edge(actions, gardner=False): # , action_mask):
    moves = jax.vmap((pgg if gardner else pgc).Action._from_label)(actions.flatten())
    moves = jax.tree_map(lambda x: jnp.reshape(x, actions.shape), moves)
    moves_from = jnp.where(
        actions >= 0, moves.from_+1, jnp.int32(0)
    )
    moves_to = jnp.where(
        actions >= 0, moves.to+1, jnp.int32(0)
    )
    moves_underpromotion = jnp.where(
        actions >= 0, moves.underpromotion, jnp.int32(-1)
    )
    return moves_from, moves_to, moves_underpromotion

def _state_edges_moves(state_obs, state_lam, n_nodes):
    MAX_LEGAL_ACTIONS = 256
    batch_size = state_obs.shape[0]
    edge_mask = state_lam
    edges = jax.vmap(partial(jnp.argwhere, size=MAX_LEGAL_ACTIONS, fill_value=-1_000_000))(edge_mask)
    edges = edges.reshape(edges.shape[:-1])
    moves_from, moves_to, moves_underpromotion = action_to_edge(edges, state_lam.shape[-1] != 4672)
    #     jnp.tile(jnp.arange(n_actions), (batch_size, 1)),
    #     edge_mask
    # )

    batch_mask = jnp.tile(jnp.arange(batch_size) * n_nodes, (MAX_LEGAL_ACTIONS, 1)).transpose()
    moves_from = (moves_from + batch_mask).reshape(-1)
    moves_to = (moves_to + batch_mask).reshape(-1)
    moves_underpromotion = moves_underpromotion.reshape(-1)

    n_edges = jnp.full(batch_size, MAX_LEGAL_ACTIONS)

    return n_edges, edges, moves_from.astype(jnp.int32), moves_to.astype(jnp.int32), moves_underpromotion

def _state_edges_grid(state_obs, n_nodes):
    batch_size = state_obs.shape[0]
    n_row, n_col = state_obs.shape[-3:-1]

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

def _flip_gen(a, do, size=8):
    from_r, from_c = a.from_ % size, a.from_ // size
    to_r, to_c = a.to % size, a.to // size
    from_r = jnp.where(do, size-1-from_r, from_r)
    to_r = jnp.where(do, size-1-to_r, to_r)
    return pgc.Action(
        from_=size*from_c + from_r,
        to=size*to_c + to_r
    )

def _flip_chess(a: pgc.Action, do):
    return _flip_gen(a, do, 8)

def _flip_gardner(a: pgg.Action, do):
    return _flip_gen(a, do, 5)

def _is_pseudo_legal_chess(state_board, a: pgc.Action, side):
    piece = state_board[a.from_] * side
    a2 = _flip_chess(a, side == -1)
    ok = (piece >= 0) # & (state._board[a.to] <= 0)
    ok &= (pgc.CAN_MOVE[piece, a2.from_] == a2.to).any()
    between_ixs = pgc.BETWEEN[a.from_, a.to]
    ok &= ((between_ixs < 0) | (state_board[between_ixs] == pgc.EMPTY)).all()
    # filter pawn move
    ok &= ~(
        (piece == pgc.PAWN)
        & (a2.to // 8 == a2.from_ // 8)
        & (state_board[a.to] != 0)
    )
    ok &= ~(
        (piece == pgc.PAWN)
        & (a2.to // 8 != a2.from_ // 8)
        & (state_board[a.to] == 0)
    )
    return (a.to >= 0) & ok

def _is_pseudo_legal_gardner(state_board, a: pgg.Action, side):
    piece = state_board[a.from_] * side
    a2 = _flip_gardner(a, side == -1)
    ok = (piece >= 0) # & (state._board[a.to] <= 0)
    ok &= (pgg.CAN_MOVE[piece, a2.from_] == a2.to).any()
    between_ixs = pgg.BETWEEN[a.from_, a.to]
    ok &= ((between_ixs < 0) | (state_board[between_ixs] == pgg.EMPTY)).all()
    # filter pawn move
    ok &= ~(
        (piece == pgg.PAWN)
        & (a2.to // 5 == a2.from_ // 5)
        & (state_board[a.to] != 0)
    )
    ok &= ~(
        (piece == pgg.PAWN)
        & (a2.to // 5 != a2.from_ // 5)
        & (state_board[a.to] == 0)
    )
    return (a.to >= 0) & ok

@partial(jax.vmap, in_axes=(None, 0, 0))
def _can_move_chess(state_board, from_, to):
    a = pgc.Action(from_=from_, to=to)
    side = jnp.where(state_board[from_] > 0, 1, -1)
    return (from_ != -1) & _is_pseudo_legal_chess(state_board, a, side=side)

@partial(jax.vmap, in_axes=(None, 0, 0))
def _can_move_gardner(state_board, from_, to):
    a = pgg.Action(from_=from_, to=to)
    side = jnp.where(state_board[from_] > 0, 1, -1)
    return (from_ != -1) & _is_pseudo_legal_gardner(state_board, a, side=side)

@partial(jax.vmap, in_axes=(0, None))
def _state_edges_vision(state_board, side) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_nodes = state_board.shape[0]
    edges_from = jnp.arange(n_nodes).repeat(n_nodes)
    edges_to = jnp.tile(jnp.arange(n_nodes), n_nodes)
    _can_move = _can_move_chess if state_board.shape[-1] == 8 else _can_move_gardner
    edges_mask = _can_move(state_board, edges_from, edges_to) & state_board[edges_from] * side > 0

    edges_from = jnp.where(edges_mask, edges_from+1, jnp.int32(0))
    edges_to = jnp.where(edges_mask, edges_to+1, jnp.int32(0))
    return n_nodes * n_nodes, edges_from, edges_to # type: ignore

def state_to_graph(state_board, state_obs, state_lam, use_embedding=True):
    node_types, n_nodes = _state_nodes(state_obs, use_embedding=use_embedding)
    n_edges, edges_actions, moves_from, moves_to, moves_underpromotion = _state_edges_moves(state_obs, state_lam, n_nodes[0])
    n_edges_grid, grid_moves_from, grid_moves_to = _state_edges_grid(state_obs, n_nodes[0])
    n_edges_active, active_moves_from, active_moves_to = _state_edges_vision(state_board, 1)
    n_edges_passive, passive_moves_from, passive_moves_to = _state_edges_vision(state_board, -1)

    return MultiGraphsTuple(
        n_node=n_nodes,
        nodes=node_types,
        n_edge=n_edges,
        senders=moves_from,
        receivers=moves_to,
        edges_actions=edges_actions,
        edges=moves_underpromotion, # .reshape(moves_underpromotion.shape + (1,)),
        n_edge_grid=n_edges_grid,
        grid_senders=grid_moves_from,
        grid_receivers=grid_moves_to,
        n_edge_active=n_edges_active.reshape((-1,)),
        active_senders=active_moves_from.reshape((-1,)),
        active_receivers=active_moves_to.reshape((-1,)),
        n_edge_passive=n_edges_passive.reshape((-1,)),
        passive_senders=passive_moves_from.reshape((-1,)),
        passive_receivers=passive_moves_to.reshape((-1,)),
        globals=None
    )

def GraphConvolution(
    update_node_fn: Callable[[jraph.NodeFeatures], jraph.NodeFeatures],
    aggregate_nodes_fn: jraph.AggregateEdgesToNodesFn = jraph.segment_sum, # type: ignore
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
        nodes, senders, receivers = graph.nodes, graph.senders, graph.receivers

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
            count_edges = lambda x: jraph.segment_sum(
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
