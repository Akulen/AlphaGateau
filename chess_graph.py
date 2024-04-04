from typing import cast, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import jraph as jraph
import pgx
import pgx.chess as pgc
import pgx.gardner_chess as pgg
from rich.pretty import pprint

def _state_nodes(observation: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    assert(observation.ndim == 3)
    n_row, n_col = observation.shape[:2]
    # cell order is the same as FEN order
    # features = observation.reshape((n_row * n_col, -1))
    # cell order is the same as pgc.Action._from_label
    features = jnp.rot90(observation, k=-1).reshape((n_row * n_col, -1))
    return n_row * n_col, features

def _state_edges( # TODO: add self-edge
    legal_action_mask: jnp.ndarray,
    offset_id: int=1
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert(legal_action_mask.ndim == 1)
    gardner = legal_action_mask.shape[-1] != 4672
    size = 5 if gardner else 8
    _pgc = pgg if gardner else pgc
    # TODO: Missing opponent's moves (pawn moves, castling, ?)
    all_moves = jax.vmap(_pgc.Action._from_label)(
        jnp.arange(legal_action_mask.shape[-1])
    )
    real_moves_id = jnp.where(
        all_moves.to != -1,
        size=455 if gardner else 1858,
        fill_value=-1
    )[0] # The fill_value should never be used
    action_edge_id = jnp.full(legal_action_mask.shape, -1) \
                        .at[real_moves_id] \
                        .set(jnp.arange(real_moves_id.shape[0]))
    all_moves = jax.tree_map(lambda x: x[real_moves_id], all_moves)
    edge_from = all_moves.from_ + offset_id
    edge_to = all_moves.to + offset_id

    n_features = 1+2+4+2*6 # legal, grid offsets, promotions, piece type
    delta_file = (all_moves.to // size) - (all_moves.from_ // size)
    delta_rank = (all_moves.to % size) - (all_moves.from_ % size)
    edge_features = jnp.where(
        (all_moves.to == -1)[:, None].repeat(n_features, axis=-1),
        0,
        jnp.stack([
            # legal
            legal_action_mask[real_moves_id],
            # legal_action_mask,
            # grid offsets
            delta_file,
            delta_rank,
            # promotions
            # doesn't distinguish between pawn promoting to queen and other
            # pieces moving to the 8th rank in a way a pawn could
            (
                  (all_moves.from_ % size == size-2)
                & (all_moves.to % size == size-1)
                & (jnp.abs(all_moves.to // size - all_moves.from_ // size) <= 1)
                & (all_moves.underpromotion == 0)
            ), # queen
            all_moves.underpromotion == 0, # rook
            all_moves.underpromotion == 1, # bishop
            all_moves.underpromotion == 2  # knight
        ] + sum([[ # TODO: add castling
            ( # pawn
                  (jnp.abs(delta_file) <= 1)
                & (
                      (delta_rank == (1 if player else -1))
                    | (
                          (delta_rank == (2 if player else -2))
                        & (all_moves.from_ % size == (1 if player else size-2))
                    ) # TODO: remove this case for gardner chess (no torpedo)
                )
            ),
            ( # knight
                  (jnp.abs(delta_file) == 1) & (jnp.abs(delta_rank) == 2)
                | (jnp.abs(delta_file) == 2) & (jnp.abs(delta_rank) == 1)
            ),
            ( # bishop
                  (jnp.abs(delta_file) == jnp.abs(delta_rank))
                & (all_moves.underpromotion < 0)
            ),
            ( # rook
                  ((jnp.abs(delta_file) == 0) | (jnp.abs(delta_rank) == 0))
                & (all_moves.underpromotion < 0)
            ),
            ( # queen (might be useless as queen == bishop | rook)
                (
                      (jnp.abs(delta_file) == jnp.abs(delta_rank))
                    | ((jnp.abs(delta_file) == 0) | (jnp.abs(delta_rank) == 0))
                )
                & (all_moves.underpromotion < 0)
            ),
            ( # king
                  ((jnp.abs(delta_file) <= 1) & (jnp.abs(delta_rank) <= 1))
                & (all_moves.underpromotion < 0)
            )
        ] for player in range(2)], []), axis=-1)
    )
    return edge_from.shape[0], edge_from, edge_to, edge_features, action_edge_id

def state_to_graph(
    observation: jnp.ndarray,
    legal_action_mask: jnp.ndarray
) -> jraph.GraphsTuple:
    n_nodes, node_features = jax.vmap(_state_nodes)(observation)
    n_nodes = cast(jnp.ndarray, n_nodes)
    node_features = node_features.reshape((-1, node_features.shape[-1]))
    offsets = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.int32),
        n_nodes[:-1]
    ]).cumsum()
    n_edges, moves_from, moves_to, edges_features, action_edge_id = (
        jax.vmap(_state_edges)(
            legal_action_mask,
            offset_id=cast(int, offsets)
        )
    )
    moves_from = moves_from.reshape((-1,)).astype(jnp.int32)
    moves_to = moves_to.reshape((-1,)).astype(jnp.int32)
    edge_offsets = jnp.arange(action_edge_id.shape[0]) * edges_features.shape[1]
    edges_features = edges_features.reshape((-1,) + edges_features.shape[2:])

    edge_offsets = edge_offsets.repeat(action_edge_id.shape[1])
    action_edge_id = action_edge_id.reshape((-1,))
    action_edge_id = jnp.where(
        action_edge_id == -1,
        -1,
        action_edge_id + edge_offsets
    )

    return jraph.GraphsTuple(
        n_node=n_nodes,
        nodes=node_features,
        n_edge=cast(jnp.ndarray, n_edges),
        edges=edges_features,
        senders=moves_from,
        receivers=moves_to,
        globals=action_edge_id
    )

def main():
    env = pgx.make("chess")
    state = env.init(jax.random.PRNGKey(0))
    state = jax.tree_map(lambda x: x[None], state)

    x = jax.jit(state_to_graph)(state.observation, state.legal_action_mask)
    # pprint(x)
    # pprint(x.n_node)
    # pprint(x.nodes.shape)
    # pprint(np.array(list(" PNBRQKpnbrqk"))[jnp.rot90((x.nodes[1:,:12] * jnp.arange(1, 13)).sum(axis=-1).reshape((-1,8,8)), axes=(1,2)).astype(jnp.int32)])
    print('   l Δx Δy pq pr pb pn  p  n  b  r  q  k  P  N  B  R  Q  K ')
    print(x.edges[jnp.where(state.legal_action_mask.reshape((-1,)))])
    pprint(jnp.where(state.legal_action_mask.reshape((-1,))))

    # states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    # x = jax.jit(state_to_graph)(states.observation, states.legal_action_mask)
    # pprint((x.n_node, x.n_edge, x.nodes.shape, x.edges.shape))
    # pprint(np.array(list(" PNBRQKpnbrqk"))[jnp.rot90((x.nodes[1:,:12] * jnp.arange(1, 13)).sum(axis=-1).reshape((-1,8,8)), axes=(1,2)).astype(jnp.int32)])
    # print('   l Δx Δy pq pr pb pn  p  n  b  r  q  k  P  N  B  R  Q  K ')
    # print(x.edges[jnp.where(state.legal_action_mask.reshape((-1)))])
    # print(x.receivers[jnp.where(states.legal_action_mask.reshape((-1,)))])
    # pprint((x.receivers.shape, x.receivers.min(), x.receivers.max()))

if __name__ == "__main__":
    main()
