from typing import cast, overload, Callable, Literal, Mapping, NamedTuple, Tuple
from functools import partial

import chex
from rich.pretty import pprint
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from jpyger import GraphConvolution, state_to_graph
import chess_graph as cg


def GCN(
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
    def _ApplyGCN(nodes, senders, receivers):
        """Applies a Graph Convolution layer."""

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
        return nodes

    return _ApplyGCN

class EGNN(nn.Module):
    out_dim: int = 128
    @nn.compact
    def __call__(self, *args, graph, **kwargs):
        x = jnp.concatenate([
            GCN(nn.Dense(self.out_dim), add_self_edges=True)(graph.nodes, graph.senders, graph.receivers),
            GCN(nn.Dense(self.out_dim), add_self_edges=True)(graph.nodes, graph.receivers, graph.senders),
            GCN(nn.Dense(self.out_dim), add_self_edges=True)(graph.nodes, graph.grid_senders, graph.grid_receivers),
            GCN(nn.Dense(self.out_dim), add_self_edges=True)(graph.nodes, graph.active_senders, graph.active_receivers),
            GCN(nn.Dense(self.out_dim), add_self_edges=True)(graph.nodes, graph.passive_senders, graph.passive_receivers),
        ], axis=-1)
        x = jax.nn.relu(nn.Dense(self.out_dim)(x))
        return graph._replace(nodes=x)

class EGNN2(nn.Module):
    out_dim: int = 128
    @nn.compact
    def __call__(self, *args, graph, training=False, **kwargs):
        i = graph.nodes
        x = nn.BatchNorm(momentum=0.9)(graph.nodes, use_running_average=not training)
        x = jax.nn.relu(x)
        x = jnp.concatenate([
            GraphConvolution(nn.Dense(self.out_dim), add_self_edges=True)(graph).nodes,
            GraphConvolution(nn.Dense(self.out_dim), add_self_edges=True)(graph._replace(
                senders=graph.receivers,
                receivers=graph.senders
            )).nodes,
            GraphConvolution(nn.Dense(self.out_dim), add_self_edges=True)(graph._replace(
                senders=graph.grid_senders,
                receivers=graph.grid_receivers
            )).nodes,
            GraphConvolution(nn.Dense(self.out_dim), add_self_edges=True)(graph._replace(
                senders=graph.active_senders,
                receivers=graph.active_receivers
            )).nodes,
            GraphConvolution(nn.Dense(self.out_dim), add_self_edges=True)(graph._replace(
                senders=graph.passive_senders,
                receivers=graph.passive_receivers
            )).nodes,
        ], axis=-1)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return graph._replace(nodes=x+i)


class NodeDot(nn.Module):
    @nn.compact
    def __call__(self, *args, x, senders, receivers, **kwargs):
        return jnp.sum(x[senders] * x[receivers], axis=-1)


class NodeDotV2(nn.Module):
    inner_size: int = 128

    @nn.compact
    def __call__(self, *args, x, senders, receivers, edge_feature, **kwargs):
        u, v = x[senders], x[receivers]
        edge_embed = nn.Embed(num_embeddings=4, features=self.inner_size)(edge_feature)
        return jnp.sum(nn.Dense(self.inner_size)(u) * nn.Dense(self.inner_size)(v) * edge_embed, axis=-1)


class AttentionPooling(nn.Module):
    @nn.compact
    def __call__(
        self,
        *args,
        x: jnp.ndarray,
        segment_ids: jnp.ndarray,
        mask: jnp.ndarray | None=None,
        num_segments: int | None=None,
        **kwargs
    ):
        if mask is None:
            segment_ids_masked = segment_ids
        else:
            segment_ids_masked = jnp.where(mask, segment_ids, -1)
        att = cast(jnp.ndarray, jraph.segment_softmax(
            nn.Dense(1)(x).squeeze(-1),
            segment_ids_masked,
            num_segments
        ))
        if mask is not None:
            att = att * mask
        att = jnp.tile(att, (x.shape[1], 1)).transpose()
        return jraph.segment_sum(x * att, segment_ids, num_segments)


class EdgeNet(nn.Module):
    n_actions: int
    inner_size: int = 128
    n_gnn_layers: int = 8
    n_eval_layers: int = 5
    dot_v2: bool = True
    use_embedding: bool = True
    attention_pooling: bool = True

    mix_edge_node: bool = False
    add_features: bool = True

    @nn.compact
    def __call__(self, *args, graphs, training=False, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.use_embedding:
            graphs = graphs._replace(nodes=nn.Embed(num_embeddings=13, features=self.inner_size)(graphs.nodes.reshape((-1)).astype(jnp.int32)))

        for _ in range(self.n_gnn_layers):
            graphs = EGNN2(out_dim=self.inner_size)(graph=graphs, training=training)

        x = nn.BatchNorm(momentum=0.9)(graphs.nodes, use_running_average=not training)
        x = jax.nn.relu(x)

        node_logits = nn.Dense(self.inner_size)(x)
        node_logits = nn.BatchNorm(momentum=0.9)(node_logits, use_running_average=not training)
        node_logits = nn.relu(node_logits)
        # node_logits = nn.Dense(self.inner_size)(node_logits)
        dot = NodeDotV2(self.inner_size) if self.dot_v2 else NodeDot()
        logits = dot(x=node_logits, senders=graphs.senders, receivers=graphs.receivers, edge_feature=graphs.edges)
        logits = logits.reshape((graphs.edges_actions.shape[0], -1))
        logits = jax.vmap(
            lambda a, ind, x: a.at[ind].set(x)
        )(logits.min() * jnp.ones((graphs.edges_actions.shape[0], self.n_actions)), graphs.edges_actions, logits)

        # global_logits = nn.Dense(graphs.senders.shape[-1])(node_logits)
        # logits = jnp.concatenate([edge_logits, global_logits], axis=-1)
        # logits = nn.BatchNorm(momentum=0.9)(logits, use_running_average=not training)
        # logits = nn.relu(logits)
        # logits = nn.Dense(graphs.senders.shape[-1])(logits)

        n_partitions = len(graphs.n_node)
        segment_ids = jnp.repeat(
            jnp.arange(n_partitions),
            graphs.n_node,
            axis=0,
            total_repeat_length=x.shape[0]
        )
        node_mask = jnp.where(
            jnp.arange(x.shape[0]) % graphs.n_node[0] == 0,
            jnp.int32(0),
            jnp.int32(1)
        )
        v = nn.Dense(self.inner_size)(x)
        v = nn.BatchNorm(momentum=0.9)(v, use_running_average=not training)
        v = jax.nn.relu(v)
        if self.attention_pooling:
            v = AttentionPooling()(x=v, segment_ids=segment_ids, mask=node_mask, num_segments=graphs.n_node.shape[0])
        else:
            # Mean Pooling
            v = v * jnp.tile(node_mask, (self.inner_size, 1)).transpose()
            v = jraph.segment_sum(v, segment_ids, graphs.n_node.shape[0])
            v /= jnp.tile(graphs.n_node - 1, (self.inner_size, 1)).transpose()
        v = jax.nn.relu(v) # Probably useless after attention pooling
        # for _ in range(self.n_eval_layers):
        #     v = nn.relu(nn.Dense(self.inner_size)(v))
        v = nn.Dense(1)(v)
        v = nn.tanh(v)

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


class BNR(nn.Module):
    momentum: float = 0.9
    @nn.compact
    def __call__(
        self,
        *args,
        x,
        training: bool=False,
        **kwargs
    ):
        training=False
        x = nn.BatchNorm(momentum=self.momentum)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        return x


class GATEAU(nn.Module):
    out_dim: int = 128
    mix_edge_node: bool = False
    add_features: bool = True
    @nn.compact
    def __call__(
        self,
        *args,
        graph: jraph.GraphsTuple,
        **kwargs
    ) -> jraph.GraphsTuple:
        try:
            sum_n_node = graph.nodes.shape[0] # type: ignore
        except IndexError:
            raise IndexError('GAT requires node features')

        node_features = cast(jnp.ndarray, graph.nodes)
        edge_features = cast(jnp.ndarray, graph.edges)

        sent_attributes_1 = nn.Dense(self.out_dim)(node_features)[graph.senders]
        sent_attributes_2 = node_features[graph.senders]
        received_attributes = nn.Dense(self.out_dim)(
            node_features
        )[graph.receivers]
        edge_features = nn.Dense(self.out_dim)(edge_features)

        if self.add_features:
            edge_features = (
                  sent_attributes_1
                + edge_features
                + received_attributes
            )
        else:
            edge_features = (
                  sent_attributes_1
                * edge_features
                * received_attributes
            )

        attention_coeffs = nn.Dense(1)(edge_features)
        attention_coeffs = nn.leaky_relu(attention_coeffs)
        attention_weights = jraph.segment_softmax(
            attention_coeffs,
            segment_ids=cast(jnp.ndarray, graph.receivers),
            num_segments=sum_n_node
        )

        if self.mix_edge_node:
            if self.add_features:
                message = sent_attributes_2 + edge_features
            else:
                message = sent_attributes_2 * edge_features
        else:
            message = sent_attributes_2
        message = nn.Dense(self.out_dim)(message)
        message = attention_weights * message
        node_features = jraph.segment_sum(
            message,
            segment_ids=cast(jnp.ndarray, graph.receivers),
            num_segments=sum_n_node
        )

        return graph._replace(
            nodes=node_features,
            edges=edge_features
        )


class EGNN3(nn.Module):
    out_dim: int = 128
    mix_edge_node: bool = False
    add_features: bool = True
    @nn.compact
    def __call__(
        self,
        *args,
        graph: jraph.GraphsTuple,
        training: bool=False,
        **kwargs
    ) -> jraph.GraphsTuple:
        i, j = map(partial(cast, jraph.ArrayTree), (graph.nodes, graph.edges))
        graph = GATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features
        )(
            graph=graph._replace(
                nodes=BNR()(x=graph.nodes, training=training),
                edges=BNR()(x=graph.edges, training=training)
            )
        )
        graph = GATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features
        )(
            graph=graph._replace(
                nodes=BNR()(x=graph.nodes, training=training),
                edges=BNR()(x=graph.edges, training=training)
            )
        )
        return graph._replace(nodes=graph.nodes+i, edges=graph.edges+j)


class EdgeNet2(nn.Module):
    n_actions: int
    inner_size: int = 128
    n_gnn_layers: int = 5
    attention_pooling: bool = True
    mix_edge_node: bool = False
    add_features: bool = True

    dot_v2: bool = True
    use_embedding: bool = True

    @nn.compact
    def __call__(
        self,
        *args,
        graphs: jraph.GraphsTuple,
        training: bool=False,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        graphs = graphs._replace(
            nodes=nn.Dense(self.inner_size)(graphs.nodes),
            edges=nn.Dense(self.inner_size)(graphs.edges)
        )

        for _ in range(self.n_gnn_layers):
            graphs = EGNN3(
                out_dim=self.inner_size,
                mix_edge_node=self.mix_edge_node,
                add_features=self.add_features
            )(
                graph=graphs,
                training=training
            )

        # TODO: merge node and edge features from all layers
        x = BNR()(x=graphs.nodes, training=training) # type: ignore
        y = BNR()(x=graphs.edges, training=training)

        logits = nn.Dense(self.inner_size)(y)
        logits = BNR()(x=logits, training=training)
        logits = nn.Dense(1)(logits).squeeze()
        logits = logits[graphs.globals]
        # logits = logits - jax.segment_max TODO!!!
        # logits = jnp.where(
        #     graphs.globals != -1,
        #     logits,
        #     jnp.finfo(logits.dtype).min
        # ) # avoid putting fake gradient on logits[-1] used as filling

        n_partitions = len(graphs.n_node)
        segment_ids = jnp.repeat(
            jnp.arange(n_partitions),
            graphs.n_node,
            axis=0,
            total_repeat_length=x.shape[0]
        )
        v = nn.Dense(self.inner_size)(x)
        v = nn.BatchNorm(momentum=0.9)(v, use_running_average=not training)
        v = jax.nn.relu(v)
        if self.attention_pooling:
            v = AttentionPooling()(
                x=v,
                segment_ids=segment_ids,
                num_segments=graphs.n_node.shape[0]
            )
        else:
            raise DeprecationWarning
            # Mean Pooling
            # v = v * jnp.tile(node_mask, (self.inner_size, 1)).transpose()
            # v = jraph.segment_sum(v, segment_ids, graphs.n_node.shape[0])
            # v /= jnp.tile(graphs.n_node - 1, (self.inner_size, 1)).transpose()
        v = v
        v = jax.nn.relu(v) # Probably useless after attention pooling
        v = nn.Dense(1)(v)
        v = nn.tanh(v)

        return logits, v


class BlockV1(nn.Module):
    num_channels: int
    name: str | None = "BlockV1"

    @nn.compact
    def __call__(self, *args, x, training, **kwargs):
        i = x
        x = nn.Conv(self.num_channels, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        return jax.nn.relu(x + i)


class BlockV2(nn.Module):
    num_channels: int
    name: str | None = "BlockV2"

    @nn.compact
    def __call__(self, *args, x, training, **kwargs):
        i = x
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3))(x)
        return x + i


class AZNet(nn.Module):
    """AlphaZero NN architecture."""
    num_actions: int
    num_channels: int = 64
    num_blocks: int = 5
    resnet_v2: bool = True
    resnet_cls = BlockV2
    name: str | None = "az_net"


    @nn.compact
    def __call__(self, *args, x, training=False, **kwargs):
        # if self.resnet_cls is None:
        #     self.resnet_cls = BlockV2 if self.resnet_v2 else BlockV1

        x = x.astype(jnp.float32)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3))(x)

        if not self.resnet_v2:
            x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
            x = jax.nn.relu(x)

        for i in range(self.num_blocks):
            x = self.resnet_cls(num_channels=self.num_channels, name=f"block_{i}")(
                x=x, training=training
            )

        if self.resnet_v2:
            x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
            x = jax.nn.relu(x)

        # policy head
        logits = nn.Conv(features=2, kernel_size=(1, 1))(x)
        logits = nn.BatchNorm(momentum=0.9)(logits, use_running_average=not training)
        logits = jax.nn.relu(logits)
        logits = logits.reshape((logits.shape[0], -1))
        logits = nn.Dense(self.num_actions)(logits)

        # value head
        v = nn.Conv(features=1, kernel_size=(1, 1))(x)
        v = nn.BatchNorm(momentum=0.9)(v, use_running_average=not training)
        v = jax.nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = nn.Dense(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v


state_to_graph = jax.jit(state_to_graph, static_argnames='use_embedding')
new_state_to_graph = jax.jit(cg.state_to_graph)
class ModelManager(NamedTuple):
    id: str
    model: nn.Module
    use_embedding: bool = True
    use_graph: bool = True
    new_graph: bool = True

    def init(self, key: chex.PRNGKey, x):
        if self.use_graph:
            return self.model.init(key, graphs=x)
        return self.model.init(key, x=x)

    @overload
    def __call__(
        self,
        x,
        legal_action_mask: jnp.ndarray,
        params: chex.ArrayTree,
        training: Literal[False]=False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def __call__(
        self,
        x,
        legal_action_mask: jnp.ndarray,
        params: chex.ArrayTree,
        training: Literal[True]
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], chex.ArrayTree]:
        ...

    def __call__(
        self,
        x,
        legal_action_mask: jnp.ndarray,
        params: chex.ArrayTree,
        training: bool=False
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[
        Tuple[jnp.ndarray, jnp.ndarray],
        chex.ArrayTree
    ]:
        if self.use_graph:
            r_tuple, batch_stats = self.model.apply(
                cast(Mapping, params),
                graphs=x,
                mutable=['batch_stats'],
                training=training
            )
        else:
            r_tuple, batch_stats = self.model.apply(
                cast(Mapping, params),
                x=x,
                mutable=['batch_stats'],
                training=training
            )
        logits, value = r_tuple
        value = jnp.reshape(value, (-1,))
        logits = logits.reshape((value.shape[-1], -1))

        # mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min
        )
        if training:
            return (logits, value), batch_stats['batch_stats']
        return logits, value

    def format_data(self, state=None, board=None, observation=None,
            legal_action_mask=None, **kwargs):
        if self.use_graph:
            if state is not None:
                board = state._board
                observation = state.observation
                legal_action_mask = state.legal_action_mask
            if self.new_graph:
                return new_state_to_graph(
                    observation, legal_action_mask,
                )
            return state_to_graph(
                board, observation, legal_action_mask,
                use_embedding=self.use_embedding
            )
        return state.observation if state is not None else observation
