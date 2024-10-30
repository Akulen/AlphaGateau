from typing import cast, Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from jpyger import GraphConvolution

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
    n_res_layers: int = 8
    n_eval_layers: int = 5
    dot_v2: bool = True
    use_embedding: bool = True
    attention_pooling: bool = True
    # Useless parameters
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    @nn.compact
    def __call__(self, *args, graphs, training=False, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.use_embedding:
            graphs = graphs._replace(nodes=nn.Embed(num_embeddings=13, features=self.inner_size)(graphs.nodes.reshape((-1)).astype(jnp.int32)))

        for _ in range(self.n_res_layers):
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


