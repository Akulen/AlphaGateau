import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from jpyger import GraphConvolution

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
