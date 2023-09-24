import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Batch

from omegaconf import DictConfig
import ipdb


class GraphSAGE(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(GraphSAGE, self).__init__()
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes
        self.node_sz = cfg.dataset.node_sz


        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                    self.convs.append(
                        SAGEConv(cfg.dataset.node_feature_sz, self.hidden_size))
            else:
                self.convs.append(SAGEConv(self.hidden_size, self.hidden_size))

        self.readout_lin = nn.Linear(
            self.node_sz * self.hidden_size, self.hidden_size)

        self.lin = nn.Linear(self.hidden_size, 1)

        # TODO: Add different pooling methods

    # def convert_edge_positive(self, edge_index, edge_weight):

    #     edge_index = edge_index[:, edge_weight > 0]
    #     edge_weight = edge_weight[edge_weight > 0]
    #     return edge_index, edge_weight

    def filter_top_edges(self, edge_index, edge_weight, batch, percentage=0.1):
        edge_list = []
        for graph_idx in batch.unique():
            mask = batch[edge_index[0]] == graph_idx
            graph_edges = edge_index[:, mask]
            graph_weights = edge_weight[mask]

            # Get the top percentage indices
            top_indices = torch.topk(graph_weights, int(
                percentage * graph_weights.size(0)), largest=True).indices

            edge_list.append(graph_edges[:, top_indices])

        return torch.cat(edge_list, dim=1)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # Filter the top 10% of edges
        edge_index = self.filter_top_edges(edge_index, edge_weight, batch)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # Don't pass edge_weight here
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)

            # if torch.isnan(x).any():
            #     print(f"Found NaN values in output tensor in layer {i}")

        xs = []
        for graph_idx in batch.unique():
            graph_nodes = x[batch == graph_idx]
            graph_nodes = graph_nodes.view(-1)
            xs.append(self.readout_lin(graph_nodes))
        x = torch.stack(xs).to(x.device)

        x = self.lin(x)

        return x
