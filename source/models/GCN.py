import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

from omegaconf import DictConfig
import ipdb

class GCN(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(GCN, self).__init__()
        self.cfg = cfg
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes
        self.node_sz = cfg.dataset.node_sz

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                self.convs.append(GCNConv(cfg.dataset.node_feature_sz, self.hidden_size))
            else:
                self.convs.append(GCNConv(self.hidden_size,self.hidden_size))

        self.readout_lin = nn.Linear(
            self.node_sz * self.hidden_size, self.hidden_size)
                
        self.lin = nn.Linear(self.hidden_size, 1)                

    def convert_edge_positive(self, edge_index, edge_weight):
        edge_index = edge_index[:, edge_weight > 0]
        edge_weight = edge_weight[edge_weight > 0]
        return edge_index, edge_weight

    def forward(self, data, **kwargs):
        self.epoch = kwargs['epoch']
        self.iteration = kwargs['iteration']
        self.test_phase = kwargs['test_phase']

        x, edge_index, edge_weight, batch, labels = data.x, data.edge_index, data.edge_weight, data.batch, data.y
        edge_index, edge_weight = self.convert_edge_positive(edge_index, edge_weight)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if i < self.num_layers:
                x = F.leaky_relu(x)

        xs = []
        for graph_idx in batch.unique():
            graph_nodes = x[batch == graph_idx]
            graph_nodes = graph_nodes.view(-1)
            xs.append(self.readout_lin(graph_nodes))
        x = torch.stack(xs).to(x.device)

        x = F.leaky_relu(x)
        x = self.lin(x)

        return x
