import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

from omegaconf import DictConfig
import ipdb

from ..components import tsne_plot_data, node_att_data_save



class Attn_Net_Gated(nn.Module):
    # Attention Network with Sigmoid Gating (3 fc layers). Args:
    # L: input feature dimension
    # D: hidden layer dimension
    # dropout: whether to use dropout (p = 0.25)
    # n_classes: number of classes """

    def __init__(self, L=64, D=256, dropout=True, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        # A = F.softmax(A, dim=0)
        return A, x



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
