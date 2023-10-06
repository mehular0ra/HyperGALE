import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch

from omegaconf import DictConfig

from .HypergraphGCNConvv2 import HypergraphGCNConvv2

from .Readouts.set_transformer_models import SetTransformer
from .Readouts.janossy_pooling import JanossyPooling


import ipdb


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
        return A, x


class HyperGALE(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(HyperGALE, self).__init__()

        self.cfg = cfg
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes
        self.node_sz = cfg.dataset.node_sz

        self.num_edges = cfg.dataset.node_sz

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(HypergraphGCNConvv2(
                    cfg, cfg.dataset.node_feature_sz, self.hidden_size, num_edges=self.num_edges))
            else:
                self.convs.append(HypergraphGCNConvv2(
                    cfg, self.hidden_size, self.hidden_size, num_edges=self.num_edges))

        if self.cfg.model.readout == 'set_transformer':
            self.readout_layer = SetTransformer(dim_input=self.hidden_size,
                                                num_outputs=1, dim_output=self.hidden_size)
        elif self.cfg.model.readout == 'janossy':
            self.readout_layer = JanossyPooling(
                num_perm=cfg.model.num_perm, in_features=self.hidden_size, fc_out_features=self.hidden_size)
        else:
            self.readout_lin = nn.Linear(
                self.node_sz * self.hidden_size, self.hidden_size)

        self.lin = nn.Linear(self.hidden_size, 1)

        # gated attention
        self.attn_gated = Attn_Net_Gated(L=self.hidden_size)

    def forward(self, data, **kwargs):
        self.epoch = kwargs['epoch']
        self.iteration = kwargs['iteration']
        self.test_phase = kwargs['test_phase']
        x, hyperedge_index, hyperedge_weight, batch, labels = data.x, data.edge_index, data.edge_weight, data.batch, data.y
        for i in range(self.num_layers):
            x = self.convs[i](x, hyperedge_index, epoch=self.epoch)

            if i < self.num_layers:
                x = F.leaky_relu(x)

        # node_attn (gated attention)
        xs = []
        saved_A = []
        for graph_idx in batch.unique():
            graph_nodes = x[batch == graph_idx]
            A, x_new = self.attn_gated(graph_nodes)
            saved_A.append(A.view(-1))
            # Broadcasting A to the same dimensions as x
            A_broadcasted = A.expand_as(x_new)
            # Performing element-wise multiplication
            x_new = A_broadcasted * x_new
            xs.append(x_new)
        x = torch.stack(xs).to(x.device)
        x = x.view(-1, self.hidden_size)


        if self.cfg.model.readout in ['set_transformer', 'janossy']:
            x = x.view(-1, self.node_sz, self.hidden_size)
            x = self.readout_layer(x)
            x = x.squeeze()
        else:
            xs = []
            for graph_idx in batch.unique():
                graph_nodes = x[batch == graph_idx]
                graph_nodes = graph_nodes.view(-1)
                xs.append(self.readout_lin(graph_nodes))
            x = torch.stack(xs).to(x.device)

        x = F.leaky_relu(x)
        x = self.lin(x)

        return x
