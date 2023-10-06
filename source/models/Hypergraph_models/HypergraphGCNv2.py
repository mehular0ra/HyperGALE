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

class HypergraphGCNv2(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(HypergraphGCNv2, self).__init__()

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

    def forward(self, data, **kwargs):
        self.epoch = kwargs['epoch']
        self.iteration = kwargs['iteration']
        self.test_phase = kwargs['test_phase']
        x, hyperedge_index, hyperedge_weight, batch, labels = data.x, data.edge_index, data.edge_weight, data.batch, data.y
        for i in range(self.num_layers):
            x = self.convs[i](x, hyperedge_index, epoch=self.epoch)
            if i < self.num_layers :
                x = F.leaky_relu(x)

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
