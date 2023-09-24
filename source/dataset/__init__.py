from omegaconf import DictConfig, open_dict
from .fc import load_fc_data
from .dataloader import init_stratified_dataloader

from typing import List
import torch.utils as utils
import logging

from .construct_graph import create_graph_data
from .construct_hyperaph import create_hypergraph_data

import ipdb


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    logging.info('cfg.dataset.name: %s', cfg.dataset.name)
    
    # create dataset
    datasets = load_fc_data(cfg)

    # graph data creation
    data_creation = cfg.model.get("data_creation", "graph")
    data_creation_func = "create_" + data_creation + "_data"
    graph_data_list, site = eval(data_creation_func)(cfg, *datasets)

    # dataloader creation
    dataloaders = init_stratified_dataloader(cfg, graph_data_list, site)

    return dataloaders
