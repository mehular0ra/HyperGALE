from typing import Optional, List
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import ipdb


def init_stratified_dataloader(cfg: DictConfig,
                               graph_data_list: List,
                               site: np.array) -> List[DataLoader]:
    
    train_length = int(cfg.dataset.train_set * len(graph_data_list))
    train_ratio = cfg.dataset.train_set
    test_ratio = cfg.dataset.test_set

    # Stratified split
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, train_size=train_ratio, random_state=42)
    for train_index, test_index in split.split(graph_data_list, site):
        train_data_list = [graph_data_list[i] for i in train_index]
        test_data_list = [graph_data_list[i] for i in test_index]


    # create pyg dataloader
    train_dataloader = DataLoader(
        train_data_list, batch_size=cfg.dataset.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_list, batch_size=cfg.dataset.batch_size, shuffle=False)

    # add total_steps and steps_per_epoch to cfg
    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    return [train_dataloader, test_dataloader]


