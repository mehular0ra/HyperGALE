import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import ipdb


def top_percent_edges(matrix, percent):
    """
    Keep the top percent of edges by absolute value.
    """
    flattened = np.sort(np.abs(matrix.flatten()))
    index = int((1.0 - percent/100.0) * len(flattened))
    threshold = flattened[index]
    mask = np.abs(matrix) >= threshold
    matrix_masked = matrix * mask
    return matrix_masked


def load_fc_data(cfg: DictConfig):

    fc_data = np.load(cfg.dataset.fc_path, allow_pickle=True).item()
    final_pearson = fc_data["corr"]
    labels = fc_data["label"]
    site = fc_data['site']

    # # Apply edge pruning
    # for i in range(final_pearson.shape[0]):
    #     final_pearson[i] = top_percent_edges(final_pearson[i], cfg.dataset.perc_edges)

    final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_pearson, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
    return final_pearson, labels, site
