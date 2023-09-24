from typing import Optional
from omegaconf import DictConfig
import torch
from torch_geometric.data import Data

import ipdb


def create_graph_data(cfg: DictConfig,
                      final_pearson: torch.Tensor,
                      labels: torch.Tensor,
                      site: torch.Tensor,
                      final_sc: Optional[torch.Tensor] = None,
                      t1: Optional[torch.Tensor] = None,):
    
    # map site names to unique integers
    unique_sites = set(site)
    site_mapping = {name: idx for idx, name in enumerate(unique_sites)}

    num_nodes = final_pearson.shape[1]



    graph_data_list = []
    for i in range(final_pearson.shape[0]):
        edge_index = final_pearson[i].nonzero(as_tuple=False).t().contiguous()

        # create edge weights tensor 'MAKING THEM ABSOLUTE'
        # edge_weight = final_pearson[i][edge_index[0], edge_index[1]].clamp(min=0)

        edge_weight = final_pearson[i][edge_index[0], edge_index[1]]

        # define node features
        if cfg.dataset.node == 'one_hot':
           # Create a tensor of indices from 0 to num_nodes
            indices = torch.arange(num_nodes)
            # Create a one-hot encoded tensor
            node_feature = torch.nn.functional.one_hot(indices).float()
        elif cfg.dataset.node == 'fc':
            # Using the fully connected (fc) tensor as node features
            node_feature = final_pearson[i]

        mapped_site = site_mapping[site[i]]

        data = Data(x=node_feature, edge_index=edge_index,
                    y=labels[i], site=mapped_site, edge_weight=edge_weight)
        
        if t1 != None:
            data.t1 = t1[i]
            data = Data(x=node_feature, edge_index=edge_index,
                        y=labels[i], site=mapped_site, edge_weight=edge_weight, t1=t1[i])
        graph_data_list.append(data)

    return graph_data_list, site
