import warnings
from typing import Any

import torch

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class HyperGraphData(Data):
    r"""
        x (torch.Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Hyperedge tensor
            with shape :obj:`[2, num_edges*num_nodes_per_edge]`.
            Where `edge_index[1]` denotes the hyperedge index and
            `edge_index[0]` denotes the node indicies that are connected
            by the hyperedge. (default: :obj:`None`)
            (default: :obj:`None`)
        edge_attr (torch.Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`.
            (default: :obj:`None`)
        y (torch.Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        pos (torch.Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         pos=pos, **kwargs)

    @property
    def num_edges(self) -> int:
        r"""Returns the number of hyperedges in the hypergraph.
        """
        if self.edge_index is None:
            return 0
        return max(self.edge_index[1]) + 1

    @property
    def num_nodes(self) -> int:
        num_nodes = super().num_nodes
        if (self.edge_index is not None and num_nodes == self.num_edges):
            return max(self.edge_index[0]) + 1
        return num_nodes

    def is_edge_attr(self, key: str) -> bool:
        val = super().is_edge_attr(key)
        if not val and self.edge_index is not None:
            return key in self and self[key].size(0) == self.num_edges

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index':
            return torch.tensor([[self.num_nodes], [self.num_edges]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


    def has_isolated_nodes(self) -> bool:
        if self.edge_index is None:
            return False
        return torch.unique(self.edge_index[0]).size(0) < self.num_nodes


    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        num_nodes = self.num_nodes
        if num_nodes is None:
            status = False
            warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'",
                          raise_on_error)

        if 'edge_index' in self:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                status = False
                warn_or_raise(
                    f"'edge_index' needs to be of shape [2, num_edges] in "
                    f"'{cls_name}' (found {self.edge_index.size()})",
                    raise_on_error)

        if 'edge_index' in self and self.edge_index.numel() > 0:
            if self.edge_index.min() < 0:
                status = False
                warn_or_raise(
                    f"'edge_index' contains negative indices in "
                    f"'{cls_name}' (found {int(self.edge_index.min())})",
                    raise_on_error)

            if num_nodes is not None and self.edge_index[0].max() >= num_nodes:
                status = False
                warn_or_raise(
                    f"'edge_index' contains larger indices than the number "
                    f"of nodes ({num_nodes}) in '{cls_name}' "
                    f"(found {int(self.edge_index.max())})", raise_on_error)

        return status


def warn_or_raise(msg: str, raise_on_error: bool = True):
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg)
