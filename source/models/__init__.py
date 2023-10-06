from omegaconf import DictConfig
from .GCN import GCN
from .GraphSAGE import GraphSAGE
from .GAT import GAT
from .Hypergraph_models.HypergraphGCN import HypergraphGCN
from .Hypergraph_models.HypergraphGCNv2 import HypergraphGCNv2
from .Hypergraph_models.HyperGALE import HyperGALE




def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config)
