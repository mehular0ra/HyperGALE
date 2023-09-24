import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

import ipdb

from .dataset import dataset_factory
from .models import model_factory
from .components import optimizers_factory, lr_scheduler_factory
from .training import training_factory

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'




def model_training(cfg: DictConfig):

    dataloaders = dataset_factory(cfg)
    model = model_factory(cfg)
    print(model)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    training = training_factory(cfg=cfg,
                                model=model,
                                optimizers=optimizers,
                                lr_schedulers=lr_schedulers,
                                dataloaders=dataloaders)
    training.train()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    data_creation = cfg.model.get('data_creation', {})
    if data_creation.get("hypergraph", "graph") == "hypergraph":
        group_name = f"{cfg.dataset.name}_{cfg.model.name}_node:{cfg.dataset.node}_Kneigs:{cfg.model.K_neigs}_layers:{cfg.model.num_layers}_hidden:{cfg.model.hidden_size}_lr:exp_1e-5"
    else:
        group_name = f"{cfg.dataset.name}_{cfg.model.name}_node:{cfg.dataset.node}_layers:{cfg.model.num_layers}_hidden:{cfg.model.hidden_size}_lr:exp_1e-5"


    for _ in range(cfg.repeat_time):

        if cfg.is_wandb:
            run = wandb.init(project=cfg.project, reinit=True,
                             group=f"{group_name}", tags=[f"{cfg.dataset.name}, {cfg.model.name}"])
        logging.info(OmegaConf.to_yaml(cfg)) 
        model_training(cfg)

        if cfg.is_wandb:
            wandb.finish()

    

if __name__ == "__main__":
    main()
