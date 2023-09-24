from source.utils import accuracy, TotalMeter, count_params, isfloat, BCEWithLogitsLossL2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import lr_scheduler_factory, LRScheduler
import logging

from scipy.special import expit



import ipdb


class Train:
    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader]) -> None:

        self.all_saved_x = []

        self.cfg = cfg
        self.device = self.cfg.device

        self.logger = logging.getLogger()
        self.model = model.to(self.device)
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = BCEWithLogitsLossL2(self.model, self.cfg.training.l2)

        # self.best_test_accuracy = 0.0
        self.best_test_metrics = {
            "accuracy": 0.0,
            "auc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

        self.init_meters()


    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy, self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.test_accuracy, self.train_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, epoch, optimizer, lr_scheduler):

        self.model.train()

        for iteration, data in enumerate(self.train_dataloader):
            optimizer.zero_grad()
            # label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer,
                                step=self.current_step) 
            
            data = data.to(self.device)
            if self.cfg.model.name in ["HyperGALE", "HypergraphGCNv2", "HypergraphGCN", "GCN"]:
                predict = self.model(
                    data, epoch=epoch, iteration=iteration, test_phase=False).squeeze()
            else: 
                predict = self.model(data).squeeze()

            label = data.y.to(self.device)
            loss = self.loss_fn(predict, label)
            loss.backward(retain_graph=True)
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.step()
            acc = accuracy(predict, label)
            self.train_accuracy.update_with_weight(acc, label.shape[0])



    def test_per_epoch(self, epoch, dataloader, loss_meter, acc_meter):
        labels = []
        logits = []

        self.model.eval()

        with torch.no_grad():

            for iteration, data in enumerate(dataloader):

                data = data.to(self.device)
                if self.cfg.model.name in ["HyperGALE", "HypergraphGCNv2", "HypergraphGCN", "GCN"]:
                    output = self.model(data, epoch=epoch, iteration=iteration, test_phase=True).squeeze()
                else:
                    output = self.model(data).squeeze()

                label = data.y.to(self.device)
                print("Epoch: ", epoch, "Iteration: ", iteration)
                if iteration == 0:
                    ipdb.set_trace()
                loss = self.loss_fn(output, label)

                loss_meter.update_with_weight(
                    loss.item(), label.shape[0])
                acc = accuracy(output, label)
                acc_meter.update_with_weight(acc, label.shape[0])
                logits += output.squeeze().tolist()
                labels += label.tolist()

        # convert logits to probabilities and predictions
        probabilities = expit(np.array(logits))
        predictions = np.round(probabilities)

        labels = np.array(labels)

        auc = roc_auc_score(labels, probabilities)

        metric = precision_recall_fscore_support(
            labels, predictions, average='micro')

        report = classification_report(
            labels, predictions, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall
    

    def train(self):
        print("\nStarting training...")
        training_process = []
        self.current_step = 0
        
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(epoch, self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(epoch, self.val_dataloader, self.val_loss, self.val_accuracy)
            test_result = self.test_per_epoch(epoch, self.test_dataloader,   
                                            self.test_loss, self.test_accuracy)
            
            
            if self.test_accuracy.avg > self.best_test_metrics["accuracy"]:
                self.best_test_metrics["accuracy"] = self.test_accuracy.avg
                auc = test_result[0]
                sensitivity = test_result[-1]
                specificity = test_result[-2]
                self.best_test_metrics["auc"] = auc
                self.best_test_metrics["sensitivity"] = sensitivity
                self.best_test_metrics["specificity"] = specificity

                # save model for best accuracy
                if self.cfg.model.model_save:
                    torch.save(self.model.state_dict(), 'best_model.pt')
                    

            self.logger.info(" | ".join([
                f'Epoch[{epoch+1}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',

                f'Val Loss:{self.val_loss.avg: .3f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Val Sen:{val_result[-1]:.4f}',
                f'Val Spe:{val_result[-2]:.4f}',

                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'Test Spe:{test_result[-2]:.4f}',
                f'Test F1:{test_result[-4]:.4f}',
                f'Test Recall:{test_result[-5]:.4f}',
                f'Test Precision:{test_result[-6]:.4f}',

                f'Best Test Accuracy:{self.best_test_metrics["accuracy"]: .3f}%',
                f'Best Test AUC:{self.best_test_metrics["auc"]:.4f}',
                f'Best Test Sensitivity:{self.best_test_metrics["sensitivity"]:.4f}',
                f'Best Test Specificity:{self.best_test_metrics["specificity"]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.7f}'

            ]))

            if self.cfg.is_wandb:
                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,

                    "Val Loss": self.val_loss.avg,
                    "Val Accuracy": self.val_accuracy.avg,
                    "Val AUC": val_result[0],
                    "Val Sensitivity": val_result[-1],

                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    "Test AUC": test_result[0],
                    "Test Sensitivity": test_result[-1],
                    "Test Specificity": test_result[-2],
                    "Test F1": test_result[-4],
                    "Test Recall": test_result[-5],
                    "Test Precision": test_result[-6],

                    "Best Test Accuracy": self.best_test_metrics["accuracy"],
                    "Best Test AUC": self.best_test_metrics["auc"],
                    "Best Test Sensitivity": self.best_test_metrics["sensitivity"],
                    "Best Test Specificity": self.best_test_metrics["specificity"],
                })


