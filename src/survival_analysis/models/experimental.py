#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.typing import np
from torch_geometric.utils import dense_to_sparse
import lightning as pl
import plotly.express as px

import survival_analysis.utils.losses as losses

class MetricLearn(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        hid_dim,
        optimizer,
        scheduler=None,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        out_dim=5
    ):

        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        
        self.phi = nn.Linear(in_dim, hid_dim)
        self.W = nn.Parameter(torch.randn(hid_dim, hid_dim))

        self.g = GCNConv(hid_dim, hid_dim)

        self.out = nn.Linear(hid_dim, out_dim)
        self.loss_manager = losses.LossManager([
            losses.CrossEntropyLoss(weight=1.0),
            losses.SmoothnessLoss(weight=self.lambda1),
            losses.ConnectivityLoss(weight=self.lambda2),
            losses.FrobeniusLoss(weight=self.lambda3),
        ])
    
    def _forward_full(self,x):
        # x: [n, d]
        z = self.phi(x)  # [n, h]
        z = torch.nn.functional.normalize(z, dim=-1)
        z = torch.nn.functional.relu(z)

        # logits edges
        projection = torch.matmul(z, self.W)
        normalized_projection = F.normalize(projection, p=2, dim=-1)
        adjacency = normalized_projection @ normalized_projection.T
        adjacency = F.relu(adjacency)

        edge_index, edge_weight = dense_to_sparse(adjacency)
        
        h = self.g(z, edge_index=edge_index, edge_weight=edge_weight)
        
        out = self.out(h)
        
        return out, adjacency, z

    def forward(self, x):
        out, *_ = self._forward_full(x)
        return out
    
    def log_losses(self, logs: dict[str, torch.Tensor], prefix: str = ""):
        for name, value in logs.items():
            self.log(
                f"{prefix}{name}",
                value,
                on_step=False,
                on_epoch=True,
            )
    
    def training_step(self, batch, batch_idx):
        output = self._forward_full(batch.x)
        pred, adjacency, z = output
        ctx = {"pred": pred, "label": batch.y, "adjacency": adjacency, "z": z}

        total, logs = self.loss_manager(ctx)
        self.log_losses(logs, prefix="train/")

        return total

    def validation_step(self, batch, batch_idx):
        output = self._forward_full(batch.x)
        pred, adjacency, z = output
        ctx = {"pred": pred, "label": batch.y, "adjacency": adjacency, "z": z}

        total, logs = self.loss_manager(ctx)
        self.log_losses(logs, prefix="val/")

        return total

    def evaluate(self, datamodule):
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            confusion_matrix,
        )

        self.eval()

        graph = datamodule.val_graph.to(self.device)
        val_idx = graph.val_idx

        with torch.no_grad():
            logits, adjacency, z = self._forward_full(graph.x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        # on ne garde que les noeuds concernés par l'évaluation
        y_pred = preds[val_idx].cpu().numpy()
        y_prob = probs[val_idx].cpu().numpy()
        y_true = graph.y[val_idx].cpu().numpy()

        n_classes = self.out.out_features

        # --- Métriques globales ---
        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

        try:
            auc_macro = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(n_classes))
            )
        except ValueError:
            auc_macro = float("nan")

        # --- Métriques par classe ---
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)

        metrics = {
            "cls/accuracy": acc,
            "cls/balanced_accuracy": balanced_acc,
            "cls/f1_macro": f1_macro,
            "cls/f1_weighted": f1_weighted,
            "cls/precision_macro": precision_macro,
            "cls/recall_macro": recall_macro,
            "cls/auc_macro_ovr": auc_macro,
        }

        for c in range(n_classes):
            metrics[f"cls/f1_class{c}"] = per_class_f1[c]
            metrics[f"cls/precision_class{c}"] = per_class_precision[c]
            metrics[f"cls/recall_class{c}"] = per_class_recall[c]

        metrics["cls/confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        return metrics

    def configure_optimizers(self):
            optimizer = self.partial_optimizer(
                self.parameters(),
            )
            if self.partial_scheduler is None:
                return optimizer

            scheduler = self.partial_scheduler(optimizer)
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                # "frequency": "indicates how often the metric is updated",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
