#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric import edge_index
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool
from sksurv.metrics import concordance_index_censored

import lightning.pytorch as pl

import survival_analysis.utils.losses as losses

class GAT(pl.LightningModule): #Graph Attention Network
    def __init__(self,  in_dim, out_features, hidden_channels=16, edge_dim=1, GNN1lay=False, heads=2, optimizer=None, scheduler=None,):
        super().__init__()

        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler

        self.conv1 = GATv2Conv(in_dim, hidden_channels, heads=heads,
                            add_self_loops=False, edge_dim=edge_dim)

        self.conv_out = GATv2Conv(hidden_channels * heads, out_features,
                                heads=1, add_self_loops=False, edge_dim=edge_dim)

        self.lin = Linear(hidden_channels * heads, out_features)
        self.GNN1lay = GNN1lay

        self.loss_manager = losses.LossManager([
            losses.CoxPHLoss(weight=1.0),
        ])

    def _forward_full(self, x, edge_index, edge_weight):

        # Première couche GAT
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        # Si 2 couches GNN
        if not self.GNN1lay:
            x = self.conv_out(x, edge_index, edge_weight)  # -> (num_nodes, out_features)
        else:
            # Si 1 seule couche : projeter via un Linear
            x = self.lin(x)  # lin doit sortir (hidden → out_features)

        # Softmax node-wise pour obtenir des probas
        # x = F.softmax(x, dim=1)

        return x

    def forward(self, x, edge_index, edge_weight):
        pred = self._forward_full(x, edge_index, edge_weight)
        return pred

    def training_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index
        edge_weight = batch.weight 

        preds = self._forward_full(x, edge_index, edge_weight)

        ctx = {
            "preds": preds,
            "targets": y
        }

        total, logs = self.loss_manager(ctx)

        self.log_losses(logs, prefix="train/")
        return total

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(self.parameters())
        if self.partial_scheduler is None:
            return optimizer

        scheduler = self.partial_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                # "frequency": "indique la fréquence de mise à jour de la métrique",
                # si "monitor" référence une métrique de validation, "frequency" doit être
                # un multiple de "trainer.check_val_every_n_epoch".
            },
        }

    def validation_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y
        val_index = batch.val_idx
        edge_index = batch.edge_index
        edge_weight = batch.weight 

        preds = self._forward_full(x, edge_index, edge_weight)

        ctx = {
            "preds": preds[val_index],
            "targets": y[val_index]
        }

        total, logs = self.loss_manager(ctx)

        self.log_losses(logs, prefix="val/")
        return total

    def log_losses(self, logs: dict, prefix: str = ""):
        for name, value in logs.items():
            self.log(
                f"{prefix}{name}",
                value,
                on_step=False,
                on_epoch=True,
            )

    def evaluate(self, datamodule):
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            for graph in datamodule.val_dataloader():
                graph = graph.to(device)
                edge_weight = graph.weight.to(device)
                risks = self(
                    graph.x,
                    graph.edge_index,
                    edge_weight
                )[graph.val_idx]
                risks = risks.squeeze(-1).cpu().numpy()

                y = graph.y[graph.val_idx].cpu().numpy()
                times, events = y[:, 0], y[:, 1].astype(bool)

        c_index = concordance_index_censored(events, times, risks)[0]
        return {"test/c_index": c_index}
