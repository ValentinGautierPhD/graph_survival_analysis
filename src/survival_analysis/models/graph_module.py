#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric import edge_index
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool

import lightning.pytorch as pl

class GAT(pl.LightningModule): #Graph Attention Network
    def __init__(self,  in_features, out_features, beta, hidden_channels=16, edge_dim=1, GNN1lay=False, heads=2, optimizer=None, scheduler=None,):
        super().__init__()

        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler

        self.conv1 = GATv2Conv(in_features, hidden_channels, heads=heads,
                            add_self_loops=False, edge_dim=edge_dim)

        self.conv_out = GATv2Conv(hidden_channels * heads, out_features,
                                heads=1, add_self_loops=False, edge_dim=edge_dim)

        self.lin = Linear(hidden_channels * heads, out_features)
        self.GNN1lay = GNN1lay
        self.beta = beta

        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_weight):

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

    # data has the shape (x, edge_index, edge_weight).
    # If no edge weight then edge_weight=None
    def training_step(self, batch, batch_idx): 
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight if hasattr(batch, "edge_weight") else torch.ones(batch.edge_index.size(1), device=batch.edge_index.device)
        target = batch.y
        logits = self.forward(x, edge_index, edge_weight)
        probas = F.softmax(x, dim=-1)
        
        # Classification loss
        loss = self.loss_func(logits, target)

        # Regularization loss
        row, col = edge_index
        # Tr(X^T A X) = sum_{(i,j) in E} w_ij * <x_i, x_j>
        dot = (probas[row] * probas[col]).sum(dim=1)   # shape (E,)
        sum_xAx = (edge_weight * dot).sum()
        # Tr(X^T X)
        sum_x2 = (probas * probas).sum()
        regularisation = sum_x2 - sum_xAx

        # progress bar logging metrics (add custom metric definitions later if useful?)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss + self.beta * regularisation

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(
            self.parameters(),
        )
        scheduler = self.partial_scheduler(optimizer)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "loss",
            # "frequency": "indicates how often the metric is updated",
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }
