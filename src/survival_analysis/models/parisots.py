#!/usr/bin/env python3

import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATv2Conv, ChebConv
import lightning as pl
from pycox.models.loss import CoxPHLoss
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from survival_analysis.utils.graph_utils import matrix_to_list

class ParisotsSimple(pl.LightningModule):
    def __init__(self, in_dim, hid_dim, optimizer, scheduler=None):
        super().__init__()
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        out_dim = 1
        
        # self.g = GCNConv(hid_dim, hid_dim)
        self.g = GATv2Conv(in_dim, hid_dim, heads=1, edge_dim=1, concat=False)
        self.out = nn.Linear(hid_dim, out_dim)
        self.loss = CoxPHLoss()

    def forward(self,x):
        # x: [n, d]
        normalized = F.normalize(x, p=2, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.T)
        adjacency = (similarity_matrix > 0.5).float()
        adjacency.fill_diagonal_(0)
        
        # Pytorch geometric format
        edge_index, edge_attr = matrix_to_list(adjacency)
        
        # messages
        h = self.g(x, edge_index=edge_index, edge_attr=edge_attr)
        # skip
        # h = h + z
        out = self.out(h)
        
        return out

    def training_step(self, batch, batch_idx):
        
        # ---- forward PyG
        pred = self.forward(batch.x)
        # pred: [b, n, C]

        times, events = batch.y[...,0], batch.y[...,1]
        
        loss = self.full_loss(pred, times, events)

        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        all_pred = self.forward(batch.x)
        pred = all_pred[batch.val_idx]
        times, events = batch.y[...,0], batch.y[...,1]
        times, events = times[batch.val_idx], events[batch.val_idx]

        loss = self.full_loss(pred, times, events)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        return loss
        
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

    def full_loss(self, pred, times, events):
        partial_likelihood = self.loss(pred, times, events)
        
        return partial_likelihood

    def evaluate(self, datamodule) -> dict:
        survival_model = CoxPH(self)
        train_x = datamodule.train_graph.x
        train_y_durations = datamodule.train_graph.y[..., 0]
        train_y_events = datamodule.train_graph.y[..., 1]

        val_x = datamodule.val_graph.x
        val_y = datamodule.val_graph.y
        val_idx = datamodule.val_graph.val_idx.numpy() # Les indices de test stockés dans le graph de val

        _ = survival_model.compute_baseline_hazards(train_x, (train_y_durations, train_y_events))

        surv = survival_model.predict_surv_df(val_x)

        # Extraction des durées et évènements réels pour le calcul des métriques
        durations_test = val_y[..., 0].numpy()
        events_test = val_y[..., 1].numpy()

        # Evaluation sur le split de validation uniquement
        ev = EvalSurv(
            surv[val_idx], 
            durations_test[val_idx], 
            events_test[val_idx], 
            censor_surv='km'
        )

        # Création de la grille temporelle pour le Brier Score
        time_grid = np.linspace(durations_test[val_idx].min(), durations_test[val_idx].max(), 100)

        c_index = ev.concordance_td()
        brier = ev.integrated_brier_score(time_grid)

        return {
            "test/c_index":    c_index,
            "test/brier_score": brier,
            "fold_index":       None,  # sera injecté par le script principal
        }

    
class ParisotsGCN(pl.LightningModule):
    """
    GCN pour analyse de survie sur graphe de population.
 
    Reprend l'architecture de Parisot et al. (arXiv:1703.03020) :
        - L couches ChebConv (K=3) + ReLU + Dropout
        - Couche de sortie linéaire → score de risque scalaire
        - Loss : Cox Partial Likelihood
 
    Le graphe est construit dynamiquement à partir de la similarité cosine
    entre les features (comme dans ParisotsSimple), sans phénotypes externes.
    """
 
    def __init__(
        self,
        in_dim:     int,
        hid_dim:    int,
        optimizer,
        scheduler=None,
        num_layers: int   = 1,      # L dans l'article (ABIDE: 1, ADNI: 5)
        K:          int   = 3,      # ordre Chebyshev (article: K=3)
        dropout:    float = 0.3,
        sim_threshold: float = 0.5, # seuil similarité cosine pour les arêtes
    ):
        super().__init__()
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        self.dropout       = dropout
        self.sim_threshold = sim_threshold
        out_dim = 1
 
        # --- L couches cachées ChebConv + ReLU (vs GATv2Conv dans ParisotsSimple) ---
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                ChebConv(
                    in_channels  = in_dim if i == 0 else hid_dim,
                    out_channels = hid_dim,
                    K            = K,
                )
            )
 
        # --- Couche de sortie : score de risque scalaire pour Cox ---
        self.out = nn.Linear(hid_dim, out_dim)
        self.loss = CoxPHLoss()
 
    # ------------------------------------------------------------------
    def forward(self, x):
        # Construction du graphe par similarité cosine (identique à ParisotsSimple)
        normalized        = F.normalize(x, p=2, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.T)
        # adjacency         = (similarity_matrix > self.sim_threshold).float()
        adjacency = F.relu(similarity_matrix)
        adjacency.fill_diagonal_(0)
 
        edge_index, edge_attr = matrix_to_list(adjacency)
 
        # L couches ChebConv + ReLU + Dropout
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight=edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
 
        out = self.out(h)
        return out
 
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        pred = self.forward(batch.x)
        times, events = batch.y[..., 0], batch.y[..., 1]
        loss = self.full_loss(pred, times, events)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss
 
    def validation_step(self, batch, batch_idx):
        all_pred = self.forward(batch.x)
        pred     = all_pred[batch.val_idx]
        times, events = batch.y[..., 0], batch.y[..., 1]
        times, events = times[batch.val_idx], events[batch.val_idx]
        loss = self.full_loss(pred, times, events)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss
 
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = self.partial_optimizer(self.parameters())
        if self.partial_scheduler is None:
            return optimizer
        scheduler = self.partial_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val/loss",
            },
        }
 
    # ------------------------------------------------------------------
    def full_loss(self, pred, times, events):
        return self.loss(pred, times, events)
 
    # ------------------------------------------------------------------
    def evaluate(self, datamodule) -> dict:
        survival_model = CoxPH(self)
        train_x = datamodule.train_graph.x
        train_y_durations = datamodule.train_graph.y[..., 0]
        train_y_events    = datamodule.train_graph.y[..., 1]
 
        val_x  = datamodule.val_graph.x
        val_y  = datamodule.val_graph.y
        val_idx = datamodule.val_graph.val_idx.numpy()
 
        _ = survival_model.compute_baseline_hazards(
            train_x, (train_y_durations, train_y_events)
        )
        surv = survival_model.predict_surv_df(val_x)
 
        durations_test = val_y[..., 0].numpy()
        events_test    = val_y[..., 1].numpy()
 
        ev = EvalSurv(
            surv[val_idx],
            durations_test[val_idx],
            events_test[val_idx],
            censor_surv='km'
        )
 
        time_grid = np.linspace(
            durations_test[val_idx].min(),
            durations_test[val_idx].max(),
            100
        )
 
        return {
            "test/c_index":     ev.concordance_td(),
            "test/brier_score": ev.integrated_brier_score(time_grid),
            "fold_index":       None,
        }


    
def matrix_to_list(A):
    row, col = torch.nonzero(A, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = A[row, col]

    return edge_index,edge_attr
