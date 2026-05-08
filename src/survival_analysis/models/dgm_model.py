#!/usr/bin/env python3

from torch import nn
import torch
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATv2Conv
from torch_geometric.typing import np
import lightning as pl
from pycox.models.loss import CoxPHLoss
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import plotly.express as px

class SurvivalDGM(pl.LightningModule):
    def __init__(self, in_dim, hid_dim, optimizer, scheduler=None, tau=0.05):
        super().__init__()
        self.lambda1 = 0
        self.lambda2 = 0
        self.tau = tau 
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        self.training_mode = True
        out_dim = 1
        
        self.phi = nn.Linear(in_dim, hid_dim)
        
        self.W = nn.Parameter(torch.randn(hid_dim, hid_dim) * 0.1)
        # self.g = nn.Linear(hid_dim, hid_dim)
        # self.g = GCNConv(hid_dim, hid_dim)
        self.g = GATv2Conv(hid_dim, hid_dim, heads=1, edge_dim=1, concat=False)
        self.out = nn.Linear(hid_dim, out_dim)
        self.loss = CoxPHLoss()


    def _forward_full(self,x):
        # x: [n, d]
        z = self.phi(x)  # [n, h]
        # z = torch.nn.functional.normalize(z, dim=-1)
        z = torch.nn.functional.relu(z)

        # logits edges
        W_sym = 0.5 * (self.W + self.W.T)
        logits = z @ W_sym @ z.T  / np.sqrt(z.size(-1)) # [n, n]
        pi = torch.sigmoid(logits/self.tau)

        if self.training_mode:
            # binary concrete
            mask_raw = binary_concrete(logits, tau=self.tau, hard=True)
        else:
            mask_raw = ((pi)>0.5).int()

        # taking upper part of mask for symetrization
        upper_mask = torch.triu(mask_raw, diagonal=1)

        # On symétrise : l'arête (i,j) devient égale à l'arête (j,i)
        adjacency = upper_mask + upper_mask.t()

        self.pi = pi
        self.adjacency = adjacency
        # self.weights = weights

        # Pytorch geometric format
        edge_index, edge_attr = matrix_to_list(adjacency)
        
        # messages
        h = self.g(z, edge_index=edge_index, edge_attr=edge_attr)
        # skip
        # h = h + z
        out = self.out(h)
        
        return out, pi

    def forward(self, x):
        out, _ = self._forward_full(x)
        return out
        
    def training_step(self, batch, batch_idx):
        eps = 1e-8
        
        # ---- forward PyG
        pred,pi = self._forward_full(batch.x)
        # pred: [b, n, C]

        # ---- reconstruire masque dense
        # y = batch.y
        times, events = batch.y[...,0], batch.y[...,1]
        
        loss, partial_likelihood, l1_loss, *_ = self.full_loss(pred, times, events, pi)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/cox_loss", partial_likelihood, on_step=False, on_epoch=True)
        self.log("train/l1_loss", self.lambda1*l1_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        all_pred, pi = self._forward_full(batch.x)
        pred = all_pred[batch.val_idx]
        times, events = batch.y[...,0], batch.y[...,1]
        times, events = times[batch.val_idx], events[batch.val_idx]

        loss, partial_likelihood, *_ = self.full_loss(pred, times, events, pi)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/partial_likelihood", loss, on_step=False, on_epoch=True)

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

    def full_loss(self, pred, times, events, pi):
        eps = 1e-8
        partial_likelihood = self.loss(pred, times, events)
        l1_loss = pi.abs().mean()
        entropy = -pi * torch.log(pi + eps) - (1 - pi)*torch.log(1 - pi + eps)
        entropy_loss = entropy.mean()
        
        loss = partial_likelihood + self.lambda1 * l1_loss + self.lambda2 * entropy_loss

        return loss, partial_likelihood, l1_loss, entropy_loss

    def evaluate(self, datamodule) -> dict:
        survival_model = CoxPH(self)
        mean_cindex, mean_brier = evaluate_bis(datamodule, survival_model, nb_tests=100)

        pi = self.pi.cpu().numpy().flatten()
        fig = plot_edge_probs(pi, log_scale=False)

        return {
            "test/c_index":    mean_cindex,
            "test/brier_score": mean_brier,
            "test/c_index_std": 0,
            "fold_index":       None,  # sera injecté par le script principal
            "edge_probs":       fig,   # figure wandb, spécifique à ce modèle
        }
#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist, x

def binary_concrete(logits, tau=1.0, hard=False, eps=1e-7):
    u = torch.rand_like(logits)
    logistic_noise = torch.log(u + eps) - torch.log(1 - u + eps)
    y = torch.sigmoid((logits + logistic_noise) / tau)

    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y

    return y

def matrix_to_list(A):
    row, col = torch.nonzero(A, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = A[row, col]

    return edge_index,edge_attr

def evaluate_bis(datamodule, survival_model, nb_tests=100):

    # On utilise les données préparées par le datamodule
    # .train_graph et .val_graph ont été créés lors du datamodule.setup()
    train_x = datamodule.train_graph.x
    train_y_durations = datamodule.train_graph.y[..., 0]
    train_y_events = datamodule.train_graph.y[..., 1]
    
    val_x = datamodule.val_graph.x
    val_y = datamodule.val_graph.y
    val_idx = datamodule.val_graph.val_idx.numpy() # Les indices de test stockés dans le graph de val

    survs = []
    
    for i in range(nb_tests):
        # Prédiction des fonctions de survie
        _ = survival_model.compute_baseline_hazards(train_x, (train_y_durations, train_y_events))
        survs.append(survival_model.predict_surv_df(val_x))


    surv = sum(survs)/nb_tests

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

    # Calcul des moyennes finales pour ce split
    return c_index, brier

def plot_edge_probs(pi, bins=100, log_scale=True):
    
    fig = px.histogram(
        x=pi,
        nbins=bins,
        log_y=log_scale,
        title="Distribution des probabilités des arêtes",
        labels={"x": "Probabilité", "y": "Nombre d'arêtes"},
    )

    fig.add_vline(
        x=pi.mean(),
        line=dict(color='red', dash='dash', width=1),
        annotation_text=f'Moyenne: {pi.mean():.4f}',
    )

    return fig
