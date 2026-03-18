#!/usr/bin/env python3

from torch import nn
import torch
import torch_geometric
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATv2Conv
from torch_geometric.typing import np
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import dense_to_sparse
import lightning as pl
from pycox.models.loss import CoxPHLoss


class MinimalDGM(pl.LightningModule):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        out_dim = 2
        
        self.phi = nn.Linear(in_dim, hid_dim)
        
        # self.W = nn.Parameter(torch.triu(torch.randn(hid_dim, hid_dim) * 0.1))
        self.W = nn.Parameter(torch.randn(hid_dim, hid_dim) * 0.1)
        # self.W_message = nn.Parameter(torch.triu(torch.randn(hid_dim, hid_dim) * 0.1))
        # self.g = nn.Linear(hid_dim, hid_dim)
        # self.g = GATv2Conv(hid_dim, hid_dim, edge_dim=1, add_self_loops=False)
        self.out = nn.Linear(hid_dim, 2)

    def forward(self, x, tau=0.5):
        # x: [n, d]
        z = self.phi(x)  # [n, h]
        # z = torch.nn.functional.normalize(z, dim=-1)
        z = torch.nn.functional.relu(z)

        # logits edges
        logits = z @ self.W @ z.T  # [n, n]
        pi = torch.sigmoid(logits)

        # row, col = torch.nonzero(pi, as_tuple=True)
        # edge_index = torch.stack([row, col], dim=0)
        # edge_attr = pi[row, col]

        # binary concrete
        mask = binary_concrete(logits, tau=tau, hard=True)
        # mask = pi
        # weights = z @ z.T
        adjacency = pi #* weights
        self.A = pi
        # edge_index, edge_attr = dense_to_sparse(self.A)

        # messages
        # h = self.g(z, edge_index=edge_index, edge_weight=edge_attr)
        h = adjacency @ z
        h = nn.functional.relu(h)
        # skip
        h = h + z

        return self.out(h), pi
        # return h

    def training_step(self, batch, batch_idx):
        eps = 0.05
        
        # ---- forward PyG
        pred,pi = self(batch.x)
        # pred: [b, n, C]

        # ---- reconstruire masque dense
        # y = batch.y
        y, mask = to_dense_batch(batch.y, batch.batch)
        y_labels = y.argmax(dim=-1)

        # ---- loss principale
        # loss = torch.nn.functional.cross_entropy(pred.view(-1,2), y_labels.view(-1), weight=torch.tensor([1.0,5.0]).to(pred.device))
        ce = torch.nn.functional.cross_entropy(pred.view(-1,2), y_labels.view(-1), weight=torch.tensor([1.0,5.7]).to(pred.device))
        kl = (
            pi * (torch.log(pi + 1e-8) - torch.log(torch.tensor(eps)))
            + (1 - pi) * (torch.log(1 - pi + 1e-8) - torch.log(torch.tensor(1 - eps)))
        ).mean()
        # kl = torch.abs(pi).sum()
        
        loss = ce + (1e-3) * kl
        
        self.log("loss", loss, on_step=False, on_epoch=True)

        # ---- accuracy
        correct = (pred.argmax(-1) == y.argmax(-1)).float().mean()
        self.log("acc", correct, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eps = 0.05
        all_pred, pi = self(batch.x)
        pred = all_pred[batch.val_idx]
        y = batch.y[batch.val_idx]
        y_labels = y.argmax(dim=-1)

        ce = torch.nn.functional.cross_entropy(pred.view(-1,2), y_labels.view(-1), weight=torch.tensor([1.0,5.7]).to(pred.device))
        kl = (
            pi * (torch.log(pi + 1e-8) - torch.log(torch.tensor(eps)))
            + (1 - pi) * (torch.log(1 - pi + 1e-8) - torch.log(torch.tensor(1 - eps)))
        ).mean()
        # kl = torch.abs(pi).sum()
        
        loss = ce + (1e-3) * kl
        
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        correct = (pred.argmax(-1) == y.argmax(-1)).float().mean()
        self.log("val_acc", correct, on_step=False, on_epoch=True)

        return loss
        
    def configure_optimizers(self):
        
        return torch.optim.Adam(self.parameters(), lr=0.02)


class SurvivalDGM(pl.LightningModule):
    def __init__(self, in_dim, hid_dim, optimizer, scheduler=None):
        super().__init__()
        self.lambda1 = 1
        self.lambda2 = 0
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        out_dim = 1
        
        self.phi = nn.Linear(in_dim, hid_dim)
        
        # self.W = nn.Parameter(torch.triu(torch.randn(hid_dim, hid_dim) * 0.1))
        self.W = nn.Parameter(torch.randn(hid_dim, hid_dim) * 0.1)
        self.W_message = nn.Parameter(torch.triu(torch.randn(hid_dim, hid_dim) * 0.1))
        # self.g = nn.Linear(hid_dim, hid_dim)
        # self.g = GCNConv(hid_dim, hid_dim)
        self.g = GATv2Conv(hid_dim, hid_dim, heads=1, edge_dim=1, concat=False)
        self.out = nn.Linear(hid_dim, out_dim)
        self.loss = CoxPHLoss()


    def _forward_full(self,x, tau=0.5):
        # x: [n, d]
        z = self.phi(x)  # [n, h]
        # z = torch.nn.functional.normalize(z, dim=-1)
        z = torch.nn.functional.relu(z)

        # logits edges
        W_sym = 0.5 * (self.W + self.W.T)
        # W_message_sym = 0.5 * (self.W_message + self.W_message.T)
        logits = z @ W_sym @ z.T  / np.sqrt(z.size(-1)) # [n, n]
        # weights = z @ 
        pi = torch.sigmoid(logits)

        # binary concrete
        mask = binary_concrete(logits, tau=tau, hard=True)
        # weights = z @ self.W_message @ z.T
        adjacency =  mask
        self.A = pi
        self.mask = mask
        # self.weights = weights

        # Pytorch geometric format
        edge_index, edge_attr = matrix_to_list(adjacency)
        
        # messages
        h = self.g(z, edge_index=edge_index, edge_attr=edge_attr)
        # h = self.g(z, edge_index=edge_index)
        # h = adjacency @ z
        # h = nn.functional.relu(h)
        # skip
        # h = h + z
        out = self.out(h)
        # out[...] = 1
        return out, pi
        # return h

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
        
        # ---- loss principale
        # loss = torch.nn.functional.cross_entropy(pred.view(-1,2), y_labels.view(-1), weight=torch.tensor([1.0,5.0]).to(pred.device))
        partial_likelihood = self.loss(pred, times, events)
        l1_loss = pi.abs().mean()
        entropy = -pi * torch.log(pi + eps) - (1 - pi)*torch.log(1 - pi + eps)
        entropy_loss = entropy.mean()
        # kl = torch.abs(pi).sum()
        
        loss = partial_likelihood + self.lambda1 * l1_loss + self.lambda2 * entropy_loss
        
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/cox_loss", partial_likelihood, on_step=False, on_epoch=True)
        self.log("train/l1_loss", self.lambda1*l1_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eps = 1e-8
        
        all_pred, pi = self._forward_full(batch.x)
        pred = all_pred[batch.val_idx]
        times, events = batch.y[...,0], batch.y[...,1]
        times, events = times[batch.val_idx], events[batch.val_idx]

        partial_likelihood = self.loss(pred, times, events)
        l1_loss = pi.abs().mean()
        entropy = -pi * torch.log(pi + eps) - (1 - pi)*torch.log(1 - pi + eps)
        entropy_loss = entropy.mean()
        
        loss = partial_likelihood + self.lambda1 * l1_loss

        
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
