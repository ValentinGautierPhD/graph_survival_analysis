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
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.lambda1 = 1
        self.lambda2 = 0
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
        
        self.log("loss", loss, on_step=False, on_epoch=True)
        self.log("cox_loss", partial_likelihood, on_step=False, on_epoch=True)
        self.log("l1_loss", self.lambda1*l1_loss, on_step=False, on_epoch=True)

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

        
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss
        
    def configure_optimizers(self):
        
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
    
    
class DGM_Model(pl.LightningModule):
    def __init__(self, in_features, out_features=1, optimizer=None, scheduler=None):
        super(DGM_Model, self).__init__()
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        
        self.f = DGM_c(MLP([in_features, in_features*2, in_features]), distance="euclidian")
        self.g = GATv2Conv(in_features, in_features, edge_dim=1, add_self_loops=False)

        self.fully_connected = MLP([in_features, in_features, out_features])

    def forward(self, data):
        x = data.x                 # [N_total, d]
        edge_index = data.edge_index
        batch = data.batch         # [N_total]
        graph_x, edges, _ = self.f(x, edge_index)

        x_dense, mask = to_dense_batch(x, batch)
        # x_dense: [b, n, d]
        
        b,n,d = x_dense.shape
        edge_index, edge_weight = matrix_to_list(edges)
        self.edge_weight = edge_weight
        x = torch.nn.functional.relu(self.g(torch.dropout(x_dense.view(-1,d), 0.5, train=self.training), edge_index, edge_attr=edge_weight)).view(b,n,-1)

        # print(x.std(), edge_weight.std())

        return self.fully_connected(x), None

    def training_step(self, batch, batch_idx):

        # ---- forward PyG
        pred, logprobs = self(batch)
        # pred: [b, n, C]

        # ---- reconstruire masque dense
        y, mask = to_dense_batch(batch.y, batch.batch)
        y_labels = y.argmax(dim=-1)
        # y = batch.y
        # y_dense: [b, n, C] ou [b, n]

        # ---- loss principale
        loss = torch.nn.functional.cross_entropy(pred.view(-1,2), y_labels.view(-1), weight=torch.tensor([1.0,5.0]).to(pred.device))

        self.log("loss", loss)

        # ---- accuracy
        correct = (pred.argmax(-1) == y.argmax(-1)).float().mean()
        self.log("acc", correct)

        return loss

    # def validation_step(self, batch, batch_idx):

    #     # ---- forward PyG
    #     pred, logprobs = self(batch)
    #     # pred: [b, n, C]

    #     # ---- reconstruire masque dense
    #     y, mask = to_dense_batch(batch.y, batch.batch)
    #     # y_dense: [b, n, C] ou [b, n]

    #     # ---- loss principale
    #     loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)

    #     self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    #     # ---- accuracy
    #     correct = (pred.argmax(-1) == y.argmax(-1)).float().mean()
    #     self.log("val_acc", correct, on_step=False, on_epoch=True, prog_bar=True)

    #     return loss
            
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

class DGM_c(nn.Module):
    input_dim = 4
    debug=False
    
    def __init__(self, embed_f, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.distance = distance
        
        self.scale = nn.Parameter(torch.tensor(-1).float(),requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1,1,DGM_c.input_dim)).float(),requires_grad=False)
        
        
    def forward(self, x, A, not_used=None, fixedges=None):

        x = self.embed_f(x, A)

        if self.scale < 0:
            self.centroid.data = x.mean(-2, keepdim=True).detach()
            self.scale.data = (0.9 / (x - self.centroid).abs().max()).detach()

        D, _ = pairwise_euclidean_distances((x - self.centroid) * self.scale)

        # logits = self.temperature * (self.threshold.abs() - D)
        logits = -D
        logits = logits - logits.mean(dim=-1, keepdim=True)

        #Sample each edge according to Bernoulli
        mask = binary_concrete(
            logits,
            tau=0.5,
            hard=True
        )
        A = torch.sigmoid(-(logits))*mask

        return x, A, None

        


class MLP(nn.Module): 
    def __init__(self, layers_size,final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1,len(layers_size)):
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1],layers_size[li]))
            if li==len(layers_size)-1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
            
            
        self.MLP = nn.Sequential(*layers)
        
    def forward(self, x, e=None):
        x = self.MLP(x)
        return x
    

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
