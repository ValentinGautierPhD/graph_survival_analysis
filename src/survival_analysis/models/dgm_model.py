#!/usr/bin/env python3

from torch import nn
import torch
import torch_geometric
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATv2Conv
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import dense_to_sparse
import lightning as pl



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
        edge_index, edge_weight = dense_to_sparse(edges)
        self.edge_weight = edge_weight
        x = torch.nn.functional.relu(self.g(torch.dropout(x_dense.view(-1,d), 0.5, train=self.training), edge_index, edge_attr=edge_weight)).view(b,n,-1)

        print(x.std(), edge_weight.std())

        return self.fully_connected(x).view(b,n), None

    def training_step(self, batch, batch_idx):

        # ---- forward PyG
        pred, logprobs = self(batch)
        # pred: [b, n, C]

        # ---- reconstruire masque dense
        y, mask = to_dense_batch(batch.y, batch.batch)
        # y_dense: [b, n, C] ou [b, n]

        # ---- loss principale
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)

        self.log("loss", loss)

        # ---- accuracy
        correct = (pred.argmax(-1) == y.argmax(-1)).float().mean()
        self.log("acc", correct)

        return loss

            
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
        
        x = self.embed_f(x,A)  
        
        # estimate normalization parameters
        if self.scale <0:            
            self.centroid.data = x.mean(-2,keepdim=True).detach()
            self.scale.data = (0.9/(x-self.centroid).abs().max()).detach()
        
        D, _x = pairwise_euclidean_distances((x-self.centroid)*self.scale)
            
        A = torch.sigmoid(self.temperature*(self.threshold.abs()-D))

        #Top k pour sparsifier
        k = 10  # ou 5, ou sqrt(n)
        vals, idx = torch.topk(A, k, dim=-1)

        mask = torch.zeros_like(A)
        mask.scatter_(-1, idx, 1)

        A = A * mask
        
        if DGM_c.debug:
            self.A = A.data.cpu()
            self._x = _x.data.cpu()
            
#         self.A=A
#         A = A/A.sum(-1,keepdim=True)
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
