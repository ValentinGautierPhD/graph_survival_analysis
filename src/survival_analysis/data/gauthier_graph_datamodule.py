import pandas as pd
import numpy as np
import json
import torch
import networkx as nx
from typing import Optional
from lightning import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler

class GauthierGraphDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        json_splits_path: str,
        split_index: int = 0, # Pour choisir quel split utiliser dans la liste JSON
    ):
        super().__init__()
        # On sauvegarde les chemins dans les hyperparamètres
        self.save_hyperparameters(logger=False)
        self.csv_path = csv_path
        self.json_splits_path = json_splits_path
        self.split_index = split_index
        
        self.out_dim = 1
        self.in_dim = None

    def prepare_data(self):
        # Optionnel : On pourrait vérifier ici si les fichiers existent
        pass

    def setup(self, stage: Optional[str] = None):
        # 1. Chargement des données
        df_raw = pd.read_csv(self.csv_path)
        
        with open(self.json_splits_path, 'r') as f:
            all_splits = json.load(f)
            # On récupère le split spécifique (la liste de dict vue précédemment)
            current_split = all_splits[self.split_index]

        # 2. Préparation des features (X)
        x_raw = df_raw.drop(
            columns=["patients_id", "pfs", "pfs_event", "pfs_2_years"]
        ).to_numpy(dtype=np.float32)
        
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_raw)
        
        # 3. Préparation des labels (Y) - Classification
        arr = df_raw["pfs_2_years"].to_numpy(dtype=np.int64)
        y_one_hot = np.zeros((arr.size, arr.max() + 1), dtype=np.float32)
        y_one_hot[np.arange(arr.size), arr] = 1

        # 4. Application des splits (indices du JSON)
        train_idx = current_split["train"]
        val_idx = current_split["test"] 

        # Conversion en tensors
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        self.in_dim = x_scaled.shape[-1]

        # Train Graph (uniquement les données de train)
        self.train_graph = Data(
            x=torch.from_numpy(x_scaled[train_idx]).float(),
            y=torch.from_numpy(y_one_hot[train_idx]).float(),
            edge_index=edge_index
        )
        
        # Val Graph 
        self.val_graph = Data(
            x=torch.from_numpy(x_scaled).float(), 
            y=torch.from_numpy(y_one_hot).float(),
            edge_index=edge_index
        )
        self.val_graph.val_idx = torch.tensor(val_idx, dtype=torch.long)

    def train_dataloader(self) -> DataLoader:
        return DataLoader([self.train_graph], batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader([self.val_graph], batch_size=1, shuffle=False)


class GauthierGraphSurvivalDataModule(GauthierGraphDataModule):
    def __init__(self, csv_path: str, json_splits_path: str, split_index: int = 0, select=None):
        self.select = select
        super().__init__(csv_path, json_splits_path, split_index)

    def setup(self, stage: Optional[str] = None):
        # On recharge les données
        df_raw = pd.read_csv(self.csv_path)
        with open(self.json_splits_path, 'r') as f:
            current_split = json.load(f)[self.split_index]

        # Features
        x_raw = df_raw.drop(
            columns=["patients_id", "pfs", "pfs_event", "pfs_2_years"]
        )

        # Allows features selection
        if self.select is not None:
            x_raw = x_raw[self.select]

        x_raw.to_numpy(dtype=np.float32)
        
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_raw)
        
        # Labels spécifiques à la Survie (Time, Event)
        y_survival = df_raw[["pfs", "pfs_event"]].to_numpy(dtype=np.float32)

        train_idx = current_split["train"]
        val_idx = current_split["test"]

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = None
            
        self.in_dim = x_scaled.shape[-1]

        self.train_graph = Data(
            x=torch.from_numpy(x_scaled[train_idx]).float(),
            y=torch.from_numpy(y_survival[train_idx]).float(),
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        
        self.val_graph = Data(
            x=torch.from_numpy(x_scaled).float(),
            y=torch.from_numpy(y_survival).float(),
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        self.val_graph.val_idx = torch.tensor(val_idx, dtype=torch.long)

class GauthierPreGraph(GauthierGraphSurvivalDataModule):
    def __init__(self, vae, csv_path: str, json_splits_path: str, split_index: int = 0, select=None, k=40):
        self.vae = vae
        super().__init__(csv_path, json_splits_path, split_index, select=select)
        self.k = k
        
    def compute_graph(self, x, k=40):
        _, mu, log_var, _ = self.vae._forward_full(x)
        std = torch.exp(0.5 * log_var)

        mu_i = mu[:, None, :]
        mu_j = mu[None, :, :]

        std_i = std[:, None, :]
        std_j = std[None, :, :]

        D2 = (
            torch.sum((mu_i - mu_j) ** 2, dim=-1)
            +
            torch.sum((std_i - std_j) ** 2, dim=-1)
        )

        D = torch.sqrt(D2)

        G = knn_graph_networkx(mu, k, dist=D)

        data = from_networkx(G)
        data.weight = data.weight.float()
        return data
        
    def setup(self, stage: Optional[str] = None):
        # On recharge les données
        df_raw = pd.read_csv(self.csv_path)
        with open(self.json_splits_path, 'r') as f:
            current_split = json.load(f)[self.split_index]

        # Features
        x_raw = df_raw.drop(
            columns=["patients_id", "pfs", "pfs_event", "pfs_2_years"]
        )

        # Allows features selection
        if self.select is not None:
            x_raw = x_raw[self.select]

        x_raw.to_numpy(dtype=np.float32)
        
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_raw)
        
        # Labels spécifiques à la Survie (Time, Event)
        y_survival = df_raw[["pfs", "pfs_event"]].to_numpy(dtype=np.float32)

        train_idx = current_split["train"]
        val_idx = current_split["test"]

        # Compute graph from vae
        x_val = torch.from_numpy(x_scaled).float().to("cuda")
        x_train = x_val[train_idx]

        val_graph = self.compute_graph(x_val, k=self.k)
        val_graph.x = x_val
        val_graph.y = torch.from_numpy(y_survival).float()
        self.val_graph = val_graph

        train_graph = self.compute_graph(x_train, k=self.k)
        train_graph.x = x_train
        train_graph.y = torch.from_numpy(y_survival[train_idx]).float()
        self.train_graph = train_graph
        
        self.in_dim = x_scaled.shape[-1]

        self.val_graph.val_idx = torch.tensor(val_idx, dtype=torch.long)


def knn_graph_networkx(X, k, include_self=False, weighted=True, dist=None):
    """
    X : tensor de forme (N, D)
    k : nombre de voisins
    Retourne un graphe NetworkX
    """
    N = X.shape[0]
    if dist is None:
        dist = torch.cdist(X, X, p=2)
    
    if not include_self:
        dist.fill_diagonal_(float('inf'))

    knn_dist, knn_idx = torch.topk(dist, k, largest=False, dim=1)
    sigma = knn_dist.std().item()

    G = nx.Graph()  # ou nx.DiGraph() si vous voulez un graphe orienté
    G.add_nodes_from(range(N))

    for i in range(N):
        for j_pos in range(k):
            j = knn_idx[i, j_pos].item()
            w = knn_dist[i, j_pos].item()
            if weighted:
                G.add_edge(i, j, weight=np.exp(-(w**2)/(2*sigma**2)))
            else:
                G.add_edge(i, j)

    return G
