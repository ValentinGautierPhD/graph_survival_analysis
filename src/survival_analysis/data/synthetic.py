#!/usr/bin/env python3

import torch
import numpy as np
import networkx as nx
from lightning import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


class SyntheticGraphClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        n_nodes: int = 300,
        n_features: int = 16,
        n_classes: int = 2,
        alpha: float = 0.8,       # 0 = graphe inutile, 1 = graphe indispensable
        noise: float = 0.5,       # bruit sur les features
        p_in: float = 0.15,       # proba arête intra-communauté
        p_out: float = 0.01,      # proba arête inter-communauté
        val_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.out_dim = 1
        self.in_dim = n_features

    def setup(self, stage=None):
        hp = self.hparams
        rng = np.random.default_rng(hp.seed)

        # 1. Graphe SBM
        sizes = [hp.n_nodes // hp.n_classes] * hp.n_classes
        P = [
            [hp.p_in if i == j else hp.p_out for j in range(hp.n_classes)]
            for i in range(hp.n_classes)
        ]
        G = nx.stochastic_block_model(sizes, P, seed=hp.seed)
        A = nx.to_numpy_array(G)
        self.true_graph = G  # utile pour évaluer la récupération du graphe

        # 2. Features
        true_labels = np.array([G.nodes[i]["block"] for i in G.nodes()])
        centers = rng.normal(0, 1, size=(hp.n_classes, hp.n_features))
        # X = centers[true_labels] + rng.normal(0, hp.noise, size=(hp.n_nodes, hp.n_features))
        X = centers[true_labels] + rng.normal(0, hp.noise, size=(hp.n_nodes, hp.n_features))

        # 3. Labels (mélange features + voisinage)
        W_cls = rng.normal(0, 1, size=(hp.n_features, hp.n_classes))
        logits_self = X @ W_cls
        true_onehot = np.eye(hp.n_classes)[true_labels]
        degrees = A.sum(axis=1, keepdims=True).clip(min=1)
        neigh_signal = (A @ true_onehot) / degrees
        logits_final = (1 - hp.alpha) * logits_self + hp.alpha * neigh_signal
        y = logits_final.argmax(axis=1)

        y_labels = y.astype(np.int64).reshape(-1,1)

        # 4. Split train/val
        all_idx = np.arange(hp.n_nodes)
        train_idx, val_idx = train_test_split(
            all_idx, test_size=hp.val_size, random_state=hp.seed, stratify=y
        )

        edge_index = torch.empty((2, 0), dtype=torch.long)
        X_scaled = (X - X[train_idx].mean(0)) / (X[train_idx].std(0) + 1e-8)

        # 5. Graphs (même structure que l'original)
        self.train_graph = Data(
            x=torch.from_numpy(X_scaled[train_idx]).float(),
            y=torch.from_numpy(y_labels[train_idx]).float(),
            edge_index=edge_index,
        )

        self.val_graph = Data(
            x=torch.from_numpy(X_scaled).float(),
            y=torch.from_numpy(y_labels).float(),
            edge_index=edge_index,
        )
        self.val_graph.val_idx = torch.tensor(val_idx, dtype=torch.long)

        # Utile pour debug
        self.true_labels = true_labels
        self.true_A = torch.tensor(A, dtype=torch.float32)

        print(f"Nodes: {hp.n_nodes} | Edges: {G.number_of_edges()}")
        print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"Label agreement with community: {(y == true_labels).mean():.2%}")
        print(set(val_idx) & set(train_idx))

    def train_dataloader(self):
        return DataLoader([self.train_graph], batch_size=1, shuffle=False)

    def val_dataloader(self):
        return DataLoader([self.val_graph], batch_size=1, shuffle=False)


class SyntheticGraphRecoveryDataModule(LightningDataModule):

    def __init__(
        self,
        n_nodes: int = 300,
        p_edge: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.in_dim = n_nodes
        self.out_dim = n_nodes

    def setup(self, stage=None):

        hp = self.hparams
        rng = np.random.default_rng(hp.seed)

        # --------------------------------------------------
        # 1. Graphe aléatoire
        # --------------------------------------------------
        G = nx.erdos_renyi_graph(
            n=hp.n_nodes,
            p=hp.p_edge,
            seed=hp.seed,
        )

        A = nx.to_numpy_array(G).astype(np.float32)

        self.true_graph = G
        self.true_A = torch.tensor(A)

        # --------------------------------------------------
        # 2. Features identité
        # --------------------------------------------------
        X = np.eye(hp.n_nodes, dtype=np.float32)

        # --------------------------------------------------
        # 3. Cibles = somme des features voisines
        # --------------------------------------------------
        Y = A

        # équivalent à :
        # Y = A

        # --------------------------------------------------
        # 4. Split train / val
        # --------------------------------------------------
        all_idx = np.arange(hp.n_nodes)
        train_idx, val_idx = all_idx, all_idx

        edge_index = torch.empty((2, 0), dtype=torch.long)

        # --------------------------------------------------
        # 5. Graphes PyG
        # --------------------------------------------------
        self.train_graph = Data(
            x=torch.from_numpy(X[train_idx]).float(),
            y=torch.from_numpy(Y[train_idx]).float(),
            edge_index=edge_index,
        )

        self.val_graph = Data(
            x=torch.from_numpy(X).float(),
            y=torch.from_numpy(Y).float(),
            edge_index=edge_index,
        )

        self.val_graph.val_idx = torch.tensor(
            val_idx,
            dtype=torch.long,
        )

        print(
            f"Nodes: {hp.n_nodes} | "
            f"Edges: {G.number_of_edges()}"
        )

        print(
            f"Train: {len(train_idx)} | "
            f"Val: {len(val_idx)}"
        )

    def train_dataloader(self):
        return DataLoader(
            [self.train_graph],
            batch_size=1,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            [self.val_graph],
            batch_size=1,
            shuffle=False,
        )
