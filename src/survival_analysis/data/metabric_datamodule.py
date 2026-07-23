#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import torch
from typing import Optional
from lightning import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pycox.datasets import metabric

class MetabricGraphSurvivalDataModule(LightningDataModule):
    def __init__(
        self,
        json_splits_path: str,
        split_index: int = 0, # Pour choisir quel split utiliser dans la liste JSON
    ):
        super().__init__()
        # On sauvegarde les chemins dans les hyperparamètres
        self.save_hyperparameters(logger=False)
        self.json_splits_path = json_splits_path
        self.split_index = split_index
        
        self.out_dim = 1
        self.in_dim = None

    def prepare_data(self):
        # Optionnel : On pourrait vérifier ici si les fichiers existent
        pass

    def setup(self, stage: Optional[str] = None):
        # 1. Chargement des données
        df_raw = metabric.read_df()
        with open(self.json_splits_path, 'r') as f:
            all_splits = json.load(f)
            current_split = all_splits[self.split_index]

        # 2. Récupération des indices
        train_idx = current_split["train"]
        val_idx = current_split["test"]

        # 3. Préparation des features (X)
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']

        preprocessor = ColumnTransformer(transformers=[
            ('scale', StandardScaler(), cols_standardize),
            ('passthrough', 'passthrough', cols_leave)
        ])

        # Fit sur le train uniquement, transform sur tout
        X = df_raw[cols_standardize + cols_leave]
        preprocessor.fit(X.iloc[train_idx])
        x_scaled_train = preprocessor.transform(X.iloc[train_idx])
        x_scaled_all = preprocessor.transform(X)  # pour le val_graph qui voit tout

        # 4. Préparation des targets (y)
        # À adapter selon ton cas (ici on suppose duration + event)
        durations = df_raw['duration'].values
        events = df_raw['event'].values
        y_one_hot = np.stack([durations, events], axis=1)

        # 5. Construction des graphes
        edge_index = torch.empty((2, 0), dtype=torch.long)

        self.in_dim = x_scaled_train.shape[-1]

        # Train Graph (uniquement les données de train)
        self.train_graph = Data(
            x=torch.from_numpy(x_scaled_train).float(),
            y=torch.from_numpy(y_one_hot[train_idx]).float(),
            edge_index=edge_index
        )

        # Val Graph (toutes les données, métriques calculées sur val_idx)
        self.val_graph = Data(
            x=torch.from_numpy(x_scaled_all).float(),
            y=torch.from_numpy(y_one_hot).float(),
            edge_index=edge_index
        )
        self.val_graph.val_idx = torch.tensor(val_idx, dtype=torch.long)

    def train_dataloader(self) -> DataLoader:
        return DataLoader([self.train_graph], batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader([self.val_graph], batch_size=1, shuffle=False)
