from typing import Optional, Any
import torch
import numpy as np

from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
from lightning import LightningDataModule
from torch_geometric.data import Data
import pandas as pd 

class GauthierGraphDataModule(LightningDataModule):

    def __init__(
        self,
        data,
        splits: dict,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data = data
        self.splits = splits
        self.out_dim = 1

    # ------------------------------------------------------------------
    # PREPARE DATA: téléchargement
    # ------------------------------------------------------------------
    def prepare_data(self):
        pass

    # ------------------------------------------------------------------
    # SETUP: chargement + splits + masques
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        x = self.data.drop(
            columns=["patients_id", "pfs", "pfs_event", "pfs_2_years"]
        ).to_numpy(dtype=np.float32)

        y = self.data["pfs_2_years"].to_numpy(dtype=np.int64)

        x_train = x[self.splits["train"]]
        y_train = y[self.splits["train"]]
        
        edge_index = torch.empty((2,0), dtype=torch.long)

        self.in_dim = x_train.shape[-1]
        self.train_graph = Data(x=torch.from_numpy(x_train).float(), y=torch.from_numpy(y_train).float(), edge_index=edge_index)
        self.val_graph = Data(x=torch.from_numpy(x), y=torch.from_numpy(y), edge_index=edge_index)


    # ------------------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        # un DataLoader qui renvoie un seul élément : le graphe
        return DataLoader([self.train_graph], batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader([self.val_graph], batch_size=1, shuffle=False)

