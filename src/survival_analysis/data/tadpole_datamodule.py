import pandas as pd
import numpy as np
import json
import torch
from typing import Optional
from lightning import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

class TadpoleGraphDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        json_splits_path: str,
        split_index: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.csv_path = csv_path
        self.json_splits_path = json_splits_path
        self.split_index = split_index

        self.out_dim = 3
        self.in_dim = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        df_raw = pd.read_csv(self.csv_path)

        with open(self.json_splits_path, 'r') as f:
            current_split = json.load(f)[self.split_index]

        x_raw = df_raw.drop(
            columns=["LABEL"]
        ).to_numpy(dtype=np.float32)

        y = df_raw["LABEL"].values - 1

        train_idx = current_split["train"]
        val_idx = current_split["test"]

        self.in_dim = x_raw.shape[-1]

        x_tensor = torch.from_numpy(x_raw).float()
        y_tensor = torch.from_numpy(y).long()

        self.train_graph = Data(
            x=x_tensor[train_idx],
            y=y_tensor[train_idx],
        )
        self.val_graph = Data(
            x=x_tensor,
            y=y_tensor,
        )
        self.val_graph.val_idx = torch.tensor(val_idx, dtype=torch.long)

    def train_dataloader(self) -> DataLoader:
        return DataLoader([self.train_graph], batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader([self.val_graph], batch_size=1, shuffle=False)


class TadpoleVAEDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        json_splits_path: str,
        split_index: int = 0,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_train: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.csv_path = csv_path
        self.json_splits_path = json_splits_path
        self.split_index = split_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

        self.out_dim = 3
        self.in_dim = None

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        df_raw = pd.read_csv(self.csv_path)

        with open(self.json_splits_path, "r") as f:
            current_split = json.load(f)[self.split_index]

        x_raw = df_raw.drop(columns=["LABEL"]).to_numpy(dtype=np.float32)
        y = df_raw["LABEL"].values - 1

        train_idx = current_split["train"]
        val_idx = current_split["test"]

        self.in_dim = x_raw.shape[-1]

        x_tensor = torch.from_numpy(x_raw).float()
        y_tensor = torch.from_numpy(y).long()

        self.train_dataset = TensorDataset(
            x_tensor[train_idx], y_tensor[train_idx]
        )
        self.val_dataset = TensorDataset(
            x_tensor[val_idx], y_tensor[val_idx]
        )

    def train_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
