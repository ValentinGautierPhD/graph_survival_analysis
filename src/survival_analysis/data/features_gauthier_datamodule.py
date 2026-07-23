import pandas as pd
import numpy as np
import json
import torch
from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class GauthierDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        json_splits_path: str,
        split_index: int = 0,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.csv_path = csv_path
        self.json_splits_path = json_splits_path
        self.split_index = split_index
        self.batch_size = batch_size

        self.out_dim = 1
        self.in_dim = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        df_raw = pd.read_csv(self.csv_path)

        with open(self.json_splits_path, 'r') as f:
            current_split = json.load(f)[self.split_index]

        x_raw = df_raw.drop(
            columns=["patients_id", "pfs", "pfs_event", "pfs_2_years"]
        ).to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_raw)

        arr = df_raw["pfs_2_years"].to_numpy(dtype=np.int64)
        y_one_hot = np.zeros((arr.size, arr.max() + 1), dtype=np.float32)
        y_one_hot[np.arange(arr.size), arr] = 1

        train_idx = current_split["train"]
        val_idx = current_split["test"]

        self.in_dim = x_scaled.shape[-1]

        x_tensor = torch.from_numpy(x_scaled).float()
        y_tensor = torch.from_numpy(y_one_hot).float()

        self.train_dataset = TensorDataset(
            x_tensor[train_idx],
            y_tensor[train_idx],
        )
        self.val_dataset = TensorDataset(
            x_tensor[val_idx],
            y_tensor[val_idx],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


class GauthierSurvivalDataModule(GauthierDataModule):
    def __init__(
        self,
        csv_path: str,
        json_splits_path: str,
        split_index: int = 0,
        batch_size: int = 32,
        select=None,
    ):
        self.select = select
        super().__init__(csv_path, json_splits_path, split_index, batch_size)

    def setup(self, stage: Optional[str] = None):
        df_raw = pd.read_csv(self.csv_path)

        with open(self.json_splits_path, 'r') as f:
            current_split = json.load(f)[self.split_index]

        x_raw = df_raw.drop(
            columns=["patients_id", "pfs", "pfs_event", "pfs_2_years"]
        )

        if self.select is not None:
            x_raw = x_raw[self.select]

        x_raw = x_raw.to_numpy(dtype=np.float32)

        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_raw)

        y_survival = df_raw[["pfs", "pfs_event"]].to_numpy(dtype=np.float32)

        train_idx = current_split["train"]
        val_idx = current_split["test"]

        self.in_dim = x_scaled.shape[-1]

        x_tensor = torch.from_numpy(x_scaled).float()
        y_tensor = torch.from_numpy(y_survival).float()

        self.train_dataset = TensorDataset(
            x_tensor[train_idx],
            y_tensor[train_idx],
        )
        self.val_dataset = TensorDataset(
            x_tensor[val_idx],
            y_tensor[val_idx],
        )
