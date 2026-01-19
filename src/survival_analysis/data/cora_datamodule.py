from typing import Optional, Any
import torch

from torch_geometric.loader import DataLoader
from lightning import LightningDataModule
from torch_geometric.datasets import Planetoid

class CoraDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "./data",
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        batch_size: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data = None
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.batch_size_per_device = batch_size

        self.data_dir = data_dir
        self.dataset = None

    # ------------------------------------------------------------------
    # PREPARE DATA: téléchargement
    # ------------------------------------------------------------------
    def prepare_data(self):
        Planetoid(root=self.data_dir, name="Cora")

    # ------------------------------------------------------------------
    # SETUP: chargement + splits + masques
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        dataset = Planetoid(root=self.data_dir, name="Cora")
        data = dataset[0]
        self.in_dims = data.x.shape[1]
        self.out_dims = dataset.num_classes
        num_nodes = data.num_nodes

        # -------------------------------
        # CAS 1 : indices fournis (K-fold)
        # -------------------------------
        if self.train_idx is not None:
            # On suppose ici que les indices fournis sont des tensors numpy/torch
            train_idx = torch.as_tensor(self.train_idx)
            val_idx = torch.as_tensor(self.val_idx) if self.val_idx is not None else None
            test_idx = torch.as_tensor(self.test_idx) if self.test_idx is not None else None

        else:
            # ------------------------------------------
            # CAS 2 : split automatique via les fractions
            # ------------------------------------------
            perm = torch.randperm(num_nodes)
            n_test = int(self.hparams.test_fraction * num_nodes)
            n_val = int(self.hparams.validation_fraction * num_nodes)

            test_idx = perm[:n_test]
            val_idx = perm[n_test:n_test + n_val]
            train_idx = perm[n_test + n_val:]

        # -------------------------------------
        # Création / remplacement des masques
        # -------------------------------------
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[train_idx] = True
        if val_idx is not None:
            data.val_mask[val_idx] = True
        if test_idx is not None:
            data.test_mask[test_idx] = True

        # Stockage
        self.data = data
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

    # ------------------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        # un DataLoader qui renvoie un seul élément : le graphe
        return DataLoader([self.data], batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader([self.data], batch_size=1, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader([self.data], batch_size=1, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader([self.data], batch_size=1, shuffle=False)

    # ------------------------------------------------------------------
    # Helpers (analogue à ton module)
    # ------------------------------------------------------------------
    def get_train_indices(self):
        return self.train_idx

    def get_val_indices(self):
        return self.val_idx

    def get_test_indices(self):
        return self.test_idx
