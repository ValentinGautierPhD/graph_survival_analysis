from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split


class FeaturesGauthierDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            data_dir: str,
            validation_fraction: float = 0.2,
            test_fraction: float = 0.2,
            train_idx=None,
            val_idx=None,
            test_idx=None,
            batch_size: int = 256,
    ) -> None:
        """Initialize the data.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction

        self.data_train: Optional[TensorDataset] = None
        self.data_val: Optional[TensorDataset] = None
        self.data_test: Optional[TensorDataset] = None

        self.batch_size_per_device = batch_size
        # Chargement du fichier de données
        self.data_path = data_dir

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # ---------------------------------------------------
        # 1. Charger et nettoyer les données (une seule fois)
        # ---------------------------------------------------
        features_df = pd.read_csv(self.data_path)

        clin_features_df = features_df.filter(regex="(clin_.+)")
        clin_features_df = pd.concat(
            [clin_features_df, features_df[['patients_id', "pfs_event", "pfs"]]],
            axis=1
        ).drop_duplicates()

        clin_features_df = clin_features_df.drop(columns=["patients_id"])
        self.full_df = clin_features_df  # je garde ça si ça sert plus tard

        # Convertit en numpy pour éviter duplication plus tard
        full_x = clin_features_df.drop(columns=["pfs", "pfs_event"]).to_numpy(dtype="float32")
        full_y = np.concatenate(self._get_target(clin_features_df), axis=1)

        # ---------------------------------------------------
        # 2. Cas où des indices sont fournis (K-Fold)
        # ---------------------------------------------------
        if self.train_idx is not None:
            # Tensorisation sélective
            self.x_train = torch.from_numpy(full_x[self.train_idx])
            self.y_train = torch.from_numpy(full_y[self.train_idx])

            if self.val_idx is not None:
                self.x_val = torch.from_numpy(full_x[self.val_idx])
                self.y_val = torch.from_numpy(full_y[self.val_idx])
            
            if self.test_idx is not None:    
                self.x_test = torch.from_numpy(full_x[self.test_idx])
                self.y_test = torch.from_numpy(full_y[self.test_idx])

        else:
            # ---------------------------------------------------
            # 3. Cas standard : split via fractions
            # ---------------------------------------------------
            df = clin_features_df.copy()

            df_test = df.sample(frac=self.test_fraction)
            df_trainval = df.drop(df_test.index)

            df_val = df_trainval.sample(frac=self.validation_fraction)
            df_train = df_trainval.drop(df_val.index)

            # Convertir en tenseurs
            self.x_train = torch.from_numpy(df_train.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32"))
            self.y_train = torch.from_numpy(np.concatenate(self._get_target(df_train), axis=1))

            self.x_val = torch.from_numpy(df_val.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32"))
            self.y_val = torch.from_numpy(np.concatenate(self._get_target(df_val), axis=1))

            self.x_test = torch.from_numpy(df_test.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32"))
            self.y_test = torch.from_numpy(np.concatenate(self._get_target(df_test), axis=1))

        # ---------------------------------------------------
        # 4. Datasets selon le stage
        # ---------------------------------------------------
        if stage in ('fit', None):
            self.train_set = TensorDataset(self.x_train, self.y_train)
            if hasattr(self, "x_val"):
                self.val_set = TensorDataset(self.x_val, self.y_val)

            self.in_dims = self.x_train.shape[1]
            self.out_dims = 1

        if stage in ('test', None):
            self.test_set = TensorDataset(self.x_test, self.y_test)

    def _get_target(cls, df : pd.DataFrame) -> np.ndarray:
        '''
        Takes pandas datframe and converts the duration, event targets into np.arrays 
        '''
        duration = df['pfs'].to_numpy().reshape(len(df['pfs']),1)
        event = df['pfs_event'].to_numpy().reshape(len(df['pfs_event']),1)

        return duration, event

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size_per_device,
            num_workers=1,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size_per_device,
            num_workers=1,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size_per_device,
            num_workers=1,
            shuffle=False,
        )

    # --- Helpers pour PyCox ---
    def get_training_data(self):
        """Helper pour compute_baseline_hazards"""
        return self.x_train.numpy(), (self.y_train.numpy()[:, 0], self.y_train.numpy()[:, 1])

    def get_test_data(self):
        """Helper pour l'évaluation finale (concordance, brier)"""
        # Renvoie X_test et (durations, events)
        return self.x_val.numpy(), self.y_val.numpy()[:, 0], self.y_val.numpy()[:, 1]
