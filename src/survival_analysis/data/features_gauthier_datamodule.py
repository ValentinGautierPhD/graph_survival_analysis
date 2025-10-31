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
        batch_size: int = 256,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
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


    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            features_df = pd.read_csv(self.data_path)

            # Nettoyage pour ne récupérer que ce qui m'intéresse
            clin_features_df = features_df.filter(like="clin_")
            clin_features_df = pd.concat([clin_features_df, features_df[['patients_id',"pfs_event","pfs"]]], axis=1)
            clin_features_df = clin_features_df.drop_duplicates()
            clin_features_df = clin_features_df.drop(columns=["patients_id"])
            # séparation en train val test
            df_train = clin_features_df.copy()
            df_test = df_train.sample(frac=self.test_fraction)
            df_train = df_train.drop(df_test.index)
            df_val = df_train.sample(frac=self.validation_fraction)
            df_train = df_train.drop(df_val.index)

            # Séparation features labels
            self.x_train = torch.from_numpy(df_train.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32"))
            self.x_test = torch.from_numpy(df_test.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32"))
            self.x_val = torch.from_numpy(df_val.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32"))

            if stage == 'fit' or stage is None:
                # Setup targets (duration, event)
                self.y_train = torch.from_numpy(np.concatenate(self._get_target(df_train), axis=1))
                self.y_val = torch.from_numpy(np.concatenate(self._get_target(df_val), axis=1))

                # Create training and validation datasets
                self.train_set = TensorDataset(self.x_train, self.y_train)
                self.val_set = TensorDataset(self.x_val, self.y_val)

                # Input and output dimensions for building net
                self.in_dims = self.x_train.shape[1]
                self.out_dims = 1

            if stage == 'test' or stage is None:    
                # Returns correctly preprocessed target y_test {torch.Tensor} and entire df_test {pd.DataFrame} for metric calculations     
                self.y_test = torch.from_numpy(np.concatenate(self._get_target(df_test), axis=1))
                self.df_test = df_test

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
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=1,
            shuffle=False,
        )

    def get_training(self):
        x_train = self.x_train.numpy()
        y_train = (self.y_train.numpy()[:,0], self.y_train.numpy()[:,1]) 
        return x_train, y_train
        
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
