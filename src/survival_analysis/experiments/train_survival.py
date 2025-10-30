from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from ..utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pathlib import Path
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

import pycox
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train_survival.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    # # train the model
    # metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value

    seed = 123
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

    # Chargement du fichier de données
    data_path = cfg.data.data_dir
    features_df = pd.read_csv(data_path)

    # Nettoyage pour ne récupérer que ce qui m'intéresse
    clin_features_df = features_df.filter(like="clin_")
    clin_features_df = pd.concat([clin_features_df, features_df[['patients_id',"pfs_event","pfs"]]], axis=1)
    clin_features_df = clin_features_df.drop_duplicates()
    clin_features_df = clin_features_df.drop(columns=["patients_id"])

    # séparation en train val test
    df_train = clin_features_df.copy()
    df_test = df_train.sample(frac=cfg.data.test_fraction)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=cfg.data.validation_fraction)
    df_train = df_train.drop(df_val.index)

    # Séparation features labels
    x_train = df_train.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32")
    x_test = df_test.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32")
    x_val = df_val.drop(columns=["pfs","pfs_event"]).to_numpy(dtype="float32")

    get_target = lambda df: (df['pfs'].to_numpy(dtype="float32"), df['pfs_event'].to_numpy(dtype="float32"))
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = x_val, y_val

    # Définition du modèle
    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                dropout, output_bias=output_bias)

    model = CoxPH(net, tt.optim.Adam)

    batch_size = 256
    # lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
    # _ = lrfinder.plot()
    model.optimizer.set_lr(0.05)

    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True

    # Entraînement
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val, val_batch_size=batch_size)

    # Evaluation
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)

    # plot survival curves
    # surv.iloc[:, :5].plot()
    # plt.ylabel('S(t | x)')
    # _ = plt.xlabel('Time')

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    # C-index
    print(ev.concordance_td())

    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    _ = ev.brier_score(time_grid).plot()

    # Score intégrés (ne marche pas si scipy trop récent à cause du changement depuis scipy.integrals.simps vers scipy.integral.simpson)
    print(ev.integrated_brier_score(time_grid))
    print(ev.integrated_nbll(time_grid))


if __name__ == "__main__":
    main()
