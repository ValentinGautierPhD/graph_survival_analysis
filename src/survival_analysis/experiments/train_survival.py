from typing import Any, Dict, List, Optional, Tuple

from lightning.pytorch import LightningModule, Trainer, Callback, LightningDataModule
from lightning.pytorch.loggers import Logger
import torch
import hydra
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

import torch

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
    # if cfg.get("seed"):
    #     L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup('fit') #allows for input / output features to be configured in the model

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, in_features=datamodule.in_dims, out_features=datamodule.out_dims)

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
        # trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        
        log.info(f"Best ckpt path: {ckpt_path}")
        
        datamodule.setup("test")
        model_surv = CoxPH(model)
        x_train = datamodule.x_train.numpy()
        y_train = (datamodule.y_train.numpy()[:,0], datamodule.y_train.numpy()[:,1]) 
        durations, events = datamodule.y_test[:,0].numpy(), datamodule.y_test[:,1].numpy()

        _ = model_surv.compute_baseline_hazards(x_train, y_train)
        surv = model_surv.predict_surv_df(datamodule.x_test)
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        time_grid = np.linspace(durations.min(), durations.max(), 100)

        print(f"Concordance: {ev.concordance_td()}")
        print(f"Brier Score: {ev.integrated_brier_score(time_grid)}")
        test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics}

    
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

    train(cfg)
    
if __name__ == "__main__":
    main()
