#!/usr/bin/env python3


from typing import Any, Dict, List, Optional, Tuple

from lightning.pytorch import LightningModule, Trainer, Callback, LightningDataModule
from lightning.pytorch.loggers import Logger
from sklearn.model_selection import KFold
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
from pathlib import Path

import torch

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

log = RankedLogger(__name__, rank_zero_only=True)

def test(model, datamodule):
    
    model_surv = CoxPH(model)
    x_train = datamodule.x_train.numpy()
    y_train = (datamodule.y_train.numpy()[:,0], datamodule.y_train.numpy()[:,1]) 
    durations, events = datamodule.y_test[:,0].numpy(), datamodule.y_test[:,1].numpy()

    _ = model_surv.compute_baseline_hazards(x_train, y_train)
    surv = model_surv.predict_surv_df(datamodule.x_test)
    ev = EvalSurv(surv, durations, events, censor_surv='km')
    time_grid = np.linspace(durations.min(), durations.max(), 100)

    return [ev.concordance_td(), ev.integrated_brier_score(time_grid)]
    
def eval(cfg: DictConfig) -> None:
    """
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    #     L.seed_everything(cfg.seed, workers=True)

    kf = KFold(n_splits=5, shuffle=True)
    all_data = hydra.utils.instantiate(cfg.data, validation_fraction=0, test_fraction=0)
    all_data.setup("fit")
    n = len(all_data.train_set)
    metrics = list()
    
    for fold_id, (tr_idx, val_idx) in enumerate(kf.split(range(n))):
    
        datamodule = hydra.utils.instantiate(cfg.data, train_idx=tr_idx, test_idx=val_idx)
        datamodule.setup('fit') #allows for input / output features to be configured in the model

        model = hydra.utils.instantiate(cfg.model, in_features=datamodule.in_dims, out_features=datamodule.out_dims)
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        metrics.append(test(model, datamodule))

    results = pd.DataFrame(metrics, columns=["c_index", "integrated_brier_score"])
    save_path = Path(cfg.paths.output_dir) / "results.csv"
    results.to_csv(save_path)

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval_survival.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    eval(cfg)
    
if __name__ == "__main__":
    main()
