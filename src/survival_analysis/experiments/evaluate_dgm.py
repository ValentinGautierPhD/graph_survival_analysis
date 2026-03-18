from typing import Optional
import numpy as np
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import hydra
from omegaconf import DictConfig
from lightning.pytorch import LightningModule, Trainer, Callback, LightningDataModule
from lightning.pytorch.loggers import Logger
from typing import Any, Dict, List, Optional, Tuple

from ..utils import (
    RankedLogger,
)

from ..utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="experiment/eval_dgm.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)
    print("instantiating loader")
    loader = hydra.utils.instantiate(cfg.data.loader)
    
    print("instantiating splitter")
    splitter = hydra.utils.instantiate(cfg.data.splitter, size=loader.size)

    all_results_concordance = []
    all_results_brier = []

    print("in loop")
    for i,(train_idx, val_idx) in enumerate(splitter.splits):

        splits = {
            "train": train_idx,
            "val": val_idx,
        }

        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, data=loader.data, splits=splits)
        datamodule.prepare_data()
        datamodule.setup()

        model: LightningModule = hydra.utils.instantiate(cfg.model, in_dim=datamodule.in_dim)
        
        if "wandb" in cfg.logger:
            # On définit un ID unique pour ce run de fold
            cfg.logger.wandb.group = f"exp_{cfg.model._target_.split('.')[-1]}" # Groupe commun
            cfg.logger.wandb.name = f"fold_{i}" # Nom spécifique au fold
            cfg.logger.wandb.job_type = "cross-val"

        logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        results_concordance = []
        results_brier = []
        nb_tests = 100

        survival_model = CoxPH(model)

        for _ in range(nb_tests):
            _ = survival_model.compute_baseline_hazards(datamodule.train_graph.x, (datamodule.train_graph.y[...,0],datamodule.train_graph.y[...,1]))
            surv = survival_model.predict_surv_df(datamodule.val_graph.x)
            y_test = datamodule.val_graph.y
            durations_test, events_test = y_test[...,0].numpy(), y_test[..., 1].numpy()

            ev = EvalSurv(surv[datamodule.val_graph.val_idx], durations_test[datamodule.val_graph.val_idx], events_test[datamodule.val_graph.val_idx], censor_surv='km')
            time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

            results_concordance.append(ev.concordance_td())
            results_brier.append(ev.integrated_brier_score(time_grid))

        all_results_concordance.append(np.mean(results_concordance))
        all_results_brier.append(np.mean(results_brier))

    print(f"C-index ||| mean: {np.mean(all_results_concordance)} | std: {np.std(all_results_concordance)}")
    print(f"Brier ||| mean: {np.mean(all_results_brier)} | std: {np.std(all_results_brier)}")

if __name__ == "__main__":
    main()

