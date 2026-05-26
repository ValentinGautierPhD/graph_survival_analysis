import hydra
import wandb
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import Optional
from lightning.pytorch import Callback, LightningModule, Trainer, LightningDataModule
from lightning.pytorch.loggers import Logger

from ..utils import (
    RankedLogger,
    instantiate_loggers,
    instantiate_callbacks,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="experiment/eval_dgm.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for a single fold evaluation.
    Le split à utiliser est défini dans cfg.data.split_index.
    """

    # 1. Instanciation du DataModule (qui gère CSV + JSON en interne)
    # On s'assure que cfg.data.datamodule contient les chemins csv_path et json_splits_path
    log.info(f"Instantiating datamodule for split index: {cfg.data.split_index}")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)
    
    # On prépare manuellement pour récupérer in_dim avant l'instanciation du modèle
    datamodule.prepare_data()
    datamodule.setup()

    # 2. Instanciation du Modèle
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, in_dim=datamodule.in_dim)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    
    # 3. Gestion spécifique du Logger (W&B)
    if "wandb" in cfg.logger:
        # On injecte dynamiquement le fold dans le nom du run
        cfg.logger.wandb.name = f"fold_{cfg.data.split_index}"
        cfg.logger.wandb.job_type = "single-fold"

    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # 4. Instanciation du Trainer et Entraînement
    log.info("Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)

    # Log des hyperparamètres vers le logger
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }
    if loggers:
        log_hyperparameters(object_dict)

    log.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    log.info("Starting survival evaluation...")
    metrics = model.evaluate(datamodule)
    metrics["fold_index"] = cfg.data.split_index  # injecté ici, pas dans le modèle

    # 3. Envoi au(x) logger(s)
    if loggers:
        for logger in loggers:
            # On utilise log_metrics pour envoyer les résultats de fin de run
            logger.log_metrics(metrics)
            
    log.info(f"Final Results for Fold {cfg.data.split_index} sent to loggers.")
    
    if "wandb" in cfg.logger:
        wandb.finish()   

    return 1

if __name__ == "__main__":
    main()
