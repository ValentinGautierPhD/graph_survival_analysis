from typing import Optional
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


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train_graphs.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    log.info(f"Loading data with loader <{cfg.data._target_}>")
    loader = hydra.utils.instantiate(cfg.data.loader)

    log.info("Splitting data between train and validation")
    splitter = hydra.utils.instantiate(cfg.data.splitter, size=loader.size)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, data=loader.data, splits=splitter.splits)
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

    # merge train and test metrics
    metric_dict = {**train_metrics}

    
    return metric_dict, object_dict
    
if __name__ == "__main__":
    main()
