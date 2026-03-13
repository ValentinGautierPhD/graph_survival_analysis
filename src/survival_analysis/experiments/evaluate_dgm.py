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

    loader = hydra.utils.instantiate(cfg.data.loader)

    splitter = hydra.utils.instantiate(cfg.data.splitter, size=loader.size)

    for train_idx, val_idx in splitter.split():

        splits = {
            "train": train_idx,
            "val": val_idx,
        }

        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, data=loader.data, splits=splits)
        datamodule.prepare_data()
        datamodule.setup()

        model: LightningModule = hydra.utils.instantiate(cfg.model, in_features=datamodule.in_dims, out_features=datamodule.out_dims)

        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        trainer.test(model, dataloaders=datamodule.val_dataloader)
    
if __name__ == "__main__":
    main()

