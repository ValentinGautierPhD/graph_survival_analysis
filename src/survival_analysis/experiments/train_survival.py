
from typing import Optional
import hydra
from omegaconf import DictConfig

from ..utils import (
    RankedLogger,
)
from ..utils.train_test import train

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

    train(cfg, log)
    
if __name__ == "__main__":
    main()
