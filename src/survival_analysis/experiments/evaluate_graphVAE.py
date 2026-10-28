import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
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


def _load_best_weights(model: LightningModule, trainer: Trainer, stage: str) -> None:
    """Recharge les meilleurs poids du checkpoint callback s'il y en a un."""
    ckpt_cb = trainer.checkpoint_callback
    best_ckpt = ckpt_cb.best_model_path if ckpt_cb is not None else ""
    if best_ckpt:
        log.info(f"[{stage}] Loading best weights: {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        log.warning(f"[{stage}] No checkpoint found, using final model state")


def _finish_wandb() -> None:
    if wandb.run is not None:
        wandb.finish()


# --------------------------------------------------------------------------- #
# Étape 1 : VAE (code d'origine, inchangé sur le fond)
# --------------------------------------------------------------------------- #
def train_vae(cfg: DictConfig) -> LightningModule:
    log.info(f"Instantiating datamodule for split index: {cfg.data.split_index}")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.prepare_data()
    datamodule.setup()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, in_dim=datamodule.in_dim)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    if "wandb" in cfg.logger:
        cfg.logger.wandb.name = f"fold_{cfg.data.split_index}"
        cfg.logger.wandb.job_type = "single-fold"

    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)

    if loggers:
        log_hyperparameters(
            {"cfg": cfg, "datamodule": datamodule, "model": model, "trainer": trainer}
        )

    log.info("Starting VAE training...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    _load_best_weights(model, trainer, "vae")

    log.info("Starting survival evaluation...")
    metrics = model.evaluate(datamodule)
    metrics["fold_index"] = cfg.data.split_index
    for logger in loggers:
        logger.log_metrics(metrics)
    log.info(f"VAE results for fold {cfg.data.split_index} sent to loggers.")

    return model


# --------------------------------------------------------------------------- #
# Étape 2 : graphe latent + GNN  (configs préfixées graph_*)
# --------------------------------------------------------------------------- #
def train_gnn(cfg: DictConfig, vae: LightningModule) -> None:
    split_index = cfg.data.split_index

    # Le VAE sert uniquement d'encodeur : GPU si dispo, eval, gelé.
    # (Lightning peut avoir repassé le modèle sur CPU après fit.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # GauthierPreGraph(vae, csv, splits, split_index) : `_partial_: true` dans la config,
    # le VAE (déjà entraîné en mémoire) est passé en premier argument positionnel.
    log.info("Building latent graph datamodule...")
    graph_datamodule: LightningDataModule = hydra.utils.instantiate(cfg.graph_data.datamodule)(vae)
    graph_datamodule.setup()

    log.info(f"Instantiating GNN <{cfg.graph_model._target_}>")
    gnn: LightningModule = hydra.utils.instantiate(cfg.graph_model, vae=vae, in_dim=graph_datamodule.in_dim)

    # Callbacks NEUFS : ne jamais réutiliser ceux du VAE
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("graph_callbacks"))

    # Copie autonome de la config logger, pour ne pas toucher à cfg.logger
    logger_cfg = OmegaConf.create(OmegaConf.to_container(cfg.graph_logger, resolve=True))
    if "wandb" in logger_cfg:
        logger_cfg.wandb.name = f"fold_{split_index}_gnn"
        logger_cfg.wandb.job_type = "single-fold-gnn"
    loggers: list[Logger] = instantiate_loggers(logger_cfg)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.graph_trainer, logger=loggers, callbacks=callbacks
    )

    graph_hparams = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        for key in ("graph_model", "graph_data", "graph_trainer", "graph_callbacks")
    }
    graph_hparams["fold_index"] = split_index
    for logger in loggers:
        logger.log_hyperparams(graph_hparams)

    log.info("Starting GNN training...")
    trainer.fit(model=gnn, datamodule=graph_datamodule, ckpt_path=cfg.get("graph_ckpt_path"))
    _load_best_weights(gnn, trainer, "gnn")

    if hasattr(gnn, "evaluate"):
        gnn.eval()
        metrics = {f"gnn/{k}": v for k, v in gnn.evaluate(graph_datamodule).items()}
        metrics["fold_index"] = split_index
        for logger in loggers:
            logger.log_metrics(metrics)
    else:
        log.warning("GNN has no .evaluate(); skipping final evaluation.")

    log.info(f"GNN results for fold {split_index} sent to loggers.")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="experiment/eval_latent_graph.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    vae = train_vae(cfg)
    _finish_wandb()  # clôt le run W&B du VAE avant d'ouvrir celui du GNN

    if cfg.get("run_graph", False):
        train_gnn(cfg, vae)
        _finish_wandb()

    return 1


if __name__ == "__main__":
    main()

