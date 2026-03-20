import hydra
import wandb
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import Optional
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from lightning.pytorch import LightningModule, Trainer, LightningDataModule
from lightning.pytorch.loggers import Logger

from ..utils import (
    RankedLogger,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)

def plot_edge_probs_matplotlib(pi, bins=100, log_scale=True):

    
    # 3. Création de la figure
    fig = plt.figure(figsize=(10, 6))
    
    # Histogramme
    n_counts, bins_edges, patches = plt.hist(
        pi, 
        bins=bins, 
        color='#3498db', 
        edgecolor='black', 
        alpha=0.7
    )
    
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Nombre d'arêtes (Échelle Log)")
    else:
        plt.ylabel("Nombre d'arêtes")

    plt.title("Distribution des probabilités des arêtes", fontsize=14)
    plt.xlabel("Probabilité")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Ajout d'une ligne pour la moyenne (optionnel)
    plt.axvline(pi.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Moyenne: {pi.mean():.4f}')
    plt.legend()

    plt.tight_layout()

    return fig

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

    # 3. Gestion spécifique du Logger (W&B)
    if "wandb" in cfg.logger:
        # On injecte dynamiquement le fold dans le nom du run
        cfg.logger.wandb.name = f"fold_{cfg.data.split_index}"
        cfg.logger.wandb.job_type = "single-fold"

    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # 4. Instanciation du Trainer et Entraînement
    log.info("Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

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

    # 5. Évaluation spécifique Survie (Pycox)
    log.info("Starting survival evaluation...")
    
    results_concordance = []
    results_brier = []
    nb_tests = 100 # Nombre de simulations pour la stabilité

    # On passe le modèle au wrapper CoxPH
    survival_model = CoxPH(model)
    pi = model.pi.cpu().numpy().flatten()

    # On utilise les données préparées par le datamodule
    # .train_graph et .val_graph ont été créés lors du datamodule.setup()
    train_x = datamodule.train_graph.x
    train_y_durations = datamodule.train_graph.y[..., 0]
    train_y_events = datamodule.train_graph.y[..., 1]
    
    val_x = datamodule.val_graph.x
    val_y = datamodule.val_graph.y
    val_idx = datamodule.val_graph.val_idx # Les indices de test stockés dans le graph de val

    # Calcul de la baseline hazard une seule fois (ou dans la boucle si besoin de stochasticité)
    _ = survival_model.compute_baseline_hazards(train_x, (train_y_durations, train_y_events))

    for i in range(nb_tests):
        # Prédiction des fonctions de survie
        surv = survival_model.predict_surv_df(val_x)
        
        # Extraction des durées et évènements réels pour le calcul des métriques
        durations_test = val_y[..., 0].numpy()
        events_test = val_y[..., 1].numpy()

        # Evaluation sur le split de validation uniquement
        ev = EvalSurv(
            surv[val_idx], 
            durations_test[val_idx], 
            events_test[val_idx], 
            censor_surv='km'
        )
        
        # Création de la grille temporelle pour le Brier Score
        time_grid = np.linspace(durations_test[val_idx].min(), durations_test[val_idx].max(), 100)

        results_concordance.append(ev.concordance_td())
        results_brier.append(ev.integrated_brier_score(time_grid))

    # Calcul des moyennes finales pour ce split
    mean_cindex = np.mean(results_concordance)
    mean_brier = np.mean(results_brier)
    std_cindex = np.std(results_concordance) # Optionnel mais utile

    fig = plot_edge_probs_matplotlib(pi)

    # 2. Préparation du dictionnaire de métriques
    metrics = {
        "test/c_index": mean_cindex,
        "test/brier_score": mean_brier,
        "test/c_index_std": std_cindex,
        "fold_index": cfg.data.split_index, # Pour filtrer facilement dans l'UI WandB
        "probs": fig
    }

    # 3. Envoi au(x) logger(s)
    if loggers:
        for logger in loggers:
            # On utilise log_metrics pour envoyer les résultats de fin de run
            logger.log_metrics(metrics)
            
    log.info(f"Final Results for Fold {cfg.data.split_index} sent to loggers.")
    
    if "wandb" in cfg.logger:
        wandb.finish()   

    return mean_cindex

if __name__ == "__main__":
    main()
