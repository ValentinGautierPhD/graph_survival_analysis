#!/usr/bin/env python3

from pycox.evaluation import EvalSurv
from torch import dropout, embedding, nn
import numpy as np
import torch
import lightning as pl
# from pycox.models.loss import CoxPHLoss
import survival_analysis.utils.losses as losses
from pycox.models import CoxPH
import plotly.express as px

class MLP(pl.LightningModule):
    def __init__(self, in_dim, out_dim, hid_dims, dropouts, optimizer, scheduler=None):
        super().__init__()
        assert len(hid_dims) >= 1, "At least one hidden layer needed"
        assert len(hid_dims) == len(dropouts), "number of hidden dimensions must match number of dropouts"

        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        self.training_mode = True
        self.out_dim = out_dim

        self.loss_manager = losses.LossManager([
            losses.CoxPHLoss(weight=1),
        ])
        dims = [in_dim] + hid_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[i + 1]),
                nn.Dropout(dropouts[i]),
            ]

        self.hid_layers = nn.Sequential(*layers)
        self.out = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, x):
        # z = self.phi(x)
        # z = self.relu(z)
        # z = self.dropout(z)
        embedding = self.hid_layers(x)
        return self.out(embedding)
        
    def training_step(self, batch, batch_idx):
        x, y = batch[:2]

        preds = self(x)
        ctx = {
            "preds": preds,
            "targets": y
        }

        total, logs = self.loss_manager(ctx)

        self.log_losses(logs, prefix="train/")
        return total

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]

        preds = self(x)
        ctx = {
            "preds": preds,
            "targets": y
        }

        total, logs = self.loss_manager(ctx)

        self.log_losses(logs, prefix="val/")
        return total
        
    def log_losses(self, logs: dict, prefix: str = ""):
        for name, value in logs.items():
            self.log(
                f"{prefix}{name}",
                value,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(
            self.parameters(),
        )
        if self.partial_scheduler is None:
            return optimizer

        scheduler = self.partial_scheduler(optimizer)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val/loss",
            # "frequency": "indicates how often the metric is updated",
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }


    def evaluate(self, datamodule) -> dict:
        survival_model = CoxPH(self)
        mean_cindex, mean_brier = evaluate_bis(datamodule, survival_model, nb_tests=100)

        return {
            "test/c_index":    mean_cindex,
            "test/brier_score": mean_brier,
            "fold_index":       None,  # sera injecté par le script principal
        }


def evaluate_bis(datamodule, survival_model, nb_tests=100):

    x_train, y_train = datamodule.train_dataset.tensors
    x_val, y_val = datamodule.val_dataset.tensors

    train_y_durations = y_train[:, 0]
    train_y_events    = y_train[:, 1]
    durations_test    = y_val[:, 0].numpy()
    events_test       = y_val[:, 1].numpy()
    survs = []
    
    for i in range(nb_tests):
        # Prédiction des fonctions de survie
        _ = survival_model.compute_baseline_hazards(x_train, (train_y_durations, train_y_events))
        survs.append(survival_model.predict_surv_df(x_val))


    surv = sum(survs)/nb_tests

    # Evaluation sur le split de validation uniquement
    ev = EvalSurv(
        surv, 
        durations_test, 
        events_test, 
        censor_surv='km'
    )

    # Création de la grille temporelle pour le Brier Score
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

    c_index = ev.concordance_td()
    brier = ev.integrated_brier_score(time_grid)

    # Calcul des moyennes finales pour ce split
    return c_index, brier
