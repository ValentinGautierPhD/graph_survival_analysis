#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch import nn
import lightning as pl
from sksurv.metrics import concordance_index_censored

import survival_analysis.utils.losses as losses


class Encoder(nn.Module):
    """Encodeur MLP pour données tabulaires / vecteurs de features.

    Produit :
      - mu, logvar : paramètres de la distribution latente (VAE classique)

    Args:
        input_dim: dimension du vecteur de features en entrée.
        hidden_dims: liste des tailles des couches cachées, ex. [128, 64].
        latent_dim: dimension de l'espace latent z.
        act_fn: fonction d'activation utilisée entre les couches.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), act_fn()]
            prev_dim = h
        self.backbone = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Décodeur MLP symétrique de l'Encoder : latent z -> reconstruction.

    Args:
        output_dim: dimension du vecteur reconstruit (= input_dim de l'encodeur).
        hidden_dims: mêmes tailles cachées que l'encodeur (utilisées en ordre inverse).
        latent_dim: dimension de l'espace latent z.
        act_fn: fonction d'activation utilisée entre les couches.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dims: list,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        super().__init__()

        layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev_dim, h), act_fn()]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class RiskHead(nn.Module):
    """Petite tête linéaire latent -> score de risque scalaire pour CoxPHLoss."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


class VAE(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        beta: float = 1.0,
        lambda_survival: float = 1.0,
        optimizer: object = None,
        scheduler: object = None,
    ):
        super().__init__()
        # Saving hyperparameters of the model
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.beta = beta
        self.lambda_survival = lambda_survival

        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler

        # Creating encoder, decoder and risk head
        self.encoder = encoder_class(in_dim, hidden_dims, latent_dim)
        self.decoder = decoder_class(in_dim, hidden_dims, latent_dim)
        self.risk_head = RiskHead(latent_dim)

        self.loss_manager = losses.LossManager([
            losses.MSELoss(weight=1.0, input_map={"preds": "x",
                                                  "labels": "reconstruction"}),
            losses.KLDivergence(weight=self.beta),
            losses.CoxPHLoss(weight=self.lambda_survival),
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _forward_full(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        risk = self.risk_head(mu)
        return x_hat, mu, logvar, risk

    def forward(self, x):
        x_hat, *_ = self._forward_full(x)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, targets, *_ = batch

        x_hat, mu, logvar, risk = self._forward_full(x)

        ctx = {
            "preds": risk,
            "reconstruction": x_hat,
            "mu": mu,
            "log_var": logvar,
            "x": x,
            "targets": targets,
        }

        total, logs = self.loss_manager(ctx)

        self.log_losses(logs, prefix="train/")
        return total

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(self.parameters())
        if self.partial_scheduler is None:
            return optimizer

        scheduler = self.partial_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                # "frequency": "indique la fréquence de mise à jour de la métrique",
                # si "monitor" référence une métrique de validation, "frequency" doit être
                # un multiple de "trainer.check_val_every_n_epoch".
            },
        }

    def validation_step(self, batch, batch_idx):
        x, targets, *_ = batch

        x_hat, mu, logvar, risk = self._forward_full(x)

        ctx = {
            "preds": risk,
            "reconstruction": x_hat,
            "x": x,
            "mu": mu,
            "log_var": logvar,
            "targets": targets,
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

    def evaluate(self, datamodule):
        all_risk = []
        all_time = []
        all_event = []
        
        with torch.no_grad():
            for x, y, patient_id in datamodule.val_dataloader():
                *_, risk = self._forward_full(x)
                all_risk.append(risk)
                time = y[:, 0]
                all_time.append(time)
                event = y[:, 1]
                all_event.append(event)
        risks = torch.cat(all_risk, dim=0).detach().cpu().numpy()
        times = torch.cat(all_time, dim=0).detach().cpu().numpy()
        events = torch.cat(all_event, dim=0).detach().cpu().numpy().astype(bool)

        c_index = concordance_index_censored(
            events,
            times,
            risks
        )[0]

        return {
            "test/c_index": c_index
        }
