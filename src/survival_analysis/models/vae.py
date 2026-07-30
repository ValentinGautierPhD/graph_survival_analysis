#!/usr/bin/env python3

from torch import nn
import torch
import lightning as pl

import survival_analysis.utils.losses as losses

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        super().__init__()


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Decoder.

        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()


class VAE(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

        self.loss_manager = losses.LossManager([
            losses.CoxPHLoss(weight=1.0),
            losses.L0LossHardConcrete(weight=self.lambda1),
            losses.L1Loss(weight=self.lambda2, input_map={"preds": "logits"}),
        ])

    def forward(self, x):
        out, _ = self._forward_full(x)
        return out
        
    def training_step(self, batch, batch_idx):
        
        # ---- forward PyG
        preds,logits = self._forward_full(batch.x)
        targets = batch.y
        # pred: [b, n, C]
        ctx = {"preds": preds,
               "targets": targets,
               "logits": logits,
               "tau": self.tau, "gamma": self.gamma, "zeta":self.zeta}
        
        total, logs = self.loss_manager(ctx)
        self.log_losses(logs, prefix="train/")

        return total

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

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


    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

    def log_losses(self, logs: dict[str, torch.Tensor], prefix: str = ""):
        for name, value in logs.items():
            self.log(
                f"{prefix}{name}",
                value,
                on_step=False,
                on_epoch=True,
            )
