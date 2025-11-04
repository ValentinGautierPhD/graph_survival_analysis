#!/usr/bin/env python3

import torch
import lightning.pytorch as pl
from torch import nn

from pycox.models.loss import CoxPHLoss

class SurvModel(pl.LightningModule):
    '''
    Defines model, optimizers, forward step, and training step. 
    Define validation step as def validation_step if needed
    Configured to use CoxPH loss from loss.CoxPHLoss()
    '''

    def __init__(self, optimizer, scheduler, in_features, out_features):
        super().__init__()

        self.save_hyperparameters()
        self.in_features = in_features
        self.out_features = out_features
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler

        # Define Model Here (in this case MLP)
        self.net = nn.Sequential(
            nn.Linear(self.in_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, self.out_features),
        )

        # Define loss function:
        self.loss_func = CoxPHLoss()

    def forward(self, x):
        batch_size, data = x.size()
        x = self.net(x)
        return x

    # Training step and validation step usually defined, this dataset only had train + test so left out val. 
    def training_step(self, batch, batch_idx): 
        x, target = batch
        output = self.forward(x)

        # target variable contains duration and event as a concatenated tensor
        loss = self.loss_func(output, target[:,0], target[:,1]) 

        # progress bar logging metrics (add custom metric definitions later if useful?)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(
            self.parameters(),
        )
        scheduler = self.partial_scheduler(optimizer)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "loss",
            # "frequency": "indicates how often the metric is updated",
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }
