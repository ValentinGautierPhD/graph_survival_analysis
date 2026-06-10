#!/usr/bin/env python3

from pycox.models.loss import CoxPHLoss as _CoxPHLoss
from torch import nn

class CoxPHLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = _CoxPHLoss()

    def forward(self, preds, targets):
        times, events = targets[...,0], targets[...,1]
        return self.loss(preds, times, events)
