#!/usr/bin/env python3

from pycox.models.loss import CoxPHLoss as _CoxPHLoss
import torch
from torch import nn
from abc import ABC, abstractmethod


class Loss(nn.Module, ABC):
    def __init__(self, weight=1.0, name=None):
        super().__init__()
        self.weight = weight
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, ctx: dict) -> torch.Tensor:
        ...

    def compute(self, ctx: dict) -> torch.Tensor:
        return self.weight * self.forward(ctx)


class LossManager(nn.Module):
    def __init__(self, losses: list[Loss]):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, ctx: dict):
        logs = {}
        total = 0.0
        for loss_fn in self.losses:
            value = loss_fn.compute(ctx)
            logs[loss_fn.name] = value
            total += value
        logs["loss"] = total
        return total, logs


class CrossEntropyLoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__(weight, name="loss_cls")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ctx):
        return self.criterion(ctx["pred"], ctx["label"])


class SmoothnessLoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__(weight, name="loss_smh")

    def forward(self, ctx):
        A, H = ctx["adjacency"], ctx["z"]
        N = H.size(0)
        sq_norms = (H ** 2).sum(dim=1)
        dot = H @ H.T
        dist_sq = (sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dot).clamp(min=0)
        return (A * dist_sq).sum() / (2 * N ** 2)


class ConnectivityLoss(Loss):
    def __init__(self, weight=1.0, eps=1e-8):
        super().__init__(weight, name="loss_con")
        self.eps = eps

    def forward(self, ctx):
        A = ctx["adjacency"]
        N = A.size(0)
        row_sum = A.sum(dim=1)
        return -torch.log(row_sum + self.eps).sum() / N


class FrobeniusLoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__(weight, name="loss_frob")

    def forward(self, ctx):
        A = ctx["adjacency"]
        N = A.size(0)
        return (A ** 2).sum() / (N ** 2)


class CoxPHLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = _CoxPHLoss()

    def forward(self, preds, targets):
        times, events = targets[...,0], targets[...,1]
        return self.loss(preds, times, events)
