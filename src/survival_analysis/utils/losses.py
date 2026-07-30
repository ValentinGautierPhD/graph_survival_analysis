#!/usr/bin/env python3

from pycox.models.loss import CoxPHLoss as _CoxPHLoss
import torch
from torch import nn
from abc import ABC, abstractmethod


class Loss(nn.Module, ABC):
    def __init__(self, weight=1.0, name=None, input_map: dict[str, str] | None = None):
        super().__init__()
        self.weight = weight
        self.name = name or self.__class__.__name__
        # mapping "nom logique attendu par la loss" -> "clé réelle dans ctx"
        self.input_map = input_map or {}

    @abstractmethod
    def forward(self, ctx: dict) -> torch.Tensor:
        ...

    def compute(self, ctx: dict) -> torch.Tensor:
        return self.weight * self.forward(ctx)

    def resolve(self, ctx: dict, key: str):
            """Va chercher `key` dans ctx en tenant compte du remapping éventuel."""
            real_key = self.input_map.get(key, key)
            if real_key not in ctx:
                raise KeyError(
                    f"[{self.name}] attend '{key}' (mappé vers '{real_key}') "
                    f"mais ctx contient: {list(ctx.keys())}"
                )
            return ctx[real_key]


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

class BCEWithLogitsLoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__(weight, name="loss_cls")
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, ctx):
        return self.criterion(ctx["preds"], ctx["labels"])

class CrossEntropyLoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__(weight, name="loss_cls")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ctx):
        return self.criterion(ctx["preds"], ctx["labels"])


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


class CoxPHLoss(Loss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = _CoxPHLoss()

    def forward(self, ctx):
        preds = ctx["preds"]
        targets = ctx["targets"]
        times, events = targets[...,0], targets[...,1]
        return self.loss(preds, times, events)

class L0LossHardConcrete(Loss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, ctx):
        logits = self.resolve(ctx, "logits")
        tau = self.resolve(ctx, "tau")
        gamma = self.resolve(ctx, "gamma")
        zeta = self.resolve(ctx, "zeta")
        
        second_term = tau * torch.log(-torch.ones_like(logits) * gamma/zeta)
        l0_loss = torch.mean(torch.sigmoid(logits - second_term))
        return l0_loss

class L1Loss(Loss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, ctx):
        preds = self.resolve(ctx, "preds")
        l1_loss = preds.abs().mean()
        return l1_loss

class MSELoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__(weight)
        self.criterion = nn.MSELoss()

    def forward(self, ctx):
        return self.criterion(ctx["preds"], ctx["labels"])

class KLDivergence(Loss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, ctx: dict) -> torch.Tensor:
        mu = self.resolve(ctx, "mu")
        logvar = self.resolve(ctx, "log_sigma")

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl
