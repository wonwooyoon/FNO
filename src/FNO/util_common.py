"""
Common Utilities for FNO Models

This module provides shared utilities used by both FNO.py and FNO_outlet.py:
- Loss functions (LpLoss)
- Learning rate schedulers (LRStepScheduler, CappedCosineAnnealingWarmRestarts)
- Other common helper functions

Refactored from FNO.py and FNO_outlet.py to eliminate code duplication.
"""

import math
import torch
import torch.nn as nn


# ==============================================================================
# Loss Functions
# ==============================================================================

class LpLoss(nn.Module):
    """Lp Loss function for neural operators.

    Computes the relative Lp norm between prediction and ground truth:
    ||pred - y||_p / ||y||_p

    Args:
        d: Spatial dimensions to compute norm over (e.g., 2 for 2D, 3 for 3D)
        p: Power for Lp norm (e.g., 2 for L2 norm)
        reduction: Reduction method ('mean' or 'sum')
    """

    def __init__(self, d=2, p=2, reduction='mean'):
        super().__init__()
        self.d = d
        self.p = p
        self.reduction = reduction

    def forward(self, pred, y):
        # Get spatial dimensions (skip batch and channel dimensions)
        if len(pred.shape) == 5:  # (N, C, nx, ny, nt)
            dims = [2, 3, 4]  # spatial and temporal dimensions
        elif len(pred.shape) == 4:  # (N, C, nx, ny)
            dims = [2, 3]  # spatial dimensions
        else:
            dims = list(range(2, len(pred.shape)))

        # Compute relative Lp norm: ||pred - y||_p / ||y||_p
        diff_norm = torch.norm(pred - y, p=self.p, dim=dims, keepdim=False)
        y_norm = torch.norm(y, p=self.p, dim=dims, keepdim=False)
        relative_error = diff_norm / (y_norm + 1e-12)  # Add small epsilon to avoid division by zero

        if self.reduction == 'mean':
            return relative_error.mean()
        elif self.reduction == 'sum':
            return relative_error.sum()
        else:
            return relative_error


# ==============================================================================
# Learning Rate Schedulers
# ==============================================================================

class LRStepScheduler:
    """Validation-plateau learning rate scheduler.

    This keeps the historical ``LRStepScheduler`` name and constructor shape,
    but ``step_size`` now means the number of consecutive non-improving
    validation epochs allowed before multiplying the learning rate by ``gamma``.

    Args:
        optimizer: Wrapped optimizer
        step_size: Validation plateau patience in epochs
        gamma: Multiplicative factor of learning rate decay
        threshold: Minimum strict improvement margin
    """

    requires_metric = True

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
        threshold: float = 0.0,
    ):
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be between 0 and 1, got {gamma}")
        self.optimizer = optimizer
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = -1
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, metric: float) -> None:
        current = float(metric)
        self.last_epoch += 1

        if self.best is None or current < (self.best - self.threshold):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.step_size:
                for group in self.optimizer.param_groups:
                    group["lr"] *= self.gamma
                self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            "step_size": self.step_size,
            "gamma": self.gamma,
            "threshold": self.threshold,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "last_epoch": self.last_epoch,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict):
        self.step_size = int(state_dict["step_size"])
        self.gamma = float(state_dict["gamma"])
        self.threshold = float(state_dict.get("threshold", 0.0))
        self.best = state_dict["best"]
        self.num_bad_epochs = int(state_dict["num_bad_epochs"])
        self.last_epoch = int(state_dict["last_epoch"])
        self._last_lr = list(state_dict["_last_lr"])


def step_scheduler(scheduler, val_loss: float) -> None:
    """Step a scheduler, passing validation loss only when required."""
    if getattr(scheduler, "requires_metric", False):
        scheduler.step(val_loss)
    else:
        scheduler.step()


class CappedCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing warm restarts scheduler with maximum period cap.

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart
        T_max: Maximum period length
        T_mult: Factor to increase period after restart
        eta_min: Minimum learning rate
        last_epoch: Index of last epoch
    """

    def __init__(self, optimizer: torch.optim.Optimizer, T_0: int, T_max: int,
                 T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1):
        self.T_0 = T_0
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.last_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        epoch_in_cycle = (self.last_epoch - self.last_restart) % self.T_i
        cycle_num = self.last_epoch // self.T_i + 1
        progress = epoch_in_cycle / self.T_i

        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + ((base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2) / cycle_num
            lrs.append(lr)

        # Check for restart
        if (self.last_epoch - self.last_restart) == self.T_i:
            self.last_restart = self.last_epoch
            self.T_i = min(self.T_i * self.T_mult, self.T_max)

        return lrs
