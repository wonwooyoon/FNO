import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from util_common import LRStepScheduler, step_scheduler


def test_lr_step_scheduler_reduces_lr_after_configured_validation_plateau():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = LRStepScheduler(optimizer, step_size=2, gamma=0.5)

    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == 1.0

    scheduler.step(1.1)
    assert optimizer.param_groups[0]["lr"] == 1.0

    scheduler.step(1.2)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_lr_step_scheduler_resets_plateau_counter_after_improvement():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = LRStepScheduler(optimizer, step_size=2, gamma=0.5)

    scheduler.step(1.0)
    scheduler.step(1.1)
    scheduler.step(0.9)
    scheduler.step(0.95)

    assert optimizer.param_groups[0]["lr"] == 1.0

    scheduler.step(0.96)

    assert optimizer.param_groups[0]["lr"] == 0.5


def test_step_scheduler_passes_validation_loss_only_to_plateau_schedulers():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    plateau = LRStepScheduler(optimizer, step_size=1, gamma=0.5)

    step_scheduler(plateau, val_loss=1.0)

    assert plateau.best == 1.0

    regular_parameter = torch.nn.Parameter(torch.tensor(1.0))
    regular_optimizer = torch.optim.SGD([regular_parameter], lr=1.0)
    regular_scheduler = torch.optim.lr_scheduler.StepLR(regular_optimizer, step_size=1, gamma=0.5)
    regular_optimizer.step()

    step_scheduler(regular_scheduler, val_loss=2.0)

    assert regular_optimizer.param_groups[0]["lr"] == 0.5
