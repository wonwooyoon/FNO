import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from util_training import model_evaluation_generic, train_model_generic


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


def test_model_evaluation_generic_creates_missing_output_dir(tmp_path):
    output_dir = tmp_path / "output_unet" / "final"
    test_loader = [
        {
            "x": torch.ones(2, 1),
            "y": torch.ones(2, 1),
        }
    ]

    results = model_evaluation_generic(
        config={"LOSS_CONFIG": {"loss_type": "mse"}},
        device="cpu",
        model=IdentityModel(),
        test_loader=test_loader,
        loss_fn=torch.nn.MSELoss(),
        output_dir=output_dir,
        verbose=False,
    )

    assert results["test_loss"] == 0.0
    assert (output_dir / "evaluation_results.pt").exists()


def test_train_model_generic_saves_training_timing_report(tmp_path):
    model = nn.Linear(1, 1)
    train_loader = [
        {
            "x": torch.ones(2, 1),
            "y": torch.ones(2, 1),
        }
    ]
    val_loader = [
        {
            "x": torch.ones(2, 1),
            "y": torch.ones(2, 1),
        }
    ]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    trained = train_model_generic(
        config={
            "N_EPOCHS": 1,
            "SCHEDULER_CONFIG": {"early_stopping": 2},
            "TIMING": {"ENABLED": True},
        },
        device="cpu",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=torch.nn.MSELoss(),
        output_dir=tmp_path,
        verbose=False,
    )

    assert trained is model
    assert (tmp_path / "training_timing.json").exists()
    assert (tmp_path / "training_timing.csv").exists()
