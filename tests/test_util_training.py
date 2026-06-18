import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from util_training import model_evaluation_generic


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
