import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

import UNET


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "x": torch.zeros(11, 64, 32, 20),
            "y": torch.zeros(1, 64, 32, 20),
        }


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_unet_preserves_spatiotemporal_shape_for_supported_depths(depth):
    model = UNET.UNet3D(
        in_channels=11,
        out_channels=1,
        init_features=2,
        depth=depth,
        dropout_rate=0.0,
        pool_kernels=UNET.CONFIG["MODEL_CONFIG"]["pool_kernels"],
    ).eval()

    with torch.no_grad():
        y = model(torch.zeros(1, 11, 64, 32, 20))

    assert y.shape == (1, 1, 64, 32, 20)


def test_unet_depth_four_uses_matching_skip_shapes_without_padding():
    model = UNET.UNet3D(
        in_channels=11,
        out_channels=1,
        init_features=2,
        depth=4,
        dropout_rate=0.0,
        pool_kernels=UNET.CONFIG["MODEL_CONFIG"]["pool_kernels"],
    ).eval()

    with torch.no_grad():
        model(torch.zeros(1, 11, 64, 32, 20))

    assert model.last_skip_shape_pairs == [
        ((8, 4, 5), (8, 4, 5)),
        ((16, 8, 5), (16, 8, 5)),
        ((32, 16, 10), (32, 16, 10)),
        ((64, 32, 20), (64, 32, 20)),
    ]


def test_unet_rejects_depth_beyond_configured_pool_schedule():
    with pytest.raises(ValueError, match="depth"):
        UNET.UNet3D(
            in_channels=11,
            out_channels=1,
            init_features=2,
            depth=5,
            dropout_rate=0.0,
            pool_kernels=UNET.CONFIG["MODEL_CONFIG"]["pool_kernels"],
        )


def test_create_model_builds_configured_conventional_unet():
    config = {
        **UNET.CONFIG,
        "MODEL_CONFIG": {
            **UNET.CONFIG["MODEL_CONFIG"],
            "in_channels": 11,
            "out_channels": 1,
            "pool_kernels": [(2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 1)],
        },
        "LOSS_CONFIG": {"loss_type": "mse"},
        "SCHEDULER_CONFIG": {
            **UNET.CONFIG["SCHEDULER_CONFIG"],
            "scheduler_type": "step",
        },
    }

    model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = UNET.create_model(
        config=config,
        train_dataset=TinyDataset(),
        val_dataset=TinyDataset(),
        test_dataset=TinyDataset(),
        device="cpu",
        depth=4,
        init_features=2,
        train_batch_size=1,
        l2_weight=0.0,
        dropout_rate=0.0,
    )

    assert isinstance(model, UNET.UNet3D)
    assert model.pool_kernels == [(2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 1)]
    assert len(train_loader) == 2
    assert len(val_loader) == 2
    assert len(test_loader) == 2
    assert isinstance(loss_fn, torch.nn.MSELoss)
    assert optimizer.__class__.__name__ == "AdamW"
    assert scheduler.__class__.__name__ == "LRStepScheduler"
