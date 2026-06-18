import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

import UNET_new
from util_ensemble import FixedSplit


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "x": torch.zeros(11, 16, 8),
            "y": torch.zeros(1, 16, 8, 20),
        }


class ConstantModel(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3], 20),
            float(self.value.detach()),
            dtype=x.dtype,
            device=x.device,
        )


def test_dataset_reduces_static_input_time_axis_and_preserves_output_shapes():
    x = torch.randn(2, 11, 4, 3, 20)
    x = x[..., 0:1].expand(-1, -1, -1, -1, 20).clone()
    y = torch.randn(2, 1, 4, 3, 20)
    y_initial = torch.randn(2, 1, 4, 3, 1)

    dataset = UNET_new.CustomDatasetPure(
        x,
        y,
        y_initial,
        input_time_index=3,
        validate_static_input=True,
    )
    item = dataset[1]

    assert item["x"].shape == (11, 4, 3)
    assert torch.equal(item["x"], x[1, :, :, :, 3])
    assert item["y"].shape == (1, 4, 3, 20)
    assert item["y_initial"].shape == (1, 4, 3, 1)


def test_dataset_rejects_nonstatic_input_time_axis_when_validation_enabled():
    x = torch.zeros(1, 11, 4, 3, 20)
    x[..., 1] = 1.0
    y = torch.zeros(1, 1, 4, 3, 20)

    with pytest.raises(ValueError, match="static"):
        UNET_new.CustomDatasetPure(x, y, validate_static_input=True)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_unet2d_time_output_preserves_output_contract_for_supported_depths(depth):
    model = UNET_new.UNet2DTimeOutput(
        in_channels=11,
        out_channels=1,
        output_time_steps=20,
        init_features=2,
        depth=depth,
        dropout_rate=0.0,
        pool_kernels=UNET_new.CONFIG["MODEL_CONFIG"]["pool_kernels"],
    ).eval()

    with torch.no_grad():
        y = model(torch.zeros(1, 11, 64, 32))

    assert y.shape == (1, 1, 64, 32, 20)
    assert model.outc.out_channels == 20


def test_unet2d_time_output_supports_multiple_output_channels():
    model = UNET_new.UNet2DTimeOutput(
        in_channels=11,
        out_channels=2,
        output_time_steps=20,
        init_features=2,
        depth=2,
        dropout_rate=0.0,
        pool_kernels=UNET_new.CONFIG["MODEL_CONFIG"]["pool_kernels"],
    ).eval()

    with torch.no_grad():
        y = model(torch.zeros(1, 11, 16, 8))

    assert model.outc.out_channels == 40
    assert y.shape == (1, 2, 16, 8, 20)


def test_create_model_builds_configured_2d_time_output_unet():
    config = {
        **UNET_new.CONFIG,
        "MODEL_CONFIG": {
            **UNET_new.CONFIG["MODEL_CONFIG"],
            "in_channels": 11,
            "out_channels": 1,
            "output_time_steps": 20,
            "pool_kernels": [(2, 2), (2, 2), (2, 2), (2, 2)],
        },
        "LOSS_CONFIG": {"loss_type": "mse"},
        "SCHEDULER_CONFIG": {
            **UNET_new.CONFIG["SCHEDULER_CONFIG"],
            "scheduler_type": "step",
        },
    }

    model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = UNET_new.create_model(
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

    assert isinstance(model, UNET_new.UNet2DTimeOutput)
    assert model.pool_kernels == [(2, 2), (2, 2), (2, 2), (2, 2)]
    assert len(train_loader) == 2
    assert len(val_loader) == 2
    assert len(test_loader) == 2
    assert isinstance(loss_fn, torch.nn.MSELoss)
    assert optimizer.__class__.__name__ == "AdamW"
    assert scheduler.__class__.__name__ == "LRStepScheduler"


def test_create_model_from_params_casts_unet_new_params(monkeypatch):
    captured = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return "components"

    monkeypatch.setattr(UNET_new, "create_model", fake_create_model)

    result = UNET_new.create_model_from_params(
        config={"OPTUNA_SEARCH_SPACE": {}},
        train_dataset="train",
        val_dataset="val",
        test_dataset="test",
        device="cpu",
        params={
            "depth": "4",
            "init_features": "8",
            "train_batch_size": "2",
            "l2_weight": "0.25",
            "dropout_rate": "0.1",
        },
    )

    assert result == "components"
    assert captured["depth"] == 4
    assert captured["init_features"] == 8
    assert captured["train_batch_size"] == 2
    assert captured["l2_weight"] == 0.25
    assert captured["dropout_rate"] == 0.1


def test_load_ensemble_for_evaluation_returns_mean_predictor(monkeypatch, tmp_path):
    state_a = tmp_path / "member_a.pt"
    state_b = tmp_path / "member_b.pt"
    torch.save(ConstantModel(1.0).state_dict(), state_a)
    torch.save(ConstantModel(3.0).state_dict(), state_b)

    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_kind": "unet_new",
                "reduction": "mean",
                "params": {
                    "depth": 2,
                    "init_features": 2,
                    "train_batch_size": 1,
                    "l2_weight": 0.0,
                    "dropout_rate": 0.0,
                },
                "members": [
                    {"member_id": 0, "model_state_path": state_a.name},
                    {"member_id": 1, "model_state_path": state_b.name},
                ],
                "test_indices": [0],
                "config": {},
            }
        )
    )

    def fake_create_model_from_params(config, train_dataset, val_dataset, test_dataset, device, params):
        return ConstantModel(), None, None, "test_loader", None, None, torch.nn.MSELoss()

    monkeypatch.setattr(UNET_new, "create_model_from_params", fake_create_model_from_params)

    predictor, test_loader, loss_fn, manifest = UNET_new.load_ensemble_for_evaluation(
        config={},
        manifest_file=manifest_path,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        device="cpu",
    )

    pred = predictor(torch.zeros(2, 11, 4, 4))

    assert torch.allclose(pred, torch.full((2, 1, 4, 4, 20), 2.0))
    assert predictor.keep_models_on_device is False
    assert test_loader == "test_loader"
    assert isinstance(loss_fn, torch.nn.MSELoss)
    assert manifest["model_kind"] == "unet_new"


def test_main_single_mode_trains_unet_new_ensemble(monkeypatch):
    captured = {}
    fake_config = {
        **UNET_new.CONFIG,
        "TRAINING_CONFIG": {**UNET_new.CONFIG["TRAINING_CONFIG"], "mode": "single"},
        "ENSEMBLE": {**UNET_new.CONFIG["ENSEMBLE"], "ENABLED": True, "N_MODELS": 2},
    }

    monkeypatch.setattr(UNET_new, "CONFIG", fake_config)
    monkeypatch.setattr(
        UNET_new,
        "preprocessing",
        lambda config, verbose, return_split: (
            "normalizer",
            "trainval_dataset",
            "test_dataset",
            FixedSplit(trainval_indices=torch.arange(4).numpy(), test_indices=torch.arange(2).numpy()),
            "train_dataset",
            "val_dataset",
            "cpu",
        ),
    )

    def fake_train_ensemble(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            predictor="ensemble_predictor",
            test_loader="test_loader",
            loss_fn="loss_fn",
            train_dataset="member_train",
            val_dataset="member_val",
            manifest_path=Path("manifest.json"),
        )

    monkeypatch.setattr(UNET_new, "train_ensemble", fake_train_ensemble, raising=False)
    monkeypatch.setattr(UNET_new, "model_evaluation", lambda **kwargs: None)
    monkeypatch.setattr(UNET_new, "visualization", lambda **kwargs: None)

    UNET_new.main()

    assert captured["model_kind"] == "unet_new"
    assert captured["params"] == fake_config["SINGLE_PARAMS"]
    assert captured["trainval_dataset"] == "trainval_dataset"
