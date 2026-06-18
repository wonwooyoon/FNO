import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

import UNET
from util_common import LpLoss as CommonLpLoss
from util_common import LRStepScheduler as CommonLRStepScheduler
from util_common import CappedCosineAnnealingWarmRestarts as CommonCappedCosineAnnealingWarmRestarts
from util_ensemble import FixedSplit


class ConstantModel(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]),
            float(self.value.detach()),
            dtype=x.dtype,
            device=x.device,
        )


def test_unet_config_exposes_ensemble_and_optuna_epoch_settings():
    assert UNET.CONFIG["TRAINING_CONFIG"]["optuna_n_epochs"] >= 1
    assert UNET.CONFIG["ENSEMBLE"]["ENABLED"] is True
    assert UNET.CONFIG["ENSEMBLE"]["N_MODELS"] >= 1
    assert UNET.CONFIG["ENSEMBLE"]["MANIFEST_NAME"] == "ensemble_manifest.json"
    assert UNET.CONFIG["OUTPUT"]["ENABLED"] is True
    assert UNET.CONFIG["OUTPUT"]["SAMPLE_INDICES"]
    assert "VISUALIZATION" not in UNET.CONFIG


def test_create_model_from_params_casts_unet_params(monkeypatch):
    captured = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return "components"

    monkeypatch.setattr(UNET, "create_model", fake_create_model)

    result = UNET.create_model_from_params(
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


def test_unet_uses_common_loss_and_scheduler_classes():
    assert UNET.LpLoss is CommonLpLoss
    assert UNET.LRStepScheduler is CommonLRStepScheduler
    assert UNET.CappedCosineAnnealingWarmRestarts is CommonCappedCosineAnnealingWarmRestarts


def test_unet_dataset_includes_initial_values_when_available():
    x = torch.zeros(2, 11, 4, 4, 3)
    y = torch.ones(2, 1, 4, 4, 3)
    y_initial = torch.full((2, 1, 4, 4, 1), 2.0)

    dataset = UNET.CustomDatasetPure(x, y, y_initial)

    item = dataset[1]
    assert set(item) == {"x", "y", "y_initial"}
    assert torch.equal(item["y_initial"], y_initial[1])


def test_unet_dataset_omits_initial_values_when_unavailable():
    x = torch.zeros(2, 11, 4, 4, 3)
    y = torch.ones(2, 1, 4, 4, 3)

    dataset = UNET.CustomDatasetPure(x, y)

    assert set(dataset[0]) == {"x", "y"}


def test_unet_visualization_delegates_to_unified_output(monkeypatch):
    captured = {}
    test_dataset = [
        {
            "x": torch.zeros(1),
            "y": torch.zeros(1),
        }
    ]

    def fake_generate_all_outputs(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(UNET, "generate_all_outputs", fake_generate_all_outputs, raising=False)

    result = UNET.visualization(
        config={"OUTPUT_DIR": "out"},
        channel_normalizer="normalizer",
        device="cpu",
        trained_model="model",
        train_dataset="train",
        val_dataset="val",
        test_dataset=test_dataset,
        verbose=False,
    )

    assert result == {"ok": True}
    assert captured["channel_normalizer"] == "normalizer"
    assert captured["trained_model"] == "model"
    assert captured["train_dataset"] == "train"
    assert captured["val_dataset"] == "val"
    assert captured["test_dataset"] is test_dataset
    assert captured["test_loader"].batch_size == 1


def test_train_model_to_dir_delegates_to_generic_training_with_output_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_train_model_generic(**kwargs):
        captured.update(kwargs)
        return "trained"

    monkeypatch.setattr(UNET, "train_model_generic", fake_train_model_generic)

    result = UNET.train_model_to_dir(
        config={"N_EPOCHS": 1},
        device="cpu",
        model="model",
        train_loader="train_loader",
        val_loader="val_loader",
        optimizer="optimizer",
        scheduler="scheduler",
        loss_fn="loss_fn",
        output_dir=tmp_path / "member_000",
        verbose=False,
    )

    assert result == "trained"
    assert captured["output_dir"] == tmp_path / "member_000"
    assert captured["verbose"] is False


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
                "model_kind": "unet",
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

    monkeypatch.setattr(UNET, "create_model_from_params", fake_create_model_from_params)

    predictor, test_loader, loss_fn, manifest = UNET.load_ensemble_for_evaluation(
        config={},
        manifest_file=manifest_path,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        device="cpu",
    )

    pred = predictor(torch.zeros(2, 11, 4, 4, 4))

    assert torch.allclose(pred, torch.full((2, 1, 4, 4, 4), 2.0))
    assert predictor.keep_models_on_device is False
    assert test_loader == "test_loader"
    assert isinstance(loss_fn, torch.nn.MSELoss)
    assert manifest["model_kind"] == "unet"


def test_main_single_mode_trains_unet_ensemble(monkeypatch):
    captured = {}
    fake_config = {
        **UNET.CONFIG,
        "TRAINING_CONFIG": {**UNET.CONFIG["TRAINING_CONFIG"], "mode": "single"},
        "ENSEMBLE": {**UNET.CONFIG["ENSEMBLE"], "ENABLED": True, "N_MODELS": 2},
    }

    monkeypatch.setattr(UNET, "CONFIG", fake_config)
    monkeypatch.setattr(
        UNET,
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

    monkeypatch.setattr(UNET, "train_ensemble", fake_train_ensemble, raising=False)
    monkeypatch.setattr(UNET, "model_evaluation", lambda **kwargs: None)
    monkeypatch.setattr(UNET, "visualization", lambda **kwargs: None)

    UNET.main()

    assert captured["model_kind"] == "unet"
    assert captured["params"] == fake_config["SINGLE_PARAMS"]
    assert captured["trainval_dataset"] == "trainval_dataset"
