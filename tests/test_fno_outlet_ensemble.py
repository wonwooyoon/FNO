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

import FNO_outlet
from util_ensemble import FixedSplit


class ConstantOutletModel(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return torch.full(
            (x.shape[0], 20),
            float(self.value.detach()),
            dtype=x.dtype,
            device=x.device,
        )


def test_outlet_config_declares_model_kind_and_timing():
    assert FNO_outlet.CONFIG["MODEL_KIND"] == "fno_outlet"
    assert FNO_outlet.CONFIG["ENSEMBLE"]["ENABLED"] is True
    assert FNO_outlet.CONFIG["TIMING"]["ENABLED"] is True
    assert FNO_outlet.CONFIG["TIMING"]["PREDICTION_WARMUP_BATCHES"] == 1
    assert FNO_outlet.CONFIG["TIMING"]["REPORT_DIR_NAME"] == "timing"


def test_load_outlet_ensemble_for_evaluation_returns_mean_predictor(monkeypatch, tmp_path):
    state_a = tmp_path / "member_a.pt"
    state_b = tmp_path / "member_b.pt"
    torch.save(ConstantOutletModel(1.0).state_dict(), state_a)
    torch.save(ConstantOutletModel(3.0).state_dict(), state_b)

    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_kind": "fno_outlet",
                "reduction": "mean",
                "params": {
                    "n_modes_1": 4,
                    "n_modes_2": 4,
                    "n_modes_3": 2,
                    "hidden_channels": 8,
                    "n_layers": 2,
                    "domain_padding": [0.1, 0.1, 0.1],
                    "train_batch_size": 1,
                    "l2_weight": 0.0,
                    "channel_mlp_expansion": 1.0,
                    "channel_mlp_skip": "linear",
                    "projection_mlp_hidden": 8,
                    "projection_mlp_layers": 2,
                    "projection_mlp_activation": "gelu",
                    "projection_mlp_dropout": 0.0,
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
        return ConstantOutletModel(), None, None, "test_loader", None, None, torch.nn.MSELoss()

    monkeypatch.setattr(FNO_outlet, "create_model_from_params", fake_create_model_from_params)

    predictor, test_loader, loss_fn, manifest = FNO_outlet.load_ensemble_for_evaluation(
        config={},
        manifest_file=manifest_path,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        device="cpu",
    )

    pred = predictor(torch.zeros(2, 11, 4, 4, 20))

    assert torch.allclose(pred, torch.full((2, 20), 2.0))
    assert predictor.keep_models_on_device is False
    assert test_loader == "test_loader"
    assert isinstance(loss_fn, torch.nn.MSELoss)
    assert manifest["model_kind"] == "fno_outlet"


def test_main_single_mode_trains_outlet_ensemble(monkeypatch):
    captured = {}
    fake_config = {
        **FNO_outlet.CONFIG,
        "TRAINING_CONFIG": {**FNO_outlet.CONFIG["TRAINING_CONFIG"], "mode": "single"},
        "ENSEMBLE": {**FNO_outlet.CONFIG["ENSEMBLE"], "ENABLED": True, "N_MODELS": 2},
    }

    monkeypatch.setattr(FNO_outlet, "CONFIG", fake_config)
    monkeypatch.setattr(
        FNO_outlet,
        "preprocessing_outlet",
        lambda config, return_split: (
            "spatial_normalizer",
            "outlet_normalizer",
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

    monkeypatch.setattr(FNO_outlet, "train_ensemble", fake_train_ensemble, raising=False)
    monkeypatch.setattr(FNO_outlet, "model_evaluation", lambda *args, **kwargs: None)
    monkeypatch.setattr(FNO_outlet, "visualize_outlet_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(FNO_outlet, "integrated_gradients_analysis_outlet", lambda *args, **kwargs: None)

    FNO_outlet.main()

    assert captured["model_kind"] == "fno_outlet"
    assert captured["params"] == fake_config["SINGLE_PARAMS"]
    assert captured["trainval_dataset"] == "trainval_dataset"
