import json
import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from util_ensemble import EnsemblePredictor
from util_output import generate_all_outputs


class IdentityNormalizer:
    def inverse_transform_output(self, y, y_initial=None):
        return y


class EchoModel(nn.Module):
    def forward(self, x):
        return x[:, :1]


def _minimal_config(tmp_path):
    return {
        "OUTPUT_DIR": str(tmp_path),
        "TIMING": {
            "ENABLED": True,
            "PREDICTION_WARMUP_BATCHES": 0,
            "REPORT_DIR_NAME": "timing",
        },
        "OUTPUT": {
            "ENABLED": True,
            "SAMPLE_INDICES": [],
            "TIME_INDICES": [],
            "IMAGE_OUTPUT": {"ENABLED": False},
            "GIF_OUTPUT": {"ENABLED": False},
            "DETAIL_EVAL": {"ENABLED": False},
            "IG_ANALYSIS": {"ENABLED": False},
        },
    }


def test_generate_all_outputs_saves_single_model_prediction_timing(tmp_path):
    test_loader = [
        {
            "x": torch.ones(2, 1, 2, 2, 2),
            "y": torch.ones(2, 1, 2, 2, 2),
        }
    ]

    results = generate_all_outputs(
        config=_minimal_config(tmp_path),
        channel_normalizer=IdentityNormalizer(),
        device="cpu",
        trained_model=EchoModel(),
        train_dataset=[],
        val_dataset=[],
        test_dataset=[],
        test_loader=test_loader,
        verbose=False,
    )

    timing_path = tmp_path / "timing" / "prediction_timing.json"
    assert timing_path.exists()
    assert results["timing"][0]["scope"] == "single_model"
    assert results["timing"][0]["n_samples"] == 2
    assert json.loads(timing_path.read_text())[0]["scope"] == "single_model"


def test_generate_all_outputs_saves_ensemble_prediction_timing(tmp_path):
    test_loader = [
        {
            "x": torch.ones(2, 1, 2, 2, 2),
            "y": torch.ones(2, 1, 2, 2, 2),
        }
    ]
    predictor = EnsemblePredictor(
        models=[EchoModel(), EchoModel()],
        device="cpu",
        keep_models_on_device=True,
    )

    results = generate_all_outputs(
        config=_minimal_config(tmp_path),
        channel_normalizer=IdentityNormalizer(),
        device="cpu",
        trained_model=predictor,
        train_dataset=[],
        val_dataset=[],
        test_dataset=[],
        test_loader=test_loader,
        verbose=False,
    )

    scopes = {record["scope"] for record in results["timing"]}
    assert scopes == {"single_model", "ensemble"}
    assert (tmp_path / "timing" / "prediction_timing.csv").exists()
