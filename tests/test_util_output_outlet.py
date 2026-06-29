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
from util_output_outlet import visualize_outlet_predictions


class IdentityOutletNormalizer:
    eps = 1e-12

    def inverse_transform(self, y):
        return y


class OutletEchoModel(nn.Module):
    def forward(self, x):
        return x[:, 0, 0, 0, :]


def _minimal_config(tmp_path):
    return {
        "MODEL_KIND": "fno_outlet",
        "OUTPUT_DIR": str(tmp_path),
        "TIMING": {
            "ENABLED": True,
            "PREDICTION_WARMUP_BATCHES": 0,
            "REPORT_DIR_NAME": "timing",
        },
        "VISUALIZATION": {
            "N_SAMPLES": 2,
        },
    }


def _test_loader():
    y = torch.ones(2, 20)
    return [
        {
            "x": torch.ones(2, 1, 2, 2, 20),
            "y": y,
        }
    ]


def test_visualize_outlet_predictions_saves_single_model_prediction_timing(tmp_path):
    results = visualize_outlet_predictions(
        config=_minimal_config(tmp_path),
        device="cpu",
        model=OutletEchoModel(),
        test_loader=_test_loader(),
        outlet_normalizer=IdentityOutletNormalizer(),
        sample_indices=[0, 1],
    )

    timing_path = tmp_path / "timing" / "prediction_timing.json"
    assert timing_path.exists()
    assert results["timing"][0]["scope"] == "single_model"
    assert results["timing"][0]["n_samples"] == 2
    assert json.loads(timing_path.read_text())[0]["model_kind"] == "fno_outlet"


def test_visualize_outlet_predictions_saves_ensemble_prediction_timing(tmp_path):
    predictor = EnsemblePredictor(
        models=[OutletEchoModel(), OutletEchoModel()],
        device="cpu",
        keep_models_on_device=True,
    )

    results = visualize_outlet_predictions(
        config=_minimal_config(tmp_path),
        device="cpu",
        model=predictor,
        test_loader=_test_loader(),
        outlet_normalizer=IdentityOutletNormalizer(),
        sample_indices=[0, 1],
    )

    scopes = {record["scope"] for record in results["timing"]}
    assert scopes == {"single_model", "ensemble"}
    assert (tmp_path / "timing" / "prediction_timing.csv").exists()
