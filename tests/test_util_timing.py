import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from util_timing import measure_forward_time, save_timing_report


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


def test_measure_forward_time_counts_batches_samples_and_elapsed_seconds():
    test_loader = [
        {"x": torch.ones(2, 1)},
        {"x": torch.ones(3, 1)},
    ]
    clock_values = iter([10.0, 10.25])

    record = measure_forward_time(
        model=IdentityModel(),
        data_loader=test_loader,
        device="cpu",
        model_kind="test",
        scope="single_model",
        n_models=1,
        warmup_batches=0,
        timer=lambda: next(clock_values),
    )

    assert record["model_kind"] == "test"
    assert record["scope"] == "single_model"
    assert record["device"] == "cpu"
    assert record["n_models"] == 1
    assert record["n_batches"] == 2
    assert record["n_samples"] == 5
    assert record["total_seconds"] == 0.25
    assert record["seconds_per_sample"] == 0.05
    assert record["seconds_per_batch"] == 0.125


def test_save_timing_report_writes_json_and_csv(tmp_path):
    records = [
        {
            "model_kind": "test",
            "scope": "single_model",
            "device": "cpu",
            "n_models": 1,
            "n_batches": 2,
            "n_samples": 5,
            "total_seconds": 0.25,
            "seconds_per_sample": 0.05,
            "seconds_per_batch": 0.125,
        }
    ]

    paths = save_timing_report(records, tmp_path / "prediction_timing")

    assert paths["json"].name == "prediction_timing.json"
    assert paths["csv"].name == "prediction_timing.csv"
    assert json.loads(paths["json"].read_text()) == records

    with paths["csv"].open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["scope"] == "single_model"
    assert rows[0]["n_samples"] == "5"
