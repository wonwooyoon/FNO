import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from eval_compare_fno_unet import (
    compute_comparison_tables,
    run_comparison,
    save_comparison_outputs,
)


def _toy_arrays():
    gt = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 1, 1, 1, 2)
    fno = gt + np.array([1.0, -1.0, 1.0, -1.0]).reshape(2, 1, 1, 1, 2)
    unet = gt + np.array([2.0, 0.0, 2.0, 0.0]).reshape(2, 1, 1, 1, 2)
    return fno, unet, gt


def test_compute_comparison_tables_uses_fno_minus_unet_delta():
    fno, unet, gt = _toy_arrays()

    tables = compute_comparison_tables(
        fno_pred=fno,
        unet_pred=unet,
        gt=gt,
        test_indices=[10, 11],
    )

    summary = tables["summary_metrics"]
    deltas = tables["metric_deltas"].set_index("metric")

    fno_rmse = summary.loc[summary["model"] == "FNO", "rmse"].iloc[0]
    unet_rmse = summary.loc[summary["model"] == "UNET", "rmse"].iloc[0]
    assert np.isclose(fno_rmse, 1.0)
    assert np.isclose(unet_rmse, np.sqrt(2.0))
    assert np.isclose(deltas.loc["rmse", "delta_fno_minus_unet"], 1.0 - np.sqrt(2.0))
    assert np.isclose(deltas.loc["bias_mean", "delta_fno_minus_unet"], -1.0)

    per_sample = tables["per_sample_metrics"]
    assert per_sample["test_index"].tolist() == [10, 11, 10, 11]
    assert set(per_sample["model"]) == {"FNO", "UNET"}


def test_distribution_and_win_rate_tables_compare_error_behavior():
    fno, unet, gt = _toy_arrays()

    tables = compute_comparison_tables(fno_pred=fno, unet_pred=unet, gt=gt)

    dist_delta = tables["error_distribution_delta"]
    signed_mean_delta = dist_delta[
        (dist_delta["error_type"] == "signed")
        & (dist_delta["statistic"] == "mean")
    ]["delta_fno_minus_unet"].iloc[0]
    assert np.isclose(signed_mean_delta, -1.0)

    win_rates = tables["win_rates"].set_index("scope")
    assert np.isclose(win_rates.loc["global", "fno_abs_error_lt_unet_rate"], 0.5)


def test_save_comparison_outputs_writes_csv_json_and_plots(tmp_path):
    fno, unet, gt = _toy_arrays()
    tables = compute_comparison_tables(fno_pred=fno, unet_pred=unet, gt=gt)
    metadata = {
        "fno_manifest": "fno.json",
        "unet_manifest": "unet.json",
        "n_samples": 2,
    }

    paths = save_comparison_outputs(
        tables=tables,
        metadata=metadata,
        output_dir=tmp_path,
    )

    expected_csvs = {
        "summary_metrics.csv",
        "metric_deltas.csv",
        "per_sample_metrics.csv",
        "per_time_metrics.csv",
        "error_distribution.csv",
        "error_distribution_delta.csv",
        "win_rates.csv",
    }
    assert expected_csvs.issubset({path.name for path in paths["csv"]})
    assert (tmp_path / "per_time_rmse_mae.png").exists()
    assert (tmp_path / "metric_deltas.png").exists()
    assert (tmp_path / "error_distribution_histograms.png").exists()
    assert json.loads((tmp_path / "comparison_metadata.json").read_text()) == metadata

    saved_summary = pd.read_csv(tmp_path / "summary_metrics.csv")
    assert saved_summary["model"].tolist() == ["FNO", "UNET"]


class _IdentityNormalizer:
    def to(self, device):
        return self

    def inverse_transform_output(self, value, y_initial=None):
        return value


class _ScaleModel(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        if x.ndim == 5:
            return x[:, :1] * self.scale
        if x.ndim == 4:
            return x[:, :1].unsqueeze(-1).expand(-1, -1, -1, -1, 2) * self.scale
        raise AssertionError(f"Unexpected input shape: {tuple(x.shape)}")


def _fake_preprocessing(expected_input_ndim, split_indices):
    dataset = SimpleNamespace(input_ndim=expected_input_ndim)
    split = SimpleNamespace(test_indices=np.asarray(split_indices))
    return (
        _IdentityNormalizer(),
        dataset,
        dataset,
        split,
        dataset,
        dataset,
        "cpu",
    )


def _fake_loader(expected_input_ndim, *, pred_value, target_value, n_samples=1):
    if expected_input_ndim == 5:
        x = torch.full((n_samples, 1, 1, 1, 2), float(pred_value))
    elif expected_input_ndim == 4:
        x = torch.full((n_samples, 1, 1, 1), float(pred_value))
    else:
        raise AssertionError(f"Unexpected expected_input_ndim={expected_input_ndim}")
    y = torch.full((n_samples, 1, 1, 1, 2), float(target_value))
    return [{"x": x, "y": y}]


def _install_fake_model_modules(
    monkeypatch,
    *,
    fno_split=(7,),
    unet_split=(7,),
    fno_batch_samples=1,
    unet_batch_samples=1,
):
    calls = []

    fake_fno = SimpleNamespace(
        CONFIG={
            "MERGED_PT_PATH": "data.pt",
            "CHANNEL_NORMALIZER_PATH": "normalizer.pkl",
            "TEST_SIZE": 0.1,
            "RANDOM_STATE": 42,
        }
    )
    fake_unet_new = SimpleNamespace(
        CONFIG={
            "MODEL_KIND": "unet_new",
            "MERGED_PT_PATH": "data.pt",
            "CHANNEL_NORMALIZER_PATH": "normalizer.pkl",
            "TEST_SIZE": 0.1,
            "RANDOM_STATE": 42,
        }
    )
    def fno_preprocessing(config, verbose, return_split):
        calls.append(("FNO.preprocessing", return_split))
        return _fake_preprocessing(5, fno_split)

    def unet_new_preprocessing(config, verbose, return_split):
        calls.append(("UNET_new.preprocessing", return_split))
        return _fake_preprocessing(4, unet_split)

    def fno_load(config, manifest_file, train_dataset, val_dataset, test_dataset, device):
        assert train_dataset.input_ndim == 5
        assert val_dataset.input_ndim == 5
        assert test_dataset.input_ndim == 5
        calls.append(("FNO.load", Path(manifest_file).name))
        return (
            _ScaleModel(1.0),
            _fake_loader(5, pred_value=2.0, target_value=1.0, n_samples=fno_batch_samples),
            None,
            {"members": [1]},
        )

    def unet_new_load(config, manifest_file, train_dataset, val_dataset, test_dataset, device):
        assert train_dataset.input_ndim == 4
        assert val_dataset.input_ndim == 4
        assert test_dataset.input_ndim == 4
        calls.append(("UNET_new.load", Path(manifest_file).name))
        return (
            _ScaleModel(2.0),
            _fake_loader(4, pred_value=2.0, target_value=1.0, n_samples=unet_batch_samples),
            None,
            {"members": [1, 2]},
        )

    fake_fno.preprocessing = fno_preprocessing
    fake_fno.load_ensemble_for_evaluation = fno_load
    fake_unet_new.preprocessing = unet_new_preprocessing
    fake_unet_new.load_ensemble_for_evaluation = unet_new_load

    monkeypatch.setitem(sys.modules, "FNO", fake_fno)
    monkeypatch.setitem(sys.modules, "UNET_new", fake_unet_new)
    return calls


def test_run_comparison_uses_unet_new_preprocessing_and_loader(monkeypatch, tmp_path):
    calls = _install_fake_model_modules(monkeypatch)

    result = run_comparison(
        fno_manifest=tmp_path / "fno_manifest.json",
        unet_manifest=tmp_path / "unet_manifest.json",
        output_dir=tmp_path / "comparison",
        device="cpu",
        verbose=False,
    )

    assert ("FNO.preprocessing", True) in calls
    assert ("UNET_new.preprocessing", True) in calls
    assert ("FNO.load", "fno_manifest.json") in calls
    assert ("UNET_new.load", "unet_manifest.json") in calls
    assert result["metadata"]["unet_model_kind"] == "unet_new"
    assert result["metadata"]["unet_n_members"] == 2
    assert result["tables"]["summary_metrics"]["model"].tolist() == ["FNO", "UNET"]


def test_run_comparison_rejects_mismatched_fno_and_unet_new_test_splits(monkeypatch, tmp_path):
    _install_fake_model_modules(monkeypatch, fno_split=(7,), unet_split=(8,))

    with pytest.raises(ValueError, match="test split"):
        run_comparison(
            fno_manifest=tmp_path / "fno_manifest.json",
            unet_manifest=tmp_path / "unet_manifest.json",
            output_dir=tmp_path / "comparison",
            device="cpu",
            verbose=False,
        )


def test_run_comparison_limits_to_shared_sample_count_when_max_batches_uses_different_batch_sizes(
    monkeypatch,
    tmp_path,
):
    _install_fake_model_modules(
        monkeypatch,
        fno_split=(7, 8),
        unet_split=(7, 8),
        fno_batch_samples=1,
        unet_batch_samples=2,
    )

    result = run_comparison(
        fno_manifest=tmp_path / "fno_manifest.json",
        unet_manifest=tmp_path / "unet_manifest.json",
        output_dir=tmp_path / "comparison",
        device="cpu",
        max_batches=1,
        verbose=False,
    )

    assert result["metadata"]["n_samples"] == 1
    assert result["metadata"]["test_indices"] == [7]
    assert set(result["tables"]["per_sample_metrics"]["sample_pos"]) == {0}
