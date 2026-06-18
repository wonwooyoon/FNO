import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from eval_compare_fno_unet import (
    compute_comparison_tables,
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
