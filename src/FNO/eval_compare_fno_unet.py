"""
Standalone final evaluation comparison for FNO and U-Net ensembles.

This script compares prediction accuracy on the shared held-out test split.
It intentionally excludes training and prediction timing because those are
already reported by the model-specific pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEFAULT_FNO_MANIFEST = Path("src/FNO/output_pure/ensemble_manifest.json")
DEFAULT_UNET_MANIFEST = Path("src/FNO/output_unet/ensemble_manifest.json")
DEFAULT_OUTPUT_DIR = Path("src/FNO/output_compare_fno_unet")
EPS = 1e-12


def compute_comparison_tables(
    *,
    fno_pred: Any,
    unet_pred: Any,
    gt: Any,
    test_indices: Optional[Sequence[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute all comparison CSV tables from physical-value predictions."""
    fno_np = _to_numpy_field(fno_pred)
    unet_np = _to_numpy_field(unet_pred)
    gt_np = _to_numpy_field(gt)
    _validate_same_shape(fno_np, unet_np, gt_np)

    fno_err = fno_np - gt_np
    unet_err = unet_np - gt_np
    n_samples = gt_np.shape[0]
    n_time = gt_np.shape[-1]
    sample_ids = _sample_ids(n_samples, test_indices)

    summary_metrics = pd.DataFrame(
        [
            _summary_row("FNO", fno_np, gt_np, fno_err),
            _summary_row("UNET", unet_np, gt_np, unet_err),
        ]
    )

    metric_deltas = _metric_deltas(summary_metrics)
    per_sample_metrics = _per_sample_metrics(
        fno_pred=fno_np,
        unet_pred=unet_np,
        gt=gt_np,
        fno_err=fno_err,
        unet_err=unet_err,
        sample_ids=sample_ids,
    )
    per_time_metrics = _per_time_metrics(
        fno_pred=fno_np,
        unet_pred=unet_np,
        gt=gt_np,
        fno_err=fno_err,
        unet_err=unet_err,
        n_time=n_time,
    )
    error_distribution = _error_distribution(fno_err, unet_err)
    error_distribution_delta = _distribution_deltas(error_distribution)
    win_rates = _win_rates(fno_err, unet_err, n_time)

    return {
        "summary_metrics": summary_metrics,
        "metric_deltas": metric_deltas,
        "per_sample_metrics": per_sample_metrics,
        "per_time_metrics": per_time_metrics,
        "error_distribution": error_distribution,
        "error_distribution_delta": error_distribution_delta,
        "win_rates": win_rates,
    }


def save_comparison_outputs(
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, List[Path]]:
    """Save comparison CSV, metadata JSON, and summary plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []
    for table_name, table in tables.items():
        csv_path = output_dir / f"{table_name}.csv"
        table.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

    metadata_path = output_dir / "comparison_metadata.json"
    metadata_path.write_text(json.dumps(_json_safe(dict(metadata)), indent=2))

    plot_paths = [
        _plot_per_time_rmse_mae(tables["per_time_metrics"], output_dir / "per_time_rmse_mae.png"),
        _plot_metric_deltas(tables["metric_deltas"], output_dir / "metric_deltas.png"),
        _plot_error_distribution(tables["error_distribution"], output_dir / "error_distribution_histograms.png"),
    ]

    return {
        "csv": csv_paths,
        "json": [metadata_path],
        "plots": plot_paths,
    }


def run_comparison(
    *,
    fno_manifest: Path,
    unet_manifest: Path,
    output_dir: Path,
    device: str = "auto",
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load both final ensembles, recompute predictions, and save comparison outputs."""
    import FNO
    import UNET

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be >= 1 when provided")

    resolved_device = _resolve_device(device)
    fno_config = dict(FNO.CONFIG)
    unet_config = dict(UNET.CONFIG)
    _validate_eval_configs(fno_config, unet_config)

    (
        channel_normalizer,
        trainval_dataset,
        test_dataset,
        fixed_split,
        train_dataset,
        val_dataset,
        _,
    ) = FNO.preprocessing(
        config=fno_config,
        verbose=verbose,
        return_split=True,
    )
    channel_normalizer = channel_normalizer.to(resolved_device)

    fno_model, fno_loader, _, fno_loaded_manifest = FNO.load_ensemble_for_evaluation(
        fno_config,
        Path(fno_manifest),
        train_dataset,
        val_dataset,
        test_dataset,
        resolved_device,
    )
    unet_model, unet_loader, _, unet_loaded_manifest = UNET.load_ensemble_for_evaluation(
        unet_config,
        Path(unet_manifest),
        train_dataset,
        val_dataset,
        test_dataset,
        resolved_device,
    )

    fno_pred, gt = _predict_physical(
        model=fno_model,
        data_loader=fno_loader,
        channel_normalizer=channel_normalizer,
        device=resolved_device,
        max_batches=max_batches,
    )
    unet_pred, unet_gt = _predict_physical(
        model=unet_model,
        data_loader=unet_loader,
        channel_normalizer=channel_normalizer,
        device=resolved_device,
        max_batches=max_batches,
    )
    if gt.shape != unet_gt.shape:
        raise ValueError(
            f"FNO and UNET loaders produced different ground-truth shapes: {gt.shape} vs {unet_gt.shape}"
        )
    if not np.allclose(gt, unet_gt, rtol=0.0, atol=1e-8):
        raise ValueError("FNO and UNET loaders produced different ground-truth tensors")

    test_indices = fixed_split.test_indices
    if max_batches is not None:
        test_indices = test_indices[: gt.shape[0]]

    tables = compute_comparison_tables(
        fno_pred=fno_pred,
        unet_pred=unet_pred,
        gt=gt,
        test_indices=test_indices,
    )
    metadata = {
        "fno_manifest": str(Path(fno_manifest)),
        "unet_manifest": str(Path(unet_manifest)),
        "output_dir": str(Path(output_dir)),
        "device": resolved_device,
        "n_samples": int(gt.shape[0]),
        "field_shape": list(gt.shape[2:]),
        "test_indices": [int(idx) for idx in test_indices],
        "max_batches": max_batches,
        "fno_n_members": len(fno_loaded_manifest.get("members", [])),
        "unet_n_members": len(unet_loaded_manifest.get("members", [])),
        "metric_basis": "physical_values",
        "error_definition": "prediction_minus_ground_truth",
        "delta_definition": "FNO - UNET",
    }
    paths = save_comparison_outputs(
        tables=tables,
        metadata=metadata,
        output_dir=Path(output_dir),
    )

    if verbose:
        print(f"\nComparison outputs saved to: {Path(output_dir)}")
        print(tables["summary_metrics"].to_string(index=False))

    return {
        "tables": tables,
        "metadata": metadata,
        "paths": paths,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare final FNO and U-Net ensemble predictions on the shared test split."
    )
    parser.add_argument("--fno-manifest", type=Path, default=DEFAULT_FNO_MANIFEST)
    parser.add_argument("--unet-manifest", type=Path, default=DEFAULT_UNET_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_comparison(
        fno_manifest=args.fno_manifest,
        unet_manifest=args.unet_manifest,
        output_dir=args.output_dir,
        device=args.device,
        max_batches=args.max_batches,
        verbose=not args.quiet,
    )


def _to_numpy_field(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = array.astype(np.float64, copy=False)
    if array.ndim != 5:
        raise ValueError(f"Expected field array with shape (N, C, nx, ny, nt), got {array.shape}")
    if array.shape[1] != 1:
        raise ValueError(f"Expected single output channel, got shape {array.shape}")
    return array


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Prediction and ground-truth shapes differ: {sorted(shapes)}")


def _sample_ids(n_samples: int, test_indices: Optional[Sequence[int]]) -> List[int]:
    if test_indices is None:
        return list(range(n_samples))
    if len(test_indices) != n_samples:
        raise ValueError(f"Expected {n_samples} test indices, got {len(test_indices)}")
    return [int(idx) for idx in test_indices]


def _summary_row(model_name: str, pred: np.ndarray, gt: np.ndarray, err: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(err)
    per_sample_rel_l2 = [_relative_l2(pred[i], gt[i]) for i in range(gt.shape[0])]
    return {
        "model": model_name,
        "n_samples": int(gt.shape[0]),
        "mse": _mse(err),
        "rmse": _rmse(err),
        "mae": _mae(err),
        "bias_mean": float(np.mean(err)),
        "abs_bias_mean": float(abs(np.mean(err))),
        "max_abs_error": float(np.max(abs_err)),
        "relative_l2_mean": float(np.mean(per_sample_rel_l2)),
        "global_r2": _global_r2(pred, gt),
    }


def _metric_deltas(summary_metrics: pd.DataFrame) -> pd.DataFrame:
    fno = summary_metrics[summary_metrics["model"] == "FNO"].iloc[0]
    unet = summary_metrics[summary_metrics["model"] == "UNET"].iloc[0]
    higher_is_better = {"global_r2"}
    metrics = [
        "mse",
        "rmse",
        "mae",
        "bias_mean",
        "abs_bias_mean",
        "max_abs_error",
        "relative_l2_mean",
        "global_r2",
    ]
    rows = []
    for metric in metrics:
        delta = float(fno[metric] - unet[metric])
        if metric == "bias_mean":
            better_model = "NA"
        elif metric in higher_is_better:
            better_model = "FNO" if delta > 0 else "UNET" if delta < 0 else "tie"
        else:
            better_model = "FNO" if delta < 0 else "UNET" if delta > 0 else "tie"
        rows.append(
            {
                "metric": metric,
                "fno": float(fno[metric]),
                "unet": float(unet[metric]),
                "delta_fno_minus_unet": delta,
                "better_model": better_model,
            }
        )
    return pd.DataFrame(rows)


def _per_sample_metrics(
    *,
    fno_pred: np.ndarray,
    unet_pred: np.ndarray,
    gt: np.ndarray,
    fno_err: np.ndarray,
    unet_err: np.ndarray,
    sample_ids: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for model_name, pred, err in (
        ("FNO", fno_pred, fno_err),
        ("UNET", unet_pred, unet_err),
    ):
        for sample_pos, test_index in enumerate(sample_ids):
            sample_err = err[sample_pos]
            rows.append(
                {
                    "model": model_name,
                    "sample_pos": sample_pos,
                    "test_index": int(test_index),
                    "mse": _mse(sample_err),
                    "rmse": _rmse(sample_err),
                    "mae": _mae(sample_err),
                    "bias_mean": float(np.mean(sample_err)),
                    "max_abs_error": float(np.max(np.abs(sample_err))),
                    "relative_l2": _relative_l2(pred[sample_pos], gt[sample_pos]),
                }
            )
    return pd.DataFrame(rows)


def _per_time_metrics(
    *,
    fno_pred: np.ndarray,
    unet_pred: np.ndarray,
    gt: np.ndarray,
    fno_err: np.ndarray,
    unet_err: np.ndarray,
    n_time: int,
) -> pd.DataFrame:
    model_rows = []
    for model_name, pred, err in (
        ("FNO", fno_pred, fno_err),
        ("UNET", unet_pred, unet_err),
    ):
        for time_idx in range(n_time):
            time_err = err[..., time_idx]
            model_rows.append(
                {
                    "model": model_name,
                    "time": time_idx,
                    "mse": _mse(time_err),
                    "rmse": _rmse(time_err),
                    "mae": _mae(time_err),
                    "bias_mean": float(np.mean(time_err)),
                    "max_abs_error": float(np.max(np.abs(time_err))),
                    "relative_l2": _relative_l2(pred[..., time_idx], gt[..., time_idx]),
                }
            )
    df = pd.DataFrame(model_rows)
    delta_rows = []
    metrics = ["mse", "rmse", "mae", "bias_mean", "max_abs_error", "relative_l2"]
    for time_idx in range(n_time):
        fno_row = df[(df["model"] == "FNO") & (df["time"] == time_idx)].iloc[0]
        unet_row = df[(df["model"] == "UNET") & (df["time"] == time_idx)].iloc[0]
        delta = {
            "model": "FNO-UNET",
            "time": time_idx,
        }
        for metric in metrics:
            delta[metric] = float(fno_row[metric] - unet_row[metric])
        delta_rows.append(delta)
    return pd.concat([df, pd.DataFrame(delta_rows)], ignore_index=True)


def _error_distribution(fno_err: np.ndarray, unet_err: np.ndarray) -> pd.DataFrame:
    rows = []
    for model_name, err in (("FNO", fno_err), ("UNET", unet_err)):
        for error_type, values in (
            ("signed", err.reshape(-1)),
            ("absolute", np.abs(err).reshape(-1)),
        ):
            stats = _distribution_stats(values)
            for statistic, value in stats.items():
                rows.append(
                    {
                        "model": model_name,
                        "error_type": error_type,
                        "statistic": statistic,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def _distribution_deltas(error_distribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = error_distribution.groupby(["error_type", "statistic"], sort=False)
    for (error_type, statistic), group in grouped:
        values = group.set_index("model")["value"]
        fno = float(values["FNO"])
        unet = float(values["UNET"])
        rows.append(
            {
                "error_type": error_type,
                "statistic": statistic,
                "fno": fno,
                "unet": unet,
                "delta_fno_minus_unet": fno - unet,
            }
        )
    return pd.DataFrame(rows)


def _win_rates(fno_err: np.ndarray, unet_err: np.ndarray, n_time: int) -> pd.DataFrame:
    fno_abs = np.abs(fno_err)
    unet_abs = np.abs(unet_err)
    rows = [
        {
            "scope": "global",
            "time": "all",
            "fno_abs_error_lt_unet_rate": float(np.mean(fno_abs < unet_abs)),
            "unet_abs_error_lt_fno_rate": float(np.mean(unet_abs < fno_abs)),
            "tie_rate": float(np.mean(fno_abs == unet_abs)),
        }
    ]
    for time_idx in range(n_time):
        fno_t = fno_abs[..., time_idx]
        unet_t = unet_abs[..., time_idx]
        rows.append(
            {
                "scope": f"time_{time_idx}",
                "time": time_idx,
                "fno_abs_error_lt_unet_rate": float(np.mean(fno_t < unet_t)),
                "unet_abs_error_lt_fno_rate": float(np.mean(unet_t < fno_t)),
                "tie_rate": float(np.mean(fno_t == unet_t)),
            }
        )
    return pd.DataFrame(rows)


def _distribution_stats(values: np.ndarray) -> Dict[str, float]:
    quantiles = {
        "q01": 0.01,
        "q05": 0.05,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q95": 0.95,
        "q99": 0.99,
    }
    stats = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
    }
    stats.update({name: float(np.quantile(values, q)) for name, q in quantiles.items()})
    stats["max"] = float(np.max(values))
    return stats


def _predict_physical(
    *,
    model: torch.nn.Module,
    data_loader: Iterable[Dict[str, torch.Tensor]],
    channel_normalizer: Any,
    device: str,
    max_batches: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_initial = batch.get("y_initial")
            if y_initial is not None:
                y_initial = y_initial.to(device)
            pred = model(x)
            pred_phys = channel_normalizer.inverse_transform_output(pred, y_initial=y_initial)
            y_phys = channel_normalizer.inverse_transform_output(y, y_initial=y_initial)
            all_pred.append(pred_phys.detach().cpu())
            all_gt.append(y_phys.detach().cpu())
    if not all_pred:
        raise ValueError("No batches were evaluated")
    return torch.cat(all_pred, dim=0).numpy(), torch.cat(all_gt, dim=0).numpy()


def _validate_eval_configs(fno_config: Mapping[str, Any], unet_config: Mapping[str, Any]) -> None:
    keys = ["MERGED_PT_PATH", "CHANNEL_NORMALIZER_PATH", "TEST_SIZE", "RANDOM_STATE"]
    mismatches = []
    for key in keys:
        if fno_config.get(key) != unet_config.get(key):
            mismatches.append((key, fno_config.get(key), unet_config.get(key)))
    if mismatches:
        details = "; ".join(f"{key}: FNO={fno!r}, UNET={unet!r}" for key, fno, unet in mismatches)
        raise ValueError(f"FNO and UNET evaluation configs must match for shared-test comparison: {details}")


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _plot_per_time_rmse_mae(per_time_metrics: pd.DataFrame, output_path: Path) -> Path:
    model_df = per_time_metrics[per_time_metrics["model"].isin(["FNO", "UNET"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
    for metric, ax in zip(["rmse", "mae"], axes):
        for model_name, group in model_df.groupby("model"):
            ax.plot(group["time"], group[metric], marker="o", linewidth=2, label=model_name)
        ax.set_xlabel("Time index")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_metric_deltas(metric_deltas: pd.DataFrame, output_path: Path) -> Path:
    plot_df = metric_deltas[metric_deltas["metric"].isin(["rmse", "mae", "relative_l2_mean", "global_r2"])]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    colors = ["#2b8cbe" if value <= 0 else "#de2d26" for value in plot_df["delta_fno_minus_unet"]]
    ax.bar(plot_df["metric"], plot_df["delta_fno_minus_unet"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("FNO - UNET")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_error_distribution(error_distribution: pd.DataFrame, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
    for error_type, ax in zip(["signed", "absolute"], axes):
        subset = error_distribution[error_distribution["error_type"] == error_type]
        pivot = subset.pivot(index="statistic", columns="model", values="value")
        stats = ["q05", "q25", "q50", "q75", "q95"]
        x = np.arange(len(stats))
        width = 0.35
        ax.bar(x - width / 2, pivot.loc[stats, "FNO"], width, label="FNO")
        ax.bar(x + width / 2, pivot.loc[stats, "UNET"], width, label="UNET")
        ax.set_xticks(x)
        ax.set_xticklabels(stats)
        ax.set_title(f"{error_type.title()} error quantiles")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _mse(err: np.ndarray) -> float:
    return float(np.mean(np.square(err)))


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(_mse(err)))


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def _relative_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm((pred - gt).reshape(-1), ord=2) / (np.linalg.norm(gt.reshape(-1), ord=2) + EPS))


def _global_r2(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    ss_res = float(np.sum(np.square(gt_flat - pred_flat)))
    ss_tot = float(np.sum(np.square(gt_flat - np.mean(gt_flat))))
    if ss_tot < EPS:
        return 1.0 if ss_res < EPS else 0.0
    return float(1.0 - ss_res / ss_tot)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
