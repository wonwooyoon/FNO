"""
Shared timing helpers for FNO-style model training and prediction.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Union

import torch


TimingRecord = Dict[str, Any]


def sync_if_cuda(device: Union[str, torch.device]) -> None:
    """Synchronize CUDA work when measuring a CUDA device."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def measure_forward_time(
    *,
    model: torch.nn.Module,
    data_loader: Iterable[Dict[str, torch.Tensor]],
    device: Union[str, torch.device],
    model_kind: str,
    scope: str,
    n_models: int = 1,
    warmup_batches: int = 0,
    timer: Callable[[], float] = time.perf_counter,
) -> TimingRecord:
    """Measure forward-only wall-clock time over every batch in a data loader."""
    was_training = model.training
    model.eval()

    warmup_batches = max(0, int(warmup_batches))
    if warmup_batches:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= warmup_batches:
                    break
                x = batch["x"].to(device)
                _ = model(x)
                del x

    n_batches = 0
    n_samples = 0
    sync_if_cuda(device)
    started_at = timer()

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            _ = model(x)
            n_batches += 1
            n_samples += int(x.shape[0])
            del x

    sync_if_cuda(device)
    total_seconds = float(timer() - started_at)

    if was_training:
        model.train()

    seconds_per_sample = total_seconds / n_samples if n_samples else None
    seconds_per_batch = total_seconds / n_batches if n_batches else None

    return {
        "model_kind": model_kind,
        "scope": scope,
        "device": str(device),
        "n_models": int(n_models),
        "n_batches": int(n_batches),
        "n_samples": int(n_samples),
        "total_seconds": total_seconds,
        "seconds_per_sample": seconds_per_sample,
        "seconds_per_batch": seconds_per_batch,
    }


def save_timing_report(records: Union[TimingRecord, Sequence[TimingRecord]], output_base: Union[str, Path]) -> Dict[str, Path]:
    """Save timing records to sibling JSON and CSV files."""
    if isinstance(records, dict):
        record_list = [records]
    else:
        record_list = list(records)

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")

    json_path.write_text(json.dumps(_json_safe(record_list), indent=2))

    fieldnames = _fieldnames(record_list)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in record_list:
            writer.writerow({field: record.get(field) for field in fieldnames})

    return {"json": json_path, "csv": csv_path}


def _fieldnames(records: List[TimingRecord]) -> List[str]:
    preferred = [
        "model_kind",
        "scope",
        "device",
        "n_models",
        "n_batches",
        "n_samples",
        "total_seconds",
        "seconds_per_sample",
        "seconds_per_batch",
        "member_id",
        "seed",
        "train_seconds",
        "single_model_train_seconds",
        "ensemble_train_total_seconds",
        "sum_member_train_seconds",
        "single_model_predict_seconds",
        "ensemble_predict_seconds",
        "epochs_completed",
        "model_state_path",
    ]
    keys = []
    seen = set()
    for field in preferred:
        if any(field in record for record in records):
            keys.append(field)
            seen.add(field)
    for record in records:
        for key in record:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
