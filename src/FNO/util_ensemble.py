"""
Shared ensemble utilities for FNO-style surrogate models.

This module keeps split handling, manifest I/O, and ensemble prediction
architecture-agnostic so spatial FNO, outlet FNO, and future architectures can
reuse the same orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset


@dataclass(frozen=True)
class FixedSplit:
    trainval_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class MemberSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass
class EnsembleRun:
    predictor: "EnsemblePredictor"
    manifest_path: Path
    member_records: List[Dict[str, Any]]
    params: Dict[str, Any]
    test_loader: Any
    loss_fn: Any
    train_dataset: Any
    val_dataset: Any


class EnsemblePredictor(nn.Module):
    """Average predictions from multiple trained models.

    The forward method intentionally mirrors a normal PyTorch model so existing
    evaluation, visualization, and Integrated Gradients code can call
    ``model(x)`` without knowing whether it is a single model or an ensemble.
    """

    def __init__(self, models: Sequence[nn.Module], device: str, reduction: str = "mean"):
        super().__init__()
        if not models:
            raise ValueError("EnsemblePredictor requires at least one model")
        if reduction != "mean":
            raise ValueError("Only mean ensemble reduction is currently supported")

        self.models = nn.ModuleList(models)
        self.device = device
        self.reduction = reduction

        for model in self.models:
            model.to(device)
            model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [model(x) for model in self.models]
        return torch.stack(predictions, dim=0).mean(dim=0)


def fixed_test_split(dataset, test_size: float, random_state: int) -> Tuple[Subset, Subset, FixedSplit]:
    """Split a dataset once into train/validation pool and held-out test set."""
    original_indices = np.arange(len(dataset))
    trainval_indices, test_indices = train_test_split(
        original_indices,
        test_size=test_size,
        random_state=random_state,
    )
    return (
        Subset(dataset, trainval_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
        FixedSplit(trainval_indices=np.asarray(trainval_indices), test_indices=np.asarray(test_indices)),
    )


def make_member_split(
    trainval_dataset,
    trainval_original_indices: Sequence[int],
    val_size_relative: float,
    random_state: int,
) -> Tuple[Subset, Subset, MemberSplit]:
    """Create one randomized train/validation split from the fixed trainval pool."""
    if not 0.0 < val_size_relative < 1.0:
        raise ValueError(f"val_size_relative must be between 0 and 1, got {val_size_relative}")

    positions = np.arange(len(trainval_dataset))
    train_positions, val_positions, train_orig, val_orig = train_test_split(
        positions,
        np.asarray(trainval_original_indices),
        test_size=val_size_relative,
        random_state=random_state,
    )

    return (
        Subset(trainval_dataset, train_positions.tolist()),
        Subset(trainval_dataset, val_positions.tolist()),
        MemberSplit(train_indices=np.asarray(train_orig), val_indices=np.asarray(val_orig)),
    )


def resolve_domain_padding(params: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Optuna's domain_padding_idx representation to domain_padding."""
    resolved = dict(params)
    if "domain_padding_idx" in resolved:
        padding_idx = int(resolved.pop("domain_padding_idx"))
        options = search_space["domain_padding_options"]
        resolved["domain_padding"] = tuple(options[padding_idx])
    return resolved


def resolve_training_params(
    mode: str,
    config: Dict[str, Any],
    optuna_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return canonical training hyperparameters for single or optuna modes."""
    if mode == "single":
        return dict(config["SINGLE_PARAMS"])
    if mode == "optuna":
        if optuna_results is None:
            raise ValueError("optuna_results is required when mode='optuna'")
        return resolve_domain_padding(
            optuna_results["best_params"],
            config["OPTUNA_SEARCH_SPACE"],
        )
    raise ValueError(f"Cannot resolve training params for mode: {mode}")


def ensemble_config(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "ENABLED": False,
        "N_MODELS": 1,
        "BASE_SEED": config.get("RANDOM_STATE", 42),
        "SPLIT_SEED_STRATEGY": "base_plus_member",
        "MEMBER_OUTPUT_PATTERN": "ensemble/member_{member_id:03d}",
        "MANIFEST_NAME": "ensemble_manifest.json",
    }
    user_config = config.get("ENSEMBLE", {})
    return {**defaults, **user_config}


def n_ensemble_members(config: Dict[str, Any]) -> int:
    cfg = ensemble_config(config)
    if not cfg.get("ENABLED", False):
        return 1
    n_models = int(cfg.get("N_MODELS", 1))
    if n_models < 1:
        raise ValueError("ENSEMBLE.N_MODELS must be >= 1")
    return n_models


def member_seed(config: Dict[str, Any], member_id: int) -> int:
    cfg = ensemble_config(config)
    strategy = cfg.get("SPLIT_SEED_STRATEGY", "base_plus_member")
    base_seed = int(cfg.get("BASE_SEED", config.get("RANDOM_STATE", 42)))
    if strategy == "base_plus_member":
        return base_seed + member_id
    if strategy == "fixed":
        return base_seed
    raise ValueError(f"Unknown SPLIT_SEED_STRATEGY: {strategy}")


def member_output_dir(config: Dict[str, Any], member_id: int) -> Path:
    cfg = ensemble_config(config)
    pattern = cfg.get("MEMBER_OUTPUT_PATTERN", "ensemble/member_{member_id:03d}")
    return Path(config["OUTPUT_DIR"]) / pattern.format(member_id=member_id)


def manifest_path(config: Dict[str, Any]) -> Path:
    cfg = ensemble_config(config)
    return Path(config["OUTPUT_DIR"]) / cfg.get("MANIFEST_NAME", "ensemble_manifest.json")


def save_split_manifest_csv(
    output_path: Path,
    test_indices: Sequence[int],
    member_splits: Iterable[Tuple[int, MemberSplit]],
) -> None:
    rows = []
    for split_pos, original_idx in enumerate(test_indices):
        rows.append(
            {
                "member_id": "all",
                "split": "test",
                "original_idx": int(original_idx),
                "split_idx": split_pos,
            }
        )

    for member_id, split in member_splits:
        for split_pos, original_idx in enumerate(split.train_indices):
            rows.append(
                {
                    "member_id": member_id,
                    "split": "train",
                    "original_idx": int(original_idx),
                    "split_idx": split_pos,
                }
            )
        for split_pos, original_idx in enumerate(split.val_indices):
            rows.append(
                {
                    "member_id": member_id,
                    "split": "val",
                    "original_idx": int(original_idx),
                    "split_idx": split_pos,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_ensemble_manifest(
    manifest_path: Path,
    model_kind: str,
    params: Dict[str, Any],
    member_records: List[Dict[str, Any]],
    test_indices: Sequence[int],
    config_subset: Dict[str, Any],
) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format_version": 1,
        "model_kind": model_kind,
        "reduction": "mean",
        "params": _json_safe(params),
        "members": _json_safe(member_records),
        "test_indices": [int(idx) for idx in test_indices],
        "config": _json_safe(config_subset),
    }
    manifest_path.write_text(json.dumps(data, indent=2))
    return manifest_path


def load_ensemble_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_ensemble_from_manifest(
    manifest_file: Path,
    model_factory: Callable[[Dict[str, Any]], nn.Module],
    device: str,
) -> Tuple[EnsemblePredictor, Dict[str, Any]]:
    data = load_ensemble_manifest(manifest_file)
    base_dir = Path(manifest_file).parent
    models = []
    for member in data["members"]:
        model = model_factory(data["params"])
        state_path = Path(member["model_state_path"])
        if not state_path.is_absolute():
            state_path = base_dir / state_path
        model.load_state_dict(torch.load(state_path, map_location=device, weights_only=False))
        models.append(model)
    return EnsemblePredictor(models=models, device=device), data


def train_ensemble(
    *,
    config: Dict[str, Any],
    model_kind: str,
    trainval_dataset,
    trainval_original_indices: Sequence[int],
    test_dataset,
    test_original_indices: Sequence[int],
    device: str,
    params: Dict[str, Any],
    create_components: Callable[[Any, Any, Any, Dict[str, Any], Path], Tuple[nn.Module, Any, Any, Any, Any, Any, Any]],
    train_one: Callable[[nn.Module, Any, Any, Any, Any, Path], nn.Module],
    verbose: bool = True,
) -> EnsembleRun:
    """Train one or more models over randomized train/validation splits."""
    n_members = n_ensemble_members(config)
    val_size_relative = config["VAL_SIZE"] / (1 - config["TEST_SIZE"])
    member_records: List[Dict[str, Any]] = []
    member_splits: List[Tuple[int, MemberSplit]] = []
    trained_models: List[nn.Module] = []
    final_test_loader = None
    final_loss_fn = None
    final_train_dataset = None
    final_val_dataset = None

    for member_id in range(n_members):
        seed = member_seed(config, member_id)
        if verbose:
            print("\n" + "-" * 70)
            print(f"Training ensemble member {member_id + 1}/{n_members} (seed={seed})")
            print("-" * 70)

        train_dataset, val_dataset, split = make_member_split(
            trainval_dataset=trainval_dataset,
            trainval_original_indices=trainval_original_indices,
            val_size_relative=val_size_relative,
            random_state=seed,
        )
        member_splits.append((member_id, split))
        final_train_dataset = train_dataset
        final_val_dataset = val_dataset

        output_dir = member_output_dir(config, member_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_components(
            train_dataset,
            val_dataset,
            test_dataset,
            params,
            output_dir,
        )
        trained_model = train_one(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            output_dir,
        )
        trained_model.eval()
        trained_models.append(trained_model)
        final_test_loader = test_loader
        final_loss_fn = loss_fn

        member_records.append(
            {
                "member_id": member_id,
                "seed": seed,
                "model_state_path": str(output_dir.relative_to(Path(config["OUTPUT_DIR"])) / "best_model_state_dict.pt"),
                "loss_history_path": str(output_dir.relative_to(Path(config["OUTPUT_DIR"])) / "loss_history.pt"),
                "train_indices": [int(idx) for idx in split.train_indices],
                "val_indices": [int(idx) for idx in split.val_indices],
            }
        )

        if device == "cuda":
            torch.cuda.empty_cache()

    split_csv_path = Path(config["OUTPUT_DIR"]) / "ensemble_split_manifest.csv"
    save_split_manifest_csv(split_csv_path, test_original_indices, member_splits)

    predictor = EnsemblePredictor(trained_models, device=device)
    m_path = save_ensemble_manifest(
        manifest_path=manifest_path(config),
        model_kind=model_kind,
        params=params,
        member_records=member_records,
        test_indices=test_original_indices,
        config_subset={
            "N_EPOCHS": config.get("N_EPOCHS"),
            "OPTUNA_N_EPOCHS": config.get("TRAINING_CONFIG", {}).get("optuna_n_epochs"),
            "VAL_SIZE": config.get("VAL_SIZE"),
            "TEST_SIZE": config.get("TEST_SIZE"),
            "RANDOM_STATE": config.get("RANDOM_STATE"),
            "ENSEMBLE": ensemble_config(config),
        },
    )

    return EnsembleRun(
        predictor=predictor,
        manifest_path=m_path,
        member_records=member_records,
        params=params,
        test_loader=final_test_loader,
        loss_fn=final_loss_fn,
        train_dataset=final_train_dataset,
        val_dataset=final_val_dataset,
    )


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
