"""
Pure 2D U-Net with time-output head.

This module is a standalone U-Net variant for the same normalized data used by
FNO.py. It removes the repeated input time axis and predicts all output
timesteps as channels from a 2D spatial U-Net.
The returned prediction shape remains (N, C, nx, ny, nt), so shared training,
evaluation, ensemble, timing, and output utilities continue to work.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append('./')
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuraloperator.neuralop.training import AdamW
from util_common import LpLoss, LRStepScheduler, CappedCosineAnnealingWarmRestarts
from util_ensemble import (
    EnsemblePredictor,
    fixed_test_split,
    load_ensemble_manifest,
    make_member_split,
    member_seed,
    resolve_training_params,
    train_ensemble,
)
from util_output import generate_all_outputs
from util_training import config_for_optuna_epochs, model_evaluation_generic, train_model_generic


preprocessing_path = Path(__file__).parent.parent / "preprocessing"
if str(preprocessing_path) not in sys.path:
    sys.path.insert(0, str(preprocessing_path))
from preprocessing_normalize import ChannelNormalizer  # noqa: F401  # Needed for pickle loading.


CONFIG = {
    "MODEL_KIND": "unet_new",
    "SPECIES_TYPE": "u",
    "MERGED_PT_PATH": "./src/preprocessing/data/normalized/lr/delta/merged_normalized_U.pt",
    "CHANNEL_NORMALIZER_PATH": "./src/preprocessing/normalizers/lr/delta/normalizer_u_delta.pkl",
    "OUTPUT_DIR": "./src/FNO/output_unet_new",
    "N_EPOCHS": 1000,
    "EVAL_INTERVAL": 1,
    "VAL_SIZE": 0.1,
    "TEST_SIZE": 0.1,
    "RANDOM_STATE": 42,
    "MODEL_CONFIG": {
        "in_channels": 11,
        "out_channels": 1,
        "output_time_steps": 20,
        "init_features": 64,
        "pool_kernels": [(2, 2), (2, 2), (2, 2), (2, 2)],
        "input_time_index": 0,
        "validate_static_input": True,
        "static_input_atol": 1e-6,
        "static_input_rtol": 1e-5,
    },
    "SCHEDULER_CONFIG": {
        "scheduler_type": "step",  # Options: "cosine", "step" (validation plateau)
        "early_stopping": 40,
        "T_0": 10,
        "T_max": 40,
        "T_mult": 2,
        "eta_min": 1e-5,
        "step_size": 25,  # Plateau patience epochs before reducing LR
        "gamma": 0.5,  # LR multiplier after plateau
        "initial_lr": 1e-2,
    },
    "OUTPUT": {
        "ENABLED": True,
        "OUTPUT_DIR": "./src/FNO/output_unet_new",
        "SAMPLE_INDICES": [0, 1, 5, 10, 15, 50, 80, 120, 230],
        "TIME_INDICES": [4, 9, 14, 19],
        "DPI": 200,
        "IMAGE_OUTPUT": {
            "ENABLED": True,
            "COMBINED_IMG": True,
            "SEPARATED_IMG": True,
        },
        "GIF_OUTPUT": {
            "ENABLED": False,
            "FPS": 2,
        },
        "DETAIL_EVAL": {
            "ENABLED": False,
            "COMPUTE_RELATIVE_L2": True,
            "COMPUTE_SSIM": True,
            "PARITY_PLOT": True,
            "ADD_MEAN_COLUMN": True,
        },
        "IG_ANALYSIS": {
            "ENABLED": False,
            "SAMPLE_IDX": 3,
            "TIME_INDICES": [4, 9, 14, 19],
            "N_STEPS": 20,
            "USE_MULTI_BASELINE": True,
            "N_BASELINES": 1,
            "BASELINE_SEED": 42,
        },
    },
    "LOSS_CONFIG": {
        "loss_type": "l2",
        "l2_d": 3,
        "l2_p": 2,
    },
    "TRAINING_CONFIG": {
        "mode": "single",
        "optuna_n_trials": 100,
        "optuna_n_epochs": 100,
        "optuna_seed": 42,
        "optuna_n_startup_trials": 5,
        "eval_model_path": "./src/FNO/output_unet_new/ensemble_manifest.json",
    },
    "ENSEMBLE": {
        "ENABLED": True,
        "N_MODELS": 20,
        "BASE_SEED": 42,
        "SPLIT_SEED_STRATEGY": "base_plus_member",
        "MEMBER_OUTPUT_PATTERN": "ensemble/member_{member_id:03d}",
        "MANIFEST_NAME": "ensemble_manifest.json",
    },
    "TIMING": {
        "ENABLED": True,
        "PREDICTION_WARMUP_BATCHES": 1,
        "REPORT_DIR_NAME": "timing",
    },
    "OPTUNA_SEARCH_SPACE": {
        "depth_range": [2, 4],
        "init_features_range": [32, 84],
        "train_batch_size_options": [128],
        "l2_weight_range": [1e-9, 1e-4],
        "dropout_rate_range": [0.0, 0.2],
    },
    "SINGLE_PARAMS": {
        "depth": 4,
        "init_features": 56,
        "train_batch_size": 128,
        "l2_weight": 7.456634017940035e-06,
        "dropout_rate": 0.02488552308598347,
    },
}


class CustomDatasetPure(Dataset):
    """Dataset that drops the repeated input time axis.

    Args:
        input_tensor: Normalized input with shape (N, C, nx, ny, nt).
        output_tensor: Normalized output with shape (N, C_out, nx, ny, nt).
        initial_tensor: Initial physical values for delta reconstruction.
        input_time_index: Time slice to use for the static 2D input.
        validate_static_input: Whether to verify all input time slices match.
    """

    def __init__(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        initial_tensor: Optional[torch.Tensor] = None,
        input_time_index: int = 0,
        validate_static_input: bool = False,
        static_input_atol: float = 1e-6,
        static_input_rtol: float = 1e-5,
    ):
        if input_tensor.ndim != 5:
            raise ValueError(f"input_tensor must be 5D (N, C, nx, ny, nt), got {tuple(input_tensor.shape)}")
        if output_tensor.ndim != 5:
            raise ValueError(f"output_tensor must be 5D (N, C, nx, ny, nt), got {tuple(output_tensor.shape)}")
        if not 0 <= input_time_index < input_tensor.shape[-1]:
            raise ValueError(
                f"input_time_index={input_time_index} is out of range for nt={input_tensor.shape[-1]}"
            )
        if validate_static_input:
            reference = input_tensor[..., 0:1].expand_as(input_tensor)
            if not torch.allclose(input_tensor, reference, atol=static_input_atol, rtol=static_input_rtol):
                raise ValueError("UNET_new expects static input over the time axis, but input_tensor varies with time")

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.initial_tensor = initial_tensor
        self.input_time_index = int(input_time_index)

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": self.input_tensor[idx, :, :, :, self.input_time_index],
            "y": self.output_tensor[idx],
        }
        if self.initial_tensor is not None:
            item["y_initial"] = self.initial_tensor[idx]
        return item


def preprocessing(config: Dict, verbose: bool = True, return_split: bool = False) -> Tuple:
    """Load normalized tensors, create 2D-input datasets, and split them."""
    if verbose:
        print("\nLoading pre-normalized data and creating 2D-input U-Net datasets...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

    try:
        if verbose:
            print("Step 1: Loading normalized tensors...")

        if not Path(config["MERGED_PT_PATH"]).exists():
            raise FileNotFoundError(f"Normalized file not found: {config['MERGED_PT_PATH']}")

        bundle = torch.load(config["MERGED_PT_PATH"], map_location="cpu", weights_only=False)
        species_map = {"u": "y_u", "ca": "y_ca", "c": "y_c"}
        species_type = config.get("SPECIES_TYPE", "u")
        if species_type not in species_map:
            raise ValueError(f"Invalid SPECIES_TYPE: {species_type}. Must be one of {list(species_map.keys())}")
        output_key = species_map[species_type]

        required_keys = ["x", output_key]
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            raise KeyError(f"Missing required keys in normalized data: {missing_keys}")

        combined_input = bundle["x"].float()
        out_data = bundle[output_key].float()
        y_initial = bundle["y_initial"].float() if "y_initial" in bundle else None
        if verbose:
            print(f"   Loaded normalized input: {tuple(combined_input.shape)}")
            print(f"   Loaded normalized output: {tuple(out_data.shape)}")
            print(f"   Species type: {species_type}, Output key: {output_key}")
            if y_initial is not None:
                print(f"   Loaded initial values for delta mode: {tuple(y_initial.shape)}")
    except Exception as e:
        raise RuntimeError(f"Failed at Step 1 (loading normalized data): {e}") from e

    try:
        if verbose:
            print("Step 2: Loading channel-wise normalizer from pickle...")
        if config.get("CHANNEL_NORMALIZER_PATH"):
            pickle_path = Path(config["CHANNEL_NORMALIZER_PATH"])
        else:
            pickle_path = Path(config["MERGED_PT_PATH"]).parent / f"normalizer_{species_type}_log.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Channel normalizer pickle not found: {pickle_path}\n"
                "Please ensure CHANNEL_NORMALIZER_PATH points to the correct file."
            )

        with open(pickle_path, "rb") as f:
            channel_normalizer = pickle.load(f)
        channel_normalizer = channel_normalizer.to(device)
        if verbose:
            print(f"   Channel normalizer loaded from: {pickle_path}")
            print(f"   Output mode: {channel_normalizer.output_mode}")
    except Exception as e:
        raise RuntimeError(f"Failed at Step 2 (loading normalizer): {e}") from e

    try:
        if verbose:
            print("Step 3: Creating train/validation/test datasets...")
        model_config = config.get("MODEL_CONFIG", {})
        full_dataset = CustomDatasetPure(
            combined_input,
            out_data,
            y_initial,
            input_time_index=int(model_config.get("input_time_index", 0)),
            validate_static_input=bool(model_config.get("validate_static_input", True)),
            static_input_atol=float(model_config.get("static_input_atol", 1e-6)),
            static_input_rtol=float(model_config.get("static_input_rtol", 1e-5)),
        )
        trainval_dataset, test_dataset, fixed_split = fixed_test_split(
            full_dataset,
            test_size=config["TEST_SIZE"],
            random_state=config["RANDOM_STATE"],
        )
        val_size_relative = config["VAL_SIZE"] / (1 - config["TEST_SIZE"])
        train_dataset, val_dataset, _ = make_member_split(
            trainval_dataset=trainval_dataset,
            trainval_original_indices=fixed_split.trainval_indices,
            val_size_relative=val_size_relative,
            random_state=config["RANDOM_STATE"],
        )
        if verbose:
            print(f"   Train dataset size: {len(train_dataset)}")
            print(f"   Validation dataset size: {len(val_dataset)}")
            print(f"   Test dataset size: {len(test_dataset)}")
    except Exception as e:
        raise RuntimeError(f"Failed at Step 3 (dataset creation): {e}") from e

    if verbose:
        print("Data preprocessing completed successfully!")

    if return_split:
        return (
            channel_normalizer,
            trainval_dataset,
            test_dataset,
            fixed_split,
            train_dataset,
            val_dataset,
            device,
        )
    return channel_normalizer, train_dataset, val_dataset, test_dataset, device


class DoubleConv2D(nn.Module):
    """Double 2D convolution block."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet2DTimeOutput(nn.Module):
    """2D U-Net that predicts all output timesteps as final head channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_time_steps: int = 20,
        init_features: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.0,
        pool_kernels: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if output_time_steps < 1:
            raise ValueError(f"output_time_steps must be >= 1, got {output_time_steps}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")
        if pool_kernels is None:
            pool_kernels = [(2, 2), (2, 2), (2, 2), (2, 2)]
        if depth > len(pool_kernels):
            raise ValueError(f"depth={depth} exceeds configured pool_kernels length={len(pool_kernels)}")

        self.depth = int(depth)
        self.out_channels = int(out_channels)
        self.output_time_steps = int(output_time_steps)
        self.pool_kernels = [tuple(kernel) for kernel in pool_kernels[:depth]]
        self.last_skip_shape_pairs = []

        encoder_channels = [init_features * (2 ** i) for i in range(depth)]

        self.encoder_blocks = nn.ModuleList()
        prev_channels = in_channels
        for features in encoder_channels:
            self.encoder_blocks.append(DoubleConv2D(prev_channels, features, dropout_rate))
            prev_channels = features

        self.pools = nn.ModuleList(
            nn.MaxPool2d(kernel_size=kernel, stride=kernel)
            for kernel in self.pool_kernels
        )

        bottleneck_channels = prev_channels * 2
        self.bottleneck = DoubleConv2D(prev_channels, bottleneck_channels, dropout_rate)

        self.up_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels, pool_kernel in zip(reversed(encoder_channels), reversed(self.pool_kernels)):
            self.up_blocks.append(
                nn.ConvTranspose2d(
                    current_channels,
                    skip_channels,
                    kernel_size=pool_kernel,
                    stride=pool_kernel,
                )
            )
            self.decoder_blocks.append(DoubleConv2D(skip_channels * 2, skip_channels, dropout_rate))
            current_channels = skip_channels

        self.outc = nn.Conv2d(
            current_channels,
            self.out_channels * self.output_time_steps,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"UNet2DTimeOutput expects input shape (N, C, nx, ny), got {tuple(x.shape)}")

        skip_connections = []
        for encoder, pool in zip(self.encoder_blocks, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        self.last_skip_shape_pairs = []

        for up, decoder in zip(self.up_blocks, self.decoder_blocks):
            skip = skip_connections.pop()
            x = up(x)
            up_shape = tuple(x.shape[2:])
            skip_shape = tuple(skip.shape[2:])
            self.last_skip_shape_pairs.append((up_shape, skip_shape))
            if up_shape != skip_shape:
                raise ValueError(f"U-Net skip shape mismatch: upsampled={up_shape}, skip={skip_shape}")
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        z = self.outc(x)
        batch_size, _, nx, ny = z.shape
        z = z.view(batch_size, self.out_channels, self.output_time_steps, nx, ny)
        return z.permute(0, 1, 3, 4, 2).contiguous()


def create_model(
    config: Dict,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str,
    depth: int,
    init_features: int,
    train_batch_size: int,
    l2_weight: float,
    dropout_rate: float,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    loss_type = config["LOSS_CONFIG"]["loss_type"]
    if loss_type == "l2":
        loss_fn = LpLoss(
            d=config["LOSS_CONFIG"]["l2_d"],
            p=config["LOSS_CONFIG"]["l2_p"],
        )
    elif loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'l2' or 'mse'.")

    model_config = config["MODEL_CONFIG"]
    model = UNet2DTimeOutput(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        output_time_steps=model_config["output_time_steps"],
        init_features=init_features,
        depth=depth,
        dropout_rate=dropout_rate,
        pool_kernels=model_config["pool_kernels"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["SCHEDULER_CONFIG"]["initial_lr"],
        weight_decay=l2_weight,
    )

    scheduler_type = config["SCHEDULER_CONFIG"]["scheduler_type"]
    if scheduler_type == "cosine":
        scheduler = CappedCosineAnnealingWarmRestarts(
            optimizer,
            config["SCHEDULER_CONFIG"]["T_0"],
            config["SCHEDULER_CONFIG"]["T_max"],
            config["SCHEDULER_CONFIG"]["T_mult"],
            config["SCHEDULER_CONFIG"]["eta_min"],
        )
    elif scheduler_type == "step":
        scheduler = LRStepScheduler(
            optimizer,
            config["SCHEDULER_CONFIG"]["step_size"],
            config["SCHEDULER_CONFIG"]["gamma"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'cosine' or 'step'.")

    return model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn


def create_model_from_params(
    config: Dict,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str,
    params: Dict[str, Any],
):
    return create_model(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        depth=int(params["depth"]),
        init_features=int(params["init_features"]),
        train_batch_size=int(params["train_batch_size"]),
        l2_weight=float(params["l2_weight"]),
        dropout_rate=float(params["dropout_rate"]),
    )


def train_model_to_dir(
    config: Dict,
    device: str,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    output_dir: Path,
    verbose: bool = True,
):
    return train_model_generic(
        config=config,
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        output_dir=output_dir,
        verbose=verbose,
    )


def load_ensemble_for_evaluation(
    config: Dict,
    manifest_file: Path,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str,
):
    manifest = load_ensemble_manifest(manifest_file)
    params = manifest["params"]
    models = []
    test_loader = None
    loss_fn = None
    base_dir = Path(manifest_file).parent

    for member in manifest["members"]:
        model, _, _, test_loader, _, _, loss_fn = create_model_from_params(
            config,
            train_dataset,
            val_dataset,
            test_dataset,
            "cpu",
            params,
        )
        state_path = Path(member["model_state_path"])
        if not state_path.is_absolute():
            state_path = base_dir / state_path
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
        model.eval()
        model.to("cpu")
        models.append(model)

    return EnsemblePredictor(models=models, device=device, keep_models_on_device=False), test_loader, loss_fn, manifest


def train_model(
    config: Dict,
    channel_normalizer,
    device: str,
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    loss_fn,
    verbose: bool = True,
):
    output_dir = Path(config["OUTPUT_DIR"]) / "final"
    return train_model_to_dir(
        config=config,
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        output_dir=output_dir,
        verbose=verbose,
    )


def model_evaluation(config: Dict, channel_normalizer, device: str, model, test_loader, loss_fn, verbose: bool = True):
    output_dir = Path(config["OUTPUT_DIR"]) / "final"
    return model_evaluation_generic(
        config=config,
        device=device,
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        output_dir=output_dir,
        compute_mse=isinstance(loss_fn, LpLoss),
        verbose=verbose,
    )


def optuna_optimization(
    config: Dict,
    channel_normalizer,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str,
    verbose: bool = True,
) -> Dict:
    if verbose:
        print("\nStarting Optuna hyperparameter optimization...")
        print(f"Number of trials: {config['TRAINING_CONFIG']['optuna_n_trials']}")

    optuna_output_dir = Path(config["OUTPUT_DIR"]) / "optuna"
    optuna_output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        search_space = config["OPTUNA_SEARCH_SPACE"]
        depth = trial.suggest_int("depth", search_space["depth_range"][0], search_space["depth_range"][1])
        init_features = trial.suggest_int(
            "init_features",
            search_space["init_features_range"][0],
            search_space["init_features_range"][1],
        )
        train_batch_size = trial.suggest_categorical("train_batch_size", search_space["train_batch_size_options"])
        l2_weight = trial.suggest_float(
            "l2_weight",
            search_space["l2_weight_range"][0],
            search_space["l2_weight_range"][1],
            log=True,
        )
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            search_space["dropout_rate_range"][0],
            search_space["dropout_rate_range"][1],
        )

        try:
            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                depth=depth,
                init_features=init_features,
                train_batch_size=train_batch_size,
                l2_weight=l2_weight,
                dropout_rate=dropout_rate,
            )
            trial_output_dir = optuna_output_dir / "trials" / f"trial_{trial.number:03d}"
            trained_model = train_model_to_dir(
                config=config_for_optuna_epochs(config),
                device=device,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                output_dir=trial_output_dir,
                verbose=False,
            )
            trial.set_user_attr("model_state_path", str(trial_output_dir / "best_model_state_dict.pt"))
            trial.set_user_attr("loss_history_path", str(trial_output_dir / "loss_history.pt"))
            loss_history_path = trial_output_dir / "loss_history.pt"
            if loss_history_path.exists():
                loss_history = torch.load(loss_history_path, map_location="cpu", weights_only=False)
                best_val_loss = min(loss_history["val_losses"])
                del loss_history
            else:
                best_val_loss = float("inf")

            del model, trained_model, optimizer, scheduler, loss_fn
            del train_loader, val_loader, test_loader
            if device == "cuda":
                torch.cuda.empty_cache()
            return best_val_loss
        except Exception as e:
            if verbose:
                print(f"Trial {trial.number} failed with error: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=config["TRAINING_CONFIG"]["optuna_n_startup_trials"],
            seed=config["TRAINING_CONFIG"]["optuna_seed"],
        ),
    )

    def progress_callback(study, trial):
        if verbose:
            print(f"Trial {trial.number:3d} completed. Value: {trial.value:.6f}, Best: {study.best_value:.6f}")
        if trial.value is not None and trial.value == study.best_value:
            source_model_path = Path(trial.user_attrs.get("model_state_path", ""))
            source_loss_path = Path(trial.user_attrs.get("loss_history_path", ""))
            best_trial_dir = optuna_output_dir / "best_trial_model"
            best_trial_dir.mkdir(parents=True, exist_ok=True)
            if source_model_path.exists():
                shutil.copy2(source_model_path, best_trial_dir / "best_model_state_dict.pt")
            if source_loss_path.exists():
                shutil.copy2(source_loss_path, best_trial_dir / "loss_history.pt")
            trial_info = {
                "trial_number": trial.number,
                "best_value": trial.value,
                "params": trial.params,
                "datetime": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            with open(best_trial_dir / "trial_info.json", "w") as f:
                json.dump(trial_info, f, indent=2)

    study.optimize(
        objective,
        n_trials=config["TRAINING_CONFIG"]["optuna_n_trials"],
        callbacks=[progress_callback] if verbose else None,
    )

    optimization_results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
        "n_trials": len(study.trials),
        "config": config,
    }
    torch.save(optimization_results, optuna_output_dir / "optimization_results.pt")
    with open(optuna_output_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        trial_numbers = [trial.number for trial in study.trials]
        trial_values = [trial.value if trial.value is not None else float("inf") for trial in study.trials]
        ax.plot(trial_numbers, trial_values, "b-", alpha=0.7, label="Trial Values")
        best_values = []
        current_best = float("inf")
        for value in trial_values:
            if value < current_best:
                current_best = value
            best_values.append(current_best)
        ax.plot(trial_numbers, best_values, "r-", linewidth=2, label="Best Value")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Optuna Optimization History")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.savefig(optuna_output_dir / "optimization_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        if verbose:
            print(f"Could not generate optimization history plot: {e}")

    if verbose:
        print("\nOptuna optimization completed!")
        print(f"Best validation loss: {study.best_value:.6f}")
        print(f"Optimization results saved to: {optuna_output_dir / 'optimization_results.pt'}")

    return optimization_results


def visualization(
    config: Dict,
    channel_normalizer,
    device: str,
    trained_model,
    train_dataset,
    val_dataset,
    test_dataset,
    verbose: bool = True,
):
    batch_size = min(8, len(test_dataset))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return generate_all_outputs(
        config=config,
        channel_normalizer=channel_normalizer,
        device=device,
        trained_model=trained_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        test_loader=test_loader,
        verbose=verbose,
    )


def main() -> None:
    try:
        print("\nU-Net-New 2D Time-Output Training Pipeline Started")
        print(f"Training Mode: {CONFIG['TRAINING_CONFIG']['mode'].upper()}")
        print(f"Species Type: {CONFIG['SPECIES_TYPE'].upper()}")

        (
            channel_normalizer,
            trainval_dataset,
            test_dataset,
            fixed_split,
            train_dataset,
            val_dataset,
            device,
        ) = preprocessing(
            config=CONFIG,
            verbose=True,
            return_split=True,
        )

        training_mode = CONFIG["TRAINING_CONFIG"]["mode"]
        if training_mode in ("single", "optuna"):
            if training_mode == "single":
                print("\nExecuting single/ensemble training mode...")
                optimization_results = None
            else:
                print("\nExecuting Optuna optimization mode...")
                optuna_train_dataset, optuna_val_dataset, _ = make_member_split(
                    trainval_dataset=trainval_dataset,
                    trainval_original_indices=fixed_split.trainval_indices,
                    val_size_relative=CONFIG["VAL_SIZE"] / (1 - CONFIG["TEST_SIZE"]),
                    random_state=member_seed(CONFIG, 0),
                )
                optimization_results = optuna_optimization(
                    config=CONFIG,
                    channel_normalizer=channel_normalizer,
                    train_dataset=optuna_train_dataset,
                    val_dataset=optuna_val_dataset,
                    test_dataset=test_dataset,
                    device=device,
                    verbose=True,
                )

            params = resolve_training_params(training_mode, CONFIG, optimization_results)
            print("\nTraining final U-Net-New ensemble with resolved hyperparameters...")

            def create_components(train_ds, val_ds, test_ds, member_params, output_dir):
                return create_model_from_params(CONFIG, train_ds, val_ds, test_ds, device, member_params)

            def train_one(model, train_loader, val_loader, optimizer, scheduler, loss_fn, output_dir):
                return train_model_to_dir(
                    CONFIG,
                    device,
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    output_dir,
                    verbose=True,
                )

            ensemble_run = train_ensemble(
                config=CONFIG,
                model_kind="unet_new",
                trainval_dataset=trainval_dataset,
                trainval_original_indices=fixed_split.trainval_indices,
                test_dataset=test_dataset,
                test_original_indices=fixed_split.test_indices,
                device=device,
                params=params,
                create_components=create_components,
                train_one=train_one,
                verbose=True,
            )
            trained_model = ensemble_run.predictor
            test_loader = ensemble_run.test_loader
            loss_fn = ensemble_run.loss_fn
            train_dataset = ensemble_run.train_dataset
            val_dataset = ensemble_run.val_dataset

            model_evaluation(
                config=CONFIG,
                channel_normalizer=channel_normalizer,
                device=device,
                model=trained_model,
                test_loader=test_loader,
                loss_fn=loss_fn,
                verbose=True,
            )
            print(f"   Ensemble manifest saved to: {ensemble_run.manifest_path}")

        elif training_mode == "eval":
            print("\nExecuting evaluation mode...")
            eval_model_path = Path(CONFIG["TRAINING_CONFIG"]["eval_model_path"])
            if not eval_model_path.exists():
                raise FileNotFoundError(f"Model file not found: {eval_model_path}")

            if eval_model_path.suffix.lower() == ".json":
                trained_model, test_loader, loss_fn, manifest = load_ensemble_for_evaluation(
                    CONFIG,
                    eval_model_path,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    device,
                )
                print(f"   Loaded ensemble manifest from: {eval_model_path}")
                print(f"   Ensemble members: {len(manifest['members'])}")
            else:
                model, _, _, test_loader, _, _, loss_fn = create_model_from_params(
                    CONFIG,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    device,
                    CONFIG["SINGLE_PARAMS"],
                )
                model.load_state_dict(torch.load(eval_model_path, map_location=device, weights_only=False))
                model.eval()
                trained_model = model
                print(f"   Loaded model from: {eval_model_path}")

            model_evaluation(
                config=CONFIG,
                channel_normalizer=channel_normalizer,
                device=device,
                model=trained_model,
                test_loader=test_loader,
                loss_fn=loss_fn,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown training mode: {training_mode}. Use 'single', 'optuna', or 'eval'.")

        visualization(
            config=CONFIG,
            channel_normalizer=channel_normalizer,
            device=device,
            trained_model=trained_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            verbose=True,
        )

        print("\nTraining pipeline completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
