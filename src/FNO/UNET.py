"""
Pure U-Net with Uniform Distribution Meta Training - Refactored Version

This module implements training pipeline for Pure 3D U-Net models with meta data incorporated
as uniform spatial channels. Meta data (e.g., permeability, porosity) is expanded to
uniform distribution across spatial dimensions and directly concatenated with input channels,
providing a straightforward approach to conditional deep learning models.

This is designed for performance comparison with FNO_pure.py, maintaining identical
data preprocessing, training, and evaluation pipelines while only changing the model architecture.

Refactored for improved readability and maintainability following CLAUDE.md guidelines:
- Keep code simple and readable
- Use functions for repetitive tasks
- Add comments to outline workflow in main()
"""

import sys
sys.path.append('./')

import math
import shutil
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.utils.data import DataLoader, Dataset

from neuraloperator.neuralop.training import AdamW
from util_training import config_for_optuna_epochs, train_model_generic, model_evaluation_generic
from util_ensemble import (
    EnsemblePredictor,
    fixed_test_split,
    load_ensemble_manifest,
    make_member_split,
    member_seed,
    resolve_training_params,
    train_ensemble,
)

# Import preprocessing normalizer (needed for loading ChannelNormalizer from pickle)
preprocessing_path = Path(__file__).parent.parent / 'preprocessing'
if str(preprocessing_path) not in sys.path:
    sys.path.insert(0, str(preprocessing_path))
from preprocessing_normalize import ChannelNormalizer

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    # Data paths - updated to use pre-normalized data (matches FNO.py)
    'SPECIES_TYPE': 'u',  # Options: 'u' (uranium), 'ca' (calcium), 'c' (carbonate)
    'MERGED_PT_PATH': './src/preprocessing/data/normalized/lr/delta/merged_normalized_U.pt',  # Pre-normalized data
    'CHANNEL_NORMALIZER_PATH': './src/preprocessing/normalizers/lr/delta/normalizer_u_delta.pkl',  # Normalizer (must match species and output mode)
    'OUTPUT_DIR': './src/FNO/output_unet',
    'N_EPOCHS': 150,
    'EVAL_INTERVAL': 1,
    'VAL_SIZE': 0.1,  # Validation set size
    'TEST_SIZE': 0.1,  # Test set size
    'RANDOM_STATE': 42,
    'MODEL_CONFIG': {
        'in_channels': 11,  # 10 original channels (with material one-hot) + 1 uniform meta channel (matches FNO)
        'out_channels': 1,
        'init_features': 32,  # Initial number of features for U-Net
        'pool_kernels': [(2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 1)],
    },
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'step',  # Options: 'cosine', 'step'
        'early_stopping': 40,
        'T_0': 10,
        'T_max': 40,
        'T_mult': 2,
        'eta_min': 1e-5,
        'step_size': 10,
        'gamma': 0.5,
        'initial_lr': 1e-2,
    },
    'VISUALIZATION': {
        'SAMPLE_NUM': [1, 5, 15, 20, 25],  # Can be single int or list: e.g., [121, 122, 123]
        'TIME_INDICES': (4, 9, 14, 19),
        'DPI': 200,
        'SAVEASCSV': True  # Save visualization data as CSV format
    },
    'LOSS_CONFIG': {
        'loss_type': 'l2',  # Options: 'l2', 'mse'
        'l2_d': 3,  # Dimension for L2 loss
        'l2_p': 2   # Power for L2 loss
    },
    'TRAINING_CONFIG': {
        'mode': 'single',  # Options: 'single', 'optuna', 'eval'
        'optuna_n_trials': 100,
        'optuna_n_epochs': 30,
        'optuna_seed': 42,
        'optuna_n_startup_trials': 10,
        'eval_model_path': './src/FNO/output_unet/final/best_model_state_dict.pt'
    },
    'ENSEMBLE': {
        'ENABLED': True,
        'N_MODELS': 10,
        'BASE_SEED': 42,
        'SPLIT_SEED_STRATEGY': 'base_plus_member',
        'MEMBER_OUTPUT_PATTERN': 'ensemble/member_{member_id:03d}',
        'MANIFEST_NAME': 'ensemble_manifest.json',
    },
    'OPTUNA_SEARCH_SPACE': {
        'depth_range': [2, 4],  # U-Net depth (number of down/up sampling stages)
        'init_features_range': [16, 64],  # Initial feature channels
        'train_batch_size_options': [16, 32],
        'l2_weight_range': [1e-9, 1e-4],  # [min, max] for log uniform
        'dropout_rate_range': [0.0, 0.3],  # Dropout rate
    },
    'SINGLE_PARAMS': {
        "depth": 4,
        "init_features": 64,
        "train_batch_size": 32,
        "l2_weight": 1e-5,
        "dropout_rate": 0.1,
    }
}

# ==============================================================================
# Data Classes and Dataset
# ==============================================================================

class CustomDatasetPure(Dataset):
    """Custom dataset for Pure U-Net training with meta data already combined as uniform spatial channels.

    Note: This version expects input_tensor to already have meta channels combined,
    unlike the original version that combines them internally.

    Args:
        input_tensor: Combined input tensor of shape (N, original_channels + meta_channels, nx, ny, nt)
        output_tensor: Output tensor of shape (N, 1, nx, ny, nt)
    """

    def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': self.input_tensor[idx],
            'y': self.output_tensor[idx]
        }

# ==============================================================================
# Data Processing Functions
# ==============================================================================

def preprocessing(config: Dict, verbose: bool = True, return_split: bool = False) -> Tuple:
    """
    Load pre-normalized data and perform train/val/test split.

    Processing Steps:
    1. Load normalized tensors (x, y_u/y_ca/y_c already combined and normalized)
    2. Load channel-wise normalizer from pickle file
    3. Perform train/val/test split and create datasets
    4. Return necessary objects for training

    Args:
        config: Configuration dictionary containing paths and parameters
        verbose: Whether to print progress information

    Returns:
        Tuple containing (channel_normalizer, train_dataset, val_dataset, test_dataset, device)
    """

    if verbose:
        print(f"\nLoading pre-normalized data and creating datasets...")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using device: {device}")

    try:
        # Step 1: Load normalized tensors
        if verbose:
            print("Step 1: Loading normalized tensors...")

        if not Path(config['MERGED_PT_PATH']).exists():
            raise FileNotFoundError(f"Normalized file not found: {config['MERGED_PT_PATH']}")

        bundle = torch.load(config['MERGED_PT_PATH'], map_location="cpu", weights_only=False)

        # Determine output key based on species type
        species_map = {'u': 'y_u', 'ca': 'y_ca', 'c': 'y_c'}
        species_type = config.get('SPECIES_TYPE', 'u')

        if species_type not in species_map:
            raise ValueError(f"Invalid SPECIES_TYPE: {species_type}. Must be one of {list(species_map.keys())}")

        output_key = species_map[species_type]

        required_keys = ["x", output_key]
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            raise KeyError(f"Missing required keys in normalized data: {missing_keys}")

        # Data is already normalized and combined (includes meta channel)
        combined_input = bundle["x"].float()   # (N, 11, nx, ny, nt) - already normalized
        out_data = bundle[output_key].float()  # (N, 1, nx, ny, nt) - already normalized

        if verbose:
            print(f"   Loaded normalized tensors - Input: {tuple(combined_input.shape)}, Output: {tuple(out_data.shape)}")
            print(f"   Species type: {species_type}, Output key: {output_key}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 1 (loading normalized data): {e}")

    try:
        # Step 2: Load channel-wise normalizer from pickle
        if verbose:
            print("Step 2: Loading channel-wise normalizer from pickle...")

        # Get pickle path from config
        if 'CHANNEL_NORMALIZER_PATH' in config and config['CHANNEL_NORMALIZER_PATH']:
            pickle_path = Path(config['CHANNEL_NORMALIZER_PATH'])
        else:
            # Fallback: same directory as normalized data
            data_path = Path(config['MERGED_PT_PATH'])
            pickle_path = data_path.parent / f'normalizer_{species_type}_log.pkl'

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Channel normalizer pickle not found: {pickle_path}\n"
                f"Please ensure CHANNEL_NORMALIZER_PATH in CONFIG points to the correct file.\n"
                f"Expected file corresponds to the species type and output mode used during preprocessing."
            )

        with open(pickle_path, 'rb') as f:
            channel_normalizer = pickle.load(f)

        # Move to device
        channel_normalizer = channel_normalizer.to(device)

        if verbose:
            print(f"   Channel normalizer loaded from: {pickle_path}")
            print(f"   Output mode: {channel_normalizer.output_mode}")
            print(f"   Moved to device: {device}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 2 (loading normalizer): {e}")

    try:
        # Step 3: Perform train/val/test split and create datasets
        if verbose:
            print("Step 3: Creating train/val/test datasets...")

        full_dataset = CustomDatasetPure(combined_input, out_data)
        trainval_dataset, test_dataset, fixed_split = fixed_test_split(
            full_dataset,
            test_size=config['TEST_SIZE'],
            random_state=config['RANDOM_STATE'],
        )
        val_size_relative = config['VAL_SIZE'] / (1 - config['TEST_SIZE'])
        train_dataset, val_dataset, _ = make_member_split(
            trainval_dataset=trainval_dataset,
            trainval_original_indices=fixed_split.trainval_indices,
            val_size_relative=val_size_relative,
            random_state=config['RANDOM_STATE'],
        )

        if verbose:
            print(f"   Train dataset size: {len(train_dataset)}")
            print(f"   Validation dataset size: {len(val_dataset)}")
            print(f"   Test dataset size: {len(test_dataset)}")
            total_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
            print(f"   Split ratios: Train {len(train_dataset)/total_size:.1%}, Val {len(val_dataset)/total_size:.1%}, Test {len(test_dataset)/total_size:.1%}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 3 (dataset creation): {e}")

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

    return (channel_normalizer, train_dataset, val_dataset, test_dataset, device)

# ==============================================================================
# Loss Function Options
# ==============================================================================

class LpLoss(nn.Module):
    """Lp Loss function for neural operators.

    Computes the relative Lp norm between prediction and ground truth:
    ||pred - y||_p / ||y||_p

    Args:
        d: Spatial dimensions to compute norm over (e.g., 2 for 2D, 3 for 3D)
        p: Power for Lp norm (e.g., 2 for L2 norm)
        reduction: Reduction method ('mean' or 'sum')
    """

    def __init__(self, d=2, p=2, reduction='mean'):
        super().__init__()
        self.d = d
        self.p = p
        self.reduction = reduction

    def forward(self, pred, y):
        # Get spatial dimensions (skip batch and channel dimensions)
        if len(pred.shape) == 5:  # (N, C, nx, ny, nt)
            dims = [2, 3, 4]  # spatial and temporal dimensions
        elif len(pred.shape) == 4:  # (N, C, nx, ny)
            dims = [2, 3]  # spatial dimensions
        else:
            dims = list(range(2, len(pred.shape)))

        # Compute relative Lp norm: ||pred - y||_p / ||y||_p
        diff_norm = torch.norm(pred - y, p=self.p, dim=dims, keepdim=False)
        y_norm = torch.norm(y, p=self.p, dim=dims, keepdim=False)
        relative_error = diff_norm / (y_norm + 1e-12)  # Add small epsilon to avoid division by zero

        if self.reduction == 'mean':
            return relative_error.mean()
        elif self.reduction == 'sum':
            return relative_error.sum()
        else:
            return relative_error


# ==============================================================================
# Scheduler Options
# ==============================================================================

class LRStepScheduler(torch.optim.lr_scheduler.StepLR):
    """Learning rate step scheduler wrapper."""

    def __init__(self, optimizer: torch.optim.Optimizer, step_size: int,
                 gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, step_size, gamma, last_epoch)

class CappedCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing warm restarts scheduler with maximum period cap.

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart
        T_max: Maximum period length
        T_mult: Factor to increase period after restart
        eta_min: Minimum learning rate
        last_epoch: Index of last epoch
    """

    def __init__(self, optimizer: torch.optim.Optimizer, T_0: int, T_max: int,
                 T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1):
        self.T_0 = T_0
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.last_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        epoch_in_cycle = (self.last_epoch - self.last_restart) % self.T_i
        cycle_num = self.last_epoch // self.T_i + 1
        progress = epoch_in_cycle / self.T_i

        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + ((base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2) / cycle_num
            lrs.append(lr)

        # Check for restart
        if (self.last_epoch - self.last_restart) == self.T_i:
            self.last_restart = self.last_epoch
            self.T_i = min(self.T_i * self.T_mult, self.T_max)

        return lrs


# ==============================================================================
# U-Net Architecture
# ==============================================================================

class DoubleConv3D(nn.Module):
    """Double 3D Convolution block: (Conv3D -> BN -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling block: configurable MaxPool3D -> DoubleConv3D"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernel: Tuple[int, int, int],
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel),
            DoubleConv3D(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling block: ConvTranspose3D -> Concat -> DoubleConv3D"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        pool_kernel: Tuple[int, int, int],
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            skip_channels,
            kernel_size=pool_kernel,
            stride=pool_kernel,
        )
        self.conv = DoubleConv3D(skip_channels * 2, skip_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.shape[2:] != x2.shape[2:]:
            raise ValueError(
                f"U-Net skip shape mismatch: upsampled={tuple(x1.shape[2:])}, "
                f"skip={tuple(x2.shape[2:])}"
            )

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for spatiotemporal prediction.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_features: Number of features in the first layer
        depth: Depth of the U-Net (number of down/up sampling stages)
        dropout_rate: Dropout rate for regularization
        pool_kernels: Downsampling schedule. Each tuple maps to one encoder level.
    """

    def __init__(self, in_channels: int, out_channels: int, init_features: int = 32,
                 depth: int = 3, dropout_rate: float = 0.0,
                 pool_kernels: Optional[List[Tuple[int, int, int]]] = None):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        if pool_kernels is None:
            pool_kernels = [(2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 1)]
        if depth > len(pool_kernels):
            raise ValueError(
                f"depth={depth} exceeds configured pool_kernels length={len(pool_kernels)}"
            )

        self.depth = depth
        self.pool_kernels = [tuple(kernel) for kernel in pool_kernels[:depth]]
        self.last_skip_shape_pairs = []

        encoder_channels = [init_features * (2 ** i) for i in range(depth)]

        self.encoder_blocks = nn.ModuleList()
        prev_channels = in_channels
        for features in encoder_channels:
            self.encoder_blocks.append(DoubleConv3D(prev_channels, features, dropout_rate))
            prev_channels = features

        self.pools = nn.ModuleList(
            nn.MaxPool3d(kernel_size=kernel, stride=kernel)
            for kernel in self.pool_kernels
        )

        bottleneck_channels = prev_channels * 2
        self.bottleneck = DoubleConv3D(prev_channels, bottleneck_channels, dropout_rate)

        self.up_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels, pool_kernel in zip(reversed(encoder_channels), reversed(self.pool_kernels)):
            self.up_blocks.append(
                nn.ConvTranspose3d(
                    current_channels,
                    skip_channels,
                    kernel_size=pool_kernel,
                    stride=pool_kernel,
                )
            )
            self.decoder_blocks.append(DoubleConv3D(skip_channels * 2, skip_channels, dropout_rate))
            current_channels = skip_channels

        self.outc = nn.Conv3d(current_channels, out_channels, kernel_size=1)

    def forward(self, x):
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
                raise ValueError(
                    f"U-Net skip shape mismatch: upsampled={up_shape}, skip={skip_shape}"
                )
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.outc(x)


# ==============================================================================
# Model Building Functions
# ==============================================================================

def create_model(config: Dict, train_dataset, val_dataset, test_dataset, device: str,
                depth: int, init_features: int, train_batch_size: int,
                l2_weight: float, dropout_rate: float):
    """
    Create complete model setup including DataLoaders, loss function, optimizer,
    scheduler, and model architecture.

    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Device to use (cuda/cpu)
        depth: U-Net depth
        init_features: Initial number of features
        train_batch_size: Training batch size
        l2_weight: L2 weight regularization
        dropout_rate: Dropout rate for regularization

    Returns:
        Tuple containing (model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn)
    """

    # 1. Create DataLoaders with pin_memory for efficient CPU-to-GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 2. Create loss function based on config
    loss_type = config['LOSS_CONFIG']['loss_type']
    if loss_type == 'l2':
        loss_fn = LpLoss(
            d=config['LOSS_CONFIG']['l2_d'],
            p=config['LOSS_CONFIG']['l2_p']
        )
    elif loss_type == 'mse':
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'l2' or 'mse'.")

    # 3. Create UNet3D model
    model = UNet3D(
        in_channels=config['MODEL_CONFIG']['in_channels'],
        out_channels=config['MODEL_CONFIG']['out_channels'],
        init_features=init_features,
        depth=depth,
        dropout_rate=dropout_rate,
        pool_kernels=config['MODEL_CONFIG']['pool_kernels'],
    ).to(device)

    # 4. Create optimizer
    optimizer = AdamW(model.parameters(), lr=config['SCHEDULER_CONFIG']['initial_lr'], weight_decay=l2_weight)

    # 5. Create scheduler based on config
    scheduler_type = config['SCHEDULER_CONFIG']['scheduler_type']
    if scheduler_type == 'cosine':
        scheduler = CappedCosineAnnealingWarmRestarts(
            optimizer,
            config['SCHEDULER_CONFIG']['T_0'],
            config['SCHEDULER_CONFIG']['T_max'],
            config['SCHEDULER_CONFIG']['T_mult'],
            config['SCHEDULER_CONFIG']['eta_min']
        )
    elif scheduler_type == 'step':
        scheduler = LRStepScheduler(
            optimizer,
            config['SCHEDULER_CONFIG']['step_size'],
            config['SCHEDULER_CONFIG']['gamma']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'cosine' or 'step'.")

    return (model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn)


def create_model_from_params(
    config: Dict,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str,
    params: Dict[str, Any],
):
    """Create U-Net model components from canonical or Optuna parameter dictionaries."""
    return create_model(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        depth=int(params['depth']),
        init_features=int(params['init_features']),
        train_batch_size=int(params['train_batch_size']),
        l2_weight=float(params['l2_weight']),
        dropout_rate=float(params['dropout_rate']),
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
    """Train one U-Net model and save artifacts to the provided directory."""
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
    """Load a U-Net ensemble manifest and return predictor, test loader, and loss."""
    manifest = load_ensemble_manifest(manifest_file)
    params = manifest['params']
    models = []
    test_loader = None
    loss_fn = None
    base_dir = Path(manifest_file).parent

    for member in manifest['members']:
        model, _, _, test_loader, _, _, loss_fn = create_model_from_params(
            config, train_dataset, val_dataset, test_dataset, device, params
        )
        state_path = Path(member['model_state_path'])
        if not state_path.is_absolute():
            state_path = base_dir / state_path
        model.load_state_dict(torch.load(state_path, map_location=device, weights_only=False))
        model.eval()
        models.append(model)

    return EnsemblePredictor(models=models, device=device), test_loader, loss_fn, manifest

# ==============================================================================
# Training Functions
# ==============================================================================

def train_model(config: Dict, channel_normalizer, device: str, model, train_loader, val_loader, test_loader,
                optimizer, scheduler, loss_fn, verbose: bool = True):
    """
    Train the U-Net model with early stopping and loss tracking.

    Note: Data is already normalized, so no transformation is applied during training.

    Args:
        config: Configuration dictionary
        channel_normalizer: ChannelNormalizer for inverse transform (not used in training)
        device: Device to use (cuda/cpu)
        model: U-Net model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        verbose: Whether to print training progress

    Returns:
        Trained model
    """
    output_dir = Path(config['OUTPUT_DIR']) / 'final'
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
    """
    Evaluate the trained model on test set and print detailed results.

    Note: Data is already normalized, so no transformation is applied during evaluation.

    Args:
        config: Configuration dictionary
        channel_normalizer: ChannelNormalizer for inverse transform (not used in evaluation)
        device: Device to use (cuda/cpu)
        model: Trained model to evaluate
        test_loader: Test data loader
        loss_fn: Loss function
        verbose: Whether to print evaluation results

    Returns:
        Dictionary containing evaluation results
    """
    output_dir = Path(config['OUTPUT_DIR']) / 'final'
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

# ==============================================================================
# Optuna Optimization Functions
# ==============================================================================

def optuna_optimization(config: Dict, channel_normalizer, train_dataset, val_dataset, test_dataset, device: str,
                        verbose: bool = True) -> Dict:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        config: Configuration dictionary
        channel_normalizer: ChannelNormalizer for inverse transform
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Device to use (cuda/cpu)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing best parameters and optimization results
    """

    if verbose:
        print(f"\nStarting Optuna hyperparameter optimization...")
        print(f"Number of trials: {config['TRAINING_CONFIG']['optuna_n_trials']}")

    # Create output directory for optuna results
    optuna_output_dir = Path(config['OUTPUT_DIR']) / 'optuna'
    optuna_output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        """Objective function for Optuna optimization."""

        # Sample hyperparameters from search space
        search_space = config['OPTUNA_SEARCH_SPACE']

        # Sample U-Net specific parameters
        depth = trial.suggest_int('depth',
                                 search_space['depth_range'][0],
                                 search_space['depth_range'][1])
        init_features = trial.suggest_int('init_features',
                                         search_space['init_features_range'][0],
                                         search_space['init_features_range'][1])

        train_batch_size = trial.suggest_categorical('train_batch_size', search_space['train_batch_size_options'])

        # Sample continuous parameters with log uniform distribution
        l2_weight = trial.suggest_float('l2_weight',
                                       search_space['l2_weight_range'][0],
                                       search_space['l2_weight_range'][1],
                                       log=True)

        dropout_rate = trial.suggest_float('dropout_rate',
                                          search_space['dropout_rate_range'][0],
                                          search_space['dropout_rate_range'][1])

        try:
            # Create model with sampled parameters
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
                dropout_rate=dropout_rate
            )

            # Train model and get best validation loss
            trial_output_dir = optuna_output_dir / 'trials' / f'trial_{trial.number:03d}'
            trial_config = config_for_optuna_epochs(config)
            trained_model = train_model_to_dir(
                config=trial_config,
                device=device,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                output_dir=trial_output_dir,
                verbose=False  # Reduce verbosity during optimization
            )
            trial.set_user_attr('model_state_path', str(trial_output_dir / 'best_model_state_dict.pt'))
            trial.set_user_attr('loss_history_path', str(trial_output_dir / 'loss_history.pt'))

            # Get best validation loss from training history
            loss_history_path = trial_output_dir / 'loss_history.pt'
            if loss_history_path.exists():
                loss_history = torch.load(loss_history_path, map_location='cpu', weights_only=False)
                best_val_loss = min(loss_history['val_losses'])
            else:
                # Fallback: return a high loss value if history not found
                best_val_loss = float('inf')

            return best_val_loss

        except Exception as e:
            if verbose:
                print(f"Trial {trial.number} failed with error: {e}")
            # Return high loss for failed trials
            return float('inf')

    # Create optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=config['TRAINING_CONFIG']['optuna_n_startup_trials'],
            seed=config['TRAINING_CONFIG']['optuna_seed']
        )
    )

    # Define callback for progress reporting and best model saving
    def progress_callback(study, trial):
        if verbose:
            current_best = study.best_value
            print(f"Trial {trial.number:3d} completed. "
                  f"Value: {trial.value:.6f}, Best: {current_best:.6f}")

        # Check if this trial is the new best
        if trial.value is not None and trial.value == study.best_value:
            print(f"New best trial found: Trial {trial.number} with loss {trial.value:.6f}")

            # Define paths
            source_model_path = Path(trial.user_attrs.get('model_state_path', ''))
            source_loss_path = Path(trial.user_attrs.get('loss_history_path', ''))

            best_trial_dir = Path(config['OUTPUT_DIR']) / 'optuna' / 'best_trial_model'
            best_trial_dir.mkdir(parents=True, exist_ok=True)

            # Copy model file
            if source_model_path.exists():
                dest_model_path = best_trial_dir / 'best_model_state_dict.pt'
                shutil.copy2(source_model_path, dest_model_path)
                print(f"   Best model saved to: {dest_model_path}")
            else:
                print(f"   Warning: Source model not found at {source_model_path}")

            # Copy loss history
            if source_loss_path.exists():
                dest_loss_path = best_trial_dir / 'loss_history.pt'
                shutil.copy2(source_loss_path, dest_loss_path)

            # Save trial information as JSON
            trial_info = {
                'trial_number': trial.number,
                'best_value': trial.value,
                'params': trial.params,
                'datetime': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            trial_info_path = best_trial_dir / 'trial_info.json'
            with open(trial_info_path, 'w') as f:
                json.dump(trial_info, f, indent=2)

    # Run optimization with progress callback
    study.optimize(
        objective,
        n_trials=config['TRAINING_CONFIG']['optuna_n_trials'],
        callbacks=[progress_callback] if verbose else None
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    if verbose:
        print(f"\nOptuna optimization completed!")
        print(f"Best validation loss: {best_value:.6f}")
        print(f"Best parameters:")
        for param_name, param_value in best_params.items():
            print(f"  {param_name}: {param_value}")

        # Check if best model was saved during optimization
        best_model_path = optuna_output_dir / 'best_trial_model' / 'best_model_state_dict.pt'
        if best_model_path.exists():
            print(f"\nBest model from optimization saved at:")
            print(f"   {best_model_path}")
            trial_info_path = optuna_output_dir / 'best_trial_model' / 'trial_info.json'
            if trial_info_path.exists():
                with open(trial_info_path, 'r') as f:
                    trial_info = json.load(f)
                print(f"   Trial number: {trial_info['trial_number']}")
                print(f"   Validation loss: {trial_info['best_value']:.6f}")
        else:
            print(f"\nWarning: Best model was not saved during optimization")

    # Save optimization results
    optimization_results = {
        'best_params': best_params,
        'best_value': best_value,
        'study': study,
        'n_trials': len(study.trials),
        'config': config
    }

    # Save results to file
    results_path = optuna_output_dir / 'optimization_results.pt'
    torch.save(optimization_results, results_path)

    # Save study as pickle for later analysis
    study_path = optuna_output_dir / 'optuna_study.pkl'
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    # Generate optimization visualizations
    try:
        # 1. Optimization history plot
        fig, ax = plt.subplots(figsize=(10, 6))
        trial_numbers = [trial.number for trial in study.trials]
        trial_values = [trial.value if trial.value is not None else float('inf') for trial in study.trials]

        ax.plot(trial_numbers, trial_values, 'b-', alpha=0.7, label='Trial Values')

        # Add best value line
        best_values = []
        current_best = float('inf')
        for value in trial_values:
            if value < current_best:
                current_best = value
            best_values.append(current_best)

        ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Optuna Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        history_plot_path = optuna_output_dir / 'optimization_history.png'
        plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2. Parameter importance plot (if available)
        if len(study.trials) > 1:
            try:
                import optuna.visualization.matplotlib as optuna_vis

                fig, ax = plt.subplots(figsize=(10, 6))
                optuna_vis.plot_param_importances(study, ax=ax)
                importance_plot_path = optuna_output_dir / 'parameter_importance.png'
                plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                if verbose:
                    print(f"Parameter importance plot saved to: {importance_plot_path}")
            except Exception as e:
                if verbose:
                    print(f"Could not generate parameter importance plot: {e}")

        if verbose:
            print(f"Optimization history plot saved to: {history_plot_path}")

    except Exception as e:
        if verbose:
            print(f"Could not generate optimization plots: {e}")

    if verbose:
        print(f"Optimization results saved to: {results_path}")
        print(f"Study object saved to: {study_path}")

    return optimization_results

# ==============================================================================
# Visualization Functions
# ==============================================================================

def visualize_single_sample(config: Dict, device: str, trained_model,
                           pred_phys: torch.Tensor, gt_phys: torch.Tensor,
                           input_phys: torch.Tensor, sample_idx: int,
                           verbose: bool = True) -> Dict:
    """
    Generate visualization for a single sample.

    Note: pred_phys and gt_phys are already in physical scale (inverse transformed).

    Args:
        config: Configuration dictionary.
        device: Device to use (e.g., 'cuda' or 'cpu').
        trained_model: The trained U-Net model.
        pred_phys: Physical scale predictions tensor (already inverse transformed).
        gt_phys: Ground truth tensor (already inverse transformed).
        input_phys: Input tensor (normalized).
        sample_idx: Index of the sample to visualize.
        verbose: If True, prints progress information.

    Returns:
        Dict containing CSV data for this sample.
    """

    # Extract the corresponding data slices and move to NumPy
    pred_sample = pred_phys[sample_idx, 0].detach().numpy()  # Shape: (nx, ny, nt) -> (64, 32, nt)
    gt_sample = gt_phys[sample_idx, 0].detach().numpy()      # Shape: (nx, ny, nt) -> (64, 32, nt)

    # Extract permeability (channel 0) and convert from log10 scale
    perm_sample = input_phys[sample_idx, 0, :, :, 0].detach().numpy() # Shape: (nx, ny)
    perm_sample = 10**perm_sample

    # Extract pyrite (channel 3)
    pyr_sample = input_phys[sample_idx, 3, :, :, 0].detach().numpy() # Shape: (nx, ny)

    # --- Plot 1: Permeability and Pyrite ---
    fig_inputs, axes_inputs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Permeability
    ax = axes_inputs[0]
    im_perm = ax.imshow(perm_sample.T, cmap='viridis', norm=colors.LogNorm())
    ax.set_title(f"Permeability (Sample {sample_idx})")
    ax.axis('off')
    fig_inputs.colorbar(im_perm, ax=ax, orientation='horizontal', pad=0.1)

    # Plot Pyrite
    ax = axes_inputs[1]
    im_pyr = ax.imshow(pyr_sample.T, cmap='cividis')
    ax.set_title(f"Pyrite (Sample {sample_idx})")
    ax.axis('off')
    fig_inputs.colorbar(im_pyr, ax=ax, orientation='horizontal', pad=0.1)

    fig_inputs.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the input data figure
    output_path_inputs = Path(config['OUTPUT_DIR']) / f'UNET_input_visualization_sample_{sample_idx}.png'
    output_path_inputs.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_inputs, dpi=config['VISUALIZATION']['DPI'], bbox_inches='tight')
    plt.close(fig_inputs)
    if verbose:
        print(f"Input visualization for sample {sample_idx} saved to: {output_path_inputs}")

    # --- Plot 2: GT, Prediction, and Error Grid ---
    t_indices = config['VISUALIZATION']['TIME_INDICES']

    # Create a 3x4 grid for GT, Prediction, and Error
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))

    # Determine shared color scale for GT and Prediction
    vmin_gt_pred = min(gt_sample[:, :, t_indices].min(), pred_sample[:, :, t_indices].min())
    vmax_gt_pred = max(gt_sample[:, :, t_indices].max(), pred_sample[:, :, t_indices].max())

    # Calculate error and determine its symmetric color scale
    error_sample = gt_sample - pred_sample
    error_max_abs = np.abs(error_sample[:, :, t_indices]).max()

    im_pred, im_err = None, None # Initialize for colorbar
    for i, t_idx in enumerate(t_indices):
        # Plot Ground Truth (Row 1)
        ax_gt = axes[0, i]
        im_gt = ax_gt.imshow(gt_sample[:, :, t_idx].T, cmap='jet', vmin=vmin_gt_pred, vmax=vmax_gt_pred)
        ax_gt.set_title(f"Ground Truth (t={t_idx})")
        ax_gt.axis('off')

        # Plot Prediction (Row 2)
        ax_pred = axes[1, i]
        im_pred = ax_pred.imshow(pred_sample[:, :, t_idx].T, cmap='jet', vmin=vmin_gt_pred, vmax=vmax_gt_pred)
        ax_pred.set_title(f"Prediction (t={t_idx})")
        ax_pred.axis('off')

        # Plot Error (Row 3)
        ax_err = axes[2, i]
        im_err = ax_err.imshow(error_sample[:, :, t_idx].T, cmap='coolwarm', vmin=-error_max_abs, vmax=error_max_abs)
        ax_err.set_title(f"Error (t={t_idx})")
        ax_err.axis('off')

    # Add shared colorbars
    if im_pred:
        fig.colorbar(im_pred, ax=axes[0:2, :].ravel().tolist(), orientation='horizontal', pad=0.05, aspect=40)
    if im_err:
        fig.colorbar(im_err, ax=axes[2, :].ravel().tolist(), orientation='horizontal', pad=0.05, aspect=40)

    # Save the main comparison figure
    output_path_grid = Path(config['OUTPUT_DIR']) / f'UNET_comparison_grid_sample_{sample_idx}.png'
    plt.savefig(output_path_grid, dpi=config['VISUALIZATION']['DPI'], bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Comparison grid for sample {sample_idx} saved to: {output_path_grid}")

    # --- Prepare CSV Data ---
    csv_data = {}
    if config['VISUALIZATION']['SAVEASCSV']:
        # Create coordinate grids
        nx, ny = gt_sample.shape[:2]  # (64, 32)
        x_coords = np.arange(nx)
        y_coords = np.arange(ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Flatten coordinate arrays
        x_flat = X.flatten()
        y_flat = Y.flatten()

        # Initialize CSV data dictionary with coordinates (only for first sample)
        csv_data = {
            'x_coord': x_flat,
            'y_coord': y_flat
        }

        # Process each time index
        for t_idx in t_indices:
            # Extract 2D slices for this time index
            gt_slice = gt_sample[:, :, t_idx]          # (nx, ny)
            pred_slice = pred_sample[:, :, t_idx]      # (nx, ny)
            error_slice = error_sample[:, :, t_idx]    # (nx, ny)

            # Flatten the 2D data to 1D arrays
            gt_flat = gt_slice.flatten()
            pred_flat = pred_slice.flatten()
            error_flat = error_slice.flatten()

            # Add columns with naming pattern: sample_time_type
            csv_data[f'{sample_idx}_{t_idx}_gt'] = gt_flat
            csv_data[f'{sample_idx}_{t_idx}_pred'] = pred_flat
            csv_data[f'{sample_idx}_{t_idx}_error'] = error_flat

    return csv_data


def visualization(config: Dict, channel_normalizer, device: str, trained_model, train_dataset,
                 test_dataset, verbose: bool = True):
    """
    Generate visualizations for multiple samples:
    1. A separate plot for permeability and pyrite maps.
    2. A 3x4 grid comparing ground truth, predictions, and their error over time.
    3. Unified CSV output for all samples (Option B).

    Args:
        config: Configuration dictionary.
        channel_normalizer: ChannelNormalizer for inverse transform.
        device: Device to use (e.g., 'cuda' or 'cpu').
        trained_model: The trained U-Net model.
        train_dataset: The training dataset.
        test_dataset: The test dataset for generating predictions.
        verbose: If True, prints progress information.
    """

    if verbose:
        print(f"\nGenerating multi-sample visualization...")

    # Ensure the model is in evaluation mode
    trained_model.eval()

    # Use smaller batch size to avoid VRAM issues
    batch_size = min(8, len(test_dataset))  # Process in smaller batches
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Store predictions, ground truth, and input data
    all_pred = []
    all_gt = []
    all_input = []

    # Generate predictions without computing gradients
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if verbose and batch_idx % 2 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(test_loader)}...")

            x, y = batch['x'].to(device), batch['y'].to(device)

            # Store original input (already normalized) moved to CPU to free GPU memory
            all_input.append(x.cpu())

            # Get model prediction (data is already normalized)
            pred = trained_model(x)

            # Inverse transform both prediction and ground truth to physical scale
            # Note: We need the original (untransformed) input for inverse transform context
            pred_phys = channel_normalizer.inverse_output_transform(pred, x_raw=x)
            y_phys = channel_normalizer.inverse_output_transform(y, x_raw=x)

            # Move to CPU immediately and clear GPU memory
            all_pred.append(pred_phys.cpu())
            all_gt.append(y_phys.cpu())

            # Clear intermediate GPU tensors
            del x, y, pred, pred_phys, y_phys
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Concatenate results from all batches (this happens on CPU)
    if verbose:
        print("  Concatenating results...")
    pred_phys = torch.cat(all_pred, dim=0)
    gt_phys = torch.cat(all_gt, dim=0)
    input_phys = torch.cat(all_input, dim=0)

    # Clear intermediate lists to free memory
    del all_pred, all_gt, all_input

    # Apply masking as per the original problem description
    pred_phys[:, :, 14:18, 14:18, :] = 0
    gt_phys[:, :, 14:18, 14:18, :] = 0

    # --- Multi-Sample Processing ---
    # Handle both single integer and list inputs for SAMPLE_NUM
    sample_config = config['VISUALIZATION']['SAMPLE_NUM']
    if isinstance(sample_config, int):
        sample_nums = [sample_config]  # Convert single int to list for consistency
    else:
        sample_nums = sample_config  # Assume it's already a list

    # Validate sample indices
    max_available_samples = len(pred_phys) - 1
    valid_sample_nums = []
    for sample_num in sample_nums:
        if sample_num <= max_available_samples:
            valid_sample_nums.append(sample_num)
        else:
            if verbose:
                print(f"Warning: Sample {sample_num} exceeds available samples ({max_available_samples}). Skipping.")

    if not valid_sample_nums:
        if verbose:
            print("Error: No valid sample indices found. Using sample 0 as fallback.")
        valid_sample_nums = [0]

    if verbose:
        print(f"Processing samples: {valid_sample_nums}")

    # --- Process Each Sample ---
    all_csv_data = []
    for i, sample_idx in enumerate(valid_sample_nums):
        if verbose:
            print(f"Processing sample {sample_idx} ({i+1}/{len(valid_sample_nums)})...")

        # Generate visualization for single sample
        csv_data = visualize_single_sample(
            config=config,
            device=device,
            trained_model=trained_model,
            pred_phys=pred_phys,
            gt_phys=gt_phys,
            input_phys=input_phys,
            sample_idx=sample_idx,
            verbose=verbose
        )

        # Store CSV data for unified output
        if csv_data:
            all_csv_data.append(csv_data)

    # --- Unified CSV Export (Option B) ---
    if config['VISUALIZATION']['SAVEASCSV'] and all_csv_data:
        if verbose:
            print(f"Creating unified CSV output for all samples...")

        # Start with coordinates from the first sample
        unified_csv_data = {
            'x_coord': all_csv_data[0]['x_coord'],
            'y_coord': all_csv_data[0]['y_coord']
        }

        # Merge data columns from all samples
        for csv_data in all_csv_data:
            for key, value in csv_data.items():
                if key not in ['x_coord', 'y_coord']:  # Skip coordinate columns
                    unified_csv_data[key] = value

        # Create unified DataFrame and save to CSV
        df_unified = pd.DataFrame(unified_csv_data)
        csv_output_path = Path(config['OUTPUT_DIR']) / 'UNET_visualization_data.csv'
        df_unified.to_csv(csv_output_path, index=False)

        if verbose:
            print(f"Unified CSV data saved to: {csv_output_path}")
            print(f"CSV shape: {df_unified.shape}")
            print(f"CSV columns: {list(df_unified.columns)}")
            print(f"Processed {len(valid_sample_nums)} samples: {valid_sample_nums}")

    if verbose:
        print(f"Multi-sample visualization completed!")

# ==============================================================================
# Utility Functions
# ==============================================================================

def main() -> None:
    """
    Main training pipeline for Pure U-Net with uniform meta channels.

    Workflow:
    1. Load and preprocess data with unified preprocessing function
    2. Configure training mode (single/optuna/eval)
    3. Execute training pipeline
    4. Generate visualization and save results
    """
    try:
        # Step 1: Unified data preprocessing
        print(f"\nU-Net-Pure Training Pipeline Started")
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

        # Step 2: Execute based on training mode
        training_mode = CONFIG['TRAINING_CONFIG']['mode']

        if training_mode in ('single', 'optuna'):
            if training_mode == 'single':
                print("\nExecuting single/ensemble training mode...")
                optimization_results = None
            else:
                print("\nExecuting Optuna optimization mode...")
                optuna_train_dataset, optuna_val_dataset, _ = make_member_split(
                    trainval_dataset=trainval_dataset,
                    trainval_original_indices=fixed_split.trainval_indices,
                    val_size_relative=CONFIG['VAL_SIZE'] / (1 - CONFIG['TEST_SIZE']),
                    random_state=member_seed(CONFIG, 0),
                )
                optimization_results = optuna_optimization(
                    config=CONFIG,
                    channel_normalizer=channel_normalizer,
                    train_dataset=optuna_train_dataset,
                    val_dataset=optuna_val_dataset,
                    test_dataset=test_dataset,
                    device=device,
                    verbose=True
                )

            params = resolve_training_params(training_mode, CONFIG, optimization_results)
            print("\nTraining final U-Net ensemble with resolved hyperparameters...")

            def create_components(train_ds, val_ds, test_ds, member_params, output_dir):
                return create_model_from_params(CONFIG, train_ds, val_ds, test_ds, device, member_params)

            def train_one(model, train_loader, val_loader, optimizer, scheduler, loss_fn, output_dir):
                return train_model_to_dir(
                    CONFIG, device, model, train_loader, val_loader,
                    optimizer, scheduler, loss_fn, output_dir, verbose=True
                )

            ensemble_run = train_ensemble(
                config=CONFIG,
                model_kind='unet',
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
                verbose=True
            )

            print(f"   Ensemble manifest saved to: {ensemble_run.manifest_path}")

        elif training_mode == 'eval':
            # Evaluation mode - load pretrained single model or ensemble manifest
            print("\nExecuting evaluation mode...")

            eval_model_path = Path(CONFIG['TRAINING_CONFIG']['eval_model_path'])
            if not eval_model_path.exists():
                raise FileNotFoundError(f"Model file not found: {eval_model_path}")

            if eval_model_path.suffix.lower() == '.json':
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
                model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model_from_params(
                    CONFIG, train_dataset, val_dataset, test_dataset, device, CONFIG['SINGLE_PARAMS']
                )
                model.load_state_dict(torch.load(eval_model_path, map_location=device, weights_only=False))
                model.eval()
                trained_model = model
                print(f"   Loaded model from: {eval_model_path}")

            # Evaluate the loaded model
            model_evaluation(
                config=CONFIG,
                channel_normalizer=channel_normalizer,
                device=device,
                model=trained_model,
                test_loader=test_loader,
                loss_fn=loss_fn,
                verbose=True
            )

        else:
            raise ValueError(f"Unknown training mode: {training_mode}. Use 'single', 'optuna', or 'eval'.")

        # Step 3: Generate visualization (for all modes)
        visualization(
            config=CONFIG,
            channel_normalizer=channel_normalizer,
            device=device,
            trained_model=trained_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            verbose=True
        )

        print("\nTraining pipeline completed successfully!")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main()
