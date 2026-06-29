"""
FNO for Outlet Prediction - Vector Output Model

This module implements a TFNO-based model for predicting outlet uranium flow time series.
Unlike standard FNO that outputs spatial fields, this model uses global pooling to produce
vector outputs, predicting outlet flow over time from spatial input fields.

Key differences from standard FNO.py:
- Uses TFNOWithPooling: custom model with global pooling instead of spatial projection
- Predicts outlet vectors (B, nt) instead of spatial fields (B, 1, nx, ny, nt)
- Uses MSE loss for vector outputs instead of LpLoss for spatial fields
- Loads and uses separate outlet normalizer for denormalization
"""

import sys
sys.path.append('./')

import shutil
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import optuna

from neuraloperator.neuralop.utils import count_model_params
from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop.training import AdamW
from neuraloperator.neuralop.layers.channel_mlp import ChannelMLP

# Import common utilities (refactored from duplicate code)
from util_common import LRStepScheduler

# Import common training utilities (refactored from duplicate code)
from util_training import config_for_optuna_epochs, train_model_generic, model_evaluation_generic
from util_ensemble import (
    EnsemblePredictor,
    fixed_test_split,
    load_ensemble_manifest,
    make_member_split,
    member_seed,
    resolve_domain_padding,
    resolve_training_params,
    train_ensemble,
)

# Import outlet-specific output utilities (refactored from duplicate code)
from util_output_outlet import visualize_outlet_predictions

# Import preprocessing normalizers
preprocessing_path = Path(__file__).parent.parent / 'preprocessing'
if str(preprocessing_path) not in sys.path:
    sys.path.insert(0, str(preprocessing_path))
from normalizer_core import OutletNormalizer
from preprocessing_normalize import ChannelNormalizer

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'MODEL_KIND': 'fno_outlet',
    # Data paths
    'MERGED_PT_PATH': './src/preprocessing/data/normalized/hr/delta/merged_normalized_out_hr.pt',
    'SPATIAL_NORMALIZER_PATH': './src/preprocessing/normalizers/lr/delta/normalizer_u_delta.pkl',
    'OUTLET_NORMALIZER_PATH': './src/preprocessing/normalizers/lr/delta/normalizer_out_delta.pkl',
    'OUTPUT_DIR': './src/FNO/output_outlet/',
    'SPLIT_INDEX_CSV_PATH': './src/FNO/output_outlet/split_index_mapping.csv',

    # Training parameters
    'N_EPOCHS': 100,
    'VAL_SIZE': 0.02,
    'TEST_SIZE': 0.96,
    'RANDOM_STATE': 42,
    'DOMAIN_PADDING_MODE': 'symmetric',

    # Model configuration (fixed architectural choices)
    'MODEL_CONFIG': {
        'in_channels': 11,  # 10 original channels + 1 meta channel
        'pool_type': 'adaptive_avg',  # Options: 'adaptive_avg', 'adaptive_max'
        'lifting_channel_ratio': 2,
        'projection_channel_ratio': 2,
        'positional_embedding': 'grid',
    },

    # Scheduler configuration
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'step',  # Validation plateau
        'early_stopping': 20,
        'step_size': 10,  # Plateau patience epochs before reducing LR
        'gamma': 0.5,  # LR multiplier after plateau
        'initial_lr': 1e-2,
    },

    # Loss configuration
    'LOSS_CONFIG': {
        'loss_type': 'l2',  # MSE for vector outputs
    },

    # Training mode
    'TRAINING_CONFIG': {
        'mode': 'eval',  # Options: 'single', 'optuna', 'eval'
        'optuna_n_trials': 100,
        'optuna_n_epochs': 30,
        'optuna_seed': 42,
        'optuna_n_startup_trials': 10,
        'eval_model_path': './src/FNO/output_outlet/final/best_model_state_dict.pt'
    },
    'ENSEMBLE': {
        'ENABLED': True,
        'N_MODELS': 50,
        'BASE_SEED': 42,
        'SPLIT_SEED_STRATEGY': 'base_plus_member',
        'MEMBER_OUTPUT_PATTERN': 'ensemble/member_{member_id:03d}',
        'MANIFEST_NAME': 'ensemble_manifest.json',
    },
    'TIMING': {
        'ENABLED': True,
        'PREDICTION_WARMUP_BATCHES': 1,
        'REPORT_DIR_NAME': 'timing',
    },

    # Optuna search space (hyperparameter ranges for optimization)
    'OPTUNA_SEARCH_SPACE': {
        # TFNO Fourier modes (outlet-specific: smaller ranges for vector output)
        'n_modes_dim1_range': [4, 12],
        'n_modes_dim2_range': [2, 8],
        'n_modes_dim3_range': [2, 6],

        # TFNO architecture
        'hidden_channels_range': [12, 48],
        'n_layers_range': [2, 6],

        # Domain and training parameters
        'domain_padding_options': [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.15, 0.1, 0.1)],
        'train_batch_size_options': [16, 32, 64],
        'l2_weight_range': [1e-9, 1e-4],  # Log uniform distribution

        # FNO block channel MLP parameters
        'channel_mlp_expansion_options': [0.5, 1.0, 2.0],
        'channel_mlp_skip_options': ['linear', 'soft-gating'],

        # Projection MLP parameters (outlet-specific feature)
        'projection_mlp_hidden_range': [64, 256],
        'projection_mlp_layers_options': [2, 3, 4],
        'projection_mlp_activation_options': ['gelu', 'relu', 'silu'],
        'projection_mlp_dropout_range': [0.0, 0.3],
    },

    # Single training parameters (hyperparameters that can be tuned)
    'SINGLE_PARAMS': {
        # TFNO architecture
        "n_modes_1": 10,
        "n_modes_2": 14,
        "n_modes_3": 4,
        "hidden_channels": 57,
        "n_layers": 5,
        "domain_padding": (0.1, 0.1, 0.1),

        # Training parameters
        "train_batch_size": 16,
        "l2_weight": 6.942933677610523e-07,

        # FNO block parameters
        "channel_mlp_expansion": 1.0,
        "channel_mlp_skip": 'soft-gating',

        # ChannelMLP projection head parameters
        "projection_mlp_hidden": 115,    # Hidden channels for C → 1 projection
        "projection_mlp_layers": 3,      # Number of Conv1d layers
        "projection_mlp_activation": 'gelu',  # Activation function
        "projection_mlp_dropout": 0.0,   # Dropout probability
    },

    # Visualization configuration
    'VISUALIZATION': {
        'N_SAMPLES': 24,  # Number of samples to visualize
    },

    # Integrated Gradients configuration
    'IG_ANALYSIS': {
        'ENABLED': False, # Perform IG analysis
        'SAMPLE_IDX': [108],  # Sample index or list of sample indices
        'TIME_INDICES': [19],  # Target times to analyze
        'N_STEPS': 50,  # Integration steps
        'N_BASELINES': 200,  # Number of baseline samples (multi-baseline mode only)
        'BASELINE_SEED': 36,  # Random seed for baseline selection (reproducibility)
    }
}

# Channel names for IG output handling
CHANNEL_NAMES_11 = [
    'Permeability', 'Calcite', 'Clinochlore', 'Pyrite', 'Smectite',
    'Material_Source', 'Material_Bentonite', 'Material_Fracture',
    'X-velocity', 'Y-velocity', 'Meta'
]
CHANNEL_SHORT_11 = [
    'Perm', 'Calcite', 'Clino', 'Pyrite', 'Smectite',
    'MatSrc', 'MatBent', 'MatFrac', 'Vx', 'Vy', 'Meta'
]
CHANNEL_NAMES_10_MERGED = [
    'Permeability', 'Calcite', 'Clinochlore', 'Pyrite', 'Smectite',
    'Material_Source', 'Material_Bentonite', 'Material_Fracture',
    'Velocity', 'Meta'
]
CHANNEL_SHORT_10_MERGED = [
    'Perm', 'Calcite', 'Clino', 'Pyrite', 'Smectite',
    'MatSrc', 'MatBent', 'MatFrac', 'Vel', 'Meta'
]


def merge_velocity_channels(ig_spatial: np.ndarray, vx_idx: int = 8, vy_idx: int = 9) -> np.ndarray:
    """
    Merge Vx/Vy IG channels into a single Velocity channel using cell-wise signed sum.

    Args:
        ig_spatial: IG attribution map of shape (11, nx, ny)
        vx_idx: Channel index for X-velocity
        vy_idx: Channel index for Y-velocity

    Returns:
        Merged IG map of shape (10, nx, ny) with channel order:
        Perm..MatFrac, Velocity, Meta
    """
    if ig_spatial.ndim != 3:
        raise ValueError(f"Expected ig_spatial with 3 dimensions (C, nx, ny), got {ig_spatial.shape}")
    if ig_spatial.shape[0] <= max(vx_idx, vy_idx):
        raise ValueError(
            f"Velocity channel indices out of bounds. shape={ig_spatial.shape}, "
            f"vx_idx={vx_idx}, vy_idx={vy_idx}"
        )

    velocity_ig = ig_spatial[vx_idx] + ig_spatial[vy_idx]
    merged_channels = []
    for ch in range(ig_spatial.shape[0]):
        if ch in (vx_idx, vy_idx):
            continue
        merged_channels.append(ig_spatial[ch])
        # Insert merged velocity channel at original Vx position.
        if ch == vx_idx - 1:
            merged_channels.append(velocity_ig)

    return np.stack(merged_channels, axis=0)

# ==============================================================================
# Custom Model: TFNOWithPooling
# ==============================================================================

class TFNOWithPooling(nn.Module):
    """
    TFNO model with spatial pooling and ChannelMLP for outlet prediction.

    Architecture:
        Input (B, C_in, nx, ny, nt)
        → Lifting → FNO blocks → Spatial Pooling (nx,ny → 1,1) → ChannelMLP (C → 1)
        → Output (B, nt)

    Key improvement: Preserves temporal dimension throughout, making the model
    dimension-independent (works with any nt without retraining).

    Args:
        base_tfno_config: Configuration dict for creating base TFNO
        pool_type: Type of spatial pooling ('adaptive_avg' or 'adaptive_max')
        channel_mlp_config: Configuration dict for ChannelMLP (C → 1 projection)
    """

    def __init__(self, base_tfno_config: Dict, pool_type: str = 'adaptive_avg',
                 channel_mlp_config: Dict = None):
        super().__init__()

        self.pool_type = pool_type

        # Default ChannelMLP config if not provided
        if channel_mlp_config is None:
            channel_mlp_config = {
                'hidden_channels': 128,
                'n_layers': 2,
                'activation': 'gelu',
                'dropout': 0.1,
            }

        # Create base TFNO model (we'll reuse its components)
        base_tfno = TFNO(**base_tfno_config)

        # Extract components from base TFNO
        self.positional_embedding = base_tfno.positional_embedding
        self.lifting = base_tfno.lifting
        self.fno_blocks = base_tfno.fno_blocks
        self.domain_padding = base_tfno.domain_padding

        # Store dimensions
        self.hidden_channels = base_tfno.hidden_channels
        self.n_layers = base_tfno.n_layers

        # Spatial pooling only (NOT temporal!)
        if pool_type == 'adaptive_avg':
            self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'adaptive_max':
            self.spatial_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. Use 'adaptive_avg' or 'adaptive_max'")

        # Map activation string to function
        activation_map = {
            'gelu': F.gelu,
            'relu': F.relu,
            'silu': F.silu,
            'tanh': torch.tanh,
        }
        activation_fn = activation_map.get(channel_mlp_config['activation'], F.gelu)

        # ChannelMLP for C → 1 projection (preserves time dimension)
        self.projection_mlp = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=1,
            hidden_channels=channel_mlp_config['hidden_channels'],
            n_layers=channel_mlp_config['n_layers'],
            n_dim=1,  # 1D conv along time dimension
            non_linearity=activation_fn,
            dropout=channel_mlp_config['dropout']
        )

    def forward(self, x, output_shape=None):
        """
        Forward pass producing dimension-independent outlet prediction.

        Args:
            x: Input tensor (B, C_in, nx, ny, nt)
            output_shape: Ignored (kept for compatibility)

        Returns:
            Vector output (B, nt) - dimension independent!

        Pipeline:
            (B, C_in, nx, ny, nt)
            → FNO blocks: (B, C_hidden, nx, ny, nt)
            → Spatial pool: (B, C_hidden, nt)
            → ChannelMLP: (B, 1, nt)
            → Squeeze: (B, nt)
        """
        # FNO pipeline: lifting → padding → blocks → unpadding
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=None)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # x shape: (B, C, nx, ny, nt)
        B, C, nx, ny, nt = x.shape

        # Spatial pooling only (preserve time dimension)
        # Reshape to apply 2D pooling on spatial dims
        x = x.permute(0, 1, 4, 2, 3)  # (B, C, nt, nx, ny)
        x = x.reshape(B * C * nt, nx, ny)  # (B*C*nt, nx, ny)
        x = self.spatial_pool(x)  # (B*C*nt, 1, 1)
        x = x.reshape(B, C, nt)  # (B, C, nt)

        # ChannelMLP projection: C → 1 (time dimension preserved)
        x = self.projection_mlp(x)  # (B, 1, nt)
        x = x.squeeze(1)  # (B, nt)

        return x


# ==============================================================================
# Dataset Class
# ==============================================================================

class CustomDatasetOutlet(Dataset):
    """
    Dataset for outlet prediction task.

    Args:
        input_tensor: Spatial input (N, 11, nx, ny, nt)
        output_outlet: Outlet time series (N, nt)
    """

    def __init__(self, input_tensor: torch.Tensor, output_outlet: torch.Tensor):
        self.input_tensor = input_tensor
        self.output_outlet = output_outlet

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': self.input_tensor[idx],      # (11, nx, ny, nt)
            'y': self.output_outlet[idx]      # (nt,)
        }


# ==============================================================================
# Data Processing Functions
# ==============================================================================

def save_split_index_mapping(train_idx: np.ndarray, val_idx: np.ndarray,
                             test_idx: np.ndarray, output_path: str) -> None:
    """Save mapping of original indices to split assignments as CSV.

    Args:
        train_idx: Original indices assigned to training set
        val_idx: Original indices assigned to validation set
        test_idx: Original indices assigned to test set
        output_path: Path to save the CSV file
    """
    rows = []
    for split_pos, orig_idx in enumerate(train_idx):
        rows.append({'original_idx': int(orig_idx), 'split': 'train', 'split_idx': split_pos})
    for split_pos, orig_idx in enumerate(val_idx):
        rows.append({'original_idx': int(orig_idx), 'split': 'val', 'split_idx': split_pos})
    for split_pos, orig_idx in enumerate(test_idx):
        rows.append({'original_idx': int(orig_idx), 'split': 'test', 'split_idx': split_pos})

    df = pd.DataFrame(rows).sort_values('original_idx').reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Split index mapping saved: {output_path}")


def preprocessing_outlet(config: Dict, verbose: bool = True, return_split: bool = False) -> Tuple:
    """
    Load pre-normalized data for outlet prediction.

    Processing Steps:
    1. Load normalized tensors (x, y_outlet)
    2. Load spatial and outlet normalizers
    3. Perform train/val/test split
    4. Create datasets

    Returns:
        Tuple containing (spatial_normalizer, outlet_normalizer,
                         train_dataset, val_dataset, test_dataset, device)
    """

    if verbose:
        print(f"\nLoading pre-normalized data for outlet prediction...")

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

        required_keys = ["x", "y_outlet"]
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            raise KeyError(f"Missing required keys in normalized data: {missing_keys}")

        # Load data
        x_input = bundle["x"].float()           # (N, 11, nx, ny, nt)
        y_outlet = bundle["y_outlet"].float()   # (N, nt)

        if verbose:
            print(f"   Loaded normalized tensors - Input: {tuple(x_input.shape)}, Outlet: {tuple(y_outlet.shape)}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 1 (loading normalized data): {e}")

    try:
        # Step 2: Load normalizers
        if verbose:
            print("Step 2: Loading normalizers...")

        # Load spatial normalizer
        spatial_normalizer_path = Path(config['SPATIAL_NORMALIZER_PATH'])
        if not spatial_normalizer_path.exists():
            raise FileNotFoundError(f"Spatial normalizer not found: {spatial_normalizer_path}")

        with open(spatial_normalizer_path, 'rb') as f:
            spatial_normalizer = pickle.load(f)
        spatial_normalizer = spatial_normalizer.to(device)

        # Load outlet normalizer
        outlet_normalizer_path = Path(config['OUTLET_NORMALIZER_PATH'])
        if not outlet_normalizer_path.exists():
            raise FileNotFoundError(f"Outlet normalizer not found: {outlet_normalizer_path}")

        with open(outlet_normalizer_path, 'rb') as f:
            outlet_normalizer = pickle.load(f)

        if verbose:
            print(f"   Spatial normalizer loaded from: {spatial_normalizer_path}")
            print(f"   Outlet normalizer loaded from: {outlet_normalizer_path}")
            print(f"   Outlet normalizer - Mean: {outlet_normalizer.mean:.4f}, Std: {outlet_normalizer.std:.4f}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 2 (loading normalizers): {e}")

    try:
        # Step 3: Perform train/val/test split
        if verbose:
            print("Step 3: Creating train/val/test datasets...")

        full_dataset = CustomDatasetOutlet(x_input, y_outlet)
        trainval_dataset, test_dataset, fixed_split = fixed_test_split(
            full_dataset,
            test_size=config['TEST_SIZE'],
            random_state=config['RANDOM_STATE'],
        )

        # Create a deterministic initial train/val split for backward-compatible
        # single-model eval paths and baseline generation.
        val_size_relative = config['VAL_SIZE'] / (1 - config['TEST_SIZE'])
        train_dataset, val_dataset, member_split = make_member_split(
            trainval_dataset=trainval_dataset,
            trainval_original_indices=fixed_split.trainval_indices,
            val_size_relative=val_size_relative,
            random_state=config['RANDOM_STATE']
        )

        # Save original-to-split index mapping for traceability
        save_split_index_mapping(member_split.train_indices, member_split.val_indices, fixed_split.test_indices,
                                 config['SPLIT_INDEX_CSV_PATH'])

        if verbose:
            print(f"   Train dataset size: {len(train_dataset)}")
            print(f"   Validation dataset size: {len(val_dataset)}")
            print(f"   Test dataset size: {len(test_dataset)}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 3 (dataset creation): {e}")

    if verbose:
        print("Data preprocessing completed successfully!")

    if return_split:
        return (
            spatial_normalizer,
            outlet_normalizer,
            trainval_dataset,
            test_dataset,
            fixed_split,
            train_dataset,
            val_dataset,
            device,
        )

    return (spatial_normalizer, outlet_normalizer,
            train_dataset, val_dataset, test_dataset, device)


# ==============================================================================
# Note: Scheduler classes moved to util_common.py
# ==============================================================================
# LRStepScheduler is now imported from util_common


# ==============================================================================
# Model Building Functions
# ==============================================================================

def create_model(config: Dict, train_dataset, val_dataset, test_dataset, device: str,
                n_modes: Tuple[int, ...], hidden_channels: int, n_layers: int,
                domain_padding: List[float], train_batch_size: int,
                l2_weight: float, channel_mlp_expansion: float,
                channel_mlp_skip: str,
                projection_mlp_hidden: int, projection_mlp_layers: int,
                projection_mlp_activation: str, projection_mlp_dropout: float):
    """
    Create complete model setup for outlet prediction.

    Args:
        config: Configuration dictionary
        train_dataset, val_dataset, test_dataset: Dataset objects
        device: Device to use
        n_modes: Fourier modes for TFNO
        hidden_channels: Hidden channels for TFNO
        n_layers: Number of TFNO layers
        domain_padding: Domain padding values
        train_batch_size: Batch size
        l2_weight: L2 regularization weight
        channel_mlp_expansion: Expansion ratio for channel MLP in FNO blocks
        channel_mlp_skip: Skip connection type for channel MLP in FNO blocks
        projection_mlp_hidden: Hidden channels for projection MLP (C → 1)
        projection_mlp_layers: Number of layers in projection MLP
        projection_mlp_activation: Activation function for projection MLP
        projection_mlp_dropout: Dropout for projection MLP

    Returns:
        Tuple containing (model, train_loader, val_loader, test_loader,
                         optimizer, scheduler, loss_fn)
    """

    # 1. Create DataLoaders
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

    # 2. Create loss function (MSE for vector outputs)
    loss_fn = nn.MSELoss()

    # 3. Create base TFNO configuration
    base_tfno_config = {
        'n_modes': n_modes,
        'in_channels': config['MODEL_CONFIG']['in_channels'],
        'out_channels': 1,  # Not used, but required for TFNO init
        'hidden_channels': hidden_channels,
        'n_layers': n_layers,
        'lifting_channel_ratio': config['MODEL_CONFIG']['lifting_channel_ratio'],
        'projection_channel_ratio': config['MODEL_CONFIG']['projection_channel_ratio'],
        'positional_embedding': config['MODEL_CONFIG']['positional_embedding'],
        'domain_padding': domain_padding,
        'domain_padding_mode': config['DOMAIN_PADDING_MODE'],
        'use_channel_mlp': True,
        'channel_mlp_expansion': channel_mlp_expansion,
        'channel_mlp_skip': channel_mlp_skip,
        'fno_skip': 'linear',
        'factorization': 'Tucker'
    }

    # 4. Create projection MLP configuration from hyperparameters
    projection_mlp_config = {
        'hidden_channels': projection_mlp_hidden,
        'n_layers': projection_mlp_layers,
        'activation': projection_mlp_activation,
        'dropout': projection_mlp_dropout,
    }

    # 5. Create TFNOWithPooling model
    model = TFNOWithPooling(
        base_tfno_config=base_tfno_config,
        pool_type=config['MODEL_CONFIG']['pool_type'],
        channel_mlp_config=projection_mlp_config
    ).to(device)

    # 6. Create optimizer
    optimizer = AdamW(model.parameters(),
                     lr=config['SCHEDULER_CONFIG']['initial_lr'],
                     weight_decay=l2_weight)

    # 7. Create scheduler
    scheduler_type = config['SCHEDULER_CONFIG']['scheduler_type']
    if scheduler_type == 'step':
        scheduler = LRStepScheduler(
            optimizer,
            config['SCHEDULER_CONFIG']['step_size'],
            config['SCHEDULER_CONFIG']['gamma']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'step'.")

    return (model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn)


def create_model_from_params(
    config: Dict,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str,
    params: Dict[str, Any],
):
    """Create outlet model components from canonical or Optuna parameter dictionaries."""
    resolved = resolve_domain_padding(params, config['OPTUNA_SEARCH_SPACE'])
    n_modes = (
        int(resolved['n_modes_1']),
        int(resolved['n_modes_2']),
        int(resolved['n_modes_3']),
    )
    return create_model(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        n_modes=n_modes,
        hidden_channels=int(resolved['hidden_channels']),
        n_layers=int(resolved['n_layers']),
        domain_padding=tuple(resolved['domain_padding']),
        train_batch_size=int(resolved['train_batch_size']),
        l2_weight=float(resolved['l2_weight']),
        channel_mlp_expansion=float(resolved['channel_mlp_expansion']),
        channel_mlp_skip=resolved['channel_mlp_skip'],
        projection_mlp_hidden=int(resolved['projection_mlp_hidden']),
        projection_mlp_layers=int(resolved['projection_mlp_layers']),
        projection_mlp_activation=resolved['projection_mlp_activation'],
        projection_mlp_dropout=float(resolved['projection_mlp_dropout']),
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
    """Train one outlet model and save artifacts to the provided directory."""
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
    """Load an outlet ensemble manifest and return predictor, test loader, and loss."""
    manifest = load_ensemble_manifest(manifest_file)
    params = manifest['params']
    models = []
    test_loader = None
    loss_fn = None
    base_dir = Path(manifest_file).parent

    for member in manifest['members']:
        model, _, _, test_loader, _, _, loss_fn = create_model_from_params(
            config, train_dataset, val_dataset, test_dataset, "cpu", params
        )
        state_path = Path(member['model_state_path'])
        if not state_path.is_absolute():
            state_path = base_dir / state_path
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
        model.eval()
        model.to("cpu")
        models.append(model)

    return EnsemblePredictor(models=models, device=device, keep_models_on_device=False), test_loader, loss_fn, manifest


# ==============================================================================
# Optuna Optimization Functions
# ==============================================================================

def optuna_optimization_outlet(
    config: Dict,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str,
    verbose: bool = True
) -> Dict:
    """
    Perform hyperparameter optimization using Optuna for outlet prediction model.

    This function creates an Optuna study and runs multiple trials to find
    optimal hyperparameters for the outlet prediction model. Each trial trains
    a model with different hyperparameter combinations and returns the best
    validation loss.

    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Device to use (cuda/cpu)
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
        - best_params: Best hyperparameters found
        - best_value: Best validation loss achieved
        - study: Optuna study object
        - n_trials: Number of trials completed
        - config: Configuration used
    """

    if verbose:
        print("\n" + "="*80)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION - OUTLET MODEL")
        print("="*80)

    # Create output directory for Optuna results
    optuna_output_dir = Path(config['OUTPUT_DIR']) / 'optuna'
    optuna_output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        """Objective function for Optuna optimization."""

        # Sample hyperparameters from search space
        search_space = config['OPTUNA_SEARCH_SPACE']

        # 1. TFNO architecture - Sample n_modes for each dimension
        n_modes_1 = trial.suggest_int('n_modes_1',
                                      search_space['n_modes_dim1_range'][0],
                                      search_space['n_modes_dim1_range'][1])
        n_modes_2 = trial.suggest_int('n_modes_2',
                                      search_space['n_modes_dim2_range'][0],
                                      search_space['n_modes_dim2_range'][1])
        n_modes_3 = trial.suggest_int('n_modes_3',
                                      search_space['n_modes_dim3_range'][0],
                                      search_space['n_modes_dim3_range'][1])
        n_modes = (n_modes_1, n_modes_2, n_modes_3)

        hidden_channels = trial.suggest_int('hidden_channels',
                                           search_space['hidden_channels_range'][0],
                                           search_space['hidden_channels_range'][1])
        n_layers = trial.suggest_int('n_layers',
                                    search_space['n_layers_range'][0],
                                    search_space['n_layers_range'][1])

        # 2. Domain and training parameters
        domain_padding_idx = trial.suggest_categorical(
            'domain_padding_idx',
            list(range(len(search_space['domain_padding_options'])))
        )
        domain_padding = search_space['domain_padding_options'][domain_padding_idx]

        train_batch_size = trial.suggest_categorical('train_batch_size',
                                                     search_space['train_batch_size_options'])

        l2_weight = trial.suggest_float('l2_weight',
                                       search_space['l2_weight_range'][0],
                                       search_space['l2_weight_range'][1],
                                       log=True)

        # 3. FNO block channel MLP parameters
        channel_mlp_expansion = trial.suggest_categorical(
            'channel_mlp_expansion',
            search_space['channel_mlp_expansion_options']
        )
        channel_mlp_skip = trial.suggest_categorical(
            'channel_mlp_skip',
            search_space['channel_mlp_skip_options']
        )

        # 4. Projection MLP parameters (outlet-specific)
        projection_mlp_hidden = trial.suggest_int(
            'projection_mlp_hidden',
            search_space['projection_mlp_hidden_range'][0],
            search_space['projection_mlp_hidden_range'][1]
        )
        projection_mlp_layers = trial.suggest_categorical(
            'projection_mlp_layers',
            search_space['projection_mlp_layers_options']
        )
        projection_mlp_activation = trial.suggest_categorical(
            'projection_mlp_activation',
            search_space['projection_mlp_activation_options']
        )
        projection_mlp_dropout = trial.suggest_float(
            'projection_mlp_dropout',
            search_space['projection_mlp_dropout_range'][0],
            search_space['projection_mlp_dropout_range'][1]
        )

        try:
            # Create model with sampled parameters
            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                n_layers=n_layers,
                domain_padding=domain_padding,
                train_batch_size=train_batch_size,
                l2_weight=l2_weight,
                channel_mlp_expansion=channel_mlp_expansion,
                channel_mlp_skip=channel_mlp_skip,
                projection_mlp_hidden=projection_mlp_hidden,
                projection_mlp_layers=projection_mlp_layers,
                projection_mlp_activation=projection_mlp_activation,
                projection_mlp_dropout=projection_mlp_dropout
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
                del loss_history  # Free memory
            else:
                # Fallback: return a high loss value if history not found
                best_val_loss = float('inf')

            # Explicitly delete all GPU objects before returning
            del model, trained_model, optimizer, scheduler, loss_fn
            del train_loader, val_loader, test_loader

            # Clear GPU cache
            if device == 'cuda':
                torch.cuda.empty_cache()

            return best_val_loss

        except Exception as e:
            if verbose:
                print(f"Trial {trial.number} failed with error: {e}")
            # Return a high loss value for failed trials
            return float('inf')

    # Progress callback to save best model
    def progress_callback(study, trial):
        """Save best model whenever a new best trial is found."""
        if trial.value == study.best_value:
            # Create directory for best trial model
            best_trial_dir = optuna_output_dir / 'best_trial_model'
            best_trial_dir.mkdir(parents=True, exist_ok=True)

            # Copy best model state dict
            source_path = Path(trial.user_attrs.get('model_state_path', ''))
            if source_path.exists():
                dest_path = best_trial_dir / 'best_model_state_dict.pt'
                shutil.copy2(source_path, dest_path)

                # Save trial info as JSON
                trial_info = {
                    'trial_number': trial.number,
                    'best_value': trial.value,
                    'params': trial.params,
                    'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
                }
                info_path = best_trial_dir / 'trial_info.json'
                with open(info_path, 'w') as f:
                    json.dump(trial_info, f, indent=2)

                if verbose:
                    print(f"\n>>> New best trial! Saved model from trial {trial.number}")
                    print(f"    Best validation loss: {trial.value:.6f}")

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(
            seed=config['TRAINING_CONFIG']['optuna_seed'],
            n_startup_trials=config['TRAINING_CONFIG']['optuna_n_startup_trials']
        )
    )

    # Run optimization
    n_trials = config['TRAINING_CONFIG']['optuna_n_trials']
    if verbose:
        print(f"\nStarting Optuna optimization with {n_trials} trials...")
        print(f"Search space: {len(config['OPTUNA_SEARCH_SPACE'])} hyperparameters")

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[progress_callback],
        show_progress_bar=verbose
    )

    # Get best parameters and value
    best_params = study.best_params
    best_value = study.best_value

    if verbose:
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best validation loss: {best_value:.6f}")
        print(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Check if best model was saved
        best_model_path = optuna_output_dir / 'best_trial_model' / 'best_model_state_dict.pt'
        if best_model_path.exists():
            print(f"\nBest model saved to: {best_model_path}")
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
        ax.set_ylabel('Validation Loss (MSE)')
        ax.set_title('Optuna Optimization History - Outlet Model')
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
# Training Functions
# ==============================================================================
# Note: Training logic moved to util_training.py
# Wrapper functions are provided below for compatibility

def train_model(config: Dict, device: str, model, train_loader, val_loader, test_loader,
                optimizer, scheduler, loss_fn, verbose: bool = True):
    """
    Train the outlet prediction model with early stopping.

    This is a wrapper around train_model_generic() from util_training.py.

    Returns:
        Trained model
    """
    output_dir = Path(config['OUTPUT_DIR']) / 'final'
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
        verbose=verbose
    )


def model_evaluation(config: Dict, device: str, model, test_loader, loss_fn,
                    verbose: bool = True):
    """
    Evaluate the trained model on test set.

    This is a wrapper around model_evaluation_generic() from util_training.py.

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
        compute_mse=False,  # Loss is already MSE for outlet model
        verbose=verbose
    )


# ==============================================================================
# Integrated Gradients Functions
# ==============================================================================

def create_multi_sample_baselines_outlet(
    train_dataset,
    val_dataset,
    test_dataset,
    n_baselines: int = 5,
    random_seed: int = 42,
    verbose: bool = True
) -> List[torch.Tensor]:
    """
    Create multiple real sample baselines for IG analysis (outlet version).

    Instead of using a single mean baseline (which may become homogeneous and outside
    the training distribution for heterogeneous-only trained models), this function
    selects n_baselines real samples from the datasets to use as baselines.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        n_baselines: Number of baseline samples to select
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        List of baseline tensors, each of shape (1, C, nx, ny, nt)
    """
    if verbose:
        print(f"Creating {n_baselines} real sample baselines from datasets...")

    # Collect all samples
    all_samples = []
    for i in range(len(train_dataset)):
        all_samples.append(train_dataset[i]['x'])
    for i in range(len(val_dataset)):
        all_samples.append(val_dataset[i]['x'])
    for i in range(len(test_dataset)):
        all_samples.append(test_dataset[i]['x'])

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_seed)

    # Randomly select n_baselines samples
    total_samples = len(all_samples)
    if n_baselines > total_samples:
        if verbose:
            print(f"  Warning: n_baselines ({n_baselines}) > total samples ({total_samples})")
            print(f"  Using all {total_samples} samples as baselines")
        selected_indices = list(range(total_samples))
    else:
        selected_indices = rng.choice(total_samples, size=n_baselines, replace=False)

    # Create list of baseline tensors
    baselines = []
    for idx in selected_indices:
        baseline = all_samples[int(idx)].unsqueeze(0)  # Add batch dimension: (1, C, nx, ny, nt)
        baselines.append(baseline)

    if verbose:
        print(f"  Total samples available: {total_samples}")
        print(f"  Selected {len(baselines)} baseline samples")
        # Convert to list for sorting (handles both list and ndarray cases)
        indices_list = selected_indices if isinstance(selected_indices, list) else selected_indices.tolist()
        print(f"  Baseline indices: {sorted(indices_list)}")
        print(f"  Each baseline shape: {baselines[0].shape}")

    return baselines


def compute_integrated_gradients_outlet(
    model: nn.Module,
    outlet_normalizer,
    device: str,
    test_sample: torch.Tensor,
    baseline: torch.Tensor,
    target_t: int,
    n_steps: int = 50,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Integrated Gradients for a specific target time in outlet prediction.

    Args:
        model: Trained outlet prediction model
        outlet_normalizer: Outlet normalizer for inverse transform
        device: Device to use
        test_sample: Test sample tensor (1, C, nx, ny, nt)
        baseline: Baseline tensor (1, C, nx, ny, nt)
        target_t: Target time index
        n_steps: Number of interpolation steps
        verbose: Whether to print progress

    Returns:
        Tuple of (ig_spatial, info_dict)
        - ig_spatial: IG attribution (C, nx, ny)
        - info_dict: Dictionary with metadata
    """
    if verbose:
        print(f"\nComputing IG for time {target_t}...")
        print(f"  Steps: {n_steps}")

    # Wrapper model to extract specific time index from outlet prediction
    class OutletTimeWrapper(nn.Module):
        def __init__(self, model, outlet_normalizer, target_t):
            super().__init__()
            self.model = model
            self.outlet_normalizer = outlet_normalizer
            self.target_t = target_t

        def forward(self, x):
            # x is already normalized
            pred = self.model(x)  # (B, nt)
            # Convert to raw physical values
            pred_phys = self.outlet_normalizer.inverse_transform(pred)  # (B, nt)
            # Extract target time (return scalar for backward)
            output_t = pred_phys[:, self.target_t]  # (B,)
            # Use absolute value to get correct IG sign interpretation
            # (outlet values are negative, but we want IG to show contribution to magnitude)
            output_t_abs = torch.abs(output_t)
            # Return sum for gradient computation
            return output_t_abs.sum()

    wrapped = OutletTimeWrapper(model, outlet_normalizer, target_t).to(device)

    # Compute gradients
    grads = []
    outputs = []
    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (test_sample - baseline)
        interpolated = interpolated.to(device).requires_grad_(True)

        output = wrapped(interpolated)
        output.backward()

        grads.append(interpolated.grad.detach().cpu().clone())
        outputs.append(output.item())
        interpolated.grad = None

        if verbose and step % 10 == 0:
            print(f"  Step {step}/{n_steps}, output={output.item():.4e}")

    # Trapezoidal integration of gradients along interpolation path:
    # integral_grad = (g0/2 + g1 + ... + gN/2) / N
    grads_tensor = torch.stack(grads, dim=0)  # (n_steps+1, 1, C, nx, ny, nt)
    integral_grad = grads_tensor[0] * 0.5 + grads_tensor[-1] * 0.5
    if n_steps > 1:
        integral_grad = integral_grad + grads_tensor[1:-1].sum(dim=0)
    integral_grad = integral_grad / n_steps

    # IG: (x - baseline) × integral_grad
    ig = (test_sample - baseline) * integral_grad

    # Sum over time dimension to get spatial attribution
    ig_spatial = ig[0, :, :, :, :].sum(dim=-1).numpy()  # (C, nx, ny)
    ig_spatial_merged = merge_velocity_channels(ig_spatial)

    ig_sum = float(ig_spatial_merged.sum())
    output_change = outputs[-1] - outputs[0]
    completeness_error = ig_sum - output_change
    completeness_rel_error = completeness_error / (abs(output_change) + 1e-12)

    # Metadata
    info = {
        'target_t': target_t,
        'n_steps': n_steps,
        'total_abs_ig': float(np.abs(ig_spatial_merged).sum()),
        'ig_sum': ig_sum,
        'output_baseline': outputs[0],
        'output_actual': outputs[-1],
        'output_change': output_change,
        'completeness_target': output_change,
        'completeness_error': completeness_error,
        'completeness_rel_error': completeness_rel_error,
    }

    if verbose:
        print(f"  Done. Total |IG|: {info['total_abs_ig']:.4e}")
        print(f"  Output change: {info['output_change']:.4e}")
        print(f"  IG sum: {info['ig_sum']:.4e}")
        print(f"  Completeness error: {info['completeness_error']:.4e}")
        print(f"  Completeness rel. error: {info['completeness_rel_error']:.2%}")

    return ig_spatial, info


def compute_integrated_gradients_outlet_multi_baseline(
    model: nn.Module,
    outlet_normalizer,
    device: str,
    test_sample: torch.Tensor,
    baselines: List[torch.Tensor],
    target_t: int,
    n_steps: int = 50,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Integrated Gradients using multiple real sample baselines for outlet prediction.

    This approach addresses the issue where a single mean baseline becomes homogeneous
    and falls outside the training distribution for models trained on heterogeneous inputs.
    By using multiple real samples as baselines, we ensure that interpolation paths
    remain within the learned distribution.

    Args:
        model: Trained outlet prediction model
        outlet_normalizer: Outlet normalizer for inverse transform
        device: Device to use
        test_sample: Test sample tensor (1, C, nx, ny, nt)
        baselines: List of baseline tensors, each of shape (1, C, nx, ny, nt)
        target_t: Target time index
        n_steps: Number of interpolation steps per baseline
        verbose: Whether to print progress

    Returns:
        Tuple of (ig_spatial_avg, info_dict)
        - ig_spatial_avg: Averaged IG attribution (C, nx, ny)
        - info_dict: Dictionary with metadata including per-baseline statistics
    """
    if verbose:
        print(f"\nComputing IG with {len(baselines)} baselines for time {target_t}...")
        print(f"  Steps per baseline: {n_steps}")

    # Storage for IG results from each baseline
    ig_results = []
    baseline_infos = []

    # Compute IG for each baseline
    for i, baseline in enumerate(baselines):
        if verbose:
            print(f"\n  Baseline {i+1}/{len(baselines)}:")

        ig_spatial, info = compute_integrated_gradients_outlet(
            model=model,
            outlet_normalizer=outlet_normalizer,
            device=device,
            test_sample=test_sample,
            baseline=baseline,
            target_t=target_t,
            n_steps=n_steps,
            verbose=verbose
        )

        ig_results.append(ig_spatial)
        baseline_infos.append(info)

    # Convert to array for easier manipulation: (n_baselines, C, nx, ny)
    ig_array = np.stack(ig_results, axis=0)

    # Compute average IG across baselines
    ig_spatial_avg = ig_array.mean(axis=0)  # (C, nx, ny)

    # Compute statistics across baselines
    ig_std = ig_array.std(axis=0)  # Standard deviation (C, nx, ny)
    ig_min = ig_array.min(axis=0)  # Minimum (C, nx, ny)
    ig_max = ig_array.max(axis=0)  # Maximum (C, nx, ny)

    # Aggregate metadata
    total_abs_igs = [info['total_abs_ig'] for info in baseline_infos]
    ig_sums = [info['ig_sum'] for info in baseline_infos]
    output_baselines = [info['output_baseline'] for info in baseline_infos]
    output_actuals = [info['output_actual'] for info in baseline_infos]
    output_changes = [info['output_change'] for info in baseline_infos]
    completeness_errors = [info['completeness_error'] for info in baseline_infos]
    completeness_rel_errors = [info['completeness_rel_error'] for info in baseline_infos]

    ig_spatial_avg_merged = merge_velocity_channels(ig_spatial_avg)
    ig_sum = float(ig_spatial_avg_merged.sum())
    output_change_mean = float(np.mean(output_changes))
    completeness_error = ig_sum - output_change_mean
    completeness_rel_error = completeness_error / (abs(output_change_mean) + 1e-12)

    info_avg = {
        'target_t': target_t,
        'n_steps': n_steps,
        'n_baselines': len(baselines),
        'total_abs_ig': float(np.abs(ig_spatial_avg_merged).sum()),
        'ig_sum': ig_sum,
        'output_actual': float(np.mean(output_actuals)),  # Should be identical for all baselines
        'completeness_target': output_change_mean,
        'completeness_error': completeness_error,
        'completeness_rel_error': completeness_rel_error,
        # Statistics across baselines
        'baseline_stats': {
            'total_abs_ig_mean': float(np.mean(total_abs_igs)),
            'total_abs_ig_std': float(np.std(total_abs_igs)),
            'total_abs_ig_min': float(np.min(total_abs_igs)),
            'total_abs_ig_max': float(np.max(total_abs_igs)),
            'ig_sum_mean': float(np.mean(ig_sums)),
            'ig_sum_std': float(np.std(ig_sums)),
            'output_baseline_mean': float(np.mean(output_baselines)),
            'output_baseline_std': float(np.std(output_baselines)),
            'output_change_mean': float(np.mean(output_changes)),
            'output_change_std': float(np.std(output_changes)),
            'completeness_error_mean': float(np.mean(completeness_errors)),
            'completeness_error_std': float(np.std(completeness_errors)),
            'completeness_error_min': float(np.min(completeness_errors)),
            'completeness_error_max': float(np.max(completeness_errors)),
            'completeness_rel_error_mean': float(np.mean(completeness_rel_errors)),
            'completeness_rel_error_std': float(np.std(completeness_rel_errors)),
        },
        # Per-channel statistics
        'channel_stats': {
            'ig_std': ig_std,  # (C, nx, ny) - spatial standard deviation
            'ig_min': ig_min,  # (C, nx, ny) - spatial minimum
            'ig_max': ig_max,  # (C, nx, ny) - spatial maximum
        }
    }

    if verbose:
        print(f"\n  Multi-baseline IG completed:")
        print(f"    Average Total |IG|: {info_avg['total_abs_ig']:.4e}")
        print(f"    Average IG sum: {info_avg['ig_sum']:.4e}")
        print(f"    Baseline variability (Total |IG|):")
        print(f"      Mean: {info_avg['baseline_stats']['total_abs_ig_mean']:.4e}")
        print(f"      Std:  {info_avg['baseline_stats']['total_abs_ig_std']:.4e}")
        print(f"      Range: [{info_avg['baseline_stats']['total_abs_ig_min']:.4e}, "
              f"{info_avg['baseline_stats']['total_abs_ig_max']:.4e}]")
        print(f"    Completeness error: {info_avg['completeness_error']:.4e}")
        print(f"    Completeness rel. error: {info_avg['completeness_rel_error']:.2%}")

    return ig_spatial_avg, info_avg


def visualize_input_channels_outlet(
    input_data: np.ndarray,
    sample_idx: int,
    output_dir: Path,
    verbose: bool = True
) -> List[Path]:
    """
    Visualize input channels for IG analysis.
    Creates one image per channel (time-invariant).

    Args:
        input_data: Input data array (C, nx, ny, nt)
        sample_idx: Sample index
        output_dir: Output directory
        verbose: Whether to print progress

    Returns:
        List of paths to saved images
    """
    channel_names = CHANNEL_NAMES_11
    channel_short = CHANNEL_SHORT_11

    saved_paths = []
    n_channels = input_data.shape[0]

    for ch in range(n_channels):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Input channel at t=0 (time-invariant)
        input_slice = input_data[ch, :, :, 0]
        input_vmin = np.percentile(input_slice, 2)
        input_vmax = np.percentile(input_slice, 98)

        im = ax.imshow(input_slice.T, cmap='viridis',
                      vmin=input_vmin, vmax=input_vmax, aspect='auto')
        ax.set_title(f'{channel_names[ch]} Input (Sample {sample_idx})', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.formatter.set_useMathText(True)
        cbar.update_ticks()

        plt.tight_layout()

        save_path = output_dir / f'ch{ch:02d}_{channel_short[ch]}_input.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        saved_paths.append(save_path)

        if verbose:
            print(f"  Saved: {save_path.name}")

    return saved_paths


def visualize_ig_attributions_outlet(
    ig_results: Dict[int, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    verbose: bool = True
) -> List[Path]:
    """
    Visualize IG attribution maps for each channel and time index.
    Creates separate images for each (channel, time) combination.

    Args:
        ig_results: Dictionary mapping time indices to IG arrays (C, nx, ny)
        sample_idx: Sample index being analyzed
        output_dir: Output directory for images
        verbose: Whether to print progress

    Returns:
        List of paths to saved images
    """
    channel_names = CHANNEL_NAMES_10_MERGED
    channel_short = CHANNEL_SHORT_10_MERGED

    # Compute global ranges for each channel (across all time indices)
    # Using symmetric colorbar centered at 0 for better interpretation
    merged_ig_results = {t_idx: merge_velocity_channels(ig_spatial) for t_idx, ig_spatial in ig_results.items()}
    n_channels = merged_ig_results[list(merged_ig_results.keys())[0]].shape[0]
    ig_ranges = {}

    for ch in range(n_channels):
        all_ig_values = []
        for t_idx, ig_spatial in merged_ig_results.items():
            all_ig_values.append(ig_spatial[ch].flatten())
        combined_ig = np.concatenate(all_ig_values)

        # Use absolute maximum to create symmetric range around 0
        # This ensures:
        # - 0 is always at the center (white in RdBu_r colormap)
        # - Red (positive) and Blue (negative) have equal scale
        # - Small contributions are visible as light colors
        abs_max = np.abs(combined_ig).max()

        # Handle edge case where all values are zero
        if abs_max < 1e-20:
            abs_max = 1e-20

        # Symmetric range: [-abs_max, +abs_max]
        ig_vmin = -abs_max
        ig_vmax = abs_max

        ig_ranges[ch] = (ig_vmin, ig_vmax)

    saved_paths = []

    # Create separate image for each (channel, time) combination
    for t_idx, ig_spatial in merged_ig_results.items():
        for ch in range(n_channels):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # IG attribution
            ig_map = ig_spatial[ch]
            ig_vmin, ig_vmax = ig_ranges[ch]
            ig_sum = ig_map.sum()

            im = ax.imshow(ig_map.T, cmap='RdBu_r',
                          vmin=ig_vmin, vmax=ig_vmax, aspect='auto')
            ax.set_title(f'{channel_names[ch]} IG Attribution (t={t_idx}, ∑IG={ig_sum:.4e})',
                        fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.formatter.set_useMathText(True)
            cbar.update_ticks()

            plt.tight_layout()

            save_path = output_dir / f'ch{ch:02d}_{channel_short[ch]}_t{t_idx:02d}_ig.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            saved_paths.append(save_path)

            if verbose:
                print(f"  Saved: {save_path.name}")

    return saved_paths


def save_ig_csv_outlet(
    ig_results: Dict[int, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    verbose: bool = True
) -> List[Path]:
    """
    Save outlet IG results as per-time CSV files (Option A, wide format).

    Format per time:
        x_coord, y_coord, Permeability_IG, Calcite_IG, ...

    Args:
        ig_results: Dictionary mapping time indices to IG arrays (C, nx, ny)
        sample_idx: Sample index being analyzed
        output_dir: Output directory
        verbose: Whether to print progress

    Returns:
        List of paths to saved CSV files
    """
    channel_names = CHANNEL_NAMES_10_MERGED

    saved_paths = []

    for t_idx in sorted(ig_results.keys()):
        ig_spatial = merge_velocity_channels(ig_results[t_idx])  # (10, nx, ny)
        n_channels, nx, ny = ig_spatial.shape

        x_coords = []
        y_coords = []
        for x in range(nx):
            for y in range(ny):
                x_coords.append(x)
                y_coords.append(y)

        data = {
            'x_coord': x_coords,
            'y_coord': y_coords
        }

        for ch in range(n_channels):
            col_name = f'{channel_names[ch]}_IG'
            vals = []
            for x in range(nx):
                for y in range(ny):
                    vals.append(ig_spatial[ch, x, y])
            data[col_name] = vals

        df = pd.DataFrame(data)
        csv_path = output_dir / f'ig_data_s{sample_idx}_t{t_idx:02d}.csv'
        df.to_csv(csv_path, index=False, float_format='%.6e')
        saved_paths.append(csv_path)

        if verbose:
            print(f"  Saved: {csv_path.name}")

    return saved_paths


def analyze_channel_importance_outlet(
    ig_results: Dict[int, np.ndarray],
    output_dir: Path,
    verbose: bool = True
) -> Tuple[Path, Path]:
    """
    Analyze channel-wise IG importance evolution over time for outlet IG.

    Importance metric:
        sum_{x,y} |IG(channel, x, y)| per time index.

    Args:
        ig_results: Dictionary mapping time indices to IG arrays (C, nx, ny)
        output_dir: Output directory
        verbose: Whether to print progress

    Returns:
        Tuple of (importance_csv_path, importance_plot_path)
    """
    channel_names = [
        'Perm', 'Calcite', 'Clino', 'Pyrite', 'Smectite',
        'MatSrc', 'MatBent', 'MatFrac', 'Velocity', 'Meta'
    ]

    times = sorted(ig_results.keys())
    n_channels = merge_velocity_channels(ig_results[times[0]]).shape[0]
    importance = np.zeros((len(times), n_channels))

    for i, t_idx in enumerate(times):
        ig_spatial = merge_velocity_channels(ig_results[t_idx])
        for ch in range(n_channels):
            importance[i, ch] = np.abs(ig_spatial[ch]).sum()

    df = pd.DataFrame(importance, index=times, columns=channel_names)
    df.index.name = 'time'
    csv_path = output_dir / 'channel_importance.csv'
    df.to_csv(csv_path, float_format='%.6e')

    plt.figure(figsize=(12, 6))
    for ch in range(n_channels):
        plt.plot(times, importance[:, ch], marker='o', label=channel_names[ch])
    plt.xlabel('Time Index')
    plt.ylabel('Total |IG|')
    plt.title('Outlet IG Channel Importance Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = output_dir / 'importance_evolution.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved importance analysis: {csv_path.name}, {plot_path.name}")

    return csv_path, plot_path


def integrated_gradients_analysis_outlet(
    config: Dict,
    outlet_normalizer,
    device: str,
    model: nn.Module,
    train_dataset,
    val_dataset,
    test_dataset,
    verbose: bool = True
) -> Dict:
    """
    Perform complete Integrated Gradients analysis for outlet prediction.

    Args:
        config: Configuration dictionary
        outlet_normalizer: Outlet normalizer for inverse transform
        device: Device to use
        model: Trained model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        verbose: Whether to print progress

    Returns:
        Dictionary containing IG results
    """
    print("\n" + "="*70)
    print("INTEGRATED GRADIENTS ANALYSIS - OUTLET PREDICTION")
    print("="*70)

    ig_config = config.get('IG_ANALYSIS', {})
    sample_idx_cfg = ig_config.get('SAMPLE_IDX', 0)
    time_indices = ig_config.get('TIME_INDICES', [4, 9, 14, 19])
    n_steps = ig_config.get('N_STEPS', 50)

    n_baselines = ig_config.get('N_BASELINES', 5)
    baseline_seed = ig_config.get('BASELINE_SEED', 42)

    # Support both int and list for sample indices
    if isinstance(sample_idx_cfg, (list, tuple, np.ndarray)):
        requested_indices = [int(idx) for idx in sample_idx_cfg]
    else:
        requested_indices = [int(sample_idx_cfg)]

    valid_sample_indices = []
    for idx in requested_indices:
        if 0 <= idx < len(test_dataset):
            valid_sample_indices.append(idx)
        else:
            print(f"Warning: sample_idx {idx} out of range. Skipping.")

    if not valid_sample_indices:
        print("Warning: No valid sample indices found. Using sample 0.")
        valid_sample_indices = [0]

    # Deduplicate while preserving order
    valid_sample_indices = list(dict.fromkeys(valid_sample_indices))

    # Multi baseline only (other baseline modes removed by design)
    baselines = create_multi_sample_baselines_outlet(
        train_dataset, val_dataset, test_dataset,
        n_baselines=n_baselines,
        random_seed=baseline_seed,
        verbose=verbose
    )

    all_results = {}

    for sample_idx in valid_sample_indices:
        # Get test sample
        test_sample = test_dataset[sample_idx]['x'].unsqueeze(0)  # (1, C, nx, ny, nt)
        test_outlet = test_dataset[sample_idx]['y']  # (nt,)

        print(f"\nAnalyzing sample {sample_idx} at times {time_indices}")
        print(f"Test sample shape: {tuple(test_sample.shape)}")
        print(f"Test outlet shape: {tuple(test_outlet.shape)}")

        # Compute IG for each time index
        ig_results = {}
        info_results = {}
        for t in time_indices:
            if t >= test_outlet.shape[0]:
                print(f"  Skipping time {t} (out of range)")
                continue

            ig_spatial, info = compute_integrated_gradients_outlet_multi_baseline(
                model=model,
                outlet_normalizer=outlet_normalizer,
                device=device,
                test_sample=test_sample,
                baselines=baselines,
                target_t=t,
                n_steps=n_steps,
                verbose=verbose
            )

            ig_results[t] = ig_spatial
            info_results[t] = info

        # Generate outputs for current sample
        output_dir = Path(config['OUTPUT_DIR']) / 'integrated_gradients' / f'sample_{sample_idx}'
        output_dir.mkdir(parents=True, exist_ok=True)

        baseline_paths = []
        if verbose:
            print("\nSkipping baseline visualization (multi-baseline only mode)")

        print("\nGenerating input channel images...")
        input_data = test_sample[0].cpu().numpy()  # (C, nx, ny, nt)
        input_paths = visualize_input_channels_outlet(
            input_data, sample_idx, output_dir, verbose
        )

        print("\nGenerating IG attribution images...")
        ig_paths = visualize_ig_attributions_outlet(
            ig_results, sample_idx, output_dir, verbose
        )

        print("\nSaving IG CSV files...")
        csv_paths = save_ig_csv_outlet(ig_results, sample_idx, output_dir, verbose)

        print("\nGenerating channel importance analysis...")
        importance_csv, importance_plot = analyze_channel_importance_outlet(
            ig_results, output_dir, verbose
        )

        summary_path = output_dir / 'ig_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Integrated Gradients Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Sample Index: {sample_idx}\n")
            f.write(f"Time Indices Analyzed: {time_indices}\n")
            f.write(f"Integration Steps: {n_steps}\n")
            f.write("Baseline Mode: multi\n")
            f.write("Velocity merged: True (Vx+Vy)\n")
            f.write(f"Number of Baselines: {n_baselines}\n")
            f.write(f"Baseline Seed: {baseline_seed}\n")
            f.write(f"IG CSV files saved: {len(csv_paths)}\n")
            f.write(f"Channel importance CSV: {importance_csv.name}\n")
            f.write(f"Channel importance plot: {importance_plot.name}\n")
            f.write("\n")

            for t in sorted(info_results.keys()):
                info = info_results[t]
                f.write(f"Time {t}:\n")

                f.write(f"  Output (actual): {info['output_actual']:.6e}\n")
                f.write(f"  Output (baseline mean): {info['baseline_stats']['output_baseline_mean']:.6e}\n")
                f.write(f"  Output (baseline std): {info['baseline_stats']['output_baseline_std']:.6e}\n")
                f.write(f"  Output change (mean): {info['baseline_stats']['output_change_mean']:.6e}\n")
                f.write(f"  Output change (std): {info['baseline_stats']['output_change_std']:.6e}\n")
                f.write(f"  Total |IG|: {info['total_abs_ig']:.6e}\n")
                f.write(f"  Total |IG| (mean): {info['baseline_stats']['total_abs_ig_mean']:.6e}\n")
                f.write(f"  Total |IG| (std): {info['baseline_stats']['total_abs_ig_std']:.6e}\n")
                f.write(f"  IG sum: {info['ig_sum']:.6e}\n")
                f.write(f"  IG sum (mean): {info['baseline_stats']['ig_sum_mean']:.6e}\n")
                f.write(f"  IG sum (std): {info['baseline_stats']['ig_sum_std']:.6e}\n")
                f.write(f"  Completeness target (mean): {info['completeness_target']:.6e}\n")
                f.write(f"  Completeness error: {info['completeness_error']:.6e}\n")
                f.write(f"  Completeness rel. error: {info['completeness_rel_error']:.6e}\n")
                f.write(f"  Completeness error (mean): {info['baseline_stats']['completeness_error_mean']:.6e}\n")
                f.write(f"  Completeness error (std): {info['baseline_stats']['completeness_error_std']:.6e}\n")
                f.write(f"  Completeness error (range): "
                        f"[{info['baseline_stats']['completeness_error_min']:.6e}, "
                        f"{info['baseline_stats']['completeness_error_max']:.6e}]\n")
                f.write(f"  Completeness rel. error (mean): "
                        f"{info['baseline_stats']['completeness_rel_error_mean']:.6e}\n")
                f.write(f"  Completeness rel. error (std): "
                        f"{info['baseline_stats']['completeness_rel_error_std']:.6e}\n")

                f.write("\n")

        print(f"\n  Saved summary: {summary_path}")

        sample_result = {
            'ig_results': ig_results,
            'info_results': info_results,
            'baseline_paths': baseline_paths,
            'input_paths': input_paths,
            'ig_paths': ig_paths,
            'csv_paths': csv_paths,
            'importance_csv': importance_csv,
            'importance_plot': importance_plot,
            'summary_path': summary_path
        }
        all_results[f'sample_{sample_idx}'] = sample_result

    print("\nIntegrated Gradients analysis complete!")

    # Backward compatibility: keep top-level single-sample keys when only one sample is analyzed
    if len(all_results) == 1:
        only_result = next(iter(all_results.values()))
        only_result['samples'] = all_results
        return only_result

    return {'samples': all_results}


# ==============================================================================
# Visualization Functions
# ==============================================================================
# Note: Outlet visualization functions moved to util_output_outlet.py
# visualize_outlet_predictions is now imported from util_output_outlet


# ==============================================================================
# Main Workflow
# ==============================================================================

def main():
    """Main workflow for outlet prediction model training."""

    print("="*80)
    print("FNO Outlet Prediction Training")
    print("="*80)

    # 1. Load data and create datasets
    (
        spatial_normalizer,
        outlet_normalizer,
        trainval_dataset,
        test_dataset,
        fixed_split,
        train_dataset,
        val_dataset,
        device,
    ) = preprocessing_outlet(CONFIG, return_split=True)

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
            optimization_results = optuna_optimization_outlet(
                config=CONFIG,
                train_dataset=optuna_train_dataset,
                val_dataset=optuna_val_dataset,
                test_dataset=test_dataset,
                device=device,
                verbose=True
            )

        params = resolve_training_params(training_mode, CONFIG, optimization_results)
        print("\nTraining final outlet ensemble with resolved hyperparameters...")
        print(f"Projection ChannelMLP hidden: {params['projection_mlp_hidden']}")
        print(f"Projection ChannelMLP layers: {params['projection_mlp_layers']}")

        def create_components(train_ds, val_ds, test_ds, member_params, output_dir):
            return create_model_from_params(CONFIG, train_ds, val_ds, test_ds, device, member_params)

        def train_one(model, train_loader, val_loader, optimizer, scheduler, loss_fn, output_dir):
            return train_model_to_dir(
                CONFIG, device, model, train_loader, val_loader,
                optimizer, scheduler, loss_fn, output_dir, verbose=True
            )

        ensemble_run = train_ensemble(
            config=CONFIG,
            model_kind='fno_outlet',
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

        model_evaluation(CONFIG, device, trained_model, test_loader, loss_fn, verbose=True)
        visualize_outlet_predictions(CONFIG, device, trained_model, test_loader, outlet_normalizer)

        if CONFIG.get('IG_ANALYSIS', {}).get('ENABLED', False):
            integrated_gradients_analysis_outlet(
                CONFIG, outlet_normalizer, device, trained_model,
                train_dataset, val_dataset, test_dataset, verbose=True
            )

        print(f"   Ensemble manifest saved to: {ensemble_run.manifest_path}")

    elif training_mode == 'eval':
        model_path = Path(CONFIG['TRAINING_CONFIG']['eval_model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_path.suffix.lower() == '.json':
            trained_model, test_loader, loss_fn, manifest = load_ensemble_for_evaluation(
                CONFIG,
                model_path,
                train_dataset,
                val_dataset,
                test_dataset,
                device,
            )
            print(f"Loaded outlet ensemble manifest from: {model_path}")
            print(f"Ensemble members: {len(manifest['members'])}")
        else:
            model, _, _, test_loader, _, _, loss_fn = create_model_from_params(
                CONFIG, train_dataset, val_dataset, test_dataset, device, CONFIG['SINGLE_PARAMS']
            )
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.eval()
            trained_model = model
            print(f"Loaded pretrained model from: {model_path}")

        model_evaluation(CONFIG, device, trained_model, test_loader, loss_fn, verbose=True)
        visualize_outlet_predictions(CONFIG, device, trained_model, test_loader, outlet_normalizer)

        if CONFIG.get('IG_ANALYSIS', {}).get('ENABLED', False):
            integrated_gradients_analysis_outlet(
                CONFIG, outlet_normalizer, device, trained_model,
                train_dataset, val_dataset, test_dataset, verbose=True
            )

    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


if __name__ == '__main__':
    main()
