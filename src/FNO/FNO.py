"""
Pure FNO with Uniform Distribution Meta Training - Refactored Version

This module implements training pipeline for Pure TFNO models with meta data incorporated 
as uniform spatial channels. Meta data (e.g., permeability, porosity) is expanded to 
uniform distribution across spatial dimensions and directly concatenated with input channels, 
providing a straightforward approach to conditional neural operators.

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
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from neuraloperator.neuralop.utils import count_model_params
from neuraloperator.neuralop.models import TFNO, FNO
from neuraloperator.neuralop.training import AdamW

# Import unified output utility
from util_output import generate_all_outputs

# Import common utilities (refactored from duplicate code)
from util_common import LpLoss, LRStepScheduler, CappedCosineAnnealingWarmRestarts

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

# Import preprocessing normalizer (needed for loading channel_normalizer from pickle)
preprocessing_path = Path(__file__).parent.parent / 'preprocessing'
if str(preprocessing_path) not in sys.path:
    sys.path.insert(0, str(preprocessing_path))
from preprocessing_normalize import ChannelNormalizer

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'MODEL_KIND': 'fno',
    # Data paths - ensure these match your preprocessing output mode (raw/log/delta)
    'MERGED_PT_PATH': './src/preprocessing/data/normalized/lr/delta/merged_normalized_U.pt',  # Pre-normalized data
    'CHANNEL_NORMALIZER_PATH': './src/preprocessing/normalizers/lr/delta/normalizer_u_delta.pkl',  # Normalizer (must match output mode)
    'OUTPUT_DIR': './src/FNO/output_pure/',
    'N_EPOCHS': 2,  
    'EVAL_INTERVAL': 1,
    'VAL_SIZE': 0.1,  # Validation set size
    'TEST_SIZE': 0.1,  # Test set size
    'RANDOM_STATE': 42,
    'DOMAIN_PADDING_MODE': 'symmetric',
    'MODEL_CONFIG': {
        'in_channels': 11,  # 10 original channels (with material one-hot) + 1 uniform meta channel
        'out_channels': 1,
        'lifting_channel_ratio': 2,
        'projection_channel_ratio': 2,
        'positional_embedding': 'grid',
    },
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'step',  # Options: 'cosine', 'step' (validation plateau)
        'early_stopping': 40,
        'T_0': 10,
        'T_max': 40,
        'T_mult': 2,
        'eta_min': 1e-5,
        'step_size': 10,  # Plateau patience epochs before reducing LR
        'gamma': 0.5,  # LR multiplier after plateau
        'initial_lr': 1e-2,
    },
    'OUTPUT': {
        'ENABLED': True,  # Master switch for all output generation
        'OUTPUT_DIR': './src/FNO/output_pure',  # Base output directory
        'SAMPLE_INDICES': [230],  # Samples to visualize
        'TIME_INDICES': [4, 9, 14, 19],  # Time indices to visualize
        'DPI': 200,  # Resolution for all images

        # Note: NORM_CHECK is now performed in preprocessing_merge.py

        # Image output configuration
        'IMAGE_OUTPUT': {
            'ENABLED': True,  # Generate static images
            'COMBINED_IMG': True,  # 3×4 grid (GT/Pred/Error)
            'SEPARATED_IMG': True,  # Individual images per time/type
        },

        # GIF generation configuration
        'GIF_OUTPUT': {
            'ENABLED': False,  # Generate animated GIFs
            'FPS': 2,  # Frames per second
            # Always uses all time steps (GIF_ALL_TIMES removed)
        },

        # Detailed evaluation configuration
        'DETAIL_EVAL': {
            'ENABLED': False,  # Compute detailed evaluation metrics
            'COMPUTE_RELATIVE_L2': True,  # Compute Relative L2-based metrics
            'COMPUTE_SSIM': True,  # Compute SSIM evolution
            'PARITY_PLOT': True,  # Generate parity plot CSV
            'ADD_MEAN_COLUMN': True,  # Add mean column to CSV
        },

        # Integrated Gradients configuration
        'IG_ANALYSIS': {
            'ENABLED': False,  # Perform IG analysis
            'SAMPLE_IDX': 3,  # Sample to analyze
            'TIME_INDICES': [4, 9, 14, 19],  # Target times
            'N_STEPS': 20,  # Integration steps
            'USE_MULTI_BASELINE': True,  # Use multiple real samples as baselines (instead of mean)
            'N_BASELINES': 1,  # Number of baseline samples (only used if USE_MULTI_BASELINE=True)
            'BASELINE_SEED': 42,  # Random seed for baseline selection (reproducibility)
        },
    },
    'LOSS_CONFIG': {
        'loss_type': 'l2',  # Options: 'l2', 'mse'
        'l2_d': 3,  # Dimension for L2 loss
        'l2_p': 2,  # Power for L2 loss
    },
    'TRAINING_CONFIG': {
        'mode': 'single',  # Options: 'single', 'optuna', 'eval'
        'optuna_n_trials': 3,
        'optuna_n_epochs': 1,
        'optuna_seed': 42,
        'optuna_n_startup_trials': 1,
        'eval_model_path': './src/FNO/output_pure/final/best_model_state_dict.pt'
    },
    'ENSEMBLE': {
        'ENABLED': True,
        'N_MODELS': 3,
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
    'OPTUNA_SEARCH_SPACE': {
        'n_modes_dim1_range': [4, 16],  # [min, max] for first dimension
        'n_modes_dim2_range': [4, 16],  # [min, max] for second dimension
        'n_modes_dim3_range': [2, 10],  # [min, max] for third dimension
        'hidden_channels_range': [12, 36],  # [min, max] for suggest_int
        'n_layers_range': [2, 8],  # [min, max] for suggest_int
        'domain_padding_options': [(0.1,0.1,0.1), (0.2,0.1,0.1)],
        'train_batch_size_options': [32, 64],
        'l2_weight_range': [1e-9, 1e-4],  # [min, max] for log uniform
        'channel_mlp_expansion_options': [0.5, 1.0, 2.0],  # categorical options
        'channel_mlp_skip_options': ['linear', 'soft-gating'],  # categorical options
        'norm_options': [None, 'group_norm'],  # Options: None, GroupNorm via neuraloperator's 'group_norm'
    },
    'SINGLE_PARAMS': {
        "n_modes_1": 8,
        "n_modes_2": 8,
        "n_modes_3": 4,
        "hidden_channels": 12,
        "n_layers": 3,
        "domain_padding": (0.1,0.1,0.1),
        "train_batch_size": 16,
        "l2_weight": 9.338206141357252e-08,
        "channel_mlp_expansion": 1.0,
        "channel_mlp_skip": 'soft-gating',
        "norm": None
    }
}

# ==============================================================================
# Data Classes and Dataset
# ==============================================================================

class CustomDatasetPure(Dataset):
    """Custom dataset for Pure FNO training with meta data already combined as uniform spatial channels.

    Note: This version expects input_tensor to already have meta channels combined,
    unlike the original version that combines them internally.

    Args:
        input_tensor: Combined input tensor of shape (N, original_channels + meta_channels, nx, ny, nt)
        output_tensor: Output tensor of shape (N, 1, nx, ny, nt)
        initial_tensor: Initial values at t=0 of shape (N, 1, nx, ny, 1) - optional, used for delta mode reconstruction
    """

    def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, initial_tensor: torch.Tensor = None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.initial_tensor = initial_tensor

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'x': self.input_tensor[idx],
            'y': self.output_tensor[idx]
        }

        # Add initial values if available (for delta mode reconstruction)
        if self.initial_tensor is not None:
            item['y_initial'] = self.initial_tensor[idx]

        return item

# ==============================================================================
# Data Processing Functions
# ==============================================================================

def preprocessing(config: Dict, verbose: bool = True, return_split: bool = False) -> Tuple:
    """
    Load pre-normalized data and perform train/val/test split.

    Processing Steps:
    1. Load normalized tensors (x, y already include meta channel and are normalized)
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

        required_keys = ["x", "y_u"]
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            raise KeyError(f"Missing required keys in normalized data: {missing_keys}")

        # Data is already normalized and combined (includes meta channel)
        combined_input = bundle["x"].float()   # (N, 11, nx, ny, nt) - already normalized
        out_data = bundle["y_u"].float()          # (N, 1, nx, ny, nt) - already normalized

        # Load initial values if available (for delta mode reconstruction)
        y_initial = None
        if "y_initial" in bundle:
            y_initial = bundle["y_initial"].float()  # (N, 1, nx, ny, 1)
            if verbose:
                print(f"   Loaded initial values for delta mode reconstruction: {tuple(y_initial.shape)}")

        if verbose:
            print(f"   Loaded normalized tensors - Input: {tuple(combined_input.shape)}, Output: {tuple(out_data.shape)}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 1 (loading normalized data): {e}")

    try:
        # Step 2: Load channel-wise normalizer from pickle
        if verbose:
            print("Step 2: Loading channel-wise normalizer from pickle...")

        # Get pickle path from config (or use default location)
        if 'CHANNEL_NORMALIZER_PATH' in config and config['CHANNEL_NORMALIZER_PATH']:
            pickle_path = Path(config['CHANNEL_NORMALIZER_PATH'])
        else:
            # Fallback: same directory as normalized data
            data_path = Path(config['MERGED_PT_PATH'])
            pickle_path = data_path.parent / 'channel_normalizer.pkl'

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Channel normalizer pickle not found: {pickle_path}\n"
                f"Please ensure CHANNEL_NORMALIZER_PATH in CONFIG points to the correct file.\n"
                f"Expected file corresponds to the output mode used during preprocessing."
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

        full_dataset = CustomDatasetPure(combined_input, out_data, y_initial)
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

    # Step 4: Return necessary objects
    return (channel_normalizer, train_dataset, val_dataset, test_dataset, device)

# ==============================================================================
# Note: Loss functions and schedulers moved to util_common.py
# ==============================================================================
# LpLoss, LRStepScheduler, CappedCosineAnnealingWarmRestarts are now imported from util_common


# ==============================================================================
# Model Building Functions
# ==============================================================================

def create_model(config: Dict, train_dataset, val_dataset, test_dataset, device: str,
                n_modes: Tuple[int, ...], hidden_channels: int, n_layers: int,
                domain_padding: List[float], train_batch_size: int,
                l2_weight: float, channel_mlp_expansion: float,
                channel_mlp_skip: str, norm: Optional[str] = None):
    """
    Create complete model setup including DataLoaders, loss function, optimizer, 
    scheduler, and model architecture.
    
    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset  
        device: Device to use (cuda/cpu)
        n_modes: Number of modes for each dimension
        hidden_channels: Number of hidden channels
        n_layers: Number of layers
        domain_padding: Domain padding values
        train_batch_size: Training batch size
        initial_lr: Initial learning rate
        l2_weight: L2 weight regularization
        channel_mlp_expansion: Expansion parameter for channel MLP
        channel_mlp_skip: Skip connection type for channel MLP
        norm: FNO block normalization option (None or 'group_norm')

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
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'l2' or 'mse'.")

    # 3. Create TFNO model
    model = TFNO(
        n_modes=n_modes,
        in_channels=config['MODEL_CONFIG']['in_channels'],
        out_channels=config['MODEL_CONFIG']['out_channels'],
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        lifting_channel_ratio=config['MODEL_CONFIG']['lifting_channel_ratio'],
        projection_channel_ratio=config['MODEL_CONFIG']['projection_channel_ratio'],
        positional_embedding=config['MODEL_CONFIG']['positional_embedding'],
        domain_padding=domain_padding,
        domain_padding_mode=config['DOMAIN_PADDING_MODE'],
        use_channel_mlp=True,
        channel_mlp_expansion=channel_mlp_expansion,
        channel_mlp_skip=channel_mlp_skip,
        norm=norm,
        fno_skip='linear',
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
    """Create FNO model components from canonical or Optuna parameter dictionaries."""
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
        norm=resolved.get('norm'),
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
    """Train one model and save checkpoints/loss curves to the provided directory."""
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
    """Load an ensemble manifest and return a predictor plus test loader/loss."""
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
# Training Functions
# ==============================================================================
# Note: Training logic moved to util_training.py
# Wrapper functions are provided below for compatibility

def train_model(config: Dict, device: str, model, train_loader, val_loader, test_loader,
                optimizer, scheduler, loss_fn, verbose: bool = True):
    """
    Train the FNO model with early stopping and loss tracking.

    This is a wrapper around train_model_generic() from util_training.py.

    Args:
        config: Configuration dictionary
        device: Device to use (cuda/cpu)
        model: TFNO model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (unused, kept for compatibility)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        verbose: Whether to print training progress

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


def model_evaluation(config: Dict, device: str, model, test_loader, loss_fn, verbose: bool = True):
    """
    Evaluate the trained model on test set and print detailed results.

    This is a wrapper around model_evaluation_generic() from util_training.py.

    Args:
        config: Configuration dictionary
        device: Device to use (cuda/cpu)
        model: Trained model to evaluate
        test_loader: Test data loader
        loss_fn: Loss function
        verbose: Whether to print evaluation results

    Returns:
        Dictionary containing evaluation results
    """
    output_dir = Path(config['OUTPUT_DIR']) / 'final'
    compute_mse = isinstance(loss_fn, LpLoss)  # Compute MSE when using LpLoss

    return model_evaluation_generic(
        config=config,
        device=device,
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        output_dir=output_dir,
        compute_mse=compute_mse,
        verbose=verbose
    )

# ==============================================================================
# Optuna Optimization Functions
# ==============================================================================

def optuna_optimization(config: Dict, train_dataset, val_dataset, test_dataset, device: str,
                        verbose: bool = True) -> Dict:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        config: Configuration dictionary
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

        # Sample n_modes for each dimension independently
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

        domain_padding_idx = trial.suggest_categorical('domain_padding_idx', list(range(len(search_space['domain_padding_options']))))
        domain_padding = search_space['domain_padding_options'][domain_padding_idx]
        
        train_batch_size = trial.suggest_categorical('train_batch_size', search_space['train_batch_size_options'])
        
        # Sample integer parameters with ranges
        hidden_channels = trial.suggest_int('hidden_channels', 
                                           search_space['hidden_channels_range'][0], 
                                           search_space['hidden_channels_range'][1])
        n_layers = trial.suggest_int('n_layers', 
                                    search_space['n_layers_range'][0], 
                                    search_space['n_layers_range'][1])
        
        # Sample continuous parameters with log uniform distribution
        l2_weight = trial.suggest_float('l2_weight',
                                       search_space['l2_weight_range'][0],
                                       search_space['l2_weight_range'][1],
                                       log=True)

        # Sample channel MLP parameters
        channel_mlp_expansion = trial.suggest_categorical('channel_mlp_expansion',
                                                          search_space['channel_mlp_expansion_options'])
        channel_mlp_skip = trial.suggest_categorical('channel_mlp_skip',
                                                     search_space['channel_mlp_skip_options'])
        norm = trial.suggest_categorical('norm', search_space['norm_options'])
        
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
                norm=norm
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

            # Clean up GPU memory even on failure
            if 'model' in locals():
                del model
            if 'trained_model' in locals():
                del trained_model
            if 'optimizer' in locals():
                del optimizer
            if 'scheduler' in locals():
                del scheduler
            if device == 'cuda':
                torch.cuda.empty_cache()

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
# Visualization Functions (Simplified - delegates to util_output.py)
# ==============================================================================

def visualization(config: Dict, channel_normalizer, device: str, trained_model, train_dataset,
                 val_dataset, test_dataset, verbose: bool = True):
    """
    Generate all outputs using the unified output system.

    This function is now a simple wrapper that delegates to generate_all_outputs()
    from util_output.py, which handles:
    - Image generation (combined grids and/or separated images)
    - GIF generation for temporal evolution
    - Detailed evaluation metrics (Relative L2, optional SSIM, parity plots)
    - Integrated Gradients analysis

    All outputs are organized into subdirectories based on configuration.

    Args:
        config: Configuration dictionary containing OUTPUT settings
        channel_normalizer: Channel-wise normalizer for inverse transform
        device: Device to use (e.g., 'cuda' or 'cpu')
        trained_model: The trained FNO model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset for generating predictions
        verbose: If True, prints progress information
    """
    # Create test loader
    batch_size = min(8, len(test_dataset))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Call unified output generation
    results = generate_all_outputs(
        config=config,
        channel_normalizer=channel_normalizer,
        device=device,
        trained_model=trained_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        test_loader=test_loader,
        verbose=verbose
    )

    return results

# ==============================================================================
# Utility Functions
# ==============================================================================

def main() -> None:
    """
    Main training pipeline for Pure FNO with uniform meta channels.
    
    Workflow:
    1. Load and preprocess data with unified preprocessing function
    2. Configure training mode (single/optuna/eval)
    3. Execute training pipeline
    4. Generate visualization and save results
    """
    try:
        # Step 1: Unified data preprocessing
        print(f"\nFNO-Pure Training Pipeline Started")
        print(f"Training Mode: {CONFIG['TRAINING_CONFIG']['mode'].upper()}")
        
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
                # Optuna searches only within the train/validation pool. The fixed test set
                # remains held out until final ensemble evaluation.
                optuna_train_dataset, optuna_val_dataset, _ = make_member_split(
                    trainval_dataset=trainval_dataset,
                    trainval_original_indices=fixed_split.trainval_indices,
                    val_size_relative=CONFIG['VAL_SIZE'] / (1 - CONFIG['TEST_SIZE']),
                    random_state=member_seed(CONFIG, 0),
                )
                optimization_results = optuna_optimization(
                    config=CONFIG,
                    train_dataset=optuna_train_dataset,
                    val_dataset=optuna_val_dataset,
                    test_dataset=test_dataset,
                    device=device,
                    verbose=True
                )

            params = resolve_training_params(training_mode, CONFIG, optimization_results)
            print("\nTraining final ensemble with resolved hyperparameters...")

            def create_components(train_ds, val_ds, test_ds, member_params, output_dir):
                return create_model_from_params(CONFIG, train_ds, val_ds, test_ds, device, member_params)

            def train_one(model, train_loader, val_loader, optimizer, scheduler, loss_fn, output_dir):
                return train_model_to_dir(
                    CONFIG, device, model, train_loader, val_loader,
                    optimizer, scheduler, loss_fn, output_dir, verbose=True
                )

            ensemble_run = train_ensemble(
                config=CONFIG,
                model_kind='fno',
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

            model_evaluation(
                config=CONFIG,
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
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            verbose=True
        )

        print("\nTraining pipeline completed successfully!")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main()
