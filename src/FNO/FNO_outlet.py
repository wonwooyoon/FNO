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

import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from neuraloperator.neuralop.utils import count_model_params
from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop.training import AdamW
from neuraloperator.neuralop.layers.channel_mlp import ChannelMLP

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
    # Data paths
    'MERGED_PT_PATH': './src/preprocessing/merged_normalized.pt',
    'SPATIAL_NORMALIZER_PATH': './src/preprocessing/normalizer_delta.pkl',
    'OUTLET_NORMALIZER_PATH': './src/preprocessing/normalizer_out_delta.pkl',
    'OUTPUT_DIR': './src/FNO/output_outlet/',

    # Training parameters
    'N_EPOCHS': 50,
    'VAL_SIZE': 0.1,
    'TEST_SIZE': 0.1,
    'RANDOM_STATE': 42,
    'DOMAIN_PADDING_MODE': 'symmetric',

    # Model configuration
    'MODEL_CONFIG': {
        'in_channels': 11,  # 10 original channels + 1 meta channel
        'pool_type': 'adaptive_avg',  # Options: 'adaptive_avg', 'adaptive_max'
        'lifting_channel_ratio': 2,
        'projection_channel_ratio': 2,
        'positional_embedding': 'grid',

        # ChannelMLP configuration (for C → 1 projection while preserving time dimension)
        # This replaces the old MLP head that lost temporal information
        'channel_mlp_config': {
            'hidden_channels': 128,  # Hidden dimension for ChannelMLP
            'n_layers': 3,           # Number of Conv1d layers
            'activation': 'gelu',    # Options: 'gelu', 'relu', 'silu', 'tanh'
            'dropout': 0.0,          # Dropout probability
        },
    },

    # Scheduler configuration
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'step',
        'early_stopping': 20,
        'step_size': 10,
        'gamma': 0.5,
        'initial_lr': 1e-2,
    },

    # Loss configuration
    'LOSS_CONFIG': {
        'loss_type': 'mse',  # MSE for vector outputs
    },

    # Training mode
    'TRAINING_CONFIG': {
        'mode': 'single',  # Options: 'single', 'eval'
        'eval_model_path': './src/FNO/output_outlet/final/best_model_state_dict.pt'
    },

    # Single training parameters
    'SINGLE_PARAMS': {
        "n_modes_1": 8,
        "n_modes_2": 4,
        "n_modes_3": 4,
        "hidden_channels": 12,
        "n_layers": 3,
        "domain_padding": (0.1, 0.1, 0.1),
        "train_batch_size": 32,
        "l2_weight": 0.0,
        "channel_mlp_expansion": 1.0,
        "channel_mlp_skip": 'soft-gating'
    }
}

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

def preprocessing_outlet(config: Dict, verbose: bool = True) -> Tuple:
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

        # First split: separate test set
        train_temp_x, test_x, train_temp_outlet, test_outlet = train_test_split(
            x_input, y_outlet,
            test_size=config['TEST_SIZE'],
            random_state=config['RANDOM_STATE']
        )

        # Second split: separate validation set
        val_size_relative = config['VAL_SIZE'] / (1 - config['TEST_SIZE'])
        train_x, val_x, train_outlet, val_outlet = train_test_split(
            train_temp_x, train_temp_outlet,
            test_size=val_size_relative,
            random_state=config['RANDOM_STATE']
        )

        # Create datasets
        train_dataset = CustomDatasetOutlet(train_x, train_outlet)
        val_dataset = CustomDatasetOutlet(val_x, val_outlet)
        test_dataset = CustomDatasetOutlet(test_x, test_outlet)

        if verbose:
            print(f"   Train dataset size: {len(train_dataset)}")
            print(f"   Validation dataset size: {len(val_dataset)}")
            print(f"   Test dataset size: {len(test_dataset)}")

    except Exception as e:
        raise RuntimeError(f"Failed at Step 3 (dataset creation): {e}")

    if verbose:
        print("Data preprocessing completed successfully!")

    return (spatial_normalizer, outlet_normalizer,
            train_dataset, val_dataset, test_dataset, device)


# ==============================================================================
# Scheduler Classes
# ==============================================================================

class LRStepScheduler(torch.optim.lr_scheduler.StepLR):
    """Learning rate step scheduler wrapper."""

    def __init__(self, optimizer: torch.optim.Optimizer, step_size: int,
                 gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, step_size, gamma, last_epoch)


# ==============================================================================
# Model Building Functions
# ==============================================================================

def create_model(config: Dict, train_dataset, val_dataset, test_dataset, device: str,
                n_modes: Tuple[int, ...], hidden_channels: int, n_layers: int,
                domain_padding: List[float], train_batch_size: int,
                l2_weight: float, channel_mlp_expansion: float,
                channel_mlp_skip: str):
    """
    Create complete model setup for outlet prediction.

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

    # 4. Create TFNOWithPooling model
    model = TFNOWithPooling(
        base_tfno_config=base_tfno_config,
        pool_type=config['MODEL_CONFIG']['pool_type'],
        channel_mlp_config=config['MODEL_CONFIG']['channel_mlp_config']
    ).to(device)

    # 5. Create optimizer
    optimizer = AdamW(model.parameters(),
                     lr=config['SCHEDULER_CONFIG']['initial_lr'],
                     weight_decay=l2_weight)

    # 6. Create scheduler
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


# ==============================================================================
# Training Functions
# ==============================================================================

def train_model(config: Dict, device: str, model, train_loader, val_loader, test_loader,
                optimizer, scheduler, loss_fn, verbose: bool = True):
    """
    Train the outlet prediction model with early stopping.

    Returns:
        Trained model
    """

    if verbose:
        print(f"\nStarting model training for {config['N_EPOCHS']} epochs...")

    # Training setup
    best_val_loss = float('inf')
    patience = 0
    early_stopping_patience = config['SCHEDULER_CONFIG']['early_stopping']

    # Track losses
    train_losses = []
    val_losses = []

    # Create output directory
    output_dir = Path(config['OUTPUT_DIR']) / 'final'
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['N_EPOCHS']):
        # Training phase
        model.train()
        total_train_loss = 0
        train_count = 0

        for batch in train_loader:
            x = batch['x'].to(device)  # (B, 11, nx, ny, nt)
            y_outlet = batch['y'].to(device)  # (B, nt)

            optimizer.zero_grad()
            pred_outlet = model(x)  # (B, nt)
            loss = loss_fn(pred_outlet, y_outlet)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_count += 1

        train_loss = total_train_loss / train_count
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y_outlet = batch['y'].to(device)

                pred_outlet = model(x)
                loss = loss_fn(pred_outlet, y_outlet)
                total_val_loss += loss.item()
                val_count += 1

        val_loss = total_val_loss / val_count
        val_losses.append(val_loss)

        # Print progress
        if verbose:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model_state_dict.pt')
            patience = 0
            if verbose:
                print(f"    New best model saved! Val loss: {val_loss:.6f}")
        else:
            patience += 1

        # Early stopping
        if patience >= early_stopping_patience:
            if verbose:
                print(f"Early stopping after {epoch} epochs")
            break

        # Update learning rate
        scheduler.step()

    # Save loss history
    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': list(range(len(train_losses)))
    }
    torch.save(loss_history, output_dir / 'loss_history.pt')

    if verbose:
        print(f"\nTraining completed!")
        print(f"Best Validation Loss: {best_val_loss:.6f}")

    # Load and return best model
    model.load_state_dict(torch.load(output_dir / 'best_model_state_dict.pt',
                                     map_location=device, weights_only=False))
    return model


def model_evaluation(config: Dict, device: str, model, test_loader, loss_fn,
                    verbose: bool = True):
    """
    Evaluate the trained model on test set.

    Returns:
        Dictionary containing evaluation results
    """

    if verbose:
        print(f"\nEvaluating model on test set...")

    model.eval()
    total_test_loss = 0
    test_count = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y_outlet = batch['y'].to(device)

            pred_outlet = model(x)
            loss = loss_fn(pred_outlet, y_outlet)
            total_test_loss += loss.item()
            test_count += 1

    test_loss = total_test_loss / test_count

    if verbose:
        print(f"Test Loss (MSE): {test_loss:.6f}")

    return {'test_loss': test_loss}


# ==============================================================================
# Visualization Functions
# ==============================================================================

def visualize_outlet_predictions(config: Dict, device: str, model, test_loader,
                                 outlet_normalizer, sample_indices: List[int] = [0, 1, 2, 3]):
    """
    Visualize outlet predictions vs ground truth for selected samples.

    Args:
        config: Configuration dictionary
        device: Device to use
        model: Trained model
        test_loader: Test data loader
        outlet_normalizer: Outlet normalizer for denormalization
        sample_indices: List of sample indices to visualize
    """

    print(f"\nGenerating outlet prediction visualizations...")

    # Create output directory
    output_dir = Path(config['OUTPUT_DIR']) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all test predictions
    model.eval()
    all_x = []
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y_outlet = batch['y'].to(device)

            pred_outlet = model(x)

            all_x.append(x.cpu())
            all_y_true.append(y_outlet.cpu())
            all_y_pred.append(pred_outlet.cpu())

    # Concatenate all batches
    all_x = torch.cat(all_x, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)

    # Denormalize outlet data
    all_y_true_denorm = outlet_normalizer.inverse_transform(all_y_true)
    all_y_pred_denorm = outlet_normalizer.inverse_transform(all_y_pred)

    # Convert to numpy and take absolute values for plotting (original values are negative)
    all_y_true_denorm = torch.abs(all_y_true_denorm).numpy()
    all_y_pred_denorm = torch.abs(all_y_pred_denorm).numpy()

    # Time points (years): [100, 200, ..., 2000] (20 points, t=0 removed during normalization)
    time_points = np.arange(100, 2001, 100)

    # 1. Individual sample plots (16 samples in 4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.flatten()

    for i, sample_idx in enumerate(sample_indices[:16]):
        if sample_idx >= len(all_y_true_denorm):
            print(f"Warning: sample_idx {sample_idx} out of range, skipping")
            continue

        ax = axes[i]
        y_true = all_y_true_denorm[sample_idx]
        y_pred = all_y_pred_denorm[sample_idx]

        ax.plot(time_points, y_true, 'o-', label='Ground Truth', linewidth=2, markersize=6)
        ax.plot(time_points, y_pred, 's--', label='Prediction', linewidth=2, markersize=5, alpha=0.8)

        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Outlet UO2++ [mol]', fontsize=11)
        ax.set_title(f'Sample {sample_idx}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plot_path = output_dir / 'outlet_predictions_samples.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")

    # 2. Error analysis plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Compute errors
    abs_errors = np.abs(all_y_pred_denorm - all_y_true_denorm)
    rel_errors = abs_errors / (all_y_true_denorm + 1e-20)

    # 2a. Mean absolute error over time
    mean_abs_error = abs_errors.mean(axis=0)
    std_abs_error = abs_errors.std(axis=0)

    axes[0].plot(time_points, mean_abs_error, 'o-', linewidth=2, markersize=6, color='crimson')
    axes[0].fill_between(time_points,
                         mean_abs_error - std_abs_error,
                         mean_abs_error + std_abs_error,
                         alpha=0.3, color='crimson')
    axes[0].set_xlabel('Time (years)', fontsize=11)
    axes[0].set_ylabel('Mean Absolute Error [mol]', fontsize=11)
    axes[0].set_title('Absolute Error vs Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 2b. Mean relative error over time
    mean_rel_error = rel_errors.mean(axis=0)
    std_rel_error = rel_errors.std(axis=0)

    axes[1].plot(time_points, mean_rel_error, 'o-', linewidth=2, markersize=6, color='steelblue')
    axes[1].fill_between(time_points,
                         mean_rel_error - std_rel_error,
                         mean_rel_error + std_rel_error,
                         alpha=0.3, color='steelblue')
    axes[1].set_xlabel('Time (years)', fontsize=11)
    axes[1].set_ylabel('Mean Relative Error', fontsize=11)
    axes[1].set_title('Relative Error vs Time', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 2c. Parity plot (all time points combined)
    axes[2].scatter(all_y_true_denorm.flatten(), all_y_pred_denorm.flatten(),
                    alpha=0.3, s=10, color='darkgreen')

    # Add diagonal line
    min_val = min(all_y_true_denorm.min(), all_y_pred_denorm.min())
    max_val = max(all_y_true_denorm.max(), all_y_pred_denorm.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    axes[2].set_xlabel('Ground Truth [mol]', fontsize=11)
    axes[2].set_ylabel('Prediction [mol]', fontsize=11)
    axes[2].set_title('Parity Plot (All Times)', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    plt.tight_layout()
    error_plot_path = output_dir / 'outlet_error_analysis.png'
    plt.savefig(error_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {error_plot_path}")

    # 3. Compute and save statistics
    stats = {
        'mean_absolute_error': float(abs_errors.mean()),
        'std_absolute_error': float(abs_errors.std()),
        'mean_relative_error': float(rel_errors.mean()),
        'std_relative_error': float(rel_errors.std()),
        'max_absolute_error': float(abs_errors.max()),
        'max_relative_error': float(rel_errors.max()),
    }

    stats_path = output_dir / 'prediction_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("Outlet Prediction Statistics\n")
        f.write("="*50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.6e}\n")

    print(f"  ✓ Saved: {stats_path}")
    print(f"\nVisualization complete!")
    print(f"  Mean Absolute Error: {stats['mean_absolute_error']:.6e} mol")
    print(f"  Mean Relative Error: {stats['mean_relative_error']:.4f}")

    return stats


# ==============================================================================
# Main Workflow
# ==============================================================================

def main():
    """Main workflow for outlet prediction model training."""

    print("="*80)
    print("FNO Outlet Prediction Training")
    print("="*80)

    # 1. Load data and create datasets
    (spatial_normalizer, outlet_normalizer,
     train_dataset, val_dataset, test_dataset, device) = preprocessing_outlet(CONFIG)

    # 2. Extract single training parameters
    params = CONFIG['SINGLE_PARAMS']
    n_modes = (params['n_modes_1'], params['n_modes_2'], params['n_modes_3'])

    # 3. Create model and training components
    (model, train_loader, val_loader, test_loader,
     optimizer, scheduler, loss_fn) = create_model(
        CONFIG, train_dataset, val_dataset, test_dataset, device,
        n_modes=n_modes,
        hidden_channels=params['hidden_channels'],
        n_layers=params['n_layers'],
        domain_padding=params['domain_padding'],
        train_batch_size=params['train_batch_size'],
        l2_weight=params['l2_weight'],
        channel_mlp_expansion=params['channel_mlp_expansion'],
        channel_mlp_skip=params['channel_mlp_skip']
    )

    # Print model info
    n_params = count_model_params(model)
    print(f"\nModel created successfully!")
    print(f"Total parameters: {n_params:,}")
    print(f"Architecture: TFNOWithPooling with {CONFIG['MODEL_CONFIG']['pool_type']} spatial pooling")
    print(f"Output: Dimension-independent (B, nt) - works with any time resolution!")

    # Print ChannelMLP configuration
    cmlp_cfg = CONFIG['MODEL_CONFIG']['channel_mlp_config']
    print(f"\nChannelMLP Configuration:")
    print(f"  Hidden channels: {cmlp_cfg['hidden_channels']}")
    print(f"  N layers: {cmlp_cfg['n_layers']}")
    print(f"  Activation: {cmlp_cfg['activation']}")
    print(f"  Dropout: {cmlp_cfg['dropout']}")
    print(f"  Projection: C={params['hidden_channels']} → 1 (time preserved)")

    # 4. Train model
    if CONFIG['TRAINING_CONFIG']['mode'] == 'single':
        trained_model = train_model(
            CONFIG, device, model, train_loader, val_loader, test_loader,
            optimizer, scheduler, loss_fn, verbose=True
        )

        # 5. Evaluate on test set
        eval_results = model_evaluation(
            CONFIG, device, trained_model, test_loader, loss_fn, verbose=True
        )

        # 6. Generate visualizations
        viz_stats = visualize_outlet_predictions(
            CONFIG, device, trained_model, test_loader, outlet_normalizer,
            sample_indices=np.random.randint(0, 150, size=16)
        )

        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)

    elif CONFIG['TRAINING_CONFIG']['mode'] == 'eval':
        # Load pretrained model for evaluation only
        model_path = Path(CONFIG['TRAINING_CONFIG']['eval_model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"Loaded pretrained model from: {model_path}")

        eval_results = model_evaluation(
            CONFIG, device, model, test_loader, loss_fn, verbose=True
        )

        # Generate visualizations for pretrained model
        viz_stats = visualize_outlet_predictions(
            CONFIG, device, model, test_loader, outlet_normalizer,
            sample_indices=[0, 1, 2, 3]
        )

    else:
        raise ValueError(f"Unknown training mode: {CONFIG['TRAINING_CONFIG']['mode']}")


if __name__ == '__main__':
    main()
