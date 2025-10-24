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
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn
import optuna
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuraloperator.neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuraloperator.neuralop.utils import count_model_params
from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop.training import AdamW

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'MERGED_PT_PATH': './src/preprocessing/merged.pt',
    'OUTPUT_DIR': './src/FNO/output_pure',
    'N_EPOCHS': 10000,  # Reduced for testing
    'EVAL_INTERVAL': 1,
    'VAL_SIZE': 0.1,  # Validation set size
    'TEST_SIZE': 0.1,  # Test set size
    'RANDOM_STATE': 42,
    'DOMAIN_PADDING_MODE': 'symmetric',
    'MODEL_CONFIG': {
        'in_channels': 9,  # 7 original channels + 1 uniform meta channel
        'out_channels': 1,
        'lifting_channel_ratio': 2,
        'projection_channel_ratio': 2,
        'positional_embedding': 'grid',
    },
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'step',  # Options: 'cosine', 'step'
        'early_stopping': 40,
        'T_0': 10,
        'T_max': 40,
        'T_mult': 2,
        'eta_min': 1e-5,
        'step_size': 20,
        'gamma': 0.5,
        'initial_lr': 1e-2,
    },
    'VISUALIZATION': {
        'SAMPLE_NUM': [1, 50, 70, 90, 111],  # Can be single int or list: e.g., [121, 122, 123]
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
        'mode': 'eval',  # Options: 'single', 'optuna', 'eval'
        'optuna_n_trials': 100,
        'optuna_seed': 42,
        'optuna_n_startup_trials': 10,
        'eval_model_path': './src/FNO/output_pure/final/best_model_state_dict.pt'
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
        'channel_mlp_skip_options': ['linear', 'soft-gating']  # categorical options
    },
    'SINGLE_PARAMS': {
        "n_modes_1": 11,
        "n_modes_2": 10,
        "n_modes_3": 5,
        "hidden_channels": 47,
        "n_layers": 8,
        "domain_padding": (0.1,0.1,0.1),
        "train_batch_size": 32,
        "l2_weight": 3.4896738256925045e-06,
        "channel_mlp_expansion": 0.5,
        "channel_mlp_skip": 'soft-gating'
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

def preprocessing(config: Dict, verbose: bool = True) -> Tuple:
    """
    Unified data preprocessing function for Pure FNO training.
    
    Processing Steps:
    1. Load saved tensors (x, y, meta)
    2. Transform meta data to uniform channels and combine with input
    3. Create normalizers and fit them, form DefaultDataProcessor
    4. Perform train/val/test split and create datasets
    5. Return necessary objects for training
    
    Args:
        config: Configuration dictionary containing paths and parameters
        verbose: Whether to print progress information
        
    Returns:
        Tuple containing (processor, train_dataset, val_dataset, test_dataset, device)
    """
    
    if verbose:
        print(f"\nStarting unified data preprocessing...")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using device: {device}")
    
    try:
        # Step 1: Load saved tensors
        if verbose:
            print("Step 1: Loading merged tensors...")
            
        if not Path(config['MERGED_PT_PATH']).exists():
            raise FileNotFoundError(f"Merged file not found: {config['MERGED_PT_PATH']}")
            
        bundle = torch.load(config['MERGED_PT_PATH'], map_location="cpu", weights_only=False)

        required_keys = ["x", "y", "meta"]
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            raise KeyError(f"Missing required keys in data: {missing_keys}")

        # Keep data on CPU to avoid VRAM issues during preprocessing
        in_data = bundle["x"].float()
        out_data = bundle["y"].float()
        meta_data = bundle["meta"].float()
        
        if verbose:
            print(f"   Loaded tensors - Input: {tuple(in_data.shape)}, Output: {tuple(out_data.shape)}, Meta: {tuple(meta_data.shape)}")
            
    except Exception as e:
        raise RuntimeError(f"Failed at Step 1 (tensor loading): {e}")
    
    try:
        # Step 2: Transform meta data to uniform channels and combine
        if verbose:
            print("Step 2: Expanding meta channels and combining with input...")
            
        # Apply output masking as in original code
        # out_data[:, :, 12:20, 12:20, :] = 0
        
        # Expand meta data to uniform spatial channels and combine with input
        N, original_channels, nx, ny, nt = in_data.shape
        
        # Handle both 1D and 2D meta tensors
        if len(meta_data.shape) == 1:
            meta_data = meta_data.unsqueeze(1)  # (N,) -> (N, 1)
            
        N_meta, meta_channels = meta_data.shape
        
        if N != N_meta:
            raise ValueError(f"Batch size mismatch: input_tensor {N}, meta_tensor {N_meta}")
        
        # Expand meta tensor to match spatial dimensions
        # Shape: (N, meta_channels) -> (N, meta_channels, nx, ny, nt)
        expanded_meta = meta_data.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (N, meta_channels, 1, 1, 1)
        expanded_meta = expanded_meta.expand(N, meta_channels, nx, ny, nt)   # (N, meta_channels, nx, ny, nt)
        
        # Concatenate along channel dimension
        combined_input = torch.cat([in_data, expanded_meta], dim=1)
        
        if verbose:
            print(f"   Combined input shape: {tuple(combined_input.shape)}")
            
    except Exception as e:
        raise RuntimeError(f"Failed at Step 2 (meta channel expansion): {e}")
    
    try:
        # Step 3: Create normalizers and processor
        if verbose:
            print("Step 3: Creating normalizers and data processor...")

        # Create normalizers for combined input and output (on CPU)
        in_normalizer = UnitGaussianNormalizer(
            mean=combined_input, std=combined_input, dim=[0,2,3,4], eps=1e-6
        )
        out_normalizer = UnitGaussianNormalizer(
            mean=out_data, std=out_data, dim=[0,2,3,4], eps=1e-6
        )

        # Fit normalizers on CPU data
        in_normalizer.fit(combined_input)
        out_normalizer.fit(out_data)

        # Create processor and move only the processor to device
        processor = DefaultDataProcessor(in_normalizer, out_normalizer).to(device)
        
        if verbose:
            print("   Normalizers and processor created successfully")
            
    except Exception as e:
        raise RuntimeError(f"Failed at Step 3 (normalizer creation): {e}")
    
    try:
        # Step 4: Perform train/val/test split and create datasets
        if verbose:
            print("Step 4: Creating train/val/test datasets...")
            
        # First split: separate test set (final 10%)
        train_temp_combined, test_combined, train_temp_out, test_out = train_test_split(
            combined_input, out_data,
            test_size=config['TEST_SIZE'], 
            random_state=config['RANDOM_STATE']
        )
        
        # Second split: separate validation set from remaining data
        # Val size relative to remaining data: 0.1 / (1 - 0.1) = ~0.111
        val_size_relative = config['VAL_SIZE'] / (1 - config['TEST_SIZE'])
        train_combined, val_combined, train_out, val_out = train_test_split(
            train_temp_combined, train_temp_out,
            test_size=val_size_relative,
            random_state=config['RANDOM_STATE']
        )
        
        # Create datasets with already combined inputs
        train_dataset = CustomDatasetPure(train_combined, train_out)
        val_dataset = CustomDatasetPure(val_combined, val_out)
        test_dataset = CustomDatasetPure(test_combined, test_out)
        
        if verbose:
            print(f"   Train dataset size: {len(train_dataset)}")
            print(f"   Validation dataset size: {len(val_dataset)}")
            print(f"   Test dataset size: {len(test_dataset)}")
            total_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
            print(f"   Split ratios: Train {len(train_dataset)/total_size:.1%}, Val {len(val_dataset)/total_size:.1%}, Test {len(test_dataset)/total_size:.1%}")
            
    except Exception as e:
        raise RuntimeError(f"Failed at Step 4 (dataset creation): {e}")
    
    if verbose:
        print("Data preprocessing completed successfully!")
    
    # Step 5: Return necessary objects
    return (processor, train_dataset, val_dataset, test_dataset, device)

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
# Model Building Functions
# ==============================================================================

def create_model(config: Dict, train_dataset, val_dataset, test_dataset, device: str,
                n_modes: Tuple[int, ...], hidden_channels: int, n_layers: int,
                domain_padding: List[float], train_batch_size: int,
                l2_weight: float, channel_mlp_expansion: float,
                channel_mlp_skip: str):
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
        fno_skip='linear'
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

# ==============================================================================
# Training Functions
# ==============================================================================

def train_model(config: Dict, processor, device: str, model, train_loader, val_loader, test_loader, 
                optimizer, scheduler, loss_fn, verbose: bool = True):
    """
    Train the FNO model with early stopping and loss tracking.
    
    Args:
        config: Configuration dictionary
        processor: Data processor for normalization
        device: Device to use (cuda/cpu)
        model: TFNO model to train
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
    
    if verbose:
        print(f"\nStarting model training for {config['N_EPOCHS']} epochs...")
    
    # Training setup
    best_val_loss = float('inf')
    patience = 0
    early_stopping_patience = config['SCHEDULER_CONFIG']['early_stopping']
    
    # Track losses for each epoch
    train_losses = []
    val_losses = []
    
    # Create output directory for saving best model
    output_dir = Path(config['OUTPUT_DIR']) / 'final'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['N_EPOCHS']):
        # Training phase
        model.train()
        total_train_loss = 0
        train_count = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            # Apply input and output normalization for consistent training
            x = processor.in_normalizer.transform(x)
            y = processor.out_normalizer.transform(y)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_count += 1
        
        train_loss = total_train_loss / train_count
        train_losses.append(train_loss)
        
        # Validation phase - compute validation loss every epoch
        model.eval()
        total_val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                x = processor.in_normalizer.transform(x)
                y = processor.out_normalizer.transform(y)
                
                pred = model(x)
                loss = loss_fn(pred, y)
                total_val_loss += loss.item()
                val_count += 1
        
        val_loss = total_val_loss / val_count
        val_losses.append(val_loss)
        
        # Print losses for every epoch
        if verbose:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Save best model based on validation loss
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
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    epochs_range = range(len(train_losses))
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FNO Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    loss_plot_path = output_dir / 'loss_curves.png'
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"\nTraining completed!")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Loss history saved to: {output_dir / 'loss_history.pt'}")
        print(f"Loss curves saved to: {loss_plot_path}")
    
    # Load and return best model
    model.load_state_dict(torch.load(output_dir / 'best_model_state_dict.pt', map_location=device, weights_only=False))
    return model


def model_evaluation(config: Dict, processor, device: str, model, test_loader, loss_fn, verbose: bool = True):
    """
    Evaluate the trained model on test set and print detailed results.
    
    Args:
        config: Configuration dictionary
        processor: Data processor for normalization
        device: Device to use (cuda/cpu)
        model: Trained model to evaluate
        test_loader: Test data loader
        loss_fn: Loss function
        verbose: Whether to print evaluation results
        
    Returns:
        Dictionary containing evaluation results
    """
    
    if verbose:
        print(f"\nEvaluating model on test set...")
    
    # Test evaluation
    model.eval()
    total_test_loss = 0
    total_test_mse_loss = 0
    test_count = 0
    
    # Create MSE loss function for additional metric when using LpLoss
    mse_loss_fn = torch.nn.MSELoss() if isinstance(loss_fn, LpLoss) else None
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            x = processor.in_normalizer.transform(x)
            y = processor.out_normalizer.transform(y)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            total_test_loss += loss.item()
            
            # Calculate MSE loss additionally when using LpLoss
            if mse_loss_fn is not None:
                mse_loss = mse_loss_fn(pred, y)
                total_test_mse_loss += mse_loss.item()
            
            test_count += 1
    
    final_test_loss = total_test_loss / test_count
    final_test_mse_loss = total_test_mse_loss / test_count if mse_loss_fn is not None else None
    
    # Create evaluation results
    eval_results = {
        'test_loss': final_test_loss,
        'test_mse_loss': final_test_mse_loss if final_test_mse_loss is not None else None
    }
    
    # Save evaluation results
    output_dir = Path(config['OUTPUT_DIR']) / 'final'
    eval_results_path = output_dir / 'evaluation_results.pt'
    torch.save(eval_results, eval_results_path)
    
    if verbose:
        print(f"Model Evaluation Results:")
        loss_type = config['LOSS_CONFIG']['loss_type']
        if loss_type == 'l2' and final_test_mse_loss is not None:
            print(f"  Test Loss (L{config['LOSS_CONFIG']['l2_p']}): {final_test_loss:.6f}")
            print(f"  Test Loss (MSE): {final_test_mse_loss:.6f}")
        else:
            print(f"  Test Loss: {final_test_loss:.6f}")
        print(f"Evaluation results saved to: {eval_results_path}")
    
    return eval_results

# ==============================================================================
# Optuna Optimization Functions
# ==============================================================================

def optuna_optimization(config: Dict, processor, train_dataset, val_dataset, test_dataset, device: str, 
                        verbose: bool = True) -> Dict:
    """
    Perform hyperparameter optimization using Optuna.
    
    Args:
        config: Configuration dictionary
        processor: Data processor for normalization
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
                initial_lr=config['SCHEDULER_CONFIG']['initial_lr'],
                l2_weight=l2_weight,
                channel_mlp_expansion=channel_mlp_expansion,
                channel_mlp_skip=channel_mlp_skip
            )
            
            # Train model and get best validation loss
            trained_model = train_model(
                config=config,
                processor=processor,
                device=device,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                verbose=False  # Reduce verbosity during optimization
            )
            
            # Get best validation loss from training history
            loss_history_path = Path(config['OUTPUT_DIR']) / 'final' / 'loss_history.pt'
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
            source_model_path = Path(config['OUTPUT_DIR']) / 'final' / 'best_model_state_dict.pt'
            source_loss_path = Path(config['OUTPUT_DIR']) / 'final' / 'loss_history.pt'

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

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def visualize_single_sample(config: Dict, processor, device: str, trained_model,
                           pred_phys: torch.Tensor, gt_phys: torch.Tensor,
                           input_phys: torch.Tensor, sample_idx: int,
                           verbose: bool = True) -> Dict:
    """
    Generate visualization for a single sample.

    Args:
        config: Configuration dictionary.
        processor: Data processor for normalization.
        device: Device to use (e.g., 'cuda' or 'cpu').
        trained_model: The trained FNO model.
        pred_phys: Physical scale predictions tensor.
        gt_phys: Ground truth tensor.
        input_phys: Input tensor.
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
    output_path_inputs = Path(config['OUTPUT_DIR']) / f'FNO_input_visualization_sample_{sample_idx}.png'
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
    output_path_grid = Path(config['OUTPUT_DIR']) / f'FNO_comparison_grid_sample_{sample_idx}.png'
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


def visualization(config: Dict, processor, device: str, trained_model, train_dataset,
                 test_dataset, verbose: bool = True):
    """
    Generate visualizations for multiple samples:
    1. A separate plot for permeability and pyrite maps.
    2. A 3x4 grid comparing ground truth, predictions, and their error over time.
    3. Unified CSV output for all samples (Option B).

    Args:
        config: Configuration dictionary.
        processor: Data processor for normalization.
        device: Device to use (e.g., 'cuda' or 'cpu').
        trained_model: The trained FNO model.
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

            # Store original input immediately moved to CPU to free GPU memory
            all_input.append(x.cpu())

            # Normalize input for the model
            x_norm = processor.in_normalizer.transform(x)

            # Get model prediction
            pred = trained_model(x_norm)

            # Inverse transform the prediction to its physical scale
            pred_phys = processor.out_normalizer.inverse_transform(pred)

            # Move to CPU immediately and clear GPU memory
            all_pred.append(pred_phys.cpu())
            all_gt.append(y.cpu())

            # Clear intermediate GPU tensors
            del x, y, x_norm, pred, pred_phys
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
            processor=processor,
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
        csv_output_path = Path(config['OUTPUT_DIR']) / 'FNO_visualization_data.csv'
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
        
        processor, train_dataset, val_dataset, test_dataset, device = preprocessing(
            config=CONFIG, 
            verbose=True
        )
        
        # Step 2: Execute based on training mode
        training_mode = CONFIG['TRAINING_CONFIG']['mode']
        
        if training_mode == 'single':
            # Single training mode - use predefined parameters
            print("\nExecuting single training mode...")

            params = CONFIG['SINGLE_PARAMS']

            # Reconstruct n_modes from individual dimension parameters
            n_modes = (params['n_modes_1'], params['n_modes_2'], params['n_modes_3'])

            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=CONFIG,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                n_modes=n_modes,
                hidden_channels=params['hidden_channels'],
                n_layers=params['n_layers'],
                domain_padding=params['domain_padding'],
                train_batch_size=params['train_batch_size'],
                l2_weight=params['l2_weight'],
                channel_mlp_expansion=params['channel_mlp_expansion'],
                channel_mlp_skip=params['channel_mlp_skip']
            )
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"   Model created - Device: {device}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
            print(f"   Optimizer: {type(optimizer).__name__}")
            print(f"   Scheduler: {type(scheduler).__name__}")
            print(f"   Loss function: {type(loss_fn).__name__}")
            print(f"   Processor: {type(processor).__name__}")
            
            # Train the model
            trained_model = train_model(
                config=CONFIG,
                processor=processor,
                device=device,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                verbose=True
            )
            
            # Evaluate the model
            model_evaluation(
                config=CONFIG,
                processor=processor,
                device=device,
                model=trained_model,
                test_loader=test_loader,
                loss_fn=loss_fn,
                verbose=True
            )
            
        elif training_mode == 'optuna':
            # Optuna optimization mode
            print("\nExecuting Optuna optimization mode...")
            
            # Run hyperparameter optimization
            optimization_results = optuna_optimization(
                config=CONFIG,
                processor=processor,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                verbose=True
            )
            
            # Train final model with best parameters
            print(f"\nTraining final model with best parameters...")
            best_params = optimization_results['best_params']

            # Reconstruct n_modes from individual dimension parameters
            n_modes = (best_params['n_modes_1'], best_params['n_modes_2'], best_params['n_modes_3'])

            # Convert index-based parameters back to actual values
            search_space = CONFIG['OPTUNA_SEARCH_SPACE']
            domain_padding = search_space['domain_padding_options'][best_params['domain_padding_idx']]

            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=CONFIG,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                n_modes=n_modes,
                hidden_channels=best_params['hidden_channels'],
                n_layers=best_params['n_layers'],
                domain_padding=domain_padding,
                train_batch_size=best_params['train_batch_size'],
                l2_weight=best_params['l2_weight'],
                channel_mlp_expansion=best_params['channel_mlp_expansion'],
                channel_mlp_skip=best_params['channel_mlp_skip']
            )
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"   Final model created - Device: {device}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Total parameters: {total_params:,}")
            
            # Train final model with best parameters
            trained_model = train_model(
                config=CONFIG,
                processor=processor,
                device=device,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                verbose=True
            )
            
            # Evaluate the final model
            model_evaluation(
                config=CONFIG,
                processor=processor,
                device=device,
                model=trained_model,
                test_loader=test_loader,
                loss_fn=loss_fn,
                verbose=True
            )
            
        elif training_mode == 'eval':
            # Evaluation mode - load pretrained model
            print("\nExecuting evaluation mode...")

            eval_model_path = CONFIG['TRAINING_CONFIG']['eval_model_path']
            if not Path(eval_model_path).exists():
                raise FileNotFoundError(f"Model file not found: {eval_model_path}")

            # Create model with single params for evaluation
            params = CONFIG['SINGLE_PARAMS']

            # Reconstruct n_modes from individual dimension parameters
            n_modes = (params['n_modes_1'], params['n_modes_2'], params['n_modes_3'])

            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=CONFIG,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                n_modes=n_modes,
                hidden_channels=params['hidden_channels'],
                n_layers=params['n_layers'],
                domain_padding=params['domain_padding'],
                train_batch_size=params['train_batch_size'],
                l2_weight=params['l2_weight'],
                channel_mlp_expansion=params['channel_mlp_expansion'],
                channel_mlp_skip=params['channel_mlp_skip']
            )
            
            # Load pretrained model
            model.load_state_dict(torch.load(eval_model_path, map_location=device, weights_only=False))
            print(f"   Loaded model from: {eval_model_path}")
            
            # Set as trained model for visualization
            trained_model = model
            
            # Evaluate the loaded model
            model_evaluation(
                config=CONFIG,
                processor=processor,
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
            processor=processor,
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