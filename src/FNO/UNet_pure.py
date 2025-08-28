"""
Pure U-Net with Uniform Distribution Meta Training - Based on FNO_pure_ref.py

This module implements training pipeline for Pure U-Net models with meta data incorporated 
as uniform spatial channels. Meta data (e.g., permeability, porosity) is expanded to 
uniform distribution across spatial dimensions and directly concatenated with input channels, 
providing a straightforward approach to conditional neural operators.

Based on FNO_pure_ref.py but adapted for 3D U-Net architecture:
- Keep code simple and readable
- Use functions for repetitive tasks
- Add comments to outline workflow in main()
"""

import sys
sys.path.append('./')

import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
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
from neuraloperator.neuralop.training import AdamW

# Import our custom U-Net model
from models.unet3d import UNet3D

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'MERGED_PT_PATH': './src/preprocessing/merged.pt',
    'OUTPUT_DIR': './src/FNO/output_unet',
    'N_EPOCHS': 10000,  # Reduced for testing
    'EVAL_INTERVAL': 1,
    'VAL_SIZE': 0.1,  # Validation set size
    'TEST_SIZE': 0.1,  # Test set size
    'RANDOM_STATE': 42,
    'MODEL_CONFIG': {
        # Fixed model architecture settings (not optimized by Optuna)
        'in_channels': 8,              # 7 original channels + 1 uniform meta channel
        'out_channels': 1,             # Single output channel
        'use_batch_norm': True,        # Batch normalization (generally always beneficial)
        'activation': 'relu',          # Activation function (standard choice)
        'final_activation': None,      # No final activation (for regression)
    },
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'step',  # Options: 'cosine', 'step'
        'early_stopping': 80,
        'T_0': 10,
        'T_max': 80,
        'T_mult': 2,
        'eta_min': 1e-8,
        'step_size': 30,
        'gamma': 0.5
    },
    'VISUALIZATION': {
        'SAMPLE_NUM': 8,
        'TIME_INDICES': (3, 7, 11, 15),
        'DPI': 200
    },
    'LOSS_CONFIG': {
        'loss_type': 'l2',  # Options: 'l2', 'mse'
        'l2_d': 3,  # Dimension for L2 loss
        'l2_p': 2   # Power for L2 loss
    },
    'TRAINING_CONFIG': {
        'mode': 'single',  # Options: 'single', 'optuna', 'eval'
        'optuna_n_trials': 50,
        'optuna_seed': 42,
        'optuna_n_startup_trials': 5,
        'eval_model_path': './src/FNO/output_unet/final/best_model_state_dict.pt'
    },
    'OPTUNA_SEARCH_SPACE': {
        'unet_depth_options': [2, 3, 4],                      # Variable U-Net depth (limited by temporal dimension)
        'base_channels_range': [16, 48],                      # [min, max] for suggest_int
        'kernel_size_options': [1, 3, 5],                     # Kernel size options (padding auto-calculated)
        'dropout_rate_range': [0.0, 0.3],                     # Dropout range
        'train_batch_size_options': [16, 32, 64],             # Batch size options
        'l2_weight_range': [1e-8, 1e-3],                      # L2 weight range [min, max] for log uniform
        'initial_lr_range': [1e-4, 1e-3]                      # Learning rate range [min, max] for log uniform
    },
    'SINGLE_PARAMS': {
        # Hyperparameters that can be optimized by Optuna
        "unet_depth": 3,           # U-Net depth (n-layer) - changed to 2 so 2^2=4 divides temporal dimension 20
        "base_channels": 32,       # Base number of channels  
        "kernel_size": 3,          # Convolution kernel size (padding auto-calculated)
        "dropout_rate": 0.1,       # Dropout probability
        "train_batch_size": 32,    # Training batch size
        "l2_weight": 0,            # L2 regularization weight
        "initial_lr": 1e-4         # Initial learning rate
    }
}

# ==============================================================================
# Data Classes and Dataset (Same as FNO version)
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
# Data Processing Functions (Same as FNO version)
# ==============================================================================

def preprocessing(config: Dict, verbose: bool = True) -> Tuple:
    """
    Unified data preprocessing function for Pure U-Net training.
    
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
            
        in_data = bundle["x"].float().to(device)[:, :, :, :, :16]
        out_data = bundle["y"].float().to(device) [:, :, :, :, :16]
        meta_data = bundle["meta"].float().to(device)

        print(f'in_data.shape: {in_data.shape}')
        
        if verbose:
            print(f"   Loaded tensors - Input: {tuple(in_data.shape)}, Output: {tuple(out_data.shape)}, Meta: {tuple(meta_data.shape)}")
            
    except Exception as e:
        raise RuntimeError(f"Failed at Step 1 (tensor loading): {e}")
    
    try:
        # Step 2: Transform meta data to uniform channels and combine
        if verbose:
            print("Step 2: Expanding meta channels and combining with input...")
            
        # Apply output masking as in original code
        out_data[:, :, 14:18, 14:18, :] = 0
        
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
            
        # Create normalizers for combined input and output
        in_normalizer = UnitGaussianNormalizer(
            mean=combined_input, std=combined_input, dim=[0,2,3,4], eps=1e-6
        )
        out_normalizer = UnitGaussianNormalizer(
            mean=out_data, std=out_data, dim=[0,2,3,4], eps=1e-6
        )
        
        # Fit normalizers
        in_normalizer.fit(combined_input)
        out_normalizer.fit(out_data)
        
        # Create processor
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
# Loss Function Options (Same as FNO version)
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
        relative_error = diff_norm / (y_norm + 1e-5)  # Add small epsilon to avoid division by zero
        
        if self.reduction == 'mean':
            return relative_error.mean()
        elif self.reduction == 'sum':
            return relative_error.sum()
        else:
            return relative_error

# ==============================================================================
# Scheduler Options (Same as FNO version)
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
        progress = epoch_in_cycle / self.T_i
        
        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            lrs.append(lr)
        
        # Check for restart
        if (self.last_epoch - self.last_restart) == self.T_i:
            self.last_restart = self.last_epoch
            self.T_i = min(self.T_i * self.T_mult, self.T_max)
        
        return lrs

# ==============================================================================
# Model Building Functions (Modified for U-Net)
# ==============================================================================

def create_model(config: Dict, train_dataset, val_dataset, test_dataset, device: str, 
                unet_depth: int, base_channels: int, kernel_size: int,
                dropout_rate: float, train_batch_size: int, 
                initial_lr: float, l2_weight: float):
    """
    Create complete U-Net model setup including DataLoaders, loss function, optimizer, 
    scheduler, and model architecture.
    
    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset  
        device: Device to use (cuda/cpu)
        unet_depth: Depth of U-Net (number of encoder/decoder levels)
        base_channels: Base number of channels
        kernel_size: Convolution kernel size
        dropout_rate: Dropout probability
        train_batch_size: Training batch size
        initial_lr: Initial learning rate
        l2_weight: L2 weight regularization
        
    Returns:
        Tuple containing (model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn)
    """
    
    # 1. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
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
    
    # 3. Create U-Net model
    # Auto-calculate padding for dimension preservation: padding = (kernel_size - 1) / 2
    auto_padding = (kernel_size - 1) // 2
    
    model = UNet3D(
        in_channels=config['MODEL_CONFIG']['in_channels'],
        out_channels=config['MODEL_CONFIG']['out_channels'],
        depth=unet_depth,
        base_channels=base_channels,
        kernel_size=kernel_size,
        padding=auto_padding,  # Auto-calculated padding
        use_batch_norm=config['MODEL_CONFIG']['use_batch_norm'],
        dropout_rate=dropout_rate,
        activation=config['MODEL_CONFIG']['activation'],
        final_activation=config['MODEL_CONFIG']['final_activation']
    ).to(device)
    
    # 4. Create optimizer
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=l2_weight)
    
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
# Training Functions (Same as FNO version)
# ==============================================================================

def train_model(config: Dict, processor, device: str, model, train_loader, val_loader, test_loader, 
                optimizer, scheduler, loss_fn, verbose: bool = True):
    """
    Train the U-Net model with early stopping and loss tracking.
    
    Args:
        config: Configuration dictionary
        processor: Data processor for normalization
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
    plt.title('U-Net Training and Validation Loss Over Time')
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
    Evaluate the trained U-Net model on test set and print detailed results.
    
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
    Perform hyperparameter optimization using Optuna for U-Net.
    
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
        print(f"\nStarting Optuna hyperparameter optimization for U-Net...")
        print(f"Number of trials: {config['TRAINING_CONFIG']['optuna_n_trials']}")
    
    # Create output directory for optuna results
    optuna_output_dir = Path(config['OUTPUT_DIR']) / 'optuna'
    optuna_output_dir.mkdir(parents=True, exist_ok=True)
    
    def objective(trial):
        """Objective function for Optuna optimization."""
        
        # Sample hyperparameters from search space
        search_space = config['OPTUNA_SEARCH_SPACE']
        
        # Sample categorical parameters
        unet_depth = trial.suggest_categorical('unet_depth', search_space['unet_depth_options'])
        kernel_size = trial.suggest_categorical('kernel_size', search_space['kernel_size_options'])
        train_batch_size = trial.suggest_categorical('train_batch_size', search_space['train_batch_size_options'])
        
        # Sample integer parameters with ranges
        base_channels = trial.suggest_int('base_channels', 
                                         search_space['base_channels_range'][0], 
                                         search_space['base_channels_range'][1])
        
        # Sample continuous parameters
        dropout_rate = trial.suggest_float('dropout_rate', 
                                          search_space['dropout_rate_range'][0], 
                                          search_space['dropout_rate_range'][1])
        l2_weight = trial.suggest_float('l2_weight', 
                                       search_space['l2_weight_range'][0], 
                                       search_space['l2_weight_range'][1], 
                                       log=True)
        initial_lr = trial.suggest_float('initial_lr', 
                                        search_space['initial_lr_range'][0], 
                                        search_space['initial_lr_range'][1], 
                                        log=True)
        
        try:
            # Create model with sampled parameters
            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                unet_depth=unet_depth,
                base_channels=base_channels,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                train_batch_size=train_batch_size,
                initial_lr=initial_lr,
                l2_weight=l2_weight
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
    
    # Define callback for progress reporting
    def progress_callback(study, trial):
        if verbose:
            current_best = study.best_value
            print(f"Trial {trial.number:3d} completed. "
                  f"Value: {trial.value:.6f}, Best: {current_best:.6f}")
    
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
        ax.set_title('U-Net Optuna Optimization History')
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
# Visualization Functions (Same as FNO version with U-Net naming)
# ==============================================================================

def visualization(config: Dict, processor, device: str, trained_model, train_dataset, 
                 test_dataset, verbose: bool = True):
    """
    Generate visualizations:
    1. A separate plot for permeability and pyrite maps.
    2. A 3x4 grid comparing ground truth, predictions, and their error over time.
    
    Args:
        config: Configuration dictionary.
        processor: Data processor for normalization.
        device: Device to use (e.g., 'cuda' or 'cpu').
        trained_model: The trained U-Net model.
        train_dataset: The training dataset.
        test_dataset: The test dataset for generating predictions.
        verbose: If True, prints progress information.
    """
    
    if verbose:
        print(f"\nGenerating visualization...")
    
    # Ensure the model is in evaluation mode
    trained_model.eval()
    
    # Create a DataLoader to handle the test data
    test_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset),  # Load all test data in a single batch
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
        for batch in test_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            
            # Store original input for visualizing permeability and pyrite
            all_input.append(x.cpu())
            
            # Normalize input for the model
            x_norm = processor.in_normalizer.transform(x)
            
            # Get model prediction
            pred = trained_model(x_norm)
            
            # Inverse transform the prediction to its physical scale
            pred_phys = processor.out_normalizer.inverse_transform(pred)
            
            all_pred.append(pred_phys.cpu())
            all_gt.append(y.cpu())
    
    # Concatenate results from all batches
    pred_phys = torch.cat(all_pred, dim=0)
    gt_phys = torch.cat(all_gt, dim=0)
    input_phys = torch.cat(all_input, dim=0)
    
    # Apply masking as per the original problem description
    pred_phys[:, :, 14:18, 14:18, :] = 0
    gt_phys[:, :, 14:18, 14:18, :] = 0
    
    # --- Data Extraction for Visualization ---
    # Select a sample index to visualize
    sample_idx = min(config['VISUALIZATION']['SAMPLE_NUM'], len(pred_phys) - 1)
    
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
    ax.set_title("Permeability")
    ax.axis('off')
    fig_inputs.colorbar(im_perm, ax=ax, orientation='horizontal', pad=0.1)

    # Plot Pyrite
    ax = axes_inputs[1]
    im_pyr = ax.imshow(pyr_sample.T, cmap='cividis')
    ax.set_title("Pyrite")
    ax.axis('off')
    fig_inputs.colorbar(im_pyr, ax=ax, orientation='horizontal', pad=0.1)
    
    fig_inputs.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the input data figure
    output_path_inputs = Path(config['OUTPUT_DIR']) / 'UNet_input_visualization.png'
    output_path_inputs.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_inputs, dpi=config['VISUALIZATION']['DPI'], bbox_inches='tight')
    plt.close(fig_inputs)
    if verbose:
        print(f"Input visualization saved to: {output_path_inputs}")

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
    output_path_grid = Path(config['OUTPUT_DIR']) / 'UNet_comparison_grid.png'
    plt.savefig(output_path_grid, dpi=config['VISUALIZATION']['DPI'], bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"Comparison grid saved to: {output_path_grid}")

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
            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=CONFIG,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                unet_depth=params['unet_depth'],
                base_channels=params['base_channels'],
                kernel_size=params['kernel_size'],
                dropout_rate=params['dropout_rate'],
                train_batch_size=params['train_batch_size'],
                initial_lr=params['initial_lr'],
                l2_weight=params['l2_weight']
            )
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"   U-Net Model created - Device: {device}")
            print(f"   Architecture: {model.get_architecture_info()}")
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
            
            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=CONFIG,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                unet_depth=best_params['unet_depth'],
                base_channels=best_params['base_channels'],
                kernel_size=best_params['kernel_size'],
                dropout_rate=best_params['dropout_rate'],
                train_batch_size=best_params['train_batch_size'],
                initial_lr=best_params['initial_lr'],
                l2_weight=best_params['l2_weight']
            )
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"   Final U-Net model created - Device: {device}")
            print(f"   Architecture: {model.get_architecture_info()}")
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
            model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
                config=CONFIG,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                unet_depth=params['unet_depth'],
                base_channels=params['base_channels'],
                kernel_size=params['kernel_size'],
                dropout_rate=params['dropout_rate'],
                train_batch_size=params['train_batch_size'],
                initial_lr=params['initial_lr'],
                l2_weight=params['l2_weight']
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
        
        print("\nU-Net training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main()