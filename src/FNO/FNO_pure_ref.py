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
from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop.training import AdamW

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'MERGED_PT_PATH': './src/preprocessing/merged.pt',
    'OUTPUT_DIR': './src/FNO/output_pure',
    'N_EPOCHS': 3,  # Reduced for testing
    'EVAL_INTERVAL': 1,
    'TEST_SIZE': 0.1,
    'RANDOM_STATE': 42,
    'DOMAIN_PADDING_MODE': 'symmetric',
    'MODEL_CONFIG': {
        'in_channels': 8,  # 7 original channels + 1 uniform meta channel
        'out_channels': 1,
        'lifting_channel_ratio': 2,
        'projection_channel_ratio': 2,
        'positional_embedding': 'grid',
    },
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'cosine',  # Options: 'cosine', 'step'
        'early_stopping': 80,
        'T_0': 10,
        'T_max': 80,
        'T_mult': 2,
        'eta_min': 1e-8,
        'step_size': 10,
        'gamma': 0.5
    },
    'VISUALIZATION': {
        'SAMPLE_NUM': 8,
        'TIME_INDICES': (4, 9, 14, 19),
        'DPI': 200
    },
    'LOSS_CONFIG': {
        'loss_type': 'l2',  # Options: 'l2', 'mse'
        'l2_d': 3,  # Dimension for L2 loss
        'l2_p': 2   # Power for L2 loss
    },
    'TRAINING_CONFIG': {
        'mode': 'single',  # Options: 'single', 'optuna', 'eval'
        'optuna_n_trials': 10,
        'optuna_seed': 42,
        'optuna_n_startup_trials': 2,
        'eval_model_path': './src/FNO/output_pure/final/best_model_state_dict.pt'
    },
    'OPTUNA_SEARCH_SPACE': {
        'n_modes_options': [(16,16,5), (16,8,5), (32,16,5)],
        'hidden_channels_options': [8, 16, 24, 32],
        'n_layers_options': [2, 3, 4],
        'domain_padding_options': [[0.125,0.25,0.4], [0.1,0.1,0.1], [0.2,0.3,0.5]],
        'train_batch_size_options': [16, 32, 64],
        'l2_weight_range': [1e-8, 1e-3],  # [min, max] for log uniform
        'initial_lr_range': [1e-4, 1e-3]  # [min, max] for log uniform
    },
    'SINGLE_PARAMS': {
        "n_modes": (16, 8, 4), 
        "hidden_channels": 12, 
        "n_layers": 4, 
        "domain_padding": [0.1, 0.1, 0.1], 
        "train_batch_size": 32, 
        "l2_weight": 0, 
        "initial_lr": 1e-3
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
    4. Perform train/test split and create datasets
    5. Return necessary objects for training
    
    Args:
        config: Configuration dictionary containing paths and parameters
        verbose: Whether to print progress information
        
    Returns:
        Tuple containing (processor, train_dataset, test_dataset, device)
    """
    
    if verbose:
        print(f"\n📊 Starting unified data preprocessing...")
    
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
            
        in_data = bundle["x"].float().to(device)
        out_data = bundle["y"].float().to(device) 
        meta_data = bundle["meta"].float().to(device)
        
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
        # Step 4: Perform train/test split and create datasets
        if verbose:
            print("Step 4: Creating train/test datasets...")
            
        # Split using the already combined input tensor
        train_combined, test_combined, train_out, test_out = train_test_split(
            combined_input, out_data,
            test_size=config['TEST_SIZE'], 
            random_state=config['RANDOM_STATE']
        )
        
        # Create datasets with already combined inputs
        train_dataset = CustomDatasetPure(train_combined, train_out)
        test_dataset = CustomDatasetPure(test_combined, test_out)
        
        if verbose:
            print(f"   Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
            
    except Exception as e:
        raise RuntimeError(f"Failed at Step 4 (dataset creation): {e}")
    
    if verbose:
        print("✅ Data preprocessing completed successfully!")
    
    # Step 5: Return necessary objects
    return (processor, train_dataset, test_dataset, device)

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
# Model Building Functions
# ==============================================================================

def create_model(config: Dict, train_dataset, test_dataset, device: str, 
                n_modes: Tuple[int, ...], hidden_channels: int, n_layers: int, 
                domain_padding: List[float], train_batch_size: int, 
                initial_lr: float, l2_weight: float):
    """
    Create complete model setup including DataLoaders, loss function, optimizer, 
    scheduler, and model architecture.
    
    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        test_dataset: Test dataset  
        device: Device to use (cuda/cpu)
        n_modes: Number of modes for each dimension
        hidden_channels: Number of hidden channels
        n_layers: Number of layers
        domain_padding: Domain padding values
        train_batch_size: Training batch size
        initial_lr: Initial learning rate
        l2_weight: L2 weight regularization
        
    Returns:
        Tuple containing (model, train_loader, test_loader, optimizer, scheduler, loss_fn)
    """
    
    # 1. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
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
        use_channel_mlp=True
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
    
    return (model, train_loader, test_loader, optimizer, scheduler, loss_fn)

# ==============================================================================
# Training Functions
# ==============================================================================

def train_model(config: Dict, processor, device: str, model, train_loader, test_loader, 
                optimizer, scheduler, loss_fn, verbose: bool = True):
    """
    Train the FNO model with early stopping and loss tracking.
    
    Args:
        config: Configuration dictionary
        processor: Data processor for normalization
        device: Device to use (cuda/cpu)
        model: TFNO model to train
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        verbose: Whether to print training progress
        
    Returns:
        Trained model
    """
    
    if verbose:
        print(f"\n🔥 Starting model training for {config['N_EPOCHS']} epochs...")
    
    # Training setup
    best_test_loss = float('inf')
    patience = 0
    early_stopping_patience = config['SCHEDULER_CONFIG']['early_stopping']
    
    # Track losses for each epoch
    train_losses = []
    test_losses = []
    
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
        
        # Evaluation phase - compute test loss every epoch
        model.eval()
        total_test_loss = 0
        test_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                x = processor.in_normalizer.transform(x)
                y = processor.out_normalizer.transform(y)
                
                pred = model(x)
                loss = loss_fn(pred, y)
                total_test_loss += loss.item()
                test_count += 1
        
        test_loss = total_test_loss / test_count
        test_losses.append(test_loss)
        
        # Print losses for every epoch
        if verbose:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Test Loss={test_loss:.6f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), output_dir / 'best_model_state_dict.pt')
            patience = 0
            if verbose:
                print(f"    ★ New best model saved! Test loss: {test_loss:.6f}")
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
        'test_losses': test_losses,
        'epochs': list(range(len(train_losses)))
    }
    torch.save(loss_history, output_dir / 'loss_history.pt')
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    epochs_range = range(len(train_losses))
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FNO Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    loss_plot_path = output_dir / 'loss_curves.png'
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"\n📊 Training completed!")
        print(f"📈 Final Train Loss: {train_losses[-1]:.6f}")
        print(f"📉 Final Test Loss: {test_losses[-1]:.6f}")
        print(f"🏆 Best Test Loss: {best_test_loss:.6f}")
        print(f"💾 Loss history saved to: {output_dir / 'loss_history.pt'}")
        print(f"📈 Loss curves saved to: {loss_plot_path}")
    
    # Load and return the best trained model
    model.load_state_dict(torch.load(output_dir / 'best_model_state_dict.pt', map_location=device, weights_only=False))
    
    return model

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
        trained_model: The trained FNO model.
        train_dataset: The training dataset.
        test_dataset: The test dataset for generating predictions.
        verbose: If True, prints progress information.
    """
    
    if verbose:
        print(f"\n📊 Generating visualization...")
    
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
    output_path_inputs = Path(config['OUTPUT_DIR']) / 'FNO_input_visualization.png'
    output_path_inputs.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_inputs, dpi=config['VISUALIZATION']['DPI'], bbox_inches='tight')
    plt.close(fig_inputs)
    if verbose:
        print(f"✅ Input visualization saved to: {output_path_inputs}")

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
    output_path_grid = Path(config['OUTPUT_DIR']) / 'FNO_comparison_grid.png'
    plt.savefig(output_path_grid, dpi=config['VISUALIZATION']['DPI'], bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"✅ Comparison grid saved to: {output_path_grid}")

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
        print(f"\n🚀 FNO-Pure Training Pipeline Started")
        print(f"Training Mode: {CONFIG['TRAINING_CONFIG']['mode'].upper()}")
        
        processor, train_dataset, test_dataset, device = preprocessing(
            config=CONFIG, 
            verbose=True
        )
        
        # Step 2: Create model setup
        print("\n🔥 Creating model setup...")
        
        # Get parameters from config for testing
        params = CONFIG['SINGLE_PARAMS']
        model, train_loader, test_loader, optimizer, scheduler, loss_fn = create_model(
            config=CONFIG,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            n_modes=params['n_modes'],
            hidden_channels=params['hidden_channels'],
            n_layers=params['n_layers'],
            domain_padding=params['domain_padding'],
            train_batch_size=params['train_batch_size'],
            initial_lr=params['initial_lr'],
            l2_weight=params['l2_weight']
        )
        
        print(f"   ✅ Model created - Device: {device}")
        print(f"   ✅ Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        print(f"   ✅ Optimizer: {type(optimizer).__name__}")
        print(f"   ✅ Scheduler: {type(scheduler).__name__}")
        print(f"   ✅ Loss function: {type(loss_fn).__name__}")
        print(f"   ✅ Processor: {type(processor).__name__}")
        
        # Step 3: Train the model
        trained_model = train_model(
            config=CONFIG,
            processor=processor,
            device=device,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            verbose=True
        )
        
        # Step 4: Generate visualization
        visualization(
            config=CONFIG,
            processor=processor,
            device=device,
            trained_model=trained_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            verbose=True
        )
        
        print("\n✅ Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()