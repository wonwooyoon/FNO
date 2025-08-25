"""
Pure FNO with Uniform Distribution Meta Training

This module implements training pipeline for Pure TFNO models with meta data incorporated 
as uniform spatial channels. Meta data (e.g., permeability, porosity) is expanded to 
uniform distribution across spatial dimensions and directly concatenated with input channels, 
providing a straightforward approach to conditional neural operators.
"""

import sys
sys.path.append('./')
sys.path.append('./neuraloperator')

import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn
import optuna
import matplotlib.pyplot as plt
import torchinfo as summary
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.utils import count_model_params
# LpLoss will be implemented inline
from neuralop.models import TFNO
from neuralop.training import AdamW

# Direct inline training - no external utilities needed

# ==============================================================================
# Inline Loss Functions
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
        # Ensure same device
        pred = pred.to(y.device)
        
        # Get spatial dimensions (assuming format: [batch, channels, spatial_dims...])
        if len(pred.shape) == 4:  # 2D: [N, C, H, W] 
            dims = (2, 3)
        elif len(pred.shape) == 5:  # 3D: [N, C, H, W, T]
            dims = (2, 3, 4)
        else:
            raise ValueError(f"Unsupported tensor shape: {pred.shape}")
        
        # Compute relative Lp norm
        diff_norm = torch.norm(pred - y, p=self.p, dim=dims, keepdim=False)
        y_norm = torch.norm(y, p=self.p, dim=dims, keepdim=False)
        
        # Avoid division by zero
        relative_error = diff_norm / (y_norm + 1e-12)
        
        if self.reduction == 'mean':
            return relative_error.mean()
        elif self.reduction == 'sum':
            return relative_error.sum()
        else:
            return relative_error


# Configuration Constants
CONFIG = {
    'MERGED_PT_PATH': './src/preprocessing/merged.pt',
    'OUTPUT_DIR': './src/FNO/output_pure',
    'N_EPOCHS': 10000,
    'EVAL_INTERVAL': 1,
    'TEST_SIZE': 0.1,
    'RANDOM_STATE': 42,
    'DOMAIN_PADDING_MODE': 'symmetric',
    'MODEL_CONFIG': {
        'in_channels': 10,  # 8 original channels + 2 uniform meta channels
        'out_channels': 1,
        'lifting_channel_ratio': 2,
        'projection_channel_ratio': 2,
        'positional_embedding': 'grid',
        'film_layer': False  # Pure FNO without FiLM modulation
    },
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'cosine',  # Options: 'cosine', 'step'
        'early_stopping': 40,
        'T_0': 10,
        'T_max': 80,
        'T_mult': 2,
        'eta_min': 1e-8,
        'step_size': 10,
        'gamma': 0.5
    },
    'VISUALIZATION': {
        'SAMPLE_NUM': 8,
        'TIME_INDICES': (0, 4, 8, 12, 16),
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
        'eval_model_path': './src/FNO/output_pure/final/best_model_state_dict.pt'  # Path for eval mode
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
    """Custom dataset for Pure FNO training with meta data as uniform spatial channels.
    
    Args:
        input_tensor: Input tensor of shape (N, original_channels, nx, ny, nt)
        output_tensor: Output tensor of shape (N, 1, nx, ny, nt)
        meta_tensor: Meta tensor of shape (N, meta_channels)
    """
    
    def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, meta_tensor: torch.Tensor):
        # Combine input tensor with uniform meta channels
        self.input_tensor = expand_meta_to_uniform_channels(input_tensor, meta_tensor)
        self.output_tensor = output_tensor
        
    def __len__(self) -> int:
        return self.input_tensor.shape[0]
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': self.input_tensor[idx], 
            'y': self.output_tensor[idx]
        }

# ==============================================================================
# Learning Rate Schedulers
# ==============================================================================
class LRStepScheduler(torch.optim.lr_scheduler.StepLR):
    """Step Learning Rate Scheduler with configurable step size and gamma.
    
    Args:
        optimizer: Wrapped optimizer
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
        last_epoch: Index of last epoch
    """
    
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
        
    def get_lr(self) -> List[float]:
        t = self.last_epoch - self.last_restart
        if t >= self.T_i:
            self.last_restart = self.last_epoch
            self.T_i = min(self.T_i * self.T_mult, self.T_max)
            t = 0
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

# ==============================================================================
# Utility Functions
# ==============================================================================
def load_merged_tensors_pure(merged_pt_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load merged tensors from .pt file including meta data for Pure FNO.
    
    Args:
        merged_pt_path: Path to merged .pt file
        
    Returns:
        Tuple of (input_tensor, output_tensor, meta_tensor)
        
    Raises:
        FileNotFoundError: If merged file doesn't exist
        KeyError: If required keys are missing from loaded data
    """
    try:
        if not Path(merged_pt_path).exists():
            raise FileNotFoundError(f"Merged file not found: {merged_pt_path}")
            
        bundle = torch.load(merged_pt_path, map_location="cpu")
        
        required_keys = ["x", "y", "meta"]
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            raise KeyError(f"Missing required keys in data: {missing_keys}")
            
        in_summation = bundle["x"].float()
        out_summation = bundle["y"].float()
        meta_summation = bundle["meta"].float()
        
        print(f"Loaded merged tensors for Pure FNO: {tuple(in_summation.shape)}, {tuple(out_summation.shape)}, {tuple(meta_summation.shape)}")
        return in_summation, out_summation, meta_summation
        
    except Exception as e:
        print(f"Error loading merged tensors: {e}")
        raise

def expand_meta_to_uniform_channels(input_tensor: torch.Tensor, meta_tensor: torch.Tensor) -> torch.Tensor:
    """Expand meta data to uniform spatial channels and combine with input tensor.
    
    This function converts meta data into spatially uniform channels where each meta value
    is broadcasted as a constant across all spatial and temporal dimensions, then concatenated
    with the original input channels for Pure FNO processing.
    
    Args:
        input_tensor: Input tensor of shape (N, original_channels, nx, ny, nt)
        meta_tensor: Meta tensor of shape (N, meta_channels)
        
    Returns:
        Combined tensor of shape (N, original_channels + meta_channels, nx, ny, nt)
    """
    N, original_channels, nx, ny, nt = input_tensor.shape
    N_meta, meta_channels = meta_tensor.shape
    
    if N != N_meta:
        raise ValueError(f"Batch size mismatch: input_tensor {N}, meta_tensor {N_meta}")
    
    # Expand meta tensor to match spatial dimensions
    # Shape: (N, meta_channels) -> (N, meta_channels, nx, ny, nt)
    expanded_meta = meta_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, meta_channels, 1, 1, 1)
    expanded_meta = expanded_meta.expand(N, meta_channels, nx, ny, nt)      # (N, meta_channels, nx, ny, nt)
    
    # Concatenate along channel dimension
    combined_tensor = torch.cat([input_tensor, expanded_meta], dim=1)  # (N, original_channels + meta_channels, nx, ny, nt)
    
    print(f"Combined tensor shape: {tuple(combined_tensor.shape)} (original: {original_channels}, meta: {meta_channels})")
    return combined_tensor

def build_model(n_modes: Tuple[int, ...], hidden_channels: int, n_layers: int, 
                domain_padding: List[float], domain_padding_mode: str, device: str):
    """Build TFNO model with given hyperparameters.
    
    Args:
        n_modes: Number of modes for each dimension
        hidden_channels: Number of hidden channels
        n_layers: Number of layers
        domain_padding: Domain padding values
        domain_padding_mode: Padding mode
        device: Device to place model on
        
    Returns:
        Configured TFNO model
    """
    model = TFNO(
        n_modes=n_modes,
        in_channels=CONFIG['MODEL_CONFIG']['in_channels'],
        out_channels=CONFIG['MODEL_CONFIG']['out_channels'],
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        lifting_channel_ratio=CONFIG['MODEL_CONFIG']['lifting_channel_ratio'],
        projection_channel_ratio=CONFIG['MODEL_CONFIG']['projection_channel_ratio'],
        positional_embedding=CONFIG['MODEL_CONFIG']['positional_embedding'],
        domain_padding=domain_padding,
        domain_padding_mode=domain_padding_mode,
        film_layer=CONFIG['MODEL_CONFIG']['film_layer'],
        use_channel_mlp=True
    ).to(device)
    return model

def setup_data_loaders(train_dataset: CustomDatasetPure, test_dataset: CustomDatasetPure, 
                       batch_size: int) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Setup train and test data loaders.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset  
        batch_size: Training batch size
        
    Returns:
        Tuple of (train_loader, test_loaders_dict)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = {'test_dataloader': DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)}
    return train_loader, test_loader

def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = None):
    """Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to wrap
        scheduler_type: Type of scheduler to create ('cosine' or 'step')
                       If None, uses CONFIG['SCHEDULER_CONFIG']['scheduler_type']
        
    Returns:
        Configured scheduler
    """
    if scheduler_type is None:
        scheduler_type = CONFIG['SCHEDULER_CONFIG']['scheduler_type']
    
    if scheduler_type == 'cosine':
        return CappedCosineAnnealingWarmRestarts(
            optimizer,
            CONFIG['SCHEDULER_CONFIG']['T_0'],
            CONFIG['SCHEDULER_CONFIG']['T_max'],
            CONFIG['SCHEDULER_CONFIG']['T_mult'],
            CONFIG['SCHEDULER_CONFIG']['eta_min']
        )
    elif scheduler_type == 'step':
        return LRStepScheduler(
            optimizer,
            CONFIG['SCHEDULER_CONFIG']['step_size'],
            CONFIG['SCHEDULER_CONFIG']['gamma']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'cosine' or 'step'.")

def create_loss_function() -> Tuple[Any, str]:
    """Create loss function based on CONFIG settings.
    
    Returns:
        Tuple of (loss_function, loss_name)
    """
    loss_type = CONFIG['LOSS_CONFIG']['loss_type']
    
    if loss_type == 'l2':
        loss_fn = LpLoss(
            d=CONFIG['LOSS_CONFIG']['l2_d'], 
            p=CONFIG['LOSS_CONFIG']['l2_p']
        )
        loss_name = 'l2'
    elif loss_type == 'mse':
        loss_fn = torch.nn.MSELoss()
        loss_name = 'mse'
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'l2' or 'mse'.")
    
    return loss_fn, loss_name

def initialize_model_weights(model):
    """Initialize model weights using Xavier uniform initialization.
    
    Args:
        model: Model to initialize
    """
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)

def _collect_visualization_data(pred_phys: torch.Tensor, gt_phys: torch.Tensor, 
                               sample_num: int, t_indices: Tuple[int, ...]) -> Tuple[List, List, List, float, float]:
    """Collect data for visualization."""
    pis, gis, ers = [], [], []
    for t in t_indices:
        pi = pred_phys[sample_num, 0, :, :, t].cpu().numpy()
        gi = gt_phys[sample_num, 0, :, :, t].cpu().numpy()
        pis.append(pi)
        gis.append(gi)
        ers.append(np.abs(pi - gi))

    vmin = min(np.min(pis), np.min(gis))
    vmax = max(np.max(pis), np.max(gis))
    return pis, gis, ers, vmin, vmax

def _create_figure_and_axes(t_indices: Tuple[int, ...]) -> Tuple[plt.Figure, List, List, List, GridSpec]:
    """Create figure and axes for visualization."""
    ncols = len(t_indices)
    fig_h = 3.6 * 3
    fig_w = 1.8 * ncols + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    gs = GridSpec(nrows=3, ncols=ncols+2, figure=fig,
                  width_ratios=[*([1]*ncols), 0.05, 0.05],
                  height_ratios=[1, 1, 1], wspace=0.08, hspace=0.12)

    axes_gt, axes_pred, axes_err = [], [], []
    for r in range(3):
        row_axes = []
        for c in range(ncols):
            ax = fig.add_subplot(gs[r, c])
            row_axes.append(ax)
        if r == 0: 
            axes_gt = row_axes
        elif r == 1: 
            axes_pred = row_axes
        else: 
            axes_err = row_axes
            
    return fig, axes_gt, axes_pred, axes_err, gs

@torch.no_grad()
def plot_compare(pred_phys: torch.Tensor, gt_phys: torch.Tensor, save_path: str, 
                sample_num: int = 0, t_indices: Tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    """Plot comparison between predictions and ground truth.
    
    Args:
        pred_phys: Predicted physical values
        gt_phys: Ground truth physical values
        save_path: Path to save the comparison plot
        sample_num: Sample index to visualize
        t_indices: Time indices to visualize
    """
    pis, gis, ers, vmin, vmax = _collect_visualization_data(pred_phys, gt_phys, sample_num, t_indices)
    fig, axes_gt, axes_pred, axes_err, gs = _create_figure_and_axes(t_indices)
    
    ims_gt, ims_pred, ims_err = [], [], []
    for c, (pi, gi, er, t) in enumerate(zip(pis, gis, ers, t_indices)):
        im1 = axes_gt[c].imshow(gi, vmin=vmin, vmax=vmax)
        im2 = axes_pred[c].imshow(pi, vmin=vmin, vmax=vmax)
        im3 = axes_err[c].imshow(er)
        ims_gt.append(im1)
        ims_pred.append(im2)
        ims_err.append(im3)

        axes_gt[c].set_title(f"GT (t={t})")
        if c == 0:
            axes_pred[c].set_ylabel("Prediction", rotation=90, labelpad=20)
            axes_err[c].set_ylabel("Abs Error", rotation=90, labelpad=20)

    for row in (axes_gt, axes_pred, axes_err):
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    ncols = len(t_indices)
    cax_main = fig.add_subplot(gs[:, ncols])
    cax_err = fig.add_subplot(gs[:, ncols+1])
    
    cb_main = fig.colorbar(ims_gt[0], cax=cax_main)
    cb_main.set_label("Value")
    cb_err = fig.colorbar(ims_err[0], cax=cax_err)
    cb_err.set_label("Abs Error")

    fig.savefig(save_path, dpi=CONFIG['VISUALIZATION']['DPI'])
    plt.close(fig)
    print(f"Saved comparison plot: {save_path}")

# ==============================================================================
# Data Processing and Training Functions
# ==============================================================================
def prepare_data_and_normalizers_pure(merged_pt_path: str) -> Tuple:
    """Prepare data and normalizers for Pure FNO with uniform meta channels.
    
    Args:
        merged_pt_path: Path to merged data file
        
    Returns:
        Tuple containing tensors, normalizers, processor, and datasets
    """
    in_summation, out_summation, meta_summation = load_merged_tensors_pure(merged_pt_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    in_summation = in_summation.to(device)
    out_summation = out_summation.to(device)
    meta_summation = meta_summation.to(device)
    
    out_summation = 10 ** out_summation
    out_summation[:, :, 14:18, 14:18, :] = 0
    
    # Combine input with uniform meta channels for full dataset
    combined_input = expand_meta_to_uniform_channels(in_summation, meta_summation)
    
    # Create normalizers for combined input and output
    in_normalizer = UnitGaussianNormalizer(mean=combined_input, std=combined_input, dim=[0,2,3,4], eps=1e-6)
    out_normalizer = UnitGaussianNormalizer(mean=out_summation, std=out_summation, dim=[0,2,3,4], eps=1e-6)
    
    in_normalizer.fit(combined_input)
    out_normalizer.fit(out_summation)
    
    processor = DefaultDataProcessor(in_normalizer, out_normalizer).to(device)
    
    # Split the original tensors, not the combined one
    train_in, test_in, train_out, test_out, train_meta, test_meta = train_test_split(
        in_summation, out_summation, meta_summation, 
        test_size=CONFIG['TEST_SIZE'], 
        random_state=CONFIG['RANDOM_STATE']
    )
    
    train_dataset = CustomDatasetPure(train_in, train_out, train_meta)
    test_dataset = CustomDatasetPure(test_in, test_out, test_meta)
    
    return (combined_input, out_summation, device, 
            in_normalizer, out_normalizer, processor, 
            train_dataset, test_dataset)

def create_objective_function(train_dataset: CustomDatasetPure, test_dataset: CustomDatasetPure, 
                             processor: DefaultDataProcessor, device: str) -> callable:
    """Create Optuna objective function.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset  
        processor: Data processor
        device: Device to use
        
    Returns:
        Objective function for Optuna optimization
    """
    def objective(trial: optuna.trial.Trial) -> float:
        """Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation L2 loss
        """
        # Get search space from CONFIG
        search_space = CONFIG['OPTUNA_SEARCH_SPACE']
        
        n_modes = trial.suggest_categorical("n_modes", search_space['n_modes_options'])
        hidden_channels = trial.suggest_categorical("hidden_channels", search_space['hidden_channels_options'])
        n_layers = trial.suggest_categorical("n_layers", search_space['n_layers_options'])
        domain_padding = trial.suggest_categorical("domain_padding", search_space['domain_padding_options'])
        train_batch_size = trial.suggest_categorical("train_batch_size", search_space['train_batch_size_options'])
        l2_weight = trial.suggest_float("l2_weight", *search_space['l2_weight_range'], log=True)
        initial_lr = trial.suggest_float("initial_lr", *search_space['initial_lr_range'], log=True)

        train_loader, test_loader = setup_data_loaders(train_dataset, test_dataset, train_batch_size)
        
        model = build_model(n_modes, hidden_channels, n_layers, domain_padding, 
                          CONFIG['DOMAIN_PADDING_MODE'], device)
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=l2_weight)
        scheduler = create_scheduler(optimizer)

        loss_fn, loss_name = create_loss_function()
        
        best_model_path = Path(CONFIG['OUTPUT_DIR']) / 'optuna'
        best_model_path.mkdir(parents=True, exist_ok=True)

        # Simple for-loop training
        best_test_loss = float('inf')
        patience = 0
        early_stopping_patience = CONFIG['SCHEDULER_CONFIG']['early_stopping']
        
        for epoch in range(CONFIG['N_EPOCHS']):
            # Training phase
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                # Apply input and output normalization for consistent training
                if processor and hasattr(processor, 'in_normalizer') and processor.in_normalizer:
                    x = processor.in_normalizer.transform(x)
                if processor and hasattr(processor, 'out_normalizer') and processor.out_normalizer:
                    y = processor.out_normalizer.transform(y)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Evaluation phase
            if epoch % CONFIG['EVAL_INTERVAL'] == 0:
                model.eval()
                total_test_loss = 0
                with torch.no_grad():
                    for batch in test_loader['test_dataloader']:
                        x = batch['x'].to(device)
                        y = batch['y'].to(device)
                        
                        if processor and hasattr(processor, 'in_normalizer') and processor.in_normalizer:
                            x = processor.in_normalizer.transform(x)
                        if processor and hasattr(processor, 'out_normalizer') and processor.out_normalizer:
                            y = processor.out_normalizer.transform(y)
                        
                        pred = model(x)
                        loss = loss_fn(pred, y)
                        total_test_loss += loss.item()
                
                avg_test_loss = total_test_loss / len(test_loader['test_dataloader'])
                
                # Save best model
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    torch.save(model.state_dict(), best_model_path / 'best_model_state_dict.pt')
                    patience = 0
                else:
                    patience += 1
                
                # Early stopping
                if patience >= early_stopping_patience:
                    break
            
            # Update learning rate
            scheduler.step()
        
        # Load best model
        model.load_state_dict(torch.load(best_model_path / 'best_model_state_dict.pt', map_location=device, weights_only=False))
        
        val_loss = best_test_loss
        
        return val_loss
    
    return objective

def train_final_model(best_params: Dict[str, Any], train_dataset: CustomDatasetPure, 
                     test_dataset: CustomDatasetPure, processor: DefaultDataProcessor,
                     in_normalizer, out_normalizer, device: str) -> None:
    """Train final model with best parameters and generate comparison plot.
    
    Args:
        best_params: Best hyperparameters from optimization
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Data processor
        in_normalizer: Input normalizer
        out_normalizer: Output normalizer  
        device: Device to use
    """
    best_model = build_model(
        best_params["n_modes"], 
        best_params["hidden_channels"], 
        best_params["n_layers"],
        best_params["domain_padding"], 
        CONFIG['DOMAIN_PADDING_MODE'], 
        device
    )

    print(f'{count_model_params(best_model)}')

    optimizer = AdamW(
        best_model.parameters(), 
        lr=best_params["initial_lr"], 
        weight_decay=best_params["l2_weight"]
    )
    scheduler = create_scheduler(optimizer)

    initialize_model_weights(best_model)

    loss_fn, loss_name = create_loss_function()
    
    train_loader, test_loader = setup_data_loaders(
        train_dataset, test_dataset, best_params["train_batch_size"]
    )

    final_model_path = Path(CONFIG['OUTPUT_DIR']) / 'final'
    final_model_path.mkdir(parents=True, exist_ok=True)

    # Simple for-loop training
    print(f"Starting training for {CONFIG['N_EPOCHS']} epochs...")
    
    best_test_loss = float('inf')
    patience = 0
    early_stopping_patience = CONFIG['SCHEDULER_CONFIG']['early_stopping']
    
    # Track losses for each epoch
    train_losses = []
    test_losses = []
    
    for epoch in range(CONFIG['N_EPOCHS']):
        # Training phase
        best_model.train()
        total_train_loss = 0
        train_count = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            # Apply input and output normalization for consistent training
            if processor and hasattr(processor, 'in_normalizer') and processor.in_normalizer:
                x = processor.in_normalizer.transform(x)
            if processor and hasattr(processor, 'out_normalizer') and processor.out_normalizer:
                y = processor.out_normalizer.transform(y)
            
            optimizer.zero_grad()
            pred = best_model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = total_train_loss / train_count
        train_losses.append(avg_train_loss)
        
        # Evaluation phase - compute test loss every epoch
        best_model.eval()
        total_test_loss = 0
        test_count = 0
        
        with torch.no_grad():
            for batch in test_loader['test_dataloader']:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                if processor and hasattr(processor, 'in_normalizer') and processor.in_normalizer:
                    x = processor.in_normalizer.transform(x)
                if processor and hasattr(processor, 'out_normalizer') and processor.out_normalizer:
                    y = processor.out_normalizer.transform(y)
                
                pred = best_model(x)
                loss = loss_fn(pred, y)
                total_test_loss += loss.item()
                test_count += 1
        
        avg_test_loss = total_test_loss / test_count
        test_losses.append(avg_test_loss)
        
        # Print losses for every epoch
        print(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.6f}, Test Loss={avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(best_model.state_dict(), final_model_path / 'best_model_state_dict.pt')
            patience = 0
            print(f"    ★ New best model saved! Test loss: {avg_test_loss:.6f}")
        else:
            patience += 1
        
        # Early stopping
        if patience >= early_stopping_patience:
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
    torch.save(loss_history, final_model_path / 'loss_history.pt')
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    epochs_range = range(len(train_losses))
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    loss_plot_path = final_model_path / 'loss_curves.png'
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Loss history saved to: {final_model_path / 'loss_history.pt'}")
    print(f"📈 Loss curves plotted: {loss_plot_path}")
    print(f"📈 Final Train Loss: {train_losses[-1]:.6f}")
    print(f"📉 Final Test Loss: {test_losses[-1]:.6f}")
    print(f"🏆 Best Test Loss: {best_test_loss:.6f}")
    
    # Load best trained model
    best_model.load_state_dict(torch.load(final_model_path / 'best_model_state_dict.pt', map_location=device, weights_only=False))

    # Generate predictions for visualization
    best_model.eval()
    all_pred = []
    all_gt = []
    
    with torch.no_grad():
        for batch in test_loader["test_dataloader"]:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            # Apply input normalization
            if in_normalizer:
                x = in_normalizer.transform(x)
            
            # Forward pass
            pred = best_model(x)
            
            # Denormalize to physical space
            if out_normalizer:
                pred_phys = out_normalizer.inverse_transform(pred)
            else:
                pred_phys = pred
            
            all_pred.append(pred_phys.cpu())
            all_gt.append(y.cpu())
    
    pred_phys = torch.cat(all_pred, dim=0)
    gt_phys = torch.cat(all_gt, dim=0)
    
    # Apply masking as in original code
    pred_phys[:, :, 14:18, 14:18, :] = 0
    gt_phys[:, :, 14:18, 14:18, :] = 0

    output_path = Path(CONFIG['OUTPUT_DIR']) / 'FNO_pure_compare.png'
    plot_compare(
        pred_phys, gt_phys, 
        save_path=str(output_path), 
        sample_num=CONFIG['VISUALIZATION']['SAMPLE_NUM'], 
        t_indices=CONFIG['VISUALIZATION']['TIME_INDICES']
    )

def run_single_training() -> Dict[str, Any]:
    """Run single training with predefined parameters.
    
    Returns:
        Dict containing the predefined parameters for training
    """
    print("=" * 50)
    print("RUNNING SINGLE TRAINING MODE")
    print("Using predefined parameters from CONFIG")
    print("=" * 50)
    
    return CONFIG['SINGLE_PARAMS'].copy()

def run_optuna_training(train_dataset: CustomDatasetPure, test_dataset: CustomDatasetPure, 
                       processor: DefaultDataProcessor, device: str) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Data processor
        device: Device to use
        
    Returns:
        Dict containing the best parameters found by Optuna
    """
    print("=" * 50)
    print("RUNNING OPTUNA OPTIMIZATION MODE")
    print(f"Number of trials: {CONFIG['TRAINING_CONFIG']['optuna_n_trials']}")
    print("=" * 50)
    
    objective_fn = create_objective_function(train_dataset, test_dataset, processor, device)
    sampler = optuna.samplers.TPESampler(
        seed=CONFIG['TRAINING_CONFIG']['optuna_seed'], 
        n_startup_trials=CONFIG['TRAINING_CONFIG']['optuna_n_startup_trials']
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective_fn, 
        n_trials=CONFIG['TRAINING_CONFIG']['optuna_n_trials'], 
        show_progress_bar=True
    )
    
    print(f"\nOptuna Best Value: {study.best_value}")
    print(f"Optuna Best Params: {study.best_params}")
    
    return study.best_params

def run_eval_mode(train_dataset: CustomDatasetPure, test_dataset: CustomDatasetPure, 
                 processor: DefaultDataProcessor, 
                 in_normalizer, out_normalizer, device: str) -> None:
    """Run evaluation mode using pre-trained model.
    
    Args:
        train_dataset: Training dataset (used for model architecture)
        test_dataset: Test dataset
        processor: Data processor
        in_normalizer: Input normalizer
        out_normalizer: Output normalizer
        device: Device to use
    """
    print("=" * 50)
    print("RUNNING EVALUATION MODE")
    print("Loading pre-trained model for evaluation")
    print("=" * 50)
    
    # Use SINGLE_PARAMS for model architecture (assuming model was trained with these)
    model_params = CONFIG['SINGLE_PARAMS']
    
    # Build model with same architecture as training
    model = build_model(
        model_params["n_modes"], 
        model_params["hidden_channels"], 
        model_params["n_layers"],
        model_params["domain_padding"], 
        CONFIG['DOMAIN_PADDING_MODE'], 
        device
    )
    
    # Load pre-trained model
    model_path = CONFIG['TRAINING_CONFIG']['eval_model_path']
    try:
        model.load_state_dict(torch.load(
            model_path, 
            map_location=device, weights_only=False
        ))
        model.eval()
        print(f"✅ Successfully loaded model from: {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pre-trained model not found at: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    # Setup test data loader
    test_loader = {'test_dataloader': DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)}
    
    print("\n" + "=" * 50)
    print("GENERATING PREDICTIONS AND VISUALIZATIONS")
    print("=" * 50)
    
    # Generate predictions for evaluation visualization
    model.eval()
    all_pred = []
    all_gt = []
    
    with torch.no_grad():
        for batch in test_loader["test_dataloader"]:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            # Apply input normalization
            if in_normalizer:
                x = in_normalizer.transform(x)
            
            # Forward pass
            pred = model(x)
            
            # Denormalize to physical space
            if out_normalizer:
                pred_phys = out_normalizer.inverse_transform(pred)
            else:
                pred_phys = pred
            
            all_pred.append(pred_phys.cpu())
            all_gt.append(y.cpu())
    
    pred_phys = torch.cat(all_pred, dim=0)
    gt_phys = torch.cat(all_gt, dim=0)
    
    # Apply masking as in original code
    pred_phys[:, :, 14:18, 14:18, :] = 0
    gt_phys[:, :, 14:18, 14:18, :] = 0

    output_path = Path(CONFIG['OUTPUT_DIR']) / 'FNO_pure_eval_compare.png'
    plot_compare(
        pred_phys, gt_phys, 
        save_path=str(output_path), 
        sample_num=CONFIG['VISUALIZATION']['SAMPLE_NUM'], 
        t_indices=CONFIG['VISUALIZATION']['TIME_INDICES']
    )
    
    print(f"✅ Evaluation completed! Results saved to: {output_path}")

def run_training_mode(train_dataset: CustomDatasetPure, test_dataset: CustomDatasetPure, 
                     processor: DefaultDataProcessor, 
                     in_normalizer, out_normalizer, device: str) -> None:
    """Run training based on configured mode.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Data processor
        in_normalizer: Input normalizer
        out_normalizer: Output normalizer
        device: Device to use
    """
    mode = CONFIG['TRAINING_CONFIG']['mode']
    
    if mode == 'single':
        best_params = run_single_training()
        print("\n" + "=" * 50)
        print("STARTING FINAL MODEL TRAINING")
        print("=" * 50)
        train_final_model(
            best_params, train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
    elif mode == 'optuna':
        best_params = run_optuna_training(train_dataset, test_dataset, processor, device)
        print("\n" + "=" * 50)
        print("STARTING FINAL MODEL TRAINING")
        print("=" * 50)
        train_final_model(
            best_params, train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
    elif mode == 'eval':
        run_eval_mode(
            train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
    else:
        raise ValueError(f"Unknown training mode: {mode}. Options: 'single', 'optuna', 'eval'")

def main() -> None:
    """Main training pipeline for Pure FNO with uniform meta channels."""
    try:
        data_results = prepare_data_and_normalizers_pure(CONFIG['MERGED_PT_PATH'])
        (
            combined_input, out_summation, device,
            in_normalizer, out_normalizer, processor,
            train_dataset, test_dataset
        ) = data_results
        
        print(f"\n🚀 FNO-Pure Training Pipeline Started")
        print(f"Training Mode: {CONFIG['TRAINING_CONFIG']['mode'].upper()}")
        
        run_training_mode(
            train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()