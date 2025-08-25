"""
2D FNO with Spatial-Only Predictions

This module implements training pipeline for 2D TFNO models that perform spatial-only
predictions without temporal forecasting. The temporal dimension is removed by taking
the final time step [:,:,:,:,-1] from the original spatiotemporal data, reducing
the problem from 3D to 2D spatial prediction.

Based on FNO_pure.py but adapted for spatial-only neural operator training.
"""

import sys
sys.path.append('./')

import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn
import optuna
import matplotlib.pyplot as plt
import torchinfo as summary
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuraloperator.neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuraloperator.neuralop.utils import count_model_params
from neuraloperator.neuralop import LpLoss
from neuraloperator.neuralop.models import TFNO2d
from neuraloperator.neuralop import Trainer
from neuraloperator.neuralop.training import AdamW


# Configuration Constants
CONFIG = {
    'MERGED_PT_PATH': './src/preprocessing/merged.pt',
    'OUTPUT_DIR': './src/FNO/output_2d',
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
        'T_max': 40,
        'T_mult': 2,
        'eta_min': 1e-6,
        'step_size': 10,
        'gamma': 0.5
    },
    'VISUALIZATION': {
        'SAMPLE_NUM': 8,
        'DPI': 200
    },
    'LOSS_CONFIG': {
        'loss_type': 'l2',  # Options: 'l2', 'mse'
        'l2_d': 2,  # Dimension for L2 loss (reduced from 3 to 2 for spatial-only)
        'l2_p': 2   # Power for L2 loss
    },
    'TRAINING_CONFIG': {
        'mode': 'single',  # Options: 'single', 'optuna', 'eval'
        'optuna_n_trials': 10,
        'optuna_seed': 42,
        'optuna_n_startup_trials': 2,
        'eval_model_path': './src/FNO/output_2d/final/best_model_state_dict.pt'  # Path for eval mode
    },
    'OPTUNA_SEARCH_SPACE': {
        'n_modes_options': [(16,16), (16,8), (32,16)],  # 2D modes only
        'hidden_channels_options': [8, 16, 24, 32],
        'n_layers_options': [2, 3, 4],
        'domain_padding_options': [[0.125,0.25], [0.1,0.1], [0.2,0.3]],  # 2D padding only
        'train_batch_size_options': [16, 32, 64],
        'l2_weight_range': [1e-8, 1e-3],  # [min, max] for log uniform
        'initial_lr_range': [1e-4, 1e-3]  # [min, max] for log uniform
    },
    'SINGLE_PARAMS': {
        "n_modes": (32, 16),  # 2D modes only (removed temporal mode)
        "hidden_channels": 24, 
        "n_layers": 5,
        "domain_padding": [0.1, 0.1],  # 2D padding only
        "train_batch_size": 32, 
        "l2_weight": 0, 
        "initial_lr": 1e-4
    }
}

# ==============================================================================
# Data Classes and Dataset
# ==============================================================================
class CustomDataset2D(Dataset):
    """Custom dataset for 2D FNO training with meta data as uniform spatial channels.
    
    Args:
        input_tensor: Input tensor of shape (N, channels, nx, ny) - temporal dimension removed
        output_tensor: Output tensor of shape (N, 1, nx, ny) - temporal dimension removed
        meta_tensor: Meta tensor of shape (N, meta_channels)
    """
    
    def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, meta_tensor: torch.Tensor):
        # Combine input tensor with uniform meta channels
        self.input_tensor = expand_meta_to_uniform_channels_2d(input_tensor, meta_tensor)
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
def reduce_temporal_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce temporal dimension by taking the final time step.
    
    Args:
        tensor: Input tensor of shape (N, channels, nx, ny, nt)
        
    Returns:
        Tensor of shape (N, channels, nx, ny) with temporal dimension removed
    """
    if len(tensor.shape) == 5:  # (N, channels, nx, ny, nt)
        return tensor[:, :, :, :, -1].contiguous()  # Take final time step
    elif len(tensor.shape) == 4:  # Already 2D
        return tensor
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tensor.shape}")

def load_merged_tensors_2d(merged_pt_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load merged tensors from .pt file and reduce temporal dimension for 2D FNO.
    
    Args:
        merged_pt_path: Path to merged .pt file
        
    Returns:
        Tuple of (input_tensor_2d, output_tensor_2d, meta_tensor)
        
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
        
        # Reduce temporal dimension by taking final time step
        in_summation_2d = reduce_temporal_dimension(in_summation)
        out_summation_2d = reduce_temporal_dimension(out_summation)
        
        print(f"Original tensor shapes: {tuple(in_summation.shape)}, {tuple(out_summation.shape)}")
        print(f"Reduced to 2D shapes: {tuple(in_summation_2d.shape)}, {tuple(out_summation_2d.shape)}, {tuple(meta_summation.shape)}")
        return in_summation_2d, out_summation_2d, meta_summation
        
    except Exception as e:
        print(f"Error loading merged tensors: {e}")
        raise

def expand_meta_to_uniform_channels_2d(input_tensor: torch.Tensor, meta_tensor: torch.Tensor) -> torch.Tensor:
    """Expand meta data to uniform spatial channels and combine with 2D input tensor.
    
    This function converts meta data into spatially uniform channels where each meta value
    is broadcasted as a constant across all spatial dimensions, then concatenated
    with the original input channels for 2D FNO processing.
    
    Args:
        input_tensor: Input tensor of shape (N, original_channels, nx, ny)
        meta_tensor: Meta tensor of shape (N, meta_channels)
        
    Returns:
        Combined tensor of shape (N, original_channels + meta_channels, nx, ny)
    """
    N, original_channels, nx, ny = input_tensor.shape
    N_meta, meta_channels = meta_tensor.shape
    
    if N != N_meta:
        raise ValueError(f"Batch size mismatch: input_tensor {N}, meta_tensor {N_meta}")
    
    # Expand meta tensor to match spatial dimensions
    # Shape: (N, meta_channels) -> (N, meta_channels, nx, ny)
    expanded_meta = meta_tensor.unsqueeze(-1).unsqueeze(-1)  # (N, meta_channels, 1, 1)
    expanded_meta = expanded_meta.expand(N, meta_channels, nx, ny)  # (N, meta_channels, nx, ny)
    
    # Concatenate along channel dimension
    combined_tensor = torch.cat([input_tensor, expanded_meta], dim=1)  # (N, original_channels + meta_channels, nx, ny)
    
    print(f"Combined 2D tensor shape: {tuple(combined_tensor.shape)} (original: {original_channels}, meta: {meta_channels})")
    return combined_tensor

def build_model_2d(n_modes: Tuple[int, int], hidden_channels: int, n_layers: int, 
                   domain_padding: List[float], domain_padding_mode: str, device: str):
    """Build TFNO2d model with given hyperparameters.
    
    Args:
        n_modes: Number of modes for each spatial dimension (height, width)
        hidden_channels: Number of hidden channels
        n_layers: Number of layers
        domain_padding: Domain padding values
        domain_padding_mode: Padding mode
        device: Device to place model on
        
    Returns:
        Configured TFNO2d model
    """
    model = TFNO2d(
        n_modes_height=n_modes[0],
        n_modes_width=n_modes[1],
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
        use_channel_mlp=False
    ).to(device)
    return model

def setup_data_loaders_2d(train_dataset: CustomDataset2D, test_dataset: CustomDataset2D, 
                          batch_size: int) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Setup train and test data loaders for 2D FNO.
    
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

@torch.no_grad()
def plot_compare_2d(pred_phys: torch.Tensor, gt_phys: torch.Tensor, save_path: str, 
                    sample_num: int = 0) -> None:
    """Plot comparison between 2D spatial predictions and ground truth.
    
    Args:
        pred_phys: Predicted physical values of shape (N, 1, nx, ny)
        gt_phys: Ground truth physical values of shape (N, 1, nx, ny)
        save_path: Path to save the comparison plot
        sample_num: Sample index to visualize
    """
    pi = pred_phys[sample_num, 0, :, :].cpu().numpy()
    gi = gt_phys[sample_num, 0, :, :].cpu().numpy()
    er = np.abs(pi - gi)

    vmin = min(np.min(pi), np.min(gi))
    vmax = max(np.max(pi), np.max(gi))
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im1 = axes[0].imshow(gi, vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    im2 = axes[1].imshow(pi, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    im3 = axes[2].imshow(er)
    axes[2].set_title("Absolute Error")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['VISUALIZATION']['DPI'])
    plt.close(fig)
    print(f"Saved 2D comparison plot: {save_path}")

# ==============================================================================
# Data Processing and Training Functions
# ==============================================================================
def prepare_data_and_normalizers_2d(merged_pt_path: str) -> Tuple:
    """Prepare data and normalizers for 2D FNO with uniform meta channels.
    
    Args:
        merged_pt_path: Path to merged data file
        
    Returns:
        Tuple containing tensors, normalizers, processor, and datasets
    """
    in_summation_2d, out_summation_2d, meta_summation = load_merged_tensors_2d(merged_pt_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    in_summation_2d = in_summation_2d.to(device)
    out_summation_2d = out_summation_2d.to(device)
    meta_summation = meta_summation.to(device)
    
    out_summation_2d = 10 ** out_summation_2d
    out_summation_2d[:, :, 14:18, 14:18] = 0
    
    # Combine input with uniform meta channels for full dataset
    combined_input_2d = expand_meta_to_uniform_channels_2d(in_summation_2d, meta_summation)
    
    # Create normalizers for combined input and output (2D dimensions)
    in_normalizer = UnitGaussianNormalizer(mean=combined_input_2d, std=combined_input_2d, dim=[0,2,3], eps=1e-6)
    out_normalizer = UnitGaussianNormalizer(mean=out_summation_2d, std=out_summation_2d, dim=[0,2,3], eps=1e-6)
    
    in_normalizer.fit(combined_input_2d)
    out_normalizer.fit(out_summation_2d)
    
    processor = DefaultDataProcessor(in_normalizer, out_normalizer).to(device)
    
    # Split the original tensors, not the combined one
    train_in, test_in, train_out, test_out, train_meta, test_meta = train_test_split(
        in_summation_2d, out_summation_2d, meta_summation, 
        test_size=CONFIG['TEST_SIZE'], 
        random_state=CONFIG['RANDOM_STATE']
    )
    
    train_dataset = CustomDataset2D(train_in, train_out, train_meta)
    test_dataset = CustomDataset2D(test_in, test_out, test_meta)
    
    return (combined_input_2d, out_summation_2d, device, 
            in_normalizer, out_normalizer, processor, 
            train_dataset, test_dataset)

def create_objective_function_2d(train_dataset: CustomDataset2D, test_dataset: CustomDataset2D, 
                                 processor: DefaultDataProcessor, device: str) -> callable:
    """Create Optuna objective function for 2D FNO.
    
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

        train_loader, test_loader = setup_data_loaders_2d(train_dataset, test_dataset, train_batch_size)
        
        model = build_model_2d(n_modes, hidden_channels, n_layers, domain_padding, 
                              CONFIG['DOMAIN_PADDING_MODE'], device)
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=l2_weight)
        scheduler = create_scheduler(optimizer)

        loss_fn, loss_name = create_loss_function()
        trainer = Trainer(
            model=model, n_epochs=CONFIG['N_EPOCHS'], device=device,
            data_processor=processor, wandb_log=False,
            eval_interval=CONFIG['EVAL_INTERVAL'], use_distributed=False, verbose=True
        )

        best_model_path = Path(CONFIG['OUTPUT_DIR']) / 'optuna'
        best_model_path.mkdir(parents=True, exist_ok=True)

        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            early_stopping=CONFIG['SCHEDULER_CONFIG']['early_stopping'],
            training_loss=loss_fn,
            eval_losses={loss_name: loss_fn},
            save_best=f'test_dataloader_{loss_name}',
            save_dir=str(best_model_path)
        )

        model.load_state_dict(torch.load(
            best_model_path / 'best_model_state_dict.pt', 
            map_location=device, weights_only=False
        ))
        model.eval()

        with torch.no_grad():
            test_batch = next(iter(test_loader['test_dataloader']))
            x = test_batch["x"].to(device)
            y = test_batch["y"].to(device)
            pred = model(x)
            val_loss = loss_fn(pred, y).item()
        
        return val_loss
    
    return objective

def train_final_model_2d(best_params: Dict[str, Any], train_dataset: CustomDataset2D, 
                         test_dataset: CustomDataset2D, processor: DefaultDataProcessor,
                         in_normalizer, out_normalizer, device: str) -> None:
    """Train final 2D model with best parameters and generate comparison plot.
    
    Args:
        best_params: Best hyperparameters from optimization
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Data processor
        in_normalizer: Input normalizer
        out_normalizer: Output normalizer  
        device: Device to use
    """
    best_model = build_model_2d(
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
    trainer = Trainer(
        model=best_model, 
        n_epochs=CONFIG['N_EPOCHS'], 
        device=device,
        data_processor=processor, 
        wandb_log=False,
        eval_interval=CONFIG['EVAL_INTERVAL'], 
        use_distributed=False, 
        verbose=True
    )
    
    train_loader, test_loader = setup_data_loaders_2d(
        train_dataset, test_dataset, best_params["train_batch_size"]
    )

    final_model_path = Path(CONFIG['OUTPUT_DIR']) / 'final'
    final_model_path.mkdir(parents=True, exist_ok=True)

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        early_stopping=CONFIG['SCHEDULER_CONFIG']['early_stopping'],
        training_loss=loss_fn,
        eval_losses={loss_name: loss_fn},
        save_best=f'test_dataloader_{loss_name}',
        save_dir=str(final_model_path)
    )
    
    best_model.load_state_dict(torch.load(
        final_model_path / 'best_model_state_dict.pt', 
        map_location=device, weights_only=False
    ))
    best_model.eval()

    with torch.no_grad():
        test_batch = next(iter(test_loader["test_dataloader"]))
        x = test_batch["x"].to(device) 
        y = test_batch["y"].to(device)
        
        pred = best_model(in_normalizer.transform(x))
        
        pred_phys = out_normalizer.inverse_transform(pred).detach().cpu()
        gt_phys = y.detach().cpu()

        pred_phys[:, :, 14:18, 14:18] = 0
        gt_phys[:, :, 14:18, 14:18] = 0

    output_path = Path(CONFIG['OUTPUT_DIR']) / 'FNO_2d_compare.png'
    plot_compare_2d(
        pred_phys, gt_phys, 
        save_path=str(output_path), 
        sample_num=CONFIG['VISUALIZATION']['SAMPLE_NUM']
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

def run_optuna_training_2d(train_dataset: CustomDataset2D, test_dataset: CustomDataset2D, 
                           processor: DefaultDataProcessor, device: str) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization for 2D FNO.
    
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
    
    objective_fn = create_objective_function_2d(train_dataset, test_dataset, processor, device)
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

def run_eval_mode_2d(train_dataset: CustomDataset2D, test_dataset: CustomDataset2D, 
                     processor: DefaultDataProcessor, 
                     in_normalizer, out_normalizer, device: str) -> None:
    """Run evaluation mode using pre-trained 2D model.
    
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
    model = build_model_2d(
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
    
    # Generate predictions and create comparison plot
    with torch.no_grad():
        test_batch = next(iter(test_loader["test_dataloader"]))
        x = test_batch["x"].to(device) 
        y = test_batch["y"].to(device)
        
        pred = model(in_normalizer.transform(x))
        
        pred[:, :, 14:18, 14:18] = 0
        y[:, :, 14:18, 14:18] = 0
        
        pred_phys = out_normalizer.inverse_transform(pred).detach().cpu()
        gt_phys = y.detach().cpu()

    output_path = Path(CONFIG['OUTPUT_DIR']) / 'FNO_2d_eval_compare.png'
    plot_compare_2d(
        pred_phys, gt_phys, 
        save_path=str(output_path), 
        sample_num=CONFIG['VISUALIZATION']['SAMPLE_NUM']
    )
    
    print(f"✅ Evaluation completed! Results saved to: {output_path}")

def run_training_mode_2d(train_dataset: CustomDataset2D, test_dataset: CustomDataset2D, 
                         processor: DefaultDataProcessor, 
                         in_normalizer, out_normalizer, device: str) -> None:
    """Run training based on configured mode for 2D FNO.
    
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
        train_final_model_2d(
            best_params, train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
    elif mode == 'optuna':
        best_params = run_optuna_training_2d(train_dataset, test_dataset, processor, device)
        print("\n" + "=" * 50)
        print("STARTING FINAL MODEL TRAINING")
        print("=" * 50)
        train_final_model_2d(
            best_params, train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
    elif mode == 'eval':
        run_eval_mode_2d(
            train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
    else:
        raise ValueError(f"Unknown training mode: {mode}. Options: 'single', 'optuna', 'eval'")

def main() -> None:
    """Main training pipeline for 2D FNO with spatial-only predictions."""
    try:
        data_results = prepare_data_and_normalizers_2d(CONFIG['MERGED_PT_PATH'])
        (
            combined_input_2d, out_summation_2d, device,
            in_normalizer, out_normalizer, processor,
            train_dataset, test_dataset
        ) = data_results
        
        print(f"\n🚀 FNO-2D Training Pipeline Started")
        print(f"Training Mode: {CONFIG['TRAINING_CONFIG']['mode'].upper()}")
        
        run_training_mode_2d(
            train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, device
        )
        
        print("\n✅ 2D Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()