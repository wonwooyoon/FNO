"""
Fourier Neural Operator (FNO) Training with Optuna Hyperparameter Optimization

This module implements training pipeline for TFNO models using Optuna TPE sampler
for hyperparameter optimization. The pipeline includes data loading, normalization,
model training, and result visualization.
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
from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop import Trainer
from neuraloperator.neuralop.training import AdamW


# Configuration Constants
CONFIG = {
    'MERGED_PT_PATH': './src/preprocessing/merged.pt',
    'OUTPUT_DIR': './src/FNO/output',
    'N_EPOCHS': 10,
    'EVAL_INTERVAL': 1,
    'TEST_SIZE': 0.1,
    'RANDOM_STATE': 42,
    'DOMAIN_PADDING_MODE': 'symmetric',
    'MODEL_CONFIG': {
        'in_channels': 8,
        'out_channels': 1,
        'lifting_channel_ratio': 2,
        'projection_channel_ratio': 2,
        'positional_embedding': 'grid',
        'film_layer': True,
        'meta_dim': 2
    },
    'SCHEDULER_CONFIG': {
        'scheduler_type': 'step',  # Options: 'cosine', 'step'
        'T_0': 10,
        'T_max': 80,
        'T_mult': 2,
        'eta_min': 1e-6,
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
    }
}

# ==============================================================================
# Data Classes and Dataset
# ==============================================================================
class CustomDataset(Dataset):
    """Custom dataset for FNO training.
    
    Args:
        input_tensor: Input tensor of shape (N, 9, nx, ny, nt)
        output_tensor: Output tensor of shape (N, 1, nx, ny, nt) 
        meta_tensor: Metadata tensor of shape (N, 2)
    """
    
    def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, meta_tensor: torch.Tensor):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.meta_tensor = meta_tensor
        
    def __len__(self) -> int:
        return self.input_tensor.shape[0]
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': self.input_tensor[idx], 
            'y': self.output_tensor[idx], 
            'meta': self.meta_tensor[idx]
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
def load_merged_tensors(merged_pt_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load merged tensors from .pt file.
    
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
        
        print(f"Loaded merged tensors: {tuple(in_summation.shape)}, {tuple(out_summation.shape)}, {tuple(meta_summation.shape)}")
        return in_summation, out_summation, meta_summation
        
    except Exception as e:
        print(f"Error loading merged tensors: {e}")
        raise

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
        meta_dim=CONFIG['MODEL_CONFIG']['meta_dim'],
        use_channel_mlp=False
    ).to(device)
    return model

def setup_data_loaders(train_dataset: CustomDataset, test_dataset: CustomDataset, 
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
def prepare_data_and_normalizers(merged_pt_path: str) -> Tuple:
    """Prepare data and normalizers.
    
    Args:
        merged_pt_path: Path to merged data file
        
    Returns:
        Tuple containing tensors, normalizers, processor, and datasets
    """
    in_summation, out_summation, meta_summation = load_merged_tensors(merged_pt_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    in_summation = in_summation.to(device)
    out_summation = out_summation.to(device)
    meta_summation = meta_summation.to(device)
    
    out_summation = 10 ** out_summation
    
    in_normalizer = UnitGaussianNormalizer(mean=in_summation, std=in_summation, dim=[0,2,3,4], eps=1e-6)
    out_normalizer = UnitGaussianNormalizer(mean=out_summation, std=out_summation, dim=[0,2,3,4], eps=1e-6)
    meta_normalizer = UnitGaussianNormalizer(mean=meta_summation, std=meta_summation, dim=[0], eps=1e-6)
    
    in_normalizer.fit(in_summation)
    out_normalizer.fit(out_summation)
    meta_normalizer.fit(meta_summation)
    
    processor = DefaultDataProcessor(in_normalizer, out_normalizer, meta_normalizer).to(device)
    
    train_in, test_in, train_out, test_out, train_meta, test_meta = train_test_split(
        in_summation, out_summation, meta_summation, 
        test_size=CONFIG['TEST_SIZE'], 
        random_state=CONFIG['RANDOM_STATE']
    )
    
    train_dataset = CustomDataset(train_in, train_out, train_meta)
    test_dataset = CustomDataset(test_in, test_out, test_meta)
    
    return (in_summation, out_summation, meta_summation, device, 
            in_normalizer, out_normalizer, meta_normalizer, processor, 
            train_dataset, test_dataset)

def create_objective_function(train_dataset: CustomDataset, test_dataset: CustomDataset, 
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
        n_modes = trial.suggest_categorical("n_modes", [(16,16,5), (16,8,5)])
        hidden_channels = trial.suggest_categorical("hidden_channels", [8])
        n_layers = trial.suggest_categorical("n_layers", [2])
        domain_padding = trial.suggest_categorical("domain_padding", [[0.125,0.25,0.4]])
        train_batch_size = trial.suggest_categorical("train_batch_size", [64])
        l2_weight = trial.suggest_float("l2_weight", 1e-8, 1e-3, log=True)
        initial_lr = trial.suggest_float("initial_lr", 1e-4, 1e-3, log=True)

        train_loader, test_loader = setup_data_loaders(train_dataset, test_dataset, train_batch_size)
        
        model = build_model(n_modes, hidden_channels, n_layers, domain_padding, 
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
            early_stopping=True,
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

def train_final_model(best_params: Dict[str, Any], train_dataset: CustomDataset, 
                     test_dataset: CustomDataset, processor: DefaultDataProcessor,
                     in_normalizer, out_normalizer, meta_normalizer, device: str) -> None:
    """Train final model with best parameters and generate comparison plot.
    
    Args:
        best_params: Best hyperparameters from optimization
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Data processor
        in_normalizer: Input normalizer
        out_normalizer: Output normalizer  
        meta_normalizer: Meta normalizer
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
    print(best_model)

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
    
    train_loader, test_loader = setup_data_loaders(
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
        early_stopping=True,
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
        meta = test_batch["meta"].to(device)
        
        pred = best_model(in_normalizer.transform(x), meta_normalizer.transform(meta))
        
        pred[:, :, 14:18, 14:18, :] = 0
        y[:, :, 14:18, 14:18, :] = 0
        
        pred_phys = out_normalizer.inverse_transform(pred).detach().cpu()
        gt_phys = y.detach().cpu()

    output_path = Path(CONFIG['OUTPUT_DIR']) / 'FNO_compare.png'
    plot_compare(
        pred_phys, gt_phys, 
        save_path=str(output_path), 
        sample_num=CONFIG['VISUALIZATION']['SAMPLE_NUM'], 
        t_indices=CONFIG['VISUALIZATION']['TIME_INDICES']
    )
def main() -> None:
    """Main training pipeline with hyperparameter optimization."""
    try:
        data_results = prepare_data_and_normalizers(CONFIG['MERGED_PT_PATH'])
        (
            in_summation, out_summation, meta_summation, device,
            in_normalizer, out_normalizer, meta_normalizer, processor,
            train_dataset, test_dataset
        ) = data_results
        
        # Example: Run with predefined best parameters
        # For actual hyperparameter optimization, uncomment the Optuna section below
        best_params = {
            "n_modes": (16, 8, 5), 
            "hidden_channels": 24, 
            "n_layers": 3, 
            "domain_padding": [0.1, 0.1, 0.1], 
            "train_batch_size": 16, 
            "l2_weight": 0, 
            "initial_lr": 1e-4
        }
        
        # Uncomment for actual hyperparameter optimization:
        # objective_fn = create_objective_function(train_dataset, test_dataset, processor, device)
        # sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=2)
        # study = optuna.create_study(direction="minimize", sampler=sampler)
        # study.optimize(objective_fn, n_trials=10, show_progress_bar=True)
        # print(f"\nOptuna Best Value: {study.best_value}")
        # print(f"Optuna Best Params: {study.best_params}")
        # best_params = study.best_params
        
        train_final_model(
            best_params, train_dataset, test_dataset, processor,
            in_normalizer, out_normalizer, meta_normalizer, device
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()