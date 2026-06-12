"""
Unified Output Utility for FNO Models

This module consolidates all output-related functions including:
- Image visualization (combined grids and separated images)
- GIF generation for temporal evolution
- Detailed evaluation metrics (Relative L2, SSIM, parity plots)
- Integrated Gradients attribution analysis

All outputs are organized into subdirectories for better file management.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim


# ==============================================================================
# Section 0: Directory Setup
# ==============================================================================

def setup_output_directories(base_dir: Path, config: Dict) -> Dict[str, Path]:
    """
    Create output directory structure based on enabled features.

    Args:
        base_dir: Base output directory path
        config: Configuration dictionary containing OUTPUT settings

    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_dir = Path(base_dir)
    dirs = {'base': base_dir}

    output_config = config.get('OUTPUT', {})

    # Image output directories
    if output_config.get('IMAGE_OUTPUT', {}).get('ENABLED', False):
        dirs['images'] = base_dir / 'images'
        dirs['images_combined'] = base_dir / 'images' / 'combined'
        dirs['images_separated'] = base_dir / 'images' / 'separated'

    # GIF output directory
    if output_config.get('GIF_OUTPUT', {}).get('ENABLED', False):
        dirs['gifs'] = base_dir / 'gifs'

    # Metrics directory
    if output_config.get('DETAIL_EVAL', {}).get('ENABLED', False):
        dirs['metrics'] = base_dir / 'metrics'

    # Integrated Gradients directory
    if output_config.get('IG_ANALYSIS', {}).get('ENABLED', False):
        dirs['ig'] = base_dir / 'integrated_gradients'
        sample_idx = output_config['IG_ANALYSIS'].get('SAMPLE_IDX', 0)
        dirs['ig_sample'] = dirs['ig'] / f'sample_{sample_idx}'

    # Create all directories
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ==============================================================================
# Section 1: Image Output Functions
# ==============================================================================

def visualize_combined_grid(
    pred_sample: np.ndarray,
    gt_sample: np.ndarray,
    sample_idx: int,
    time_indices: List[int],
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> Path:
    """
    Create 3×4 grid showing GT, Prediction, and Error over time.

    Args:
        pred_sample: Prediction array (nx, ny, nt)
        gt_sample: Ground truth array (nx, ny, nt)
        sample_idx: Sample index
        time_indices: Time indices to visualize
        output_dir: Directory to save the image (images/combined/)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        Path to saved image
    """
    # Create 3×4 grid
    fig, axes = plt.subplots(3, len(time_indices), figsize=(4*len(time_indices), 8))

    # Determine shared color scale for GT and Prediction
    vmin_gt_pred = min(gt_sample[:, :, time_indices].min(),
                       pred_sample[:, :, time_indices].min())
    vmax_gt_pred = max(gt_sample[:, :, time_indices].max(),
                       pred_sample[:, :, time_indices].max())

    # Calculate error
    error_sample = gt_sample - pred_sample
    error_max_abs = np.abs(error_sample[:, :, time_indices]).max()

    im_pred, im_err = None, None
    for i, t_idx in enumerate(time_indices):
        # Plot Ground Truth (Row 1)
        ax_gt = axes[0, i]
        im_gt = ax_gt.imshow(gt_sample[:, :, t_idx].T, cmap='RdBu_r',
                            vmin=vmin_gt_pred, vmax=vmax_gt_pred)
        ax_gt.set_title(f"Ground Truth (t={t_idx})")
        ax_gt.axis('off')

        # Plot Prediction (Row 2)
        ax_pred = axes[1, i]
        im_pred = ax_pred.imshow(pred_sample[:, :, t_idx].T, cmap='RdBu_r',
                                vmin=vmin_gt_pred, vmax=vmax_gt_pred)
        ax_pred.set_title(f"Prediction (t={t_idx})")
        ax_pred.axis('off')

        # Plot Error (Row 3)
        ax_err = axes[2, i]
        im_err = ax_err.imshow(error_sample[:, :, t_idx].T, cmap='coolwarm',
                              vmin=-error_max_abs, vmax=error_max_abs)
        ax_err.set_title(f"Error (t={t_idx})")
        ax_err.axis('off')

    # Add shared colorbars
    if im_pred:
        fig.colorbar(im_pred, ax=axes[0:2, :].ravel().tolist(),
                    orientation='horizontal', pad=0.05, aspect=40)
    if im_err:
        fig.colorbar(im_err, ax=axes[2, :].ravel().tolist(),
                    orientation='horizontal', pad=0.05, aspect=40)

    # Save
    output_path = output_dir / f'sample_{sample_idx}_grid.png'
    dpi = config.get('OUTPUT', {}).get('DPI', 200)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved combined grid: {output_path.name}")

    return output_path


def visualize_separated_images(
    pred_sample: np.ndarray,
    gt_sample: np.ndarray,
    sample_idx: int,
    time_indices: List[int],
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> List[Path]:
    """
    Create separate image files for each time and type (GT, Pred, Error).

    Args:
        pred_sample: Prediction array (nx, ny, nt)
        gt_sample: Ground truth array (nx, ny, nt)
        sample_idx: Sample index
        time_indices: Time indices to visualize
        output_dir: Directory to save images (images/separated/)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        List of paths to saved images
    """
    saved_paths = []

    # Determine color scales
    vmin_gt_pred = min(gt_sample[:, :, time_indices].min(),
                       pred_sample[:, :, time_indices].min())
    vmax_gt_pred = max(gt_sample[:, :, time_indices].max(),
                       pred_sample[:, :, time_indices].max())

    error_sample = gt_sample - pred_sample
    error_max_abs = np.abs(error_sample[:, :, time_indices]).max()

    dpi = config.get('OUTPUT', {}).get('DPI', 200)

    for t_idx in time_indices:
        # Ground Truth
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(gt_sample[:, :, t_idx].T, cmap='RdBu_r',
                      vmin=vmin_gt_pred, vmax=vmax_gt_pred)
        ax.set_title(f"Ground Truth (Sample {sample_idx}, t={t_idx})")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        path = output_dir / f'sample_{sample_idx}_t{t_idx:02d}_gt.png'
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)

        # Prediction
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(pred_sample[:, :, t_idx].T, cmap='RdBu_r',
                      vmin=vmin_gt_pred, vmax=vmax_gt_pred)
        ax.set_title(f"Prediction (Sample {sample_idx}, t={t_idx})")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        path = output_dir / f'sample_{sample_idx}_t{t_idx:02d}_pred.png'
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)

        # Error
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(error_sample[:, :, t_idx].T, cmap='coolwarm',
                      vmin=-error_max_abs, vmax=error_max_abs)
        ax.set_title(f"Error (Sample {sample_idx}, t={t_idx})")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        path = output_dir / f'sample_{sample_idx}_t{t_idx:02d}_error.png'
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)

    if verbose:
        print(f"  Saved {len(saved_paths)} separated images")

    return saved_paths


# ==============================================================================
# Section 2: GIF Generation Functions
# ==============================================================================

def create_single_type_gif(
    data: np.ndarray,
    data_type: str,
    sample_idx: int,
    output_dir: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    config: Dict,
    verbose: bool = True
) -> Path:
    """
    Create an animated GIF for a single data type (GT, Prediction, or Error).

    Args:
        data: Data array of shape (nx, ny, nt)
        data_type: Type of data - 'gt', 'pred', or 'error'
        sample_idx: Index of the sample
        output_dir: Directory where the GIF will be saved (gifs/)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        cmap: Colormap name
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        Path to the saved GIF file
    """
    nx, ny, nt = data.shape
    aspect_ratio = ny / nx
    fig_width = 8
    fig_height = fig_width * aspect_ratio

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Initialize image
    im = ax.imshow(
        data[:, :, 0].T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
        interpolation='nearest'
    )

    def init():
        im.set_array(data[:, :, 0].T)
        return [im]

    def animate(frame_idx):
        im.set_array(data[:, :, frame_idx].T)
        return [im]

    # Create animation - always use all time indices
    fps = config.get('OUTPUT', {}).get('GIF_OUTPUT', {}).get('FPS', 2)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=nt,  # All time steps
        interval=1000/fps,
        blit=True,
        repeat=True
    )

    # Save
    gif_filename = f'sample_{sample_idx}_{data_type}.gif'
    gif_path = output_dir / gif_filename

    writer = animation.PillowWriter(fps=fps)
    dpi = config.get('OUTPUT', {}).get('DPI', 100)
    anim.save(gif_path, writer=writer, dpi=dpi)

    plt.close(fig)

    if verbose:
        print(f"    {data_type.upper()} GIF saved: {gif_filename}")

    return gif_path


def create_colorbar_png(
    vmin: float,
    vmax: float,
    cmap: str,
    label: str,
    output_path: Path,
    config: Dict,
    orientation: str = 'horizontal',
    verbose: bool = True
) -> Path:
    """
    Create a standalone colorbar as a PNG file.

    Args:
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        cmap: Colormap name
        label: Label for the colorbar
        output_path: Full path where the PNG will be saved
        config: Configuration dictionary
        orientation: 'horizontal' or 'vertical'
        verbose: Whether to print progress

    Returns:
        Path to the saved colorbar PNG
    """
    if orientation == 'horizontal':
        fig, ax = plt.subplots(figsize=(8, 1))
    else:
        fig, ax = plt.subplots(figsize=(1, 8))

    # Create colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation=orientation
    )
    cb.set_label(label, fontsize=12)

    # Save
    dpi = config.get('OUTPUT', {}).get('DPI', 200)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"    Colorbar saved: {output_path.name}")

    return output_path


def create_all_gifs(
    gt_sample: np.ndarray,
    pred_sample: np.ndarray,
    sample_idx: int,
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Create GIFs and colorbars for GT, Prediction, and Error.
    Always uses all time indices.

    Args:
        gt_sample: Ground truth array (nx, ny, nt)
        pred_sample: Prediction array (nx, ny, nt)
        sample_idx: Sample index
        output_dir: Directory to save files (gifs/)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        Dictionary containing paths to generated files
    """
    if verbose:
        print(f"  Creating GIFs for sample {sample_idx}...")

    # Calculate error
    error_sample = gt_sample - pred_sample

    # Determine color scales
    vmin_gt_pred = min(gt_sample.min(), pred_sample.min())
    vmax_gt_pred = max(gt_sample.max(), pred_sample.max())
    error_max_abs = np.abs(error_sample).max()
    vmin_error = -error_max_abs
    vmax_error = error_max_abs

    output_paths = {}

    # Create GT GIF
    output_paths['gt_gif'] = create_single_type_gif(
        data=gt_sample,
        data_type='gt',
        sample_idx=sample_idx,
        output_dir=output_dir,
        vmin=vmin_gt_pred,
        vmax=vmax_gt_pred,
        cmap='RdBu_r',
        config=config,
        verbose=verbose
    )

    # Create Prediction GIF
    output_paths['pred_gif'] = create_single_type_gif(
        data=pred_sample,
        data_type='pred',
        sample_idx=sample_idx,
        output_dir=output_dir,
        vmin=vmin_gt_pred,
        vmax=vmax_gt_pred,
        cmap='RdBu_r',
        config=config,
        verbose=verbose
    )

    # Create Error GIF
    output_paths['error_gif'] = create_single_type_gif(
        data=error_sample,
        data_type='error',
        sample_idx=sample_idx,
        output_dir=output_dir,
        vmin=vmin_error,
        vmax=vmax_error,
        cmap='coolwarm',
        config=config,
        verbose=verbose
    )

    # Create colorbar for GT and Prediction
    colorbar_gt_pred_path = output_dir / f'sample_{sample_idx}_colorbar_gt_pred.png'
    output_paths['gt_pred_colorbar'] = create_colorbar_png(
        vmin=vmin_gt_pred,
        vmax=vmax_gt_pred,
        cmap='RdBu_r',
        label='Concentration',
        output_path=colorbar_gt_pred_path,
        config=config,
        orientation='horizontal',
        verbose=verbose
    )

    # Create colorbar for Error
    colorbar_error_path = output_dir / f'sample_{sample_idx}_colorbar_error.png'
    output_paths['error_colorbar'] = create_colorbar_png(
        vmin=vmin_error,
        vmax=vmax_error,
        cmap='coolwarm',
        label='Error (GT - Pred)',
        output_path=colorbar_error_path,
        config=config,
        orientation='horizontal',
        verbose=verbose
    )

    if verbose:
        print(f"    Completed: 3 GIFs + 2 colorbars")

    return output_paths


# ==============================================================================
# Section 3: Detailed Evaluation (Metrics) Functions
# ==============================================================================

def compute_ssim_per_time(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute SSIM for each time index of a single sample.

    Args:
        pred: Prediction array of shape (nx, ny, nt)
        gt: Ground truth array of shape (nx, ny, nt)

    Returns:
        Array of SSIM values of shape (nt,)
    """
    nx, ny, nt = pred.shape
    ssim_values = np.zeros(nt)

    for t in range(nt):
        pred_t = pred[:, :, t]
        gt_t = gt[:, :, t]

        # Compute data range
        data_range = max(gt_t.max() - gt_t.min(), pred_t.max() - pred_t.min())

        if data_range < 1e-10:
            ssim_values[t] = 1.0  # Perfect similarity for constant images
        else:
            ssim_values[t] = ssim(
                gt_t,
                pred_t,
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False
            )

    return ssim_values


def compute_case_relative_l2(
    pred: np.ndarray,
    gt: np.ndarray
) -> float:
    """
    Compute case-wise Relative L2 error on physical values.

    Matches the training loss definition for p=2:
        ||pred - gt||_2 / (||gt||_2 + eps)

    Args:
        pred: Prediction array of shape (nx, ny, nt)
        gt: Ground truth array of shape (nx, ny, nt)

    Returns:
        Relative L2 value for the case (scalar in range [0, ∞))
    """
    eps = 0
    diff_norm = np.sqrt(np.sum((pred - gt) ** 2))
    gt_norm = np.sqrt(np.sum(gt ** 2))
    return float(diff_norm / (gt_norm + eps))


def compute_relative_l2_per_time(
    pred: np.ndarray,
    gt: np.ndarray
) -> np.ndarray:
    """
    Compute time-wise Relative L2 on physical values.

    For each time t:
        RelL2_t = ||pred_t - gt_t||_2 / (||gt_t||_2 + eps)

    Args:
        pred: Prediction array of shape (nx, ny, nt)
        gt: Ground truth array of shape (nx, ny, nt)

    Returns:
        Array RelL2_t of shape (nt,)
    """
    eps = 0
    diff_norm_per_time = np.sqrt(np.sum((pred - gt) ** 2, axis=(0, 1)))
    gt_norm_per_time = np.sqrt(np.sum(gt ** 2, axis=(0, 1)))
    return diff_norm_per_time / (gt_norm_per_time + eps)


def compute_r2_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) score.

    R² measures the proportion of variance in the ground truth that is
    predictable from the model. It is a standard metric for regression
    model quality.

    Formula:
        R² = 1 - (SS_res / SS_tot)
        where:
            SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
            SS_tot = Σ(y_true - y_mean)²  (total sum of squares)

    Args:
        pred: Prediction array of shape (nx, ny, nt)
        gt: Ground truth array of shape (nx, ny, nt)

    Returns:
        R² score (scalar, typically in range (-∞, 1])
        - R² = 1.0: Perfect prediction (all variance explained)
        - R² = 0.0: Model as good as mean baseline
        - R² < 0.0: Model worse than mean baseline
    """
    # Flatten arrays for scalar computation
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    # Compute mean of ground truth
    gt_mean = np.mean(gt_flat)

    # Sum of squared residuals (prediction error)
    ss_res = np.sum((gt_flat - pred_flat) ** 2)

    # Total sum of squares (variance in ground truth)
    ss_tot = np.sum((gt_flat - gt_mean) ** 2)

    # Handle edge case: zero variance in ground truth
    if ss_tot < 1e-20:
        # If ground truth is constant and prediction matches, R² = 1
        return 1.0 if ss_res < 1e-20 else 0.0

    # Compute R² score
    r2 = 1.0 - (ss_res / ss_tot)

    return r2


def compute_global_r2_score(
    pred_phys: np.ndarray,
    gt_phys: np.ndarray
) -> float:
    """
    Compute global R² score across ALL samples.

    Unlike computing R² per sample and averaging them, this function
    computes a single R² value by treating all samples as one large
    dataset. This is mathematically correct for evaluating overall
    model performance across the entire test set.

    Mathematical difference:
        Method 1 (incorrect): Mean(R²_i) = (1/N) Σ R²_i
        Method 2 (correct):   R²_global = 1 - (Σ SS_res_i / Σ SS_tot_i)

    These are NOT equivalent because:
        - SS_res is linear: Σ SS_res_i = SS_res_global ✓
        - SS_tot is NOT linear: Σ SS_tot_i ≠ SS_tot_global ✗
          (each sample has different mean)

    Args:
        pred_phys: Predictions of shape (N, nx, ny, nt)
        gt_phys: Ground truth of shape (N, nx, ny, nt)

    Returns:
        Global R² score (scalar, typically in range (-∞, 1])
        - R² = 1.0: Perfect prediction across all samples
        - R² = 0.0: Model as good as global mean baseline
        - R² < 0.0: Model worse than global mean baseline

    Note:
        This value is typically higher than the mean of per-sample R²
        values because it also accounts for inter-sample variance.
    """
    # Flatten ALL data (concatenate all samples into one long vector)
    pred_all = pred_phys.flatten()  # (N×nx×ny×nt,)
    gt_all = gt_phys.flatten()

    # Compute GLOBAL mean (mean of all data points across all samples)
    gt_mean_global = np.mean(gt_all)

    # Global sum of squared residuals
    ss_res_global = np.sum((gt_all - pred_all) ** 2)

    # Global total sum of squares
    ss_tot_global = np.sum((gt_all - gt_mean_global) ** 2)

    # Handle edge case: zero variance in ground truth
    if ss_tot_global < 1e-20:
        return 1.0 if ss_res_global < 1e-20 else 0.0

    # Compute global R² score
    r2_global = 1.0 - (ss_res_global / ss_tot_global)

    return r2_global


def add_mean_column(df: pd.DataFrame, exclude_col: str = 'time') -> pd.DataFrame:
    """
    Add a mean column to the DataFrame, excluding specified columns.

    Args:
        df: DataFrame with time as index or column
        exclude_col: Column name to exclude from mean calculation

    Returns:
        DataFrame with added 'mean' column
    """
    # Get numeric columns (exclude 'time' or other non-sample columns)
    numeric_cols = [col for col in df.columns if col != exclude_col]
    
    # Calculate mean across samples
    df['mean'] = df[numeric_cols].mean(axis=1)
    
    return df


def generate_parity_csv(
    pred_phys: torch.Tensor,
    gt_phys: torch.Tensor,
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> List[Path]:
    """
    Generate parity plot data (GT vs Prediction) for each time index.
    Saves separate CSV files per time index and a combined parity plot image.

    Args:
        pred_phys: Predictions (N, C, nx, ny, nt)
        gt_phys: Ground truth (N, C, nx, ny, nt)
        output_dir: Directory to save CSV files (metrics/)
        config: Configuration dictionary containing DPI settings
        verbose: Whether to print progress

    Returns:
        List of paths to saved CSV files
    """
    pred_np = pred_phys[:, 0].detach().cpu().numpy()  # (N, nx, ny, nt)
    gt_np = gt_phys[:, 0].detach().cpu().numpy()

    n_samples, nx, ny, n_time = pred_np.shape
    saved_paths = []

    for t_idx in range(n_time):
        # Extract all pixels for this time index
        gt_t = gt_np[:, :, :, t_idx].flatten()
        pred_t = pred_np[:, :, :, t_idx].flatten()

        # Create DataFrame
        parity_df = pd.DataFrame({
            'ground_truth': gt_t,
            'prediction': pred_t
        })

        # Save
        csv_path = output_dir / f'parity_t{t_idx:02d}.csv'
        parity_df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

    if verbose:
        print(f"  Saved {len(saved_paths)} parity plot CSV files")

    # Generate combined parity plot for selected time indices
    # Plot in reverse order so t=0 appears in front
    selected_times = [19, 14, 9, 4, 0]
    # Continuous color gradient from dark to light
    colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef']

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')

    # Collect all data points to determine axis limits
    all_gt = []
    all_pred = []

    for i, t_idx in enumerate(selected_times):
        if t_idx < n_time:
            gt_t = gt_np[:, :, :, t_idx].flatten()
            pred_t = pred_np[:, :, :, t_idx].flatten()
            all_gt.extend(gt_t)
            all_pred.extend(pred_t)

            # Scatter plot for this time index
            ax.scatter(gt_t, pred_t, c=colors[i], alpha=0.3, s=10,
                      label=f't={t_idx}', edgecolors='none')

    # Determine axis limits from data
    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    min_val = min(all_gt.min(), all_pred.min())
    max_val = max(all_gt.max(), all_pred.max())

    # Add 5% margin for better visualization
    margin = (max_val - min_val) * 0.05
    plot_min = max(0, min_val - margin)  # Don't go below 0 for concentrations
    plot_max = max_val + margin

    # Plot 1:1 line
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'y--', linewidth=2, label='1:1 line')

    # Set axis range and ticks (dynamic)
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)

    # Create approximately 5 ticks
    tick_interval = (plot_max - plot_min) / 5
    ticks = np.arange(plot_min, plot_max + tick_interval/2, tick_interval)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Set labels with larger font (no Arial specification)
    ax.set_xlabel('Ground Truth', fontweight='bold', fontsize=16)
    ax.set_ylabel('Prediction', fontweight='bold', fontsize=16)

    # Set tick parameters with thicker axes
    ax.tick_params(direction='in', labelsize=14, width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Add legend with larger font
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save the plot
    parity_plot_path = output_dir / 'parity_plot_combined.png'
    plt.savefig(parity_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved combined parity plot: {parity_plot_path.name}")

    return saved_paths


def detailed_evaluation(
    config: Dict,
    channel_normalizer,
    device: str,
    model: nn.Module,
    test_loader,
    output_dir: Path,
    verbose: bool = True
) -> Dict:
    """
    Perform detailed evaluation computing Relative-L2-based metrics and optional SSIM.

    Args:
        config: Configuration dictionary
        channel_normalizer: Channel-wise normalizer for inverse transform
        device: Device to use
        model: Trained model
        test_loader: Test data loader
        output_dir: Directory to save results (metrics/)
        verbose: Whether to print progress

    Returns:
        Dictionary containing evaluation results
    """
    if verbose:
        print("\nComputing detailed evaluation metrics...")

    model.eval()

    # Storage for predictions and ground truth
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)  # Already normalized
            y = batch['y'].to(device)  # Already normalized

            # Get initial values if available (for delta mode reconstruction)
            y_initial_batch = None
            if 'y_initial' in batch:
                y_initial_batch = batch['y_initial'].to(device)

            # Predict in normalized space
            pred = model(x)

            # Convert prediction to raw physical values (with initial values for delta mode)
            pred_phys = channel_normalizer.inverse_transform_output(pred, y_initial=y_initial_batch)

            # Convert ground truth to raw physical values (with initial values for delta mode)
            y_phys = channel_normalizer.inverse_transform_output(y, y_initial=y_initial_batch)

            all_pred.append(pred_phys.cpu())
            all_gt.append(y_phys.cpu())

    # Concatenate
    pred_phys = torch.cat(all_pred, dim=0)
    gt_phys = torch.cat(all_gt, dim=0)

    # Note: Inverse transform (log→raw) is now handled by channel_normalizer
    # Apply additional masking if needed (e.g., source region masking)
    # pred_phys[:, :, 14:18, 14:18, :] = 0
    # gt_phys[:, :, 14:18, 14:18, :] = 0

    n_samples = pred_phys.shape[0]
    n_time = pred_phys.shape[-1]

    # Check metric toggles
    compute_relative_l2 = config.get('OUTPUT', {}).get('DETAIL_EVAL', {}).get('COMPUTE_RELATIVE_L2', False)
    compute_ssim = config.get('OUTPUT', {}).get('DETAIL_EVAL', {}).get('COMPUTE_SSIM', True)

    if compute_relative_l2:
        if verbose:
            print("  Relative L2 metrics are enabled (physical-value basis).")
    if compute_ssim:
        if verbose:
            print("  SSIM metrics are enabled.")

    # Compute time-wise Relative L2 and optional SSIM per sample
    relative_l2_data = {'time': list(range(n_time))} if compute_relative_l2 else None
    ssim_data = {'time': list(range(n_time))} if compute_ssim else None

    for sample_idx in range(n_samples):
        pred_sample = pred_phys[sample_idx, 0].numpy()  # (nx, ny, nt)
        gt_sample = gt_phys[sample_idx, 0].numpy()

        # Compute time-wise Relative L2 if enabled
        if compute_relative_l2:
            relative_l2_values = compute_relative_l2_per_time(pred_sample, gt_sample)
            relative_l2_data[f'sample_{sample_idx}'] = relative_l2_values

        # Compute SSIM if enabled
        if compute_ssim:
            ssim_values = compute_ssim_per_time(pred_sample, gt_sample)
            ssim_data[f'sample_{sample_idx}'] = ssim_values

    # Create DataFrames for per-time metrics
    relative_l2_df = pd.DataFrame(relative_l2_data) if compute_relative_l2 else None
    ssim_df = pd.DataFrame(ssim_data) if compute_ssim else None

    # Add mean columns to per-time metrics
    add_mean = config.get('OUTPUT', {}).get('DETAIL_EVAL', {}).get('ADD_MEAN_COLUMN', True)
    if add_mean:
        if compute_relative_l2 and relative_l2_df is not None:
            relative_l2_df = add_mean_column(relative_l2_df, exclude_col='time')
        if compute_ssim and ssim_df is not None:
            ssim_df = add_mean_column(ssim_df, exclude_col='time')

    # Compute global metrics (case-wise Relative L2 and global R²) if enabled
    global_metrics_df = None
    if compute_relative_l2:
        if verbose:
            print("  Computing global metrics (case Relative L2 and global R²)...")

        case_relative_l2_values = []

        # Compute per-sample case Relative L2
        for sample_idx in range(n_samples):
            pred_sample = pred_phys[sample_idx, 0].numpy()  # (nx, ny, nt)
            gt_sample = gt_phys[sample_idx, 0].numpy()

            case_rel_l2 = compute_case_relative_l2(pred_sample, gt_sample)
            case_relative_l2_values.append(case_rel_l2)

        # Compute global R² across ALL samples (mathematically correct)
        gt_np = gt_phys[:, 0].numpy()    # (N, nx, ny, nt)
        pred_np = pred_phys[:, 0].numpy()  # (N, nx, ny, nt)
        r2_global = compute_global_r2_score(pred_np, gt_np)

        # Create DataFrame for global metrics
        global_metrics_df = pd.DataFrame({
            'sample': [f'sample_{i}' for i in range(n_samples)],
            'case_relative_l2': case_relative_l2_values
        })

        # Add summary row with mean Relative L2 and global R²
        if add_mean:
            mean_row = pd.DataFrame({
                'sample': ['mean'],
                'case_relative_l2': [np.mean(case_relative_l2_values)]
            })
            global_metrics_df = pd.concat([global_metrics_df, mean_row], ignore_index=True)

        # Add global R² as a separate column (single value for entire dataset)
        global_metrics_df['r2_score'] = np.nan  # Initialize with NaN
        # Only set R² in the mean/summary row
        if add_mean:
            global_metrics_df.loc[global_metrics_df['sample'] == 'mean', 'r2_score'] = r2_global

    # Save SSIM if enabled
    ssim_path = None
    if compute_ssim and ssim_df is not None:
        ssim_path = output_dir / 'ssim_evolution.csv'
        ssim_df.to_csv(ssim_path, index=False)
        if verbose:
            print(f"  SSIM evolution saved: {ssim_path.name}")

    # Save Relative L2 evolution if enabled
    relative_l2_path = None
    if compute_relative_l2 and relative_l2_df is not None:
        relative_l2_path = output_dir / 'relative_l2_evolution.csv'
        relative_l2_df.to_csv(relative_l2_path, index=False)
        if verbose:
            print(f"  Relative L2 evolution saved: {relative_l2_path.name}")

    # Save global metrics if enabled
    global_metrics_path = None
    if compute_relative_l2 and global_metrics_df is not None:
        global_metrics_path = output_dir / 'global_metrics.csv'
        global_metrics_df.to_csv(global_metrics_path, index=False)

        if verbose:
            print(f"  Global metrics saved: {global_metrics_path.name}")

            # Print summary statistics
            mean_idx = global_metrics_df[global_metrics_df['sample'] == 'mean'].index
            if len(mean_idx) > 0:
                mean_relative_l2 = global_metrics_df.loc[mean_idx[0], 'case_relative_l2']
                global_r2 = global_metrics_df.loc[mean_idx[0], 'r2_score']
                print(f"    Mean Case Relative L2: {mean_relative_l2:.6f}")
                print(f"    Global R² Score (all samples): {global_r2:.6f}")

    # Generate parity plot data if enabled
    parity_paths = []
    if config.get('OUTPUT', {}).get('DETAIL_EVAL', {}).get('PARITY_PLOT', True):
        parity_paths = generate_parity_csv(pred_phys, gt_phys, output_dir, config, verbose)

    return {
        'relative_l2_df': relative_l2_df,
        'ssim_df': ssim_df,
        'global_metrics_df': global_metrics_df,
        'relative_l2_path': relative_l2_path,
        'ssim_path': ssim_path,
        'global_metrics_path': global_metrics_path,
        'parity_paths': parity_paths
    }


# ==============================================================================
# Section 4: Integrated Gradients Analysis Functions
# ==============================================================================

def create_mean_baseline(
    train_dataset,
    val_dataset,
    test_dataset,
    verbose: bool = True
) -> torch.Tensor:
    """
    Create mean baseline from all datasets for IG analysis.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        verbose: Whether to print progress

    Returns:
        Baseline tensor of shape (1, C, nx, ny, nt)
    """
    if verbose:
        print("Creating mean baseline from all datasets...")

    all_samples = []

    # Collect all samples
    for i in range(len(train_dataset)):
        all_samples.append(train_dataset[i]['x'])
    for i in range(len(val_dataset)):
        all_samples.append(val_dataset[i]['x'])
    for i in range(len(test_dataset)):
        all_samples.append(test_dataset[i]['x'])

    # Compute mean
    baseline = torch.stack(all_samples).mean(dim=0, keepdim=True)

    if verbose:
        print(f"  Total samples: {len(all_samples)}")
        print(f"  Baseline shape: {baseline.shape}")
        print(f"  Channel means:")
        for ch in range(baseline.shape[1]):
            mean_val = baseline[0, ch].mean().item()
            print(f"    Ch{ch}: {mean_val:.4e}")

    return baseline


def create_multi_sample_baselines(
    train_dataset,
    val_dataset,
    test_dataset,
    n_baselines: int = 5,
    random_seed: int = 42,
    verbose: bool = True
) -> List[torch.Tensor]:
    """
    Create multiple real sample baselines for IG analysis.

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


def compute_integrated_gradients(
    model: nn.Module,
    channel_normalizer,
    device: str,
    test_sample: torch.Tensor,
    baseline: torch.Tensor,
    target_t: int,
    y_initial: torch.Tensor = None,
    n_steps: int = 50,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Integrated Gradients for a specific target time.

    Args:
        model: Trained model
        channel_normalizer: Channel-wise normalizer for inverse transform
        device: Device to use
        test_sample: Test sample tensor (1, C, nx, ny, nt)
        baseline: Baseline tensor (1, C, nx, ny, nt)
        target_t: Target time index
        y_initial: Initial values at t=0 (1, 1, nx, ny, 1) - for delta mode reconstruction
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

    # Wrapper model for sum-of-squares aggregation
    class SumSquaresWrapper(nn.Module):
        def __init__(self, model, channel_normalizer, target_t, y_initial):
            super().__init__()
            self.model = model
            self.channel_normalizer = channel_normalizer
            self.target_t = target_t
            self.y_initial = y_initial

        def forward(self, x):
            # x is already normalized
            pred = self.model(x)
            # Convert to raw physical values (with initial values for delta mode)
            pred_phys = self.channel_normalizer.inverse_transform_output(pred, y_initial=self.y_initial)
            output_slice = pred_phys[:, 0, :, :, self.target_t]
            return (output_slice ** 2).sum(dim=[1, 2])

    wrapped = SumSquaresWrapper(model, channel_normalizer, target_t, y_initial).to(device)

    # Compute gradients
    grads = []
    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (test_sample - baseline)
        interpolated = interpolated.to(device).requires_grad_(True)

        output = wrapped(interpolated)
        output.backward()

        grads.append(interpolated.grad.detach().cpu().clone())
        interpolated.grad = None

        if verbose and step % 10 == 0:
            print(f"  Step {step}/{n_steps}, output={output.item():.4e}")

    # Average gradient
    avg_grad = torch.stack(grads).mean(dim=0)

    # IG: (x - baseline) × avg_grad
    ig = (test_sample - baseline) * avg_grad

    # Sum over time dimension
    ig_spatial = ig[0, :, :, :, :].sum(dim=-1).numpy()  # (C, nx, ny)

    # Metadata
    info = {
        'target_t': target_t,
        'n_steps': n_steps,
        'total_abs_ig': float(np.abs(ig_spatial).sum()),
        'ig_sum': float(ig_spatial.sum()),
        'output_baseline': float(wrapped(baseline.to(device)).item()),
        'output_actual': float(wrapped(test_sample.to(device)).item())
    }

    if verbose:
        print(f"  Done. Total |IG|: {info['total_abs_ig']:.4e}")
        print(f"  Output change: {info['output_actual'] - info['output_baseline']:.4e}")
        print(f"  IG sum: {info['ig_sum']:.4e}")

    return ig_spatial, info


def compute_integrated_gradients_multi_baseline(
    model: nn.Module,
    channel_normalizer,
    device: str,
    test_sample: torch.Tensor,
    baselines: List[torch.Tensor],
    target_t: int,
    y_initial: torch.Tensor = None,
    n_steps: int = 50,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Integrated Gradients using multiple real sample baselines and average the results.

    This approach addresses the issue where a single mean baseline becomes homogeneous
    and falls outside the training distribution for models trained on heterogeneous inputs.
    By using multiple real samples as baselines, we ensure that interpolation paths
    remain within the learned distribution.

    Args:
        model: Trained model
        channel_normalizer: Channel-wise normalizer for inverse transform
        device: Device to use
        test_sample: Test sample tensor (1, C, nx, ny, nt)
        baselines: List of baseline tensors, each of shape (1, C, nx, ny, nt)
        target_t: Target time index
        y_initial: Initial values at t=0 (1, 1, nx, ny, 1) - for delta mode reconstruction
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

        ig_spatial, info = compute_integrated_gradients(
            model=model,
            channel_normalizer=channel_normalizer,
            device=device,
            test_sample=test_sample,
            baseline=baseline,
            target_t=target_t,
            y_initial=y_initial,
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

    info_avg = {
        'target_t': target_t,
        'n_steps': n_steps,
        'n_baselines': len(baselines),
        'total_abs_ig': float(np.abs(ig_spatial_avg).sum()),
        'ig_sum': float(ig_spatial_avg.sum()),
        'output_actual': float(np.mean(output_actuals)),  # Should be identical for all baselines
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

    return ig_spatial_avg, info_avg


def compute_global_ranges(
    ig_results: Dict[int, np.ndarray],
    input_data: np.ndarray
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    """
    Compute global vmin/vmax ranges for IG and input channels.
    Same channel uses same colorbar across all time indices.

    Args:
        ig_results: Dictionary mapping time indices to IG arrays (C, nx, ny)
        input_data: Input data array (C, nx, ny, nt)

    Returns:
        Tuple of (ig_ranges, input_ranges)
        - ig_ranges: Dict[channel_idx] = (vmin, vmax)
        - input_ranges: Dict[channel_idx] = (vmin, vmax)
    """
    n_channels = ig_results[list(ig_results.keys())[0]].shape[0]

    ig_ranges = {}
    input_ranges = {}

    for ch in range(n_channels):
        # IG ranges: aggregate all time indices
        all_ig_values = []
        for t_idx, ig_spatial in ig_results.items():
            all_ig_values.append(ig_spatial[ch].flatten())
        combined_ig = np.concatenate(all_ig_values)

        ig_pos = combined_ig[combined_ig > 0]
        ig_neg = combined_ig[combined_ig < 0]

        if len(ig_pos) > 0:
            ig_vmax = np.percentile(ig_pos, 99)
        else:
            ig_vmax = combined_ig.max()

        if len(ig_neg) > 0:
            ig_vmin = np.percentile(ig_neg, 1)
        else:
            ig_vmin = combined_ig.min()

        # Handle edge case
        if abs(ig_vmax - ig_vmin) < 1e-20:
            if abs(ig_vmax) < 1e-20:
                ig_vmax = 1e-20
                ig_vmin = -1e-20
            else:
                max_abs = max(abs(ig_vmax), abs(ig_vmin))
                ig_vmax = max_abs
                ig_vmin = -max_abs

        ig_ranges[ch] = (ig_vmin, ig_vmax)

        # Input ranges: aggregate all time (though it's static)
        input_vmin = np.percentile(input_data[ch], 2)
        input_vmax = np.percentile(input_data[ch], 98)

        input_ranges[ch] = (input_vmin, input_vmax)

    return ig_ranges, input_ranges


def visualize_baseline_channels(
    baseline_data: np.ndarray,
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> List[Path]:
    """
    Visualize baseline channels for IG analysis.
    Creates one image per channel showing the mean baseline (time-invariant).

    Args:
        baseline_data: Baseline data array (C, nx, ny, nt)
        output_dir: Output directory (integrated_gradients/sample_{idx}/)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        List of paths to saved images
    """
    channel_names = [
        'Permeability', 'Calcite', 'Clinochlore', 'Pyrite',
        'Smectite', 'Material_Source', 'Material_Bentonite', 'Material_Fracture',
        'X-velocity', 'Y-velocity', 'Meta'
    ]

    channel_short = [
        'Perm', 'Calcite', 'Clino', 'Pyrite',
        'Smectite', 'MatSrc', 'MatBent', 'MatFrac',
        'Vx', 'Vy', 'Meta'
    ]

    saved_paths = []
    dpi = config.get('OUTPUT', {}).get('DPI', 200)

    # Baseline channels are time-invariant, use t=0
    n_channels = baseline_data.shape[0]
    for ch in range(n_channels):
        # Create single subplot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Baseline channel at t=0 (time-invariant)
        baseline_slice = baseline_data[ch, :, :, 0]
        baseline_vmin = np.percentile(baseline_slice, 2)
        baseline_vmax = np.percentile(baseline_slice, 98)

        im = ax.imshow(baseline_slice.T, cmap='viridis',
                      vmin=baseline_vmin, vmax=baseline_vmax, aspect='auto')
        ax.set_title(f'{channel_names[ch]} Baseline (Mean)', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.formatter.set_useMathText(True)
        cbar.update_ticks()

        plt.tight_layout()

        # Save
        save_path = output_dir / f'ch{ch}_{channel_short[ch]}_baseline.png'
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        saved_paths.append(save_path)

        if verbose:
            print(f"  Saved: {save_path.name}")

    return saved_paths


def visualize_ig_input_channels(
    input_data: np.ndarray,
    sample_idx: int,
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> List[Path]:
    """
    Visualize input channels for IG analysis.
    Creates one image per channel (time-invariant).

    Args:
        input_data: Input data array (C, nx, ny, nt)
        sample_idx: Sample index
        output_dir: Output directory (integrated_gradients/sample_{idx}/)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        List of paths to saved images
    """
    channel_names = [
        'Permeability', 'Calcite', 'Clinochlore', 'Pyrite',
        'Smectite', 'Material_Source', 'Material_Bentonite', 'Material_Fracture',
        'X-velocity', 'Y-velocity', 'Meta'
    ]

    channel_short = [
        'Perm', 'Calcite', 'Clino', 'Pyrite',
        'Smectite', 'MatSrc', 'MatBent', 'MatFrac',
        'Vx', 'Vy', 'Meta'
    ]

    saved_paths = []
    dpi = config.get('OUTPUT', {}).get('DPI', 200)

    # Input channels are time-invariant, use t=0
    n_channels = input_data.shape[0]
    for ch in range(n_channels):
        # Create single subplot
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

        # Save
        save_path = output_dir / f'ch{ch}_{channel_short[ch]}_input.png'
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        saved_paths.append(save_path)

        if verbose:
            print(f"  Saved: {save_path.name}")

    return saved_paths


def visualize_ig_attributions(
    ig_results: Dict[int, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> List[Path]:
    """
    Visualize IG attribution maps for each channel and time index.
    Creates separate images for each (channel, time) combination.

    Args:
        ig_results: Dictionary mapping time indices to IG arrays (C, nx, ny)
        sample_idx: Sample index
        output_dir: Output directory (integrated_gradients/sample_{idx}/)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        List of paths to saved images
    """
    channel_names = [
        'Permeability', 'Calcite', 'Clinochlore', 'Pyrite',
        'Smectite', 'Material_Source', 'Material_Bentonite', 'Material_Fracture',
        'X-velocity', 'Y-velocity', 'Meta'
    ]

    channel_short = [
        'Perm', 'Calcite', 'Clino', 'Pyrite',
        'Smectite', 'MatSrc', 'MatBent', 'MatFrac',
        'Vx', 'Vy', 'Meta'
    ]

    # Compute global ranges for each channel (across all time indices)
    # Using symmetric colorbar centered at 0 for better interpretation
    n_channels = ig_results[list(ig_results.keys())[0]].shape[0]
    ig_ranges = {}

    for ch in range(n_channels):
        all_ig_values = []
        for t_idx, ig_spatial in ig_results.items():
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
    dpi = config.get('OUTPUT', {}).get('DPI', 200)

    for t_idx, ig_spatial in ig_results.items():
        for ch in range(n_channels):
            # Create single subplot for IG attribution
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

            # Save
            save_path = output_dir / f'ch{ch}_{channel_short[ch]}_t{t_idx:02d}_ig.png'
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

            saved_paths.append(save_path)

            if verbose:
                print(f"  Saved: {save_path.name}")

    return saved_paths


def save_ig_csv(
    ig_results: Dict[int, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    verbose: bool = True
) -> List[Path]:
    """
    Save IG results as CSV files.
    Format: x_coord, y_coord, Permeability_IG, Calcite_IG, ...

    Args:
        ig_results: Dictionary mapping time indices to IG arrays
        sample_idx: Sample index
        output_dir: Output directory (integrated_gradients/)
        verbose: Whether to print progress

    Returns:
        List of paths to saved CSV files
    """
    channel_names = [
        'Permeability', 'Calcite', 'Clinochlore', 'Pyrite',
        'Smectite', 'Material_Source', 'Material_Bentonite', 'Material_Fracture',
        'X-velocity', 'Y-velocity', 'Meta'
    ]

    saved_paths = []

    for t_idx, ig_spatial in ig_results.items():
        C, nx, ny = ig_spatial.shape

        # Create coordinate arrays
        x_coords = []
        y_coords = []
        for x in range(nx):
            for y in range(ny):
                x_coords.append(x)
                y_coords.append(y)

        # Data dictionary
        data = {
            'x_coord': x_coords,
            'y_coord': y_coords
        }

        # Add channel columns
        for ch in range(C):
            channel_col_name = f'{channel_names[ch]}_IG'
            ig_values = []
            for x in range(nx):
                for y in range(ny):
                    ig_values.append(ig_spatial[ch, x, y])
            data[channel_col_name] = ig_values

        df = pd.DataFrame(data)
        csv_path = output_dir / f'ig_data_s{sample_idx}_t{t_idx:02d}.csv'
        df.to_csv(csv_path, index=False)

        saved_paths.append(csv_path)

        if verbose:
            print(f"  Saved: {csv_path.name}")

    return saved_paths


def analyze_channel_importance(
    ig_results: Dict[int, np.ndarray],
    output_dir: Path,
    verbose: bool = True
) -> Tuple[Path, Path]:
    """
    Analyze and visualize channel importance evolution over time.

    Args:
        ig_results: Dictionary mapping time indices to IG arrays
        output_dir: Output directory (integrated_gradients/)
        verbose: Whether to print progress

    Returns:
        Tuple of (csv_path, plot_path)
    """
    channel_names = [
        'Perm', 'Calcite', 'Clino', 'Pyrite',
        'Smectite', 'MatSrc', 'MatBent', 'MatFrac',
        'Vx', 'Vy', 'Meta'
    ]

    times = sorted(ig_results.keys())
    n_channels = ig_results[times[0]].shape[0]

    # Compute importance (sum of absolute IG)
    importance = np.zeros((len(times), n_channels))

    for i, t in enumerate(times):
        for ch in range(n_channels):
            importance[i, ch] = np.abs(ig_results[t][ch]).sum()

    # Save CSV
    df = pd.DataFrame(importance, index=times, columns=channel_names)
    df.index.name = 'time'
    csv_path = output_dir / 'channel_importance.csv'
    df.to_csv(csv_path)

    # Create plot
    plt.figure(figsize=(12, 6))
    for ch in range(n_channels):
        plt.plot(times, importance[:, ch], marker='o', label=channel_names[ch])

    plt.xlabel('Time Index')
    plt.ylabel('Total |IG|')
    plt.title('Channel Importance Evolution Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / 'importance_evolution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved importance analysis: {csv_path.name}, {plot_path.name}")

    return csv_path, plot_path


def integrated_gradients_analysis(
    config: Dict,
    channel_normalizer,
    device: str,
    model: nn.Module,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dirs: Dict[str, Path],
    verbose: bool = True
) -> Dict:
    """
    Perform complete Integrated Gradients analysis.

    Args:
        config: Configuration dictionary
        processor: Data processor
        device: Device to use
        model: Trained model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        output_dirs: Dictionary of output directories
        verbose: Whether to print progress

    Returns:
        Dictionary containing IG results
    """
    print("\n" + "="*70)
    print("INTEGRATED GRADIENTS ANALYSIS")
    print("="*70)

    ig_config = config.get('OUTPUT', {}).get('IG_ANALYSIS', {})
    sample_idx = ig_config.get('SAMPLE_IDX', 0)
    time_indices = ig_config.get('TIME_INDICES', [5, 10, 15, 19])
    n_steps = ig_config.get('N_STEPS', 50)

    # Check if multi-baseline mode is enabled
    use_multi_baseline = ig_config.get('USE_MULTI_BASELINE', False)
    n_baselines = ig_config.get('N_BASELINES', 5)
    baseline_seed = ig_config.get('BASELINE_SEED', 42)

    # Get test sample
    test_sample_dict = test_dataset[sample_idx]
    test_sample = test_sample_dict['x'].unsqueeze(0)  # (1, C, nx, ny, nt)
    input_data = test_sample[0].cpu().numpy()  # (C, nx, ny, nt)

    # Get initial values if available (for delta mode reconstruction)
    y_initial = None
    if 'y_initial' in test_sample_dict:
        y_initial = test_sample_dict['y_initial'].unsqueeze(0)  # (1, 1, nx, ny, 1)
        if verbose:
            print(f"  Initial values loaded for delta mode: {tuple(y_initial.shape)}")

    print(f"\nAnalyzing sample {sample_idx} at times {time_indices}")

    # Create baseline(s) based on configuration
    if use_multi_baseline:
        # Multi-baseline mode: use multiple real samples
        baselines = create_multi_sample_baselines(
            train_dataset, val_dataset, test_dataset,
            n_baselines=n_baselines,
            random_seed=baseline_seed,
            verbose=verbose
        )
        baseline_data = None  # Will visualize individual baselines instead

        # Compute IG for each time index using multi-baseline approach
        ig_results = {}
        for t in time_indices:
            ig_spatial, info = compute_integrated_gradients_multi_baseline(
                model=model,
                channel_normalizer=channel_normalizer,
                device=device,
                test_sample=test_sample,
                baselines=baselines,
                target_t=t,
                y_initial=y_initial,
                n_steps=n_steps,
                verbose=verbose
            )
            ig_results[t] = ig_spatial

    else:
        # Single baseline mode: use mean baseline (original behavior)
        baseline = create_mean_baseline(train_dataset, val_dataset, test_dataset, verbose)
        baseline_data = baseline[0].cpu().numpy()  # (C, nx, ny, nt) for visualization

        # Compute IG for each time index
        ig_results = {}
        for t in time_indices:
            ig_spatial, info = compute_integrated_gradients(
                model=model,
                channel_normalizer=channel_normalizer,
                device=device,
                test_sample=test_sample,
                baseline=baseline,
                target_t=t,
                y_initial=y_initial,
                n_steps=n_steps,
                verbose=verbose
            )
            ig_results[t] = ig_spatial

    # Generate outputs
    print("\nGenerating outputs...")

    # Baseline channel visualizations (time-invariant)
    # Only visualize if using single mean baseline
    baseline_viz_paths = []
    if baseline_data is not None:
        print("\n  Generating baseline channel images...")
        baseline_viz_paths = visualize_baseline_channels(
            baseline_data,
            output_dirs['ig_sample'], config, verbose
        )
    else:
        if verbose:
            print("\n  Skipping mean baseline visualization (using multi-baseline mode)")

    # Input channel visualizations (time-invariant)
    print("\n  Generating input channel images...")
    input_viz_paths = visualize_ig_input_channels(
        input_data, sample_idx,
        output_dirs['ig_sample'], config, verbose
    )

    # IG attribution visualizations (N channels × N time indices)
    print("\n  Generating IG attribution images...")
    ig_viz_paths = visualize_ig_attributions(
        ig_results, sample_idx,
        output_dirs['ig_sample'], config, verbose
    )

    # CSV files
    csv_paths = save_ig_csv(ig_results, sample_idx, output_dirs['ig'], verbose)

    # Channel importance analysis
    importance_csv, importance_plot = analyze_channel_importance(
        ig_results, output_dirs['ig'], verbose
    )

    print("\n" + "="*70)
    print(f"COMPLETED! Results in: {output_dirs['ig']}")
    print("="*70)

    return {
        'ig_results': ig_results,
        'baseline_viz_paths': baseline_viz_paths,
        'input_viz_paths': input_viz_paths,
        'ig_viz_paths': ig_viz_paths,
        'csv_paths': csv_paths,
        'importance_csv': importance_csv,
        'importance_plot': importance_plot
    }


# ==============================================================================
# Section 5: Master Output Generation Function
# ==============================================================================

def generate_all_outputs(
    config: Dict,
    channel_normalizer,
    device: str,
    trained_model: nn.Module,
    train_dataset,
    val_dataset,
    test_dataset,
    test_loader,
    verbose: bool = True
) -> Dict:
    """
    Master function to generate all outputs based on configuration.

    This function:
    1. Sets up output directories
    2. Generates predictions for all test samples
    3. Creates images (combined/separated) if enabled
    4. Creates GIFs if enabled
    5. Computes detailed metrics if enabled
    6. Performs IG analysis if enabled

    Args:
        config: Configuration dictionary
        channel_normalizer: Channel-wise normalizer for inverse transform
        device: Device to use
        trained_model: Trained FNO model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        test_loader: Test data loader
        verbose: Whether to print progress

    Returns:
        Dictionary containing all output results
    """
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    # Setup output directories
    base_dir = Path(config.get('OUTPUT_DIR', config.get('OUTPUT', {}).get('OUTPUT_DIR', './output')))
    output_dirs = setup_output_directories(base_dir, config)

    if verbose:
        print(f"\nOutput directory: {base_dir}")
        print(f"Subdirectories created: {list(output_dirs.keys())}")

    # Get configuration
    output_config = config.get('OUTPUT', {})
    sample_indices = output_config.get('SAMPLE_INDICES')
    time_indices = output_config.get('TIME_INDICES')

    # Generate predictions
    print("\nGenerating predictions...")
    trained_model.eval()

    all_pred = []
    all_gt = []
    all_input = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if verbose and batch_idx % 2 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(test_loader)}...")

            x, y = batch['x'].to(device), batch['y'].to(device)  # Already normalized

            # Get initial values if available (for delta mode reconstruction)
            y_initial_batch = None
            if 'y_initial' in batch:
                y_initial_batch = batch['y_initial'].to(device)

            all_input.append(x.cpu())

            # Predict in normalized space
            pred = trained_model(x)

            # Convert to raw physical values (with initial values for delta mode)
            pred_phys = channel_normalizer.inverse_transform_output(pred, y_initial=y_initial_batch)
            y_phys = channel_normalizer.inverse_transform_output(y, y_initial=y_initial_batch)

            all_pred.append(pred_phys.cpu())
            all_gt.append(y_phys.cpu())

            del x, y, pred, pred_phys, y_phys
            if y_initial_batch is not None:
                del y_initial_batch
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Concatenate
    pred_phys = torch.cat(all_pred, dim=0)
    gt_phys = torch.cat(all_gt, dim=0)
    input_phys = torch.cat(all_input, dim=0)

    del all_pred, all_gt, all_input

    # Note: Inverse transform (log→raw) is now handled by channel_normalizer
    # Apply additional masking if needed (e.g., source region masking)
    # pred_phys[:, :, 14:18, 14:18, :] = 0
    # gt_phys[:, :, 14:18, 14:18, :] = 0
    # pred_phys[:, :, 28:37, 28:37, :] = 0
    # gt_phys[:, :, 28:37, 28:37, :] = 0


    results = {}

    # ==== Image Output ====
    if output_config.get('IMAGE_OUTPUT', {}).get('ENABLED', False):
        print("\n" + "="*50)
        print("IMAGE OUTPUT")
        print("="*50)

        image_config = output_config['IMAGE_OUTPUT']
        combined_enabled = image_config.get('COMBINED_IMG', True)
        separated_enabled = image_config.get('SEPARATED_IMG', False)

        results['images'] = {}

        for sample_idx in sample_indices:
            if sample_idx >= len(pred_phys):
                if verbose:
                    print(f"Warning: Sample {sample_idx} exceeds available samples. Skipping.")
                continue

            print(f"\nProcessing sample {sample_idx}...")

            # Extract sample data
            pred_sample = pred_phys[sample_idx, 0].detach().cpu().numpy()
            gt_sample = gt_phys[sample_idx, 0].detach().cpu().numpy()

            # Combined grid
            if combined_enabled:
                combined_path = visualize_combined_grid(
                    pred_sample, gt_sample, sample_idx, time_indices,
                    output_dirs['images_combined'], config, verbose
                )
                results['images'][f'sample_{sample_idx}_combined'] = combined_path

            # Separated images
            if separated_enabled:
                separated_paths = visualize_separated_images(
                    pred_sample, gt_sample, sample_idx, time_indices,
                    output_dirs['images_separated'], config, verbose
                )
                results['images'][f'sample_{sample_idx}_separated'] = separated_paths

    # ==== GIF Output ====
    if output_config.get('GIF_OUTPUT', {}).get('ENABLED', False):
        print("\n" + "="*50)
        print("GIF OUTPUT")
        print("="*50)

        results['gifs'] = {}

        for sample_idx in sample_indices:
            if sample_idx >= len(pred_phys):
                continue

            gt_sample = gt_phys[sample_idx, 0].cpu().numpy()
            pred_sample = pred_phys[sample_idx, 0].cpu().numpy()

            gif_paths = create_all_gifs(
                gt_sample, pred_sample, sample_idx,
                output_dirs['gifs'], config, verbose
            )
            results['gifs'][f'sample_{sample_idx}'] = gif_paths

    # ==== Detailed Evaluation ====
    if output_config.get('DETAIL_EVAL', {}).get('ENABLED', False):
        print("\n" + "="*50)
        print("DETAILED EVALUATION")
        print("="*50)

        eval_results = detailed_evaluation(
            config, channel_normalizer, device, trained_model,
            test_loader, output_dirs['metrics'], verbose
        )
        results['metrics'] = eval_results

    # ==== Integrated Gradients ====
    if output_config.get('IG_ANALYSIS', {}).get('ENABLED', False):
        ig_results = integrated_gradients_analysis(
            config, channel_normalizer, device, trained_model,
            train_dataset, val_dataset, test_dataset,
            output_dirs, verbose
        )
        results['ig'] = ig_results

    print("\n" + "="*70)
    print("ALL OUTPUTS COMPLETED")
    print("="*70)

    return results
