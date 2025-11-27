"""
Unified Output Utility for FNO Models

This module consolidates all output-related functions including:
- Image visualization (combined grids and separated images)
- GIF generation for temporal evolution
- Detailed evaluation metrics (RMSE, SSIM, parity plots)
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
        im_gt = ax_gt.imshow(gt_sample[:, :, t_idx].T, cmap='jet',
                            vmin=vmin_gt_pred, vmax=vmax_gt_pred)
        ax_gt.set_title(f"Ground Truth (t={t_idx})")
        ax_gt.axis('off')

        # Plot Prediction (Row 2)
        ax_pred = axes[1, i]
        im_pred = ax_pred.imshow(pred_sample[:, :, t_idx].T, cmap='jet',
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
        im = ax.imshow(gt_sample[:, :, t_idx].T, cmap='jet',
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
        im = ax.imshow(pred_sample[:, :, t_idx].T, cmap='jet',
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
        cmap='jet',
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
        cmap='jet',
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
        cmap='jet',
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

def compute_rmse_per_time(
    pred: np.ndarray,
    gt: np.ndarray,
    normalize: bool = False,
    min_per_time: Optional[np.ndarray] = None,
    max_per_time: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute RMSE for each time index of a single sample.

    Args:
        pred: Prediction array of shape (nx, ny, nt)
        gt: Ground truth array of shape (nx, ny, nt)
        normalize: If True, apply MinMax normalization before computing RMSE
        min_per_time: Min values across all samples for each time (nt,).
                      Required if normalize=True.
        max_per_time: Max values across all samples for each time (nt,).
                      Required if normalize=True.

    Returns:
        Array of RMSE (or NRMSE if normalized) values of shape (nt,)
    """
    nx, ny, nt = pred.shape
    rmse_values = np.zeros(nt)

    for t in range(nt):
        pred_t = pred[:, :, t]
        gt_t = gt[:, :, t]

        if normalize:
            if min_per_time is None or max_per_time is None:
                raise ValueError("min_per_time and max_per_time must be provided when normalize=True")

            # MinMax normalization
            min_t = min_per_time[t]
            max_t = max_per_time[t]
            range_t = max_t - min_t

            pred_t_norm = (pred_t - min_t) / range_t
            gt_t_norm = (gt_t - min_t) / range_t

            # RMSE on normalized values
            rmse_values[t] = np.sqrt(np.mean((pred_t_norm - gt_t_norm) ** 2))
        else:
            # Absolute RMSE
            rmse_values[t] = np.sqrt(np.mean((pred_t - gt_t) ** 2))

    return rmse_values


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

    # Plot 1:1 line
    ax.plot([0, 1e-6], [0, 1e-6], 'y--', linewidth=2, label='1:1 line')

    # Set axis range and ticks
    ax.set_xlim(0, 1e-6)
    ax.set_ylim(0, 1e-6)
    ax.set_xticks(np.arange(0, 1.2e-6, 0.2e-6))
    ax.set_yticks(np.arange(0, 1.2e-6, 0.2e-6))

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
    Perform detailed evaluation computing RMSE, NRMSE, and SSIM per time index.

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

            # Predict in normalized space
            pred = model(x)

            # Convert prediction to raw physical values
            pred_phys = channel_normalizer.inverse_transform_output_to_raw(pred)

            # Convert ground truth to raw physical values
            y_phys = channel_normalizer.inverse_transform_output_to_raw(y)

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

    # Check if NRMSE computation is enabled
    compute_nrmse = config.get('OUTPUT', {}).get('DETAIL_EVAL', {}).get('COMPUTE_NRMSE', False)

    # Compute min/max values if NRMSE is enabled
    min_per_time = None
    max_per_time = None
    if compute_nrmse:
        gt_np = gt_phys[:, 0].numpy()  # (N, nx, ny, nt)
        min_per_time = np.zeros(n_time)
        max_per_time = np.zeros(n_time)
        for t in range(n_time):
            min_per_time[t] = gt_np[:, :, :, t].min()
            max_per_time[t] = gt_np[:, :, :, t].max()

    # Compute RMSE and optionally NRMSE per time for each sample
    rmse_data = {'time': list(range(n_time))}
    nrmse_data = {'time': list(range(n_time))} if compute_nrmse else None
    ssim_data = {'time': list(range(n_time))}

    for sample_idx in range(n_samples):
        pred_sample = pred_phys[sample_idx, 0].numpy()  # (nx, ny, nt)
        gt_sample = gt_phys[sample_idx, 0].numpy()

        # Compute absolute RMSE
        rmse_values = compute_rmse_per_time(pred_sample, gt_sample, normalize=False)
        rmse_data[f'sample_{sample_idx}'] = rmse_values

        # Compute normalized RMSE (NRMSE) if enabled
        if compute_nrmse:
            nrmse_values = compute_rmse_per_time(
                pred_sample, gt_sample,
                normalize=True,
                min_per_time=min_per_time,
                max_per_time=max_per_time
            )
            nrmse_data[f'sample_{sample_idx}'] = nrmse_values

        # Compute SSIM
        ssim_values = compute_ssim_per_time(pred_sample, gt_sample)
        ssim_data[f'sample_{sample_idx}'] = ssim_values

    # Create DataFrames
    rmse_df = pd.DataFrame(rmse_data)
    nrmse_df = pd.DataFrame(nrmse_data) if compute_nrmse else None
    ssim_df = pd.DataFrame(ssim_data)

    # Add mean columns
    add_mean = config.get('OUTPUT', {}).get('DETAIL_EVAL', {}).get('ADD_MEAN_COLUMN', True)
    if add_mean:
        rmse_df = add_mean_column(rmse_df, exclude_col='time')
        if compute_nrmse and nrmse_df is not None:
            nrmse_df = add_mean_column(nrmse_df, exclude_col='time')
        ssim_df = add_mean_column(ssim_df, exclude_col='time')

    # Save RMSE and SSIM
    rmse_path = output_dir / 'rmse_evolution.csv'
    ssim_path = output_dir / 'ssim_evolution.csv'

    rmse_df.to_csv(rmse_path, index=False)
    ssim_df.to_csv(ssim_path, index=False)

    if verbose:
        print(f"  RMSE evolution saved: {rmse_path.name}")
        print(f"  SSIM evolution saved: {ssim_path.name}")

    # Save NRMSE if enabled
    nrmse_path = None
    if compute_nrmse and nrmse_df is not None:
        nrmse_path = output_dir / 'nrmse_evolution.csv'
        nrmse_df.to_csv(nrmse_path, index=False)
        if verbose:
            print(f"  NRMSE evolution saved: {nrmse_path.name}")

    # Generate parity plot data if enabled
    parity_paths = []
    if config.get('OUTPUT', {}).get('DETAIL_EVAL', {}).get('PARITY_PLOT', True):
        parity_paths = generate_parity_csv(pred_phys, gt_phys, output_dir, config, verbose)

    return {
        'rmse_df': rmse_df,
        'nrmse_df': nrmse_df,
        'ssim_df': ssim_df,
        'rmse_path': rmse_path,
        'nrmse_path': nrmse_path,
        'ssim_path': ssim_path,
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


def compute_integrated_gradients(
    model: nn.Module,
    channel_normalizer,
    device: str,
    test_sample: torch.Tensor,
    baseline: torch.Tensor,
    target_t: int,
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
        def __init__(self, model, channel_normalizer, target_t):
            super().__init__()
            self.model = model
            self.channel_normalizer = channel_normalizer
            self.target_t = target_t

        def forward(self, x):
            # x is already normalized
            pred = self.model(x)
            # Convert to raw physical values
            pred_phys = self.channel_normalizer.inverse_transform_output_to_raw(pred)
            output_slice = pred_phys[:, 0, :, :, self.target_t]
            return (output_slice ** 2).sum(dim=[1, 2])

    wrapped = SumSquaresWrapper(model, channel_normalizer, target_t).to(device)

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
    n_channels = ig_results[list(ig_results.keys())[0]].shape[0]
    ig_ranges = {}

    for ch in range(n_channels):
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

    # Create baseline
    baseline = create_mean_baseline(train_dataset, val_dataset, test_dataset, verbose)

    # Get test sample
    test_sample = test_dataset[sample_idx]['x'].unsqueeze(0)  # (1, C, nx, ny, nt)
    input_data = test_sample[0].cpu().numpy()  # (C, nx, ny, nt)

    print(f"\nAnalyzing sample {sample_idx} at times {time_indices}")

    # Compute IG for each time index
    ig_results = {}
    for t in time_indices:
        ig_spatial, info = compute_integrated_gradients(
            model, channel_normalizer, device,
            test_sample, baseline, t,
            n_steps=n_steps, verbose=verbose
        )
        ig_results[t] = ig_spatial

    # Generate outputs
    print("\nGenerating outputs...")

    # Baseline channel visualizations (time-invariant)
    print("\n  Generating baseline channel images...")
    baseline_data = baseline[0].cpu().numpy()  # (C, nx, ny, nt)
    baseline_viz_paths = visualize_baseline_channels(
        baseline_data,
        output_dirs['ig_sample'], config, verbose
    )

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
    sample_indices = output_config.get('SAMPLE_INDICES', [0])
    time_indices = output_config.get('TIME_INDICES', [0, 5, 10, 15, 19])

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

            all_input.append(x.cpu())

            # Predict in normalized space
            pred = trained_model(x)

            # Convert to raw physical values
            pred_phys = channel_normalizer.inverse_transform_output_to_raw(pred)
            y_phys = channel_normalizer.inverse_transform_output_to_raw(y)

            all_pred.append(pred_phys.cpu())
            all_gt.append(y_phys.cpu())

            del x, y, pred, pred_phys, y_phys
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Concatenate
    pred_phys = torch.cat(all_pred, dim=0)
    gt_phys = torch.cat(all_gt, dim=0)
    input_phys = torch.cat(all_input, dim=0)

    del all_pred, all_gt, all_input

    # Note: Inverse transform (log→raw) is now handled by channel_normalizer
    # Apply additional masking if needed (e.g., source region masking)
    pred_phys[:, :, 28:36, 28:36, :] = 0
    gt_phys[:, :, 28:36, 28:36, :] = 0


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
