"""
GIF Generation Utility for FNO Visualization

This module provides functions to create animated GIFs showing the temporal evolution
of ground truth, predictions, and errors from FNO model outputs.

Each data type (GT, Prediction, Error) is saved as a separate GIF file containing
only the image data without legends, titles, or colorbars. Colorbars are saved as
separate PNG files for reference.
"""

from pathlib import Path
from typing import Dict, Union, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def create_single_type_gif(
    data: np.ndarray,
    data_type: str,
    sample_idx: int,
    time_indices: List[int],
    output_dir: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    config: Dict,
    verbose: bool = True
) -> Path:
    """
    Create an animated GIF for a single data type (GT, Prediction, or Error).

    The GIF contains only the image data without any legends, titles, colorbars,
    or time annotations for clean visualization.

    Args:
        data: Data array of shape (nx, ny, nt)
        data_type: Type of data - 'gt', 'pred', or 'error'
        sample_idx: Index of the sample being visualized
        time_indices: List of time indices to include in the GIF
        output_dir: Directory where the GIF will be saved
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        cmap: Colormap name (e.g., 'jet', 'coolwarm')
        config: Configuration dictionary containing visualization settings
        verbose: Whether to print progress information

    Returns:
        Path to the saved GIF file
    """

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine figure size based on data shape to maintain aspect ratio
    nx, ny, nt = data.shape
    aspect_ratio = ny / nx
    fig_width = 8
    fig_height = fig_width * aspect_ratio

    # Create figure with no frame
    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    ax.axis('off')

    # Initialize image
    im = ax.imshow(
        data[:, :, time_indices[0]].T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
        interpolation='nearest'
    )

    def init():
        """Initialize animation - return image artist."""
        im.set_array(data[:, :, time_indices[0]].T)
        return [im]

    def animate(frame_idx):
        """Update function for each frame of the animation."""
        t_idx = time_indices[frame_idx]
        im.set_array(data[:, :, t_idx].T)
        return [im]

    # Create animation
    fps = config['VISUALIZATION'].get('GIF_FPS', 2)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(time_indices),
        interval=1000/fps,
        blit=True,
        repeat=True
    )

    # Save as GIF
    gif_filename = f'FNO_animation_sample_{sample_idx}_{data_type}.gif'
    gif_path = output_dir / gif_filename

    writer = animation.PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer, dpi=config['VISUALIZATION'].get('DPI', 100))

    plt.close(fig)

    if verbose:
        print(f"    {data_type.upper()} GIF saved: {gif_filename}")

    return gif_path


def create_colorbar_legend(
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
        cmap: Colormap name (e.g., 'jet', 'coolwarm')
        label: Label for the colorbar
        output_path: Full path where the PNG will be saved
        config: Configuration dictionary containing visualization settings
        orientation: Colorbar orientation ('horizontal' or 'vertical')
        verbose: Whether to print progress information

    Returns:
        Path to the saved colorbar PNG file
    """

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure for colorbar
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

    # Save colorbar
    dpi = config['VISUALIZATION'].get('DPI', 200)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"    Colorbar saved: {output_path.name}")

    return output_path


def create_gif_for_sample(
    gt_data: np.ndarray,
    pred_data: np.ndarray,
    sample_idx: int,
    time_indices: Union[range, Tuple, list],
    output_dir: Path,
    config: Dict,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Create separate animated GIFs for ground truth, prediction, and error,
    along with standalone colorbar PNG files.

    Args:
        gt_data: Ground truth data array of shape (nx, ny, nt)
        pred_data: Prediction data array of shape (nx, ny, nt)
        sample_idx: Index of the sample being visualized
        time_indices: Time indices to include in the GIF (can be range, tuple, or list)
        output_dir: Directory where files will be saved
        config: Configuration dictionary containing visualization settings
        verbose: Whether to print progress information

    Returns:
        Dictionary containing paths to all generated files:
        {
            'gt_gif': Path,
            'pred_gif': Path,
            'error_gif': Path,
            'gt_pred_colorbar': Path,
            'error_colorbar': Path
        }
    """

    if verbose:
        print(f"  Creating GIFs for sample {sample_idx}...")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert time_indices to list for consistent iteration
    if isinstance(time_indices, range):
        time_list = list(time_indices)
    elif isinstance(time_indices, tuple):
        time_list = list(time_indices)
    else:
        time_list = time_indices

    # Calculate error
    error_data = gt_data - pred_data

    # Determine color scales for consistent visualization across frames
    # GT and Prediction share the same color scale
    vmin_gt_pred = min(gt_data[:, :, time_list].min(), pred_data[:, :, time_list].min())
    vmax_gt_pred = max(gt_data[:, :, time_list].max(), pred_data[:, :, time_list].max())

    # Error uses symmetric scale
    error_max_abs = np.abs(error_data[:, :, time_list]).max()
    vmin_error = -error_max_abs
    vmax_error = error_max_abs

    # Initialize output dictionary
    output_paths = {}

    # Create GIF for Ground Truth
    output_paths['gt_gif'] = create_single_type_gif(
        data=gt_data,
        data_type='gt',
        sample_idx=sample_idx,
        time_indices=time_list,
        output_dir=output_dir,
        vmin=vmin_gt_pred,
        vmax=vmax_gt_pred,
        cmap='jet',
        config=config,
        verbose=verbose
    )

    # Create GIF for Prediction
    output_paths['pred_gif'] = create_single_type_gif(
        data=pred_data,
        data_type='pred',
        sample_idx=sample_idx,
        time_indices=time_list,
        output_dir=output_dir,
        vmin=vmin_gt_pred,
        vmax=vmax_gt_pred,
        cmap='jet',
        config=config,
        verbose=verbose
    )

    # Create GIF for Error
    output_paths['error_gif'] = create_single_type_gif(
        data=error_data,
        data_type='error',
        sample_idx=sample_idx,
        time_indices=time_list,
        output_dir=output_dir,
        vmin=vmin_error,
        vmax=vmax_error,
        cmap='coolwarm',
        config=config,
        verbose=verbose
    )

    # Create colorbar for GT and Prediction (shared colorbar)
    colorbar_gt_pred_path = output_dir / f'FNO_colorbar_sample_{sample_idx}_gt_pred.png'
    output_paths['gt_pred_colorbar'] = create_colorbar_legend(
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
    colorbar_error_path = output_dir / f'FNO_colorbar_sample_{sample_idx}_error.png'
    output_paths['error_colorbar'] = create_colorbar_legend(
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
        print(f"    Completed sample {sample_idx}: 3 GIFs + 2 colorbars")

    return output_paths


def create_gifs_for_samples(
    gt_phys: np.ndarray,
    pred_phys: np.ndarray,
    sample_indices: list,
    config: Dict,
    output_dir: Path,
    verbose: bool = True
) -> Dict[int, Dict[str, Path]]:
    """
    Create GIFs and colorbars for multiple samples.

    Args:
        gt_phys: Ground truth tensor of shape (N, C, nx, ny, nt)
        pred_phys: Prediction tensor of shape (N, C, nx, ny, nt)
        sample_indices: List of sample indices to create GIFs for
        config: Configuration dictionary
        output_dir: Directory where GIFs will be saved
        verbose: Whether to print progress information

    Returns:
        Dictionary mapping sample indices to their output file paths:
        {
            sample_idx: {
                'gt_gif': Path,
                'pred_gif': Path,
                'error_gif': Path,
                'gt_pred_colorbar': Path,
                'error_colorbar': Path
            },
            ...
        }
    """

    if verbose:
        print(f"\nGenerating GIFs for {len(sample_indices)} samples...")

    # Determine time indices to use
    if config['VISUALIZATION'].get('GIF_ALL_TIMES', True):
        # Use all time indices
        nt = gt_phys.shape[-1]
        time_indices = range(nt)
        if verbose:
            print(f"  Using all time indices: 0 to {nt-1}")
    else:
        # Use only specified TIME_INDICES
        time_indices = config['VISUALIZATION']['TIME_INDICES']
        if verbose:
            print(f"  Using specified time indices: {time_indices}")

    all_output_paths = {}

    for i, sample_idx in enumerate(sample_indices):
        if verbose:
            print(f"\n  Processing sample {sample_idx} ({i+1}/{len(sample_indices)})...")

        # Extract data for this sample (remove batch and channel dimensions)
        gt_sample = gt_phys[sample_idx, 0].numpy()   # Shape: (nx, ny, nt)
        pred_sample = pred_phys[sample_idx, 0].numpy()  # Shape: (nx, ny, nt)

        # Create GIFs and colorbars for this sample
        sample_outputs = create_gif_for_sample(
            gt_data=gt_sample,
            pred_data=pred_sample,
            sample_idx=sample_idx,
            time_indices=time_indices,
            output_dir=output_dir,
            config=config,
            verbose=verbose
        )

        all_output_paths[sample_idx] = sample_outputs

    if verbose:
        total_gifs = len(all_output_paths) * 3
        total_colorbars = len(all_output_paths) * 2
        print(f"\nGIF generation completed!")
        print(f"  Created {total_gifs} GIF files (GT, Pred, Error for {len(sample_indices)} samples)")
        print(f"  Created {total_colorbars} colorbar PNG files")

    return all_output_paths
