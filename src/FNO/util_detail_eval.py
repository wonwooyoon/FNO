"""
Detailed Evaluation Utility for FNO Models

This module provides functions to compute detailed metrics (MSE and SSIM)
for each test sample across all time indices. Results are saved as CSV files
showing how metrics evolve over time for each sample.

Usage:
    from util_detail_eval import detailed_evaluation

    detailed_evaluation(
        config=CONFIG,
        processor=processor,
        device=device,
        model=trained_model,
        test_loader=test_loader,
        verbose=True
    )
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim


def compute_mse_per_time(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute MSE for each time index of a single sample.

    Args:
        pred: Prediction array of shape (nx, ny, nt)
        gt: Ground truth array of shape (nx, ny, nt)

    Returns:
        Array of MSE values of shape (nt,), one for each time index
    """
    nx, ny, nt = pred.shape
    mse_values = np.zeros(nt)

    for t in range(nt):
        pred_t = pred[:, :, t]
        gt_t = gt[:, :, t]
        mse_values[t] = np.mean((pred_t - gt_t) ** 2)

    return mse_values


def compute_ssim_per_time(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute SSIM for each time index of a single sample.

    Args:
        pred: Prediction array of shape (nx, ny, nt)
        gt: Ground truth array of shape (nx, ny, nt)

    Returns:
        Array of SSIM values of shape (nt,), one for each time index
    """
    nx, ny, nt = pred.shape
    ssim_values = np.zeros(nt)

    for t in range(nt):
        pred_t = pred[:, :, t]
        gt_t = gt[:, :, t]

        # Compute data range for this specific time slice
        data_range = max(gt_t.max() - gt_t.min(), pred_t.max() - pred_t.min())

        # Avoid SSIM computation if data_range is too small
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


def generate_parity_plot_data(
    pred_phys: torch.Tensor,
    gt_phys: torch.Tensor,
    config: Dict,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Generate parity plot data with pixel-wise GT vs Prediction for each time index.
    Saves separate CSV files for each time index to reduce file size.

    Each CSV file contains two columns: 'ground_truth' and 'prediction',
    with rows representing all pixels from all test samples at that time index.

    Args:
        pred_phys: Prediction tensor of shape (N, C, nx, ny, nt)
        gt_phys: Ground truth tensor of shape (N, C, nx, ny, nt)
        config: Configuration dictionary
        verbose: Whether to print progress information

    Returns:
        Dictionary mapping time indices to their saved CSV file paths:
        {0: Path('detail_eval_parity_t00.csv'), 1: Path('detail_eval_parity_t01.csv'), ...}
    """

    if verbose:
        print(f"\n  Generating parity plot data (separate files per time index)...")

    # Get dimensions
    n_samples, n_channels, nx, ny, n_time = pred_phys.shape
    total_pixels = n_samples * nx * ny

    # Create output directory
    output_dir = Path(config['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store file paths
    parity_file_paths = {}

    # Process each time index and save separately
    for t_idx in range(n_time):
        if verbose and t_idx % 5 == 0:
            print(f"    Processing time index {t_idx}/{n_time}...")

        # Extract data for this time index across all samples
        # Shape: (N, C, nx, ny) -> (N, nx, ny) -> flatten to (N*nx*ny,)
        gt_t = gt_phys[:, 0, :, :, t_idx].numpy().flatten()
        pred_t = pred_phys[:, 0, :, :, t_idx].numpy().flatten()

        # Create DataFrame for this time index
        parity_df_t = pd.DataFrame({
            'ground_truth': gt_t,
            'prediction': pred_t
        })

        # Save to CSV with zero-padded time index in filename
        csv_filename = f'detail_eval_parity_t{t_idx:02d}.csv'
        csv_path = output_dir / csv_filename
        parity_df_t.to_csv(csv_path, index=False)

        # Store path
        parity_file_paths[t_idx] = csv_path

    if verbose:
        print(f"    Parity plot data saved: {n_time} CSV files")
        print(f"    Each file contains {total_pixels} pixels (rows)")
        print(f"    File naming: detail_eval_parity_t00.csv to detail_eval_parity_t{n_time-1:02d}.csv")

    return parity_file_paths


def detailed_evaluation(
    config: Dict,
    processor,
    device: str,
    model,
    test_loader,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Perform detailed evaluation computing MSE and SSIM for each test sample
    across all time indices.

    Args:
        config: Configuration dictionary
        processor: Data processor for normalization
        device: Device to use (cuda/cpu)
        model: Trained model to evaluate
        test_loader: Test data loader
        verbose: Whether to print progress information

    Returns:
        Dictionary containing MSE and SSIM DataFrames:
        {
            'mse_df': DataFrame with rows=time indices, cols=sample indices
            'ssim_df': DataFrame with rows=time indices, cols=sample indices
        }
    """

    if verbose:
        print(f"\nPerforming detailed evaluation (MSE and SSIM per time index)...")

    # Set model to evaluation mode
    model.eval()

    # Storage for all predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    # Generate predictions for all test samples
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if verbose and batch_idx % 5 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(test_loader)}...")

            x = batch['x'].to(device)
            y = batch['y'].to(device)

            # Normalize input
            x_norm = processor.in_normalizer.transform(x)

            # Get model prediction
            pred_norm = model(x_norm)

            # Inverse transform to physical scale
            pred_phys = processor.out_normalizer.inverse_transform(pred_norm)

            # Move to CPU and store
            all_predictions.append(pred_phys.cpu())
            all_ground_truths.append(y.cpu())

            # Free GPU memory
            del x, y, x_norm, pred_norm, pred_phys
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Concatenate all batches
    if verbose:
        print("  Concatenating results...")
    pred_phys = torch.cat(all_predictions, dim=0)  # Shape: (N, C, nx, ny, nt)
    gt_phys = torch.cat(all_ground_truths, dim=0)  # Shape: (N, C, nx, ny, nt)

    # Free memory
    del all_predictions, all_ground_truths

    # Apply masking (same as visualization)
    pred_phys[:, :, 14:18, 14:18, :] = 0
    gt_phys[:, :, 14:18, 14:18, :] = 0

    # Get dimensions
    n_samples = pred_phys.shape[0]
    n_time = pred_phys.shape[-1]

    if verbose:
        print(f"  Computing metrics for {n_samples} samples across {n_time} time indices...")

    # Initialize storage for metrics
    # Each row is a time index, each column is a sample
    mse_matrix = np.zeros((n_time, n_samples))
    ssim_matrix = np.zeros((n_time, n_samples))

    # Compute metrics for each sample
    for sample_idx in range(n_samples):
        if verbose and sample_idx % 10 == 0:
            print(f"    Processing sample {sample_idx}/{n_samples}...")

        # Extract sample data (remove batch and channel dimensions)
        pred_sample = pred_phys[sample_idx, 0].numpy()  # Shape: (nx, ny, nt)
        gt_sample = gt_phys[sample_idx, 0].numpy()      # Shape: (nx, ny, nt)

        # Compute MSE and SSIM for all time indices
        mse_values = compute_mse_per_time(pred_sample, gt_sample)
        ssim_values = compute_ssim_per_time(pred_sample, gt_sample)

        # Store in matrix (column for this sample)
        mse_matrix[:, sample_idx] = mse_values
        ssim_matrix[:, sample_idx] = ssim_values

    # Create DataFrames with proper column and index names
    sample_cols = [f'sample_{i}' for i in range(n_samples)]
    time_indices = list(range(n_time))

    mse_df = pd.DataFrame(
        mse_matrix,
        index=time_indices,
        columns=sample_cols
    )
    mse_df.index.name = 'time_index'

    ssim_df = pd.DataFrame(
        ssim_matrix,
        index=time_indices,
        columns=sample_cols
    )
    ssim_df.index.name = 'time_index'

    # Generate parity plot data (saves separate CSV files internally)
    parity_file_paths = generate_parity_plot_data(
        pred_phys=pred_phys,
        gt_phys=gt_phys,
        config=config,
        verbose=verbose
    )

    # Save MSE and SSIM to CSV files
    output_dir = Path(config['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)

    mse_csv_path = output_dir / 'detail_eval_mse.csv'
    ssim_csv_path = output_dir / 'detail_eval_ssim.csv'

    mse_df.to_csv(mse_csv_path)
    ssim_df.to_csv(ssim_csv_path)

    if verbose:
        print(f"\nDetailed evaluation completed!")
        print(f"  MSE results saved to: {mse_csv_path}")
        print(f"  SSIM results saved to: {ssim_csv_path}")
        print(f"  Parity plot data: {len(parity_file_paths)} CSV files in {output_dir}")
        print(f"  MSE/SSIM shape: {n_time} time indices × {n_samples} samples")
        print(f"\n  Summary statistics:")
        print(f"    MSE  - Mean: {mse_matrix.mean():.6f}, Std: {mse_matrix.std():.6f}")
        print(f"    SSIM - Mean: {ssim_matrix.mean():.6f}, Std: {ssim_matrix.std():.6f}")

    return {
        'mse_df': mse_df,
        'ssim_df': ssim_df,
        'parity_file_paths': parity_file_paths
    }
