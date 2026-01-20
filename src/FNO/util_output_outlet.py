"""
Outlet-Specific Output Utilities for FNO Models

This module provides visualization and analysis functions specific to outlet prediction:
- Outlet time series visualization (PNG + CSV)
- Parity plots for outlet predictions (PNG + CSV)
- Error statistics computation (TXT)

All visualization functions generate both image files and corresponding CSV data files
for external analysis and plotting.

Refactored from FNO_outlet.py for better code organization.
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ==============================================================================
# Visualization Functions
# ==============================================================================

def visualize_outlet_predictions(config: Dict, device: str, model, test_loader,
                                 outlet_normalizer, sample_indices: List[int] = None):
    """
    Visualize outlet predictions vs ground truth for selected samples.

    Generates the following outputs:
    1. Prediction samples plot (PNG) + CSV data
    2. Parity plot (PNG) + CSV data
    3. Statistical summary (TXT)

    Args:
        config: Configuration dictionary
        device: Device to use
        model: Trained model
        test_loader: Test data loader
        outlet_normalizer: Outlet normalizer for denormalization
        sample_indices: List of sample indices to visualize (if None, randomly selected)

    Returns:
        Dictionary containing error statistics
    """

    print(f"\nGenerating outlet prediction visualizations...")

    # Create output directory
    output_dir = Path(config['OUTPUT_DIR']) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all test predictions
    model.eval()
    all_x = []
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y_outlet = batch['y'].to(device)

            pred_outlet = model(x)

            all_x.append(x.cpu())
            all_y_true.append(y_outlet.cpu())
            all_y_pred.append(pred_outlet.cpu())

    # Concatenate all batches
    all_x = torch.cat(all_x, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)

    # Denormalize outlet data
    all_y_true_denorm = outlet_normalizer.inverse_transform(all_y_true)
    all_y_pred_denorm = outlet_normalizer.inverse_transform(all_y_pred)

    # Convert to numpy and take absolute values for plotting (original values are negative)
    all_y_true_denorm = torch.abs(all_y_true_denorm).numpy()
    all_y_pred_denorm = torch.abs(all_y_pred_denorm).numpy()

    # Time points (years): [100, 200, ..., 2000] (20 points, t=0 removed during normalization)
    time_points = np.arange(100, 2001, 100)

    # Generate sample indices if not provided
    if sample_indices is None:
        n_samples_viz = config.get('VISUALIZATION', {}).get('N_SAMPLES', 16)
        n_samples_viz = min(n_samples_viz, len(all_y_true_denorm))
        sample_indices = np.random.choice(len(all_y_true_denorm), size=n_samples_viz, replace=False)
        print(f"  Randomly selected {n_samples_viz} samples for visualization")

    # 1. Individual sample plots (16 samples in 4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.flatten()

    for i, sample_idx in enumerate(sample_indices[:16]):
        if sample_idx >= len(all_y_true_denorm):
            print(f"Warning: sample_idx {sample_idx} out of range, skipping")
            continue

        ax = axes[i]
        y_true = all_y_true_denorm[sample_idx]
        y_pred = all_y_pred_denorm[sample_idx]

        ax.plot(time_points, y_true, 'o-', label='Ground Truth', linewidth=2, markersize=6)
        ax.plot(time_points, y_pred, 's--', label='Prediction', linewidth=2, markersize=5, alpha=0.8)

        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Outlet UO2++ [mol]', fontsize=11)
        ax.set_title(f'Sample {sample_idx}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plot_path = output_dir / 'outlet_predictions_samples.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")

    # 1-1. Save prediction samples data as CSV
    # Create a CSV with columns: sample_idx, time_years, ground_truth, prediction
    csv_data = []
    for sample_idx in sample_indices[:16]:
        if sample_idx >= len(all_y_true_denorm):
            continue

        y_true = all_y_true_denorm[sample_idx]
        y_pred = all_y_pred_denorm[sample_idx]

        for t_idx, time_year in enumerate(time_points):
            csv_data.append({
                'sample_idx': sample_idx,
                'time_years': time_year,
                'ground_truth': y_true[t_idx],
                'prediction': y_pred[t_idx],
                'absolute_error': abs(y_pred[t_idx] - y_true[t_idx]),
                'relative_error': abs(y_pred[t_idx] - y_true[t_idx]) / (y_true[t_idx] + 1e-20)
            })

    samples_df = pd.DataFrame(csv_data)
    samples_csv_path = output_dir / 'outlet_predictions_samples.csv'
    samples_df.to_csv(samples_csv_path, index=False, float_format='%.6e')
    print(f"  ✓ Saved: {samples_csv_path}")

    # 2. Parity plot following FNO.py style
    # Compute errors for statistics
    abs_errors = np.abs(all_y_pred_denorm - all_y_true_denorm)
    rel_errors = abs_errors / (all_y_true_denorm + 1e-20)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')

    # Plot all data points with unified green color
    ax.scatter(all_y_true_denorm.flatten(), all_y_pred_denorm.flatten(),
              c='darkgreen', alpha=0.3, s=10, edgecolors='none',
              label='Outlet Predictions')

    # Determine axis limits based on data
    all_data = np.concatenate([all_y_true_denorm.flatten(), all_y_pred_denorm.flatten()])
    data_min = np.min(all_data[all_data > 0]) if np.any(all_data > 0) else 1e-10
    data_max = np.max(all_data)

    # Set log scale limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(data_min * 0.5, data_max * 2)
    ax.set_ylim(data_min * 0.5, data_max * 2)

    # Plot 1:1 line
    ax.plot([data_min * 0.5, data_max * 2], [data_min * 0.5, data_max * 2],
            'y--', linewidth=2, label='1:1 line')

    # Set labels with larger font (matching FNO.py style)
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

    error_plot_path = output_dir / 'outlet_error_analysis.png'
    plt.savefig(error_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {error_plot_path}")

    # 2-1. Save parity plot data as CSV
    # Flatten all predictions and ground truth for parity analysis
    parity_df = pd.DataFrame({
        'ground_truth': all_y_true_denorm.flatten(),
        'prediction': all_y_pred_denorm.flatten(),
        'absolute_error': abs_errors.flatten(),
        'relative_error': rel_errors.flatten()
    })
    parity_csv_path = output_dir / 'outlet_parity_data.csv'
    parity_df.to_csv(parity_csv_path, index=False, float_format='%.6e')
    print(f"  ✓ Saved: {parity_csv_path}")

    # 3. Compute and save statistics
    # Compute RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(abs_errors ** 2))

    # Compute NRMSE (Normalized RMSE using MinMax normalization)
    # MinMax normalization: NRMSE = RMSE / (max - min)
    data_min = all_y_true_denorm.min()
    data_max = all_y_true_denorm.max()
    data_range = data_max - data_min
    nrmse = rmse / data_range if data_range > 0 else 0.0

    stats = {
        'mean_absolute_error': float(abs_errors.mean()),
        'std_absolute_error': float(abs_errors.std()),
        'mean_relative_error': float(rel_errors.mean()),
        'std_relative_error': float(rel_errors.std()),
        'max_absolute_error': float(abs_errors.max()),
        'max_relative_error': float(rel_errors.max()),
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'data_min': float(data_min),
        'data_max': float(data_max),
    }

    stats_path = output_dir / 'prediction_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("Outlet Prediction Statistics\n")
        f.write("="*50 + "\n\n")
        f.write("Error Metrics:\n")
        f.write(f"  mean_absolute_error: {stats['mean_absolute_error']:.6e}\n")
        f.write(f"  std_absolute_error: {stats['std_absolute_error']:.6e}\n")
        f.write(f"  mean_relative_error: {stats['mean_relative_error']:.6e}\n")
        f.write(f"  std_relative_error: {stats['std_relative_error']:.6e}\n")
        f.write(f"  max_absolute_error: {stats['max_absolute_error']:.6e}\n")
        f.write(f"  max_relative_error: {stats['max_relative_error']:.6e}\n\n")
        f.write("RMSE Metrics:\n")
        f.write(f"  rmse: {stats['rmse']:.6e}\n")
        f.write(f"  nrmse: {stats['nrmse']:.6e}\n\n")
        f.write("Data Range:\n")
        f.write(f"  data_min: {stats['data_min']:.6e}\n")
        f.write(f"  data_max: {stats['data_max']:.6e}\n")

    print(f"  ✓ Saved: {stats_path}")
    print(f"\nVisualization complete!")
    print(f"  Mean Absolute Error: {stats['mean_absolute_error']:.6e} mol")
    print(f"  Mean Relative Error: {stats['mean_relative_error']:.4f}")
    print(f"  RMSE: {stats['rmse']:.6e} mol")
    print(f"  NRMSE: {stats['nrmse']:.6e}")

    return stats
