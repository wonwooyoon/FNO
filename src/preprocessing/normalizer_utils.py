#!/usr/bin/env python3
"""
Normalization Utility Functions

통계 분석 및 시각화 기능:
- Normalization 전후 통계 계산
- Distribution histogram 생성
- CSV 저장
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict


def compute_normalization_stats(
    data_before: torch.Tensor,
    data_after: torch.Tensor,
    channel_configs: List[Tuple]
) -> List[Dict]:
    """
    Compute statistics before and after normalization

    Args:
        data_before: (N, C, nx, ny, nt) - data before normalization
        data_after: (N, C, nx, ny, nt) - data after normalization
        channel_configs: List of (idx, name, ...) or just (name,)

    Returns:
        List of dicts with statistics for each channel
    """
    N, C, nx, ny, nt = data_before.shape
    stats = []

    for idx in range(C):
        # Extract channel name from config
        if len(channel_configs[idx]) > 1:
            # Input config: (idx, name, transform, normalizer)
            channel_name = channel_configs[idx][1]
        else:
            # Output config: (name,)
            channel_name = channel_configs[idx][0]

        # Flatten data for statistics
        before = data_before[:, idx].flatten().cpu().numpy()
        after = data_after[:, idx].flatten().cpu().numpy()

        stats.append({
            'channel': channel_name,
            'before_mean': float(np.mean(before)),
            'before_std': float(np.std(before)),
            'before_min': float(np.min(before)),
            'before_max': float(np.max(before)),
            'before_median': float(np.median(before)),
            'after_mean': float(np.mean(after)),
            'after_std': float(np.std(after)),
            'after_min': float(np.min(after)),
            'after_max': float(np.max(after)),
            'after_median': float(np.median(after)),
        })

    return stats


def save_stats_csv(stats: List[Dict], csv_path: Path):
    """
    Save statistics to CSV file

    Args:
        stats: List of statistics dicts
        csv_path: Output CSV path
    """
    df = pd.DataFrame(stats)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f"  Stats saved: {csv_path.name}")


def visualize_distributions(
    data_before: torch.Tensor,
    data_after: torch.Tensor,
    output_path: Path,
    channel_names: List[str],
    dpi: int = 150,
    sample_size: int = None
):
    """
    Create histogram comparison plots for before/after normalization

    Args:
        data_before: (N, C, nx, ny, nt) - data before normalization
        data_after: (N, C, nx, ny, nt) - data after normalization
        output_path: Output image path
        channel_names: List of channel names
        dpi: Image DPI
        sample_size: Number of samples to plot (None = all)
    """
    N, C, nx, ny, nt = data_before.shape

    # Sample if needed
    if sample_size is not None and N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        data_before = data_before[indices]
        data_after = data_after[indices]

    # Create figure with 2 columns (before/after) and C rows (channels)
    fig, axes = plt.subplots(C, 2, figsize=(14, 3 * C))

    # Handle single channel case
    if C == 1:
        axes = axes.reshape(1, -1)

    for i in range(C):
        before = data_before[:, i].flatten().cpu().numpy()
        after = data_after[:, i].flatten().cpu().numpy()

        # Before normalization (left column)
        ax_before = axes[i, 0]
        ax_before.hist(before, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax_before.set_title(f'{channel_names[i]} - Before Normalization', fontweight='bold')
        ax_before.set_xlabel('Value')
        ax_before.set_ylabel('Frequency')
        ax_before.grid(True, alpha=0.3)
        ax_before.axvline(np.mean(before), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(before):.4e}')
        ax_before.legend()

        # After normalization (right column)
        ax_after = axes[i, 1]
        ax_after.hist(after, bins=100, alpha=0.7, color='green', edgecolor='black')
        ax_after.set_title(f'{channel_names[i]} - After Normalization', fontweight='bold')
        ax_after.set_xlabel('Value')
        ax_after.set_ylabel('Frequency')
        ax_after.grid(True, alpha=0.3)
        ax_after.axvline(np.mean(after), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(after):.4e}')
        ax_after.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Visualization saved: {output_path.name}")


def visualize_normalized_summary(
    data_normalized: torch.Tensor,
    output_path: Path,
    channel_names: List[str],
    dpi: int = 150,
    sample_size: int = None
):
    """
    Create summary plot showing all normalized channels

    Args:
        data_normalized: (N, C, nx, ny, nt) - normalized data
        output_path: Output image path
        channel_names: List of channel names
        dpi: Image DPI
        sample_size: Number of samples to plot (None = all)
    """
    N, C, nx, ny, nt = data_normalized.shape

    # Sample if needed
    if sample_size is not None and N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        data_normalized = data_normalized[indices]

    # Create grid layout
    n_cols = 3
    n_rows = (C + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i in range(C):
        data = data_normalized[:, i].flatten().cpu().numpy()

        ax = axes[i]
        ax.hist(data, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_title(f'{channel_names[i]} (Normalized)', fontweight='bold', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(0.02, 0.98, f'μ={np.mean(data):.2f}\nσ={np.std(data):.2f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for i in range(C, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Normalized Distribution Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary plot saved: {output_path.name}")


def print_stats_table(stats: List[Dict], title: str = "Statistics"):
    """
    Print statistics table to console

    Args:
        stats: List of statistics dicts
        title: Table title
    """
    print(f"\n{title}")
    print("=" * 100)

    df = pd.DataFrame(stats)

    # Format floats for better readability
    pd.options.display.float_format = '{:.4e}'.format

    print(df.to_string(index=False))
    print("=" * 100)
