#!/usr/bin/env python3
"""
Channel-wise Normalization for FNO Preprocessing

This module implements channel-specific normalization strategies for different
input and output channels in the FNO groundwater uranium transport model.

Each channel has its own normalization strategy based on its physical characteristics:
- Permeability: UnitGaussian (log-normal distribution)
- Minerals (Calcite, Clino, Pyrite): MinMax after shifted log
- Smectite: MinMax (sparse, mostly zeros)
- Material IDs: No normalization (one-hot encoded)
- Velocities: MinMax (bounded physical quantities)
- Meta: MinMax (uniform LHS distribution)
- Uranium: UnitGaussian (concentration prediction)
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Import neuralop normalizers
import sys
sys.path.append('./')
from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer, MinMaxNormalizer


class ChannelWiseNormalizer:
    """
    Channel-specific transformation and normalization for FNO input/output tensors.

    This class handles TWO stages:
    1. Data Transformation: Convert raw data using log, shifted log
    2. Normalization: Apply channel-specific normalizers

    Raw Input channels (11 - already one-hot encoded from preprocessing_revised.py):
        0: Perm (raw)             → log10 → UnitGaussian
        1: Calcite (raw)          → shifted log → MinMax
        2: Clino (raw)            → shifted log → MinMax
        3: Pyrite (raw)           → shifted log → MinMax
        4: Smectite (raw)         → MinMax
        5: Material_Source        → None (already one-hot)
        6: Material_Bentonite     → None (already one-hot)
        7: Material_Fracture      → None (already one-hot)
        8: Vx (raw)               → MinMax
        9: Vy (raw)               → MinMax
        10: Meta (raw)            → MinMax

    Output channel (1):
        0: Uranium - UnitGaussianNormalizer (concentration)
    """

    def __init__(self, output_mode: str = 'log'):
        """
        Initialize channel-wise normalizer with proper configuration.

        Args:
            output_mode: Output transformation mode
                - 'raw': Use raw concentration values (linear scale)
                - 'log': Apply log10 transformation
                - 'delta': Compute delta from t=0 initial state (requires t=0 in data)
        """
        valid_modes = ['raw', 'log', 'delta']
        if output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be one of {valid_modes}")

        self.output_mode = output_mode

        # Channel names (11 channels)
        self.input_channel_names = [
            'Perm', 'Calcite', 'Clino', 'Pyrite', 'Smectite',
            'Material_Source', 'Material_Bentonite', 'Material_Fracture',
            'Vx', 'Vy', 'Meta'
        ]
        self.output_channel_names = ['Uranium']

        # Transformation configuration for RAW input (11 channels)
        self.transform_config = {
            0: {'type': 'log10'},                              # Perm
            1: {'type': 'shifted_log', 'eps': 1e-6},          # Calcite
            2: {'type': 'shifted_log', 'eps': 1e-6},          # Clino
            3: {'type': 'shifted_log', 'eps': 1e-9},          # Pyrite
            4: {'type': 'none'},                               # Smectite (keep raw)
            5: {'type': 'none'},                               # Material_Source (already one-hot)
            6: {'type': 'none'},                               # Material_Bentonite (already one-hot)
            7: {'type': 'none'},                               # Material_Fracture (already one-hot)
            8: {'type': 'none'},                               # Vx
            9: {'type': 'none'},                               # Vy
            10: {'type': 'none'},                              # Meta (keep raw)
        }

        # Normalization configuration for TRANSFORMED input (11 channels)
        self.norm_config = {
            'input': {
                0: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},   # Perm (after log10)
                1: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},         # Calcite (after shifted log)
                2: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},         # Clino (after shifted log)
                3: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},         # Pyrite (after shifted log)
                4: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},         # Smectite
                5: {'type': 'None'},  # Material_Source (one-hot)
                6: {'type': 'None'},  # Material_Bentonite (one-hot)
                7: {'type': 'None'},  # Material_Fracture (one-hot)
                8: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},         # Vx
                9: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},         # Vy
                10: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},        # Meta
            },
            'output': {
                0: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 0},  # Uranium
            }
        }

        # Initialize normalizers (will be populated during fit)
        self.input_normalizers = {}   # Dict[int, Normalizer]
        self.output_normalizer = None

    def apply_raw_transformations(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Apply transformations to raw input data.

        Args:
            raw_input: (N, 11, nx, ny, nt) - raw data (already one-hot encoded from preprocessing)

        Returns:
            transformed_input: (N, 11, nx, ny, nt) - transformed data
        """
        N, C_raw, nx, ny, nt = raw_input.shape
        assert C_raw == 11, f"Expected 11 raw input channels, got {C_raw}"

        transformed_channels = []

        for ch_idx in range(C_raw):
            transform_cfg = self.transform_config[ch_idx]
            ch_data = raw_input[:, ch_idx:ch_idx+1, :, :, :]  # (N, 1, nx, ny, nt)

            if transform_cfg['type'] == 'log10':
                # Apply log10 transformation
                transformed = torch.log10(ch_data)  # Add small epsilon for stability
                transformed_channels.append(transformed)

            elif transform_cfg['type'] == 'shifted_log':
                # Apply shifted log: log10(x + eps) - log10(eps)
                eps = transform_cfg['eps']
                transformed = torch.log10(ch_data + eps) - np.log10(eps)
                transformed_channels.append(transformed)

            elif transform_cfg['type'] == 'none':
                # Keep as-is (includes one-hot encoded material channels)
                transformed_channels.append(ch_data)

            else:
                raise ValueError(f"Unknown transformation type: {transform_cfg['type']}")

        # Concatenate all transformed channels
        transformed_input = torch.cat(transformed_channels, dim=1)  # (N, 11, nx, ny, nt)

        return transformed_input

    def apply_output_transformation(self, raw_output: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to raw output data based on output_mode.

        Args:
            raw_output: (N, 1, nx, ny, nt) - RAW uranium concentration

        Returns:
            transformed_output: (N, 1, nx, ny, nt) - transformed based on mode
        """
        N, C, nx, ny, nt = raw_output.shape
        assert C == 1, f"Expected 1 output channel, got {C}"

        if self.output_mode == 'raw':
            raw_output = raw_output[:, :, :, :, 1:]
            # Keep raw concentration
            return raw_output

        elif self.output_mode == 'log':
            raw_output = raw_output[:, :, :, :, 1:]
            # Apply log10 transformation
            return torch.log10(raw_output + 1e-12)

        elif self.output_mode == 'delta':
            # Compute delta from t=0 (first timestep)
            # Assuming t=0 is already included in the raw data
            if nt < 2:
                raise ValueError(f"Delta mode requires at least 2 timesteps, got {nt}")

            # Apply log first, then compute delta
            initial = raw_output[:, :, :, :, 0:1]  # (N, 1, nx, ny, 1) - reference at t=0
            delta_all = raw_output - initial  # (N, 1, nx, ny, nt) - delta from t=0
            delta = delta_all[:, :, :, :, 1:]  # (N, 1, nx, ny, nt-1) - exclude t=0

            return delta

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def fit(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, verbose: bool = True):
        """
        Fit normalizers on the entire dataset.

        Args:
            input_tensor: (N, 11, nx, ny, nt) - RAW input data (already one-hot encoded)
            output_tensor: (N, 1, nx, ny, nt) - output data
            verbose: whether to print progress
        """
        N, C_raw, nx, ny, nt = input_tensor.shape
        _, C_out, _, _, _ = output_tensor.shape

        if verbose:
            print(f"\n{'='*70}")
            print("Fitting Channel-Wise Normalizers")
            print(f"{'='*70}")
            print(f"Raw input shape: {tuple(input_tensor.shape)}")
            print(f"Raw output shape: {tuple(output_tensor.shape)}")
            print(f"Output mode: {self.output_mode}")

        # Step 1: Apply transformations to raw input data
        if verbose:
            print("\nApplying input transformations (log, shifted log)...")

        transformed_input = self.apply_raw_transformations(input_tensor)

        if verbose:
            print(f"Transformed input shape: {tuple(transformed_input.shape)}")

        # Step 2: Apply transformation to raw output data
        if verbose:
            print(f"\nApplying output transformation (mode={self.output_mode})...")

        transformed_output = self.apply_output_transformation(output_tensor)

        if verbose:
            print(f"Transformed output shape: {tuple(transformed_output.shape)}")
            print()

        # Step 3: Fit normalizers on transformed data
        C_in = transformed_input.shape[1]
        for ch_idx in range(C_in):
            ch_name = self.input_channel_names[ch_idx]
            ch_config = self.norm_config['input'][ch_idx]

            if ch_config['type'] == 'None':
                if verbose:
                    print(f"[{ch_idx:2d}] {ch_name:20s} - No normalization (one-hot)")
                continue

            # Extract channel data from TRANSFORMED input
            ch_data = transformed_input[:, ch_idx:ch_idx+1, :, :, :]  # (N, 1, nx, ny, nt)

            # Create normalizer
            if ch_config['type'] == 'UnitGaussian':
                normalizer = UnitGaussianNormalizer(
                    dim=ch_config['dim'],
                    eps=ch_config['eps']
                )
            elif ch_config['type'] == 'MinMax':
                normalizer = MinMaxNormalizer(
                    dim=ch_config['dim'],
                    eps=ch_config['eps']
                )
            else:
                raise ValueError(f"Unknown normalizer type: {ch_config['type']}")

            # Fit normalizer
            normalizer.fit(ch_data)
            self.input_normalizers[ch_idx] = normalizer

            # Print statistics
            if verbose:
                if ch_config['type'] == 'UnitGaussian':
                    mean_val = normalizer.mean.item() if normalizer.mean.numel() == 1 else normalizer.mean.mean().item()
                    std_val = normalizer.std.item() if normalizer.std.numel() == 1 else normalizer.std.mean().item()
                    print(f"[{ch_idx:2d}] {ch_name:20s} - UnitGaussian: mean={mean_val:10.4e}, std={std_val:10.4e}")
                elif ch_config['type'] == 'MinMax':
                    min_val = normalizer.data_min.item() if normalizer.data_min.numel() == 1 else normalizer.data_min.min().item()
                    max_val = normalizer.data_max.item() if normalizer.data_max.numel() == 1 else normalizer.data_max.max().item()
                    print(f"[{ch_idx:2d}] {ch_name:20s} - MinMax:       min={min_val:10.4e}, max={max_val:10.4e}")

        # Fit output normalizer
        if verbose:
            print()
            print("Output Normalizer:")

        out_config = self.norm_config['output'][0]
        if out_config['type'] == 'UnitGaussian':
            self.output_normalizer = UnitGaussianNormalizer(
                dim=out_config['dim'],
                eps=out_config['eps']
            )
            self.output_normalizer.fit(transformed_output)  # Fit on transformed output

            if verbose:
                mean_val = self.output_normalizer.mean.item() if self.output_normalizer.mean.numel() == 1 else self.output_normalizer.mean.mean().item()
                std_val = self.output_normalizer.std.item() if self.output_normalizer.std.numel() == 1 else self.output_normalizer.std.mean().item()
                print(f"[ 0] {self.output_channel_names[0]:20s} - UnitGaussian: mean={mean_val:10.4e}, std={std_val:10.4e}")

        if verbose:
            print(f"{'='*70}")
            print()

    def transform(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformations and channel-wise normalization.

        Args:
            input_tensor: (N, 11, nx, ny, nt) - RAW input data (already one-hot encoded)
            output_tensor: (N, 1, nx, ny, nt) - RAW output data

        Returns:
            normalized_input: (N, 11, nx, ny, nt-1) - transformed and normalized (t=0 removed)
            normalized_output: (N, 1, nx, ny, nt-1) - transformed and normalized (t=0 removed)
        """
        # Step 1: Apply raw transformations to input
        transformed_input = self.apply_raw_transformations(input_tensor)

        # Step 2: Remove t=0 from input to match output timesteps
        # (Output transformation will remove t=0, so input must match)
        transformed_input = transformed_input[:, :, :, :, 1:]  # (N, 11, nx, ny, nt-1)

        # Step 3: Apply normalization to transformed input
        N, C_in, nx, ny, nt = transformed_input.shape

        normalized_channels = []
        for ch_idx in range(C_in):
            if ch_idx in self.input_normalizers:
                # Apply normalization
                ch_data = transformed_input[:, ch_idx:ch_idx+1, :, :, :]
                ch_normalized = self.input_normalizers[ch_idx].transform(ch_data)
                normalized_channels.append(ch_normalized)
            else:
                # No normalization (material channels)
                normalized_channels.append(transformed_input[:, ch_idx:ch_idx+1, :, :, :])

        normalized_input = torch.cat(normalized_channels, dim=1)

        # Step 4: Apply transformation to output (will remove t=0)
        transformed_output = self.apply_output_transformation(output_tensor)

        # Step 5: Normalize transformed output
        normalized_output = self.output_normalizer.transform(transformed_output)

        return normalized_input, normalized_output

    def inverse_transform_input(self, normalized_input: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform for input channels.

        Args:
            normalized_input: (N, 11, nx, ny, nt)

        Returns:
            original_input: (N, 11, nx, ny, nt)
        """
        N, C_in, nx, ny, nt = normalized_input.shape

        original_channels = []
        for ch_idx in range(C_in):
            if ch_idx in self.input_normalizers:
                # Apply inverse normalization
                ch_normalized = normalized_input[:, ch_idx:ch_idx+1, :, :, :]
                ch_original = self.input_normalizers[ch_idx].inverse_transform(ch_normalized)
                original_channels.append(ch_original)
            else:
                # No normalization (already original)
                original_channels.append(normalized_input[:, ch_idx:ch_idx+1, :, :, :])

        return torch.cat(original_channels, dim=1)

    def inverse_transform_output(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform for output channel (normalization only).

        Args:
            normalized_output: (N, 1, nx, ny, nt) - normalized

        Returns:
            transformed_output: (N, 1, nx, ny, nt) - denormalized but still in transformed space (log/delta)
        """
        return self.output_normalizer.inverse_transform(normalized_output)

    def inverse_transform_output_to_raw(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        Complete inverse transform: normalized → transformed → raw physical values.

        This is the full pipeline inverse:
        1. Inverse normalization: normalized → transformed (log/delta space)
        2. Inverse transformation: transformed → raw physical values

        Args:
            normalized_output: (N, 1, nx, ny, nt) - normalized prediction from model

        Returns:
            raw_output: (N, 1, nx, ny, nt) - raw physical concentration values
        """
        # Get device from input tensor
        device = normalized_output.device

        # Ensure output_normalizer is on the same device
        if hasattr(self.output_normalizer, 'mean'):
            if self.output_normalizer.mean.device != device:
                self.output_normalizer.mean = self.output_normalizer.mean.to(device)
                self.output_normalizer.std = self.output_normalizer.std.to(device)

        # Step 1: Inverse normalization (normalized → transformed)
        transformed = self.output_normalizer.inverse_transform(normalized_output)

        # Step 2: Inverse transformation (transformed → raw)
        if self.output_mode == 'raw':
            # No transformation was applied, already in raw space
            return transformed

        elif self.output_mode == 'log':
            # Inverse log transformation: log(C) → C
            # transformed is in log10 space, convert back to linear
            return torch.pow(10, transformed)

        elif self.output_mode == 'delta':
            # Delta mode: Cannot fully reverse to absolute raw values
            # because we don't have the t=0 reference anymore
            # Return in log space (best approximation we can do)
            # User needs to be aware that delta predictions are relative changes
            return transformed

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def get_state_dict(self) -> Dict:
        """
        Serialize normalizer state to dictionary.

        Returns:
            Dictionary containing all normalizer parameters
        """
        state = {
            'transform_config': self.transform_config,
            'norm_config': self.norm_config,
            'output_mode': self.output_mode,
            'input_channel_names': self.input_channel_names,
            'output_channel_names': self.output_channel_names,
            'input_normalizers': {},
            'output_normalizer': None
        }

        # Save input normalizers
        for ch_idx, normalizer in self.input_normalizers.items():
            if isinstance(normalizer, UnitGaussianNormalizer):
                state['input_normalizers'][ch_idx] = {
                    'type': 'UnitGaussian',
                    'mean': normalizer.mean.cpu(),
                    'std': normalizer.std.cpu(),
                    'eps': normalizer.eps
                }
            elif isinstance(normalizer, MinMaxNormalizer):
                state['input_normalizers'][ch_idx] = {
                    'type': 'MinMax',
                    'data_min': normalizer.data_min.cpu(),
                    'data_max': normalizer.data_max.cpu(),
                    'data_range': normalizer.data_range.cpu(),
                    'eps': normalizer.eps
                }

        # Save output normalizer
        if isinstance(self.output_normalizer, UnitGaussianNormalizer):
            state['output_normalizer'] = {
                'type': 'UnitGaussian',
                'mean': self.output_normalizer.mean.cpu(),
                'std': self.output_normalizer.std.cpu(),
                'eps': self.output_normalizer.eps
            }
        elif isinstance(self.output_normalizer, MinMaxNormalizer):
            state['output_normalizer'] = {
                'type': 'MinMax',
                'data_min': self.output_normalizer.data_min.cpu(),
                'data_max': self.output_normalizer.data_max.cpu(),
                'data_range': self.output_normalizer.data_range.cpu(),
                'eps': self.output_normalizer.eps
            }

        return state

    @staticmethod
    def load_from_state_dict(state_dict: Dict) -> 'ChannelWiseNormalizer':
        """
        Restore normalizer from saved state.

        Args:
            state_dict: Dictionary from get_state_dict()

        Returns:
            ChannelWiseNormalizer instance with loaded parameters
        """
        # Restore output_mode
        output_mode = state_dict.get('output_mode', 'log')  # Default to 'log' for backward compatibility
        normalizer = ChannelWiseNormalizer(output_mode=output_mode)

        # Restore config and names
        normalizer.transform_config = state_dict['transform_config']
        normalizer.norm_config = state_dict['norm_config']
        normalizer.input_channel_names = state_dict['input_channel_names']
        normalizer.output_channel_names = state_dict['output_channel_names']

        # Restore input normalizers
        for ch_idx, norm_state in state_dict['input_normalizers'].items():
            ch_idx = int(ch_idx)  # JSON keys are strings

            if norm_state['type'] == 'UnitGaussian':
                norm = UnitGaussianNormalizer(
                    mean=norm_state['mean'],
                    std=norm_state['std'],
                    eps=norm_state['eps']
                )
            elif norm_state['type'] == 'MinMax':
                norm = MinMaxNormalizer(
                    data_min=norm_state['data_min'],
                    data_max=norm_state['data_max'],
                    eps=norm_state['eps']
                )
                norm.data_range = norm_state['data_range']
            else:
                raise ValueError(f"Unknown normalizer type: {norm_state['type']}")

            normalizer.input_normalizers[ch_idx] = norm

        # Restore output normalizer
        out_state = state_dict['output_normalizer']
        if out_state['type'] == 'UnitGaussian':
            normalizer.output_normalizer = UnitGaussianNormalizer(
                mean=out_state['mean'],
                std=out_state['std'],
                eps=out_state['eps']
            )
        elif out_state['type'] == 'MinMax':
            norm = MinMaxNormalizer(
                data_min=out_state['data_min'],
                data_max=out_state['data_max'],
                eps=out_state['eps']
            )
            norm.data_range = out_state['data_range']
            normalizer.output_normalizer = norm

        return normalizer

    def to(self, device):
        """Move all normalizers to device."""
        for normalizer in self.input_normalizers.values():
            normalizer.to(device)
        if self.output_normalizer is not None:
            self.output_normalizer.to(device)
        return self

    def cpu(self):
        """Move all normalizers to CPU."""
        for normalizer in self.input_normalizers.values():
            normalizer.cpu()
        if self.output_normalizer is not None:
            self.output_normalizer.cpu()
        return self

    def cuda(self):
        """Move all normalizers to CUDA."""
        for normalizer in self.input_normalizers.values():
            normalizer.cuda()
        if self.output_normalizer is not None:
            self.output_normalizer.cuda()
        return self


def analyze_normalization_distribution(
    data_before: torch.Tensor,
    data_after: torch.Tensor,
    data_type: str,
    output_dir: Path,
    config: Dict,
    channel_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Analyze and visualize distribution before and after normalization.

    Args:
        data_before: Data before normalization (N, C, nx, ny, nt)
        data_after: Data after normalization (N, C, nx, ny, nt)
        data_type: Type of data ('input' or 'output')
        output_dir: Output directory (normalization_check/)
        config: Configuration dictionary
        channel_names: List of channel names (optional)
        verbose: Whether to print progress

    Returns:
        Dictionary containing paths to generated files
    """
    if verbose:
        print(f"\nAnalyzing {data_type} normalization distribution...")

    N, C, nx, ny, nt = data_before.shape
    dpi = config.get('OUTPUT', {}).get('DPI', 150)

    # Default channel names
    if channel_names is None:
        if data_type == 'input':
            channel_names = [
                'Perm', 'Calcite', 'Clino', 'Pyrite',
                'Smectite', 'MatSrc', 'MatBent', 'MatFrac',
                'Vx', 'Vy', 'Meta'
            ]
        else:
            channel_names = ['Concentration']

    output_paths = {}

    # 1. Compute statistics for each channel
    stats_data = []
    for ch in range(C):
        before_ch = data_before[:, ch].cpu().numpy().flatten()
        after_ch = data_after[:, ch].cpu().numpy().flatten()

        stats = {
            'channel': channel_names[ch] if ch < len(channel_names) else f'Ch{ch}',
            'before_mean': float(np.mean(before_ch)),
            'before_std': float(np.std(before_ch)),
            'before_min': float(np.min(before_ch)),
            'before_max': float(np.max(before_ch)),
            'after_mean': float(np.mean(after_ch)),
            'after_std': float(np.std(after_ch)),
            'after_min': float(np.min(after_ch)),
            'after_max': float(np.max(after_ch)),
        }
        stats_data.append(stats)

    # Save statistics CSV
    stats_df = pd.DataFrame(stats_data)
    stats_path = output_dir / f'{data_type}_normalization_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    output_paths['stats_csv'] = stats_path

    if verbose:
        print(f"  Statistics saved: {stats_path.name}")
        print("\n  Summary:")
        print(stats_df.to_string(index=False))

    # 2. Create histograms for each channel
    hist_paths = []
    for ch in range(C):
        ch_name = channel_names[ch] if ch < len(channel_names) else f'Ch{ch}'

        before_ch = data_before[:, ch].cpu().numpy().flatten()
        after_ch = data_after[:, ch].cpu().numpy().flatten()

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Before normalization
        ax_before = axes[0]
        ax_before.hist(before_ch, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax_before.set_title(f'{ch_name} - Before Normalization', fontweight='bold', fontsize=12)
        ax_before.set_xlabel('Value')
        ax_before.set_ylabel('Frequency')
        ax_before.grid(True, alpha=0.3)
        ax_before.axvline(np.mean(before_ch), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(before_ch):.4e}')
        ax_before.legend()

        # After normalization
        ax_after = axes[1]
        ax_after.hist(after_ch, bins=100, alpha=0.7, color='green', edgecolor='black')
        ax_after.set_title(f'{ch_name} - After Normalization', fontweight='bold', fontsize=12)
        ax_after.set_xlabel('Value')
        ax_after.set_ylabel('Frequency')
        ax_after.grid(True, alpha=0.3)
        ax_after.axvline(np.mean(after_ch), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(after_ch):.4e}')
        ax_after.legend()

        plt.tight_layout()

        # Save
        hist_path = output_dir / f'{data_type}_ch{ch}_{ch_name}_histogram.png'
        plt.savefig(hist_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        hist_paths.append(hist_path)

        if verbose:
            print(f"  Saved: {hist_path.name}")

    output_paths['histograms'] = hist_paths

    # 3. Create combined summary plot (all channels)
    n_cols = 3
    n_rows = (C + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Handle different subplot configurations
    if n_rows == 1 and n_cols == 1:
        axes = [axes]  # Single subplot: wrap in list
    elif n_rows == 1 or n_cols == 1:
        axes = axes  # Already 1D array
    else:
        axes = axes.flatten()  # 2D array: flatten to 1D

    for ch in range(C):
        ch_name = channel_names[ch] if ch < len(channel_names) else f'Ch{ch}'
        after_ch = data_after[:, ch].cpu().numpy().flatten()

        ax = axes[ch]
        ax.hist(after_ch, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_title(f'{ch_name} (Normalized)', fontweight='bold', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Hide unused subplots
    for i in range(C, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{data_type.capitalize()} Data - Normalized Distribution Summary',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    summary_path = output_dir / f'{data_type}_normalized_summary.png'
    plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    output_paths['summary_plot'] = summary_path

    if verbose:
        print(f"  Summary plot saved: {summary_path.name}")

    return output_paths


def apply_channel_normalization(
    merged_data_path: str,
    output_path: str,
    config: Dict,
    output_mode: str = 'log',
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Apply channel-wise normalization to merged preprocessing data.

    Args:
        merged_data_path: Path to merged data (e.g., merged_U_raw.pt with RAW output)
        output_path: Path to save normalized data
        config: Configuration dictionary with normalization check options
        output_mode: Output transformation mode ('raw', 'log', 'delta')
        verbose: Whether to print progress

    Returns:
        Dictionary containing paths to generated files:
            - normalized_data_path: Path to normalized .pt file
            - normalization_check_dir: Directory with statistics
            - stats_csv: CSV files with statistics
            - histograms: List of histogram image paths
            - summary_plot: Summary plot path
    """
    if verbose:
        print(f"\n{'='*70}")
        print("Channel-Wise Normalization Pipeline")
        print(f"{'='*70}")
        print(f"Input:  {merged_data_path}")
        print(f"Output: {output_path}")
        print(f"Output mode: {output_mode}")
        print()

    # Load merged data
    if verbose:
        print("Step 1: Loading merged data...")

    data = torch.load(merged_data_path, map_location='cpu')
    x = data['x']        # (N, 11, nx, ny, nt) - RAW data (already one-hot encoded with meta)
    y = data['y']        # (N, 1, nx, ny, nt) - RAW uranium concentration
    xc = data['xc']
    yc = data['yc']
    time_keys = data['time_keys']

    N, C_raw, nx, ny, nt = x.shape

    if verbose:
        print(f"   Raw input shape: {tuple(x.shape)} (11 channels: already one-hot encoded + meta)")
        print(f"   Raw output shape: {tuple(y.shape)} (RAW uranium concentration)")

    # Data is already prepared with one-hot encoding and meta channel
    raw_input_with_meta = x  # (N, 11, nx, ny, nt)

    if verbose:
        print("\nStep 2: Raw data is already prepared (one-hot + meta from preprocessing_revised.py)")

    # Create normalizer and fit on RAW data (will apply transformations internally)
    if verbose:
        print("\nStep 3: Creating and fitting normalizers...")
        print(f"   (Input transformations: log, shifted log)")
        print(f"   (Output transformation: {output_mode})")

    normalizer = ChannelWiseNormalizer(output_mode=output_mode)
    normalizer.fit(raw_input_with_meta, y, verbose=verbose)

    # Apply transformations and normalization
    if verbose:
        print("\nStep 4: Applying transformations and normalization...")

    normalized_input, normalized_output = normalizer.transform(raw_input_with_meta, y)

    if verbose:
        print(f"   Transformed and normalized input shape: {tuple(normalized_input.shape)}")
        print(f"   Transformed and normalized output shape: {tuple(normalized_output.shape)}")

    # Perform normalization distribution check (always enabled)
    result_paths = {}

    if verbose:
        print("\nStep 5: Performing normalization distribution check...")

    # Setup output directory
    output_path_obj = Path(output_path)
    norm_check_dir = output_path_obj.parent / 'normalization_check'
    norm_check_dir.mkdir(parents=True, exist_ok=True)

    # Take samples for analysis
    n_samples = config.get('OUTPUT', {}).get('N_SAMPLES')

    # Apply transformations to samples (for before/after comparison)
    sample_raw = raw_input_with_meta[:n_samples]
    sample_transformed = normalizer.apply_raw_transformations(sample_raw)  # BEFORE normalization
    sample_normalized = normalized_input[:n_samples]  # AFTER normalization

    sample_output_before = y[:n_samples]
    sample_output_after = normalized_output[:n_samples]

    # Analyze input distribution (transformed before vs normalized after)
    input_results = analyze_normalization_distribution(
        sample_transformed, sample_normalized,
        'input', norm_check_dir, config, verbose=verbose
    )

    # Analyze output distribution
    output_results = analyze_normalization_distribution(
        sample_output_before, sample_output_after,
        'output', norm_check_dir, config, verbose=verbose
    )

    result_paths['normalization_check_dir'] = norm_check_dir
    result_paths['input_stats_csv'] = input_results['stats_csv']
    result_paths['output_stats_csv'] = output_results['stats_csv']
    result_paths['input_histograms'] = input_results['histograms']
    result_paths['output_histograms'] = output_results['histograms']
    result_paths['input_summary'] = input_results['summary_plot']
    result_paths['output_summary'] = output_results['summary_plot']

    if verbose:
        print(f"   Normalization check completed. Results saved to: {norm_check_dir}")

    # Save normalized data with normalizer state
    if verbose:
        print("\nStep 6: Saving normalized data...")

    save_data = {
        'x': normalized_input,
        'y': normalized_output,
        'xc': xc,
        'yc': yc,
        'time_keys': time_keys,
        'normalizer_state': normalizer.get_state_dict()
    }

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path_obj)

    result_paths['normalized_data_path'] = output_path_obj

    if verbose:
        print(f"   Saved normalized data: {output_path_obj}")
        print(f"   File size: {output_path_obj.stat().st_size / 1024 / 1024:.2f} MB")

    # Save channel_normalizer as pickle for direct use in FNO.py
    if verbose:
        print("\nStep 7: Saving channel_normalizer as pickle...")

    # Move normalizer to CPU before saving (for compatibility)
    normalizer_cpu = normalizer.cpu()

    # Save as pickle in the same directory
    pickle_path = output_path_obj.parent / 'channel_normalizer.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(normalizer_cpu, f)

    result_paths['channel_normalizer_pkl'] = pickle_path

    if verbose:
        print(f"   Saved channel_normalizer: {pickle_path}")
        print(f"   Pickle size: {pickle_path.stat().st_size / 1024:.2f} KB")
        print(f"\n{'='*70}")
        print("Channel-wise normalization completed successfully!")
        print(f"{'='*70}\n")

    return result_paths


if __name__ == "__main__":
    """
    Test the channel-wise normalizer with sample data.
    """
    print("Testing ChannelWiseNormalizer with RAW data...")

    # Create dummy RAW data (9 channels: 8 + meta)
    N, nx, ny, nt = 100, 64, 32, 20
    # Raw: Perm, Calcite, Clino, Pyrite, Smectite, Material(1-3), Vx, Vy, Meta
    raw_input = torch.abs(torch.randn(N, 9, nx, ny, nt))  # Use abs to ensure positive values
    # Set material channel to valid IDs (1, 2, or 3)
    raw_input[:, 5] = torch.randint(1, 4, (N, nx, ny, nt)).float()

    raw_output = torch.randn(N, 1, nx, ny, nt)

    # Test fit and transform
    normalizer = ChannelWiseNormalizer()
    normalizer.fit(raw_input, raw_output, verbose=True)

    norm_input, norm_output = normalizer.transform(raw_input, raw_output)

    print(f"\nTransformed and normalized input shape: {norm_input.shape}")
    print(f"Normalized output shape: {norm_output.shape}")

    # Verify transformation worked (11 channels from 9 raw channels)
    assert norm_input.shape[1] == 11, f"Expected 11 transformed channels, got {norm_input.shape[1]}"

    # Test inverse transform for output (should work)
    recovered_output = normalizer.inverse_transform_output(norm_output)
    print(f"\nInverse transform error (output): {(recovered_output - raw_output).abs().max().item():.6e}")

    # Test save and load
    state = normalizer.get_state_dict()
    loaded_normalizer = ChannelWiseNormalizer.load_from_state_dict(state)

    norm_input2, norm_output2 = loaded_normalizer.transform(raw_input, raw_output)

    print(f"\nLoad/save error (input): {(norm_input2 - norm_input).abs().max().item():.6e}")
    print(f"Load/save error (output): {(norm_output2 - norm_output).abs().max().item():.6e}")

    print("\nAll tests passed!")
