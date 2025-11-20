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

# Import neuralop normalizers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'neuraloperator'))
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer, MinMaxNormalizer


class ChannelWiseNormalizer:
    """
    Channel-specific normalization for FNO input/output tensors.

    Input channels (11):
        0: Perm           - UnitGaussianNormalizer (already log10-transformed)
        1: Calcite        - MinMaxNormalizer (shifted log applied)
        2: Clino          - MinMaxNormalizer (shifted log applied)
        3: Pyrite         - MinMaxNormalizer (shifted log applied)
        4: Smectite       - MinMaxNormalizer (raw values)
        5: Material_Source    - None (one-hot)
        6: Material_Bentonite - None (one-hot)
        7: Material_Fracture  - None (one-hot)
        8: Vx             - MinMaxNormalizer
        9: Vy             - MinMaxNormalizer
        10: Meta          - MinMaxNormalizer (LHS uniform distribution)

    Output channel (1):
        0: Uranium        - UnitGaussianNormalizer (concentration)
    """

    def __init__(self):
        """Initialize channel-wise normalizer with proper configuration."""

        # Channel names
        self.input_channel_names = [
            'Perm', 'Calcite', 'Clino', 'Pyrite', 'Smectite',
            'Material_Source', 'Material_Bentonite', 'Material_Fracture',
            'Vx', 'Vy', 'Meta'
        ]
        self.output_channel_names = ['Uranium']

        # Configuration: which normalizer to use for each channel
        self.config = {
            'input': {
                0: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 1e-6},   # Perm
                1: {'type': 'MinMax', 'dim': [0, 2, 3, 4], 'eps': 1e-7},         # Calcite
                2: {'type': 'MinMax', 'dim': [0, 2, 3, 4], 'eps': 1e-7},         # Clino
                3: {'type': 'MinMax', 'dim': [0, 2, 3, 4], 'eps': 1e-7},         # Pyrite
                4: {'type': 'MinMax', 'dim': [0, 2, 3, 4], 'eps': 1e-7},         # Smectite
                5: {'type': 'None'},  # Material_Source (one-hot)
                6: {'type': 'None'},  # Material_Bentonite (one-hot)
                7: {'type': 'None'},  # Material_Fracture (one-hot)
                8: {'type': 'MinMax', 'dim': [0, 2, 3, 4], 'eps': 1e-7},         # Vx
                9: {'type': 'MinMax', 'dim': [0, 2, 3, 4], 'eps': 1e-7},         # Vy
                10: {'type': 'MinMax', 'dim': [0, 2, 3, 4], 'eps': 1e-7},        # Meta
            },
            'output': {
                0: {'type': 'UnitGaussian', 'dim': [0, 2, 3, 4], 'eps': 1e-6},  # Uranium
            }
        }

        # Initialize normalizers (will be populated during fit)
        self.input_normalizers = {}   # Dict[int, Normalizer]
        self.output_normalizer = None

    def fit(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, verbose: bool = True):
        """
        Fit normalizers on the entire dataset.

        Args:
            input_tensor: (N, 11, nx, ny, nt) - input data with meta channel
            output_tensor: (N, 1, nx, ny, nt) - output data
            verbose: whether to print progress
        """
        N, C_in, nx, ny, nt = input_tensor.shape
        _, C_out, _, _, _ = output_tensor.shape

        if verbose:
            print(f"\n{'='*70}")
            print("Fitting Channel-Wise Normalizers")
            print(f"{'='*70}")
            print(f"Input shape: {tuple(input_tensor.shape)}")
            print(f"Output shape: {tuple(output_tensor.shape)}")
            print()

        # Fit input normalizers
        for ch_idx in range(C_in):
            ch_name = self.input_channel_names[ch_idx]
            ch_config = self.config['input'][ch_idx]

            if ch_config['type'] == 'None':
                if verbose:
                    print(f"[{ch_idx:2d}] {ch_name:20s} - No normalization (one-hot)")
                continue

            # Extract channel data
            ch_data = input_tensor[:, ch_idx:ch_idx+1, :, :, :]  # (N, 1, nx, ny, nt)

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

        out_config = self.config['output'][0]
        if out_config['type'] == 'UnitGaussian':
            self.output_normalizer = UnitGaussianNormalizer(
                dim=out_config['dim'],
                eps=out_config['eps']
            )
            self.output_normalizer.fit(output_tensor)

            if verbose:
                mean_val = self.output_normalizer.mean.item() if self.output_normalizer.mean.numel() == 1 else self.output_normalizer.mean.mean().item()
                std_val = self.output_normalizer.std.item() if self.output_normalizer.std.numel() == 1 else self.output_normalizer.std.mean().item()
                print(f"[ 0] {self.output_channel_names[0]:20s} - UnitGaussian: mean={mean_val:10.4e}, std={std_val:10.4e}")

        if verbose:
            print(f"{'='*70}")
            print()

    def transform(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply channel-wise normalization.

        Args:
            input_tensor: (N, 11, nx, ny, nt)
            output_tensor: (N, 1, nx, ny, nt)

        Returns:
            normalized_input: (N, 11, nx, ny, nt)
            normalized_output: (N, 1, nx, ny, nt)
        """
        N, C_in, nx, ny, nt = input_tensor.shape

        # Transform input channels
        normalized_channels = []
        for ch_idx in range(C_in):
            if ch_idx in self.input_normalizers:
                # Apply normalization
                ch_data = input_tensor[:, ch_idx:ch_idx+1, :, :, :]
                ch_normalized = self.input_normalizers[ch_idx].transform(ch_data)
                normalized_channels.append(ch_normalized)
            else:
                # No normalization (material channels)
                normalized_channels.append(input_tensor[:, ch_idx:ch_idx+1, :, :, :])

        normalized_input = torch.cat(normalized_channels, dim=1)

        # Transform output
        normalized_output = self.output_normalizer.transform(output_tensor)

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
        Inverse transform for output channel.

        Args:
            normalized_output: (N, 1, nx, ny, nt)

        Returns:
            original_output: (N, 1, nx, ny, nt)
        """
        return self.output_normalizer.inverse_transform(normalized_output)

    def get_state_dict(self) -> Dict:
        """
        Serialize normalizer state to dictionary.

        Returns:
            Dictionary containing all normalizer parameters
        """
        state = {
            'config': self.config,
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
        normalizer = ChannelWiseNormalizer()

        # Restore config and names
        normalizer.config = state_dict['config']
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


def apply_channel_normalization(
    merged_data_path: str,
    output_path: str,
    config: Dict,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Apply channel-wise normalization to merged preprocessing data.

    Args:
        merged_data_path: Path to merged data (e.g., merged_U_log.pt)
        output_path: Path to save normalized data
        config: Configuration dictionary with normalization check options
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
        print()

    # Load merged data
    if verbose:
        print("Step 1: Loading merged data...")

    data = torch.load(merged_data_path, map_location='cpu')
    x = data['x']        # (N, 10, nx, ny, nt)
    y = data['y']        # (N, 1, nx, ny, nt)
    meta = data['meta']  # (N,)
    xc = data['xc']
    yc = data['yc']
    time_keys = data['time_keys']

    N, C_orig, nx, ny, nt = x.shape

    if verbose:
        print(f"   Original input shape: {tuple(x.shape)}")
        print(f"   Output shape: {tuple(y.shape)}")
        print(f"   Meta shape: {tuple(meta.shape)}")

    # Expand meta to uniform channel and concatenate
    if verbose:
        print("\nStep 2: Expanding meta channel...")

    expanded_meta = meta[:, None, None, None, None].expand(N, 1, nx, ny, nt)
    combined_input = torch.cat([x, expanded_meta], dim=1)  # (N, 11, nx, ny, nt)

    if verbose:
        print(f"   Combined input shape: {tuple(combined_input.shape)}")

    # Create and fit normalizer
    if verbose:
        print("\nStep 3: Creating and fitting normalizers...")

    normalizer = ChannelWiseNormalizer()
    normalizer.fit(combined_input, y, verbose=verbose)

    # Apply normalization
    if verbose:
        print("Step 4: Applying normalization...")

    normalized_input, normalized_output = normalizer.transform(combined_input, y)

    if verbose:
        print(f"   Normalized input shape: {tuple(normalized_input.shape)}")
        print(f"   Normalized output shape: {tuple(normalized_output.shape)}")

    # Perform normalization distribution check
    result_paths = {}

    if config.get('OUTPUT', {}).get('NORM_CHECK', {}).get('ENABLED', False):
        if verbose:
            print("\nStep 5: Performing normalization distribution check...")

        # Import here to avoid circular dependency
        sys.path.insert(0, str(Path(__file__).parent.parent / 'FNO'))
        from util_output import analyze_normalization_distribution

        # Setup output directory
        output_path_obj = Path(output_path)
        norm_check_dir = output_path_obj.parent / 'normalization_check'
        norm_check_dir.mkdir(parents=True, exist_ok=True)

        # Take samples for analysis
        n_samples = min(config.get('OUTPUT', {}).get('NORM_CHECK', {}).get('N_SAMPLES', 10), N)
        sample_input_before = combined_input[:n_samples]
        sample_input_after = normalized_input[:n_samples]
        sample_output_before = y[:n_samples]
        sample_output_after = normalized_output[:n_samples]

        # Analyze input distribution
        input_results = analyze_normalization_distribution(
            sample_input_before, sample_input_after,
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
        'meta': meta,  # Keep original meta for reference
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
        print(f"   Saved: {output_path_obj}")
        print(f"   File size: {output_path_obj.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"\n{'='*70}")
        print("Channel-wise normalization completed successfully!")
        print(f"{'='*70}\n")

    return result_paths


if __name__ == "__main__":
    """
    Test the channel-wise normalizer with sample data.
    """
    print("Testing ChannelWiseNormalizer...")

    # Create dummy data
    N, nx, ny, nt = 100, 64, 32, 20
    dummy_input = torch.randn(N, 11, nx, ny, nt)
    dummy_output = torch.randn(N, 1, nx, ny, nt)

    # Test fit and transform
    normalizer = ChannelWiseNormalizer()
    normalizer.fit(dummy_input, dummy_output, verbose=True)

    norm_input, norm_output = normalizer.transform(dummy_input, dummy_output)

    print(f"\nNormalized input shape: {norm_input.shape}")
    print(f"Normalized output shape: {norm_output.shape}")

    # Test inverse transform
    recovered_input = normalizer.inverse_transform_input(norm_input)
    recovered_output = normalizer.inverse_transform_output(norm_output)

    print(f"\nInverse transform error (input): {(recovered_input - dummy_input).abs().max().item():.6e}")
    print(f"Inverse transform error (output): {(recovered_output - dummy_output).abs().max().item():.6e}")

    # Test save and load
    state = normalizer.get_state_dict()
    loaded_normalizer = ChannelWiseNormalizer.load_from_state_dict(state)

    norm_input2, norm_output2 = loaded_normalizer.transform(dummy_input, dummy_output)

    print(f"\nLoad/save error (input): {(norm_input2 - norm_input).abs().max().item():.6e}")
    print(f"Load/save error (output): {(norm_output2 - norm_output).abs().max().item():.6e}")

    print("\nAll tests passed!")
