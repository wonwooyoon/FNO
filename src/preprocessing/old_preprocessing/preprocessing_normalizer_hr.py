#!/usr/bin/env python3
"""
High-Resolution Data Normalization using Pre-trained Normalizer

This script applies a pre-trained normalizer (from low-resolution training data)
to high-resolution test data for zero-shot super-resolution evaluation.

Key principle: For zero-shot super-resolution, the normalizer MUST be identical
to the one used during training, regardless of resolution differences.

Usage:
    python preprocessing_normalizer_hr.py \
        --hr_data merged_U_raw_highres.pt \
        --normalizer channel_normalizer_log.pkl \
        --output merged_U_log_normalized_highres.pt
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

# Import preprocessing normalizer module
sys.path.append('./')
from preprocessing_normalizer import ChannelWiseNormalizer


def apply_pretrained_normalizer_to_hr_data(
    hr_data_path: str,
    normalizer_path: str,
    output_path: str,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Apply pre-trained normalizer to high-resolution data for zero-shot super-resolution.

    This function ensures that high-resolution test data is normalized using the
    EXACT SAME normalizer that was fitted on low-resolution training data.

    Args:
        hr_data_path: Path to high-resolution raw data (e.g., merged_U_raw_highres.pt)
        normalizer_path: Path to pre-trained normalizer pickle file (e.g., channel_normalizer_log.pkl)
        output_path: Path to save normalized high-resolution data
        verbose: Whether to print progress information

    Returns:
        Dictionary containing paths to generated files:
            - normalized_data_path: Path to normalized .pt file
            - normalizer_path: Path to normalizer used (same as input)

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If data shapes or normalizer configurations are incompatible
    """
    if verbose:
        print(f"\n{'='*70}")
        print("High-Resolution Data Normalization with Pre-trained Normalizer")
        print(f"{'='*70}")
        print(f"HR Data:    {hr_data_path}")
        print(f"Normalizer: {normalizer_path}")
        print(f"Output:     {output_path}")
        print()

    # Validate input files
    hr_data_path_obj = Path(hr_data_path)
    normalizer_path_obj = Path(normalizer_path)

    if not hr_data_path_obj.exists():
        raise FileNotFoundError(f"High-resolution data not found: {hr_data_path}")

    if not normalizer_path_obj.exists():
        raise FileNotFoundError(f"Normalizer not found: {normalizer_path}")

    # Step 1: Load pre-trained normalizer
    if verbose:
        print("Step 1: Loading pre-trained normalizer...")

    with open(normalizer_path_obj, 'rb') as f:
        normalizer = pickle.load(f)

    if not isinstance(normalizer, ChannelWiseNormalizer):
        raise TypeError(
            f"Expected ChannelWiseNormalizer, got {type(normalizer).__name__}"
        )

    if verbose:
        print(f"   Normalizer type: ChannelWiseNormalizer")
        print(f"   Output mode: {normalizer.output_mode}")
        print(f"   Input channels: {len(normalizer.input_channel_names)}")
        print(f"   Output channels: {len(normalizer.output_channel_names)}")

        # Show channel-specific normalizer info
        print(f"\n   Fitted normalizers:")
        for ch_idx, norm in normalizer.input_normalizers.items():
            ch_name = normalizer.input_channel_names[ch_idx]
            norm_type = type(norm).__name__
            print(f"      [{ch_idx:2d}] {ch_name:20s} - {norm_type}")

        if normalizer.output_normalizer is not None:
            out_norm_type = type(normalizer.output_normalizer).__name__
            print(f"      [ 0] {normalizer.output_channel_names[0]:20s} - {out_norm_type}")

    # Step 2: Load high-resolution raw data
    if verbose:
        print("\nStep 2: Loading high-resolution raw data...")

    data_bundle = torch.load(hr_data_path_obj, map_location='cpu', weights_only=False)

    required_keys = ['x', 'y', 'xc', 'yc', 'time_keys']
    missing_keys = [key for key in required_keys if key not in data_bundle]
    if missing_keys:
        raise KeyError(f"Missing required keys in HR data: {missing_keys}")

    x_hr_raw = data_bundle['x']  # (N, 11, nx_hr, ny_hr, nt)
    y_hr_raw = data_bundle['y']  # (N, 1, nx_hr, ny_hr, nt)
    xc_hr = data_bundle['xc']
    yc_hr = data_bundle['yc']
    time_keys = data_bundle['time_keys']

    N, C_in, nx_hr, ny_hr, nt = x_hr_raw.shape
    _, C_out, _, _, _ = y_hr_raw.shape

    if verbose:
        print(f"   HR input shape:  {tuple(x_hr_raw.shape)}")
        print(f"   HR output shape: {tuple(y_hr_raw.shape)}")
        print(f"   Spatial resolution: {nx_hr} × {ny_hr}")
        print(f"   Time steps: {nt}")

    # Validate channel counts
    expected_input_channels = len(normalizer.input_channel_names)
    if C_in != expected_input_channels:
        raise RuntimeError(
            f"Channel count mismatch! "
            f"Normalizer expects {expected_input_channels} input channels, "
            f"but HR data has {C_in} channels."
        )

    if C_out != len(normalizer.output_channel_names):
        raise RuntimeError(
            f"Output channel mismatch! "
            f"Normalizer expects {len(normalizer.output_channel_names)} output channels, "
            f"but HR data has {C_out} channels."
        )

    # Step 3: Apply same transformations and normalization
    if verbose:
        print("\nStep 3: Applying pre-trained normalizer to HR data...")
        print(f"   (Using SAME statistics as low-resolution training data)")

    # Transform and normalize using pre-trained normalizer
    # The normalizer is resolution-independent!
    x_hr_normalized, y_hr_normalized = normalizer.transform(x_hr_raw, y_hr_raw)

    if verbose:
        print(f"   Normalized HR input shape:  {tuple(x_hr_normalized.shape)}")
        print(f"   Normalized HR output shape: {tuple(y_hr_normalized.shape)}")

        # Show normalization statistics
        print(f"\n   Normalization check:")
        print(f"      Input  - mean: {x_hr_normalized.mean().item():10.4e}, std: {x_hr_normalized.std().item():10.4e}")
        print(f"      Output - mean: {y_hr_normalized.mean().item():10.4e}, std: {y_hr_normalized.std().item():10.4e}")

    # Step 4: Save normalized high-resolution data
    if verbose:
        print("\nStep 4: Saving normalized HR data...")

    save_data = {
        'x': x_hr_normalized,
        'y': y_hr_normalized,
        'xc': xc_hr,
        'yc': yc_hr,
        'time_keys': time_keys,
    }

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path_obj)

    if verbose:
        print(f"   Saved: {output_path_obj}")
        print(f"   File size: {output_path_obj.stat().st_size / 1024 / 1024:.2f} MB")

    # Step 5: Summary and important notes
    if verbose:
        print(f"\n{'='*70}")
        print("SUCCESS: High-resolution data normalized!")
        print(f"{'='*70}")
        print("\n⚠️  IMPORTANT NOTES:")
        print(f"   1. This HR data was normalized using the SAME normalizer")
        print(f"      as the low-resolution training data.")
        print(f"   2. For predictions, use the SAME normalizer for inverse transform:")
        print(f"      normalizer_path: {normalizer_path}")
        print(f"   3. DO NOT create a new normalizer for this HR data!")
        print(f"   4. The resolution difference is handled automatically by FNO.")
        print(f"{'='*70}\n")

    return {
        'normalized_data_path': output_path_obj,
        'normalizer_path': normalizer_path_obj
    }


def main():
    """
    Command-line interface for high-resolution data normalization.
    """
    parser = argparse.ArgumentParser(
        description='Apply pre-trained normalizer to high-resolution data for zero-shot super-resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply normalizer trained on low-res data to high-res test data
  python preprocessing_normalizer_hr.py \\
      --hr_data ./src/preprocessing/merged_U_raw_highres.pt \\
      --normalizer ./src/preprocessing/channel_normalizer_log.pkl \\
      --output ./src/preprocessing/merged_U_log_normalized_highres.pt

  # With custom output mode specified in normalizer name
  python preprocessing_normalizer_hr.py \\
      --hr_data merged_U_raw_hr.pt \\
      --normalizer channel_normalizer_delta.pkl \\
      --output merged_U_delta_normalized_hr.pt

Notes:
  - The normalizer MUST be from the low-resolution training dataset
  - Do NOT create a new normalizer for high-resolution data
  - Resolution differences are handled automatically by FNO architecture
        """
    )

    parser.add_argument(
        '--hr_data',
        type=str,
        required=True,
        help='Path to high-resolution raw data file (merged_U_raw_*.pt)'
    )

    parser.add_argument(
        '--normalizer',
        type=str,
        required=True,
        help='Path to pre-trained normalizer pickle file (channel_normalizer_*.pkl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save normalized high-resolution data'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print verbose progress information (default: True)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output (overrides --verbose)'
    )

    args = parser.parse_args()

    # Handle verbosity
    verbose = args.verbose and not args.quiet

    # Run normalization
    try:
        result = apply_pretrained_normalizer_to_hr_data(
            hr_data_path=args.hr_data,
            normalizer_path=args.normalizer,
            output_path=args.output,
            verbose=verbose
        )

        if verbose:
            print("\n✓ Normalization completed successfully!")
            print(f"  Output: {result['normalized_data_path']}")
            print(f"  Use normalizer: {result['normalizer_path']} for inverse transform")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
