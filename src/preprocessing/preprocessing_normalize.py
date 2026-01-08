#!/usr/bin/env python3
"""
Data Normalization Main Script

채널별 transformation과 normalization을 적용하고,
LR/HR 모드에 따라 normalizer를 생성하거나 재사용

Usage:
    # LR mode: Create new normalizer
    python preprocessing_normalize.py \\
        --mode lr \\
        --input merged_raw.pt \\
        --output-mode log \\
        --output merged_normalized.pt

    # HR mode: Use existing normalizer
    python preprocessing_normalize.py \\
        --mode hr \\
        --input merged_raw_hr.pt \\
        --normalizer normalizer_log.pkl \\
        --output merged_normalized_hr.pt
"""

import argparse
import sys
from pathlib import Path
import torch
import pickle

from normalizer_core import ChannelNormalizer, OutletNormalizer
from normalizer_utils import (
    compute_normalization_stats,
    save_stats_csv,
    visualize_distributions,
    visualize_normalized_summary,
    print_stats_table
)


# ============================================================================
# Channel Configuration
# ============================================================================

CHANNEL_CONFIG = [
    # (channel_idx, name, transform_type, normalizer_type)
    # transform_type: 'log10', 'none', or ('shifted_log', eps)
    (0,  'Perm',              'log10',                'UnitGaussian'),
    (1,  'Calcite',           ('shifted_log', 1e-6), 'UnitGaussian'),
    (2,  'Clino',             ('shifted_log', 1e-6), 'UnitGaussian'),
    (3,  'Pyrite',            ('shifted_log', 1e-9), 'UnitGaussian'),
    (4,  'Smectite',          'none',                 'UnitGaussian'),
    (5,  'Material_Source',   'none',                 'none'),  # one-hot
    (6,  'Material_Bentonite','none',                 'none'),  # one-hot
    (7,  'Material_Fracture', 'none',                 'none'),  # one-hot
    (8,  'Vx',                'none',                 'UnitGaussian'),
    (9,  'Vy',                'none',                 'UnitGaussian'),
    (10, 'Meta',              'none',                 'UnitGaussian'),
]

OUTPUT_TRANSFORM_CONFIG = {
    'log': {
        'transform': 'log10',
        'remove_t0': True,
    },
    'raw': {
        'transform': 'none',
        'remove_t0': True,
    },
    'delta': {
        'transform': 'delta',
        'remove_t0': True,
        'mask_source': True
    }
}


# ============================================================================
# LR Mode: Create New Normalizer
# ============================================================================

def normalize_lr(
    raw_data_path: str,
    output_mode: str,
    output_path: str,
    run_analysis: bool = True,
    analysis_sample_size: int = 3000
):
    """
    LR 데이터에 대해 새로운 normalizer 생성 및 적용

    흐름:
    1. Raw 데이터 로드
    2. ChannelNormalizer 생성 (CHANNEL_CONFIG 기반)
    3. Fit (통계 계산)
    4. Transform (정규화 적용)
    5. 결과 저장 (normalized data + normalizer pickle)
    6. (Optional) 통계 분석 및 시각화

    Args:
        raw_data_path: Path to raw merged data
        output_mode: 'log', 'raw', 'delta'
        output_path: Path to save normalized data
        run_analysis: Whether to run statistical analysis
        analysis_sample_size: Number of samples for analysis plots

    Returns:
        dict with paths to generated files
    """
    print(f"\n{'='*70}")
    print(f"LR Mode: Creating New Normalizer")
    print(f"{'='*70}")
    print(f"Input:       {raw_data_path}")
    print(f"Output:      {output_path}")
    print(f"Output mode: {output_mode}")
    print(f"{'='*70}\n")

    # Validate output mode
    if output_mode not in OUTPUT_TRANSFORM_CONFIG:
        raise ValueError(f"Invalid output_mode: {output_mode}. Must be one of {list(OUTPUT_TRANSFORM_CONFIG.keys())}")

    # Set preprocessing directory base (always run from FNO root)
    PREPROC_DIR = Path('./src/preprocessing')

    # Process raw_data_path
    raw_data_path = PREPROC_DIR / raw_data_path

    # 1. Load raw data
    print("\nStep 1: Loading raw data...")
    data = torch.load(raw_data_path, map_location='cpu')

    required_keys = ['x', 'y', 'y_outlet', 'xc', 'yc', 'time_keys']
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in data: {missing}")

    x_raw = data['x']              # (N, 11, nx, ny, nt)
    y_raw = data['y']              # (N, 1, nx, ny, nt)
    y_outlet_raw = data['y_outlet']  # (N, nt)

    print(f"  Raw input shape:   {tuple(x_raw.shape)}")
    print(f"  Raw output shape:  {tuple(y_raw.shape)}")
    print(f"  Raw outlet shape:  {tuple(y_outlet_raw.shape)}")
    print(f"  Samples: {x_raw.shape[0]}")
    print(f"  Spatial: {x_raw.shape[2]} × {x_raw.shape[3]}")
    print(f"  Time steps: {x_raw.shape[4]}")

    # 2. Create normalizers
    print("\nStep 2: Creating normalizers...")
    print("  2.1 Creating spatial normalizer...")
    normalizer = ChannelNormalizer(
        input_config=CHANNEL_CONFIG,
        output_mode=output_mode,
        output_config=OUTPUT_TRANSFORM_CONFIG[output_mode]
    )

    print("  2.2 Creating outlet normalizer...")
    outlet_normalizer = OutletNormalizer()

    # 3. Fit normalizers
    print("\nStep 3: Fitting normalizers on data...")
    print("  3.1 Fitting spatial normalizer...")
    normalizer.fit(x_raw, y_raw, verbose=True)

    print("  3.2 Fitting outlet normalizer...")
    outlet_normalizer.fit(y_outlet_raw, verbose=True)

    # 4. Transform data
    print("\nStep 4: Applying transformations and normalization...")
    print("  4.1 Transforming spatial data...")
    x_norm, y_norm = normalizer.transform(x_raw, y_raw)

    print("  4.2 Transforming outlet data...")
    y_outlet_norm = outlet_normalizer.transform(y_outlet_raw)

    # Remove t=0 from outlet to match input (input removes t=0 in normalizer)
    y_outlet_norm = y_outlet_norm[:, 1:]  # (N, nt) → (N, nt-1)

    print(f"  Normalized input shape:  {tuple(x_norm.shape)}")
    print(f"  Normalized output shape: {tuple(y_norm.shape)}")
    print(f"  Normalized outlet shape: {tuple(y_outlet_norm.shape)}")

    # 5. Save results
    print("\nStep 5: Saving results...")

    # Process output_path
    output_path = PREPROC_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save normalized data
    torch.save({
        'x': x_norm,
        'y': y_norm,
        'y_outlet': y_outlet_norm,
        'xc': data['xc'],
        'yc': data['yc'],
        'time_keys': data['time_keys']
    }, output_path)

    print(f"  ✓ Saved normalized data: {output_path}")
    print(f"    File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save normalizers (always in src/preprocessing/)
    normalizer_path = PREPROC_DIR / f'normalizer_{output_mode}.pkl'
    outlet_normalizer_path = PREPROC_DIR / f'normalizer_out_{output_mode}.pkl'

    normalizer_cpu = normalizer.cpu()  # Move to CPU before saving
    outlet_normalizer_cpu = outlet_normalizer.cpu()

    with open(normalizer_path, 'wb') as f:
        pickle.dump(normalizer_cpu, f)

    with open(outlet_normalizer_path, 'wb') as f:
        pickle.dump(outlet_normalizer_cpu, f)

    print(f"  ✓ Saved spatial normalizer: {normalizer_path}")
    print(f"    File size: {normalizer_path.stat().st_size / 1024:.2f} KB")
    print(f"  ✓ Saved outlet normalizer: {outlet_normalizer_path}")
    print(f"    File size: {outlet_normalizer_path.stat().st_size / 1024:.2f} KB")

    result_paths = {
        'normalized_data': output_path,
        'normalizer': normalizer_path,
        'outlet_normalizer': outlet_normalizer_path
    }

    # 6. Optional: Statistical analysis
    if run_analysis:
        print(f"\nStep 6: Running statistical analysis (sample_size={analysis_sample_size})...")
        # Always save stats to src/preprocessing/normalization_stats/
        stats_dir = PREPROC_DIR / 'normalization_stats'
        stats_dir.mkdir(exist_ok=True)

        # Transform data for comparison (before normalization)
        x_transformed = normalizer.apply_input_transforms(x_raw)
        y_transformed = normalizer.apply_output_transform(y_raw, x_raw=x_raw)

        # Remove t=0 from input to match output
        x_transformed = x_transformed[:, :, :, :, 1:]

        # Sample for visualization
        n_samples = min(analysis_sample_size, x_raw.shape[0])
        sample_indices = torch.randperm(x_raw.shape[0])[:n_samples]

        x_trans_sample = x_transformed[sample_indices]
        x_norm_sample = x_norm[sample_indices]
        y_trans_sample = y_transformed[sample_indices]
        y_norm_sample = y_norm[sample_indices]

        # Compute statistics
        print("  Computing statistics...")
        input_stats = compute_normalization_stats(x_trans_sample, x_norm_sample, CHANNEL_CONFIG)
        output_stats = compute_normalization_stats(y_trans_sample, y_norm_sample, [('Uranium',)])

        # Save CSV
        save_stats_csv(input_stats, stats_dir / 'input_normalization_stats.csv')
        save_stats_csv(output_stats, stats_dir / 'output_normalization_stats.csv')

        # Print tables
        print_stats_table(input_stats, "Input Channel Statistics")
        print_stats_table(output_stats, "Output Channel Statistics")

        # Visualize distributions
        print("  Creating visualizations...")
        channel_names = [cfg[1] for cfg in CHANNEL_CONFIG]

        visualize_distributions(
            x_trans_sample, x_norm_sample,
            stats_dir / 'input_distributions.png',
            channel_names,
            dpi=150
        )

        visualize_distributions(
            y_trans_sample, y_norm_sample,
            stats_dir / 'output_distributions.png',
            ['Uranium'],
            dpi=150
        )

        # Summary plots
        visualize_normalized_summary(
            x_norm_sample,
            stats_dir / 'input_normalized_summary.png',
            channel_names,
            dpi=150
        )

        visualize_normalized_summary(
            y_norm_sample,
            stats_dir / 'output_normalized_summary.png',
            ['Uranium'],
            dpi=150
        )

        result_paths['stats_dir'] = stats_dir
        print(f"  ✓ Analysis saved to: {stats_dir}")

    print(f"\n{'='*70}")
    print("LR Normalization Complete!")
    print(f"{'='*70}\n")

    return result_paths


# ============================================================================
# HR Mode: Use Existing Normalizer
# ============================================================================

def normalize_hr(
    raw_data_path: str,
    normalizer_path: str,
    outlet_normalizer_path: str,
    output_path: str
):
    """
    HR 데이터에 대해 기존 LR normalizer 적용

    흐름:
    1. Raw HR 데이터 로드
    2. LR normalizers 로드 (spatial + outlet)
    3. Transform만 수행 (fit 안 함)
    4. 결과 저장 (normalized data만)

    Args:
        raw_data_path: Path to raw HR data
        normalizer_path: Path to LR spatial normalizer pickle
        outlet_normalizer_path: Path to LR outlet normalizer pickle
        output_path: Path to save normalized HR data

    Returns:
        dict with paths to generated files
    """
    # Set preprocessing directory base (always run from FNO root)
    PREPROC_DIR = Path('./src/preprocessing')

    # Process paths
    raw_data_path = PREPROC_DIR / raw_data_path
    normalizer_path = PREPROC_DIR / normalizer_path
    outlet_normalizer_path = PREPROC_DIR / outlet_normalizer_path

    print(f"\n{'='*70}")
    print(f"HR Mode: Applying Existing Normalizers")
    print(f"{'='*70}")
    print(f"HR data:           {raw_data_path}")
    print(f"Spatial normalizer: {normalizer_path}")
    print(f"Outlet normalizer:  {outlet_normalizer_path}")
    print(f"Output:            {output_path}")
    print(f"{'='*70}\n")

    # Validate paths
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw HR data not found: {raw_data_path}")

    if not normalizer_path.exists():
        raise FileNotFoundError(f"Spatial normalizer not found: {normalizer_path}")

    if not outlet_normalizer_path.exists():
        raise FileNotFoundError(f"Outlet normalizer not found: {outlet_normalizer_path}")

    # 1. Load HR data
    print("Step 1: Loading HR data...")
    data = torch.load(raw_data_path, map_location='cpu')

    required_keys = ['x', 'y', 'y_outlet', 'xc', 'yc', 'time_keys']
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in HR data: {missing}")

    x_raw = data['x']              # (N, 11, nx_hr, ny_hr, nt)
    y_raw = data['y']              # (N, 1, nx_hr, ny_hr, nt)
    y_outlet_raw = data['y_outlet']  # (N, nt)

    print(f"  HR input shape:   {tuple(x_raw.shape)}")
    print(f"  HR output shape:  {tuple(y_raw.shape)}")
    print(f"  HR outlet shape:  {tuple(y_outlet_raw.shape)}")
    print(f"  Samples: {x_raw.shape[0]}")
    print(f"  Spatial resolution: {x_raw.shape[2]} × {x_raw.shape[3]}")
    print(f"  Time steps: {x_raw.shape[4]}")

    # 2. Load LR normalizers
    print("\nStep 2: Loading LR normalizers...")
    with open(normalizer_path, 'rb') as f:
        normalizer = pickle.load(f)

    with open(outlet_normalizer_path, 'rb') as f:
        outlet_normalizer = pickle.load(f)

    if not isinstance(normalizer, ChannelNormalizer):
        raise TypeError(f"Expected ChannelNormalizer, got {type(normalizer).__name__}")

    if not isinstance(outlet_normalizer, OutletNormalizer):
        raise TypeError(f"Expected OutletNormalizer, got {type(outlet_normalizer).__name__}")

    print(f"  Spatial normalizer type: ChannelNormalizer")
    print(f"  Outlet normalizer type:  OutletNormalizer")
    print(f"  Output mode: {normalizer.output_mode}")
    print(f"  Input channels: {len(normalizer.input_config)}")

    # Validate channel count
    if x_raw.shape[1] != len(normalizer.input_config):
        raise ValueError(
            f"Channel count mismatch! "
            f"Normalizer expects {len(normalizer.input_config)} channels, "
            f"but HR data has {x_raw.shape[1]} channels"
        )

    # 3. Transform HR data (no fitting)
    print("\nStep 3: Applying transformations and normalization...")
    print("  (Using SAME statistics as LR training data)")

    print("  3.1 Transforming spatial data...")
    x_norm, y_norm = normalizer.transform(x_raw, y_raw)

    print("  3.2 Transforming outlet data...")
    y_outlet_norm = outlet_normalizer.transform(y_outlet_raw)
    y_outlet_norm = y_outlet_norm[:, 1:]  # Remove t=0

    print(f"  Normalized HR input shape:  {tuple(x_norm.shape)}")
    print(f"  Normalized HR output shape: {tuple(y_norm.shape)}")
    print(f"  Normalized HR outlet shape: {tuple(y_outlet_norm.shape)}")

    # Check normalization quality with detailed statistics
    print(f"\nStep 3.5: Normalization quality check...")

    # Transform data for comparison (before normalization)
    x_transformed = normalizer.apply_input_transforms(x_raw)
    y_transformed = normalizer.apply_output_transform(y_raw, x_raw=x_raw)

    # Remove t=0 from input to match output
    x_transformed = x_transformed[:, :, :, :, 1:]

    # Compute statistics
    print("  Computing channel-wise statistics...")
    input_stats = compute_normalization_stats(x_transformed, x_norm, CHANNEL_CONFIG)
    output_stats = compute_normalization_stats(y_transformed, y_norm, [('Uranium',)])

    # Print tables
    print_stats_table(input_stats, "HR Input Channel Statistics")
    print_stats_table(output_stats, "HR Output Channel Statistics")

    # 4. Save normalized HR data
    print("\nStep 4: Saving normalized HR data...")

    # Process output_path
    output_path = PREPROC_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'x': x_norm,
        'y': y_norm,
        'y_outlet': y_outlet_norm,
        'xc': data['xc'],
        'yc': data['yc'],
        'time_keys': data['time_keys']
    }, output_path)

    print(f"  ✓ Saved normalized HR data: {output_path}")
    print(f"    File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"\n{'='*70}")
    print("HR Normalization Complete!")
    print(f"{'='*70}")
    print(f"\n⚠️  IMPORTANT:")
    print(f"  - Use {normalizer_path} for spatial inverse transform")
    print(f"  - Use {outlet_normalizer_path} for outlet inverse transform")
    print(f"  - DO NOT create new normalizers for this HR data")
    print(f"  - The resolution difference is handled by FNO architecture")
    print(f"{'='*70}\n")

    return {
        'normalized_data': output_path,
        'normalizer': normalizer_path,
        'outlet_normalizer': outlet_normalizer_path
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Apply channel-wise normalization to preprocessing data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LR mode: Create new normalizer
  python preprocessing_normalize.py \\
      --mode lr \\
      --input merged_raw.pt \\
      --output-mode log \\
      --output merged_normalized.pt

  # HR mode: Use existing normalizer
  python preprocessing_normalize.py \\
      --mode hr \\
      --input merged_raw_hr.pt \\
      --normalizer normalizer_log.pkl \\
      --output merged_normalized_hr.pt
        """
    )

    # Common arguments
    parser.add_argument('--mode', choices=['lr', 'hr'], required=True,
                        help='Processing mode: lr (create normalizer) or hr (use existing)')
    parser.add_argument('--input', required=True,
                        help='Path to raw merged data')
    parser.add_argument('--output', required=True,
                        help='Path to save normalized data')

    # LR-specific arguments
    parser.add_argument('--output-mode', choices=['log', 'raw', 'delta'],
                        help='Output transformation mode (required for LR mode)')
    parser.add_argument('--no-analysis', action='store_true',
                        help='Skip statistical analysis (LR mode only)')
    parser.add_argument('--analysis-samples', type=int, default=3000,
                        help='Number of samples for analysis plots (default: 3000)')

    # HR-specific arguments
    parser.add_argument('--normalizer',
                        help='Path to LR spatial normalizer pickle (required for HR mode)')
    parser.add_argument('--outlet-normalizer',
                        help='Path to LR outlet normalizer pickle (required for HR mode)')

    args = parser.parse_args()

    try:
        if args.mode == 'lr':
            # LR mode: create new normalizer
            if not args.output_mode:
                parser.error("LR mode requires --output-mode")

            result = normalize_lr(
                raw_data_path=args.input,
                output_mode=args.output_mode,
                output_path=args.output,
                run_analysis=not args.no_analysis,
                analysis_sample_size=args.analysis_samples
            )

            print("Generated files:")
            print(f"  - Normalized data: {result['normalized_data']}")
            print(f"  - Spatial normalizer: {result['normalizer']}")
            print(f"  - Outlet normalizer: {result['outlet_normalizer']}")
            if 'stats_dir' in result:
                print(f"  - Statistics: {result['stats_dir']}/")

        else:  # hr mode
            # HR mode: use existing normalizer
            if not args.normalizer or not args.outlet_normalizer:
                parser.error("HR mode requires --normalizer and --outlet-normalizer")

            result = normalize_hr(
                raw_data_path=args.input,
                normalizer_path=args.normalizer,
                outlet_normalizer_path=args.outlet_normalizer,
                output_path=args.output
            )

            print("Generated files:")
            print(f"  - Normalized HR data: {result['normalized_data']}")
            print(f"  - Spatial normalizer used: {result['normalizer']}")
            print(f"  - Outlet normalizer used: {result['outlet_normalizer']}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
