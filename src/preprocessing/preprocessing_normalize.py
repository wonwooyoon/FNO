#!/usr/bin/env python3
"""
Data Normalization Main Script

채널별 transformation과 normalization을 적용하고,
LR/HR 모드에 따라 normalizer를 생성하거나 재사용합니다.
U (Uranium), Ca (Calcium), C (Carbonate) 3개 species를 동시에 처리합니다.

Input Files (must exist in ./src/preprocessing/):
    LR mode:
        - merged_raw_U.pt   : U (Uranium) raw data
        - merged_raw_Ca.pt  : Ca (Calcium) raw data
        - merged_raw_C.pt   : C (Carbonate) raw data
        - merged_raw_out.pt : Outlet data (for U only)

    HR mode:
        - merged_raw_U_hr.pt   : U (Uranium) HR raw data
        - merged_raw_Ca_hr.pt  : Ca (Calcium) HR raw data
        - merged_raw_C_hr.pt   : C (Carbonate) HR raw data
        - merged_raw_out_hr.pt : Outlet HR data (for U only)
        - normalizer_u_{mode}.pkl   : Pre-trained U normalizer
        - normalizer_ca_{mode}.pkl  : Pre-trained Ca normalizer
        - normalizer_c_{mode}.pkl   : Pre-trained C normalizer
        - normalizer_outlet_{mode}.pkl : Pre-trained outlet normalizer

Output Files (saved to ./src/preprocessing/):
    LR mode:
        - merged_normalized_U.pt   : Normalized U data
        - merged_normalized_Ca.pt  : Normalized Ca data
        - merged_normalized_C.pt   : Normalized C data
        - merged_normalized_out.pt : Normalized outlet data
        - normalizer_u_{mode}.pkl       : Fitted U normalizer
        - normalizer_ca_{mode}.pkl      : Fitted Ca normalizer
        - normalizer_c_{mode}.pkl       : Fitted C normalizer
        - normalizer_outlet_{mode}.pkl  : Fitted outlet normalizer
        - normalization_stats_{u,ca,c}/ : Statistical analysis & plots

    HR mode:
        - merged_normalized_U_hr.pt   : Normalized U HR data
        - merged_normalized_Ca_hr.pt  : Normalized Ca HR data
        - merged_normalized_C_hr.pt   : Normalized C HR data
        - merged_normalized_out_hr.pt : Normalized outlet HR data

Output Modes:
    - log   : Apply log10 transform to output (for concentration data)
    - raw   : No transform, use raw values (removes t=0)
    - delta : Compute delta from t=0, with source region masking

Usage Examples:
    # LR mode: Create new normalizers for U, Ca, C
    python preprocessing_normalize.py \\
        --mode lr \\
        --output-mode log

    # LR mode: Create normalizers with delta mode (includes source masking)
    python preprocessing_normalize.py \\
        --mode lr \\
        --output-mode delta

    # LR mode: Skip statistical analysis (faster)
    python preprocessing_normalize.py \\
        --mode lr \\
        --output-mode log \\
        --no-analysis

    # LR mode: Custom analysis sample size
    python preprocessing_normalize.py \\
        --mode lr \\
        --output-mode log \\
        --analysis-samples 5000

    # HR mode: Apply existing normalizers to HR data
    python preprocessing_normalize.py \\
        --mode hr \\
        --output-mode log

Notes:
    - Always run from FNO project root directory
    - LR mode: Fits normalizers on LR data and saves them
    - HR mode: Loads existing normalizers and applies to HR data
    - Output mode must match between LR and HR processing
    - Delta mode applies source region masking (14:18, 14:18)
    - Analysis plots use sampled data to reduce memory usage
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
        'mask_source': False
    }
}

# Data type configuration
DATA_TYPE_CONFIG = {
    'u': {
        'data_key': 'y_u',
        'species_name': 'Uranium',
        'full_name': 'Total UO2++ [M]'
    },
    'ca': {
        'data_key': 'y_ca',
        'species_name': 'Calcium',
        'full_name': 'Total Ca++ [M]'
    },
    'c': {
        'data_key': 'y_c',
        'species_name': 'Carbonate',
        'full_name': 'Total CO3-- [M]'
    }
}


# ============================================================================
# LR Mode: Create New Normalizer
# ============================================================================

def normalize_lr(
    output_mode: str,
    run_analysis: bool = True,
    analysis_sample_size: int = 3000
):
    """
    LR 데이터에 대해 새로운 normalizer 생성 및 적용

    U, Ca, C 3개 species에 대해 동시에 처리하고 각각의 normalizer를 저장

    흐름:
    1. Raw 데이터 로드 (merged_raw_U.pt, merged_raw_Ca.pt, merged_raw_C.pt, merged_raw_out.pt)
    2. 각 species에 대해 ChannelNormalizer 생성
    3. Fit (통계 계산)
    4. Transform (정규화 적용)
    5. 결과 저장 (normalized data + normalizer pickle for each species)
    6. (Optional) 통계 분석 및 시각화

    Args:
        output_mode: 'log', 'raw', 'delta' - 모든 species에 동일하게 적용
        run_analysis: Whether to run statistical analysis
        analysis_sample_size: Number of samples for analysis plots

    Returns:
        dict with paths to generated files
    """
    print(f"\n{'='*70}")
    print(f"LR Mode: Creating New Normalizers for U, Ca, C")
    print(f"{'='*70}")
    print(f"Output mode: {output_mode}")
    print(f"{'='*70}\n")

    # Validate output mode
    if output_mode not in OUTPUT_TRANSFORM_CONFIG:
        raise ValueError(f"Invalid output_mode: {output_mode}. Must be one of {list(OUTPUT_TRANSFORM_CONFIG.keys())}")

    # Set preprocessing directory base (always run from FNO root)
    PREPROC_DIR = Path('./src/preprocessing')
    RAW_DIR = PREPROC_DIR / 'data/raw/lr'

    # Define species to process
    species_list = ['u', 'ca', 'c']

    # Storage for results
    all_results = {}

    # 1. Load raw data for all species
    print("\nStep 1: Loading raw data for U, Ca, C...")

    # Load U, Ca, C data
    data_u = torch.load(RAW_DIR / 'merged_raw_U.pt', map_location='cpu')
    data_ca = torch.load(RAW_DIR / 'merged_raw_Ca.pt', map_location='cpu')
    data_c = torch.load(RAW_DIR / 'merged_raw_C.pt', map_location='cpu')

    # Load outlet data (only for U)
    data_out = torch.load(RAW_DIR / 'merged_raw_out.pt', map_location='cpu')

    # Extract input (same for all species)
    x_raw = data_u['x']  # (N, 11, nx, ny, nt)

    # Extract outputs for each species
    y_u_raw = data_u['y_u']    # (N, 1, nx, ny, nt)
    y_ca_raw = data_ca['y_ca']  # (N, 1, nx, ny, nt)
    y_c_raw = data_c['y_c']    # (N, 1, nx, ny, nt)
    y_outlet_raw = data_out['y_out']  # (N, nt)

    # Validate shapes
    assert y_u_raw.shape == y_ca_raw.shape == y_c_raw.shape, "Output shapes must match"
    assert x_raw.shape[0] == y_u_raw.shape[0], "Batch size mismatch"

    print(f"  ✓ Loaded U:  {tuple(y_u_raw.shape)}")
    print(f"  ✓ Loaded Ca: {tuple(y_ca_raw.shape)}")
    print(f"  ✓ Loaded C:  {tuple(y_c_raw.shape)}")
    print(f"  ✓ Loaded outlet: {tuple(y_outlet_raw.shape)}")
    print(f"  Input shape: {tuple(x_raw.shape)}")
    print(f"  Samples: {x_raw.shape[0]}")
    print(f"  Spatial: {x_raw.shape[2]} × {x_raw.shape[3]}")
    print(f"  Time steps: {x_raw.shape[4]}")

    # 2. Create normalizers for each species
    print("\nStep 2: Creating normalizers for U, Ca, C...")

    normalizers = {}
    for species in species_list:
        print(f"  2.{species_list.index(species)+1} Creating normalizer for {species.upper()}...")
        normalizers[species] = ChannelNormalizer(
            input_config=CHANNEL_CONFIG,
            output_mode=output_mode,
            output_config=OUTPUT_TRANSFORM_CONFIG[output_mode]
        )

    print("  2.4 Creating outlet normalizer...")
    outlet_normalizer = OutletNormalizer()

    # 3. Fit normalizers
    print("\nStep 3: Fitting normalizers on data...")

    y_raw_dict = {'u': y_u_raw, 'ca': y_ca_raw, 'c': y_c_raw}

    for species in species_list:
        print(f"  3.{species_list.index(species)+1} Fitting normalizer for {species.upper()}...")
        normalizers[species].fit(x_raw, y_raw_dict[species], verbose=True)

    print("  3.4 Fitting outlet normalizer...")
    outlet_normalizer.fit(y_outlet_raw, verbose=True)

    # 4. Transform data
    print("\nStep 4: Applying transformations and normalization...")

    # Transform input once (same for all species)
    print("  4.1 Transforming input data...")
    x_norm, _ = normalizers['u'].transform(x_raw, y_u_raw)  # Use U normalizer for input

    # Transform outputs for each species
    y_norm_dict = {}
    for species in species_list:
        print(f"  4.{species_list.index(species)+2} Transforming output for {species.upper()}...")
        _, y_norm_dict[species] = normalizers[species].transform(x_raw, y_raw_dict[species])

    print("  4.5 Transforming outlet data...")
    y_outlet_norm = outlet_normalizer.transform(y_outlet_raw)
    y_outlet_norm = y_outlet_norm[:, 1:]  # Remove t=0 to match input (N, nt) → (N, nt-1)

    print(f"\n  ✓ Normalized input shape:  {tuple(x_norm.shape)}")
    print(f"  ✓ Normalized U shape:      {tuple(y_norm_dict['u'].shape)}")
    print(f"  ✓ Normalized Ca shape:     {tuple(y_norm_dict['ca'].shape)}")
    print(f"  ✓ Normalized C shape:      {tuple(y_norm_dict['c'].shape)}")
    print(f"  ✓ Normalized outlet shape: {tuple(y_outlet_norm.shape)}")

    # 5. Save results for each species
    print("\nStep 5: Saving results for U, Ca, C...")

    # Define output directories
    DATA_DIR = PREPROC_DIR / 'data/normalized/lr' / output_mode
    NORMALIZER_DIR = PREPROC_DIR / 'normalizers/lr' / output_mode
    STATS_DIR = PREPROC_DIR / 'analysis/normalization_stats/lr' / output_mode

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Extract initial values if delta mode (for visualization reconstruction)
    y_initial_dict = {}
    if output_mode == 'delta':
        y_initial_dict['u'] = y_u_raw[:, :, :, :, 0:1]    # (N, 1, nx, ny, 1)
        y_initial_dict['ca'] = y_ca_raw[:, :, :, :, 0:1]
        y_initial_dict['c'] = y_c_raw[:, :, :, :, 0:1]
        print("  Delta mode: Saving initial values (t=0) for visualization reconstruction")

    # Save normalized data for each species
    for species in species_list:
        species_config = DATA_TYPE_CONFIG[species]
        data_key = species_config['data_key']
        species_name = species_config['species_name']

        output_path = DATA_DIR / f'merged_normalized_{species.upper()}.pt'

        save_dict = {
            'x': x_norm,
            data_key: y_norm_dict[species],
            'xc': data_u['xc'],
            'yc': data_u['yc'],
            'time_keys': data_u['time_keys']
        }

        # Add initial values for delta mode
        if output_mode == 'delta' and species in y_initial_dict:
            save_dict['y_initial'] = y_initial_dict[species]

        torch.save(save_dict, output_path)
        print(f"  ✓ Saved {species_name:10s}: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

        all_results[f'normalized_data_{species}'] = output_path

    # Save outlet data separately
    outlet_output_path = DATA_DIR / 'merged_normalized_out.pt'
    torch.save({
        'x': x_norm,
        'y_outlet': y_outlet_norm,
        'xc': data_u['xc'],
        'yc': data_u['yc'],
        'time_keys': data_u['time_keys']
    }, outlet_output_path)
    print(f"  ✓ Saved Outlet    : {outlet_output_path.name} ({outlet_output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    all_results['normalized_data_out'] = outlet_output_path

    # Save normalizers for each species
    print("\nStep 6: Saving normalizers...")
    for species in species_list:
        species_name = DATA_TYPE_CONFIG[species]['species_name']
        normalizer_path = NORMALIZER_DIR / f'normalizer_{species}_{output_mode}.pkl'

        normalizer_cpu = normalizers[species].cpu()
        with open(normalizer_path, 'wb') as f:
            pickle.dump(normalizer_cpu, f)

        print(f"  ✓ Saved {species_name:10s} normalizer: {normalizer_path.name} ({normalizer_path.stat().st_size / 1024:.2f} KB)")
        all_results[f'normalizer_{species}'] = normalizer_path

    # Save outlet normalizer
    outlet_normalizer_path = NORMALIZER_DIR / f'normalizer_out_{output_mode}.pkl'
    outlet_normalizer_cpu = outlet_normalizer.cpu()
    with open(outlet_normalizer_path, 'wb') as f:
        pickle.dump(outlet_normalizer_cpu, f)

    print(f"  ✓ Saved Outlet     normalizer: {outlet_normalizer_path.name} ({outlet_normalizer_path.stat().st_size / 1024:.2f} KB)")
    all_results['normalizer_out'] = outlet_normalizer_path

    # 7. Optional: Statistical analysis for each species
    if run_analysis:
        print(f"\nStep 7: Running statistical analysis (sample_size={analysis_sample_size})...")

        # Sample indices (same for all species)
        n_samples = min(analysis_sample_size, x_raw.shape[0])
        sample_indices = torch.randperm(x_raw.shape[0])[:n_samples]

        x_norm_sample = x_norm[sample_indices]
        channel_names = [cfg[1] for cfg in CHANNEL_CONFIG]

        for species in species_list:
            species_config = DATA_TYPE_CONFIG[species]
            species_name = species_config['species_name']

            print(f"\n  Processing {species_name} statistics...")
            stats_dir = STATS_DIR / species
            stats_dir.mkdir(parents=True, exist_ok=True)

            # Transform data for comparison (before normalization)
            x_transformed = normalizers[species].apply_input_transforms(x_raw)
            y_transformed = normalizers[species].apply_output_transform(y_raw_dict[species], x_raw=x_raw)

            # Remove t=0 from input to match output
            x_transformed = x_transformed[:, :, :, :, 1:]

            # Sample
            x_trans_sample = x_transformed[sample_indices]
            y_trans_sample = y_transformed[sample_indices]
            y_norm_sample = y_norm_dict[species][sample_indices]

            # Compute statistics
            input_stats = compute_normalization_stats(x_trans_sample, x_norm_sample, CHANNEL_CONFIG)
            output_stats = compute_normalization_stats(y_trans_sample, y_norm_sample, [(species_name,)])

            # Save CSV
            save_stats_csv(input_stats, stats_dir / 'input_normalization_stats.csv')
            save_stats_csv(output_stats, stats_dir / 'output_normalization_stats.csv')

            # Print tables
            print_stats_table(input_stats, f"Input Channel Statistics ({species_name})")
            print_stats_table(output_stats, f"Output Channel Statistics ({species_name})")

            # Visualize distributions
            visualize_distributions(
                x_trans_sample, x_norm_sample,
                stats_dir / 'input_distributions.png',
                channel_names,
                dpi=150
            )

            visualize_distributions(
                y_trans_sample, y_norm_sample,
                stats_dir / 'output_distributions.png',
                [species_name],
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
                [species_name],
                dpi=150
            )

            all_results[f'stats_dir_{species}'] = stats_dir
            print(f"  ✓ {species_name} analysis saved to: {stats_dir}")

    print(f"\n{'='*70}")
    print("LR Normalization Complete!")
    print(f"{'='*70}\n")

    return all_results


# ============================================================================
# HR Mode: Use Existing Normalizer
# ============================================================================

def normalize_hr(output_mode: str):
    """
    HR 데이터에 대해 기존 LR normalizer 적용

    U, Ca, C 3개 species에 대해 동시에 처리

    흐름:
    1. Raw HR 데이터 로드 (merged_raw_U_hr.pt, merged_raw_Ca_hr.pt, merged_raw_C_hr.pt, merged_raw_out_hr.pt)
    2. LR normalizers 로드 (각 species 별)
    3. Transform만 수행 (fit 안 함)
    4. 결과 저장 (normalized HR data for each species)

    Args:
        output_mode: 'log', 'raw', 'delta' - LR normalization과 동일한 mode 사용

    Returns:
        dict with paths to generated files
    """
    # Set preprocessing directory base (always run from FNO root)
    PREPROC_DIR = Path('./src/preprocessing')

    # Define species to process
    species_list = ['u', 'ca', 'c']

    # Storage for results
    all_results = {}

    print(f"\n{'='*70}")
    print(f"HR Mode: Applying Existing Normalizers for U, Ca, C")
    print(f"{'='*70}")
    print(f"Output mode: {output_mode}")
    print(f"{'='*70}\n")

    # 1. Load HR data for all species
    print("\nStep 1: Loading HR data for U, Ca, C...")

    # Define raw data directory for HR
    RAW_DIR_HR = PREPROC_DIR / 'data/raw/hr'

    # Load U, Ca, C data
    data_u = torch.load(RAW_DIR_HR / 'merged_raw_U_hr.pt', map_location='cpu')
    data_ca = torch.load(RAW_DIR_HR / 'merged_raw_Ca_hr.pt', map_location='cpu')
    data_c = torch.load(RAW_DIR_HR / 'merged_raw_C_hr.pt', map_location='cpu')

    # Load outlet data (only for U)
    data_out = torch.load(RAW_DIR_HR / 'merged_raw_out_hr.pt', map_location='cpu')

    # Extract input (same for all species)
    x_raw = data_u['x']  # (N, 11, nx_hr, ny_hr, nt)

    # Extract outputs for each species
    y_u_raw = data_u['y_u']    # (N, 1, nx_hr, ny_hr, nt)
    y_ca_raw = data_ca['y_ca']  # (N, 1, nx_hr, ny_hr, nt)
    y_c_raw = data_c['y_c']    # (N, 1, nx_hr, ny_hr, nt)
    y_outlet_raw = data_out['y_out']  # (N, nt)

    # Validate shapes
    assert y_u_raw.shape == y_ca_raw.shape == y_c_raw.shape, "Output shapes must match"
    assert x_raw.shape[0] == y_u_raw.shape[0], "Batch size mismatch"

    print(f"  ✓ Loaded U:  {tuple(y_u_raw.shape)}")
    print(f"  ✓ Loaded Ca: {tuple(y_ca_raw.shape)}")
    print(f"  ✓ Loaded C:  {tuple(y_c_raw.shape)}")
    print(f"  ✓ Loaded outlet: {tuple(y_outlet_raw.shape)}")
    print(f"  HR input shape: {tuple(x_raw.shape)}")
    print(f"  Samples: {x_raw.shape[0]}")
    print(f"  Spatial resolution: {x_raw.shape[2]} × {x_raw.shape[3]}")
    print(f"  Time steps: {x_raw.shape[4]}")

    # 2. Load LR normalizers for each species
    print("\nStep 2: Loading LR normalizers...")
    NORMALIZER_DIR_LR = PREPROC_DIR / 'normalizers/lr' / output_mode
    normalizers = {}
    for species in species_list:
        normalizer_path = NORMALIZER_DIR_LR / f'normalizer_{species}_{output_mode}.pkl'
        if not normalizer_path.exists():
            raise FileNotFoundError(f"Normalizer not found: {normalizer_path}")

        with open(normalizer_path, 'rb') as f:
            normalizers[species] = pickle.load(f)

        if not isinstance(normalizers[species], ChannelNormalizer):
            raise TypeError(f"Expected ChannelNormalizer, got {type(normalizers[species]).__name__}")

        print(f"  ✓ Loaded {species.upper()} normalizer: {normalizer_path.name}")

    # Load outlet normalizer
    outlet_normalizer_path = NORMALIZER_DIR_LR / f'normalizer_out_{output_mode}.pkl'
    if not outlet_normalizer_path.exists():
        raise FileNotFoundError(f"Outlet normalizer not found: {outlet_normalizer_path}")

    with open(outlet_normalizer_path, 'rb') as f:
        outlet_normalizer = pickle.load(f)

    if not isinstance(outlet_normalizer, OutletNormalizer):
        raise TypeError(f"Expected OutletNormalizer, got {type(outlet_normalizer).__name__}")

    print(f"  ✓ Loaded outlet normalizer: {outlet_normalizer_path.name}")

    # Validate channel count
    if x_raw.shape[1] != len(normalizers['u'].input_config):
        raise ValueError(
            f"Channel count mismatch! "
            f"Normalizer expects {len(normalizers['u'].input_config)} channels, "
            f"but HR data has {x_raw.shape[1]} channels"
        )

    # 3. Transform HR data (no fitting)
    print("\nStep 3: Applying transformations and normalization...")
    print("  (Using SAME statistics as LR training data)")

    # Transform input once (same for all species)
    print("  3.1 Transforming input data...")
    x_norm, _ = normalizers['u'].transform(x_raw, y_u_raw)  # Use U normalizer for input

    # Transform outputs for each species
    y_raw_dict = {'u': y_u_raw, 'ca': y_ca_raw, 'c': y_c_raw}
    y_norm_dict = {}
    for species in species_list:
        print(f"  3.{species_list.index(species)+2} Transforming output for {species.upper()}...")
        _, y_norm_dict[species] = normalizers[species].transform(x_raw, y_raw_dict[species])

    print("  3.5 Transforming outlet data...")
    y_outlet_norm = outlet_normalizer.transform(y_outlet_raw)
    y_outlet_norm = y_outlet_norm[:, 1:]  # Remove t=0

    print(f"\n  ✓ Normalized HR input shape:  {tuple(x_norm.shape)}")
    print(f"  ✓ Normalized U shape:         {tuple(y_norm_dict['u'].shape)}")
    print(f"  ✓ Normalized Ca shape:        {tuple(y_norm_dict['ca'].shape)}")
    print(f"  ✓ Normalized C shape:         {tuple(y_norm_dict['c'].shape)}")
    print(f"  ✓ Normalized outlet shape:    {tuple(y_outlet_norm.shape)}")

    # 4. Save normalized HR data for each species
    print("\nStep 4: Saving normalized HR data for U, Ca, C...")

    # Define output directory for HR
    DATA_DIR_HR = PREPROC_DIR / 'data/normalized/hr' / output_mode
    DATA_DIR_HR.mkdir(parents=True, exist_ok=True)

    # Extract initial values if delta mode (for visualization reconstruction)
    y_initial_dict = {}
    if output_mode == 'delta':
        y_initial_dict['u'] = y_u_raw[:, :, :, :, 0:1]    # (N, 1, nx_hr, ny_hr, 1)
        y_initial_dict['ca'] = y_ca_raw[:, :, :, :, 0:1]
        y_initial_dict['c'] = y_c_raw[:, :, :, :, 0:1]
        print("  Delta mode: Saving initial values (t=0) for visualization reconstruction")

    # Save normalized data for each species
    for species in species_list:
        species_config = DATA_TYPE_CONFIG[species]
        data_key = species_config['data_key']
        species_name = species_config['species_name']

        output_path = DATA_DIR_HR / f'merged_normalized_{species.upper()}_hr.pt'

        save_dict = {
            'x': x_norm,
            data_key: y_norm_dict[species],
            'xc': data_u['xc'],
            'yc': data_u['yc'],
            'time_keys': data_u['time_keys']
        }

        # Add initial values for delta mode
        if output_mode == 'delta' and species in y_initial_dict:
            save_dict['y_initial'] = y_initial_dict[species]

        torch.save(save_dict, output_path)
        print(f"  ✓ Saved {species_name:10s}: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

        all_results[f'normalized_data_{species}'] = output_path

    # Save outlet data separately
    outlet_output_path = DATA_DIR_HR / 'merged_normalized_out_hr.pt'
    torch.save({
        'x': x_norm,
        'y_outlet': y_outlet_norm,
        'xc': data_u['xc'],
        'yc': data_u['yc'],
        'time_keys': data_u['time_keys']
    }, outlet_output_path)
    print(f"  ✓ Saved Outlet    : {outlet_output_path.name} ({outlet_output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    all_results['normalized_data_out'] = outlet_output_path

    print(f"\n{'='*70}")
    print("HR Normalization Complete!")
    print(f"{'='*70}")
    print(f"\n⚠️  IMPORTANT:")
    print(f"  - DO NOT create new normalizers for this HR data")
    print(f"  - The resolution difference is handled by FNO architecture")
    print(f"  - Use LR normalizers for inverse transform:")
    for species in species_list:
        print(f"    * {species.upper()}: normalizer_{species}_{output_mode}.pkl")
    print(f"    * Outlet: normalizer_out_{output_mode}.pkl")
    print(f"{'='*70}\n")

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Apply channel-wise normalization to preprocessing data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LR mode: Create normalizers for U, Ca, C
  python preprocessing_normalize.py \\
      --mode lr \\
      --output-mode log

  # HR mode: Use existing normalizers
  python preprocessing_normalize.py \\
      --mode hr \\
      --output-mode log
        """
    )

    # Common arguments
    parser.add_argument('--mode', choices=['lr', 'hr'], required=True,
                        help='Processing mode: lr (create normalizer) or hr (use existing)')
    parser.add_argument('--output-mode', choices=['log', 'raw', 'delta'], required=True,
                        help='Output transformation mode for all species')

    # LR-specific arguments
    parser.add_argument('--no-analysis', action='store_true',
                        help='Skip statistical analysis (LR mode only)')
    parser.add_argument('--analysis-samples', type=int, default=3000,
                        help='Number of samples for analysis plots (default: 3000)')

    args = parser.parse_args()

    try:
        if args.mode == 'lr':
            # LR mode: create new normalizers for U, Ca, C
            result = normalize_lr(
                output_mode=args.output_mode,
                run_analysis=not args.no_analysis,
                analysis_sample_size=args.analysis_samples
            )

            print("\n" + "="*70)
            print("Generated files:")
            print("="*70)
            print("\nNormalized data:")
            print(f"  - U:  {result['normalized_data_u']}")
            print(f"  - Ca: {result['normalized_data_ca']}")
            print(f"  - C:  {result['normalized_data_c']}")
            print(f"  - Outlet: {result['normalized_data_out']}")

            print("\nNormalizers:")
            print(f"  - U:  {result['normalizer_u']}")
            print(f"  - Ca: {result['normalizer_ca']}")
            print(f"  - C:  {result['normalizer_c']}")
            print(f"  - Outlet: {result['normalizer_out']}")

            if 'stats_dir_u' in result:
                print("\nStatistics:")
                print(f"  - U:  {result['stats_dir_u']}/")
                print(f"  - Ca: {result['stats_dir_ca']}/")
                print(f"  - C:  {result['stats_dir_c']}/")

        else:  # hr mode
            # HR mode: use existing normalizers
            result = normalize_hr(output_mode=args.output_mode)

            print("\n" + "="*70)
            print("Generated files:")
            print("="*70)
            print("\nNormalized HR data:")
            print(f"  - U:  {result['normalized_data_u']}")
            print(f"  - Ca: {result['normalized_data_ca']}")
            print(f"  - C:  {result['normalized_data_c']}")
            print(f"  - Outlet: {result['normalized_data_out']}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
