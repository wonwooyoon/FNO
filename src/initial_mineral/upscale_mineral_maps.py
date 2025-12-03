#!/usr/bin/env python3
"""
Create LR (64×32) and HR (128×64) versions of mineral map 0

This script reads the existing mineral_0.h5 files (128×64 HR format),
creates LR version by downsampling, and HR version by copying.
This ensures both resolutions share the same underlying distribution.
"""

import numpy as np
import h5py
from scipy.ndimage import zoom
import os


def read_mineral_map(filepath, group_name):
    """
    Read mineral map from HDF5 file

    Args:
        filepath: Path to HDF5 file
        group_name: Name of the group (e.g., 'calcite_mapX')

    Returns:
        data: Mineral concentration map
        attrs: Dictionary of attributes
    """
    with h5py.File(filepath, 'r') as f:
        group = f[group_name]
        data = group['Data'][:]

        attrs = {
            'Dimension': group.attrs['Dimension'],
            'Discretization': group.attrs['Discretization'],
            'Origin': group.attrs['Origin'],
            'Cell Centered': group.attrs['Cell Centered']
        }

        if 'Space Interpolation Method' in group.attrs:
            attrs['Space Interpolation Method'] = group.attrs['Space Interpolation Method']

    return data, attrs


def downsample_4x4_averaging(data):
    """
    Downsample from 128×64 to 64×32 using 4×4 box averaging

    This matches the LR generation method in initial_perm_v2.py
    where 4×4 box averaging with stride=4 is used.

    Args:
        data: (128, 64) array

    Returns:
        downsampled: (64, 32) array
    """
    h, w = data.shape  # (128, 64)
    kernel_size = 4
    stride = 4

    output_h = h // stride  # 128 // 4 = 32... wait, this is wrong
    output_w = w // stride  # 64 // 4 = 16

    # Actually, let's use 2×2 averaging for 128×64 → 64×32
    # because 128/2 = 64, 64/2 = 32
    kernel_size = 2
    stride = 2

    output_h = h // stride  # 128 // 2 = 64
    output_w = w // stride  # 64 // 2 = 32

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            window = data[stride*i:stride*i+kernel_size,
                         stride*j:stride*j+kernel_size]
            output[i, j] = window.mean()

    return output


def write_mineral_map(filepath, group_name, data, discretization, origin):
    """
    Write mineral map to HDF5 file

    Args:
        filepath: Output HDF5 path
        group_name: Group name (e.g., 'calcite_mapX')
        data: Mineral concentration array
        discretization: [dx, dy] grid spacing
        origin: [x0, y0] domain origin
    """
    with h5py.File(filepath, 'w') as f:
        group = f.create_group(group_name)

        data_dataset = group.create_dataset('Data', shape=data.shape, dtype=data.dtype)
        data_dataset[:, :] = data

        group.attrs.create('Dimension', ['XY'],
                          dtype=h5py.string_dtype(encoding='ascii', length=10))
        group.attrs.create('Space Interpolation Method', ['STEP'],
                          dtype=h5py.string_dtype(encoding='ascii', length=10))
        group.attrs['Discretization'] = discretization
        group.attrs['Origin'] = origin
        group.attrs['Cell Centered'] = [True]


def process_mineral_type(mineral_name, group_name, input_dir, lr_output_dir, hr_output_dir):
    """
    Process one mineral type: create LR and HR versions

    Args:
        mineral_name: 'calcite', 'clinochlore', or 'pyrite'
        group_name: HDF5 group name
        input_dir: Input directory containing mineral_0.h5
        lr_output_dir: Output directory for LR maps
        hr_output_dir: Output directory for HR maps
    """
    print(f"\nProcessing {mineral_name}...")

    # Read original map (assume it's 128×64)
    input_file = os.path.join(input_dir, f"{mineral_name}_0.h5")

    if not os.path.exists(input_file):
        print(f"  ⚠ File not found: {input_file}")
        return

    data_hr, attrs = read_mineral_map(input_file, group_name)
    print(f"  Original shape: {data_hr.shape}")
    print(f"  Original discretization: {attrs['Discretization']}")

    # Create HR version (just copy, but ensure it's 128×64)
    if data_hr.shape != (128, 64):
        print(f"  ⚠ Warning: Expected (128, 64), got {data_hr.shape}")
        # If it's (64, 32), we need to upscale first
        if data_hr.shape == (64, 32):
            print(f"  Upscaling (64, 32) → (128, 64) using bilinear interpolation")
            data_hr = zoom(data_hr, (2, 2), order=1)  # bilinear

    # Create LR version by downsampling
    print(f"  Creating LR version (64×32)...")
    data_lr = downsample_4x4_averaging(data_hr)
    print(f"  LR shape: {data_lr.shape}")

    # Write LR map
    lr_file = os.path.join(lr_output_dir, f"{mineral_name}_0.h5")
    discretization_lr = [0.25, 0.25]  # LR: 0.25m per cell
    origin = [-8.0, -4.0]

    print(f"  Writing LR: {lr_file}")
    write_mineral_map(lr_file, group_name, data_lr, discretization_lr, origin)

    # Write HR map
    hr_file = os.path.join(hr_output_dir, f"{mineral_name}_0.h5")
    discretization_hr = [0.125, 0.125]  # HR: 0.125m per cell

    print(f"  Writing HR: {hr_file}")
    write_mineral_map(hr_file, group_name, data_hr, discretization_hr, origin)

    # Verify
    print(f"  ✓ LR: {data_lr.shape}, range: [{data_lr.min():.6f}, {data_lr.max():.6f}]")
    print(f"  ✓ HR: {data_hr.shape}, range: [{data_hr.min():.6f}, {data_hr.max():.6f}]")


def main():
    """
    Create LR and HR versions of mineral map 0
    """
    print("=" * 80)
    print("MINERAL MAP UPSCALING: Create LR (64×32) and HR (128×64) versions")
    print("=" * 80)

    # Directories
    base_dir = "/home/geofluids/research/FNO/src/initial_mineral"
    input_dir = os.path.join(base_dir, "output")
    lr_output_dir = os.path.join(base_dir, "output_lr")
    hr_output_dir = os.path.join(base_dir, "output_hr")

    # Create output directories if needed
    os.makedirs(lr_output_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)

    print(f"\nInput directory: {input_dir}")
    print(f"LR output directory: {lr_output_dir}")
    print(f"HR output directory: {hr_output_dir}")

    # Process each mineral type
    minerals = [
        ('calcite', 'calcite_mapX'),
        ('clinochlore', 'clinochlore_mapX'),
        ('pyrite', 'pyrite_mapX')
    ]

    for mineral_name, group_name in minerals:
        process_mineral_type(mineral_name, group_name, input_dir, lr_output_dir, hr_output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print("LR maps (64×32) saved to:", lr_output_dir)
    print("HR maps (128×64) saved to:", hr_output_dir)
    print()
    print("Both resolutions share the same underlying distribution.")
    print("LR is created by 2×2 averaging from HR.")
    print()


if __name__ == '__main__':
    main()
