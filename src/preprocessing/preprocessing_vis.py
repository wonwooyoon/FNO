import torch
import h5py
import numpy as np
from pathlib import Path


def save_samples_to_hdf5(data_path, output_path, n_samples=1):
    """
    Extract samples from PyTorch tensor and save to HDF5 for visualization

    Args:
        data_path: Path to .pt file containing 'x' and 'y'
        output_path: Path to output HDF5 file
        n_samples: Number of samples to extract
    """
    # Load data
    print(f"Loading data from {data_path.name}...")
    data = torch.load(data_path, map_location='cpu')
    x = data['x']  # (N, C, nx, ny, nt)
    y = data['y']  # (N, C, nx, ny, nt)

    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")

    N, C_in, nx, ny, nt = x.shape
    _, C_out, _, _, _ = y.shape

    # Limit samples to available data
    n_samples = min(n_samples, N)
    print(f"  Extracting {n_samples} samples...")

    # Create HDF5 file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Save each sample
        for i in range(n_samples):
            sample_group = f.create_group(f'sample_{i:03d}')

            # Save input channels
            input_group = sample_group.create_group('input')
            for ch in range(C_in):
                channel_data = x[i, ch].numpy()  # (nx, ny, nt)
                input_group.create_dataset(f'channel_{ch:02d}', data=channel_data)

            # Save output channels
            output_group = sample_group.create_group('output')
            for ch in range(C_out):
                channel_data = y[i, ch].numpy()  # (nx, ny, nt)
                output_group.create_dataset(f'channel_{ch:02d}', data=channel_data)

            print(f"    Saved sample {i}")

    print(f"Saved to {output_path}")


def main():
    PREPROC_DIR = Path('./src/preprocessing')

    # Number of samples to extract
    n_samples = 5

    # Process merged_normalized.pt
    data_path_lr = PREPROC_DIR / 'merged_normalized.pt'
    if data_path_lr.exists():
        output_path_lr = PREPROC_DIR / 'merged_normalized_vis.h5'
        print("\n" + "="*60)
        print("Processing Low Resolution Data")
        print("="*60)
        save_samples_to_hdf5(data_path_lr, output_path_lr, n_samples)
    else:
        print(f"File not found: {data_path_lr}")

    # Process merged_normalized_hr.pt
    data_path_hr = PREPROC_DIR / 'merged_normalized_hr.pt'
    if data_path_hr.exists():
        output_path_hr = PREPROC_DIR / 'merged_normalized_hr_vis.h5'
        print("\n" + "="*60)
        print("Processing High Resolution Data")
        print("="*60)
        save_samples_to_hdf5(data_path_hr, output_path_hr, n_samples)
    else:
        print(f"File not found: {data_path_hr}")

    print("\n" + "="*60)
    print("Visualization files created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
