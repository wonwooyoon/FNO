#!/usr/bin/env python3
"""
Test data exploration and small dataset creation script
"""

import torch
import numpy as np
from pathlib import Path

def main():
    # Load original data
    merged_path = './src/preprocessing/merged.pt'
    if Path(merged_path).exists():
        print("Loading original merged.pt...")
        data = torch.load(merged_path, weights_only=False)
        
        input_tensor = data['input_tensor']
        output_tensor = data['output_tensor']
        meta_tensor = data['meta_tensor']
        
        print(f"Original data shapes:")
        print(f"  Input tensor: {input_tensor.shape}")
        print(f"  Output tensor: {output_tensor.shape}")
        print(f"  Meta tensor: {meta_tensor.shape}")
        
        # Create small test dataset
        n_samples = min(32, input_tensor.shape[0])  # Use at most 32 samples
        
        print(f"\nCreating test dataset with {n_samples} samples...")
        
        # Select subset of data
        test_input = input_tensor[:n_samples]
        test_output = output_tensor[:n_samples]
        test_meta = meta_tensor[:n_samples]
        
        print(f"Test data shapes:")
        print(f"  Test input: {test_input.shape}")
        print(f"  Test output: {test_output.shape}")
        print(f"  Test meta: {test_meta.shape}")
        
        # Save test dataset
        test_data = {
            'input_tensor': test_input,
            'output_tensor': test_output,
            'meta_tensor': test_meta
        }
        
        test_path = './merged_test.pt'
        torch.save(test_data, test_path)
        print(f"Test dataset saved to: {test_path}")
        
    else:
        print(f"Original data file not found: {merged_path}")
        print("Creating synthetic test data...")
        
        # Create synthetic test data
        n_samples = 32
        n_channels = 7
        nx, ny, nt = 32, 32, 20
        
        # Synthetic input data (N, channels, nx, ny, nt)
        test_input = torch.randn(n_samples, n_channels, nx, ny, nt)
        
        # Synthetic output data (N, 1, nx, ny, nt)
        test_output = torch.randn(n_samples, 1, nx, ny, nt)
        
        # Synthetic meta data (N, 2) - permeability and porosity values
        test_meta = torch.randn(n_samples, 2)
        
        test_data = {
            'input_tensor': test_input,
            'output_tensor': test_output,
            'meta_tensor': test_meta
        }
        
        test_path = './merged_test.pt'
        torch.save(test_data, test_path)
        
        print(f"Synthetic test data shapes:")
        print(f"  Input tensor: {test_input.shape}")
        print(f"  Output tensor: {test_output.shape}")
        print(f"  Meta tensor: {test_meta.shape}")
        print(f"Synthetic test dataset saved to: {test_path}")

if __name__ == "__main__":
    main()