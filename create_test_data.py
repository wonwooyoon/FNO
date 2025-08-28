#!/usr/bin/env python3
"""
Create test dataset script
"""

import torch
import numpy as np
from pathlib import Path

def main():
    # Load original data
    merged_path = './src/preprocessing/merged.pt'
    print("Creating test dataset...")
    
    if Path(merged_path).exists():
        print("Loading original merged.pt...")
        data = torch.load(merged_path, weights_only=False)
        
        input_tensor = data['x']        # (460, 7, 64, 32, 20)
        output_tensor = data['y']       # (460, 1, 64, 32, 20)  
        meta_tensor = data['meta']      # (460,)
        
        print(f"Original data shapes:")
        print(f"  Input tensor: {input_tensor.shape}")
        print(f"  Output tensor: {output_tensor.shape}")
        print(f"  Meta tensor: {meta_tensor.shape}")
        
        # Create small test dataset - use first 32 samples
        n_samples = 32
        
        print(f"\nCreating test dataset with {n_samples} samples...")
        
        # Select subset of data
        test_input = input_tensor[:n_samples]
        test_output = output_tensor[:n_samples] 
        test_meta = meta_tensor[:n_samples]
        
        # Expand meta tensor to 2D for consistency with the models
        if len(test_meta.shape) == 1:
            test_meta = test_meta.unsqueeze(1)  # (32, 1)
        
        print(f"Test data shapes:")
        print(f"  Test input: {test_input.shape}")
        print(f"  Test output: {test_output.shape}")
        print(f"  Test meta: {test_meta.shape}")
        
        # Save test dataset with original key names
        test_data = {
            'x': test_input,
            'y': test_output,
            'meta': test_meta,
            'xc': data['xc'],
            'yc': data['yc'],
            'time_keys': data['time_keys']
        }
        
        test_path = './merged_test.pt'
        torch.save(test_data, test_path)
        print(f"Test dataset saved to: {test_path}")
        
    else:
        print(f"Original data file not found: {merged_path}")

if __name__ == "__main__":
    main()