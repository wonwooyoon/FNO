#!/usr/bin/env python3
"""
Check data structure script
"""

import torch
from pathlib import Path

def main():
    merged_path = './src/preprocessing/merged.pt'
    if Path(merged_path).exists():
        print("Loading merged.pt to check structure...")
        data = torch.load(merged_path, weights_only=False)
        
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} - {type(value)}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")

if __name__ == "__main__":
    main()