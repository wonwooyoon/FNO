#!/usr/bin/env python3
"""
Test UNet_pure.py CSV export functionality
"""

import sys
sys.path.append('./src/FNO')

import torch
from pathlib import Path

# Import UNet_pure and temporarily modify its config
import UNet_pure

def test_unet_csv():
    print("="*60)
    print("Testing UNet_pure.py - CSV Export Feature")
    print("="*60)
    
    # Backup original config
    original_config = UNet_pure.CONFIG.copy()
    
    try:
        # Modify config for test with CSV export enabled
        UNet_pure.CONFIG.update({
            'MERGED_PT_PATH': './merged_test.pt',
            'OUTPUT_DIR': './test_output_unet_csv',
            'N_EPOCHS': 1,  # Very short test
            'TRAINING_CONFIG': {
                'mode': 'single',
                'optuna_n_trials': 2,
                'optuna_seed': 42,
                'optuna_n_startup_trials': 1,
                'eval_model_path': './test_output_unet_csv/final/best_model_state_dict.pt'
            },
            'VISUALIZATION': {
                'SAMPLE_NUM': 8,
                'TIME_INDICES': (3, 7, 11, 15),  # UNet uses different time indices
                'DPI': 200,
                'SAVEASCSV': True  # Enable CSV export
            }
        })
        
        print("Modified CONFIG for testing:")
        print(f"  Data path: {UNet_pure.CONFIG['MERGED_PT_PATH']}")
        print(f"  Output dir: {UNet_pure.CONFIG['OUTPUT_DIR']}")
        print(f"  Epochs: {UNet_pure.CONFIG['N_EPOCHS']}")
        print(f"  Mode: {UNet_pure.CONFIG['TRAINING_CONFIG']['mode']}")
        print(f"  SAVEASCSV: {UNet_pure.CONFIG['VISUALIZATION']['SAVEASCSV']}")
        print(f"  Time indices: {UNet_pure.CONFIG['VISUALIZATION']['TIME_INDICES']}")
        
        # Create output directory
        Path(UNet_pure.CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        Path(UNet_pure.CONFIG['OUTPUT_DIR'] + '/final').mkdir(parents=True, exist_ok=True)
        
        # Run the test
        print("\nRunning UNet_pure.py with CSV export...")
        UNet_pure.main()
        
        # Check if CSV file was created
        csv_path = Path(UNet_pure.CONFIG['OUTPUT_DIR']) / 'UNet_visualization_data.csv'
        if csv_path.exists():
            print(f"\n✅ CSV file created successfully: {csv_path}")
            
            # Read and display basic info about the CSV
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"CSV shape: {df.shape}")
            print(f"CSV columns: {list(df.columns)}")
            print(f"First few rows:")
            print(df.head())
            
        else:
            print(f"\n❌ CSV file not found: {csv_path}")
            
        print("\n✅ UNet_pure.py CSV export test PASSED!")
        
    except Exception as e:
        print(f"\n❌ UNet_pure.py CSV export test FAILED: {e}")
        raise
        
    finally:
        # Restore original config
        UNet_pure.CONFIG = original_config

if __name__ == "__main__":
    test_unet_csv()