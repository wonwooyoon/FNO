#!/usr/bin/env python3
"""
Test FNO_pure.py CSV export functionality
"""

import sys
sys.path.append('./src/FNO')

import torch
from pathlib import Path

# Import FNO_pure and temporarily modify its config
import FNO_pure

def test_fno_csv():
    print("="*60)
    print("Testing FNO_pure.py - CSV Export Feature")
    print("="*60)
    
    # Backup original config
    original_config = FNO_pure.CONFIG.copy()
    
    try:
        # Modify config for test with CSV export enabled
        FNO_pure.CONFIG.update({
            'MERGED_PT_PATH': './merged_test.pt',
            'OUTPUT_DIR': './test_output_fno_csv',
            'N_EPOCHS': 1,  # Very short test
            'TRAINING_CONFIG': {
                'mode': 'single',
                'optuna_n_trials': 2,
                'optuna_seed': 42,
                'optuna_n_startup_trials': 1,
                'eval_model_path': './test_output_fno_csv/final/best_model_state_dict.pt'
            },
            'VISUALIZATION': {
                'SAMPLE_NUM': 8,
                'TIME_INDICES': (4, 9, 14, 19),
                'DPI': 200,
                'SAVEASCSV': True  # Enable CSV export
            }
        })
        
        print("Modified CONFIG for testing:")
        print(f"  Data path: {FNO_pure.CONFIG['MERGED_PT_PATH']}")
        print(f"  Output dir: {FNO_pure.CONFIG['OUTPUT_DIR']}")
        print(f"  Epochs: {FNO_pure.CONFIG['N_EPOCHS']}")
        print(f"  Mode: {FNO_pure.CONFIG['TRAINING_CONFIG']['mode']}")
        print(f"  SAVEASCSV: {FNO_pure.CONFIG['VISUALIZATION']['SAVEASCSV']}")
        
        # Create output directory
        Path(FNO_pure.CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        Path(FNO_pure.CONFIG['OUTPUT_DIR'] + '/final').mkdir(parents=True, exist_ok=True)
        
        # Run the test
        print("\nRunning FNO_pure.py with CSV export...")
        FNO_pure.main()
        
        # Check if CSV file was created
        csv_path = Path(FNO_pure.CONFIG['OUTPUT_DIR']) / 'FNO_visualization_data.csv'
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
            
        print("\n✅ FNO_pure.py CSV export test PASSED!")
        
    except Exception as e:
        print(f"\n❌ FNO_pure.py CSV export test FAILED: {e}")
        raise
        
    finally:
        # Restore original config
        FNO_pure.CONFIG = original_config

if __name__ == "__main__":
    test_fno_csv()