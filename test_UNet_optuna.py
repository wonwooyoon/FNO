#!/usr/bin/env python3
"""
Test UNet_pure.py optuna mode
"""

import sys
sys.path.append('./src/FNO')

import torch
from pathlib import Path

# Import UNet_pure and temporarily modify its config
import UNet_pure

def test_unet_optuna():
    print("="*60)
    print("Testing UNet_pure.py - OPTUNA mode")
    print("="*60)
    
    # Backup original config
    original_config = UNet_pure.CONFIG.copy()
    
    try:
        # Modify config for test - very short optuna test
        UNet_pure.CONFIG.update({
            'MERGED_PT_PATH': './merged_test.pt',
            'OUTPUT_DIR': './test_output_unet_optuna',
            'N_EPOCHS': 1,  # Very short test
            'TRAINING_CONFIG': {
                'mode': 'optuna',
                'optuna_n_trials': 2,  # Very short optuna test
                'optuna_seed': 42,
                'optuna_n_startup_trials': 1,
                'eval_model_path': './test_output_unet_optuna/final/best_model_state_dict.pt'
            }
        })
        
        print("Modified CONFIG for testing:")
        print(f"  Data path: {UNet_pure.CONFIG['MERGED_PT_PATH']}")
        print(f"  Output dir: {UNet_pure.CONFIG['OUTPUT_DIR']}")
        print(f"  Epochs: {UNet_pure.CONFIG['N_EPOCHS']}")
        print(f"  Mode: {UNet_pure.CONFIG['TRAINING_CONFIG']['mode']}")
        print(f"  Optuna trials: {UNet_pure.CONFIG['TRAINING_CONFIG']['optuna_n_trials']}")
        
        # Create output directory
        Path(UNet_pure.CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        Path(UNet_pure.CONFIG['OUTPUT_DIR'] + '/final').mkdir(parents=True, exist_ok=True)
        
        # Run the test
        print("\nRunning UNet_pure.py optuna mode test...")
        UNet_pure.main()
        
        print("\n✅ UNet_pure.py optuna mode test PASSED!")
        
    except Exception as e:
        print(f"\n❌ UNet_pure.py optuna mode test FAILED: {e}")
        raise
        
    finally:
        # Restore original config
        UNet_pure.CONFIG = original_config

if __name__ == "__main__":
    test_unet_optuna()