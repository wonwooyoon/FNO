#!/usr/bin/env python3
"""
Test FNO_pure.py single mode
"""

import sys
sys.path.append('./src/FNO')

import torch
from pathlib import Path

# Import FNO_pure and temporarily modify its config
import FNO_pure

def test_fno_single():
    print("="*60)
    print("Testing FNO_pure.py - SINGLE mode")
    print("="*60)
    
    # Backup original config
    original_config = FNO_pure.CONFIG.copy()
    
    try:
        # Modify config for test
        FNO_pure.CONFIG.update({
            'MERGED_PT_PATH': './merged_test.pt',
            'OUTPUT_DIR': './test_output_fno',
            'N_EPOCHS': 2,  # Very short test
            'TRAINING_CONFIG': {
                'mode': 'single',
                'optuna_n_trials': 3,
                'optuna_seed': 42,
                'optuna_n_startup_trials': 2,
                'eval_model_path': './test_output_fno/final/best_model_state_dict.pt'
            }
        })
        
        print("Modified CONFIG for testing:")
        print(f"  Data path: {FNO_pure.CONFIG['MERGED_PT_PATH']}")
        print(f"  Output dir: {FNO_pure.CONFIG['OUTPUT_DIR']}")
        print(f"  Epochs: {FNO_pure.CONFIG['N_EPOCHS']}")
        print(f"  Mode: {FNO_pure.CONFIG['TRAINING_CONFIG']['mode']}")
        
        # Create output directory
        Path(FNO_pure.CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        Path(FNO_pure.CONFIG['OUTPUT_DIR'] + '/final').mkdir(parents=True, exist_ok=True)
        
        # Run the test
        print("\nRunning FNO_pure.py single mode test...")
        FNO_pure.main()
        
        print("\n✅ FNO_pure.py single mode test PASSED!")
        
    except Exception as e:
        print(f"\n❌ FNO_pure.py single mode test FAILED: {e}")
        raise
        
    finally:
        # Restore original config
        FNO_pure.CONFIG = original_config

if __name__ == "__main__":
    test_fno_single()