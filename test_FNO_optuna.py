#!/usr/bin/env python3
"""
Test FNO_pure.py optuna mode
"""

import sys
sys.path.append('./src/FNO')

import torch
from pathlib import Path

# Import FNO_pure and temporarily modify its config
import FNO_pure

def test_fno_optuna():
    print("="*60)
    print("Testing FNO_pure.py - OPTUNA mode")
    print("="*60)
    
    # Backup original config
    original_config = FNO_pure.CONFIG.copy()
    
    try:
        # Modify config for test - very short optuna test
        FNO_pure.CONFIG.update({
            'MERGED_PT_PATH': './merged_test.pt',
            'OUTPUT_DIR': './test_output_fno_optuna',
            'N_EPOCHS': 1,  # Very short test
            'TRAINING_CONFIG': {
                'mode': 'optuna',
                'optuna_n_trials': 2,  # Very short optuna test
                'optuna_seed': 42,
                'optuna_n_startup_trials': 1,
                'eval_model_path': './test_output_fno_optuna/final/best_model_state_dict.pt'
            }
        })
        
        print("Modified CONFIG for testing:")
        print(f"  Data path: {FNO_pure.CONFIG['MERGED_PT_PATH']}")
        print(f"  Output dir: {FNO_pure.CONFIG['OUTPUT_DIR']}")
        print(f"  Epochs: {FNO_pure.CONFIG['N_EPOCHS']}")
        print(f"  Mode: {FNO_pure.CONFIG['TRAINING_CONFIG']['mode']}")
        print(f"  Optuna trials: {FNO_pure.CONFIG['TRAINING_CONFIG']['optuna_n_trials']}")
        
        # Create output directory
        Path(FNO_pure.CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        Path(FNO_pure.CONFIG['OUTPUT_DIR'] + '/final').mkdir(parents=True, exist_ok=True)
        
        # Run the test
        print("\nRunning FNO_pure.py optuna mode test...")
        FNO_pure.main()
        
        print("\n✅ FNO_pure.py optuna mode test PASSED!")
        
    except Exception as e:
        print(f"\n❌ FNO_pure.py optuna mode test FAILED: {e}")
        raise
        
    finally:
        # Restore original config
        FNO_pure.CONFIG = original_config

if __name__ == "__main__":
    test_fno_optuna()