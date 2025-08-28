#!/usr/bin/env python3
"""
Test FNO_pure.py eval mode
"""

import sys
sys.path.append('./src/FNO')

import torch
from pathlib import Path

# Import FNO_pure and temporarily modify its config
import FNO_pure

def test_fno_eval():
    print("="*60)
    print("Testing FNO_pure.py - EVAL mode")
    print("="*60)
    
    # Backup original config
    original_config = FNO_pure.CONFIG.copy()
    
    try:
        # Check if model file exists
        model_path = './test_output_fno/final/best_model_state_dict.pt'
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print("Run single mode test first to generate a model.")
            return
        
        # Modify config for test
        FNO_pure.CONFIG.update({
            'MERGED_PT_PATH': './merged_test.pt',
            'OUTPUT_DIR': './test_output_fno_eval',
            'N_EPOCHS': 1,  # Not used in eval mode
            'TRAINING_CONFIG': {
                'mode': 'eval',
                'optuna_n_trials': 2,
                'optuna_seed': 42,
                'optuna_n_startup_trials': 1,
                'eval_model_path': model_path
            }
        })
        
        print("Modified CONFIG for testing:")
        print(f"  Data path: {FNO_pure.CONFIG['MERGED_PT_PATH']}")
        print(f"  Output dir: {FNO_pure.CONFIG['OUTPUT_DIR']}")
        print(f"  Mode: {FNO_pure.CONFIG['TRAINING_CONFIG']['mode']}")
        print(f"  Model path: {FNO_pure.CONFIG['TRAINING_CONFIG']['eval_model_path']}")
        
        # Create output directory
        Path(FNO_pure.CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        Path(FNO_pure.CONFIG['OUTPUT_DIR'] + '/final').mkdir(parents=True, exist_ok=True)
        
        # Run the test
        print("\nRunning FNO_pure.py eval mode test...")
        FNO_pure.main()
        
        print("\n✅ FNO_pure.py eval mode test PASSED!")
        
    except Exception as e:
        print(f"\n❌ FNO_pure.py eval mode test FAILED: {e}")
        raise
        
    finally:
        # Restore original config
        FNO_pure.CONFIG = original_config

if __name__ == "__main__":
    test_fno_eval()