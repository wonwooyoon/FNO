#!/usr/bin/env python3
"""
Test UNet_pure.py eval mode
"""

import sys
sys.path.append('./src/FNO')

import torch
from pathlib import Path

# Import UNet_pure and temporarily modify its config
import UNet_pure

def test_unet_eval():
    print("="*60)
    print("Testing UNet_pure.py - EVAL mode")
    print("="*60)
    
    # Backup original config
    original_config = UNet_pure.CONFIG.copy()
    
    try:
        # Check if model file exists
        model_path = './test_output_unet/final/best_model_state_dict.pt'
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print("Run single mode test first to generate a model.")
            return
        
        # Modify config for test
        UNet_pure.CONFIG.update({
            'MERGED_PT_PATH': './merged_test.pt',
            'OUTPUT_DIR': './test_output_unet_eval',
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
        print(f"  Data path: {UNet_pure.CONFIG['MERGED_PT_PATH']}")
        print(f"  Output dir: {UNet_pure.CONFIG['OUTPUT_DIR']}")
        print(f"  Mode: {UNet_pure.CONFIG['TRAINING_CONFIG']['mode']}")
        print(f"  Model path: {UNet_pure.CONFIG['TRAINING_CONFIG']['eval_model_path']}")
        
        # Create output directory
        Path(UNet_pure.CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        Path(UNet_pure.CONFIG['OUTPUT_DIR'] + '/final').mkdir(parents=True, exist_ok=True)
        
        # Run the test
        print("\nRunning UNet_pure.py eval mode test...")
        UNet_pure.main()
        
        print("\n✅ UNet_pure.py eval mode test PASSED!")
        
    except Exception as e:
        print(f"\n❌ UNet_pure.py eval mode test FAILED: {e}")
        raise
        
    finally:
        # Restore original config
        UNet_pure.CONFIG = original_config

if __name__ == "__main__":
    test_unet_eval()