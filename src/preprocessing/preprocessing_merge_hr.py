#!/usr/bin/env python3
"""
High-Resolution Data Merge and Normalization with Pre-trained Normalizer

This script merges high-resolution preprocessing shards and applies a pre-trained
normalizer (from low-resolution training data) for zero-shot super-resolution.

Key Principle:
    For zero-shot super-resolution, the normalizer MUST be identical to the one
    used during low-resolution training, regardless of resolution differences.

Usage:
    python preprocessing_merge_hr.py
"""

import torch
from pathlib import Path
import glob
import sys

def load_cpu(path):
    """Load torch file to CPU memory."""
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("High-Resolution Data Merge and Normalization")
    print(f"{'='*70}\n")

    # ==============================================================================
    # Step 1: Merge High-Resolution Shards
    # ==============================================================================
    print("Step 1: Merging high-resolution data shards...")
    print("-" * 70)

    # Hardcoded paths for high-resolution data
    input_dir = "./src/preprocessing"
    output_file = "./src/preprocessing/merged_U_raw_hr.pt"  # High-res RAW data

    # Automatically find all high-resolution input_output files
    # Pattern: input_output_com*_hr.pt or input_output_hr*.pt
    patterns = [
        f"{input_dir}/input_output_com*_hr.pt",
        f"{input_dir}/input_output_hr*.pt",
        f"{input_dir}/input_output_*highres*.pt",
    ]

    input_paths = []
    for pattern in patterns:
        input_paths.extend(glob.glob(pattern))

    # Remove duplicates and sort
    input_paths = sorted(list(set(input_paths)))

    print(f"Searching for high-resolution files...")
    print(f"Patterns: {patterns}")
    print(f"Found {len(input_paths)} file(s):")
    for path in input_paths:
        print(f"  - {path}")

    if len(input_paths) < 1:
        print("\n[ERROR] No high-resolution files found!")
        print("Expected filename patterns:")
        print("  - input_output_com*_hr.pt")
        print("  - input_output_hr*.pt")
        print("  - input_output_*highres*.pt")
        sys.exit(1)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load files
    print("\nLoading shards...")
    shards = [load_cpu(pth) for pth in input_paths]
    ref = shards[0]

    # Validate consistency across shards
    def check_same(a, b, name):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or not torch.allclose(a, b):
                raise RuntimeError(f"Shard mismatch in {name}")
        elif isinstance(a, list) and isinstance(b, list):
            if a != b:
                raise RuntimeError(f"Shard mismatch in list {name}")
        else:
            raise RuntimeError(f"Unsupported meta type for {name}")

    if len(shards) > 1:
        print("Validating shard consistency...")
        for i, sh in enumerate(shards[1:], start=1):
            check_same(ref["xc"], sh["xc"], "xc")
            check_same(ref["yc"], sh["yc"], "yc")
            check_same(ref["time_keys"], sh["time_keys"], "time_keys")
            if sh["x"].shape[1:] != ref["x"].shape[1:] or sh["y"].shape[1:] != ref["y"].shape[1:]:
                raise RuntimeError(f"Shape mismatch at shard {i}: x{tuple(sh['x'].shape)} y{tuple(sh['y'].shape)}")

    # Merge shards
    print("Merging data...")
    X = torch.cat([sh["x"] for sh in shards], dim=0).contiguous()
    Y = torch.cat([sh["y"] for sh in shards], dim=0).contiguous()

    payload = {
        "x": X, "y": Y,
        "xc": ref["xc"], "yc": ref["yc"],
        "time_keys": ref["time_keys"],
    }

    torch.save(payload, out_path)

    print(f"\n[OK] Merge complete: {len(shards)} shard(s) → {out_path}")
    print(f"Final shapes: x{tuple(X.shape)} y{tuple(Y.shape)}")
    print(f"Total samples: {X.shape[0]}")
    print(f"Spatial resolution: {X.shape[2]} × {X.shape[3]}")
    print(f"Time steps: {X.shape[4]}")

    # ==============================================================================
    # Step 2: Apply Pre-trained Normalizer
    # ==============================================================================
    print(f"\n{'='*70}")
    print("Step 2: Applying Pre-trained Normalizer to High-Resolution Data")
    print(f"{'='*70}\n")

    from preprocessing_normalizer_hr import apply_pretrained_normalizer_to_hr_data

    # Get normalizer path from user
    print("⚠️  IMPORTANT: Use the normalizer from LOW-RESOLUTION training data!")
    print("   (e.g., channel_normalizer_log.pkl from low-res training)\n")

    default_normalizer = "./src/preprocessing/channel_normalizer_log.pkl"
    normalizer_path = input(f"Enter normalizer path [{default_normalizer}]: ").strip()

    if not normalizer_path:
        normalizer_path = default_normalizer

    normalizer_path_obj = Path(normalizer_path)

    if not normalizer_path_obj.exists():
        print(f"\n[ERROR] Normalizer not found: {normalizer_path}")
        print("\nPlease ensure you have:")
        print("  1. Trained a model on low-resolution data")
        print("  2. Generated normalizer using preprocessing_merge.py")
        print("  3. The normalizer file exists at the specified path")
        sys.exit(1)

    # Determine normalized output path
    # Infer output mode from normalizer filename
    # E.g., channel_normalizer_log.pkl → log
    normalizer_name = normalizer_path_obj.stem  # e.g., 'channel_normalizer_log'

    if 'log' in normalizer_name:
        output_mode = 'log'
    elif 'delta' in normalizer_name:
        output_mode = 'delta'
    elif 'raw' in normalizer_name:
        output_mode = 'raw'
    else:
        # Default to log if cannot infer
        output_mode = 'log'
        print(f"⚠️  Cannot infer output mode from normalizer filename, using default: '{output_mode}'")

    print(f"Inferred output mode from normalizer: '{output_mode}'")

    # E.g., merged_U_raw_hr.pt → merged_U_<mode>_normalized_hr.pt
    normalized_path = out_path.parent / f"merged_U_{output_mode}_normalized_hr.pt"

    print(f"\nApplying normalizer...")
    print(f"  Input (raw):      {out_path}")
    print(f"  Normalizer:       {normalizer_path}")
    print(f"  Output (normed):  {normalized_path}")
    print()

    # Apply pre-trained normalizer
    try:
        result = apply_pretrained_normalizer_to_hr_data(
            hr_data_path=str(out_path),
            normalizer_path=str(normalizer_path),
            output_path=str(normalized_path),
            verbose=True
        )

        print(f"\n{'='*70}")
        print("High-Resolution Data Processing Complete!")
        print(f"{'='*70}")
        print(f"Raw data:        {out_path}")
        print(f"Normalized data: {result['normalized_data_path']}")
        print(f"Normalizer used: {result['normalizer_path']}")
        print(f"\n⚠️  Remember: Use the SAME normalizer for inverse transform!")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n[ERROR] Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)