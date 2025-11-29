#!/usr/bin/env python3
"""
PFLOTRAN Data Collection and Merging

역할:
- SSH를 통한 원격 서버 데이터 수집
- HDF5 → PyTorch tensor 변환
- 데이터 병합 및 raw 형태로 저장
- LR/HR 모드 통합 지원

Usage:
    # LR mode
    python preprocessing_collect.py --mode lr --config servers.yaml

    # HR mode
    python preprocessing_collect.py --mode hr --config servers.yaml

    # Local only (no remote servers)
    python preprocessing_collect.py --mode lr --local-only
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import torch
import h5py


# ============================================================================
# Configuration
# ============================================================================

# Get project root (2 levels up from script: src/preprocessing/ -> src/ -> root/)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CONFIG = {
    'lr': {
        'mode': 'lr',
        'pflotran_dir': PROJECT_ROOT / 'src/pflotran_run/output',
        'meta_csv': PROJECT_ROOT / 'src/initial_others/output/others.csv',
        'preprocessing_script': 'preprocessing_collect.py',
        'output_prefix': 'input_output_com',
        'output_file': SCRIPT_DIR / 'merged_raw.pt'  # Always save to src/preprocessing/
    },
    'hr': {
        'mode': 'hr',
        'pflotran_dir': PROJECT_ROOT / 'src/pflotran_run/output_hr',
        'meta_csv': PROJECT_ROOT / 'src/initial_others/output_hr/others.csv',
        'preprocessing_script': 'preprocessing_collect.py',
        'output_prefix': 'input_output_hr_com',
        'output_file': SCRIPT_DIR / 'merged_raw_hr.pt'  # Always save to src/preprocessing/
    }
}


# ============================================================================
# PFLOTRAN HDF5 Reader
# ============================================================================

def read_pflotran_h5(h5_path: Path, meta_value: float) -> Tuple:
    """
    Read PFLOTRAN HDF5 output and convert to RAW tensor

    Args:
        h5_path: Path to HDF5 file
        meta_value: Meta parameter value (added as 11th channel)

    Returns:
        x: Input tensor (11, nx, ny, nt) - includes meta
        y: Output tensor (1, nx, ny, nt) - RAW uranium concentration
        coords: (xc_unique, yc_unique) coordinate arrays
        t_labels: List of time labels
    """
    with h5py.File(h5_path, "r") as f:
        keys_list = list(f.keys())

        # Select Z=0 plane
        XC = np.array(f["Domain"]["XC"][:])
        YC = np.array(f["Domain"]["YC"][:])
        ZC = np.array(f["Domain"]["ZC"][:])
        zc_mask = (ZC >= -1e-5) & (ZC <= 1e-5)

        XC_m = XC[zc_mask]
        YC_m = YC[zc_mask]
        x_round = np.round(XC_m, 5)
        y_round = np.round(YC_m, 5)

        xc_unique, inv_x = np.unique(x_round, return_inverse=True)
        yc_unique, inv_y = np.unique(y_round, return_inverse=True)
        nx, ny = len(xc_unique), len(yc_unique)

        def to_grid(vec):
            grid = np.zeros((nx, ny), dtype=np.float32)
            grid[inv_x, inv_y] = vec.astype(np.float32, copy=False)
            return grid

        # Load t=0 input components (RAW format)
        grp0 = f["   0 Time  0.00000E+00 y"]

        perm = np.array(grp0["Permeability [m^2]"][:])[zc_mask]
        calcite = np.array(grp0["Calcite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        clino = np.array(grp0["Clinochlore VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        pyrite = np.array(grp0["Pyrite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        smectite = np.array(grp0["Smectite_MX80 VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        material = np.array(grp0["Material ID"][:])[zc_mask]

        # Load velocity from 50y
        grp1 = f["   1 Time  5.00000E+01 y"]
        x_velo = np.array(grp1["Liquid X-Velocity [m_per_yr]"][:])[zc_mask]
        y_velo = np.array(grp1["Liquid Y-Velocity [m_per_yr]"][:])[zc_mask]

        # Convert to grids
        perm_grid = to_grid(perm)
        calcite_grid = to_grid(calcite)
        clino_grid = to_grid(clino)
        pyrite_grid = to_grid(pyrite)
        smectite_grid = to_grid(smectite)
        material_grid = to_grid(material)
        x_velo_grid = to_grid(x_velo)
        y_velo_grid = to_grid(y_velo)

        # One-hot encode material (1, 2, 3 → 3 channels)
        material_source = (material_grid == 1).astype(np.float32)
        material_bentonite = (material_grid == 2).astype(np.float32)
        material_fracture = (material_grid == 3).astype(np.float32)

        # Stack input channels (10 channels)
        input_base = np.stack(
            [perm_grid, calcite_grid, clino_grid, pyrite_grid, smectite_grid,
             material_source, material_bentonite, material_fracture,
             x_velo_grid, y_velo_grid], axis=0
        )[:, :, :, np.newaxis]  # (10, nx, ny, 1)

        # Collect timesteps (100~2000y, 100y intervals)
        available = {}
        for X in range(0, 2001, 100):
            token = f"{int(X/50)} Time"
            match = next((k for k in keys_list if token in k), None)
            if match is not None:
                available[int(X/50)] = match

        times_sorted = sorted(available.keys())
        t_labels = []
        in_slices, out_slices = [], []

        # Collect all timesteps
        for tnum in times_sorted:
            key = available[tnum]

            # Load RAW UO2 concentration
            total_uo2_raw = np.array(f[key]["Total UO2++ [M]"][:])[zc_mask]

            out_grid = to_grid(total_uo2_raw)
            out_slices.append(out_grid[np.newaxis, :, :, np.newaxis])  # (1,nx,ny,1)
            in_slices.append(input_base)  # (10,nx,ny,1)
            t_labels.append(key.strip())

        x = np.concatenate(in_slices, axis=3).astype(np.float32)  # (10,nx,ny,nt)
        y = np.concatenate(out_slices, axis=3).astype(np.float32)  # (1,nx,ny,nt)

        # Add meta as 11th channel
        nt = x.shape[3]
        meta_channel = np.full((1, nx, ny, nt), meta_value, dtype=np.float32)
        x = np.concatenate([x, meta_channel], axis=0)  # (11, nx, ny, nt)

        return x, y, (xc_unique.astype(np.float32), yc_unique.astype(np.float32)), t_labels


def get_available_ids(base_dir: str) -> List[int]:
    """
    Automatically detect available pflotran simulation IDs

    Args:
        base_dir: PFLOTRAN output directory

    Returns:
        Sorted list of available simulation IDs
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return []

    available_ids = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("pflotran_"):
            match = re.match(r"pflotran_(\d+)", item.name)
            if match:
                id_num = int(match.group(1))
                h5_file = item / f"pflotran_{id_num}.h5"
                if h5_file.exists():
                    available_ids.append(id_num)

    available_ids.sort()
    return available_ids


# ============================================================================
# Local Data Processing
# ============================================================================

def process_local_data(config: dict) -> Path:
    """
    Process local PFLOTRAN data

    Args:
        config: Configuration dictionary for current mode

    Returns:
        Path to generated .pt file
    """
    print(f"\n{'='*70}")
    print("Processing Local Data")
    print(f"{'='*70}\n")

    pflotran_dir = config['pflotran_dir']
    meta_csv = config['meta_csv']
    output_prefix = config['output_prefix']

    print(f"PFLOTRAN dir: {pflotran_dir}")
    print(f"Meta CSV:     {meta_csv}")

    # Get available IDs
    print(f"\nScanning {pflotran_dir}...")
    available_ids = get_available_ids(pflotran_dir)

    if not available_ids:
        
        print(f"No PFLOTRAN data found in local")

        return None
    
    else:
    
        print(f"Found {len(available_ids)} simulations: IDs {min(available_ids)}-{max(available_ids)}")

        # Load meta data
        others = pd.read_csv(meta_csv)
        others_data = others.to_numpy(dtype=np.float32)
        max_csv_rows = len(others_data)

        # Validate IDs
        missing_ids = [id for id in available_ids if id >= max_csv_rows]
        if missing_ids:
            print(f"[ERROR] Missing meta data for IDs: {missing_ids}")
            sys.exit(1)

        # Extract meta values
        meta_values = others_data[available_ids, 2]
        print(f"Meta values: {len(meta_values)} simulations")

        # Process each simulation
        xs, ys = [], []
        coords_saved, times_saved = None, None

        for idx, sim_id in enumerate(available_ids):
            h5_path = Path(pflotran_dir) / f"pflotran_{sim_id}" / f"pflotran_{sim_id}.h5"
            meta_val = meta_values[idx]
            print(f"  Processing {h5_path.name} (meta={meta_val:.6f})...")

            x, y, coords, tlabels = read_pflotran_h5(h5_path, meta_value=meta_val)
            xs.append(x[np.newaxis, ...])
            ys.append(y[np.newaxis, ...])

            if coords_saved is None:
                coords_saved = coords
                times_saved = tlabels

        if len(xs) == 0:
            raise RuntimeError("No simulations were processed successfully")

        # Concatenate and save
        X = torch.from_numpy(np.concatenate(xs, axis=0))
        Y = torch.from_numpy(np.concatenate(ys, axis=0))

        payload = {
            "x": X, "y": Y,
            "xc": torch.from_numpy(coords_saved[0]),
            "yc": torch.from_numpy(coords_saved[1]),
            "time_keys": times_saved,
        }

        # Save to script directory (src/preprocessing/)
        output_file = SCRIPT_DIR / f"{output_prefix}localhost.pt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_file)

        print(f"\n✓ Saved: {output_file}")
        print(f"  Shape: x{tuple(X.shape)} y{tuple(Y.shape)}")

        return output_file


# ============================================================================
# Remote Data Processing (SSH)
# ============================================================================

def sync_script_on_server(host: str, user: str, port: int, script_name: str) -> bool:
    """Sync preprocessing script from git repository on remote server"""
    sync_command = (
        "cd research/FNO && "
        "git fetch origin && "
        f"git checkout origin/master -- src/preprocessing/{script_name}"
    )

    ssh_cmd = [
        'ssh', '-p', str(port), '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}', sync_command
    ]

    print(f"  Syncing {script_name} on {host}...")

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 or True  # Allow warnings
    except Exception as e:
        print(f"  Warning: Sync failed - {e}")
        return False


def execute_remote_preprocessing(
    host: str,
    user: str,
    port: int,
    script_name: str,
    mode: str
) -> bool:
    """Execute preprocessing script on remote server via SSH"""
    remote_command = (
        "cd research/FNO && "
        "source .venv_FNO/bin/activate && "
        f"python3 src/preprocessing/{script_name} --mode {mode} --local-only"
    )

    ssh_cmd = [
        'ssh', '-p', str(port), '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}', remote_command
    ]

    print(f"  Executing preprocessing on {host}...")

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            print(f"  ✓ Processing completed on {host}")
            return True
        else:
            print(f"  ✗ Processing failed on {host} (code: {result.returncode})")
            if result.stderr.strip():
                print(f"    {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Processing timed out on {host}")
        return False
    except Exception as e:
        print(f"  ✗ SSH error: {e}")
        return False


def download_result_file(
    host: str,
    user: str,
    port: int,
    output_suffix: str,
    output_prefix: str
) -> Optional[Path]:
    """Download generated .pt file from remote server using SCP"""
    remote_file = f"research/FNO/src/preprocessing/{output_prefix}localhost.pt"
    # Always save to script directory (src/preprocessing/)
    local_file = SCRIPT_DIR / f"{output_prefix}{output_suffix}.pt"

    local_file.parent.mkdir(parents=True, exist_ok=True)

    scp_cmd = [
        'scp', '-P', str(port), '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}:{remote_file}', str(local_file)
    ]

    print(f"  Downloading result from {host}...")

    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print(f"  ✓ Downloaded: {local_file.name}")
            return local_file
        else:
            print(f"  ✗ Download failed (code: {result.returncode})")
            return None

    except subprocess.TimeoutExpired:
        print(f"  ✗ Download timed out")
        return None
    except Exception as e:
        print(f"  ✗ SCP error: {e}")
        return None


def process_remote_data(server_config: dict, mode_config: dict) -> List[Path]:
    """
    Process data on remote servers via SSH

    Args:
        server_config: Server configuration from YAML
        mode_config: Configuration for current mode (lr/hr)

    Returns:
        List of paths to downloaded .pt files
    """
    print(f"\n{'='*70}")
    print("Processing Remote Servers")
    print(f"{'='*70}\n")

    if 'servers' not in server_config or len(server_config['servers']) == 0:
        print("No remote servers configured")
        return []

    script_name = mode_config['preprocessing_script']
    output_prefix = mode_config['output_prefix']

    downloaded_files = []

    for server in server_config['servers']:
        host = server['host']
        user = server['user']
        port = server.get('port', 22)
        output_suffix = f"{host}"

        print(f"\n--- Processing {host} ---")

        # Step 1: Sync script
        sync_script_on_server(host, user, port, script_name)

        # Step 2: Execute preprocessing
        success = execute_remote_preprocessing(host, user, port, script_name, mode_config['mode'])

        if not success:
            print(f"✗ Failed to process {host}")
            sys.exit(1)

        # Step 3: Download result
        local_file = download_result_file(host, user, port, output_suffix, output_prefix)

        if local_file is None:
            print(f"✗ Failed to download from {host}")
            sys.exit(1)

        downloaded_files.append(local_file)

    return downloaded_files


# ============================================================================
# Data Merging
# ============================================================================

def merge_data_shards(shard_files: List[Path], output_path: Path):
    """
    Merge multiple .pt files into a single file

    Args:
        shard_files: List of .pt file paths
        output_path: Output merged file path
    """
    print(f"\n{'='*70}")
    print("Merging Data Shards")
    print(f"{'='*70}\n")

    print(f"Found {len(shard_files)} shard(s):")
    for f in shard_files:
        print(f"  - {f}")

    if len(shard_files) < 1:
        raise RuntimeError("Need at least 1 shard to merge")

    # Load shards
    print("\nLoading shards...")
    shards = [torch.load(f, map_location='cpu') for f in shard_files]
    ref = shards[0]

    # Validate consistency
    if len(shards) > 1:
        print("Validating consistency...")
        for i, sh in enumerate(shards[1:], start=1):
            if not torch.allclose(ref['xc'], sh['xc']):
                raise RuntimeError(f"xc mismatch at shard {i}")
            if not torch.allclose(ref['yc'], sh['yc']):
                raise RuntimeError(f"yc mismatch at shard {i}")
            if ref['time_keys'] != sh['time_keys']:
                raise RuntimeError(f"time_keys mismatch at shard {i}")
            if sh['x'].shape[1:] != ref['x'].shape[1:] or sh['y'].shape[1:] != ref['y'].shape[1:]:
                raise RuntimeError(f"Shape mismatch at shard {i}")

    # Merge
    print("Merging...")
    X = torch.cat([sh['x'] for sh in shards], dim=0).contiguous()
    Y = torch.cat([sh['y'] for sh in shards], dim=0).contiguous()

    payload = {
        'x': X, 'y': Y,
        'xc': ref['xc'], 'yc': ref['yc'],
        'time_keys': ref['time_keys'],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(f"\n✓ Merge complete: {output_path}")
    print(f"  Final shape: x{tuple(X.shape)} y{tuple(Y.shape)}")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Spatial: {X.shape[2]} × {X.shape[3]}")
    print(f"  Time steps: {X.shape[4]}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect and merge PFLOTRAN data from local and remote servers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LR mode with remote servers
  python preprocessing_collect.py --mode lr --config servers.yaml

  # HR mode with remote servers
  python preprocessing_collect.py --mode hr --config servers.yaml

  # LR mode, local only
  python preprocessing_collect.py --mode lr --local-only
        """
    )

    parser.add_argument('--mode', choices=['lr', 'hr'], required=True,
                        help='Processing mode: lr (low-res) or hr (high-res)')
    parser.add_argument('--config', type=str,
                        help='Path to server configuration YAML file')
    parser.add_argument('--local-only', action='store_true',
                        help='Process local data only, skip remote servers')

    args = parser.parse_args()

    # Validate arguments
    if not args.local_only and not args.config:
        parser.error("--config is required unless --local-only is specified")

    # Get mode configuration
    mode_config = CONFIG[args.mode]

    print(f"\n{'='*70}")
    print(f"PFLOTRAN Data Collection - {args.mode.upper()} Mode")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Local only: {args.local_only}")
    print(f"Output: {mode_config['output_file']}")
    print(f"{'='*70}")

    # Collect shard files
    shard_files = []

    # 1. Process local data
    local_file = process_local_data(mode_config)
    
    if local_file is not None:
        shard_files.append(local_file)

    # 2. Process remote data (if not local-only)
    if not args.local_only:
        # Load server config
        with open(args.config, 'r') as f:
            server_config = yaml.safe_load(f)

        remote_files = process_remote_data(server_config, mode_config)
        shard_files.extend(remote_files)

    # 3. Merge all shards
    output_path = Path(mode_config['output_file'])
    merge_data_shards(shard_files, output_path)

    print(f"\n{'='*70}")
    print("Data Collection Complete!")
    print(f"{'='*70}")
    print(f"\nOutput file: {output_path}")
    print("\nNext step: Run normalization")
    print(f"  python preprocessing_normalize.py --mode {args.mode} --input {output_path} ...")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
