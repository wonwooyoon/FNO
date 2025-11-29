# preprocess_pflotran.py
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import sys
import re

def read_one_h5(h5_path: Path, meta_value: float):
    """
    Read PFLOTRAN HDF5 output and save as RAW data.

    Args:
        h5_path: Path to HDF5 file
        meta_value: Meta parameter value to add as 11th channel

    Returns:
        x: Input tensor (11, nx, ny, nt) - includes meta as 11th channel
        y: Output tensor (1, nx, ny, nt) - RAW uranium concentration
        coords: (xc_unique, yc_unique) coordinate arrays
        t_labels: List of time labels
    """

    with h5py.File(h5_path, "r") as f:
        keys_list = list(f.keys())

        # Z=0 단면 선택
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

        # t=0 고정 입력 성분
        # NOTE: All data is stored in RAW format (no transformations applied)
        # Transformations (log, shifted log, etc.) will be applied during normalization
        grp0 = f["   0 Time  0.00000E+00 y"]

        # Load all data in raw format
        perm = np.array(grp0["Permeability [m^2]"][:])[zc_mask]  # Raw permeability (will be log10 in normalizer)
        calcite = np.array(grp0["Calcite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]  # Raw
        clino = np.array(grp0["Clinochlore VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]  # Raw
        pyrite = np.array(grp0["Pyrite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]  # Raw
        smectite = np.array(grp0["Smectite_MX80 VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]  # Raw
        material = np.array(grp0["Material ID"][:])[zc_mask]  # Raw (will be one-hot in normalizer)

        # 예: 50y 속도(필요시 args로 시간 바꿔도 됨)
        grp1 = f["   1 Time  5.00000E+01 y"]
        x_velo    = np.array(grp1["Liquid X-Velocity [m_per_yr]"][:])[zc_mask]
        y_velo    = np.array(grp1["Liquid Y-Velocity [m_per_yr]"][:])[zc_mask]

        perm_grid      = to_grid(perm)
        calcite_grid   = to_grid(calcite)
        clino_grid     = to_grid(clino)
        pyrite_grid    = to_grid(pyrite)
        smectite_grid  = to_grid(smectite)
        material_grid  = to_grid(material)
        x_velo_grid    = to_grid(x_velo)
        y_velo_grid    = to_grid(y_velo)

        # Apply one-hot encoding to material (1, 2, 3 → 3 channels)
        material_source = (material_grid == 1).astype(np.float32)     # Material ID 1: Source
        material_bentonite = (material_grid == 2).astype(np.float32)  # Material ID 2: Bentonite
        material_fracture = (material_grid == 3).astype(np.float32)   # Material ID 3: Fracture

        # Stack all input channels (10 channels: 5 properties + 3 materials + 2 velocities)
        input_base = np.stack(
            [perm_grid, calcite_grid, clino_grid, pyrite_grid, smectite_grid,
             material_source, material_bentonite, material_fracture,
             x_velo_grid, y_velo_grid], axis=0
        )[:, :, :, np.newaxis]  # (10, nx, ny, 1) - raw data with one-hot encoded material

        # Time key collection (100~2000y, 100y intervals)
        # Always include all timesteps in RAW format (t=0 excluded, 20 timesteps total)
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

            # Load RAW UO2 concentration (no transformation)
            total_uo2_raw = np.array(f[key]["Total UO2++ [M]"][:])[zc_mask]

            out_grid = to_grid(total_uo2_raw)
            out_slices.append(out_grid[np.newaxis, :, :, np.newaxis])  # (1,nx,ny,1)
            in_slices.append(input_base)                                # (10,nx,ny,1)
            t_labels.append(key.strip())

        x = np.concatenate(in_slices, axis=3).astype(np.float32)   # (10,nx,ny,nt)
        y = np.concatenate(out_slices, axis=3).astype(np.float32)  # (1,nx,ny,nt) - RAW concentration

        # Add meta value as 11th channel (constant across space and time)
        nt = x.shape[3]
        meta_channel = np.full((1, nx, ny, nt), meta_value, dtype=np.float32)  # (1, nx, ny, nt)
        x = np.concatenate([x, meta_channel], axis=0)  # (11, nx, ny, nt)

        return x, y, (xc_unique.astype(np.float32), yc_unique.astype(np.float32)), t_labels

def get_available_ids(base_dir: str):
    """Automatically detect available pflotran IDs from output directory"""
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

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    # This script saves all data in RAW format.
    # Transformations (log, delta) will be applied during normalization step.
    # ========================================================================

    # Default paths
    base_dir = "./src/pflotran_run/output"
    others_csv = "./src/initial_others/output/others.csv"

    # Get output suffix from command line argument
    # This is required for distributed_preprocessing.py integration
    if len(sys.argv) > 1:
        out_suffix = sys.argv[1]
    else:
        out_suffix = input("저장할 .pt 파일 suffix를 입력하세요 (예: server1_0_14): ").strip()
        if not out_suffix:
            print("[ERROR] Output suffix is required!")
            sys.exit(1)

    print("=" * 70)
    print("PFLOTRAN Data Preprocessing (RAW Data)")
    print("=" * 70)
    print(f"Output suffix:        {out_suffix}")
    print(f"Output mode:          RAW (transformations will be applied in normalizer)")
    print(f"Base directory:       {base_dir}")
    print(f"Others CSV:           {others_csv}")
    print("=" * 70)
    
    # Automatically detect available IDs
    print(f"Scanning {base_dir} for available pflotran data...")
    available_ids = get_available_ids(base_dir)
    
    if not available_ids:
        print(f"[ERROR] No valid pflotran data found in {base_dir}")
        sys.exit(1)
    
    min_id = min(available_ids)
    max_id = max(available_ids)
    print(f"Found {len(available_ids)} simulations: IDs {min_id}-{max_id}")
    print(f"Available IDs: {available_ids}")
    
    # Process data
    others = pd.read_csv(others_csv)

    # Extract meta data only for available IDs to handle non-sequential simulation runs
    others_data = others.to_numpy(dtype=np.float32)
    max_csv_rows = len(others_data)

    # Validate that all available_ids exist in CSV
    missing_ids = [id for id in available_ids if id >= max_csv_rows]
    if missing_ids:
        print(f"[ERROR] Missing meta data for IDs: {missing_ids}")
        print(f"CSV has {max_csv_rows} rows, but need data for IDs up to {max(available_ids)}")
        sys.exit(1)

    # Extract meta values only for available simulation IDs
    meta_values = others_data[available_ids, 2]
    print(f"Meta values: {len(meta_values)} simulations")

    xs, ys = [], []
    coords_saved, times_saved = None, None

    for idx, sim_id in enumerate(available_ids):
        h5_path = Path(base_dir) / f"pflotran_{sim_id}" / f"pflotran_{sim_id}.h5"
        meta_val = meta_values[idx]
        print(f"Processing {h5_path} (meta={meta_val:.6f})...")
        x, y, coords, tlabels = read_one_h5(h5_path, meta_value=meta_val)
        xs.append(x[np.newaxis, ...])
        ys.append(y[np.newaxis, ...])
        if coords_saved is None:
            coords_saved = coords
            times_saved = tlabels

    if len(xs) == 0:
        raise RuntimeError("No simulations were processed successfully.")

    X = torch.from_numpy(np.concatenate(xs, axis=0))
    Y = torch.from_numpy(np.concatenate(ys, axis=0))

    payload = {
        "x": X, "y": Y,
        "xc": torch.from_numpy(coords_saved[0]),
        "yc": torch.from_numpy(coords_saved[1]),
        "time_keys": times_saved,
    }
    
    out_pt = f"./src/preprocessing/input_output_com{out_suffix}.pt"
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    print(f"[OK] Saved shard: {out_pt} | x{tuple(X.shape)} y{tuple(Y.shape)}")