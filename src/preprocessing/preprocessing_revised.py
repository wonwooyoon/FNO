# preprocess_pflotran.py
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import sys
import re

def read_one_h5(h5_path: Path, preprocessing_mode: str = 'log'):
    """
    Read and preprocess PFLOTRAN HDF5 output.

    Args:
        h5_path: Path to HDF5 file
        preprocessing_mode: How to process Total UO2++ concentration
            - 'raw': Use raw concentration values (linear scale, absolute)
            - 'log': Apply log10 transformation with epsilon (log10(C + 1e-12))
            - 'delta': Compute delta from t=0 initial state (excludes t=0 from output)

    Returns:
        x: Input tensor (channels, nx, ny, nt)
        y: Output tensor (1, nx, ny, nt)
        coords: (xc_unique, yc_unique) coordinate arrays
        t_labels: List of time labels
    """
    valid_modes = ['raw', 'log', 'delta']
    if preprocessing_mode not in valid_modes:
        raise ValueError(f"Invalid preprocessing_mode: {preprocessing_mode}. Must be one of {valid_modes}")

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
        grp0 = f["   0 Time  0.00000E+00 y"]

        # Permeability: log10 transformation (현재 유지)
        perm = np.log10(np.array(grp0["Permeability [m^2]"][:])[zc_mask])

        # Minerals: shifted log transformation
        # Calcite, Clino: log10(x + 1e-6) - log10(1e-6)
        # Pyrite: log10(x + 1e-9) - log10(1e-9)
        # Smectite: raw values (no transformation)
        calcite_raw = np.array(grp0["Calcite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        clino_raw = np.array(grp0["Clinochlore VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        pyrite_raw = np.array(grp0["Pyrite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        smectite = np.array(grp0["Smectite_MX80 VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]

        # Apply shifted log transformations
        calcite = np.log10(calcite_raw + 1e-6) - np.log10(1e-6)
        clino = np.log10(clino_raw + 1e-6) - np.log10(1e-6)
        pyrite = np.log10(pyrite_raw + 1e-9) - np.log10(1e-9)

        # Material ID (will be converted to one-hot encoding later)
        material = np.array(grp0["Material ID"][:])[zc_mask]

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

        # Convert material ID to one-hot encoding
        # Material IDs: 1=Source, 2=Bentonite, 3=Fracture
        material_source    = (material_grid == 1).astype(np.float32)
        material_bentonite = (material_grid == 2).astype(np.float32)
        material_fracture  = (material_grid == 3).astype(np.float32)

        input_base = np.stack(
            [perm_grid, calcite_grid, clino_grid, pyrite_grid, smectite_grid,
             material_source, material_bentonite, material_fracture,
             x_velo_grid, y_velo_grid], axis=0
        )[:, :, :, np.newaxis]  # (10, nx, ny, 1)

        # Time key collection (0~2000y, 100y intervals)
        # Include t=0 for delta mode, optional for others
        start_time = 0 if preprocessing_mode == 'delta' else 100
        available = {}
        for X in range(start_time, 2001, 100):
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

            # Load raw UO2 concentration
            total_uo2_raw = np.array(f[key]["Total UO2++ [M]"][:])[zc_mask]

            # Apply preprocessing based on mode
            if preprocessing_mode == 'raw':
                total_uo2 = total_uo2_raw  # Linear scale, absolute concentration
            elif preprocessing_mode == 'log':
                total_uo2 = np.log10(total_uo2_raw + 1e-12)  # Log scale with epsilon
            elif preprocessing_mode == 'delta':
                # For delta mode, we'll compute log first, then delta later
                total_uo2 = np.log10(total_uo2_raw + 1e-12)

            out_grid = to_grid(total_uo2)
            out_slices.append(out_grid[np.newaxis, :, :, np.newaxis])  # (1,nx,ny,1)
            in_slices.append(input_base)                                # (10,nx,ny,1)
            t_labels.append(key.strip())

        x = np.concatenate(in_slices, axis=3).astype(np.float32)   # (10,nx,ny,nt)
        y = np.concatenate(out_slices, axis=3).astype(np.float32)  # (1,nx,ny,nt)

        # Apply delta transformation if requested
        if preprocessing_mode == 'delta':
            # y[:, :, :, 0] is the initial state (reference at t=0)
            # Compute delta for t=1,2,...,nt (exclude t=0 from output)
            y_initial = y[:, :, :, 0:1]                    # (1, nx, ny, 1) - reference state at t=0
            y_delta_all = y - y_initial                     # (1, nx, ny, nt) - delta from t=0
            y = y_delta_all[:, :, :, 1:]                   # (1, nx, ny, nt-1) - exclude t=0

            # Update input to match output timesteps (exclude t=0)
            x = x[:, :, :, 1:]                             # (10, nx, ny, nt-1) - match output timesteps
            t_labels = t_labels[1:]                        # Remove t=0 label

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
    # CONFIGURATION - Preprocessing mode 설정 (이 부분만 수정하세요)
    # ========================================================================

    # Preprocessing mode 선택:
    #   - 'raw':   원시 농도 값 (선형 스케일, 절대 농도)
    #   - 'log':   Log10 변환 적용 (log10(C + 1e-12)) [기본값]
    #   - 'delta': t=0 시점 대비 변화량 (log 변환 후 delta 계산)
    preprocessing_mode = 'log'

    # ========================================================================
    # 이하 코드는 수정 불필요 (distributed_preprocessing.py와 연동)
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

    # Validate preprocessing mode
    valid_modes = ['raw', 'log', 'delta']
    if preprocessing_mode not in valid_modes:
        print(f"[ERROR] Invalid preprocessing mode: {preprocessing_mode}")
        print(f"Valid modes: {valid_modes}")
        print("  - 'raw':   Use raw concentration values (linear scale)")
        print("  - 'log':   Apply log10 transformation")
        print("  - 'delta': Compute delta from t=0 initial state")
        sys.exit(1)

    print("=" * 70)
    print("PFLOTRAN Data Preprocessing")
    print("=" * 70)
    print(f"Output suffix:        {out_suffix}")
    print(f"Preprocessing mode:   {preprocessing_mode}")
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
    meta = others_data[available_ids, 2]
    print(f"Meta shape: {meta.shape} (for {len(available_ids)} available simulations)")

    xs, ys = [], []
    coords_saved, times_saved = None, None

    for i in available_ids:
        h5_path = Path(base_dir) / f"pflotran_{i}" / f"pflotran_{i}.h5"
        print(f"Processing {h5_path}...")
        x, y, coords, tlabels = read_one_h5(h5_path, preprocessing_mode=preprocessing_mode)
        xs.append(x[np.newaxis, ...])
        ys.append(y[np.newaxis, ...])
        if coords_saved is None:
            coords_saved = coords
            times_saved = tlabels

    if len(xs) == 0:
        raise RuntimeError("No simulations were processed successfully.")

    X = torch.from_numpy(np.concatenate(xs, axis=0))
    Y = torch.from_numpy(np.concatenate(ys, axis=0))
    meta = torch.from_numpy(meta)
    
    payload = {
        "x": X, "y": Y, "meta": meta,
        "xc": torch.from_numpy(coords_saved[0]),
        "yc": torch.from_numpy(coords_saved[1]),
        "time_keys": times_saved,
    }
    
    out_pt = f"./src/preprocessing/input_output_com{out_suffix}.pt"
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    print(f"[OK] Saved shard: {out_pt} | x{tuple(X.shape)} y{tuple(Y.shape)} meta{tuple(meta.shape)}")