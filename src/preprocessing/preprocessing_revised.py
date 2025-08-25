# preprocess_pflotran.py
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch

def read_one_h5(h5_path: Path):
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
        perm      = np.log10(np.array(grp0["Permeability [m^2]"][:])[zc_mask])
        calcite   = np.array(grp0["Calcite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        clino     = np.array(grp0["Clinochlore VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        pyrite    = np.array(grp0["Pyrite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
        smectite  = np.array(grp0["Smectite_MX80 VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]

        # 예: 50y 속도(필요시 args로 시간 바꿔도 됨)
        grp1 = f["   1 Time  5.00000E+01 y"]
        x_velo    = np.array(grp1["Liquid X-Velocity [m_per_yr]"][:])[zc_mask]
        y_velo    = np.array(grp1["Liquid Y-Velocity [m_per_yr]"][:])[zc_mask]

        perm_grid      = to_grid(perm)
        calcite_grid   = to_grid(calcite)
        clino_grid     = to_grid(clino)
        pyrite_grid    = to_grid(pyrite)
        smectite_grid  = to_grid(smectite)
        x_velo_grid    = to_grid(x_velo)
        y_velo_grid    = to_grid(y_velo)

        input_base = np.stack(
            [perm_grid, calcite_grid, clino_grid, pyrite_grid, smectite_grid,
             x_velo_grid, y_velo_grid], axis=0
        )[:, :, :, np.newaxis]  # (9, nx, ny, 1)

        # 시간 키 수집(100~2000y, 100 간격)
        available = {}
        for X in range(100, 2001, 100):
            token = f"{int(X/50)} Time"
            match = next((k for k in keys_list if token in k), None)
            if match is not None:
                available[int(X/50)] = match

        times_sorted = sorted(available.keys())
        t_labels = []
        in_slices, out_slices = [], []

        for tnum in times_sorted:
            key = available[tnum]
            total_uo2 = np.array(f[key]["Total UO2++ [M]"][:])[zc_mask]
            out_grid = to_grid(total_uo2)
            out_slices.append(out_grid[np.newaxis, :, :, np.newaxis])  # (1,nx,ny,1)
            in_slices.append(input_base)                                # (9,nx,ny,1)
            t_labels.append(key.strip())

        x = np.concatenate(in_slices, axis=3).astype(np.float32)   # (9,nx,ny,nt)
        y = np.concatenate(out_slices, axis=3).astype(np.float32)  # (1,nx,ny,nt)
        return x, y, (xc_unique.astype(np.float32), yc_unique.astype(np.float32)), t_labels

if __name__ == "__main__":
    # 실행 후 대화형 입력
    base_dir = input("HDF5 base 디렉토리 경로를 입력하세요 (기본값: ./src/pflotran_run/output): ").strip()
    if not base_dir:
        base_dir = "./src/pflotran_run/output"

    others_csv = input("others.csv 경로를 입력하세요 (기본값: ./src/initial_others/output/others.csv): ").strip()
    if not others_csv:
        others_csv = "./src/initial_others/output/others.csv"

    min_id = int(input("처리할 최소 시뮬레이션 ID (예: 0): "))
    max_id = int(input("처리할 최대 시뮬레이션 ID (예: 46): "))

    out_pt = input("저장할 .pt 파일 경로 번호 입력 (예: ./src/preprocessing/input_output_com-번호-.pt로 저장됨): ").strip()
    out_pt = f"./src/preprocessing/input_output_com{out_pt}.pt"

    others = pd.read_csv(others_csv)
    meta = others.to_numpy(dtype=np.float32)[min_id:max_id+1, 2]
    print(meta)

    xs, ys = [], []
    coords_saved, times_saved = None, None

    for i in range(min_id, max_id + 1):
        h5_path = Path(base_dir) / f"pflotran_{i}" / f"pflotran_{i}.h5"
        if not h5_path.exists():
            print(f"[WARN] Skip missing file: {h5_path}")
            continue
        x, y, coords, tlabels = read_one_h5(h5_path)
        xs.append(x[np.newaxis, ...])
        ys.append(y[np.newaxis, ...])
        if coords_saved is None:
            coords_saved = coords
            times_saved = tlabels

    import torch, numpy as np
    if len(xs) == 0:
        raise RuntimeError("No simulations found in given range.")

    X = torch.from_numpy(np.concatenate(xs, axis=0))
    Y = torch.from_numpy(np.concatenate(ys, axis=0))
    meta = torch.from_numpy(meta)
    payload = {
        "x": X, "y": Y, "meta": meta,
        "xc": torch.from_numpy(coords_saved[0]),
        "yc": torch.from_numpy(coords_saved[1]),
        "time_keys": times_saved,
    }
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    print(f"[OK] Saved shard: {out_pt} | x{tuple(X.shape)} y{tuple(Y.shape)} meta{tuple(meta.shape)}")