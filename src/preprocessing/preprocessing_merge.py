# merge_shards.py
import torch
from pathlib import Path
import glob

def load_cpu(path):
    return torch.load(path, map_location="cpu")

if __name__ == "__main__":
    # Hardcoded paths
    input_dir = "./src/preprocessing"
    output_file = "./src/preprocessing/merged_U_raw.pt"  # Now saves RAW data
    
    # Automatically find all input_output_com*.pt files
    pattern = f"{input_dir}/input_output_com*.pt"
    input_paths = glob.glob(pattern)
    input_paths.sort()  # Sort for consistent order
    
    print(f"Searching for files matching: {pattern}")
    print(f"Found {len(input_paths)} files:")
    for path in input_paths:
        print(f"  - {path}")
    
    if len(input_paths) < 1:
        raise RuntimeError("병합할 파일이 최소 1개 필요합니다.")
    
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 파일 로드
    shards = [load_cpu(pth) for pth in input_paths]
    ref = shards[0]

    # 메타데이터 일관성 확인
    def check_same(a, b, name):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or not torch.allclose(a, b):
                raise RuntimeError(f"Shard mismatch in {name}")
        elif isinstance(a, list) and isinstance(b, list):
            if a != b:
                raise RuntimeError(f"Shard mismatch in list {name}")
        else:
            raise RuntimeError(f"Unsupported meta type for {name}")

    for i, sh in enumerate(shards[1:], start=1):
        check_same(ref["xc"], sh["xc"], "xc")
        check_same(ref["yc"], sh["yc"], "yc")
        check_same(ref["time_keys"], sh["time_keys"], "time_keys")
        if sh["x"].shape[1:] != ref["x"].shape[1:] or sh["y"].shape[1:] != ref["y"].shape[1:]:
            raise RuntimeError(f"Shape mismatch at shard {i}: x{tuple(sh['x'].shape)} y{tuple(sh['y'].shape)}")

    # 병합
    X = torch.cat([sh["x"] for sh in shards], dim=0).contiguous()
    Y = torch.cat([sh["y"] for sh in shards], dim=0).contiguous()

    payload = {
        "x": X, "y": Y,
        "xc": ref["xc"], "yc": ref["yc"],
        "time_keys": ref["time_keys"],
    }
    torch.save(payload, out_path)
    print(f"[OK] 병합 완료: {len(shards)} shards → {out_path}")
    print(f"Final shapes: x{tuple(X.shape)} y{tuple(Y.shape)}")
    print(f"Total samples: {X.shape[0]}")

    # ==============================================================================
    # Apply Channel-wise Normalization
    # ==============================================================================
    print(f"\n{'='*70}")
    print("Applying Channel-wise Normalization")
    print(f"{'='*70}\n")

    from preprocessing_normalizer import apply_channel_normalization

    # Configuration for normalization check
    norm_config = {
        'OUTPUT': {
            'DPI': 150,
            'NORM_CHECK': {
                'ENABLED': True,
                'N_SAMPLES': 10
            }
        }
    }

    # Get output mode from user (default: log)
    print("Output transformation modes:")
    print("  - 'raw':   Use raw concentration values (linear scale)")
    print("  - 'log':   Apply log10 transformation (default)")
    print("  - 'delta': Compute delta from t=0 initial state")
    output_mode = input("\nEnter output transformation mode [log]: ").strip().lower()
    if not output_mode:
        output_mode = 'log'

    if output_mode not in ['raw', 'log', 'delta']:
        print(f"[ERROR] Invalid output mode: {output_mode}")
        sys.exit(1)

    # Determine normalized output path
    # E.g., merged_U_raw.pt → merged_U_<mode>_normalized.pt
    normalized_path = out_path.parent / f"merged_U_{output_mode}_normalized.pt"

    # Apply normalization
    result = apply_channel_normalization(
        merged_data_path=str(out_path),
        output_path=str(normalized_path),
        config=norm_config,
        output_mode=output_mode,
        verbose=True
    )

    print(f"\n{'='*70}")
    print("Normalization Results:")
    print(f"{'='*70}")
    print(f"Normalized data: {result['normalized_data_path']}")
    if 'normalization_check_dir' in result:
        print(f"Statistics dir:  {result['normalization_check_dir']}")
        print(f"Input stats:     {result['input_stats_csv']}")
        print(f"Output stats:    {result['output_stats_csv']}")
    print(f"{'='*70}\n")