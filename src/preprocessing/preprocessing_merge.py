# merge_shards.py
import torch
from pathlib import Path
import glob

def load_cpu(path):
    return torch.load(path, map_location="cpu")

if __name__ == "__main__":
    # Hardcoded paths
    input_dir = "./src/preprocessing"
    output_file = "./src/preprocessing/merged_U.pt"
    
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
    
    if len(input_paths) == 1:
        print("[INFO] 파일이 1개만 있습니다. 단순 복사합니다.")
        import shutil
        shutil.copy2(input_paths[0], output_file)
        print(f"[OK] 파일 복사 완료: {input_paths[0]} → {output_file}")
        exit(0)

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
    META = torch.cat([sh["meta"] for sh in shards], dim=0).contiguous()

    payload = {
        "x": X, "y": Y, "meta": META,
        "xc": ref["xc"], "yc": ref["yc"],
        "time_keys": ref["time_keys"],
    }
    torch.save(payload, out_path)
    print(f"[OK] 병합 완료: {len(shards)} shards → {out_path}")
    print(f"Final shapes: x{tuple(X.shape)} y{tuple(Y.shape)} meta{tuple(META.shape)}")
    print(f"Total samples: {X.shape[0]}")