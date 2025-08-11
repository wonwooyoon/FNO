# merge_shards.py
import torch
from pathlib import Path

def load_cpu(path):
    return torch.load(path, map_location="cpu")

if __name__ == "__main__":
    # 실행 중 입력
    print("병합할 .pt 파일 번호를 한 줄씩 입력하세요. (입력 종료: 빈 줄 + Enter)")
    input_paths = []
    while True:
        p = input(f"파일 번호 {len(input_paths)+1}: ").strip()
        if p == "":
            break
        if not Path(f"./src/preprocessing/input_output_com{p}.pt").exists():
            print(f"[WARN] 파일 없음: {p}")
            continue
        input_paths.append(f"./src/preprocessing/input_output_com{p}.pt")

    if len(input_paths) < 2:
        raise RuntimeError("병합할 파일이 2개 이상 필요합니다.")

    out_pt = input("저장할 .pt 파일 경로를 입력하세요. (기본값: ./src/preprocessing/merged.pt): ").strip()
    if not out_pt:
        out_pt = "./src/preprocessing/merged.pt"

    out_path = Path(out_pt)
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
    print(f"Shapes: x{tuple(X.shape)} y{tuple(Y.shape)}")