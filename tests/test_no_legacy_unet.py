from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_unet_files_are_removed():
    legacy_paths = [
        REPO_ROOT / "src" / "FNO" / "UNET.py",
        REPO_ROOT / "src" / "FNO" / "__pycache__" / "UNET.cpython-312.pyc",
        REPO_ROOT / "tests" / "test_unet_conventional.py",
        REPO_ROOT / "tests" / "test_unet_ensemble.py",
        REPO_ROOT / "src" / "FNO" / "output_unet" / "ensemble_manifest.json",
        REPO_ROOT / "src" / "FNO" / "output_unet" / "ensemble_split_manifest.csv",
    ]

    remaining = [path.relative_to(REPO_ROOT).as_posix() for path in legacy_paths if path.exists()]

    assert remaining == []
