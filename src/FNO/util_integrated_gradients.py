"""
Integrated Gradients for FNO Spatial Attribution Analysis

단순화 버전:
- Baseline: 전체 데이터의 평균
- Aggregation: Sum of Squares
- Temporal: Sum
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def create_mean_baseline(
    train_dataset,
    val_dataset,
    test_dataset,
    verbose: bool = True
) -> torch.Tensor:
    """
    전체 데이터셋의 평균을 baseline으로 생성

    Returns:
        baseline: (1, C, nx, ny, nt)
    """

    if verbose:
        print("Creating mean baseline from all datasets...")

    all_samples = []

    # Train
    for i in range(len(train_dataset)):
        all_samples.append(train_dataset[i]['x'])

    # Val
    for i in range(len(val_dataset)):
        all_samples.append(val_dataset[i]['x'])

    # Test
    for i in range(len(test_dataset)):
        all_samples.append(test_dataset[i]['x'])

    # 평균
    baseline = torch.stack(all_samples).mean(dim=0, keepdim=True)

    if verbose:
        total = len(all_samples)
        print(f"  Total samples: {total}")
        print(f"  Baseline shape: {baseline.shape}")
        print(f"  Channel means:")
        for ch in range(baseline.shape[1]):
            mean_val = baseline[0, ch].mean().item()
            print(f"    Ch{ch}: {mean_val:.4e}")

    return baseline


def compute_integrated_gradients(
    model: nn.Module,
    processor,
    device: str,
    test_sample: torch.Tensor,  # (1, C, nx, ny, nt)
    baseline: torch.Tensor,     # (1, C, nx, ny, nt)
    target_t: int,
    n_steps: int = 50,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    IG 계산 (고정 설정)

    - 출력 집계: Sum of Squares
    - 시간 차원: Sum

    Returns:
        ig_spatial: (C, nx, ny)
        info: Dict
    """

    if verbose:
        print(f"\nComputing IG for time {target_t}...")
        print(f"  Steps: {n_steps}")

    # Wrapper: Sum of Squares aggregation
    class SumSquaresWrapper(nn.Module):
        def __init__(self, model, processor, target_t):
            super().__init__()
            self.model = model
            self.processor = processor
            self.target_t = target_t

        def forward(self, x):
            x_norm = self.processor.in_normalizer.transform(x)
            pred = self.model(x_norm)
            pred_phys = self.processor.out_normalizer.inverse_transform(pred)
            output_slice = pred_phys[:, 0, :, :, self.target_t]
            # Sum of Squares
            return (output_slice ** 2).sum(dim=[1, 2])

    wrapped = SumSquaresWrapper(model, processor, target_t).to(device)

    # IG 계산
    grads = []

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (test_sample - baseline)
        interpolated = interpolated.to(device).requires_grad_(True)

        output = wrapped(interpolated)
        output.backward()

        grads.append(interpolated.grad.detach().cpu().clone())
        interpolated.grad = None

        if verbose and step % 10 == 0:
            print(f"  Step {step}/{n_steps}, output={output.item():.4e}")

    # 평균 gradient
    avg_grad = torch.stack(grads).mean(dim=0)

    # IG: (x - baseline) × avg_grad
    ig = (test_sample - baseline) * avg_grad

    # 시간 차원 합산
    ig_spatial = ig[0, :, :, :, :].sum(dim=-1).numpy()  # (C, nx, ny)

    # Info
    info = {
        'target_t': target_t,
        'n_steps': n_steps,
        'total_abs_ig': float(np.abs(ig_spatial).sum()),
        'ig_sum': float(ig_spatial.sum()),
        'output_baseline': float(wrapped(baseline.to(device)).item()),
        'output_actual': float(wrapped(test_sample.to(device)).item())
    }

    if verbose:
        print(f"  Done. Total |IG|: {info['total_abs_ig']:.4e}")
        print(f"  Output change: {info['output_actual'] - info['output_baseline']:.4e}")
        print(f"  IG sum: {info['ig_sum']:.4e}")

    return ig_spatial, info


def visualize_ig(
    ig_results: Dict[int, np.ndarray],
    input_data: np.ndarray,  # (C, nx, ny, nt)
    model_outputs: Dict[int, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    verbose: bool = True
):
    """
    IG 시각화 - 채널별 개별 PNG 파일 생성
    실제 input channel data 위에 반투명 마스킹으로 중요도 표시
    """

    channel_names = [
        'Permeability', 'Calcite', 'Clinochlore', 'Pyrite',
        'Smectite', 'Material_ID', 'X-velocity', 'Y-velocity', 'Meta'
    ]

    channel_short = [
        'Perm', 'Calcite', 'Clino', 'Pyrite',
        'Smectite', 'MatID', 'Vx', 'Vy', 'Meta'
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for t_idx, ig_spatial in ig_results.items():
        # 각 채널별로 개별 파일 생성
        for ch in range(9):
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))

            # 실제 input data (해당 시간의 데이터)
            input_slice = input_data[ch, :, :, t_idx]

            # IG attribution
            ig_map = ig_spatial[ch]

            # 1. Input data를 배경으로 표시 (grayscale)
            input_vmin = np.percentile(input_slice, 2)
            input_vmax = np.percentile(input_slice, 98)
            ax.imshow(input_slice.T, cmap='gray', alpha=0.8,
                     vmin=input_vmin, vmax=input_vmax, aspect='auto')

            # 2. IG attribution을 반투명 마스킹으로 overlay
            # 양수와 음수의 분포를 모두 고려하여 vmin, vmax 설정
            ig_positive = ig_map[ig_map > 0]
            ig_negative = ig_map[ig_map < 0]

            if len(ig_positive) > 0:
                ig_vmax = np.percentile(ig_positive, 99)
            else:
                ig_vmax = ig_map.max()

            if len(ig_negative) > 0:
                ig_vmin = np.percentile(ig_negative, 1)
            else:
                ig_vmin = ig_map.min()

            # vmin과 vmax가 같으면 약간의 범위 추가
            if abs(ig_vmax - ig_vmin) < 1e-20:
                if abs(ig_vmax) < 1e-20:
                    # 모든 값이 0에 가까운 경우
                    ig_vmax = 1e-20
                    ig_vmin = -1e-20
                else:
                    # 대칭으로 범위 설정
                    max_abs = max(abs(ig_vmax), abs(ig_vmin))
                    ig_vmax = max_abs
                    ig_vmin = -max_abs

            # IG의 부호에 따라 red/blue로 표시
            im = ax.imshow(ig_map.T, cmap='RdBu_r',
                          vmin=ig_vmin, vmax=ig_vmax,
                          alpha=0.6, aspect='auto')

            # Title with exponential format
            ig_sum = ig_map.sum()
            ax.set_title(f'{channel_names[ch]} - Sample {sample_idx}, t={t_idx}\n∑IG={ig_sum:.4e}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Colorbar with better formatting
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('IG Attribution', rotation=270, labelpad=20)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.formatter.set_useMathText(True)
            cbar.update_ticks()

            plt.tight_layout()
            save_path = output_dir / f'ig_ch{ch}_{channel_short[ch]}_s{sample_idx}_t{t_idx:02d}.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()

            if verbose:
                print(f"  Saved: {save_path.name}")


def save_ig_csv(
    ig_results: Dict[int, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    verbose: bool = True
):
    """
    CSV 저장 - x, y 좌표와 채널별 column으로 분리
    """

    channel_names = [
        'Permeability', 'Calcite', 'Clinochlore', 'Pyrite',
        'Smectite', 'Material_ID', 'X-velocity', 'Y-velocity', 'Meta'
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for t_idx, ig_spatial in ig_results.items():
        C, nx, ny = ig_spatial.shape

        # x, y 좌표 생성
        x_coords = []
        y_coords = []
        for x in range(nx):
            for y in range(ny):
                x_coords.append(x)
                y_coords.append(y)

        # 데이터 딕셔너리 생성
        data = {
            'x_coord': x_coords,
            'y_coord': y_coords
        }

        # 각 채널별로 column 추가
        for ch in range(C):
            channel_col_name = f'{channel_names[ch]}_IG'
            ig_values = []
            for x in range(nx):
                for y in range(ny):
                    ig_values.append(ig_spatial[ch, x, y])
            data[channel_col_name] = ig_values

        df = pd.DataFrame(data)
        csv_path = output_dir / f'ig_s{sample_idx}_t{t_idx:02d}.csv'
        df.to_csv(csv_path, index=False)

        if verbose:
            print(f"  Saved: {csv_path.name}")


def analyze_time_evolution(
    ig_results: Dict[int, np.ndarray],
    output_dir: Path,
    verbose: bool = True
):
    """
    시간에 따른 채널 중요도
    """

    channel_names = [
        'Perm', 'Calcite', 'Clino', 'Pyrite',
        'Smectite', 'MatID', 'Vx', 'Vy', 'Meta'
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = sorted(ig_results.keys())
    n_channels = ig_results[times[0]].shape[0]

    importance = np.zeros((len(times), n_channels))

    for i, t in enumerate(times):
        for ch in range(n_channels):
            importance[i, ch] = np.abs(ig_results[t][ch]).sum()

    # CSV
    df = pd.DataFrame(importance, index=times, columns=channel_names)
    df.index.name = 'time'
    df.to_csv(output_dir / 'channel_importance.csv')

    # Plot
    plt.figure(figsize=(12, 6))
    for ch in range(n_channels):
        plt.plot(times, importance[:, ch], marker='o', label=channel_names[ch])

    plt.xlabel('Time')
    plt.ylabel('Total |IG|')
    plt.title('Channel Importance Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'importance_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved evolution analysis")


def integrated_gradients_analysis(
    config: Dict,
    processor,
    device: str,
    model: nn.Module,
    train_dataset,
    val_dataset,
    test_dataset,
    verbose: bool = True
):
    """
    전체 IG 분석
    """

    print("\n" + "="*70)
    print("INTEGRATED GRADIENTS ANALYSIS")
    print("="*70)

    # Config
    ig_config = config.get('IG_ANALYSIS', {})
    sample_idx = ig_config.get('SAMPLE_IDX', 5)
    time_indices = ig_config.get('TIME_INDICES', [5, 10, 15, 19])

    # Baseline 생성
    baseline = create_mean_baseline(
        train_dataset, val_dataset, test_dataset, verbose
    )

    # Test sample
    test_sample = test_dataset[sample_idx]['x'].unsqueeze(0)  # (1, C, nx, ny, nt)
    input_data = test_sample[0].cpu().numpy()  # (C, nx, ny, nt)

    print(f"\nAnalyzing sample {sample_idx} at times {time_indices}")

    # IG 계산
    ig_results = {}
    for t in time_indices:
        ig_spatial, info = compute_integrated_gradients(
            model, processor, device,
            test_sample, baseline, t,
            n_steps=50, verbose=verbose
        )
        ig_results[t] = ig_spatial

    # 모델 출력
    model_outputs = {}
    with torch.no_grad():
        x_norm = processor.in_normalizer.transform(test_sample.to(device))
        pred = model(x_norm)
        pred_phys = processor.out_normalizer.inverse_transform(pred)
        for t in time_indices:
            model_outputs[t] = pred_phys[0, 0, :, :, t].cpu().numpy()

    # 출력
    output_dir = Path(config['OUTPUT_DIR']) / 'integrated_gradients'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating outputs...")
    visualize_ig(ig_results, input_data, model_outputs, sample_idx, output_dir, verbose)
    save_ig_csv(ig_results, sample_idx, output_dir, verbose)
    analyze_time_evolution(ig_results, output_dir, verbose)

    print("\n" + "="*70)
    print(f"COMPLETED! Results in: {output_dir}")
    print("="*70)

    return ig_results
