import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def visualize_permeability_maps():
    """투과도 맵을 2D 이미지로 시각화 (Log10 변환, 동일한 컬러 스케일)"""
    
    # 경로 설정
    perm_dir = '/home/geofluids/research/FNO/src/initial_perm/output'
    output_dir = '/home/geofluids/research/FNO/src/initial_perm'
    
    # 5개 샘플 선택
    sample_indices = [0, 1, 2, 3, 4]
    
    print("Setting up fixed color scale...")
    
    # 컬러 스케일을 -18에서 -12로 고정
    global_min = -18.0
    global_max = -12.0
    
    print(f"Fixed Log10 range: {global_min:.2f} - {global_max:.2f}")
    print("Generating permeability maps with consistent color scale...")
    
    for i, idx in enumerate(sample_indices):
        perm_file = f'{perm_dir}/perm_map_{idx}.h5'
        
        if not os.path.exists(perm_file):
            print(f"Warning: File not found: {perm_file}")
            continue
        
        try:
            # HDF5 파일에서 투과도 데이터 읽기
            with h5py.File(perm_file, 'r') as hdf:
                perm_data_full = hdf['/permsX/Data'][:]
            
            # 아래쪽 절반만 사용 (실제 사용되는 데이터 영역)
            height = perm_data_full.shape[0]
            perm_data = perm_data_full[height//2:, :]  # 아래쪽 절반
            
            # 64x32 크기로 crop (가로 64, 세로 32)
            perm_data = perm_data[:32, :64]
            
            # Log10 변환 (0 값 처리를 위해 작은 값 추가)
            perm_log = np.log10(perm_data + 1e-20)
            
            # 이미지 생성 (가로형 64x32 비율)
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))  # 2:1 비율 (64:32)
            
            # 컬러맵으로 투과도 시각화 (viridis - 노란색-보라색 계열)
            im = ax.imshow(perm_log, cmap='viridis', origin='lower', 
                          interpolation='nearest',
                          extent=[0, perm_log.shape[1], 0, perm_log.shape[0]],
                          vmin=global_min, vmax=global_max)
            
            # 축과 테두리 제거 (정확한 경계 설정)
            ax.set_xlim(0, perm_log.shape[1])
            ax.set_ylim(0, perm_log.shape[0])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 모든 여백 완전 제거
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.margins(0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            
            # 이미지 저장 (모든 여백 제거)
            output_file = f'{output_dir}/perm_map_sample_{idx}.png'
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, 
                       facecolor='none', edgecolor='none')
            plt.close()
            
            print(f"Generated: perm_map_sample_{idx}.png")
            print(f"  Original data shape: {perm_data_full.shape}")
            print(f"  Bottom half shape: {perm_data_full[height//2:, :].shape}")
            print(f"  Final cropped shape: {perm_data.shape}")
            print(f"  Permeability range: {perm_data.min():.2e} - {perm_data.max():.2e}")
            print(f"  Log10 range: {perm_log.min():.2f} - {perm_log.max():.2f}")
            
        except Exception as e:
            print(f"Error processing {perm_file}: {e}")
            continue
    
    # 컬러바 이미지 생성
    generate_colorbar_legend(global_min, global_max, output_dir)
    
    # GIF 애니메이션 생성
    create_permeability_gif(sample_indices, output_dir)
    
    print("All permeability map visualizations completed!")

def generate_colorbar_legend(vmin, vmax, output_dir):
    """컬러바 범례를 별도 이미지로 생성"""
    
    print("Generating colorbar legend...")
    
    # 세로형 컬러바
    fig, ax = plt.subplots(figsize=(2, 8))
    
    # 더미 이미지 생성 (컬러바만 표시하기 위해)
    dummy_data = np.linspace(vmin, vmax, 100).reshape(100, 1)
    im = ax.imshow(dummy_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    # 축 숨기기
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax, shrink=1.0, aspect=20)
    cbar.set_label('Log10 Permeability', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 컬러바만 남기고 메인 축 제거
    ax.remove()
    
    # 저장
    colorbar_file = f'{output_dir}/permeability_colorbar.png'
    plt.tight_layout()
    plt.savefig(colorbar_file, dpi=300, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Generated: permeability_colorbar.png")

def create_permeability_gif(sample_indices, output_dir):
    """투과도 맵 이미지들로 GIF 애니메이션 생성"""
    
    print("Creating GIF animation...")
    
    # 이미지 파일들 로드
    images = []
    for idx in sample_indices:
        img_path = f'{output_dir}/perm_map_sample_{idx}.png'
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
            print(f"  Added: perm_map_sample_{idx}.png")
        else:
            print(f"  Warning: Image not found: {img_path}")
    
    if len(images) == 0:
        print("No images found for GIF creation")
        return
    
    # GIF 저장 (2초 = 2000ms per frame)
    gif_path = f'{output_dir}/permeability_animation.gif'
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=2000,  # 2초 간격
        loop=0  # 무한 반복
    )
    
    print(f"Generated: permeability_animation.gif")
    print(f"  Frames: {len(images)}")
    print(f"  Duration: 2 seconds per frame")
    print(f"  Total cycle time: {len(images) * 2} seconds")

def check_perm_files():
    """투과도 파일들의 존재 여부 확인"""
    perm_dir = '/home/geofluids/research/FNO/src/initial_perm/output'
    
    print("Checking permeability files...")
    
    available_files = []
    for i in range(10):  # 처음 10개 파일 체크
        perm_file = f'{perm_dir}/perm_map_{i}.h5'
        if os.path.exists(perm_file):
            available_files.append(i)
            print(f"✓ perm_map_{i}.h5 exists")
        else:
            print(f"✗ perm_map_{i}.h5 not found")
    
    print(f"\nFound {len(available_files)} permeability files")
    return available_files

if __name__ == "__main__":
    # 먼저 파일 존재 여부 확인
    available_files = check_perm_files()
    
    if len(available_files) >= 5:
        print("\nProceeding with visualization...")
        visualize_permeability_maps()
    else:
        print(f"\nNot enough permeability files found. Need at least 5, but found {len(available_files)}")
        print("Run initial_perm.py first to generate permeability maps.")