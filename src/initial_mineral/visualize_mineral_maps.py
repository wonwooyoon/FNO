import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def visualize_mineral_maps():
    """미네랄 맵을 2D 이미지로 시각화 (로그 스케일, 미네랄별 개별 스케일)"""
    
    # 경로 설정
    mineral_dir = '/home/geofluids/research/FNO/src/initial_mineral/output'
    output_dir = '/home/geofluids/research/FNO/src/initial_mineral'
    
    # 미네랄 종류
    minerals = ['clinochlore', 'calcite', 'pyrite']
    
    # 5개 샘플 선택
    sample_indices = [0, 1, 2, 3, 4]
    
    print("Setting up fixed log scale ranges for minerals...")
    
    # 미네랄별 고정 스케일 범위 설정
    mineral_ranges = {
        'clinochlore': (-4.0, 0.0),
        'calcite': (-4.0, 0.0),
        'pyrite': (-7.0, -2.0)
    }
    
    for mineral in minerals:
        vmin, vmax = mineral_ranges[mineral]
        print(f"  {mineral} Fixed Log10 range: {vmin:.1f} - {vmax:.1f}")
    
    print("\nGenerating mineral maps with fixed log scales...")
    
    for mineral in minerals:
        print(f"\nProcessing {mineral} maps...")
        mineral_min, mineral_max = mineral_ranges[mineral]
        
        for i, idx in enumerate(sample_indices):
            mineral_file = f'{mineral_dir}/{mineral}_{idx}.h5'
            
            if not os.path.exists(mineral_file):
                print(f"Warning: File not found: {mineral_file}")
                continue
            
            try:
                # HDF5 파일에서 미네랄 데이터 읽기
                with h5py.File(mineral_file, 'r') as hdf:
                    mineral_data_full = hdf[f'/{mineral}_mapX/Data'][:]
                
                # 데이터가 128x64라면 이미 하부만 있는 것, 아니라면 하부 절반 추출
                if mineral_data_full.shape[0] == mineral_data_full.shape[1]:  # 정사각형인 경우
                    height = mineral_data_full.shape[0]
                    mineral_data = mineral_data_full[height//2:, :]  # 하부 절반
                else:
                    mineral_data = mineral_data_full  # 이미 하부만 있음
                
                # 로그 변환 (0 값 처리)
                mineral_log = np.log10(mineral_data + 1e-10)
                
                # 이미지 생성 (가로형 64x32 비율)
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))  # 2:1 비율 (64:32)
                
                # 컬러맵으로 미네랄 시각화 (viridis, 미네랄별 개별 스케일)
                im = ax.imshow(mineral_log, cmap='viridis', origin='lower', 
                              interpolation='nearest',
                              extent=[0, mineral_log.shape[1], 0, mineral_log.shape[0]],
                              vmin=mineral_min, vmax=mineral_max)
                
                # 축과 테두리 제거 (정확한 경계 설정)
                ax.set_xlim(0, mineral_log.shape[1])
                ax.set_ylim(0, mineral_log.shape[0])
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
                output_file = f'{output_dir}/{mineral}_map_sample_{idx}.png'
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, 
                           facecolor='none', edgecolor='none')
                plt.close()
                
                print(f"Generated: {mineral}_map_sample_{idx}.png")
                print(f"  Original data shape: {mineral_data_full.shape}")
                if mineral_data_full.shape[0] == mineral_data_full.shape[1]:
                    print(f"  Bottom half shape: {mineral_data_full[mineral_data_full.shape[0]//2:, :].shape}")
                else:
                    print(f"  Full data shape: {mineral_data_full.shape}")
                print(f"  Final cropped shape: {mineral_data.shape}")
                print(f"  Mineral range: {mineral_data.min():.6f} - {mineral_data.max():.6f}")
                print(f"  Log10 range: {mineral_log.min():.2f} - {mineral_log.max():.2f}")
                
            except Exception as e:
                print(f"Error processing {mineral_file}: {e}")
                continue
    
    # 각 미네랄별 컬러바 이미지 생성 (개별 범위 사용)
    for mineral in minerals:
        mineral_min, mineral_max = mineral_ranges[mineral]
        generate_mineral_colorbar_legend(mineral_min, mineral_max, output_dir, mineral)
    
    # 각 미네랄별 GIF 애니메이션 생성
    for mineral in minerals:
        create_mineral_gif(mineral, sample_indices, output_dir)
    
    print("\nAll mineral map visualizations completed!")

def generate_mineral_colorbar_legend(vmin, vmax, output_dir, mineral):
    """미네랄별 컬러바 범례를 별도 이미지로 생성"""
    
    print(f"Generating {mineral} colorbar legend...")
    
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
    cbar.set_label(f'{mineral.capitalize()} Log10 Concentration', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 컬러바만 남기고 메인 축 제거
    ax.remove()
    
    # 저장
    colorbar_file = f'{output_dir}/{mineral}_colorbar.png'
    plt.tight_layout()
    plt.savefig(colorbar_file, dpi=300, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Generated: {mineral}_colorbar.png")

def create_mineral_gif(mineral, sample_indices, output_dir):
    """미네랄 맵 이미지들로 GIF 애니메이션 생성"""
    
    print(f"Creating {mineral} GIF animation...")
    
    # 이미지 파일들 로드
    images = []
    for idx in sample_indices:
        img_path = f'{output_dir}/{mineral}_map_sample_{idx}.png'
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
            print(f"  Added: {mineral}_map_sample_{idx}.png")
        else:
            print(f"  Warning: Image not found: {img_path}")
    
    if len(images) == 0:
        print(f"No images found for {mineral} GIF creation")
        return
    
    # GIF 저장 (2초 = 2000ms per frame)
    gif_path = f'{output_dir}/{mineral}_animation.gif'
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=2000,  # 2초 간격
        loop=0  # 무한 반복
    )
    
    print(f"Generated: {mineral}_animation.gif")
    print(f"  Frames: {len(images)}")
    print(f"  Duration: 2 seconds per frame")
    print(f"  Total cycle time: {len(images) * 2} seconds")

def check_mineral_files():
    """미네랄 파일들의 존재 여부 확인"""
    mineral_dir = '/home/geofluids/research/FNO/src/initial_mineral/output'
    minerals = ['clinochlore', 'calcite', 'pyrite']
    
    print("Checking mineral files...")
    
    for mineral in minerals:
        print(f"\n{mineral.upper()} files:")
        available_files = []
        for i in range(10):  # 처음 10개 파일 체크
            mineral_file = f'{mineral_dir}/{mineral}_{i}.h5'
            if os.path.exists(mineral_file):
                available_files.append(i)
                print(f"  ✓ {mineral}_{i}.h5 exists")
            else:
                print(f"  ✗ {mineral}_{i}.h5 not found")
        
        print(f"  Found {len(available_files)} {mineral} files")
    
    return True

if __name__ == "__main__":
    # 먼저 파일 존재 여부 확인
    files_exist = check_mineral_files()
    
    if files_exist:
        print("\nProceeding with visualization...")
        visualize_mineral_maps()
    else:
        print("\nMineral files not found. Run initial_mineral.py first to generate mineral maps.")