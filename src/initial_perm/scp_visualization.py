import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_scp_method():
    """Stratified Continuum Percolation 방법의 사각형 생성 과정 시각화"""
    
    # 파라미터 설정
    N = 3  # 각 레벨에서 생성할 사각형 수
    b = 2  # 분할 비율
    size = 10  # 전체 도메인 크기
    level_max = 3  # 최대 레벨
    
    # 시각화 설정
    fig, axes = plt.subplots(1, level_max, figsize=(15, 5))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    def draw_rectangles_visual(level, parent_coords, ax):
        """각 레벨의 사각형들을 시각화"""
        side_length = size / (b ** level)
        new_coords = []
        
        for i, (px, py) in enumerate(parent_coords):
            # 부모 영역 표시 (점선)
            parent_size = size / (b ** (level-1)) if level > 1 else size
            parent_rect = plt.Rectangle((px, py), parent_size, parent_size, 
                                      fill=False, edgecolor='gray', linestyle='--', alpha=0.5)
            ax.add_patch(parent_rect)
            
            # N개의 사각형 생성
            for _ in range(N):
                # 부모 영역 내에서 무작위 위치 선택
                x = px + random.uniform(0, side_length * b)
                y = py + random.uniform(0, side_length * b)
                
                # 경계 조건 (wrapping)
                if x >= size:
                    x = x - size
                if y >= size:
                    y = y - size
                
                new_coords.append((x, y))
                
                # 사각형 그리기
                rect = plt.Rectangle((x, y), side_length, side_length, 
                                   fill=True, facecolor=colors[level-1], 
                                   alpha=0.7, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
        
        return new_coords
    
    # 각 레벨별로 시각화
    current_coords = [(0, 0)]  # 초기 좌표
    
    for level in range(1, level_max + 1):
        ax = axes[level - 1]
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Level {level}', fontsize=12)
        
        # 이전 레벨까지의 모든 사각형들을 다시 그리기
        temp_coords = [(0, 0)]
        for prev_level in range(1, level + 1):
            side_length = size / (b ** prev_level)
            temp_coords = draw_rectangles_visual(prev_level, temp_coords, ax)
    
    plt.tight_layout()
    plt.savefig('/home/geofluids/research/FNO/src/initial_perm/scp_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def visualize_stepwise_scp():
    """단계별 SCP 시각화 - 각 레벨을 누적으로 보여줌 + 밀도 맵"""
    
    N = 5  # 각 부모로부터 5개 박스 생성
    b = 2.64  # 크기 감소 비율
    size = 64  # 크기 증가 (마지막 레벨 박스 크기 확보)
    level_max = 3  # 3레벨 유지
    density_map_size = 32  # 밀도 맵 해상도
    
    # 색상 설정 (3레벨용)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 빨강, 청록, 파랑
    alphas = [0.4, 0.6, 0.8]  # 레벨별 투명도
    
    def draw_wrapped_rectangle(ax, x, y, width, height, color, alpha, linewidth):
        """래핑을 고려한 사각형 그리기"""
        rectangles = []
        
        # 기본 사각형
        if x + width <= size and y + height <= size:
            # 완전히 도메인 내부
            rect = plt.Rectangle((x, y), width, height, 
                               fill=True, facecolor=color, alpha=alpha, 
                               edgecolor='black', linewidth=linewidth)
            rectangles.append(rect)
        else:
            # 래핑 필요
            # 우하단 부분 (기본)
            w1 = min(width, size - x)
            h1 = min(height, size - y)
            if w1 > 0 and h1 > 0:
                rect = plt.Rectangle((x, y), w1, h1, 
                                   fill=True, facecolor=color, alpha=alpha, 
                                   edgecolor='black', linewidth=linewidth)
                rectangles.append(rect)
            
            # 우상단 부분 (y 래핑)
            if y + height > size and w1 > 0:
                h2 = (y + height) - size
                rect = plt.Rectangle((x, 0), w1, h2, 
                                   fill=True, facecolor=color, alpha=alpha, 
                                   edgecolor='black', linewidth=linewidth)
                rectangles.append(rect)
            
            # 좌하단 부분 (x 래핑)
            if x + width > size and h1 > 0:
                w2 = (x + width) - size
                rect = plt.Rectangle((0, y), w2, h1, 
                                   fill=True, facecolor=color, alpha=alpha, 
                                   edgecolor='black', linewidth=linewidth)
                rectangles.append(rect)
            
            # 좌상단 부분 (x, y 모두 래핑)
            if x + width > size and y + height > size:
                w2 = (x + width) - size
                h2 = (y + height) - size
                rect = plt.Rectangle((0, 0), w2, h2, 
                                   fill=True, facecolor=color, alpha=alpha, 
                                   edgecolor='black', linewidth=linewidth)
                rectangles.append(rect)
        
        for rect in rectangles:
            ax.add_patch(rect)
    
    def update_density_map(density_map, x, y, side_length, size, density_map_size):
        """원본 initial_perm.py와 동일한 밀도 맵 업데이트 - 박스 내부 모든 픽셀에 1 추가"""
        for i in range(int(side_length)):
            for j in range(int(side_length)):
                # 박스 내부의 실제 좌표 (래핑 적용)
                actual_x = (x + i) % size
                actual_y = (y + j) % size
                
                # 밀도 맵 인덱스로 변환
                xi = int(actual_x // (size / density_map_size))
                yj = int(actual_y // (size / density_map_size))
                
                # 경계 체크 (밀도 맵 크기 내에서)
                xi = min(xi, density_map_size - 1)
                yj = min(yj, density_map_size - 1)
                
                density_map[yj, xi] += 1
    
    # 밀도 맵 초기화
    density_map = np.ones((density_map_size, density_map_size))
    
    # 모든 레벨의 좌표를 저장할 리스트
    all_level_coords = []
    current_coords = [(0, 0)]  # 초기 좌표
    
    # 각 레벨별로 좌표 생성
    for level in range(1, level_max + 1):
        side_length = size / (b ** level)  # 현재 레벨의 사각형 크기
        new_coords = []
        
        for px, py in current_coords:
            for _ in range(N):
                # 부모 영역 내에서 무작위 위치 선택
                search_range = side_length * b  # 부모 크기의 b배 범위
                x = px + random.uniform(0, search_range)
                y = py + random.uniform(0, search_range)
                
                # 래핑 처리
                x = x % size
                y = y % size
                
                new_coords.append((x, y))
        
        all_level_coords.append((new_coords.copy(), side_length, level))
        current_coords = new_coords
        
        # 마지막 레벨에서 밀도 맵 업데이트
        if level == level_max:
            for x, y in new_coords:
                update_density_map(density_map, x, y, side_length, size, density_map_size)
    
    # 각 단계별로 이미지 생성
    for step in range(1, level_max + 1):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 현재 단계까지의 모든 레벨 그리기
        for level_idx in range(step):
            coords, side_length, level = all_level_coords[level_idx]
            
            for x, y in coords:
                draw_wrapped_rectangle(ax, x, y, side_length, side_length, 
                                     colors[level-1], alphas[level-1], 
                                     4-level)  # 레벨이 높을수록 얇은 선
        
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'/home/geofluids/research/FNO/src/initial_perm/scp_level_{step}.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Generated: scp_level_{step}.png")
    
    # 밀도 맵 시각화
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 밀도 맵을 컬러맵으로 표시
    im = ax.imshow(density_map, cmap='viridis', origin='lower', 
                   extent=[0, size, 0, size], interpolation='nearest')
    
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/home/geofluids/research/FNO/src/initial_perm/scp_density_map.png', 
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Generated: scp_density_map.png")
    
    print("All step-by-step images and density map generated!")

if __name__ == "__main__":
    random.seed(42)  # 재현 가능한 결과를 위해
    visualize_stepwise_scp()