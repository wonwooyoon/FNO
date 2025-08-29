import h5py
import numpy as np
import os
import time
import gc

def load_mesh_data(mesh_file_path):
    """메쉬 데이터를 로드하고 좌표를 사전 계산"""
    print("Loading mesh data...")
    with h5py.File(mesh_file_path, 'r') as hdf:
        domain_cells = hdf['/Domain/Cells'][:]
        domain_vertices = hdf['/Domain/Vertices'][:]
        materials_cell_ids = hdf['/Materials/Cell Ids'][:]
        materials_material_ids = hdf['/Materials/Material Ids'][:]
        
        # material_id가 1인 인덱스 찾기
        indices = np.where(materials_material_ids == 1)[0]
        
        print(f"Found {len(indices)} cells with material_id = 1")
        
        # 좌표 사전 계산
        coordinates = precompute_coordinates(domain_cells, domain_vertices, materials_cell_ids, indices)
        
        return indices, materials_cell_ids, coordinates

def precompute_coordinates(domain_cells, domain_vertices, materials_cell_ids, indices):
    """모든 셀의 좌표를 한 번에 계산"""
    print("Precomputing coordinates...")
    coordinates = np.zeros((len(indices), 2), dtype=np.int32)
    
    for i, idx in enumerate(indices):
        # 셀 정보 가져오기
        cell_row = domain_cells[materials_cell_ids[idx] - 1]
        # 버텍스 좌표들 가져오기
        vertices = domain_vertices[cell_row[1:] - 1]
        # 평균 좌표 계산 및 변환
        mean_x = int((np.mean(vertices[:, 0]) + 8.0) / 0.125)
        mean_y = int((np.mean(vertices[:, 1]) + 4.0) / 0.125)
        coordinates[i] = [mean_x, mean_y]
    
    print(f"Precomputed coordinates for {len(indices)} cells")
    return coordinates

def process_batch(mineral_output_dir, indices, materials_cell_ids, coordinates, start_n, end_n):
    """배치 단위로 미네랄 데이터 처리"""
    mineral_names = ['clinochlore', 'calcite', 'pyrite']
    
    print(f"Processing batch: {start_n} to {end_n-1}")
    
    for n in range(start_n, end_n):
        if n % 10 == 0:
            print(f"  Processing n={n}")
        
        # 결과 배열 초기화
        results = {}
        for mineral in mineral_names:
            results[mineral] = np.zeros((len(indices), 2), dtype=np.float32)
            results[mineral][:, 0] = materials_cell_ids[indices]  # Cell IDs
        
        # 각 미네랄 데이터 처리
        for mineral in mineral_names:
            file_path = f'{mineral_output_dir}/{mineral}_{n}.h5'
            
            if os.path.exists(file_path):
                try:
                    with h5py.File(file_path, 'r') as hdf:
                        data = hdf[f'/{mineral}_mapX/Data'][:]
                        
                        # 벡터화된 인덱싱으로 값 추출
                        x_coords = coordinates[:, 0]
                        y_coords = coordinates[:, 1]
                        
                        # 경계 체크
                        valid_mask = (
                            (x_coords >= 0) & (x_coords < data.shape[0]) &
                            (y_coords >= 0) & (y_coords < data.shape[1])
                        )
                        
                        # 유효한 인덱스에서 값 추출
                        results[mineral][valid_mask, 1] = data[x_coords[valid_mask], y_coords[valid_mask]]
                        
                except Exception as e:
                    print(f"Warning: Error processing {file_path}: {e}")
                    continue
            else:
                print(f"Warning: File not found: {file_path}")
        
        # 결과 저장
        save_results(mineral_output_dir, results, n)

def save_results(output_dir, results, n):
    """결과를 HDF5 파일로 저장 (압축 적용)"""
    mineral_names = ['clinochlore', 'calcite', 'pyrite']
    
    for mineral in mineral_names:
        output_file = f'{output_dir}/{mineral}_cell_{n}.h5'
        
        try:
            with h5py.File(output_file, 'w') as output_hdf:
                # 압축 적용하여 저장
                output_hdf.create_dataset(
                    'Cell Ids', 
                    data=results[mineral][:, 0].astype('int32'),
                    compression='gzip', compression_opts=6
                )
                output_hdf.create_dataset(
                    f'{mineral}_cell', 
                    data=results[mineral][:, 1],
                    compression='gzip', compression_opts=6
                )
        except Exception as e:
            print(f"Error saving {output_file}: {e}")

def process_all_data(mesh_file_path, mineral_output_dir, total_files=3000, batch_size=50):
    """전체 데이터 처리 메인 함수"""
    print("=" * 60)
    print("OPTIMIZED MINERAL CELL PROCESSING")
    print("=" * 60)
    
    # 메쉬 데이터 로드
    indices, materials_cell_ids, coordinates = load_mesh_data(mesh_file_path)
    
    # 배치 단위로 처리
    print(f"Processing {total_files} files in batches of {batch_size}")
    
    for start in range(0, total_files, batch_size):
        end = min(start + batch_size, total_files)
        
        print(f"\nBatch {start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        
        try:
            process_batch(mineral_output_dir, indices, materials_cell_ids, coordinates, start, end)
            
            # 메모리 정리
            if start > 0 and start % (batch_size * 5) == 0:  # 5개 배치마다
                gc.collect()
                print(f"  Memory cleanup completed")
                
        except Exception as e:
            print(f"Error processing batch {start}-{end}: {e}")
            print("Continuing with next batch...")
            continue

def main():
    """메인 함수 - 전체 워크플로우"""
    # 설정
    mesh_file_path = '/home/geofluids/research/FNO/src/mesh/output/mesh.h5'
    mineral_output_dir = '/home/geofluids/research/FNO/src/initial_mineral/output'
    
    # 파일 존재 확인
    if not os.path.exists(mesh_file_path):
        print(f"Error: Mesh file not found: {mesh_file_path}")
        return
    
    if not os.path.exists(mineral_output_dir):
        print(f"Error: Mineral output directory not found: {mineral_output_dir}")
        return
    
    # 처리 시작
    start_time = time.time()
    
    try:
        process_all_data(mesh_file_path, mineral_output_dir, total_files=3000, batch_size=50)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETED!")
        print(f"Total time: {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
        print(f"Processing rate: {3000/processing_time:.2f} files/second")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        end_time = time.time()
        print(f"Total execution time: {(end_time - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()