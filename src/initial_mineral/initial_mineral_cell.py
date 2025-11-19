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
        material_1_indices = np.where(materials_material_ids == 1)[0]
        
        print(f"Total cells: {len(materials_cell_ids)}")
        print(f"Cells with material_id = 1: {len(material_1_indices)}")
        
        # 모든 셀의 좌표 사전 계산 (material_id=1인 경우만)
        coordinates = precompute_coordinates(domain_cells, domain_vertices, materials_cell_ids, material_1_indices)
        
        return material_1_indices, materials_cell_ids, materials_material_ids, coordinates

def precompute_coordinates(domain_cells, domain_vertices, materials_cell_ids, material_1_indices):
    """material_id=1인 셀들의 좌표를 계산"""
    print("Precomputing coordinates for material_id=1 cells...")
    coordinates = {}
    
    for idx in material_1_indices:
        # 셀 정보 가져오기
        cell_row = domain_cells[materials_cell_ids[idx] - 1]
        # 버텍스 좌표들 가져오기
        vertices = domain_vertices[cell_row[1:] - 1]
        # 평균 좌표 계산 및 변환
        mean_x = int((np.mean(vertices[:, 0]) + 8.0) / 0.125)
        mean_y = int((np.mean(vertices[:, 1]) + 4.0) / 0.125)
        coordinates[idx] = (mean_x, mean_y)
    
    print(f"Precomputed coordinates for {len(material_1_indices)} material_id=1 cells")
    return coordinates

def process_batch(mineral_output_dir, material_1_indices, materials_cell_ids, materials_material_ids, coordinates, start_n, end_n):
    """배치 단위로 미네랄 데이터 처리 - 벡터화 최적화"""
    mineral_names = ['clinochlore', 'calcite', 'pyrite']
    
    print(f"Processing batch: {start_n} to {end_n-1}")
    
    # 좌표 배열 미리 준비 (벡터화를 위해)
    coord_indices = []
    coord_values = []
    for idx in material_1_indices:
        if idx in coordinates:
            coord_indices.append(idx)
            coord_values.append(coordinates[idx])
    
    coord_indices = np.array(coord_indices)
    coord_values = np.array(coord_values)
    
    for n in range(start_n, end_n):
        if n % 10 == 0:
            print(f"  Processing n={n}")
        
        # 원본과 동일하게 모든 셀에 대한 배열 초기화
        values_clinochlore = np.zeros((len(materials_cell_ids), 2))
        values_clinochlore[:, 0] = materials_cell_ids
        values_calcite = np.zeros((len(materials_cell_ids), 2))
        values_calcite[:, 0] = materials_cell_ids
        values_pyrite = np.zeros((len(materials_cell_ids), 2))
        values_pyrite[:, 0] = materials_cell_ids
        
        results = {
            'clinochlore': values_clinochlore,
            'calcite': values_calcite,
            'pyrite': values_pyrite
        }
        
        # 각 미네랄 파일을 벡터화로 처리
        for mineral in mineral_names:
            file_path = f'{mineral_output_dir}/{mineral}_{n}.h5'
            
            if os.path.exists(file_path):
                try:
                    with h5py.File(file_path, 'r') as hdf:
                        data = hdf[f'/{mineral}_mapX/Data'][:]
                        
                        # 벡터화된 경계 체크
                        x_coords = coord_values[:, 0]
                        y_coords = coord_values[:, 1]
                        valid_mask = (
                            (x_coords >= 0) & (x_coords < data.shape[0]) &
                            (y_coords >= 0) & (y_coords < data.shape[1])
                        )
                        
                        # 유효한 좌표에서 벡터화된 값 추출
                        valid_indices = coord_indices[valid_mask]
                        valid_x = x_coords[valid_mask]
                        valid_y = y_coords[valid_mask]
                        
                        results[mineral][valid_indices, 1] = data[valid_x, valid_y]
                        
                except Exception as e:
                    print(f"Warning: Error processing {file_path}: {e}")
                    continue
            else:
                print(f"Warning: File not found: {file_path}")
        
        # 결과 저장
        save_results(mineral_output_dir, results, n)

def save_results(output_dir, results, n):
    """결과를 HDF5 파일로 저장 - 원본과 동일한 형식"""
    mineral_names = ['clinochlore', 'calcite', 'pyrite']
    
    for mineral in mineral_names:
        output_file = f'{output_dir}/{mineral}_cell_{n}.h5'
        
        try:
            with h5py.File(output_file, 'w') as output_hdf:
                # 원본과 동일하게 저장 (압축 없이)
                output_hdf.create_dataset('Cell Ids', data=results[mineral][:, 0].astype('int32'))
                output_hdf.create_dataset(f'{mineral}_cell', data=results[mineral][:, 1])
        except Exception as e:
            print(f"Error saving {output_file}: {e}")

def process_all_data(mesh_file_path, mineral_output_dir, total_files=3000, batch_size=50):
    """전체 데이터 처리 메인 함수"""
    print("=" * 60)
    print("FAST OPTIMIZED MINERAL CELL PROCESSING")
    print("=" * 60)
    
    # 메쉬 데이터 로드
    material_1_indices, materials_cell_ids, materials_material_ids, coordinates = load_mesh_data(mesh_file_path)
    
    # 배치 단위로 처리
    print(f"Processing {total_files} files in batches of {batch_size}")
    print(f"Output will include all {len(materials_cell_ids)} cells (values only for material_id=1)")
    
    for start in range(0, total_files, batch_size):
        end = min(start + batch_size, total_files)
        
        print(f"\nBatch {start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        
        try:
            process_batch(mineral_output_dir, material_1_indices, materials_cell_ids, materials_material_ids, coordinates, start, end)
            
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
    mesh_file_path = '/home/geofluids/research/FNO/src/mesh/output_hr/mesh.h5'
    mineral_output_dir = '/home/geofluids/research/FNO/src/initial_mineral/output_hr'
    
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
        process_all_data(mesh_file_path, mineral_output_dir, total_files=100, batch_size=50)
        
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