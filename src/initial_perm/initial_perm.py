import numpy as np
import random
import h5py
from multiprocessing import Pool


def generate_scp_field(N, b, size_x, size_y, level_max, density_map_ratio, iteration):

    # Calculate density map dimensions based on ratio
    density_map_size_x = int(size_x * density_map_ratio)
    density_map_size_y = int(size_y * density_map_ratio)
    density_map = np.ones((density_map_size_x, density_map_size_y))  # (rows, cols) = (X, Y)

    def draw_squares(N, b, size_x, size_y, level=1, parent_coords=[(0, 0)]):
        
        if level > level_max:
            return
        
        # Calculate side_length as average of dimensions divided by b^level
        side_length = (size_x + size_y) / 2 / (b ** level)
        new_coords = []
        for px, py in parent_coords:
            for _ in range(N):
                if level == 1:  # First level: distribute across entire domain
                    x = random.uniform(0, size_x)
                    y = random.uniform(0, size_y)
                else:  # Subsequent levels: fractal subdivision
                    x = px + random.uniform(0, side_length * b)
                    y = py + random.uniform(0, side_length * b)

                # Handle wrapping for rectangular domain
                if x >= size_x:
                    x = x - size_x
                if y >= size_y:
                    y = y - size_y

                new_coords.append((x, y))

                if level == level_max:
                    update_density_map(density_map, x, y, side_length, size_x, size_y, density_map_size_x, density_map_size_y)
                    
        draw_squares(N, b, size_x, size_y, level + 1, new_coords)

    def update_density_map(density_map, x, y, side_length, size_x, size_y, density_map_size_x, density_map_size_y):

        for i in range(int(side_length)):
            for j in range(int(side_length)):
                
                # Calculate grid indices for rectangular domain
                xi = int((x + i) // (size_x / density_map_size_x))  # X coordinate -> row index
                yj = int((y + j) // (size_y / density_map_size_y))  # Y coordinate -> column index
                
                # Handle wrapping for rectangular density map
                if xi > density_map_size_x-1:
                    xi = xi - density_map_size_x
                if yj > density_map_size_y-1:
                    yj = yj - density_map_size_y

                density_map[xi, yj] += 1  # [X_row_index, Y_col_index]

    def generate_perm_map(density_map, iteration, size_x, size_y, density_map_size_x, density_map_size_y):
        
        average_aperature = np.random.uniform(0.00001, 0.0001)

        perm_dir = f"/home/geofluids/research/FNO/src/initial_perm/output/perm_map_{iteration}.h5"
        poro_dir = f"/home/geofluids/research/FNO/src/initial_perm/output/poro_map_{iteration}.h5"
        
        domain_size = density_map.shape[0] * density_map.shape[1]
        total_density = np.sum(density_map)
        aperature_ratio = average_aperature * domain_size / total_density
        aperature_map = np.minimum(0.005, aperature_ratio * density_map)

        perm_map = aperature_map ** 3 / 12.0 / 100.0 / 0.005
        poro_map = aperature_map / 0.005 + 0.01 * (1 - aperature_map / 0.005)
        
        data_shape = perm_map.shape
        data_dtype = perm_map.dtype
            
        with h5py.File(perm_dir, 'w') as hdf5_file:
            
            permsX_group = hdf5_file.create_group('permsX')

            data_dataset = permsX_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
            data_dataset[:, :] = perm_map
            permsX_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
            permsX_group.attrs['Discretization'] = [0.25, 0.25]
            permsX_group.attrs['Origin'] = [-8, -4]
            permsX_group.attrs['Cell Centered'] = [True]
        
        with h5py.File(poro_dir, 'w') as hdf5_file:

            porosX_group = hdf5_file.create_group('porosX')

            data_dataset = porosX_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
            data_dataset[:, :] = poro_map
            porosX_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
            porosX_group.attrs['Discretization'] = [0.25, 0.25]
            porosX_group.attrs['Origin'] = [-8, -4]
            porosX_group.attrs['Cell Centered'] = [True]
        
    draw_squares(N, b, size_x, size_y)
    generate_perm_map(density_map, iteration, size_x, size_y, density_map_size_x, density_map_size_y)


if __name__ == '__main__':
    
    map_num = 3000
    N = 9
    b = 2.64
    size_x = 256  # Width of rectangular domain
    size_y = 128  # Height of rectangular domain  
    level_max = 5
    density_map_ratio = 0.25  # Ratio for density map resolution (0.25 * 256 = 64)

    with Pool(30) as p:
        p.starmap(generate_scp_field, [(N, b, size_x, size_y, level_max, density_map_ratio, i) for i in range(map_num)])
        