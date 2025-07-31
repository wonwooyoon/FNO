import numpy as np
import random
import h5py
from multiprocessing import Pool


def generate_scp_field(N, b, size, level_max, density_map_size, iteration):

    density_map = np.ones((density_map_size, density_map_size))

    def draw_squares(N, b, size, level=1, parent_coords=[(0, 0)]):
        
        if level > level_max:
            return
        
        side_length = size / (b ** level)
        new_coords = []
        for px, py in parent_coords:
            for _ in range(N):
                x = px + random.uniform(0, side_length * b)
                y = py + random.uniform(0, side_length * b)

                if x >= size:
                    x = x - size
                if y >= size:
                    y = y - size

                new_coords.append((x, y))

                if level == level_max:
                    update_density_map(density_map, x, y, side_length, size)
                    
        draw_squares(N, b, size, level + 1, new_coords)

    def update_density_map(density_map, x, y, side_length, size):

        for i in range(int(side_length)):
            for j in range(int(side_length)):
                xi = int((x + i) // (size / density_map_size))
                yj = int((y + j) // (size / density_map_size))
                density_map[yj, xi] += 1

    def generate_perm_map(density_map, iteration):
        
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
            permsX_group.attrs['Discretization'] = [0.125, 0.125]
            permsX_group.attrs['Origin'] = [-8.0, -4.0]
            permsX_group.attrs['Cell Centered'] = [True]
        
        with h5py.File(poro_dir, 'w') as hdf5_file:

            porosX_group = hdf5_file.create_group('porosX')

            data_dataset = porosX_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
            data_dataset[:, :] = poro_map
            porosX_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
            porosX_group.attrs['Discretization'] = [0.125, 0.125]
            porosX_group.attrs['Origin'] = [-8.0, -4.0]
            porosX_group.attrs['Cell Centered'] = [True]
        
    draw_squares(N, b, size)
    generate_perm_map(density_map, iteration)


if __name__ == '__main__':
    
    map_num = 1000
    N = 9
    b = 2.64
    size = 256
    level_max = 5
    density_map_size = 128

    with Pool(10) as p:
        p.starmap(generate_scp_field, [(N, b, size, level_max, density_map_size, i) for i in range(map_num)])
        