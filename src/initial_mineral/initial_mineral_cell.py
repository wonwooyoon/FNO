import h5py
import numpy as np

# Path to the HDF5 file
file_path = '/home/geofluids/research/FNO/src/mesh/output/mesh.h5'

# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as hdf:
    # Read data from the specified paths
    domain_cells = hdf['/Domain/Cells'][:]
    domain_vertices = hdf['/Domain/Vertices'][:]
    materials_cell_ids = hdf['/Materials/Cell Ids'][:]
    materials_material_ids = hdf['/Materials/Material Ids'][:]

    # Find indices where materials_material_ids equals 1
    indices = np.where(materials_material_ids == 1)[0]

    # Initialize a list to store the averages

    values_calcite = np.zeros((len(materials_cell_ids), 2))
    values_calcite[:, 0] = materials_cell_ids
    values_pyrite = np.zeros((len(materials_cell_ids), 2))
    values_pyrite[:, 0] = materials_cell_ids

    for n in range(1000):
        for idx in indices:
            # Get the corresponding row in domain_cells
            cell_row = domain_cells[materials_cell_ids[idx]-1]
            averages = []
            
            # Iterate through the values in the cell_row
            for cell_value in cell_row[1:]:
                # Get the corresponding row in domain_vertices
                vertex_row = domain_vertices[cell_value - 1]
                # Append the mean to the averages list
                averages.append(vertex_row)
            
            # Convert averages to a numpy array for easier manipulation
            averages_array = np.array(averages)

            # Calculate the mean for x, y, z coordinates
            mean_x = (np.mean(averages_array[:, 0]) + 8.0) / 0.125
            mean_y = (np.mean(averages_array[:, 1]) + 4.0) / 0.125

            calcite_file_path = f'/home/geofluids/research/FNO/src/initial_mineral/output/calcite_{n}.h5'
            pyrite_file_path = f'/home/geofluids/research/FNO/src/initial_mineral/output/pyrite_{n}.h5'

            # Open the calcite HDF5 file and read data
            with h5py.File(calcite_file_path, 'r') as calcite_hdf:
                calcite_data = calcite_hdf['/calcite_mapX/Data'][:]

                # Get the value at the calculated mean_x and mean_y indices
                x_index = int(mean_x)
                y_index = int(mean_y)
                calcite_data[x_index, y_index]
                values_calcite[idx, 1] = calcite_data[x_index, y_index]

            # Open the pyrite HDF5 file and read data
            with h5py.File(pyrite_file_path, 'r') as pyrite_hdf:
                pyrite_data = pyrite_hdf['/pyrite_mapX/Data'][:]

                # Get the value at the calculated mean_x and mean_y indices
                x_index = int(mean_x)
                y_index = int(mean_y)
                pyrite_data[x_index, y_index]
                values_pyrite[idx, 1] = pyrite_data[x_index, y_index]
            

        with h5py.File(f'/home/geofluids/research/FNO/src/initial_mineral/output/calcite_cell_{n}.h5', 'w') as output_hdf:
            output_hdf.create_dataset('Cell Ids', data=values_calcite[:, 0].astype('int32'))
            output_hdf.create_dataset('calcite_cell', data=values_calcite[:, 1])

        with h5py.File(f'/home/geofluids/research/FNO/src/initial_mineral/output/pyrite_cell_{n}.h5', 'w') as output_hdf:
            output_hdf.create_dataset('Cell Ids', data=values_pyrite[:, 0].astype('int32'))
            output_hdf.create_dataset('pyrite_cell', data=values_pyrite[:, 1])

