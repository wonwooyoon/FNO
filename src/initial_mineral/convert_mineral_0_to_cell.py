#!/usr/bin/env python3
"""
Convert mineral_0 maps (LR and HR) to cell format

This script converts the upscaled mineral_0.h5 files to cell-based format
for use with PFLOTRAN simulations.
"""

import h5py
import numpy as np
import os


def load_mesh_data(mesh_file_path):
    """
    Load mesh data and precompute coordinates

    Args:
        mesh_file_path: Path to mesh.h5 file

    Returns:
        material_1_indices: Indices of cells with material_id=1
        materials_cell_ids: Cell IDs for all materials
        coordinates: Dictionary of precomputed coordinates
    """
    print(f"Loading mesh data from: {mesh_file_path}")

    with h5py.File(mesh_file_path, 'r') as hdf:
        domain_cells = hdf['/Domain/Cells'][:]
        domain_vertices = hdf['/Domain/Vertices'][:]
        materials_cell_ids = hdf['/Materials/Cell Ids'][:]
        materials_material_ids = hdf['/Materials/Material Ids'][:]

        # Find indices where material_id == 1
        material_1_indices = np.where(materials_material_ids == 1)[0]

        print(f"  Total cells: {len(materials_cell_ids)}")
        print(f"  Cells with material_id=1: {len(material_1_indices)}")

        # Precompute coordinates for material_id=1 cells
        coordinates = precompute_coordinates(
            domain_cells, domain_vertices, materials_cell_ids, material_1_indices
        )

        return material_1_indices, materials_cell_ids, coordinates


def precompute_coordinates(domain_cells, domain_vertices, materials_cell_ids,
                          material_1_indices, discretization=0.125):
    """
    Precompute coordinates for material_id=1 cells

    Args:
        domain_cells: Cell connectivity
        domain_vertices: Vertex coordinates
        materials_cell_ids: Cell IDs
        material_1_indices: Indices of material_id=1 cells
        discretization: Grid spacing (0.125 for HR, 0.25 for LR)

    Returns:
        coordinates: Dictionary mapping index to (x, y) grid coordinates
    """
    print(f"  Precomputing coordinates (discretization={discretization})...")
    coordinates = {}

    for idx in material_1_indices:
        # Get cell information
        cell_row = domain_cells[materials_cell_ids[idx] - 1]

        # Get vertex coordinates
        vertices = domain_vertices[cell_row[1:] - 1]

        # Calculate mean coordinates and convert to grid indices
        mean_x = int((np.mean(vertices[:, 0]) + 8.0) / discretization)
        mean_y = int((np.mean(vertices[:, 1]) + 4.0) / discretization)

        coordinates[idx] = (mean_x, mean_y)

    print(f"  Precomputed {len(coordinates)} coordinates")
    return coordinates


def convert_mineral_to_cell(mineral_file_path, mineral_name, mesh_material_1_indices,
                            mesh_materials_cell_ids, mesh_coordinates):
    """
    Convert one mineral map to cell format

    Args:
        mineral_file_path: Path to mineral .h5 file
        mineral_name: Name of mineral (calcite, clinochlore, pyrite)
        mesh_material_1_indices: Indices of material_id=1 cells
        mesh_materials_cell_ids: Cell IDs for all materials
        mesh_coordinates: Precomputed coordinates

    Returns:
        cell_data: (N, 2) array with [cell_id, mineral_value]
    """
    print(f"  Converting {mineral_name} to cell format...")

    # Read mineral map
    group_name = f'{mineral_name}_mapX'

    with h5py.File(mineral_file_path, 'r') as hdf:
        data = hdf[group_name]['Data'][:]

    print(f"    Map shape: {data.shape}")

    # Initialize result array (all cells)
    cell_data = np.zeros((len(mesh_materials_cell_ids), 2))
    cell_data[:, 0] = mesh_materials_cell_ids

    # Vectorized coordinate extraction
    coord_indices = []
    coord_values = []

    for idx in mesh_material_1_indices:
        if idx in mesh_coordinates:
            coord_indices.append(idx)
            coord_values.append(mesh_coordinates[idx])

    coord_indices = np.array(coord_indices)
    coord_values = np.array(coord_values)

    # Vectorized boundary check
    x_coords = coord_values[:, 0]
    y_coords = coord_values[:, 1]

    valid_mask = (
        (x_coords >= 0) & (x_coords < data.shape[0]) &
        (y_coords >= 0) & (y_coords < data.shape[1])
    )

    # Extract values for valid coordinates
    valid_indices = coord_indices[valid_mask]
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]

    cell_data[valid_indices, 1] = data[valid_x, valid_y]

    print(f"    Assigned values to {np.sum(valid_mask)} cells")

    return cell_data


def save_cell_data(output_file_path, mineral_name, cell_data):
    """
    Save cell data to HDF5 file

    Args:
        output_file_path: Output file path
        mineral_name: Name of mineral
        cell_data: (N, 2) array with [cell_id, mineral_value]
    """
    with h5py.File(output_file_path, 'w') as hdf:
        hdf.create_dataset('Cell Ids', data=cell_data[:, 0].astype('int32'))
        hdf.create_dataset(f'{mineral_name}_cell', data=cell_data[:, 1])

    print(f"    Saved to: {output_file_path}")


def process_resolution(resolution_name, mineral_input_dir, mesh_file_path, output_dir):
    """
    Process one resolution (LR or HR)

    Args:
        resolution_name: 'LR' or 'HR'
        mineral_input_dir: Directory containing mineral_0.h5 files
        mesh_file_path: Path to mesh.h5 file
        output_dir: Output directory for cell files
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING {resolution_name}")
    print("=" * 80)

    # Determine discretization
    discretization = 0.25 if resolution_name == 'LR' else 0.125

    # Load mesh data
    material_1_indices, materials_cell_ids, coordinates = load_mesh_data(mesh_file_path)

    # Note: coordinates need to be recomputed for different discretization
    # The mesh is fixed, but grid indices change based on discretization
    print(f"\nRecomputing coordinates for discretization={discretization}...")

    with h5py.File(mesh_file_path, 'r') as hdf:
        domain_cells = hdf['/Domain/Cells'][:]
        domain_vertices = hdf['/Domain/Vertices'][:]

        coordinates = {}
        for idx in material_1_indices:
            cell_row = domain_cells[materials_cell_ids[idx] - 1]
            vertices = domain_vertices[cell_row[1:] - 1]
            mean_x = int((np.mean(vertices[:, 0]) + 8.0) / discretization)
            mean_y = int((np.mean(vertices[:, 1]) + 4.0) / discretization)
            coordinates[idx] = (mean_x, mean_y)

    # Process each mineral
    minerals = ['calcite', 'clinochlore', 'pyrite']

    for mineral in minerals:
        print(f"\nProcessing {mineral}...")

        input_file = os.path.join(mineral_input_dir, f"{mineral}_0.h5")

        if not os.path.exists(input_file):
            print(f"  ⚠ File not found: {input_file}")
            continue

        # Convert to cell format
        cell_data = convert_mineral_to_cell(
            input_file, mineral, material_1_indices,
            materials_cell_ids, coordinates
        )

        # Save
        output_file = os.path.join(output_dir, f"{mineral}_cell_0.h5")
        save_cell_data(output_file, mineral, cell_data)

        # Verify
        non_zero = np.sum(cell_data[:, 1] > 0)
        print(f"    ✓ Non-zero cells: {non_zero}/{len(cell_data)}")
        print(f"    ✓ Value range: [{cell_data[:, 1].min():.6f}, {cell_data[:, 1].max():.6f}]")


def main():
    """
    Main function
    """
    print("\n" + "#" * 80)
    print("# CONVERT MINERAL_0 MAPS TO CELL FORMAT")
    print("#" * 80)

    base_dir = "/home/geofluids/research/FNO/src/initial_mineral"

    # LR processing
    lr_input_dir = os.path.join(base_dir, "output_lr")
    lr_mesh_file = "/home/geofluids/research/FNO/src/mesh/output_test_lr/mesh.h5"
    lr_output_dir = lr_input_dir

    if os.path.exists(lr_mesh_file):
        process_resolution("LR", lr_input_dir, lr_mesh_file, lr_output_dir)
    else:
        print(f"\n⚠ LR mesh not found: {lr_mesh_file}")
        print("  Skipping LR processing")

    # HR processing
    hr_input_dir = os.path.join(base_dir, "output_hr")
    hr_mesh_file = "/home/geofluids/research/FNO/src/mesh/output_test_hr/mesh.h5"
    hr_output_dir = hr_input_dir

    if os.path.exists(hr_mesh_file):
        process_resolution("HR", hr_input_dir, hr_mesh_file, hr_output_dir)
    else:
        print(f"\n⚠ HR mesh not found: {hr_mesh_file}")
        print("  Skipping HR processing")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
