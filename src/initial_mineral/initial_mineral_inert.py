import h5py
import numpy as np
import os


for n in range(1000):

    # Define file paths
    output_dir = "/home/geofluids/research/FNO/src/initial_mineral/output"
    calcite_file = os.path.join(output_dir, f"calcite_cell_{n}.h5")
    pyrite_file = os.path.join(output_dir, f"pyrite_cell_{n}.h5")
    inert_file = os.path.join(output_dir, f"inert_cell_{n}.h5")

    # Read data from calcite_cell_n.h5 and pyrite_cell_n.h5
    with h5py.File(calcite_file, "r") as calcite_h5, h5py.File(pyrite_file, "r") as pyrite_h5:
        calcite_cell = calcite_h5["calcite_cell"][:]
        pyrite_cell = pyrite_h5["pyrite_cell"][:]
        cell_ids = calcite_h5["Cell Ids"][:]

    # Ensure the data shapes match
    if calcite_cell.shape != pyrite_cell.shape:
        raise ValueError("The shapes of calcite_cell and pyrite_cell do not match.")

    # Compute inert and 0.7 - inert
    inert = calcite_cell + pyrite_cell
    inert = 0.7 - inert

    # Save the new data to inert_cell_n.h5
    with h5py.File(inert_file, "w") as inert_h5:
        inert_h5.create_dataset("inert_cell", data=inert)
        inert_h5.create_dataset("Cell Ids", data=cell_ids)
