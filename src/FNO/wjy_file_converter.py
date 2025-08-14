import numpy as np
import os

def csv_to_npy(csv_path, npy_path=None, delimiter=','):
    # Load all data from CSV (including the first row)
    data = np.loadtxt(csv_path, delimiter=delimiter)
    # Set npy_path if not provided
    if npy_path is None:
        npy_path = os.path.splitext(csv_path)[0] + '.npy'
    # Save as .npy
    np.save(npy_path, data)
    print(f"Saved numpy array to {npy_path}")
    print(f"Shape of the array: {data.shape}")

# Example usage: 
csv_to_npy('./src/preprocessing/input.csv')