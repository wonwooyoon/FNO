# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a machine learning pipeline for groundwater uranium transport modeling using Fourier Neural Operators (FNO). The project combines PFLOTRAN-based geochemical simulations with neural operator training to predict uranium migration patterns in subsurface environments.

## Key Components

### Neural Operator Library (`neuraloperator/`)
- Contains the NeuralOperator library - a comprehensive PyTorch implementation for learning neural operators
- Provides FNO, TFNO, and other neural operator architectures
- Includes training utilities, data loaders, and loss functions
- Has its own test suite in `neuralop/*/tests/`

### Source Code (`src/`)
- **FNO/**: Main FNO training scripts with Optuna hyperparameter optimization
- **preprocessing/**: Data preprocessing and merging utilities for PFLOTRAN outputs
- **initial_perm/**: Generates initial permeability and porosity maps using stochastic processes
- **initial_mineral/**: Creates initial mineral distribution maps
- **initial_others/**: Handles other simulation parameters
- **pflotran_code/**: Generates PFLOTRAN input files
- **pflotran_run/**: Executes PFLOTRAN simulations in parallel
- **CNN/**: Alternative CNN-based models for comparison

## Common Development Tasks

### Training FNO Models
```bash
# Navigate to FNO directory and run training
cd src/FNO
python FNO.py
```

### Data Preprocessing
```bash
# Process PFLOTRAN output files
cd src/preprocessing
python preprocessing_revised.py --input_dir <pflotran_output_dir> --output <output_file>

# Merge preprocessed data shards
python preprocessing_merge.py
```

### Running PFLOTRAN Simulations
```bash
# Generate input files first
cd src/pflotran_code
python generate_input.py

# Run simulations
cd ../pflotran_run
python RunPFLOTRAN.py
```

### Testing Neural Operator Components
```bash
# Run tests for the neuraloperator library
cd neuraloperator
python -m pytest neuralop/*/tests/ -v
```

## Architecture Details

### Data Pipeline Flow
1. **Initial Conditions Generation**: Scripts in `initial_*` directories create stochastic initial conditions (permeability, minerals, etc.)
2. **PFLOTRAN Input Generation**: `pflotran_code/generate_input.py` creates simulation input files
3. **PFLOTRAN Execution**: `pflotran_run/RunPFLOTRAN.py` runs parallel geochemical simulations
4. **Data Preprocessing**: `preprocessing/preprocessing_revised.py` converts PFLOTRAN HDF5 outputs to ML-ready tensors
5. **Data Merging**: `preprocessing/preprocessing_merge.py` combines multiple simulation results
6. **Model Training**: `FNO/FNO.py` trains neural operators with hyperparameter optimization

### FNO Training Configuration
The main FNO training script uses a centralized CONFIG dictionary containing:
- Model hyperparameters (channels, layers, modes)
- Training settings (epochs, batch size, learning rates)
- Data paths and output directories
- Visualization parameters

### Key Data Structures
- **Input tensors**: (N, 9, nx, ny, nt) - permeability, minerals, velocities over time
- **Output tensors**: (N, 1, nx, ny, nt) - uranium concentration predictions
- **Meta tensors**: (N, 2) - additional metadata for FiLM conditioning

## Dependencies and Setup

### Core Dependencies
- PyTorch for neural networks
- NeuralOperator library (included in repo)
- PFLOTRAN for geochemical simulations
- Optuna for hyperparameter optimization
- h5py for HDF5 file handling
- matplotlib for visualization

### Installation
```bash
# Install neuraloperator library in development mode
cd neuraloperator
pip install -e .
pip install -r requirements.txt

# The PFLOTRAN_DIR environment variable should be set to your PFLOTRAN installation
export PFLOTRAN_DIR=/path/to/pflotran
```

## File Organization Patterns

### Output Directories
- `src/FNO/output/`: Training outputs, model checkpoints, comparison plots
- `src/initial_perm/output/`: Generated permeability/porosity maps (HDF5 format)
- `src/preprocessing/`: Merged tensor data files (.pt format)

### Naming Conventions
- HDF5 files: `perm_map_{id}.h5`, `poro_map_{id}.h5`
- PyTorch tensors: `input_output_com{N}.pt`, `merged.pt`
- Model checkpoints: `best_model_state_dict.pt`

## Development Notes

### Model Training
- The FNO implementation supports both single training and Optuna-based hyperparameter search
- Training uses early stopping and saves best models based on validation loss
- Visualization includes ground truth vs prediction comparisons across multiple time steps

### Data Preprocessing
- PFLOTRAN outputs are processed to extract 2D slices at Z=0
- Log transformation is applied to uranium concentrations
- Data normalization uses UnitGaussianNormalizer from the neuralop library

### Parallel Execution
- PFLOTRAN simulations run with MPI parallelization (default: 36 processes)
- Multiple simulation cases can be processed in batch using the provided scripts

### Memory Considerations
- Large tensor files are stored as .pt format for efficient loading
- Data loaders use appropriate batch sizing based on available GPU memory
- The preprocessing pipeline handles data in chunks to manage memory usage