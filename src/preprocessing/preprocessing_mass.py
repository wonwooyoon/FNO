#!/usr/bin/env python3
"""
PFLOTRAN Mass Balance Collection

Collect mass balance data from all PFLOTRAN simulations (local and remote servers)
and save to 10 separate CSV files (one per variable).

Usage:
    # Local only
    python preprocessing_mass.py --local-only

    # Local + remote servers
    python preprocessing_mass.py --config ../../servers.yaml
"""

import argparse
import sys
import re
import pickle
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import yaml
import h5py
import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CONFIG = {
    'pflotran_dir': PROJECT_ROOT / 'src/pflotran_run/output',
    'output_dir': SCRIPT_DIR,  # Save CSVs to src/preprocessing/
    'final_timestep': '2000.0000yr',
    'preprocessing_script': 'preprocessing_mass.py'
}


def extract_time_from_key(key: str) -> float:
    """
    Extract time value in years from HDF5 group key

    Args:
        key: HDF5 group key (e.g., "   0 Time  0.00000E+00 y")

    Returns:
        Time value in years
    """
    import re
    match = re.search(r'Time\s+([\d.E+-]+)\s*y', key)
    if match:
        return float(match.group(1))
    return 0.0


def calculate_mass_balance(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate uranium adsorption and dissolution over time, separated by domain

    Args:
        h5_path: Path to PFLOTRAN HDF5 output file

    Returns:
        times: Time points in years (nt,)
        source_aqueous: Aqueous U in source region (Material ID = 3) in mol (nt,)
        nonsource_aqueous: Aqueous U in non-source region (Material ID ≠ 3) in mol (nt,)
        adsorbed: Total adsorbed uranium in mol (nt,)
        dissolved: Total dissolved UO2 volume in m^3 (nt,)
    """
    with h5py.File(h5_path, 'r') as f:
        # Get all timestep groups
        timestep_keys = [key for key in f.keys() if 'Time' in key]
        timestep_keys = sorted(timestep_keys)  # Sort by timestep number

        # Load Material ID (constant across time)
        first_key = timestep_keys[0]
        material_id = np.array(f[first_key]['Material ID'][:])
        source_mask = (material_id == 3)

        print(f"Domain info: {source_mask.sum()} source cells, {(~source_mask).sum()} non-source cells")

        n_timesteps = len(timestep_keys)
        times = np.zeros(n_timesteps, dtype=np.double)
        source_aqueous = np.zeros(n_timesteps, dtype=np.double)
        nonsource_aqueous = np.zeros(n_timesteps, dtype=np.double)
        adsorbed = np.zeros(n_timesteps, dtype=np.double)
        dissolved = np.zeros(n_timesteps, dtype=np.double)

        print(f"Processing {n_timesteps} timesteps...")

        for idx, key in enumerate(timestep_keys):
            # Extract time
            time_yr = extract_time_from_key(key)
            times[idx] = time_yr

            # Load data from this timestep
            grp = f[key]

            # Extract arrays
            aq = np.array(grp['Total UO2++ [M]'][:])*np.array(grp['Porosity'][:]) # [mol/L]
            sorbed = np.array(grp['Total Sorbed UO2++ [mol_m^3 bulk]'][:])  # [mol/m^3]
            uo2_vf = np.array(grp['UO2:2H2O(am) VF [m^3 mnrl_m^3 bulk]'][:])  # [m^3/m^3]
            volume = np.array(grp['Volume [m^3]'][:])  # [m^3]

            # Calculate aqueous U separately for source and non-source regions
            source_aq = np.sum(aq[source_mask] * volume[source_mask] * 1000)
            nonsource_aq = np.sum(aq[~source_mask] * volume[~source_mask] * 1000)
            source_aqueous[idx] = source_aq
            nonsource_aqueous[idx] = nonsource_aq

            # Calculate total adsorbed uranium [mol]
            # sorbed [mol/m^3] * volume [m^3] = [mol]
            total_adsorbed = np.sum(sorbed * volume)
            adsorbed[idx] = total_adsorbed

            # Calculate total dissolved UO2 volume [m^3]
            # current mineral = uo2_vf * volume
            total_dissolved = np.sum(uo2_vf * volume) / 500 * 10**6
            dissolved[idx] = total_dissolved

            if idx % 5 == 0:  # Progress indicator
                print(f"  Processed timestep {idx+1}/{n_timesteps}: {time_yr:.1f} years")

    return times, source_aqueous, nonsource_aqueous, adsorbed, dissolved


def load_outlet_data(h5_path: Path) -> np.ndarray:
    """
    Load outlet uranium data from PFLOTRAN mass balance file

    Args:
        h5_path: Path to PFLOTRAN HDF5 output file

    Returns:
        outlet_data: Array of cumulative outlet UO2++ in mol (nt,)
                    Starting with 0.0 at year 0, then every 50 years
    """
    # Find corresponding -mas.dat file
    dat_path = h5_path.parent / f"{h5_path.stem}-mas.dat"

    if not dat_path.exists():
        print(f"[WARNING] Mass balance file not found: {dat_path}")
        print("          Returning zeros for outlet data")
        return np.zeros(41, dtype=np.double)  # 0, 50, 100, ..., 2000 years

    # Read header to get column names
    with open(dat_path, 'r') as f:
        header_line = f.readline().strip()

    # Parse header (comma-separated, quoted column names)
    header = [col.strip().strip('"') for col in header_line.split('","')]
    header[0] = header[0].lstrip('"')  # Remove leading quote from first column
    header[-1] = header[-1].rstrip('"')  # Remove trailing quote from last column

    # Find OUTLET UO2++ [mol] column
    try:
        outlet_col_idx = header.index('OUTLET UO2++ [mol]')
    except ValueError:
        print(f"[ERROR] Could not find 'OUTLET UO2++ [mol]' column in {dat_path}")
        return np.zeros(41, dtype=np.double)

    # Read data with space delimiter
    df = pd.read_csv(dat_path, sep=r'\s+', engine='python', skiprows=1, header=None)

    # Extract OUTLET UO2++ column
    outlet_col = df.iloc[:, outlet_col_idx].values

    # Sample every 50 years: years 50, 100, 150, ..., 2000
    # .dat file has year 1-2000 in rows 0-1999 (0-indexed)
    # Year 50 = row 49, year 100 = row 99, etc.
    sampled_outlet = -outlet_col[49::50]  # Start at index 49, step by 50

    # Prepend year 0 with value 0.0
    outlet_data = np.concatenate([[0.0], sampled_outlet])

    print(f"✓ Loaded outlet data from: {dat_path}")
    print(f"  Sample times: 0, 50, 100, ..., 2000 years ({len(outlet_data)} points)")

    return outlet_data


# ============================================================================
# Batch Processing Functions
# ============================================================================

def get_available_ids(base_dir: Path, final_timestep: str = '2000.0000yr') -> List[int]:
    """
    Automatically detect available pflotran simulation IDs

    Only includes simulations that have completed to the final timestep.

    Args:
        base_dir: PFLOTRAN output directory
        final_timestep: Final timestep filename suffix

    Returns:
        Sorted list of available simulation IDs that completed successfully
    """
    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return []

    available_ids = []
    incomplete_ids = []

    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("pflotran_"):
            match = re.match(r"pflotran_(\d+)", item.name)
            if match:
                id_num = int(match.group(1))
                h5_file = item / f"pflotran_{id_num}.h5"
                final_h5_file = item / f"pflotran_{id_num}-{final_timestep}.h5"

                # Check if simulation completed
                if h5_file.exists() and final_h5_file.exists():
                    available_ids.append(id_num)
                elif h5_file.exists():
                    incomplete_ids.append(id_num)

    available_ids.sort()

    # Report incomplete simulations
    if incomplete_ids:
        incomplete_ids.sort()
        print(f"[WARNING] Found {len(incomplete_ids)} incomplete simulation(s): {incomplete_ids}")
        print(f"          These will be skipped.")

    return available_ids


def process_single_case(h5_path: Path, sim_id: int) -> Dict[str, np.ndarray]:
    """
    Process a single PFLOTRAN case and return all mass balance data

    Args:
        h5_path: Path to PFLOTRAN HDF5 file
        sim_id: Simulation ID

    Returns:
        Dictionary with keys: 'times', 'source_aq', 'delta_source_aq',
        'nonsource_aq', 'delta_nonsource_aq', 'adsorbed', 'delta_adsorbed',
        'dissolved', 'delta_dissolved', 'outlet', 'delta_outlet'
    """
    # Calculate mass balance
    times, source_aq, nonsource_aq, adsorbed, dissolved = calculate_mass_balance(h5_path)

    # Load outlet data
    outlet = load_outlet_data(h5_path)

    # Calculate delta values (change from t=0)
    delta_source_aq = source_aq - source_aq[0]
    delta_nonsource_aq = nonsource_aq - nonsource_aq[0]
    delta_adsorbed = adsorbed - adsorbed[0]
    delta_dissolved = dissolved - dissolved[0]
    delta_outlet = outlet - outlet[0]

    return {
        'times': times,
        'source_aq': source_aq,
        'delta_source_aq': delta_source_aq,
        'nonsource_aq': nonsource_aq,
        'delta_nonsource_aq': delta_nonsource_aq,
        'adsorbed': adsorbed,
        'delta_adsorbed': delta_adsorbed,
        'dissolved': dissolved,
        'delta_dissolved': delta_dissolved,
        'outlet': outlet,
        'delta_outlet': delta_outlet
    }


def process_local_data() -> Optional[Path]:
    """
    Process all local PFLOTRAN simulations

    Returns:
        Path to generated .pkl file (or None if no data)
    """
    print(f"\n{'='*70}")
    print("Processing Local Data")
    print(f"{'='*70}\n")

    pflotran_dir = CONFIG['pflotran_dir']
    final_timestep = CONFIG['final_timestep']

    print(f"Scanning {pflotran_dir}...")
    available_ids = get_available_ids(pflotran_dir, final_timestep)

    if not available_ids:
        print("No PFLOTRAN data found locally")
        return None

    print(f"Found {len(available_ids)} completed simulation(s): {available_ids}")

    # Process each simulation
    results = {}
    for sim_id in available_ids:
        h5_path = pflotran_dir / f"pflotran_{sim_id}" / f"pflotran_{sim_id}.h5"
        print(f"  Processing pflotran_{sim_id}...")
        try:
            results[sim_id] = process_single_case(h5_path, sim_id)
        except Exception as e:
            print(f"    [ERROR] Failed to process pflotran_{sim_id}: {e}")
            continue

    if not results:
        print("[ERROR] No simulations processed successfully")
        return None

    # Save intermediate pickle
    output_file = CONFIG['output_dir'] / 'mass_balance_localhost.pkl'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Saved intermediate file: {output_file.name}")
    print(f"  Processed {len(results)} case(s)")

    return output_file


# ============================================================================
# Remote Data Processing (SSH)
# ============================================================================

def sync_script_on_server(host: str, user: str, port: int, script_name: str) -> bool:
    """
    Sync preprocessing script from git repository on remote server

    Only downloads if there are changes in the remote repository.

    Args:
        host: Remote server hostname
        user: SSH username
        port: SSH port
        script_name: Name of the script to sync (e.g., 'preprocessing_mass.py')

    Returns:
        True if sync successful or no changes needed, False on error
    """
    # Step 1: Check if there are remote changes
    check_command = (
        "cd research/FNO && "
        "git fetch origin --quiet && "
        f"git diff --quiet HEAD origin/master -- src/preprocessing/{script_name}"
    )

    ssh_check = [
        'ssh', '-p', str(port), '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}', check_command
    ]

    print(f"  Checking for updates to {script_name} on {host}...")

    try:
        check_result = subprocess.run(ssh_check, capture_output=True, text=True, timeout=30)

        # git diff --quiet returns:
        # - 0 if no differences (files are identical)
        # - 1 if there are differences
        # - >1 on error

        if check_result.returncode == 0:
            print(f"    ✓ No changes detected, using existing version")
            return True
        elif check_result.returncode == 1:
            print(f"    → Changes detected, downloading update...")

            # Step 2: Download the updated file
            sync_command = (
                "cd research/FNO && "
                f"git checkout origin/master -- src/preprocessing/{script_name}"
            )

            ssh_sync = [
                'ssh', '-p', str(port), '-o', 'StrictHostKeyChecking=no',
                f'{user}@{host}', sync_command
            ]

            sync_result = subprocess.run(ssh_sync, capture_output=True, text=True, timeout=30)

            if sync_result.returncode == 0:
                print(f"    ✓ Successfully updated {script_name}")
                return True
            else:
                print(f"    ✗ Failed to download update (code: {sync_result.returncode})")
                if sync_result.stderr.strip():
                    print(f"      Error: {sync_result.stderr.strip()}")
                return False
        else:
            print(f"    ✗ Error checking for changes (code: {check_result.returncode})")
            if check_result.stderr.strip():
                print(f"      Error: {check_result.stderr.strip()}")
            # Continue anyway with existing version
            return True

    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout while checking for updates")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def execute_remote_preprocessing(host: str, user: str, port: int, script_name: str) -> bool:
    """Execute preprocessing script on remote server via SSH"""
    remote_command = (
        "cd research/FNO && "
        "source .venv_FNO/bin/activate && "
        f"python3 src/preprocessing/{script_name} --local-only"
    )

    ssh_cmd = [
        'ssh', '-p', str(port), '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}', remote_command
    ]

    print(f"  Executing preprocessing on {host}...")

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            print(f"  ✓ Processing completed on {host}")
            return True
        else:
            print(f"  ✗ Processing failed on {host} (code: {result.returncode})")
            if result.stderr.strip():
                print(f"    {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Processing timed out on {host}")
        return False
    except Exception as e:
        print(f"  ✗ SSH error: {e}")
        return False


def download_result_file(host: str, user: str, port: int, output_suffix: str) -> Optional[Path]:
    """Download generated .pkl file from remote server using SCP"""
    remote_file = "research/FNO/src/preprocessing/mass_balance_localhost.pkl"
    local_file = CONFIG['output_dir'] / f"mass_balance_{output_suffix}.pkl"

    local_file.parent.mkdir(parents=True, exist_ok=True)

    scp_cmd = [
        'scp', '-P', str(port), '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}:{remote_file}', str(local_file)
    ]

    print(f"  Downloading result from {host}...")

    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=6000)

        if result.returncode == 0:
            print(f"  ✓ Downloaded: {local_file.name}")
            return local_file
        else:
            print(f"  ✗ Download failed (code: {result.returncode})")
            return None

    except subprocess.TimeoutExpired:
        print(f"  ✗ Download timed out")
        return None
    except Exception as e:
        print(f"  ✗ SCP error: {e}")
        return None


def process_remote_data(server_config: dict) -> List[Path]:
    """
    Process data on remote servers via SSH

    Args:
        server_config: Server configuration from YAML

    Returns:
        List of paths to downloaded .pkl files
    """
    print(f"\n{'='*70}")
    print("Processing Remote Servers")
    print(f"{'='*70}\n")

    if 'servers' not in server_config or len(server_config['servers']) == 0:
        print("No remote servers configured")
        return []

    script_name = CONFIG['preprocessing_script']
    downloaded_files = []

    for server in server_config['servers']:
        host = server['host']
        user = server['user']
        port = server.get('port', 22)
        output_suffix = host.replace('.', '_')

        print(f"\n--- Processing {host} ---")

        # Step 1: Sync script
        sync_script_on_server(host, user, port, script_name)

        # Step 2: Execute preprocessing
        success = execute_remote_preprocessing(host, user, port, script_name)

        if not success:
            print(f"✗ Failed to process {host}")
            sys.exit(1)

        # Step 3: Download result
        local_file = download_result_file(host, user, port, output_suffix)

        if local_file is None:
            print(f"✗ Failed to download from {host}")
            sys.exit(1)

        downloaded_files.append(local_file)

    return downloaded_files


# ============================================================================
# Data Merging and CSV Export
# ============================================================================

def merge_mass_balance_data(pkl_files: List[Path]) -> Tuple[Dict[int, dict], np.ndarray]:
    """
    Merge multiple .pkl files into a single dictionary

    Args:
        pkl_files: List of .pkl file paths

    Returns:
        all_results: Dictionary mapping sim_id to result dict
        times: Reference time array
    """
    print(f"\n{'='*70}")
    print("Merging Data")
    print(f"{'='*70}\n")

    print(f"Loading {len(pkl_files)} file(s):")
    for f in pkl_files:
        print(f"  - {f.name}")

    all_results = {}

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            shard = pickle.load(f)
            all_results.update(shard)

    if not all_results:
        raise RuntimeError("No data to merge")

    # Validate consistent times
    ref_times = None
    for sim_id, data in all_results.items():
        if ref_times is None:
            ref_times = data['times']
        elif not np.allclose(ref_times, data['times']):
            raise ValueError(f"Time mismatch at pflotran_{sim_id}")

    print(f"\n✓ Merged {len(all_results)} case(s)")
    print(f"  Time points: {len(ref_times)}")

    return all_results, ref_times


def save_to_separate_csvs(all_results: Dict[int, dict], times: np.ndarray, output_dir: Path):
    """
    Save mass balance data to 10 separate CSV files

    Args:
        all_results: Dictionary mapping sim_id to result dict
        times: Time array
        output_dir: Output directory
    """
    print(f"\n{'='*70}")
    print("Saving CSV Files")
    print(f"{'='*70}\n")

    # Variable mapping: (csv_name, dict_key)
    variables = [
        ('mass_source_aqueous.csv', 'source_aq'),
        ('mass_delta_source_aqueous.csv', 'delta_source_aq'),
        ('mass_nonsource_aqueous.csv', 'nonsource_aq'),
        ('mass_delta_nonsource_aqueous.csv', 'delta_nonsource_aq'),
        ('mass_adsorbed.csv', 'adsorbed'),
        ('mass_delta_adsorbed.csv', 'delta_adsorbed'),
        ('mass_dissolved.csv', 'dissolved'),
        ('mass_delta_dissolved.csv', 'delta_dissolved'),
        ('mass_outlet.csv', 'outlet'),
        ('mass_delta_outlet.csv', 'delta_outlet')
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_name, dict_key in variables:
        # Create DataFrame with times as first column
        df = pd.DataFrame({'time_years': times})

        # Add each case as a column
        for sim_id in sorted(all_results.keys()):
            df[f'pflotran_{sim_id}'] = all_results[sim_id][dict_key]

        # Save to CSV
        output_path = output_dir / csv_name
        df.to_csv(output_path, index=False, float_format='%.6e')
        print(f"  ✓ {csv_name:40s} ({len(all_results)} cases)")

    print(f"\n✓ Saved 10 CSV files to: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Main workflow:
    1. Parse command line arguments
    2. Process local PFLOTRAN data
    3. Process remote PFLOTRAN data (if not --local-only)
    4. Merge all data
    5. Save to 10 separate CSV files
    """
    parser = argparse.ArgumentParser(
        description='Collect mass balance data from all PFLOTRAN simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local only
  python preprocessing_mass.py --local-only

  # Local + remote servers
  python preprocessing_mass.py --config ../../servers.yaml
        """
    )

    parser.add_argument('--config', type=str, default='../../servers.yaml',
                        help='Server configuration YAML file')
    parser.add_argument('--local-only', action='store_true',
                        help='Process local data only, skip remote servers')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("PFLOTRAN Mass Balance Collection")
    print(f"{'='*70}")
    print(f"Local only: {args.local_only}")
    print(f"Output dir: {CONFIG['output_dir']}")
    print(f"{'='*70}")

    pkl_files = []

    # 1. Process local data
    local_pkl = process_local_data()
    if local_pkl:
        pkl_files.append(local_pkl)

    # 2. Process remote data (if not local-only)
    if not args.local_only:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            server_config = yaml.safe_load(f)

        remote_pkls = process_remote_data(server_config)
        pkl_files.extend(remote_pkls)

    # 3. Merge all results
    if not pkl_files:
        print("\n[ERROR] No data to process")
        sys.exit(1)

    all_results, times = merge_mass_balance_data(pkl_files)

    # 4. Save to 10 CSV files
    output_dir = Path(CONFIG['output_dir'])
    save_to_separate_csvs(all_results, times, output_dir)

    print(f"\n{'='*70}")
    print("Mass Balance Collection Complete!")
    print(f"{'='*70}")
    print(f"\n✓ Generated 10 CSV files")
    print(f"  Total cases: {len(all_results)}")
    print(f"  Time points: {len(times)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
