#!/usr/bin/env python3
"""
Simple SSH-based distributed preprocessing
Only executes preprocessing_revised.py on remote servers
"""

import subprocess
import yaml
import sys

def load_config(config_path: str):
    """Load server configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def execute_remote_preprocessing(host: str, user: str, port: int, output_suffix: str):
    """Execute preprocessing_revised.py on remote server via SSH"""
    
    # SSH command to run preprocessing_revised.py with simplified input
    # Use single line command with && separators
    remote_command = f"cd research/FNO && source .venv_FNO/bin/activate && python3 src/preprocessing/preprocessing_revised.py {output_suffix}"
    
    # Build SSH command
    ssh_cmd = [
        'ssh', 
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}',
        remote_command
    ]
    
    print(f"Executing preprocessing on {host}:{port} with suffix '{output_suffix}'")
    print(f"SSH command: {' '.join(ssh_cmd)}")  # Debug: show actual command
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=3600)
        
        print(f"Return code: {result.returncode}")  # Debug info
        
        if result.returncode == 0:
            print(f"✅ {host}: Processing completed successfully")
            if result.stdout.strip():
                print(f"Output:\n{result.stdout}")
        else:
            print(f"❌ {host}: Processing failed (return code: {result.returncode})")
            if result.stdout.strip():
                print(f"Stdout:\n{result.stdout}")
            if result.stderr.strip():
                print(f"Stderr:\n{result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ {host}: Processing timed out")
        return False
    except Exception as e:
        print(f"❌ {host}: SSH error - {e}")
        return False

def download_result_file(host: str, user: str, port: int, output_suffix: str):
    """Download the generated .pt file from remote server using SCP"""
    
    remote_file = f"research/FNO/src/preprocessing/input_output_com{output_suffix}.pt"
    local_file = f"./src/preprocessing/input_output_com{output_suffix}.pt"
    
    # Ensure local directory exists
    from pathlib import Path
    Path(local_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Build SCP command
    scp_cmd = [
        'scp',
        '-P', str(port),  # Note: SCP uses capital P
        '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}:{remote_file}',
        local_file
    ]
    
    print(f"Downloading result from {host}:{port}...")
    print(f"SCP command: {' '.join(scp_cmd)}")  # Debug: show actual command
    
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
        
        print(f"SCP return code: {result.returncode}")  # Debug info
        
        if result.returncode == 0:
            print(f"✅ {host}: File downloaded successfully to {local_file}")
            return True
        else:
            print(f"❌ {host}: Download failed (return code: {result.returncode})")
            if result.stdout.strip():
                print(f"SCP Stdout:\n{result.stdout}")
            if result.stderr.strip():
                print(f"SCP Stderr:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {host}: Download timed out")
        return False
    except Exception as e:
        print(f"❌ {host}: SCP error - {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 distributed_preprocessing.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Process each server
    for server in config['servers']:
        host = server['host']
        user = server['user'] 
        port = server.get('port', 22)
        # Create output suffix from server info (no longer need ID range)
        output_suffix = f"{host}"
        
        # Execute preprocessing on remote server
        preprocessing_success = execute_remote_preprocessing(host, user, port, output_suffix)
        
        if not preprocessing_success:
            print(f"Failed to process server {host}")
            sys.exit(1)
        
        # Download result file from remote server
        download_success = download_result_file(host, user, port, output_suffix)
        
        if not download_success:
            print(f"Failed to download result from server {host}")
            sys.exit(1)
    
    print("🎉 All servers processed and files downloaded successfully!")
    print("\nDownloaded files:")
    for server in config['servers']:
        host = server['host']
        output_suffix = f"{host}"
        local_file = f"./src/preprocessing/input_output_com{output_suffix}.pt"
        print(f"  - {local_file}")

if __name__ == "__main__":
    main()