#!/bin/bash

# auto_setting.sh
# Automated installation script for PFLOTRAN and dependencies
# This script installs PFLOTRAN without PyTorch (manual installation recommended)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Get current user's home directory
USER_HOME=$(eval echo ~$USER)
log "Using home directory: $USER_HOME"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   warn "This script is running as root. Some operations will use sudo explicitly."
fi

log "Starting PFLOTRAN auto-installation..."

# 1. Install necessary programs
log "Step 1: Installing necessary programs..."
sudo apt update --fix-missing || error "Failed to update apt packages"
sudo apt install -y gcc gfortran make cmake git python3 python3.12-venv python3-pip libtool autoconf build-essential pkg-config automake tcsh mpich || error "Failed to install required packages"

# 2. Install PETSc
log "Step 2: Installing PETSc..."

# Go to home directory
cd "$USER_HOME" || error "Failed to change to home directory"

# Clone PETSc repository
if [ -d "petsc" ]; then
    warn "PETSc directory already exists. Removing and re-cloning..."
    rm -rf petsc
fi

log "Cloning PETSc repository..."
git clone https://gitlab.com/petsc/petsc petsc || error "Failed to clone PETSc repository"

# Change to petsc directory
cd petsc || error "Failed to change to petsc directory"

# Checkout specific version
log "Checking out PETSc main..."
git checkout main || error "Failed to checkout PETSc v3.21.5"

# Configure PETSc
log "Configuring PETSc..."
./configure --COPTFLAGS='-O3' --CXXOPTFLAGS='-O3' --FOPTFLAGS='-O3 -Wno-unused-function -fallow-argument-mismatch' --with-debugging=no --download-mpich=yes --download-hdf5=yes --download-hdf5-fortran-bindings=yes --download-fblaslapack=yes --download-metis=yes --download-parmetis=yes || error "Failed to configure PETSc"

# 3. Set environment variables in .bashrc
log "Step 3: Setting up environment variables..."

# Define .bashrc path
BASHRC_FILE="$USER_HOME/.bashrc"

# Check if .bashrc exists, if not create it
if [ ! -f "$BASHRC_FILE" ]; then
    touch "$BASHRC_FILE"
    log "Created .bashrc file"
fi

# Backup .bashrc
cp "$BASHRC_FILE" "$USER_HOME/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"

# Function to safely add environment variables
add_env_var() {
    local var_name="$1"
    local var_value="$2"
    local comment="$3"
    
    # Check if .bashrc is writable
    if [ ! -w "$BASHRC_FILE" ]; then
        error ".bashrc is not writable. Cannot add environment variables."
    fi
    
    if ! grep -q "export $var_name=" "$BASHRC_FILE"; then
        {
            if [ -n "$comment" ]; then
                echo ""
                echo "# $comment"
            fi
            echo "export $var_name=$var_value"
        } >> "$BASHRC_FILE" || error "Failed to write $var_name to .bashrc"
        
        log "Added $var_name to .bashrc"
        return 0
    else
        log "$var_name already exists in .bashrc"
        return 1
    fi
}

# Add environment variables
add_env_var "PETSC_DIR" "$USER_HOME/petsc" "PFLOTRAN/PETSc environment variables"
add_env_var "PETSC_ARCH" "arch-linux-c-opt"
add_env_var "PFLOTRAN_DIR" "$USER_HOME/pflotran"

# Verify .bashrc file permissions and readability
if [ ! -r "$BASHRC_FILE" ]; then
    warn ".bashrc is not readable. Attempting to fix permissions..."
    chmod 644 "$BASHRC_FILE" || error "Failed to fix .bashrc permissions"
fi

# Source the updated .bashrc for current session
if source "$BASHRC_FILE" 2>/dev/null; then
    log "Successfully sourced .bashrc"
else
    warn "Failed to source .bashrc, setting variables manually for current session"
fi

# Set environment variables for current session (regardless of sourcing success)
export PETSC_DIR="$USER_HOME/petsc"
export PETSC_ARCH="arch-linux-c-opt"
export PFLOTRAN_DIR="$USER_HOME/pflotran"

log "Environment variables set and sourced."
log "PETSC_DIR: $PETSC_DIR"
log "PETSC_ARCH: $PETSC_ARCH"
log "PFLOTRAN_DIR: $PFLOTRAN_DIR"

# 4. Make PETSc
log "Step 4: Building PETSc..."
cd "$PETSC_DIR" || error "Failed to change to PETSC_DIR"
make all || error "Failed to build PETSc"

# 5. Install PFLOTRAN
log "Step 5: Installing PFLOTRAN..."

# Go back to home directory
cd "$USER_HOME" || error "Failed to change to home directory"

# Clone PFLOTRAN repository
if [ -d "pflotran" ]; then
    warn "PFLOTRAN directory already exists. Removing and re-cloning..."
    rm -rf pflotran
fi

log "Cloning PFLOTRAN repository..."
git clone https://bitbucket.org/pflotran/pflotran || error "Failed to clone PFLOTRAN repository"

# Build PFLOTRAN
log "Building PFLOTRAN..."
cd pflotran/src/pflotran || error "Failed to change to PFLOTRAN source directory"
make pflotran || error "Failed to build PFLOTRAN"

# 6. Install user code
log "Step 6: Setting up research code..."

# Go back to home directory
cd "$USER_HOME" || error "Failed to change to home directory"

# Create research directory if it doesn't exist
if [ ! -d "research" ]; then
    mkdir research || error "Failed to create research directory"
    log "Created research directory"
else
    log "Research directory already exists"
fi

cd research || error "Failed to change to research directory"

# Clone the FNO repository
if [ -d "FNO" ]; then
    warn "FNO directory already exists. Skipping clone..."
else
    log "Cloning FNO repository..."
    git clone https://github.com/wonwooyoon/FNO || error "Failed to clone FNO repository"
fi

# Set up Python virtual environment and install requirements
log "Setting up Python virtual environment..."

cd FNO || error "Failed to change to FNO directory"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv_FNO" ]; then
    log "Creating .venv_FNO virtual environment..."
    python3 -m venv .venv_FNO || error "Failed to create virtual environment"
else
    log ".venv_FNO virtual environment already exists"
fi

# Activate virtual environment
log "Activating virtual environment..."
source .venv_FNO/bin/activate || error "Failed to activate virtual environment"

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip || warn "Failed to upgrade pip, continuing..."

# Install requirements if requirements files exist
if [ -f "requirements_simple.txt" ]; then
    log "Installing requirements from requirements_simple.txt..."
    pip install -r requirements_simple.txt || error "Failed to install requirements_simple.txt"
else
    warn "requirements_simple.txt not found, skipping basic requirements installation"
fi

if [ -f "requirements.txt" ]; then
    log "Installing requirements from requirements.txt..."
    pip install -r requirements.txt || warn "Failed to install requirements.txt, continuing..."
fi

# Install neuraloperator in development mode
if [ -d "neuraloperator" ]; then
    log "Installing neuraloperator library in development mode..."
    cd neuraloperator || error "Failed to change to neuraloperator directory"
    pip install -e . || warn "Failed to install neuraloperator, continuing..."
    cd .. || error "Failed to return to FNO directory"
fi

# Deactivate virtual environment
deactivate

log "Virtual environment setup completed successfully!"

log "Installation completed successfully!"
log ""
log "Next steps:"
log "1. Run 'source ~/.bashrc' to load environment variables"
log "2. Activate the Python environment: 'cd ~/research/FNO && source .venv_FNO/bin/activate'"
log "3. For FNO training, you may need to install additional PyTorch dependencies"
log ""
log "PFLOTRAN executable location: $USER_HOME/pflotran/src/pflotran/pflotran"
log "Research code location: $USER_HOME/research/FNO"
log "Python virtual environment: $USER_HOME/research/FNO/.venv_FNO"