#!/bin/bash

# Script to set up storage directories for TorchCell Neo4j database
# This should be run once before the first build

PROJECT_DIR="/home/rocky/projects/torchcell"
# Always use rocky's home directory, even when run with sudo
if [ "$SUDO_USER" ]; then
    LOCAL_DATA_DIR="/home/$SUDO_USER/neo4j-data"
else
    LOCAL_DATA_DIR="$HOME/neo4j-data"
fi
DELTA_DATA_DIR="/mnt/delta_bbub/mjvolk3/torchcell"

echo "Setting up storage directories for TorchCell database..."
echo ""
echo "This script will create:"
echo "  - Local directories at: $LOCAL_DATA_DIR"
echo "  - Delta mount directories at: $DELTA_DATA_DIR"
echo ""

# Create local directories
echo "Creating local directories..."
mkdir -p $LOCAL_DATA_DIR/{neo4j-data,logs,biocypher-out}
chmod -R 777 $LOCAL_DATA_DIR
# If run with sudo, change ownership back to the user
if [ "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER $LOCAL_DATA_DIR
fi
echo "✓ Local directories created"

# Create Delta directories
echo ""
echo "Creating Delta mount directories..."

# First check if we can access the parent directory
if [ ! -d "/mnt/delta_bbub" ] || [ ! -r "/mnt/delta_bbub" ]; then
    echo "ERROR: Cannot access /mnt/delta_bbub"
    echo "Please ensure you have access to the Delta mount."
    exit 1
fi

# Try to create as regular user first (if in delta_nfs group)
echo "Attempting to create directories as user..."
if mkdir -p $DELTA_DATA_DIR 2>/dev/null; then
    echo "Created base directory as user"
    mkdir -p $DELTA_DATA_DIR/{biocypher-out,torchcell,sgd}
    chmod -R 777 $DELTA_DATA_DIR/{biocypher-out,torchcell,sgd} 2>/dev/null || true
    echo "✓ Delta directories created"
elif [ ! -d "$DELTA_DATA_DIR" ]; then
    # If that fails, try with sudo
    echo "Regular user creation failed, trying with sudo..."
    sudo mkdir -p $DELTA_DATA_DIR
    sudo mkdir -p $DELTA_DATA_DIR/{biocypher-out,torchcell,sgd}
    sudo chmod -R 777 $DELTA_DATA_DIR/{biocypher-out,torchcell,sgd}
    echo "✓ Delta directories created with sudo"
else
    # Directory exists, just create subdirectories
    if [ ! -d "$DELTA_DATA_DIR/biocypher-out" ] || [ ! -d "$DELTA_DATA_DIR/torchcell" ] || [ ! -d "$DELTA_DATA_DIR/sgd" ]; then
        echo "Creating subdirectories..."
        mkdir -p $DELTA_DATA_DIR/{biocypher-out,torchcell,sgd}
        chmod -R 777 $DELTA_DATA_DIR/{biocypher-out,torchcell,sgd} 2>/dev/null || true
        echo "✓ Delta subdirectories created"
    else
        echo "✓ Delta directories already exist"
    fi
fi

# Verify permissions
echo ""
echo "Verifying directory permissions..."
if [ -w "$LOCAL_DATA_DIR" ]; then
    echo "✓ Local directory is writable"
else
    echo "⚠ Warning: Local directory may not be writable"
fi

if [ -w "$DELTA_DATA_DIR/biocypher-out" ]; then
    echo "✓ Delta directories are writable"
else
    echo "⚠ Warning: Delta directories may not be writable"
fi

echo ""
echo "Storage setup complete!"
echo ""
echo "Directory structure:"
echo "Local (for Neo4j system files):"
echo "  $LOCAL_DATA_DIR/"
echo "  ├── neo4j-data/     # Neo4j database system files"
echo "  ├── logs/           # Log files"
echo "  └── biocypher-out/  # Temporary build artifacts (if needed locally)"
echo ""
echo "Delta mount (for large data):"
echo "  $DELTA_DATA_DIR/"
echo "  ├── biocypher-out/  # BioCypher output files"
echo "  ├── torchcell/      # TorchCell data"
echo "  └── sgd/            # SGD data"
echo ""
echo "You can now run: bash $PROJECT_DIR/database/build/build_openstack.sh"