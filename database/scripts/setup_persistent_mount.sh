#!/bin/bash
# Script to set up persistent NFS mount for Delta storage

echo "Setting up persistent NFS mount for Delta storage..."

# Check if already mounted
if mountpoint -q /mnt/delta_bbub; then
    echo "✓ /mnt/delta_bbub is already mounted"
else
    echo "Mounting Delta storage..."
    sudo mount -t nfs taiga-nfs.ncsa.illinois.edu:/taiga/nsf/delta/bbub /mnt/delta_bbub
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully mounted Delta storage"
    else
        echo "✗ Failed to mount Delta storage"
        exit 1
    fi
fi

# Check if entry exists in fstab
if grep -q "taiga-nfs.ncsa.illinois.edu:/taiga/nsf/delta/bbub" /etc/fstab; then
    echo "✓ Mount entry already exists in /etc/fstab"
else
    echo "Adding mount to /etc/fstab for persistence..."
    echo "# Delta NFS storage for TorchCell database" | sudo tee -a /etc/fstab
    echo "taiga-nfs.ncsa.illinois.edu:/taiga/nsf/delta/bbub /mnt/delta_bbub nfs defaults,_netdev,auto 0 0" | sudo tee -a /etc/fstab
    echo "✓ Added persistent mount entry to /etc/fstab"
fi

# Verify mount
df -h /mnt/delta_bbub
echo ""
echo "Mount is configured to persist across reboots."
echo "The '_netdev' option ensures the mount waits for network before mounting."