#!/bin/bash

# Rsync Transfer Script: GilaHyper -> Delta NCSA
# Transfers the torchcell database dump to Delta for production use

set -e

# =============================================================================
# Configuration
# =============================================================================

# Delta NCSA connection (use data transfer node for large files)
DELTA_USER="mjvolk3"  # Same username on Delta
DELTA_HOST="dt-login.delta.ncsa.illinois.edu"  # Data transfer node (faster)
# Alternative: login.delta.ncsa.illinois.edu

# Paths
SOURCE_SYMLINK="/scratch/projects/torchcell-scratch/database/export/torchcell_latest.dump"
DEST_DIR="/projects/bbub/mjvolk3/torchcell/database/import/"

# Resolve symlink to get the actual timestamped file (e.g., torchcell_20251208_152655.dump)
SOURCE_FILE=$(readlink -f "$SOURCE_SYMLINK" 2>/dev/null)

# =============================================================================
# Main
# =============================================================================

echo "========================================="
echo "Rsync Transfer: GilaHyper -> Delta NCSA"
echo "========================================="
echo ""
echo "Source symlink: $SOURCE_SYMLINK"
echo "Actual file:    $SOURCE_FILE"
echo "Destination:    ${DELTA_USER}@${DELTA_HOST}:${DEST_DIR}"
echo ""

# Check source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "ERROR: Source file not found: $SOURCE_FILE"
    exit 1
fi

FILE_SIZE=$(ls -lh "$SOURCE_FILE" | awk '{print $5}')
echo "File size: $FILE_SIZE"
echo ""

# Confirm before transfer
read -p "Start transfer? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Transfer cancelled."
    exit 0
fi

echo ""
echo "Starting rsync transfer..."
echo ""

# Rsync with progress, compression, archive mode, and checksum verification
# --checksum: verifies file integrity after transfer
# --partial: keeps partial files if interrupted (allows resume)
# --partial-dir: stores partial files in hidden dir
rsync -avz --progress --checksum --partial --partial-dir=.rsync-partial \
    "$SOURCE_FILE" \
    "${DELTA_USER}@${DELTA_HOST}:${DEST_DIR}"

RSYNC_EXIT=$?

echo ""
if [ $RSYNC_EXIT -eq 0 ]; then
    echo "========================================="
    echo "Transfer completed successfully!"
    echo "========================================="

    # Verify by comparing checksums
    echo ""
    echo "Verifying transfer integrity..."
    LOCAL_MD5=$(md5sum "$SOURCE_FILE" | awk '{print $1}')
    REMOTE_MD5=$(ssh "${DELTA_USER}@${DELTA_HOST}" "md5sum ${DEST_DIR}$(basename $SOURCE_FILE)" | awk '{print $1}')

    echo "Local MD5:  $LOCAL_MD5"
    echo "Remote MD5: $REMOTE_MD5"

    if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
        echo ""
        echo "✓ Checksums match - transfer verified!"

        # Show remote file details
        echo ""
        echo "Remote file:"
        ssh "${DELTA_USER}@${DELTA_HOST}" "ls -lh ${DEST_DIR}$(basename $SOURCE_FILE)"
    else
        echo ""
        echo "✗ WARNING: Checksums do not match!"
        echo "  The transfer may be corrupted. Re-run this script to retry."
        exit 1
    fi
else
    echo "========================================="
    echo "Transfer FAILED (exit code: $RSYNC_EXIT)"
    echo "========================================="
    echo ""
    echo "Common rsync exit codes:"
    echo "  1  - Syntax error"
    echo "  12 - Error in rsync protocol"
    echo "  23 - Partial transfer (some files not transferred)"
    echo "  24 - Source file vanished"
    echo "  255 - SSH connection failed"
    echo ""
    echo "To RETRY, simply re-run this script."
    echo "Rsync will resume from where it left off (--partial flag)."
    echo ""
    echo "  bash database/scripts/rsync_transfer_gh_to_delta.sh"
    echo ""
    exit $RSYNC_EXIT
fi

echo ""
echo "========================================="
