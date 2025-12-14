#!/bin/bash
#SBATCH --job-name=rsync_to_delta
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell-scratch/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# =============================================================================
# Rsync Transfer: GilaHyper -> Delta NCSA (SLURM version)
# Transfers the torchcell database dump to Delta for production use
# =============================================================================

echo "========================================="
echo "Rsync Transfer: GilaHyper -> Delta NCSA"
echo "========================================="
echo "Running on $(hostname)"
echo "Started at: $(date)"
echo ""

# Configuration
DELTA_USER="mjvolk3"
DELTA_HOST="dt-login.delta.ncsa.illinois.edu"
SOURCE_SYMLINK="/scratch/projects/torchcell-scratch/database/export/torchcell_latest.dump"
DEST_DIR="/projects/bbub/mjvolk3/torchcell/database/import/"

# Resolve symlink to get the actual timestamped file (e.g., torchcell_20251208_152655.dump)
SOURCE_FILE=$(readlink -f "$SOURCE_SYMLINK" 2>/dev/null)

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

# Compute local checksum before transfer
echo "Computing local checksum..."
LOCAL_MD5=$(md5sum "$SOURCE_FILE" | awk '{print $1}')
echo "Local MD5: $LOCAL_MD5"
echo ""

# Rsync transfer
echo "Starting rsync transfer at $(date)..."
echo ""

# Transfer the actual timestamped file (not symlink)
rsync -avz --progress --checksum --partial --partial-dir=.rsync-partial \
    "$SOURCE_FILE" \
    "${DELTA_USER}@${DELTA_HOST}:${DEST_DIR}"

RSYNC_EXIT=$?

echo ""
echo "Rsync finished at $(date)"
echo ""

if [ $RSYNC_EXIT -eq 0 ]; then
    echo "========================================="
    echo "Transfer completed successfully!"
    echo "========================================="

    # Verify by comparing checksums
    echo ""
    echo "Verifying transfer integrity..."
    REMOTE_MD5=$(ssh "${DELTA_USER}@${DELTA_HOST}" "md5sum ${DEST_DIR}$(basename $SOURCE_FILE)" | awk '{print $1}')

    echo "Local MD5:  $LOCAL_MD5"
    echo "Remote MD5: $REMOTE_MD5"

    if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
        echo ""
        echo "✓ Checksums match - transfer verified!"

        # Create/update torchcell_latest.dump symlink on Delta
        FILENAME=$(basename $SOURCE_FILE)
        echo ""
        echo "Creating symlink: torchcell_latest.dump -> $FILENAME"
        ssh "${DELTA_USER}@${DELTA_HOST}" "cd ${DEST_DIR} && ln -sf ${FILENAME} torchcell_latest.dump"

        echo ""
        echo "Remote files:"
        ssh "${DELTA_USER}@${DELTA_HOST}" "ls -lh ${DEST_DIR}"
        echo ""
        echo "========================================="
        echo "SUCCESS - Transfer complete and verified"
        echo "Finished at: $(date)"
        echo "========================================="
        exit 0
    else
        echo ""
        echo "✗ WARNING: Checksums do not match!"
        echo "  The transfer may be corrupted."
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
    echo "To retry, resubmit this job:"
    echo "  sbatch database/slurm/scripts/rsync_transfer_gh_to_delta.slurm"
    echo ""
    exit $RSYNC_EXIT
fi
