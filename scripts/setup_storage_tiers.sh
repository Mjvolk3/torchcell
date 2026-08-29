#!/bin/bash
# scripts/setup_storage_tiers.sh
# [[scripts.setup_storage_tiers]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/setup_storage_tiers
#
# Bring GilaHyper's three unused drives into service as two storage tiers. Run as root.
# DRY RUN BY DEFAULT: prints every command and changes nothing until --yes-destroy.
#
#   sudo bash scripts/setup_storage_tiers.sh              # dry run, review the plan
#   sudo bash scripts/setup_storage_tiers.sh --yes-destroy
#
# WHAT IT BUILDS
#   /bulk   mdadm RAID1 over the two WD Gold 26 TB (sdb, sdc) -> 23.6 TiB usable, xfs
#   /db     the WD_BLACK SN850X 8 TB (nvme1n1), single device, xfs
#
# It formats and mounts only. It MOVES NO DATA -- migration is a separate, reviewable
# step, because a half-finished move of a 3 TB dataset tree is far worse than a delay.
#
# WHY THIS SHAPE
#   Crash exposure, not speed, sets the split. The Micron 7400 behind /scratch is an
#   enterprise drive with power-loss protection; the SN850X is a consumer drive without
#   it. So /scratch becomes the ML tier (dataset LMDBs, training data, wandb), /db
#   takes the neo4j store + serving container -- a TEMPORARY serving arrangement whose
#   contents must stay REGENERABLE, since the SN850X cannot guarantee committed writes
#   across power loss (the store re-imports from archived CSV in ~19 min). /bulk takes
#   cold bytes: CSV archives, dumps, raw mirrors, the graveyard, archived experiments.
#   Nothing irreplaceable ever lives on /db.
#
#   RAID1 rather than RAID0 or JBOD. Derived data is regenerable, but the raw mirrors
#   and OCR artifacts at the root of the provenance chain are not: several sources are
#   already un-refetchable (Science.org 403s, dead portals). Halving capacity to 23.6 TiB
#   is the price of not breaking the rebuild guarantee on one drive failure. Set
#   RAID_LEVEL=0 to take 47 TiB with no redundancy instead.
#
#   No ZFS. It would give compression and end-to-end checksums, but OpenZFS on Rocky 9
#   is a DKMS module, and a kernel update landing ahead of the rebuild leaves the pool
#   unmountable at boot. This machine both builds and serves the graph.
set -euo pipefail

RAID_DEV="${RAID_DEV:-/dev/md0}"
RAID_LEVEL="${RAID_LEVEL:-1}"
RAID_MEMBERS=("${RAID_MEMBERS[@]:-/dev/sdb /dev/sdc}")
FAST_DEV="${FAST_DEV:-/dev/nvme1n1}"
BULK_MNT="${BULK_MNT:-/bulk}"
FAST_MNT="${FAST_MNT:-/db}"
OWNER="${OWNER:-michaelvolk:michaelvolk}"

APPLY=0
[ "${1:-}" = "--yes-destroy" ] && APPLY=1

run() {
    if [ "$APPLY" = "1" ]; then
        echo "+ $*"
        "$@"
    else
        echo "  [dry-run] $*"
    fi
}

fail() {
    echo "ABORT: $*" >&2
    exit 1
}

# --- guards ---------------------------------------------------------------
# Every one of these has to pass before a single byte is written. The cost of a wrong
# device name here is the 3 TB dataset tree or the OS.

[ "$(id -u)" = "0" ] || fail "run as root (sudo)"

for dev in "${RAID_MEMBERS[@]}" "$FAST_DEV"; do
    [ -b "$dev" ] || fail "$dev is not a block device"
    if findmnt -S "$dev" >/dev/null 2>&1; then
        fail "$dev is MOUNTED -- refusing"
    fi
    if [ -n "$(lsblk -no FSTYPE,PARTTYPE "$dev" 2>/dev/null | tr -d ' \n')" ]; then
        fail "$dev already carries a filesystem or partition table -- refusing"
    fi
    sig="$(wipefs -n "$dev" 2>/dev/null | tail -n +2 || true)"
    [ -z "$sig" ] || fail "$dev has existing signatures:
$sig"
    # A device holding one of the mounted filesystems must never appear here.
    if lsblk -no MOUNTPOINT "$dev" | grep -q .; then
        fail "$dev has a mounted child -- refusing"
    fi
done

for mnt in "$BULK_MNT" "$FAST_MNT"; do
    if findmnt "$mnt" >/dev/null 2>&1; then
        fail "$mnt is already a mount point"
    fi
done

echo "=== devices to be DESTROYED and reformatted ==="
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL "${RAID_MEMBERS[@]}" "$FAST_DEV"
echo
echo "  RAID${RAID_LEVEL} over ${RAID_MEMBERS[*]}  ->  $RAID_DEV  ->  $BULK_MNT (xfs)"
echo "  ${FAST_DEV}                                ->  $FAST_MNT (xfs)"
echo
if [ "$APPLY" != "1" ]; then
    echo "DRY RUN. Re-run with --yes-destroy to apply."
fi
echo

# --- bulk tier: mdadm + xfs ----------------------------------------------
# --bitmap=internal makes a resync resumable across a reboot. A RAID1 resync of 23.6 TiB
# runs on the order of a day and a half at ~200 MB/s; the array is usable throughout,
# just slower, so there is no reason to skip it with --assume-clean.
echo "=== 1/5 create $RAID_DEV ==="
run mdadm --create "$RAID_DEV" \
    --level="$RAID_LEVEL" \
    --raid-devices="${#RAID_MEMBERS[@]}" \
    --bitmap=internal \
    --run \
    "${RAID_MEMBERS[@]}"

echo "=== 2/5 persist the array so it assembles at boot ==="
run bash -c "mdadm --detail --scan >> /etc/mdadm.conf"
run dracut --force

echo "=== 3/5 make filesystems ==="
run mkfs.xfs -f -L bulk "$RAID_DEV"
run mkfs.xfs -f -L db "$FAST_DEV"

echo "=== 4/5 mount points + fstab (by UUID, nofail) ==="
run mkdir -p "$BULK_MNT" "$FAST_MNT"
if [ "$APPLY" = "1" ]; then
    BULK_UUID="$(blkid -s UUID -o value "$RAID_DEV")"
    FAST_UUID="$(blkid -s UUID -o value "$FAST_DEV")"
    cp -a /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d%H%M%S)"
    # nofail so a degraded or absent array can never block boot to a rescue shell.
    printf 'UUID=%s %s xfs defaults,nofail 0 2\n' "$BULK_UUID" "$BULK_MNT" >>/etc/fstab
    printf 'UUID=%s %s xfs defaults,nofail 0 2\n' "$FAST_UUID" "$FAST_MNT" >>/etc/fstab
else
    echo "  [dry-run] append UUID= lines for $RAID_DEV and $FAST_DEV to /etc/fstab (backed up first)"
fi

echo "=== 5/5 mount and hand ownership over ==="
run mount -a
run chown "$OWNER" "$BULK_MNT" "$FAST_MNT"

echo
if [ "$APPLY" = "1" ]; then
    findmnt "$BULK_MNT" "$FAST_MNT" || true
    df -h "$BULK_MNT" "$FAST_MNT" || true
    echo
    echo "RAID resync progress (array is usable while this runs):"
    cat /proc/mdstat
fi
