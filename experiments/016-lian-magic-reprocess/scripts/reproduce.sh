#!/bin/bash
# experiments/016-lian-magic-reprocess/scripts/reproduce.sh
# End-to-end regeneration of the Lian 2019 MAGIC per-guide enrichment table
# (guide_enrichment_final.tsv, sha256 f9af849f...) from raw SRA + the sha256-pinned inputs.
# This is what makes the ~12 GB of downloaded fastqs SAFE TO DELETE: they are fully
# regenerable. See README.md for the environment + provenance details.
set -euo pipefail

# Pinned inputs live in the library mirror; the work dir is scratch (large intermediates).
: "${DATA_ROOT:?set DATA_ROOT}"
export LIAN_INPUTS="${LIAN_INPUTS:-$DATA_ROOT/torchcell-library/lianMultifunctionalGenomewideCRISPR2019/data/inputs}"
export LIAN_WORK="${LIAN_WORK:-$DATA_ROOT/lian2019-magic-reprocess}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${TORCHCELL_PYTHON:-$HOME/miniconda3/envs/torchcell/bin/python}"
SRA_ENV="${LIAN_SRA_BIN:-$HOME/miniconda3/envs/lian-sra/bin}"  # bowtie 1.3.1, sra-tools 3.4.1
export PATH="$SRA_ENV:$PATH"

mkdir -p "$LIAN_WORK"/{sra,counts}
cp "$HERE/run_manifest.tsv" "$LIAN_WORK/run_manifest.tsv"

echo "[1/6] verify pinned inputs"
( cd "$LIAN_INPUTS" && sha256sum -c "$HERE/inputs.sha256" )

echo "[2/6] download 21 SRA runs (PRJNA504483)"
bash "$HERE/download.sh"

echo "[3/6] build guide_id -> (modality, gene, spacer) map"
"$PY" "$HERE/build_guide_map.py"

echo "[4/6] count barcodes per run (offset-27 exact match)"
"$PY" "$HERE/count_guides.py"

echo "[5/6] CPM + per-round log2(after/before) enrichment"
"$PY" "$HERE/enrichment.py"

echo "[6/6] finalize (control/corrupted flags) + verify output sha256"
"$PY" "$HERE/finalize_enrichment.py"
echo "f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647  $LIAN_WORK/guide_enrichment_final.tsv" | sha256sum -c
echo "REPRODUCE_OK: guide_enrichment_final.tsv matches the pinned sha256"
