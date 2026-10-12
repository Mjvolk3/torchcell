#!/bin/bash
#SBATCH --job-name=036-ocr-nichols2011-mmc8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/ecoli-nichols2011/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/ecoli-nichols2011/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_ocr_nichols2011_mmc8.slurm
#
# Issue #691: OCR the owner's deposit of the Nichols 2011 Cell supplemental information
# PDF (si/mmc8.pdf: the article, pages 1 to 14, then the Supplemental Information) with
# MinerU, in place, beside the PDF in the raw mirror. The literature mirror has no key for
# this paper (the Zotero item REEL3RAD has no PDF), so the loader quotes this OCR.
# torchcell.literature.ocr.ocr_pdf is the function `lit_capture_si --ocr-recorded` and
# `reocr_si` call; it writes si/mmc8.md, the layout JSON, the figures under
# si/images/mmc8/ and the processing record si/mmc8_ocr_provenance.json (MinerU version,
# effective DPI, exact command, the PDF's sha256). CPU mode: the GPU cards are held by
# training. Sizing: the 96-page Ishii 2007 SOM took 21 min 50 s and 7.2 GB RSS on 16 CPUs
# (job 4156); this PDF has 23 pages, so 4 CPUs and 12 G fit a backfill slot.
#
#   sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_ocr_nichols2011_mmc8.slurm
set -euo pipefail
echo "Running on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
PROJECT_ROOT="/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/ecoli-nichols2011"
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
PDF="$DATA_ROOT/torchcell-raw/nicholsPhenotypicLandscapeBacterial2011/si/mmc8.pdf"
echo "e0c4e40678d83bc47252d8e47cad64bdd8178ca4713cad2496e0a72b77909e2c  $PDF" | sha256sum -c -
python -c "
import sys
from torchcell.literature.ocr import ocr_pdf
print(ocr_pdf(sys.argv[1], device_mode='cpu', timeout=6000))
" "$PDF"
echo "finished at $(date)"
