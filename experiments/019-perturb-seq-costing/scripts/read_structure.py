# experiments/019-perturb-seq-costing/scripts/read_structure.py
# [[experiments.019-perturb-seq-costing.scripts.read_structure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/read_structure
"""Sequencing read configurations for droplet vs split-pool single-cell RNA-seq.

Both families need PAIRED-END runs, but for opposite reasons, and the two put
the cell barcode on opposite reads. Getting this wrong at submission wastes a
whole flow cell, so it is recorded here rather than left to the core.

* **Droplet (10x, Drop-seq):** one short read is *only* barcode + UMI and one
  long read is *only* cDNA. The cDNA itself is sequenced single-end -- "paired"
  here means "barcode read plus insert read", not two ends of one fragment.
* **Split-pool (SPLiT-seq, microSPLiT, Parse):** reversed. Read 1 is the cDNA
  and Read 2 is the long read, because three ligated barcodes plus linkers plus
  a UMI need ~86 nt. This is why split-pool cannot shrink onto a short kit the
  way a barcode-only read could.

Neither can run on a single-read kit: with one read you get barcodes and no
transcript, or transcript and no cell identity.
"""

from __future__ import annotations

from pydantic import BaseModel


class ReadConfig(BaseModel):
    method: str
    read1_nt: int
    read1_content: str
    read2_nt: int
    read2_content: str
    index_nt: str
    paired_end_required: bool
    min_kit_cycles: int
    phix_fraction: float = 0.0
    citation_key: str | None = None
    line: int | None = None
    quote: str | None = None
    # A second quote from the same paper, used when the read LAYOUT and the read
    # ORDER are stated in different places. Kept separate rather than
    # concatenated so each claim keeps its own line number.
    quote_order: str | None = None
    quote_order_line: int | None = None
    note: str = ""


READ_CONFIGS: list[ReadConfig] = [
    ReadConfig(
        method="SPLiT-seq / microSPLiT (Brettner, Gaisser, Parse)",
        read1_nt=74,
        read1_content="cDNA",
        read2_nt=86,
        # Order matters and is NOT arbitrary: the UMI arrives on the round-3
        # oligo, so it sits distal to BC3 on the assembled strand and is
        # therefore read FIRST. An earlier version of this row listed the UMI
        # last, which would place it beyond BC1 -- physically impossible, since
        # BC1 is on the round-1 RT primer, adjacent to the cDNA. The STARsolo
        # offsets in the protocol settle it exactly (see quote): UMI 0-9,
        # BC3 10-17, BC2 48-55, BC1 78-85, linkers filling 18-47 and 56-77.
        read2_content="10 nt UMI + BC3 + BC2 + BC1 (with linkers)",
        index_nt="6 nt single, or dual 8 nt -- the sublibrary index (barcode round 4)",
        paired_end_required=True,
        min_kit_cycles=160,
        phix_fraction=0.10,
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        line=1012,
        quote=(
            "Use a paired-end sequencing run with a >=150-bp kit. Set read2 to "
            "86 nt (cell-specific barcodes and UMI). Set read1 to >=74 nt (cDNA "
            "sequence)... Include a 6-nt read 1 index to read sub-library indices "
            "(this is the fourth round of barcodes). Use >=10% PhiX spike-in."
        ),
        # Second, independent quote pinning the ORDER of read 2, which the
        # sequencing-setup step above does not state. STARsolo positions are
        # 0-based, inclusive, as start_end pairs on the named read.
        quote_order=(
            "--soloCBposition 0_10_0_17 0_48_0_55 0_78_0_85 "
            "--soloUMIposition 0_0_0_9"
        ),
        quote_order_line=1046,
        note=(
            "The 10% PhiX is not optional hygiene. Read 2 carries fixed linker "
            "sequence at defined cycles, so base diversity collapses there and "
            "patterned flow cells mis-register clusters without a spike-in. "
            "Budget it as a 10% tax on usable reads."
        ),
    ),
    ReadConfig(
        method="10x Chromium 3' (GEM-X / v3.1)",
        read1_nt=28,
        read1_content="16 nt cell barcode + 12 nt UMI",
        read2_nt=90,
        read2_content="cDNA",
        index_nt="dual 10 nt (i7 + i5), unique dual indices",
        paired_end_required=True,
        min_kit_cycles=118,
        phix_fraction=0.0,
        note=(
            "v2 chemistry (what Jariani used) is 26 nt on Read 1: 16 nt barcode "
            "+ 10 nt UMI. Only ~118 of a 2x150 kit's 300 cycles are used, but "
            "the lane price is flat, so the waste costs nothing extra."
        ),
    ),
    ReadConfig(
        method="Drop-seq / yeastDrop-Seq (Urbonaite)",
        read1_nt=20,
        read1_content="12 nt cell barcode + 8 nt UMI",
        read2_nt=100,
        read2_content="cDNA",
        index_nt="single 8 nt",
        paired_end_required=True,
        min_kit_cycles=120,
        citation_key="urbonaiteYeastoptimizedSinglecellTranscriptomics2021",
        line=118,
        quote=(
            "Sequencing was done using the Illumina HiSeq2500 platform with "
            "2x100 read pairs."
        ),
    ),
]

BRETTNER_SEQUENCING_QUOTE = (
    "Experiment 1 was paired end sequenced on a full lane of the HiSeq X "
    "platform with a six basepair index on read 2. Experiments 2 and 3 were "
    "also sequenced on full lanes of the HiSeq X platform but with dual 8 bp "
    "indices."
)  # brettnerUltraHighthroughputMassively2024 paper.md:187

# --- Why paired-end costs nothing extra at UIUC ------------------------------
# The instinct is that requiring paired-end forces you off the cheap single-read
# rate. At UIUC's published rates it does not, because the largest flow cell is
# paired-end only and its per-read price still wins:
#
#   Nova X 25B PE150, 8+ lanes : $3,180 / 3.20e9 pairs = $0.994 per M read pairs
#   Nova X 10B SR100, 8+ lanes : $1,270 / 1.20e9 reads = $1.058 per M reads
#
# So the paired-end requirement is free here. It would NOT be free at a core
# whose cheapest option is a single-read flow cell.

# --- Index hopping: a split-pool-specific hazard on patterned flow cells ------
# NovaSeq X uses ExAmp clustering on a patterned flow cell, where free adapters
# cause index hopping at ~0.1-2%. For an ordinary RNA-seq library that is a
# minor sample-bleed. For SPLiT-seq it is worse in kind: the sublibrary index IS
# barcode round 4, so a hopped index moves a cell into another sublibrary's
# barcode namespace and manufactures a barcode collision. UNIQUE DUAL INDICES
# are therefore mandatory, not advisory -- which is what Brettner switched to
# after Experiment 1's single 6 bp index.
INDEX_HOPPING_NOTE = (
    "Unique dual indices are mandatory for split-pool libraries on NovaSeq X: "
    "the sublibrary index is a barcode dimension, so index hopping creates "
    "barcode collisions rather than mere sample bleed."
)


def usable_read_fraction(cfg: ReadConfig, valid_barcode_fraction: float = 0.8) -> float:
    """Fraction of purchased reads that become assignable cell data.

    Two independent taxes: the PhiX spike (bought, never mapped) and reads whose
    barcode does not resolve. Gaisser: "Successful experiments will have ~70-90%
    of reads containing valid barcodes."
    """
    return (1.0 - cfg.phix_fraction) * valid_barcode_fraction


if __name__ == "__main__":
    for c in READ_CONFIGS:
        print(f"\n{c.method}")
        print(f"  R1 {c.read1_nt:>3} nt  {c.read1_content}")
        print(f"  R2 {c.read2_nt:>3} nt  {c.read2_content}")
        print(f"  idx        {c.index_nt}")
        print(f"  paired-end required: {c.paired_end_required}")
        print(f"  min kit cycles: {c.min_kit_cycles}, PhiX: {c.phix_fraction:.0%}")
        print(f"  usable read fraction: {usable_read_fraction(c):.2f}")
