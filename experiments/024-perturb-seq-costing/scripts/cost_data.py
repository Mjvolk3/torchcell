# experiments/024-perturb-seq-costing/scripts/cost_data.py
# [[experiments.024-perturb-seq-costing.scripts.cost_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/cost_data
"""Sourced cost line items for split-pool and droplet single-cell RNA-seq.

Two independent published cost models are transcribed here and cross-checked
against each other:

* **Brettner et al. 2024** (yeast SPLiT-seq), Methods 5.1.9 "Detailed cost
  breakdown" -- a prose paragraph, transcribed line by line.
* **Gaisser et al. 2024** (microSPLiT, Nat Protoc), Supplementary Tables 1-2 --
  a spreadsheet, retrieved from Springer ESM
  (``41596_2024_1007_MOESM2_ESM.xlsx``, sha256 ``2b7813a7...``).

They agree on the item that dominates: the custom IDT barcode plates, $7,699.40
(Brettner) vs $7,844.52 (Gaisser). Two labs, two organisms, two years apart,
1.9% apart on the number that IS the start-up cost.

Costs NOT covered by either source (sequencing, oligo pool synthesis, cell
counting) are in ``QUOTES_NEEDED`` with an explicit ``status`` so nothing
unpriced silently reads as priced.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CostItem(BaseModel):
    """One line item, with the scaling rule that decides when it is paid again.

    ``scaling`` is the whole point of this file. Confusing a per-run cost with a
    one-time cost is what produces a staging plan that spends more to do less.
    """

    name: str
    usd: float
    # one_time | per_run | per_sublibrary | per_cell | per_read -- what
    # multiplies this item.
    scaling: str
    # For a durable item: how many runs the single purchase covers.
    runs_covered: int | None = None
    citation_key: str
    line: int | None = None
    quote: str
    note: str | None = None


# =============================================================================
# Brettner 2024 -- yeast SPLiT-seq. All quotes from paper.md line 183.
# =============================================================================

_B = "brettnerUltraHighthroughputMassively2024"

BRETTNER_ITEMS: list[CostItem] = [
    CostItem(
        name="IDT barcode plates (RT r1 + ligation r2 + r3)",
        usd=7699.40,
        scaling="one_time",
        runs_covered=215,
        citation_key=_B,
        line=183,
        quote=(
            "300uL 100uM prediluted plates for the round 1 reverse transcription "
            "and rounds 2 and 3 ligations were ordered from Integrated DNA "
            "Technologies (IDT) for $7699.40. These plates contain enough volume "
            "to complete 300 RT, 250 r2, and 215 r3 ligations with careful "
            "pipetting. Using the lowest capacity of the round 3 plate, this "
            "comes to $36 in barcoding oligos per protocol."
        ),
        note=(
            "The round-3 plate is the binding constraint at 215 uses, so 215 is "
            "the honest run count -- not the 300 the RT plate would allow."
        ),
    ),
    CostItem(
        name="Standalone oligos (lyophilized)",
        usd=270.60,
        scaling="one_time",
        citation_key=_B,
        line=183,
        quote=(
            "The remaining standalone oligos were purchased as lyophilized DNA "
            "for $270.60"
        ),
    ),
    CostItem(
        name="Maxima H Minus Reverse Transcriptase (ThermoFisher EP0753)",
        usd=840.00,
        scaling="per_run",
        citation_key=_B,
        line=183,
        quote=(
            "Thermo Fisher Maxima H minus Reverse Transcriptase (EP0753) now "
            "sells for between $835 and $845 depending on the vendor. ... These "
            "reagents are enough for just over 1 experiment, but not enough for "
            "2, so we calculate them to be repurchased for every instance of the "
            "protocol."
        ),
        note="Midpoint of the quoted $835-845 band.",
    ),
    CostItem(
        name="T4 DNA Ligase (NEB M0202M)",
        usd=270.00,
        scaling="per_run",
        citation_key=_B,
        line=183,
        quote=(
            "The New England Biolabs T4 DNA Ligase (M0202M) is currently listed "
            "at $270"
        ),
        note=(
            "Covers the two published ligation rounds. A third ligation round "
            "scales this term, not the plate term."
        ),
    ),
    CostItem(
        name="RNase inhibitors (SUPERase-In + Enzymatics)",
        usd=127.00,
        scaling="per_run",
        citation_key=_B,
        line=183,
        quote=(
            "Both RNAse inhibitors together cost approximately $630 and last for "
            "about 5 repetitions of the protocol, totaling $127 per run."
        ),
    ),
    CostItem(
        name="Barcoding oligos drawn from the plates",
        usd=36.00,
        scaling="per_run",
        citation_key=_B,
        line=183,
        quote="this comes to $36 in barcoding oligos per protocol",
        note=(
            "This is the amortized draw-down of the $7,699.40 plate purchase. "
            "Counting both the plate as start-up AND this as per-run would "
            "double-count; the model below charges the plate once and this "
            "per run, which is the correct accounting only because the plate is "
            "treated as a capital asset with 215 uses."
        ),
    ),
    CostItem(
        name="Dynabeads MyOne Streptavidin C1",
        usd=14.00,
        scaling="per_sublibrary",
        citation_key=_B,
        line=183,
        quote=(
            "The Dynabeads cost $638 per 2ml, and 44uL are needed per sublibrary "
            "sample, giving a cost-per-sample of $14"
        ),
    ),
    CostItem(
        name="WGS fragmentation + ligation library prep (Enzymatics)",
        usd=39.00,
        scaling="per_sublibrary",
        citation_key=_B,
        line=183,
        quote=(
            "using our method with the WGs fragmentation and ligation kits from "
            "Enzymatics, it comes to $39 per sample"
        ),
    ),
]

# Brettner's own rolled-up totals, kept verbatim so the model can be checked
# against them rather than silently diverging.
BRETTNER_STARTUP_USD = 10_000.0
BRETTNER_PER_RUN_LOW = 1_300.0
BRETTNER_PER_RUN_HIGH = 1_700.0
BRETTNER_PER_SUBLIBRARY = 55.0
BRETTNER_TOTALS_QUOTE = (
    "This gives a start-up cost of approximately $10,000, a per-protocol cost of "
    "approximately $1300-1700 including added leeway for the unpriced "
    "consumables, and an additional $55 per sublibrary."
)  # paper.md:183 (OCR renders "$12300-1700"; the intended range is $1300-1700,
# confirmed by summing the line items above: 840+270+127+36 = $1,273.)

BRETTNER_CELLS_CLAIM = (
    "our yeast-optimized SPLiT-seq method can process approximately 400,000 "
    "cells for approximately $2000 (a full cost breakdown is provided in the "
    "methods), while droplet-based methods would cost approximately $2000 per "
    "sample, with typically up to 10K cells per sample and up to eight samples"
)  # paper.md:31


# =============================================================================
# Gaisser 2024 -- microSPLiT, Supplementary Tables 1-2 (Springer ESM xlsx).
# Independent replication of the same platform's economics.
# =============================================================================

_G = "gaisserHighthroughputSinglecellTranscriptomics2024"
_G_ESM = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41596-024-01007-w/"
    "MediaObjects/41596_2024_1007_MOESM2_ESM.xlsx"
)
_G_ESM_SHA256 = "2b7813a72fd813167de4dbb547385cb7e912a34153b39048a1c529821b9dfd3c"

# Supplementary Table 1, "Cost Estimate of microSPLiT". Header notes: "*All
# prices listed are calculated before tax", "**All prices are calaulated for the
# use of a 96-well plate with one sublibrary" (sic).
GAISSER_TABLE1 = {
    "1_sample": {
        "reagents": 710.24,
        "consumables": 80.54,
        "barcodes": 20.67,
        "total": 790.78,
    },
    "24_samples": {
        "reagents": 1858.92,
        "consumables": 156.23,
        "barcodes": 20.67,
        "total": 2035.82,
    },
    "48_samples": {
        "reagents": 2929.26,
        "consumables": 246.60,
        "barcodes": 20.67,
        "total": 3196.53,
    },
    "start_up": 8142.52,
}

GAISSER_BARCODE_PLATE_USD = 7844.52  # "Bardcode Plates, 100uM, Custom IDT" (sic)

# Supplementary Table 2 -- cross-method comparison, verbatim strings.
GAISSER_TABLE2 = {
    "microSPLiT": {
        "cost_usd": 773.15,
        "cells": 45_000,
        "startup_usd": 8142.52,
        "quote": "One sample (up to 45,000 cells) = $773.15",
    },
    "microSPLiT_1M": {
        "cost_usd": 2510.00,
        "cells": 1_000_000,
        "startup_usd": 8142.52,
        "quote": "one sample (up to 1M cells) = $2,510",
    },
    "PETRI-seq": {
        "cost_usd": 367.00,
        "cells": 10_000,
        "startup_usd": 4593.00,
        "quote": "Cell and library preparation (~10,000 cells) = $367",
    },
    "ProBac-seq": {
        "cost_usd": 1649.00,
        "cells": 10_000,
        "startup_usd": 942.25,
        "quote": "Cost per sample (up to 10,000 cells) = $1,649",
    },
    "MATQ-seq": {
        "cost_usd": 1000.00,
        "cells": 20,
        "startup_usd": None,
        "quote": "Cost per library (20 cells) = $800-$1,200",
    },
}

# Inconsistencies inside the authors' own workbook. Recorded, not silently fixed.
GAISSER_CAVEATS = [
    "24-sample reagent total: overview sheet 1858.92 vs reagents sheet 1870.28.",
    "One-sample microSPLiT total: Table 1 says 790.78, Table 2 says 773.15.",
    "The $2,510 (1M cells) figure has no line-item breakdown in the workbook.",
]

# Largest reagent line items, 1-sample column (for the "what actually costs
# money" narrative).
GAISSER_TOP_REAGENTS = {
    "Maxima H Minus RT": 330.85,
    "T4 DNA ligase": 104.00,
    "Superase-IN": 89.96,
    "Kapa HiFi": 26.52,
    "Poly(A) Polymerase": 16.06,
}


# =============================================================================
# Published per-cell library-prep costs, for the droplet/plate comparison.
# =============================================================================


class PerCellCost(BaseModel):
    """A published or derived dollars-per-cell figure.

    ``derived`` is load-bearing. Only two of these are quoted as a per-cell rate
    by their authors (Jariani's $0.30 and $4.15). The rest are a total divided
    by a cell count that the same source supplies -- honest arithmetic, but not
    something any paper says, so it must not be cited as if it were.

    Every entry here is LIBRARY PREP ONLY. None includes sequencing, and none
    is per *usable* cell. Both omissions flatter split-pool most, because it is
    the method that spends the most reads per usable UMI and discards the
    largest fraction of the cells it sequences.
    """

    label: str
    usd_per_cell: float
    organism: str
    isolation: str
    scope: str
    citation_key: str
    line: int | None = None
    quote: str
    derived: bool = False
    derivation: str | None = None


PER_CELL_COSTS: list[PerCellCost] = [
    PerCellCost(
        label="MATQ-seq",
        usd_per_cell=50.0,
        organism="bacteria",
        isolation="plate",
        scope="library preparation only",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        quote="Cost per library (20 cells) = $800-$1,200",
        derived=True,
        derivation="midpoint $1,000 / 20 cells (Suppl. Table 2)",
    ),
    PerCellCost(
        label="Plate FACS + STRT-seq (Nadal-Ribelles 2019)",
        usd_per_cell=4.15,
        organism="S. cerevisiae",
        isolation="plate",
        scope="library preparation only, excludes sequencing",
        citation_key="jarianiNewProtocolSinglecell2020",
        line=172,
        quote="compared to previously reported costs of $4.15/cell",
    ),
    PerCellCost(
        label="10x Chromium 3' v2 (Jariani 2020)",
        usd_per_cell=0.30,
        organism="S. cerevisiae",
        isolation="droplet",
        scope="library preparation only, excludes sequencing",
        citation_key="jarianiNewProtocolSinglecell2020",
        line=172,
        quote=(
            "the ability to increase sample size in droplet-based methods allows "
            "the library preparation cost to be greatly reduced to $0.3/cell "
            "compared to previously reported costs of $4.15/cell "
            "(Nadal-Ribelles et al., 2019)"
        ),
    ),
    PerCellCost(
        label="ProBac-seq",
        usd_per_cell=0.1649,
        organism="bacteria",
        isolation="droplet",
        scope="library preparation only",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        quote="Cost per sample (up to 10,000 cells) = $1,649",
        derived=True,
        derivation="$1,649 / 10,000 cells (Suppl. Table 2)",
    ),
    PerCellCost(
        label="PETRI-seq",
        usd_per_cell=0.0367,
        organism="bacteria",
        isolation="split_pool",
        scope="cell and library preparation",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        quote="Cell and library preparation (~10,000 cells) = $367",
        derived=True,
        derivation="$367 / 10,000 cells (Suppl. Table 2)",
    ),
    PerCellCost(
        label="microSPLiT (1 sublibrary)",
        usd_per_cell=0.01718,
        organism="bacteria",
        isolation="split_pool",
        scope="cell and library preparation",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        quote="One sample (up to 45,000 cells) = $773.15",
        derived=True,
        derivation="$773.15 / 45,000 cells (Suppl. Table 2)",
    ),
    PerCellCost(
        label="SPLiT-seq (Brettner 2024)",
        usd_per_cell=0.005,
        organism="S. cerevisiae",
        isolation="split_pool",
        scope="reagents only, excludes sequencing",
        citation_key="brettnerUltraHighthroughputMassively2024",
        line=31,
        quote=(
            "our yeast-optimized SPLiT-seq method can process approximately "
            "400,000 cells for approximately $2000"
        ),
        derived=True,
        derivation="$2,000 / 400,000 cells",
    ),
    PerCellCost(
        label="microSPLiT (full scale)",
        usd_per_cell=0.00251,
        organism="bacteria",
        isolation="split_pool",
        scope="cell and library preparation",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        quote="one sample (up to 1M cells) = $2,510",
        derived=True,
        derivation=(
            "$2,510 / 1,000,000 cells. NOTE: the authors give no line-item "
            "breakdown for the $2,510 figure anywhere in the workbook."
        ),
    ),
]

# The spread these figures cover, which is the reason the comparison is worth
# making at all: MATQ-seq to microSPLiT is a factor of ~20,000 in reagent cost
# per cell. It is also why the figure needs a log axis and a health warning.
PER_CELL_SPREAD = 50.0 / 0.00251


# =============================================================================
# Everything neither paper prices. Explicit, so an unpriced item cannot be
# mistaken for a priced one.
# =============================================================================


class QuoteNeeded(BaseModel):
    item: str
    why_it_matters: str
    status: str  # "needs_quote" | "list_price_found"
    working_value_usd: float | None = None
    source: str | None = None


QUOTES_NEEDED: list[QuoteNeeded] = [
    QuoteNeeded(
        item="Illumina sequencing (per flow cell / per lane)",
        why_it_matters=(
            "The single largest term at genome scale, and the ONLY term that "
            "scales with cells x depth. Neither Brettner nor Gaisser prices it."
        ),
        status="needs_quote",
        source="UIUC Roy J. Carver Biotechnology Center DNA Services",
    ),
    QuoteNeeded(
        item="sgRNA oligo pool synthesis (~38,000 members, ~100 nt)",
        why_it_matters=(
            "One-time per library. MAGIC used CustomArray chips; the pools are "
            "NOT on Addgene, so a rebuild means re-synthesis."
        ),
        status="needs_quote",
    ),
    QuoteNeeded(
        item="Cell counting / concentration QC",
        why_it_matters=(
            "SPLiT-seq loads a fixed cells-per-well; a miscount propagates "
            "straight into the collision rate. Brettner used a Beckman Coulter "
            "cell counter, Jariani a Bio-Rad TC20."
        ),
        status="needs_quote",
    ),
    QuoteNeeded(
        item="rRNA depletion reagents (Cas9 + transcribed guides)",
        why_it_matters=(
            "Brandner's version cut reads-per-usable-UMI ~13x. Cheap in "
            "reagents, enormous in sequencing saved."
        ),
        status="needs_quote",
    ),
]
