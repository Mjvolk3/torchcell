# experiments/019-perturb-seq-costing/scripts/uiuc_core_data.py
# [[experiments.019-perturb-seq-costing.scripts.uiuc_core_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/uiuc_core_data
"""UIUC Roy J. Carver Biotechnology Center published rates, transcribed verbatim.

Source: https://biotech.illinois.edu/dna-services-core/dna-services-pricing/
Retrieved 2026-08-18. The page states: "All pricing is updated, effective
Aug. 1, 2026. Listed rates do not include the off-campus overhead" and "This is
an abridged list of our service prices and subject to change at any time."

Listed rates are the on-campus (internal) rate. External users pay +30.8%
overhead, capped at 20% for State-of-Illinois-subsidized institutions.

Two facts about this core reshape the plan and are easy to miss:

1. **The only Illumina sequencers are the NovaSeq X Plus and the MiSeq i100.**
   There is no NextSeq and no NovaSeq 6000 (two pages still mention a 6000 in
   stale prose; the instrument list and rate table do not). So the mid-scale
   option most split-pool protocols are written against -- Gaisser recommends a
   NextSeq 2000 P2 -- does not exist here. The real choice is a ~25M-read MiSeq
   run or a >=800M-read NovaSeq X lane, with nothing between.
2. **10x CRISPR screening is an offered add-on**, at $263/sample on top of the
   library, so the guide-capture library does not need to be built in-house.
"""

from __future__ import annotations

from pydantic import BaseModel

PRICING_URL = "https://biotech.illinois.edu/dna-services-core/dna-services-pricing/"
INSTRUMENTS_URL = "https://biotech.illinois.edu/dna-services-core/dna-services-instruments/"
TENX_URL = (
    "https://biotech.illinois.edu/dna-services-core/"
    "dna-10x-single-cell-submission-requirements-and-other-information/"
)
RATES_EFFECTIVE = "2026-08-01"
RETRIEVED = "2026-08-18"

EXTERNAL_OVERHEAD = 0.308
EXTERNAL_OVERHEAD_STATE_CAP = 0.20
OVERHEAD_QUOTE = (
    "External users are charged an additional 30.8% overhead fee, which is "
    "capped at 20% for those users from State of Illinois subsidized "
    "universities and government agencies."
)


class SeqOption(BaseModel):
    """One sequencing configuration, priced per lane."""

    label: str
    instrument: str
    flow_cell: str
    read_type: str
    usd_per_lane: float
    read_pairs_per_lane: int  # fragments, NOT total reads
    lanes_per_flow_cell: int | None = None

    @property
    def usd_per_million_read_pairs(self) -> float:
        """The only number that lets configurations be compared honestly."""
        return self.usd_per_lane / (self.read_pairs_per_lane / 1e6)


# NovaSeq X Plus. read_pairs_per_lane from the instruments-page output table
# ("# Single-Reads or Read Pairs Per lane"). 25B is 8 lanes per flow cell.
NOVASEQ_X: list[SeqOption] = [
    SeqOption(
        label="Nova X 1.5B SR100, per lane",
        instrument="NovaSeq X Plus",
        flow_cell="1.5B",
        read_type="single-read 100 bp",
        usd_per_lane=1360.0,
        read_pairs_per_lane=800_000_000,
    ),
    SeqOption(
        label="Nova X 1.5B PE150, per lane",
        instrument="NovaSeq X Plus",
        flow_cell="1.5B",
        read_type="paired-read 2x150",
        usd_per_lane=1580.0,
        read_pairs_per_lane=800_000_000,
    ),
    SeqOption(
        label="Nova X 10B PE150, SPLIT LANE",
        instrument="NovaSeq X Plus",
        flow_cell="10B",
        read_type="paired-read 2x150",
        usd_per_lane=1160.0,
        read_pairs_per_lane=600_000_000,  # DERIVED: half of the 1.2B lane
    ),
    SeqOption(
        label="Nova X 10B PE150, per lane",
        instrument="NovaSeq X Plus",
        flow_cell="10B",
        read_type="paired-read 2x150",
        usd_per_lane=2000.0,
        read_pairs_per_lane=1_200_000_000,
        lanes_per_flow_cell=8,
    ),
    SeqOption(
        label="Nova X 10B PE150, 8+ lanes",
        instrument="NovaSeq X Plus",
        flow_cell="10B",
        read_type="paired-read 2x150",
        usd_per_lane=1710.0,
        read_pairs_per_lane=1_200_000_000,
        lanes_per_flow_cell=8,
    ),
    SeqOption(
        label="Nova X 25B PE150, per lane",
        instrument="NovaSeq X Plus",
        flow_cell="25B",
        read_type="paired-read 2x150",
        usd_per_lane=3380.0,
        read_pairs_per_lane=3_200_000_000,
        lanes_per_flow_cell=8,
    ),
    SeqOption(
        label="Nova X 25B PE150, 8+ lanes",
        instrument="NovaSeq X Plus",
        flow_cell="25B",
        read_type="paired-read 2x150",
        usd_per_lane=3180.0,
        read_pairs_per_lane=3_200_000_000,
        lanes_per_flow_cell=8,
    ),
]

# MiSeq i100. Its role in this project is QC, not production -- see the note.
MISEQ_I100: list[SeqOption] = [
    SeqOption(
        label="i100 titration run (26nt + indexes only)",
        instrument="MiSeq i100",
        flow_cell="titration",
        read_type="26 nt + indexes",
        usd_per_lane=398.0,
        read_pairs_per_lane=5_000_000,  # DERIVED: titration uses the 5M kit
    ),
    SeqOption(
        label="i100 5M 2x150nt",
        instrument="MiSeq i100",
        flow_cell="5M",
        read_type="paired-read 2x150",
        usd_per_lane=661.0,
        read_pairs_per_lane=5_000_000,
    ),
    SeqOption(
        label="i100 25M SR100nt",
        instrument="MiSeq i100",
        flow_cell="25M",
        read_type="single-read 100",
        usd_per_lane=986.0,
        read_pairs_per_lane=25_000_000,
    ),
    SeqOption(
        label="i100 25M 2x150nt",
        instrument="MiSeq i100",
        flow_cell="25M",
        read_type="paired-read 2x150",
        usd_per_lane=1450.0,
        read_pairs_per_lane=25_000_000,
    ),
]

# The cheapest per-read production option, used as the default in the model.
CHEAPEST_PRODUCTION = NOVASEQ_X[-1]  # Nova X 25B PE150, 8+ lanes


# --- Flow-cell scale comparison ----------------------------------------------
# Reads per lane and lanes per flow cell, so the platforms can be compared on
# total output rather than on a per-lane price alone. The NovaSeq X rows are
# derived from the UIUC page's own per-lane read counts and check out against
# Illumina's flow-cell naming (800M x 2 = 1.6B ~ "1.5B"; 1.2B x 8 = 9.6B ~ "10B";
# 3.2B x 8 = 25.6B ~ "25B"), which is a useful confirmation that the core's
# quoted per-lane numbers are read PAIRS.
#
# NOT_AT_UIUC rows are MANUFACTURER SPECIFICATIONS, not core rates -- the core
# does not have these instruments, so there is no price and nothing to verify
# against a rate sheet. They are here only to answer "how does what we have
# compare", and they are flagged wherever they are rendered.


class FlowCell(BaseModel):
    instrument: str
    flow_cell: str
    read_pairs_per_lane: int
    lanes: int
    available_at_uiuc: bool
    usd_per_lane: float | None = None

    @property
    def total_read_pairs(self) -> int:
        return self.read_pairs_per_lane * self.lanes


FLOW_CELLS: list[FlowCell] = [
    # --- available at UIUC, priced ---
    FlowCell(instrument="MiSeq i100", flow_cell="5M", read_pairs_per_lane=5_000_000,
             lanes=1, available_at_uiuc=True, usd_per_lane=661.0),
    FlowCell(instrument="MiSeq i100", flow_cell="25M", read_pairs_per_lane=25_000_000,
             lanes=1, available_at_uiuc=True, usd_per_lane=1450.0),
    FlowCell(instrument="MiSeq i100", flow_cell="50M", read_pairs_per_lane=50_000_000,
             lanes=1, available_at_uiuc=True, usd_per_lane=2050.0),
    FlowCell(instrument="NovaSeq X Plus", flow_cell="1.5B",
             read_pairs_per_lane=800_000_000, lanes=2, available_at_uiuc=True,
             usd_per_lane=1580.0),
    FlowCell(instrument="NovaSeq X Plus", flow_cell="10B",
             read_pairs_per_lane=1_200_000_000, lanes=8, available_at_uiuc=True,
             usd_per_lane=1710.0),
    FlowCell(instrument="NovaSeq X Plus", flow_cell="25B",
             read_pairs_per_lane=3_200_000_000, lanes=8, available_at_uiuc=True,
             usd_per_lane=3180.0),
    # --- reference only: manufacturer specs, NOT at UIUC, unpriced ---
    FlowCell(instrument="NovaSeq 6000", flow_cell="SP",
             read_pairs_per_lane=400_000_000, lanes=2, available_at_uiuc=False),
    FlowCell(instrument="NovaSeq 6000", flow_cell="S1",
             read_pairs_per_lane=800_000_000, lanes=2, available_at_uiuc=False),
    FlowCell(instrument="NovaSeq 6000", flow_cell="S2",
             read_pairs_per_lane=2_050_000_000, lanes=2, available_at_uiuc=False),
    FlowCell(instrument="NovaSeq 6000", flow_cell="S4",
             read_pairs_per_lane=2_500_000_000, lanes=4, available_at_uiuc=False),
    FlowCell(instrument="NextSeq 2000", flow_cell="P2",
             read_pairs_per_lane=400_000_000, lanes=1, available_at_uiuc=False),
    FlowCell(instrument="NextSeq 2000", flow_cell="P3",
             read_pairs_per_lane=1_200_000_000, lanes=1, available_at_uiuc=False),
]

# --- 10x Genomics at UIUC (Chromium X) ---------------------------------------
TENX_PRICES = {
    "library_kit_provided": 503.0,
    "feature_barcoding_addon_per_sample": 263.0,
    "rna_library_1": 2650.0,
    "rna_library_2_to_7_each": 2240.0,
    "rna_library_8plus_each": 1880.0,
}
TENX_INSTRUMENT_QUOTE = (
    "The 10x Chromium X system produces libraries for 5' or 3' transcriptome "
    "sequencing from single-cells, ATAC-seq, and multiomics"
)
TENX_CRISPR_QUOTE = (
    "Add-ons to the RNASeq libraries include V(D)J, Cell Surface Protein, and "
    "CRISPR Screening"
)
TENX_DEPTH_QUOTE = (
    "For 10x 3' transcriptome V3.1 GEM-X single cell kits, we generally "
    "recommend at least 50k sequencing reads per targeted cell"
)
TENX_DOUBLET_QUOTE = "~8% at 20,000 cells with the latest GEM-X kits"
TENX_CELLS_PER_CHANNEL = 20_000
TENX_RECOMMENDED_READS_PER_CELL = 50_000

# --- QC, cell counting -------------------------------------------------------
QC_PRICES = {
    "qubit": 7.0,
    "fragment_analyzer_1_to_12": 118.0,
    "fragment_analyzer_up_to_48": 307.0,
    "fragment_analyzer_up_to_96": 449.0,
    "library_qc_single": 118.0,
    "library_qc_2_to_8": 270.0,
    "library_pooling_lt_12": 89.0,
    "library_pooling_lt_96": 357.0,
    "quantstudio7_qpcr": 77.0,
    "cmto_cell_counter_self_per_slide": 3.3,  # Bio-Rad TC20, CMtO
    "cmto_yeast_sort_decontamination": 313.0,  # per sort, flat surcharge
}

# Cell counting for 10x is free, and this is stated unusually plainly.
CELL_COUNTER_INSTRUMENT = "Nexcelom (Revvity) Cellometer K2, brightfield + AO/PI"
CELL_COUNT_FREE_QUOTE = (
    "There is no charge to run test counts, either independently or with "
    "assistance from DNA Services employees. None. All consumables for counting "
    "are also included."
)
VIABILITY_GATE_QUOTE = (
    "Cell viability should be >70%, but >80% is better, and over 90% is ideal."
)

# Concentration/QC instruments the core does NOT have (checked explicitly):
# NanoDrop, Agilent TapeStation, Qiagen QIAxpert, and Bioanalyzer-as-a-service.
# Their capillary QC platform is the Fragment Analyzer / Femto Pulse, and
# quantification is Qubit fluorometry + qPCR.
ABSENT_QC_INSTRUMENTS = ["NanoDrop", "TapeStation", "Qiagen QIAxpert", "Bioanalyzer"]
ABSENT_SEQUENCERS = ["NextSeq 500/550/1000/2000", "NovaSeq 6000", "iSeq 100"]

TURNAROUND = {
    "ngs_general": "one to three weeks from the day we receive the samples or libraries",
    "tenx_test_counts": "~1-2+ weeks from when you contact us",
    "tenx_library_construction": "~4-8 weeks from when you contact us",
    "fragment_analyzer_qc": "typically with 24 hours",
}


def cost_for_read_pairs(n_read_pairs: float, option: SeqOption | None = None) -> float:
    """Cost of n read pairs, billed in whole lanes.

    Whole lanes, not a pro-rated rate: a lane is the indivisible purchase (the
    SPLIT LANE rows are the only exception and are priced separately). Rounding
    up is what makes an under-filled run the main way to waste money here.
    """
    opt = option or CHEAPEST_PRODUCTION
    import math

    lanes = math.ceil(n_read_pairs / opt.read_pairs_per_lane)
    return lanes * opt.usd_per_lane
