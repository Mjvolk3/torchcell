# experiments/024-perturb-seq-costing/scripts/cost_model.py
# [[experiments.024-perturb-seq-costing.scripts.cost_model]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/cost_model
"""Genome-scale CRISPRi perturb-seq cost model for S. cerevisiae.

Turns the sourced constants in ``cost_data``/``method_data``/``uiuc_core_data``
into an end-to-end budget: reagents, library construction, and sequencing at
UIUC rates.

Three modelling decisions drive every number and are the ones to argue with:

1. **You pay to sequence cells you will throw away.** Reads are spent on every
   barcoded cell that lands in a sublibrary, not on the subset that survives QC
   and guide assignment. Brandner sequenced 365,000 E. coli and kept 76,068 --
   a 4.8x multiplier on the sequencing bill that a naive per-usable-cell model
   makes invisible.
2. **Sequencing is billed in whole lanes.** Under-filling a lane is the main
   way to waste money, so the model rounds up rather than pro-rating.
3. **mRNA read fraction is the dominant lever, not cells and not reagents.** At
   the published 5.75% mRNA fraction, ~94 of every 100 reads buy rRNA. Every
   sequencing figure below is roughly inversely proportional to this one number.

Run:  python experiments/024-perturb-seq-costing/scripts/cost_model.py
"""

from __future__ import annotations

import math
import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import cost_data as CD
import method_data as MD
import uiuc_core_data as UC

load_dotenv()
RESULTS_DIR = osp.join(
    os.environ["EXPERIMENT_ROOT"], "024-perturb-seq-costing", "results"
)


# =============================================================================
# Design
# =============================================================================


class ScreenDesign(BaseModel):
    """What we are trying to measure, independent of how."""

    n_genes: int = 6_000
    guides_per_gene: int = 6  # MAGIC LibI is exactly 6/gene over 6,295 genes
    n_controls: int = 100  # MAGIC includes 100 non-targeting per library
    n_environments: int = 1
    cells_per_gene: int = 500

    @property
    def n_elements(self) -> int:
        return self.n_genes * self.guides_per_gene + self.n_controls

    @property
    def usable_cells_needed(self) -> int:
        """Cells that must survive QC *and* carry an identified guide."""
        return self.cells_per_gene * self.n_genes * self.n_environments

    @property
    def cells_per_guide(self) -> float:
        """Gene-level power pools all guides for a gene, so this is only the
        per-guide QC/concordance budget -- it does not set the cell count."""
        return self.cells_per_gene / self.guides_per_gene


# =============================================================================
# Platform
# =============================================================================


class Platform(BaseModel):
    """A measurement technology, parameterized by what it actually delivers."""

    name: str
    mrna_umis_per_cell: float
    mrna_read_fraction: float
    # Fraction of barcoded+sequenced cells that pass QC and get a guide call.
    usable_fraction: float
    reads_per_cell: float
    # Split-pool: cells barcoded per protocol run. Droplet: cells per channel.
    cells_per_batch: int
    cost_per_batch_usd: float
    # Split-pool only. Set to None for droplet.
    cells_per_sublibrary: int | None = None
    cost_per_sublibrary_usd: float = 0.0
    startup_usd: float = 0.0
    # Reads bought but never usable: PhiX spike-in plus reads whose barcode does
    # not resolve. Split-pool pays both (>=10% PhiX is a protocol requirement
    # because Read 2's fixed linkers collapse base diversity); droplet pays only
    # the barcode tax.
    phix_fraction: float = 0.0
    valid_barcode_fraction: float = 0.80
    # True when at least one parameter is a projection rather than a measurement
    # in yeast. A field rather than a naming convention, because every table that
    # renders this platform has to mark it, and a marker derived from the name
    # would be lost the moment a label is shortened for a caption.
    projected: bool = False
    notes: str = ""

    @property
    def reads_per_mrna_umi(self) -> float:
        return self.reads_per_cell / self.mrna_umis_per_cell

    @property
    def usable_read_fraction(self) -> float:
        return (1.0 - self.phix_fraction) * self.valid_barcode_fraction


# ---- SPLiT-seq, exactly as Brettner published it -----------------------------
# cells_per_batch: Brettner ran 48 wells x 5,000 cells; scaled to a full 96-well
# RT plate that is 480,000. NOT taken from microSPLiT's 40,000-100,000 cells per
# well -- that is a bacterial loading density and yeast cells are far larger.
# This is the single most consequential unverified number in the model.
SPLITSEQ_PUBLISHED = Platform(
    name="SPLiT-seq (Brettner, as published)",
    mrna_umis_per_cell=410.0,
    mrna_read_fraction=0.0575,
    usable_fraction=0.25,
    reads_per_cell=15_000.0,
    cells_per_batch=480_000,
    cost_per_batch_usd=CD.BRETTNER_PER_RUN_HIGH,
    cells_per_sublibrary=MD.CELLS_PER_SUBLIBRARY_AT_CEILING,
    cost_per_sublibrary_usd=CD.BRETTNER_PER_SUBLIBRARY,
    startup_usd=CD.BRETTNER_STARTUP_USD,
    phix_fraction=0.10,
    notes="93.7% of recovered transcripts are rRNA; polydT+random hexamer priming.",
)

# ---- SPLiT-seq with rRNA depletion, per Brandner's mapSPLiT -------------------
# mRNA fraction 4.6% -> 60.2% and median mRNA UMIs 54 -> 115 (2.1x) in E. coli.
# Applying the same 2.1x to Brettner's 410 gives 861. Both are transfers from a
# bacterial result to yeast and are the model's second-largest assumption.
SPLITSEQ_DEPLETED = Platform(
    name="SPLiT-seq + rRNA depletion",
    mrna_umis_per_cell=861.0,
    mrna_read_fraction=0.602,
    usable_fraction=0.25,
    reads_per_cell=3_000.0,
    cells_per_batch=480_000,
    cost_per_batch_usd=CD.BRETTNER_PER_RUN_HIGH + 200.0,  # +Cas9 & guide synthesis
    cells_per_sublibrary=MD.CELLS_PER_SUBLIBRARY_AT_CEILING,
    cost_per_sublibrary_usd=CD.BRETTNER_PER_SUBLIBRARY,
    startup_usd=CD.BRETTNER_STARTUP_USD,
    phix_fraction=0.10,
    notes="Brandner's Cas9-based depletion: 253 guides vs 5S/16S/23S rRNA.",
)

# ---- 10x Chromium X at UIUC --------------------------------------------------
# Depth from Jackson 2020 (~2,000 mRNA/cell in yeast on 3' v2). UIUC's own
# recommendation is >=50k reads/cell, which is calibrated to mammalian cells; a
# yeast cell has ~1/5 the mRNA, so 20,000 is used here and flagged as a
# deviation from the core's guidance.
TENX = Platform(
    name="10x Chromium X (GEM-X 3')",
    mrna_umis_per_cell=2_000.0,
    mrna_read_fraction=0.85,
    usable_fraction=0.55,
    reads_per_cell=20_000.0,
    cells_per_batch=UC.TENX_CELLS_PER_CHANNEL,
    cost_per_batch_usd=(
        UC.TENX_PRICES["rna_library_8plus_each"]
        + UC.TENX_PRICES["feature_barcoding_addon_per_sample"]
    ),
    cells_per_sublibrary=None,
    startup_usd=0.0,
    notes="No instrument purchase; core runs it. CRISPR screening is a $263 add-on.",
)

# ---- 10x Chromium with scifi preindexing, PROJECTED --------------------------
# A single-variable counterfactual, and it is built that way on purpose. Every
# parameter is copied verbatim from TENX except `cells_per_batch`, which becomes
# Datlinger et al.'s MEASURED recovery of 151,788 transcriptomes from one
# overloaded channel. So the row answers exactly one question -- what does the
# droplet budget look like if a channel goes further, and nothing else changes --
# rather than blending a throughput measurement with guesses about yeast depth.
#
# Three assumptions are carried, and none of them is measured in yeast:
#   1. DEPTH IS HELD AT THE 10x VALUE. scifi's own comparison says it "did not
#      reach the performance of the Chromium workflow on fresh cells", so 2,000
#      UMIs per cell is the optimistic end. What supports holding it rather than
#      discounting it by a guessed factor is that complexity did NOT fall as
#      droplets filled: no trend down to 15 nuclei per droplet. The overloading
#      itself is therefore not what would cost depth; in-cell RT on a fixed,
#      wall-digested yeast cell might be, and that is unmeasured.
#   2. THE CHANNEL PRICE IS THE RNA-LIBRARY PRICE. scifi runs on Chromium ATAC
#      reagents, which are not on the UIUC rate card at all, so the 3' RNA
#      channel rate stands in for them. Substituted, not sourced.
#   3. NO YEAST RUN EXISTS. Every scifi number is from human or mouse material.
#      Preindexing needs reagents to reach the transcriptome in situ, which in
#      yeast means the fixation and wall digestion of Sec. 3.1.
# The startup term is the one addition: a 96-well plate of barcoded oligo-dT RT
# primers, priced at Brettner et al.'s three-plate purchase divided evenly, the
# same per-plate figure Sec. 5.4 already uses to cost a fourth ligation round.
SCIFI_ROUND1_PLATE_USD = CD.BRETTNER_ITEMS[0].usd / 3.0

TENX_SCIFI_PROJECTED = Platform(
    name="10x + scifi preindexing (projected)",
    mrna_umis_per_cell=TENX.mrna_umis_per_cell,
    mrna_read_fraction=TENX.mrna_read_fraction,
    usable_fraction=TENX.usable_fraction,
    reads_per_cell=TENX.reads_per_cell,
    cells_per_batch=MD.SCIFI_RECOVERED_LARGE_RUN,
    cost_per_batch_usd=TENX.cost_per_batch_usd,
    cells_per_sublibrary=None,
    startup_usd=SCIFI_ROUND1_PLATE_USD,
    projected=True,
    notes=(
        "PROJECTED, not measured in yeast. Only cells_per_batch differs from the "
        "10x row: 151,788 recovered per overloaded channel (Datlinger 2021) "
        "against 20,000 at the UIUC baseline."
    ),
)

PLATFORMS = [SPLITSEQ_PUBLISHED, SPLITSEQ_DEPLETED, TENX, TENX_SCIFI_PROJECTED]


# =============================================================================
# Costing
# =============================================================================


class Budget(BaseModel):
    platform: str
    usable_cells: int
    sequenced_cells: int
    n_batches: int
    n_sublibraries: int
    read_pairs: float
    lanes_25b: float
    startup_usd: float
    protocol_usd: float
    sublibrary_usd: float
    sequencing_usd: float
    recurring_usd: float
    first_time_usd: float
    usd_per_usable_cell: float
    usd_per_gene: float
    detail: dict = Field(default_factory=dict)


def budget_for(design: ScreenDesign, platform: Platform) -> Budget:
    usable = design.usable_cells_needed
    sequenced = math.ceil(usable / platform.usable_fraction)
    n_batches = math.ceil(sequenced / platform.cells_per_batch)

    if platform.cells_per_sublibrary:
        n_sub = math.ceil(sequenced / platform.cells_per_sublibrary)
        sublib_usd = n_sub * platform.cost_per_sublibrary_usd
    else:
        n_sub = 0
        sublib_usd = 0.0

    protocol_usd = n_batches * platform.cost_per_batch_usd
    # Reads that must be BOUGHT, which exceeds reads that will be used: the PhiX
    # spike and unresolvable barcodes are paid for at the same per-lane rate.
    read_pairs = sequenced * platform.reads_per_cell / platform.usable_read_fraction
    seq_usd = UC.cost_for_read_pairs(read_pairs)
    lanes = read_pairs / UC.CHEAPEST_PRODUCTION.read_pairs_per_lane

    recurring = protocol_usd + sublib_usd + seq_usd
    return Budget(
        platform=platform.name,
        usable_cells=usable,
        sequenced_cells=sequenced,
        n_batches=n_batches,
        n_sublibraries=n_sub,
        read_pairs=read_pairs,
        lanes_25b=lanes,
        startup_usd=platform.startup_usd,
        protocol_usd=protocol_usd,
        sublibrary_usd=sublib_usd,
        sequencing_usd=seq_usd,
        recurring_usd=recurring,
        first_time_usd=recurring + platform.startup_usd,
        usd_per_usable_cell=recurring / usable,
        usd_per_gene=recurring / design.n_genes,
        detail={
            "reads_per_cell": platform.reads_per_cell,
            "reads_per_mrna_umi": round(platform.reads_per_mrna_umi, 1),
            "mrna_read_fraction": platform.mrna_read_fraction,
            "usable_fraction": platform.usable_fraction,
            "pseudobulk_umis_per_gene": design.cells_per_gene
            * platform.mrna_umis_per_cell,
        },
    )


# =============================================================================
# Power: how many cells per perturbation
# =============================================================================

# Field-standard floor, stated explicitly by Brandner et al.:
# "at least 100 cells following standard approaches in scRNA-seq perturbation
# screens" (brandnerPooledSinglecellCRISPRa2025 paper.md:89).
CELLS_FLOOR = 100

# z for 80% power at alpha=0.05 two-sided.
Z_80_05 = 2.80


def umis_needed_for_fold_change(log2_fc: float, z: float = Z_80_05) -> float:
    """Pseudobulk UMIs of ONE gene needed to call a fold change.

    Shot-noise limit only. For pseudobulk counts K ~ Poisson, Var(log K) ~ 1/K,
    so detecting an effect of size ``log2_fc`` at 80% power needs
    ``K >= (z / (log2_fc * ln2))**2``. This is a floor: it ignores biological
    cell-to-cell variance, which no amount of sequencing depth can beat and
    which is why ``CELLS_FLOOR`` is a separate, independent constraint.
    """
    return (z / (log2_fc * math.log(2))) ** 2


def cells_per_perturbation(
    mrna_umis_per_cell: float,
    target_pseudobulk_umis: float,
    floor: int = CELLS_FLOOR,
) -> int:
    """Binding constraint of the two: biological floor vs depth requirement."""
    return max(floor, math.ceil(target_pseudobulk_umis / mrna_umis_per_cell))


# Pseudobulk UMI tiers, and what each buys. A gene is measurable if its share of
# the mRNA pool puts >= ~16 UMIs (the 2-fold detection threshold) in the pool.
PSEUDOBULK_TIERS = {
    "screen (strong coherent effects only)": 5.0e4,
    "standard (Brettner's own 500-cell heuristic)": 2.0e5,
    "deep (reaches low-abundance regulators)": 1.0e6,
}


def min_copies_per_cell_detectable(
    pseudobulk_umis: float,
    total_mrna_per_cell: int = MD.YEAST_MRNA_PER_CELL_WORKING,
    log2_fc: float = 1.0,
) -> float:
    """Lowest per-cell mRNA copy number whose 2-fold change is detectable."""
    needed = umis_needed_for_fold_change(log2_fc)
    p_min = needed / pseudobulk_umis  # minimum share of the mRNA pool
    return p_min * total_mrna_per_cell


# =============================================================================
# Multiplex (the end goal)
# =============================================================================


def cells_for_main_effects_kplex(
    cells_per_gene_singles: int, n_genes: int, plex: int
) -> int:
    """Total cells to reach a given main-effect precision with k-plex guides.

    Under an additive model, gene g's main effect is informed by every cell whose
    combination contains g. With random k-subsets of T genes and N cells total,
    that is N*k/T cells -- independent of how many distinct combinations the
    library holds. So k-plexing divides the cell requirement by k, which is the
    real economic argument for multiplexed screens.
    """
    return math.ceil(cells_per_gene_singles * n_genes / plex)


def cells_per_named_pair(n_cells: int, n_genes: int, plex: int) -> float:
    """Cells informing one SPECIFIC gene pair under random k-plex assignment.

    P(both g and h in a random k-subset) = k(k-1) / (T(T-1)). This is the number
    that kills random genome-scale multiplexing for interaction mapping, and it
    is why a focused sub-library is the only way to get named interactions.
    """
    return n_cells * plex * (plex - 1) / (n_genes * (n_genes - 1))


# --- Second-order interactions, after Yao et al. 2024 -------------------------
# Yao gives the cell requirement for powering ALL pairwise interactions among p
# targets at k guides per cell:
#
#     n = 400 * C(p,2) / C(k,2)
#
# quoting: "400 represents the number of cells needed to learn a single
# second-order interaction effect with the same power as our conventional screens
# had to learn first-order effects (~100 cells/perturbations), assuming that four
# times as many cells are needed to estimate a second-order interaction effect as
# a first-order effect with the same magnitude"
# (yaoScalableGeneticScreening2024, si1.md:203-211).
#
# This is the same combinatorics as cells_per_named_pair() -- pairs needed over
# pairs delivered per cell -- so the two agree; what Yao supplies is the
# authoritative CONSTANT. 400 = 4x the 100-cell first-order floor, which replaces
# the ~3x variance-summing argument with a published, empirically-anchored value.
CELLS_PER_PAIR_YAO = 400


def cells_for_all_pairs(n_targets: int, plex: int, per_pair: int = CELLS_PER_PAIR_YAO) -> float:
    """Cells to power every pairwise interaction among ``n_targets`` at k-plex."""
    if plex < 2:
        return math.inf
    n_pairs = n_targets * (n_targets - 1) / 2
    pairs_per_cell = plex * (plex - 1) / 2
    return per_pair * n_pairs / pairs_per_cell


def max_targets_for_pairs(n_cells: int, plex: int, per_pair: int = CELLS_PER_PAIR_YAO) -> float:
    """Inverse: how many targets can have ALL their pairs powered, given n cells.

    Reproduces Yao's worked cases -- 100,000 cells at k=3 gives ~39 targets;
    1,000,000 cells at k=8 gives ~374 (si1.md:211).
    """
    if plex < 2:
        return 0.0
    pairs_per_cell = plex * (plex - 1) / 2
    n_pairs = n_cells * pairs_per_cell / per_pair
    # n_pairs = p(p-1)/2  ->  solve for p
    return (1 + math.sqrt(1 + 8 * n_pairs)) / 2


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Sequencing options, ranked by cost per million read pairs --------
    seq_rows = [
        {
            "label": o.label,
            "instrument": o.instrument,
            "flow_cell": o.flow_cell,
            "read_type": o.read_type,
            "usd_per_lane": o.usd_per_lane,
            "read_pairs_per_lane": o.read_pairs_per_lane,
            "usd_per_million_read_pairs": round(o.usd_per_million_read_pairs, 3),
        }
        for o in UC.NOVASEQ_X + UC.MISEQ_I100
    ]
    seq_df = pd.DataFrame(seq_rows).sort_values("usd_per_million_read_pairs")
    seq_df.to_csv(osp.join(RESULTS_DIR, "sequencing_options.csv"), index=False)

    # --- 2. Cells per perturbation, by method depth --------------------------
    pp_rows = []
    for m in MD.METHODS:
        if m.mrna_umis_per_cell is None:
            continue
        for tier, target in PSEUDOBULK_TIERS.items():
            n = cells_per_perturbation(m.mrna_umis_per_cell, target)
            pp_rows.append(
                {
                    "method": m.label,
                    "organism": m.organism,
                    "isolation": m.isolation,
                    "mrna_umis_per_cell": m.mrna_umis_per_cell,
                    "tier": tier,
                    "target_pseudobulk_umis": target,
                    "cells_per_perturbation": n,
                    "binding_constraint": (
                        "biological floor"
                        if n == CELLS_FLOOR
                        else "sequencing depth"
                    ),
                    "cells_for_6000_genes": n * 6000,
                }
            )
    pp_df = pd.DataFrame(pp_rows)
    pp_df.to_csv(osp.join(RESULTS_DIR, "cells_per_perturbation.csv"), index=False)

    # --- 3. Detection ladder -------------------------------------------------
    det_rows = [
        {
            "tier": tier,
            "pseudobulk_umis_per_perturbation": target,
            "min_umis_for_2fold": round(umis_needed_for_fold_change(1.0), 1),
            "min_copies_per_cell_2fold": round(
                min_copies_per_cell_detectable(target, log2_fc=1.0), 2
            ),
            "min_copies_per_cell_1.5fold": round(
                min_copies_per_cell_detectable(target, log2_fc=math.log2(1.5)), 2
            ),
        }
        for tier, target in PSEUDOBULK_TIERS.items()
    ]
    pd.DataFrame(det_rows).to_csv(
        osp.join(RESULTS_DIR, "detection_ladder.csv"), index=False
    )

    # --- 4. Genome-scale budgets --------------------------------------------
    budget_rows = []
    for cells_per_gene in (100, 250, 500):
        design = ScreenDesign(cells_per_gene=cells_per_gene)
        for plat in PLATFORMS:
            b = budget_for(design, plat)
            budget_rows.append(
                {
                    "cells_per_gene": cells_per_gene,
                    "platform": b.platform,
                    "projected": plat.projected,
                    "usable_cells": b.usable_cells,
                    "sequenced_cells": b.sequenced_cells,
                    "n_batches": b.n_batches,
                    "n_sublibraries": b.n_sublibraries,
                    "read_pairs_billions": round(b.read_pairs / 1e9, 2),
                    "lanes_25B_equivalent": round(b.lanes_25b, 2),
                    "protocol_usd": round(b.protocol_usd),
                    "sublibrary_usd": round(b.sublibrary_usd),
                    "sequencing_usd": round(b.sequencing_usd),
                    "recurring_usd": round(b.recurring_usd),
                    "startup_usd": round(b.startup_usd),
                    "first_time_usd": round(b.first_time_usd),
                    "usd_per_usable_cell": round(b.usd_per_usable_cell, 4),
                    "usd_per_gene": round(b.usd_per_gene, 2),
                    "reads_per_mrna_umi": b.detail["reads_per_mrna_umi"],
                    "pseudobulk_umis_per_gene": b.detail["pseudobulk_umis_per_gene"],
                }
            )
    bud_df = pd.DataFrame(budget_rows)
    bud_df.to_csv(osp.join(RESULTS_DIR, "genome_scale_budgets.csv"), index=False)

    # --- 5. Multiplex scaling ------------------------------------------------
    mx_rows = []
    base_cells_per_gene = 250
    for plex in (1, 2, 3, 5, 10):
        n = cells_for_main_effects_kplex(base_cells_per_gene, 6000, plex)
        mx_rows.append(
            {
                "plex": plex,
                "cells_total_for_main_effects": n,
                "cells_per_named_pair": round(cells_per_named_pair(n, 6000, plex), 4),
                "named_pairs_in_design_space": 6000 * 5999 // 2,
                # A focused 200-gene sub-library is the tractable alternative.
                "cells_per_named_pair_200gene_sublibrary": round(
                    cells_per_named_pair(n, 200, plex), 2
                ),
            }
        )
    pd.DataFrame(mx_rows).to_csv(
        osp.join(RESULTS_DIR, "multiplex_scaling.csv"), index=False
    )

    # --- 6. Barcode space ----------------------------------------------------
    bc_rows = []
    for rounds in (3, 4):
        for sublibs in (1, 10, 24, 48, 96):
            B = MD.barcode_space(rounds=rounds, sublibraries=sublibs)
            for n_cells in (45_000, 480_000, 3_000_000, 12_000_000):
                bc_rows.append(
                    {
                        "barcoding_rounds": rounds,
                        "sublibraries": sublibs,
                        "barcode_space": B,
                        "cells_barcoded_together": n_cells,
                        "collision_rate_pct": round(
                            100 * MD.collision_rate(n_cells, B), 3
                        ),
                    }
                )
    pd.DataFrame(bc_rows).to_csv(
        osp.join(RESULTS_DIR, "barcode_collision.csv"), index=False
    )

    # --- console summary -----------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== Cheapest production sequencing (UIUC, eff. 2026-08-01) ===")
    print(seq_df.head(4).to_string(index=False))
    print("\n=== Genome-scale budgets (6,000 genes, 1 environment) ===")
    print(
        bud_df[
            [
                "cells_per_gene",
                "platform",
                "usable_cells",
                "sequenced_cells",
                "n_batches",
                "read_pairs_billions",
                "protocol_usd",
                "sequencing_usd",
                "recurring_usd",
            ]
        ].to_string(index=False)
    )
    print("\n=== Multiplex scaling (target = 250 cells/gene main effect) ===")
    print(pd.DataFrame(mx_rows).to_string(index=False))
    print(f"\nWrote CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
