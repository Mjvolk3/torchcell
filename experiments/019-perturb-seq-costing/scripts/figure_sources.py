# experiments/019-perturb-seq-costing/scripts/figure_sources.py
# [[experiments.019-perturb-seq-costing.scripts.figure_sources]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/figure_sources
"""Where every hand-drawn number in this document's draw.io figures comes from.

Companion to ``notes-tex/common/figure_provenance.py``, which defines the model
and the check; this file holds the records. The matplotlib figures need nothing
here -- their scripts read the data -- so this covers Figs. 1--3 only.

The throughput band of Fig. 1 is the reason this file exists. Reviewing it
against the primaries turned up three separate errors of the same family, all of
them "a number that is true of something, attached to the wrong thing":

* the C1 row showed 163 cells, which is Gasch et al.'s total across four chip
  runs, on an axis reading "cells profiled per experiment";
* the microwell row showed ~75,000 from mDrop-seq, which is a droplet method,
  and predated the 1.06-million-cell microwell atlas that now exists;
* the combinatorial row's upper bound of 1,000,000 was Gaisser et al.'s protocol
  capacity claim, the same figure that had to be removed from Fig. 4.

Run:  python experiments/019-perturb-seq-costing/scripts/figure_sources.py
"""

from __future__ import annotations

import os.path as osp
import sys

sys.path.insert(
    0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(
        osp.abspath(__file__))))), "notes-tex", "common")
)

from figure_provenance import FigureNumber, check  # noqa: E402

TAXONOMY = "scrnaseq-method-taxonomy"
SPLITSEQ = "splitseq-barcoding"
DROPLET = "droplet-10x-barcoding"
SCIFI = "scifi-fluidic-indexing"

RECORDS: list[FigureNumber] = [
    # --- Fig. 1a, the throughput band ---------------------------------------
    FigureNumber(
        figure=TAXONOMY, panel="a", element="Microfluidic chip (C1) throughput",
        value="up to 96 per chip",
        citation_key="nadal-ribellesRiseSinglecellTranscriptomics2024",
        quote=(
            "It uses an integrated fluidic circuit to isolate and process up to "
            "96 cells at once."
        ),
        note=(
            "Was drawn as ~160 cells, which is Gasch et al.'s 163-cell study "
            "total across four chip runs -- a study total on a per-experiment "
            "axis. The capacity of one chip is the comparable quantity."
        ),
    ),
    FigureNumber(
        figure=TAXONOMY, panel="a", element="FACS into plates throughput",
        value="~285 cells",
        citation_key="nadal-ribellesSensitiveHighthroughputSinglecell2019",
        quote="we applied yscRNA-seq to 285 individual yeast cells",
    ),
    FigureNumber(
        figure=TAXONOMY, panel="a", element="Microdissection throughput",
        value="~2,000 cells",
        quote=(
            "Microdissection coupled with imaging has been used to profile the "
            "transcriptomes of Schizosaccharomyces pombe across a variety of "
            "environmental stressors and S. cerevisiae during aging (Saint et "
            "al., 2019; Wang et al., 2022)"
        ),
        note=(
            "NOT SOURCED TO A NUMBER. The review names the studies but states no "
            "cell count, and neither primary is in the mirror. The ~2,000 is an "
            "order-of-magnitude placeholder and is the one number in this figure "
            "that should not be quoted."
        ),
    ),
    FigureNumber(
        figure=TAXONOMY, panel="a", element="Droplet throughput range",
        value="6,000-100,000",
        citation_key="boocockSinglecellEQTLMapping2025",
        quote=(
            "applied it to over 100,000 single cells from three crosses "
            "[upper bound; lower bound is Jariani et al.'s 6,118 cells]"
        ),
    ),
    FigureNumber(
        figure=TAXONOMY, panel="a", element="Microwell array throughput",
        value="up to 1,061,865",
        citation_key="nadal-ribellesSinglecellResolvedGenotypephenotype2025",
        quote=(
            "we used a microwell-based platform for single-cell isolation ... "
            "We profiled a total of 1.061.865 cells"
        ),
        note=(
            "Was ~75,000, taken from mDrop-seq -- which is a droplet method, not "
            "a microwell one. Replaced with the microwell study that exists."
        ),
    ),
    FigureNumber(
        figure=TAXONOMY, panel="a", element="Combinatorial indexing throughput",
        value="25,000-240,000 per run",
        citation_key="brettnerUltraHighthroughputMassively2024",
        quote=(
            "Reverse transcription (in situ) was performed in 25 uL reactions in "
            "48 wells of a 100 uL 96-well plate ... 200,000 cells/mL "
            "[= ~5,000 cells x 48 wells = ~240,000 barcoded per run; lower bound "
            "is Kuchina et al.'s 25,214-cell combined dataset]"
        ),
        note=(
            "The upper bound was 1,000,000, which is Gaisser et al.'s protocol "
            "capacity claim (\"enables ... up to 1 million\"), not a profiled "
            "count. Same correction as Fig. 4."
        ),
    ),
    # --- Fig. 2, split-pool barcoding ---------------------------------------
    FigureNumber(
        figure=SPLITSEQ, panel="1", element="round-1 wells and cells per well",
        value="48 wells, ~5,000 cells each",
        citation_key="brettnerUltraHighthroughputMassively2024",
        quote=(
            "Reverse transcription (in situ) was performed in 25 uL reactions in "
            "48 wells of a 100 uL 96-well plate with each well containing "
            "wellspecific, barcoded primers ... 200,000 cells/mL"
        ),
    ),
    FigureNumber(
        figure=SPLITSEQ, panel="3", element="barcode space over three rounds",
        value="963 = 884,736",
        citation_key="brettnerUltraHighthroughputMassively2024",
        quote=(
            "Cells are then pooled and randomly split into a new 96-well plate, "
            "and a well-specific barcode is attached"
        ),
        note=(
            "Protocol capacity. The published run loaded 48 wells in round 1, so "
            "its realized space was 48x96x96 = 442,368 (Sec. 3.1)."
        ),
    ),
    FigureNumber(
        figure=SPLITSEQ, panel="4", element="sublibrary size",
        value="5,000-20,000 cells",
        citation_key="brettnerUltraHighthroughputMassively2024",
        quote=(
            "In experiment 3, the barcoded cells were evenly divided into 10 "
            "sublibraries. The two sequenced sublibraries returned ~5,500 and "
            "10,000 barcoded cells that passed computational filtering."
        ),
    ),
    FigureNumber(
        figure=SPLITSEQ, panel="1", element="rRNA fraction of recovered transcripts",
        value="93.7%",
        citation_key="brettnerUltraHighthroughputMassively2024",
        line=70,
        quote=(
            "On average, the transcript proportions we recover are 93.7% rRNA, "
            "0.005% tRNA, 0.04% ncRNA, and 5.75% mRNA."
        ),
    ),
    # --- Fig. 3, droplet barcoding ------------------------------------------
    FigureNumber(
        figure=DROPLET, panel="a", element="doublet rate at 20,000 cells",
        value="~8% at 20,000 recovered",
        citation_key="jarianiNewProtocolSinglecell2020",
        quote=(
            "[10x Chromium specification for the GEM-X/v3.1 chemistry priced in "
            "Sec. 5; the protocol paper used 3' v2]"
        ),
        note=(
            "MANUFACTURER SPECIFICATION, not a measurement from a mirrored paper. "
            "Verify against the current 10x user guide before quoting."
        ),
    ),
    FigureNumber(
        figure=DROPLET, panel="b", element="digestion timing and cell-count drop",
        value="6 min hold, 200-fold drop within 20 min at 53 C",
        citation_key="jarianiNewProtocolSinglecell2020",
        quote=(
            "cell counts hold on ice and through the 6 min of droplet "
            "generation, then drop 200-fold within 20 min at 53 C"
        ),
    ),
    FigureNumber(
        figure=DROPLET, panel="c", element="bead barcode and UMI lengths",
        value="16 nt CB, 12 nt UMI, 118 cycles",
        citation_key="jarianiNewProtocolSinglecell2020",
        quote=(
            "[10x 3' v3/GEM-X read layout; cycle count derived in "
            "experiments/019-perturb-seq-costing/scripts/read_structure.py]"
        ),
        note="Layout is the current GEM-X/v3.1 chemistry, not the v2 Jariani ran.",
    ),

    # --- Fig. 4 (scifi-RNA-seq), Datlinger et al. 2021 -----------------------
    # Panel a is the measured droplet-imaging series; panel b's barcode-space
    # figure is a MODELED bound and is labeled as such on the canvas, which is
    # the distinction the first review found missing elsewhere.
    FigureNumber(
        figure=SCIFI, panel="a", element="maximum recommended load",
        value="15,300 nuclei per channel",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=26,
        quote=(
            "We first assessed the maximum recommended loading concentration "
            "(15,300 nuclei per microfluidic channel)."
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="a", element="occupancy at recommended load",
        value="16.4% occupied; mean 0.2 nuclei per droplet",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=26,
        quote=(
            "Counting 609 droplet images, we found that only 16.4% of droplets "
            "contained one or more nuclei (mean number of nuclei per droplet: 0.2)."
        ),
        note="Measured by counting 609 droplet images, not inferred from Poisson.",
    ),
    FigureNumber(
        figure=SCIFI, panel="a", element="overloaded condition",
        value="100x overloaded, 1.53 million nuclei per channel",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=26,
        quote=(
            "Remarkably, even 100-fold overloading (1.53 million nuclei per "
            "channel) resulted in a stable droplet emulsion and did not clog the "
            "microfluidic system"
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="a", element="occupancy when overloaded",
        value="95.5% fill; mean 9.6 nuclei per droplet",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=26,
        quote=(
            "up to a droplet fill rate of 95.5% and an average of 9.6 nuclei per "
            "droplet (1.53 million nuclei per channel), with highly consistent "
            "droplet diameter"
        ),
        note=(
            "The droplet cartoons in panel a show three nuclei, not 9.6; they are "
            "schematic and the canvas says so. The number is the measured mean."
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="b", element="round1 plate format",
        value="one 384-well plate",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=34,
        quote=(
            "permeabilized cells or nuclei are first preindexed with barcoded "
            "oligo-dT primers by reverse transcription on multiwell plates (we use "
            "one 384-well plate with well-specific primers)"
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="b", element="round2 barcode space",
        value="737,280 round2 barcodes",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=30,
        quote=(
            "Using the 737,280 distinct microfluidic (round2) barcodes provided by "
            "the Chromium ATAC reagents"
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="b", element="resolvable transcriptomes at 96 round1 wells",
        value="1 million from 96 round1 wells",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=30,
        quote=(
            "our analyses indicated that scifi-RNA-seq can resolve 1 million "
            "single-cell transcriptomes already with 96 round1 indices"
        ),
        note=(
            "MODELED, not observed: it comes from a zero-inflated Poisson model "
            "plus Monte Carlo simulation of barcode collisions. The canvas says "
            "'a modeled bound, not a measurement' for exactly this reason."
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="a", element="large-run yield, left caption",
        value="383,000 loaded returned 151,788 transcriptomes from one channel",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=57,
        quote=(
            "we performed a large-scale scifi-RNA-seq experiment with 383,000 "
            "nuclei loaded into a single microfluidic channel of the Chromium "
            "system ... This experiment resulted in 151,788 single-cell "
            "transcriptomes passing quality control"
        ),
        note=(
            "Deliberately separated from the occupancy numbers beside it. Those "
            "were counted under a microscope at a different loading "
            "concentration; this is a sequencing yield, and it is the only scifi "
            "number the cost model consumes."
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="bottom note",
        element="channel saving carried into the budget",
        value="7.6x more cells, so 137 channels become 18",
        citation_key=None,
        quote=(
            "[DERIVED HERE, no source sentence: 151,788 recovered per channel / "
            "20,000 at the UIUC baseline = 7.6x; channel counts from "
            "cost_model.py, ScreenDesign(cells_per_gene=250)]"
        ),
        note=(
            "OURS, not the paper's. 151,788 recovered per channel over the UIUC "
            "baseline of 20,000 is 7.6x; the channel counts come straight from "
            "cost_model.py at 250 cells per target gene. Datlinger et al.'s own "
            "headline is 15-fold, against a ~10,000-cell standard Chromium run "
            "rather than against UIUC's more generous 20,000, so the figure "
            "drawn here is the conservative one. Recompute if the rate card's "
            "cells-per-channel changes."
        ),
    ),
    FigureNumber(
        figure=SCIFI, panel="c", element="round2 attaches by ligation, not RT",
        value="round2 BC (ligated)",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=34,
        quote=(
            "oligonucleotides carrying the microfluidic (round2) barcode (delivered "
            "via Chromium gel beads) are ligated to the cDNA via the 5'-phosphate "
            "group of the reverse transcription primer, directed by a complementary "
            "3'-blocked bridge oligonucleotide"
        ),
    ),
]


def main() -> None:
    figures = [f"{TAXONOMY}.pdf", f"{SPLITSEQ}.pdf", f"{DROPLET}.pdf",
               f"{SCIFI}.pdf"]
    problems = check(RECORDS, figures)
    n_src = sum(1 for r in RECORDS if r.sourced)
    print(f"{len(RECORDS)} drawn numbers, {n_src} tied to a mirrored source, "
          f"{len(RECORDS) - n_src} flagged")
    for r in RECORDS:
        tag = r.citation_key or "UNSOURCED"
        print(f"  [{r.figure} {r.panel}] {r.element}: {r.value}   <- {tag}")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print("\nevery figure has records; every unsourced number carries a note")


if __name__ == "__main__":
    main()
