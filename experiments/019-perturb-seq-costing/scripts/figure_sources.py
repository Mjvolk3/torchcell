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
]


def main() -> None:
    figures = [f"{TAXONOMY}.pdf", f"{SPLITSEQ}.pdf", f"{DROPLET}.pdf"]
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
