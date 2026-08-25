# experiments/024-perturb-seq-costing/scripts/figure_sources.py
# [[experiments.024-perturb-seq-costing.scripts.figure_sources]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/figure_sources
"""Where every hand-drawn number in this document's draw.io figures comes from.

Companion to ``notes-tex/common/figure_provenance.py``, which defines the model
and the check; this file holds the records. The matplotlib figures need nothing
here -- their scripts read the data -- so this covers the draw.io figures only.

The throughput band of Fig. 1 is the reason this file exists. Reviewing it
against the primaries turned up three separate errors of the same family, all of
them "a number that is true of something, attached to the wrong thing":

* the C1 row showed 163 cells, which is Gasch et al.'s total across four chip
  runs, on an axis reading "cells profiled per experiment";
* the microwell row showed ~75,000 from mDrop-seq, which is a droplet method,
  and predated the 1.06-million-cell microwell atlas that now exists;
* the combinatorial row's upper bound of 1,000,000 was Gaisser et al.'s protocol
  capacity claim, the same figure that had to be removed from Fig. 4.

Run:  python experiments/024-perturb-seq-costing/scripts/figure_sources.py
"""

from __future__ import annotations

import os.path as osp
import sys

sys.path.insert(
    0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(
        osp.abspath(__file__))))), "notes-tex", "common")
)

from figure_provenance import FigureNumber, HandDrawnFigure, check  # noqa: E402

TAXONOMY = "scrnaseq-method-taxonomy"
SPLITSEQ = "splitseq-barcoding"
DROPLET = "droplet-10x-barcoding"
SCIFI = "scifi-fluidic-indexing"
WORKFLOW = "perturb-seq-workflow"
RRNA = "rrna-depletion-options"

# ONLY draw.io figures appear here. The matplotlib figures (including the
# library-ceiling and Poisson-primer panels added in review) carry their own
# provenance: their scripts read the data, and the caption names the script.
#
# Every hand-drawn figure gets its own provenance table, keyed to the figure's
# own \label so the table caption and the figure caption cross-reference by
# NUMBER in both directions. Adding a figure means adding a row here and its
# records below; render_tex_tables.t15 loops over this and needs no edit.
FIGURES: list[HandDrawnFigure] = [
    HandDrawnFigure(slug=TAXONOMY, tex_label="fig:methods-map",
                    title="the yeast scRNA-seq method taxonomy"),
    HandDrawnFigure(slug=SPLITSEQ, tex_label="fig:splitseq",
                    title="split-pool barcoding"),
    HandDrawnFigure(slug=DROPLET, tex_label="fig:droplet",
                    title="droplet barcoding"),
    HandDrawnFigure(slug=SCIFI, tex_label="fig:scifi",
                    title="combinatorial fluidic indexing"),
    HandDrawnFigure(slug=WORKFLOW, tex_label="fig:workflow",
                    title="the end-to-end workflow"),
    HandDrawnFigure(slug=RRNA, tex_label="fig:rrna",
                    title="ribosomal RNA depletion options"),
]

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
    FigureNumber(
        figure=TAXONOMY, panel="a", element="preindexed droplet throughput",
        value="152,000 preindexed (human; Fig. 4)",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021", line=57,
        quote=(
            "we performed a large-scale scifi-RNA-seq experiment with 383,000 "
            "nuclei loaded into a single microfluidic channel of the Chromium "
            "system ... This experiment resulted in 151,788 single-cell "
            "transcriptomes passing quality control"
        ),
        note=(
            "Sits under the droplet column rather than in a seventh box, because "
            "preindexing is a modifier of droplet capture and not a separate way "
            "of keeping cells apart. Rounded to 152,000 on the canvas, where "
            "every other entry is rounded too. The organism caveat is IN the "
            "label rather than in a footnote: this is a figure of yeast methods "
            "and the one number in it that is not from a microbe has to say so "
            "even when the panel is screenshotted out of the document."
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
            "experiments/024-perturb-seq-costing/scripts/read_structure.py]"
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
        value="383,000 loaded; 151,788 recovered",
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
    # --- Fig. 5, the end-to-end workflow ------------------------------------
    # Panels a and b are structural rather than numeric; the numbers all live in
    # the funnel of panel c, and every one of them is already load-bearing
    # somewhere else in the document.
    FigureNumber(
        figure=WORKFLOW, panel="c", element="cells recovered from those loaded",
        value="25-50% recovered",
        citation_key="brettnerUltraHighthroughputMassively2024", line=68,
        quote=(
            "The two sequenced sublibraries returned ~5,500 and 10,000 barcoded "
            "cells that passed computational filtering. Given we started the "
            "experiment with between 200,000 and 300,000 cells, this estimates a "
            "capturing efficiency between 25% and 50%."
        ),
        note=(
            "A CELL-level efficiency, not a read-level one, and it is the "
            "authors' own back-calculation from a starting count rather than a "
            "directly measured recovery."
        ),
    ),
    FigureNumber(
        figure=WORKFLOW, panel="c", element="sequenced cells discarded at QC",
        value="~75% of sequenced cells discarded",
        citation_key=None,
        quote=(
            "[DERIVED HERE: cost_per_cell_table.py, the ratio of cells passing "
            "QC with an identified guide to cells whose reads were paid for]"
        ),
        note=(
            "OURS, not a published figure. It is the gap between 'sequenced' "
            "and 'usable' that Sec. 5.2 is built on, and it is the reason a "
            "quoted cost per cell understates the real one by ~4x in split-pool."
        ),
    ),
    FigureNumber(
        figure=WORKFLOW, panel="c", element="guide assignment rate q",
        value="21-25% in bacteria, >71% in yeast",
        citation_key="nadal-ribellesSinglecellResolvedGenotypephenotype2025",
        quote=(
            "[Yeast end: genotype identity assigned for more than 71% of cells. "
            "Bacterial end: Brandner et al. raise correct assignment from 5.5% "
            "to 21-25% by adding an enrichment primer at cDNA amplification.]"
        ),
        note=(
            "TWO SOURCES IN ONE ROW, deliberately: the range is the span between "
            "two different organisms and two different constructs, not a "
            "measurement with an uncertainty. Sec. 4.5 shows it enters the "
            "multiplex ceiling as q^k, which is why the span matters so much."
        ),
    ),
    FigureNumber(
        figure=WORKFLOW, panel="c", element="reads surviving all four tolls",
        value="~1 read in 100; ~11 with depletion",
        citation_key=None,
        quote=(
            "[DERIVED HERE: plot_economics.py panel d, composing the PhiX/"
            "barcode, rRNA and cell-QC fractions]"
        ),
        note=(
            "OURS. The rRNA term is Brettner et al.'s 93.7%; the cell-QC term is "
            "the row above. Composing them is this document's arithmetic, and it "
            "is the number that makes depletion the largest single lever."
        ),
    ),
    FigureNumber(
        figure=WORKFLOW, panel="c", element="reads lost to spike-in and barcode",
        value="28 lost to the PhiX spike and to unresolved barcodes",
        citation_key=None,
        quote=(
            "[DERIVED HERE: cost_model.py, phix_fraction=0.10 and "
            "valid_barcode_fraction=0.80, i.e. usable_read_fraction=0.72]"
        ),
        note=(
            "OURS. The canvas used to say 10, which was the PhiX spike alone; the "
            "model applies a 10% spike AND a separate 80% valid-barcode rate, so "
            "usable_read_fraction is 0.72 and 28 reads in 100 are lost at this "
            "stage. The 1-in-100 survival figure was always computed from 0.72 "
            "and was right; only the drawn breakdown was wrong."
        ),
    ),
    FigureNumber(
        figure=WORKFLOW, panel="c", element="ribosomal share of remaining reads",
        value="94 of the remainder are ribosomal",
        citation_key="brettnerUltraHighthroughputMassively2024", line=70,
        quote=(
            "On average, the transcript proportions we recover are 93.7% rRNA, "
            "0.005% tRNA, 0.04% ncRNA, and 5.75% mRNA."
        ),
        note="Drawn rounded to 94 from the sourced 93.7%.",
    ),
    FigureNumber(
        figure=WORKFLOW, panel="b", element="usable fragments, bulk vs single cell",
        value="6 and 2 usable in bulk; 1 and 1 in single cell",
        citation_key=None,
        quote=(
            "[SCHEMATIC, not measured: the cut positions are drawn to illustrate "
            "that reads scale with length in bulk and do not in a 3' assay]"
        ),
        note=(
            "The only numbers on this canvas that are illustrative rather than "
            "sourced. The RATIO is the claim -- six against two is the 3 kb "
            "against 1 kb case in Sec. 2.4 -- and the cut count itself is not a "
            "measurement of anything."
        ),
    ),
    FigureNumber(
        figure=WORKFLOW, panel="a", element="value of rRNA depletion, drawn on canvas",
        value="worth ~$99k (Fig. 6)",
        citation_key=None,
        quote=(
            "[DERIVED HERE: tables/t8-budgets.tex, the 250-cells-per-gene "
            "split-pool rows, $156,670 undepleted less $57,510 depleted = $99,160]"
        ),
        note=(
            "OURS. A difference of two budget rows rather than a figure any "
            "source states, and it is the number the whole rRNA argument turns "
            "on, so it is recorded here as well as in the text."
        ),
    ),

    # --- Fig. 6, rRNA depletion options -------------------------------------
    FigureNumber(
        figure=RRNA, panel="a", element="workflow step numbers",
        value="fragment 134-139, ligate 152-154, index 156-157",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        line=82,
        quote=(
            "The library then undergoes fragmentation and adapter ligation "
            "(Part 2, Steps 134-139 and 152-154). ... Illumina P5 and P7 "
            "sequence adapters and a final sub-library index are also appended "
            "at this final PCR step (Part 2, Step 157). e, A 0.5-0.7x double "
            "size selection then selects out unwanted fragments"
        ),
        note=(
            "This is the correction. An earlier version of Sec. 3.5 placed "
            "depletion BEFORE fragmentation and indexing and argued it had to "
            "be there; the protocol says otherwise, and Brandner et al.'s stated "
            "input is the post-size-selection library. One precision caveat: the "
            "quote names step 157 for the index PCR, and 156 comes from the "
            "numbered protocol body rather than from this sentence."
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="b", element="Cas9 route, guide pool and conditions",
        value="253 sgRNAs",
        citation_key="brandnerPooledSinglecellCRISPRa2025", line=151,
        quote=(
            "We designed 253 sgRNA spacer sequences to target the 5S, 16S, and "
            "23S rRNA sequences in E. coli and P. putida, weighted "
            "proportionally to the regions in the rRNA that appeared more "
            "frequently during NGS after microSPLiT. ... 100 pmoles of the "
            "transcribed, column-purified sgRNA molecules were pre-incubated "
            "with 10 pmoles spCas9 (NEB) ... then incubated at 37 C for 2 hours."
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="b", element="Cas9 route, measured outcome",
        value="4.6% -> 60.2%; 2.2% -> 10.4%",
        citation_key="brandnerPooledSinglecellCRISPRa2025", line=71,
        quote=(
            "With rRNA depletion, the mRNA fraction increased from 4.6% to 60.2% "
            "(Fig. 2B) and median mRNA transcripts per cell more than doubled, "
            "from 54 to 115 [E. coli]; ... the mRNA fraction increased from 2.2% "
            "to 10.4% [P. putida, md:119]"
        ),
        note=(
            "Two organisms, one protocol, and the gap between them is the "
            "transfer risk the figure is drawing attention to. Both figures are "
            "proportions of ALIGNED transcripts computed before single-cell "
            "filtering, not of reads purchased."
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="b", element="RNase H chosen over Cas9",
        value="75-80% rRNA left by the Cas9 arm",
        citation_key="wangSinglecellMassivelyparallelMultiplexed2023", line=23,
        quote=(
            "after testing two approaches for depleting ribosomal sequences from "
            "bulk libraries (Extended Data Fig. 3a-c), we chose an RNase H-based "
            "approach to complete our pipeline [and, md:92:] this is consistent "
            "with our trial runs using Cas9-based rRNA depletion (75-80% rRNA in "
            "the final library, Extended Data Fig. 3a)"
        ),
        note=(
            "The head-to-head. An earlier version of Sec. 3.5 cited M3-seq as "
            "corroborating the Cas9 route; it tested both and chose the other one."
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="b", element="in-situ route, best published result",
        value="~2.5-fold, ~7% of total RNA",
        citation_key="kuchinaMicrobialSinglecellRNA2021", line=36,
        quote=(
            "we then tested polyadenylation with E. coli poly(A) polymerase I "
            "(PAP) ...; 5'-phosphate-dependent exonuclease (\"Terminator\", "
            "Epicentre); and RT with ribosomal RNA-specific probes followed by "
            "RNaseH mediated degradation ... We found that the treatment of "
            "fixed and permeabilized cells with PAP resulted in the highest "
            "(about 2.5-fold, or about 7% of total RNA) enrichment of mRNA reads"
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="b", element="why M3-seq avoids in-situ depletion",
        value="in situ can decrease mRNA capture",
        citation_key="wangSinglecellMassivelyparallelMultiplexed2023", line=23,
        quote=(
            "we noted that depletion of rRNA in situ can decrease mRNA capture "
            "efficiency and thus focused on depleting rRNAs after amplification"
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="b", element="undepleted yeast transcript proportions",
        value="93.7% rRNA, 5.75% mRNA",
        citation_key="brettnerUltraHighthroughputMassively2024", line=70,
        quote=(
            "On average, the transcript proportions we recover are 93.7% rRNA, "
            "0.005% tRNA, 0.04% ncRNA, and 5.75% mRNA."
        ),
        note=(
            "Computed with multi-mapping reads RETAINED (STARsolo, 'the most "
            "basic uniform multi-mapping algorithm', md:195), which is the "
            "convention that inflates apparent rRNA. That makes it comparable to "
            "microSPLiT's numbers and NOT to any unique-only figure -- see the "
            "reporting-convention paragraph in Sec. 3.5."
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="a", element="reads spent on rRNA, drawn on canvas",
        value="roughly 94 of every 100 reads",
        citation_key="brettnerUltraHighthroughputMassively2024", line=70,
        quote=(
            "On average, the transcript proportions we recover are 93.7% rRNA, "
            "0.005% tRNA, 0.04% ncRNA, and 5.75% mRNA."
        ),
        note="Drawn rounded to 94 from the sourced 93.7%.",
    ),
    FigureNumber(
        figure=RRNA, panel="b", element="post-hoc against in-situ enrichment",
        value="against 13-fold post hoc",
        citation_key="brandnerPooledSinglecellCRISPRa2025", line=71,
        quote="the mRNA fraction increased from 4.6% to 60.2%",
        note=(
            "13-fold is 60.2/4.6, computed here rather than stated by the "
            "source, and it is set against microSPLiT's 2.5-fold in-situ result "
            "to make the point that the two routes are not close."
        ),
    ),
    FigureNumber(
        figure=RRNA, panel="a", element="rRNA share of yeast RNA",
        value="about 85%",
        citation_key="nadal-ribellesRiseSinglecellTranscriptomics2024", line=49,
        quote="Total RNA per cell (~85% rRNA) 0.7-1 pg",
    ),
]


def main() -> None:
    figures = [f"{f.slug}.pdf" for f in FIGURES]
    problems = check(RECORDS, figures, FIGURES)
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
