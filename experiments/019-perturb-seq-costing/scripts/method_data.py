# experiments/019-perturb-seq-costing/scripts/method_data.py
# [[experiments.019-perturb-seq-costing.scripts.method_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/method_data
"""Sourced constants for the microbial perturb-seq method review.

Every value here is transcribed from a sha256-verified paper artifact in the
torchcell-library mirror (served by ``tc-lit``). Each record carries the citation
key, the line number in that paper's ``paper.md``, and a verbatim quote, so the
provenance chain runs from this file back to the exact sentence.

Nothing in this module is estimated. Values we DERIVE (cost per usable UMI,
cells per perturbation) live in the scripts that import this module, never here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Where a number came from: citation key, line in paper.md, verbatim quote."""

    citation_key: str
    line: int | None = None
    quote: str


class Method(BaseModel):
    """One published single-cell RNA-seq run, as reported.

    ``mrna_umis_per_cell`` is deliberately separate from ``genes_per_cell``: a
    method can detect one gene many times. Both are needed -- UMIs set the
    shot-noise floor on an expression estimate, genes set how much of the
    transcriptome is observable at all in one cell.
    """

    label: str
    # "First-author Year", the form every figure and table uses to name a study.
    # Separate from `label`, which may carry the protocol name, because the two
    # SPLiT rows were being drawn as "microSPLiT"/"mapSPLiT" while every other
    # point was drawn as an author-year -- one figure, two naming schemes. One
    # field, consumed by both the plot and the tables, is what keeps them
    # consistent; it defaults to the label so a new row cannot silently opt out.
    study: str = ""
    citation_key: str
    year: int
    organism: str
    platform: str
    # droplet | plate | microdissection | split_pool -- the isolation principle,
    # which is what actually drives the cells-vs-depth tradeoff.
    isolation: str
    cells_profiled: int
    mrna_umis_per_cell: float | None = None
    mrna_umis_low: float | None = None
    mrna_umis_high: float | None = None
    umi_statistic: str | None = None  # "median" | "mean"
    # What mrna_umis_low..high actually spans. Auditing every row turned up that
    # NOT ONE of them is a dispersion statistic over cells -- they are all min-max
    # ranges over conditions or experiments, and one is a garbled second-hand
    # figure. Drawn as identical vertical bars they read as error bars, which is
    # a stronger and quite different claim, so the kind is carried per row.
    #   across_conditions  -- range of per-condition summaries within one study
    #   across_experiments -- range over separate runs of differing depth/design
    #   across_cells       -- a genuine cell-to-cell range. Rare: exactly one row
    #                         (Gasch) reports one, and only its GENE count does.
    #   unclear            -- the source figure cannot be interpreted (see notes)
    spread_kind: str = "across_conditions"
    # How mrna_umis_per_cell itself was arrived at. Same discipline as the
    # "quoted vs derived" column already used for cost per cell: a value the
    # paper states and a value we computed must not look alike.
    #   reported     -- the source states this single number
    #   midpoint     -- midpoint of a reported range; a CONVENTION, not a
    #                   measurement, used where the source gives only bounds
    #   weighted     -- cell-weighted mean of reported per-dataset summaries
    #   median_of    -- median of the reported per-condition summaries
    #   derived      -- computed from two quantities the source states separately,
    #                   because it never prints the one we need. Weaker than every
    #                   basis above and must stay visible wherever it is plotted.
    value_basis: str = "reported"
    genes_per_cell: float | None = None
    genes_low: float | None = None
    genes_high: float | None = None
    mrna_read_fraction: float | None = None  # fraction of reads that are mRNA
    has_perturbation_readout: bool = False
    # True when the values come from a REVIEW's summary table rather than the
    # primary paper, and the primary paper is not in our mirror / Zotero
    # collection. A first-class field rather than a comment, so the flag cannot
    # be dropped when a row is rendered into a table or a figure.
    secondhand: bool = False
    # Set ONLY for a `secondhand` row: the primary paper is not in Zotero, so a
    # \cite key does not exist for it and the reader is otherwise given no way to
    # reach the source. The URL and PMID are taken from the reference list and
    # summary table of the review we DID read, never from recall -- that keeps the
    # chain "our figure -> the review we hold -> the primary paper" intact. Clear
    # this field the moment the paper enters the library and gets a citekey.
    url: str | None = None
    pmid: str | None = None
    provenance: list[Provenance] = Field(default_factory=list)


# --- Yeast total mRNA content -------------------------------------------------
# The denominator for every "what fraction of the transcriptome did we see"
# claim. The literature does not agree on one number, so we carry the range and
# a working point rather than pretending to a single value.

YEAST_MRNA_PER_CELL_LOW = 20_000
YEAST_MRNA_PER_CELL_HIGH = 60_000
YEAST_MRNA_PER_CELL_WORKING = 30_000
YEAST_PROTEIN_CODING_GENES = 6_000

YEAST_MRNA_PROVENANCE = [
    Provenance(
        citation_key="jarianiNewProtocolSinglecell2020",
        line=31,
        quote=(
            "the total number of mRNA molecules in yeast cells reported to be "
            "between 20,000 and 60,000, compared to 50,000 to 300,000 mRNA "
            "molecules in a typical mammalian cell"
        ),
    ),
    Provenance(
        citation_key="brettnerUltraHighthroughputMassively2024",
        line=70,
        quote=(
            "Given that there are approximately 30,000 mRNA molecules per cell "
            "for Saccharomyces cereivisiae, (Miura, 2008)"
        ),
    ),
    Provenance(
        citation_key="nadal-ribellesRiseSinglecellTranscriptomics2024",
        line=107,
        quote=(
            "the basal transcriptome of a wild-type S. cereveisae cell contains "
            "nearly half of the yeast genome (3000 genes) ... with an average "
            "expression of 3.5 molecules/gene per cell"
        ),
    ),
    Provenance(
        citation_key="nadal-ribellesRiseSinglecellTranscriptomics2024",
        line=49,
        quote=(
            "the minuscule amount of RNA within a yeast cell, ranging from 0.7 "
            "to 1 pg, with approximately 85% attributed to rRNA"
        ),
    ),
]

# Nadal-Ribelles' 3000 genes x 3.5 molecules = 10,500, materially below Jariani's
# 20,000 floor. Both are cited; the disagreement is real and is flagged in the
# note rather than averaged away.


# --- Bacterial mRNA content, for the cross-organism comparison ---------------
# NOT IN THE MIRROR. None of the mirrored bacterial papers (Gaisser, Brandner)
# states a total mRNA content per cell -- they report only what their assay
# *detected*, which is a different quantity and roughly 30x smaller. These come
# from BioNumbers and are marked external so they are never mistaken for a
# mirror-backed, sha256-pinned value. If they enter a proposal, pull the primary
# paper into the mirror first.

ECOLI_MRNA_PER_CELL_RICH = 7_800  # LB
ECOLI_MRNA_PER_CELL_MINIMAL = 2_400  # minimal medium
ECOLI_PROTEIN_CODING_GENES = 4_400

ECOLI_MRNA_PROVENANCE = [
    Provenance(
        citation_key="EXTERNAL:bionumbers-112795",
        quote=(
            "Cells grown in LB medium have about three times more mRNA "
            "transcripts than cells grown in MM, with approximately 7800 mRNA "
            "copies/cell versus approximately 2400 mRNA copies/cell. "
            "[BNID 112795; Bartholomaus A et al., Philos Trans A Math Phys Eng "
            "Sci, 2016; https://bionumbers.hms.harvard.edu/bionumber.aspx?id=112795]"
        ),
    ),
    Provenance(
        citation_key="EXTERNAL:bionumbers-taniguchi2010",
        quote=(
            "Under exponential growth at medium growth rate, E. coli is known to "
            "contain about 3000 mRNA [Taniguchi et al., Science 329:533, 2010]"
        ),
    ),
    # The mirrored papers' contribution is the rRNA burden and the detected
    # counts, not the content. Both are needed to explain why bacteria are hard.
    Provenance(
        citation_key="brandnerPooledSinglecellCRISPRa2025",
        line=71,
        quote="rRNA comprises over 95% of all transcripts in E. coli",
    ),
    Provenance(
        citation_key="brandnerPooledSinglecellCRISPRa2025",
        line=39,
        quote=(
            "sparse transcript capture in bacterial systems, typically tens to "
            "hundreds of transcripts per cell"
        ),
    ),
]

METHODS: list[Method] = [
    Method(
        # NO mRNA-UMI VALUE, deliberately. This row used to be plotted at 2,351
        # UMIs per cell, and the primary paper -- now in the mirror -- shows that
        # is a UNITS ERROR inherited from the review's summary table.
        #
        # The sentence is "Each transcriptome encompassed 735-5,437 mRNAs (median
        # = 2,351), with a total of 5,667 out of the 5,888 yeast transcripts
        # covered by at least 5 reads in >=5% of cells". Three things settle that
        # 735-5,437 counts GENES DETECTED, not molecules:
        #   1. The same sentence puts it against "5,667 out of 5,888 yeast
        #      transcripts" -- a gene-count denominator -- and the upper bound
        #      5,437 sits just under it. Molecule counts have no reason to stop
        #      dead at the number of genes in the genome.
        #   2. The cells yielded "a median of 1.4 million mapped de-duplicated
        #      fragments per cell". 1.4M DISTINCT post-deduplication fragments
        #      cannot come from 2,351 molecules; they can easily come from 2,351
        #      genes tiled along their length by a full-length library.
        #   3. Nadal-Ribelles et al. 2019, comparing their own method to this one,
        #      report the pair as "a higher number of genes per cell (3,399
        #      versus 2,392)" -- naming Gasch's quantity as genes outright.
        # Fluidigm C1 + SMART-Seq carries no UMI at all, so there is no molecule
        # count to plot here under any reading. The value moves to the genes
        # column and the row leaves the depth axis.
        label="Gasch 2017",
        citation_key="gaschSinglecellRNASequencing2017",
        year=2017,
        organism="S. cerevisiae",
        platform="Fluidigm C1 / SMART-Seq v4",
        isolation="plate",
        cells_profiled=163,  # 83 + 80 successful libraries
        mrna_umis_per_cell=None,
        genes_per_cell=2351,
        genes_low=735,
        genes_high=5437,
        # And the review's "735 +/- 5437" IS a garbled range, now confirmed
        # against the primary rather than merely suspected: the primary prints
        # "735-5,437". It is a spread across CELLS, the only one in this table.
        spread_kind="across_cells",
        provenance=[
            Provenance(
                citation_key="gaschSinglecellRNASequencing2017",
                line=56,
                quote=(
                    "we performed two C1 chip runs for each of the unstressed and "
                    "stressed cultures and selected 85 and 81 captured "
                    "single-cells, respectively; 83 and 80 yielded successful "
                    "libraries ... producing a median of 1.4 million mapped "
                    "de-duplicated fragments per cell ... Each transcriptome "
                    "encompassed 735-5,437 mRNAs (median = 2,351), with a total "
                    "of 5,667 out of the 5,888 yeast transcripts covered by at "
                    "least 5 reads in >=5% of cells"
                ),
            ),
            Provenance(
                citation_key="nadal-ribellesSensitiveHighthroughputSinglecell2019",
                quote=(
                    "yscRNA-seq yielded a higher number of genes per cell (3,399 "
                    "versus 2,392)"
                ),
            ),
        ],
    ),
    Method(
        label="Nadal-Ribelles 2019",
        # Was also plotted as UMIs using a gene count -- 3,339 "transcripts", as
        # Jariani's summary paraphrases it. The primary calls the same quantity
        # genes ("3,399 ... genes per cell"), and note the digits: the figure in
        # circulation, 3,339, is a transposition of 3,399.
        #
        # This method DOES carry a UMI, so a molecule count exists; the paper just
        # never prints it per cell. It prints the two factors instead -- genes per
        # cell and UMIs per detected gene -- so the product is recorded here as a
        # DERIVED value, flagged as such, rather than leaving the deepest plate
        # method off an axis about depth.
        citation_key="nadal-ribellesSensitiveHighthroughputSinglecell2019",
        year=2019,
        organism="S. cerevisiae",
        platform="FACS + STRT-seq (5' end)",
        isolation="plate",
        cells_profiled=285,
        mrna_umis_per_cell=11_760,  # 3,399 genes x 3.46 UMIs per gene
        umi_statistic="derived: genes per cell x mean UMIs per detected gene",
        value_basis="derived",
        genes_per_cell=3399,
        provenance=[
            Provenance(
                citation_key="nadal-ribellesSensitiveHighthroughputSinglecell2019",
                quote=(
                    "we applied yscRNA-seq to 285 individual yeast cells ... "
                    "yscRNA-seq yielded a higher number of genes per cell (3,399 "
                    "versus 2,392) ... On average, each gene expressed 3.46 "
                    "transcript molecules per cell [3,399 x 3.46 = 11,760 UMIs "
                    "per cell, derived here, not stated by the source]"
                ),
            ),
            Provenance(
                citation_key="jarianiNewProtocolSinglecell2020",
                line=33,
                quote=(
                    "studied 285 single Saccharomyces cerevisiae yeast cells "
                    "grown in rich media, and detected on average 3339 transcripts"
                ),
            ),
        ],
    ),
    Method(
        # Primary now in the mirror, so the second-hand flag is gone -- and the
        # numbers moved. The review gave "~2000 mRNAs per cell and ~700 genes",
        # both rounded and neither attributed to a growth condition. The primary
        # reports a median of 2,250 UMIs from a median of 695 genes in YPD, and
        # says explicitly that depth is condition-dependent, so naming the
        # condition is part of the number.
        label="Jackson 2020",
        citation_key="jacksonGeneRegulatoryNetwork2020",
        year=2020,
        organism="S. cerevisiae",
        platform="10x Chromium 3' v2",
        isolation="droplet",
        cells_profiled=38_225,
        mrna_umis_per_cell=2250,
        umi_statistic="median (YPD)",
        genes_per_cell=695,
        has_perturbation_readout=True,  # 72 barcoded deletion strains
        provenance=[
            Provenance(
                citation_key="jacksonGeneRegulatoryNetwork2020",
                quote=(
                    "we recovered 83,703,440 transcript counts from a total of "
                    "38,225 individual cells ... For cells growing in rich medium "
                    "(YPD) we recover a median of 2250 unique transcripts per "
                    "cell, from a median of 695 distinct genes, indicating a "
                    "capture rate of approximately 3-5% of total transcripts from "
                    "each cell"
                ),
            ),
        ],
    ),
    Method(
        label="Jariani 2020",
        citation_key="jarianiNewProtocolSinglecell2020",
        year=2020,
        organism="S. cerevisiae",
        platform="10x Chromium 3' v2 + in-droplet zymolyase",
        isolation="droplet",
        cells_profiled=6_118,
        mrna_umis_per_cell=894,
        mrna_umis_low=528,
        mrna_umis_high=1261,
        umi_statistic="median",
        # Five per-condition medians (1261, 894, 528, 607, 975); the plotted
        # point is their median, 894, not a midpoint of the extremes.
        value_basis="median_of",
        provenance=[
            Provenance(
                citation_key="jarianiNewProtocolSinglecell2020",
                line=51,
                quote=(
                    "The number of cells analyzed ... was 1695, 1096, 1172, 1469 "
                    "and 686, respectively ... and the per-cell median number of "
                    "transcripts in each condition was 1261, 894, 528, 607 and "
                    "975, respectively"
                ),
            )
        ],
    ),
    Method(
        label="Urbonaite 2021",
        citation_key="urbonaiteYeastoptimizedSinglecellTranscriptomics2021",
        year=2021,
        organism="S. cerevisiae",
        platform="yeastDrop-Seq (table-top)",
        isolation="droplet",
        cells_profiled=844,
        mrna_umis_per_cell=1088,
        mrna_umis_low=840,
        mrna_umis_high=1336,
        umi_statistic="mean",
        # The source gives only bounds -- "840.19 to 1335.62 across the four
        # treatment conditions" -- with no per-condition value to weight by, so
        # the plotted point is the midpoint by convention. Declared, not passed
        # off as a measurement. TODO: the primary paper IS in the mirror; pull
        # the per-condition means and cell counts and switch to a weighted mean,
        # as was done for Boocock.
        value_basis="midpoint",
        genes_per_cell=548,
        genes_low=443,
        genes_high=619,
        provenance=[
            Provenance(
                citation_key="urbonaiteYeastoptimizedSinglecellTranscriptomics2021",
                line=42,
                quote=(
                    "With the total number of cells identified in each condition "
                    "ranging from 85 to 268, the mean transcript counts per cell "
                    "ranged from 840.19 to 1335.62 across the four treatment "
                    "conditions ... 443 (G), 554 (M), 578 (MG), and 619 (D)"
                ),
            )
        ],
    ),
    Method(
        label="Dohn 2021 (mDrop-seq)",
        study="Dohn 2021",
        secondhand=True,
        # Dohn, R. et al. mDrop-seq, Vaccines 10, 30. The review's reference list
        # (paper.md line 176) prints the *bioRxiv* DOI 10.1101/2021.07.06.451338
        # for what it cites as the Vaccines paper -- an error in the review, so
        # linking it would send the reader to the preprint rather than to the
        # version these numbers were tabulated from. The PMID in the review's own
        # Table 1 (line 62) is unambiguous, so resolve through that instead. Do
        # not substitute a journal DOI from memory; get it when the paper is
        # deposited in Zotero and this field is retired.
        url="https://pubmed.ncbi.nlm.nih.gov/35062691/",
        pmid="35062691",
        citation_key="nadal-ribellesRiseSinglecellTranscriptomics2024",
        year=2021,
        organism="S. cerevisiae + C. albicans",
        platform="mDrop-seq",
        isolation="droplet",
        cells_profiled=74_814,  # 35,109 S.c. + 39,705 C.a.
        genes_per_cell=580,
        genes_low=193,
        genes_high=967,
        provenance=[
            Provenance(
                citation_key="nadal-ribellesRiseSinglecellTranscriptomics2024",
                line=62,
                quote="35,109 in total (SC) | 39,705 ... in total (CA) ... 193-967 genes",
            )
        ],
    ),
    Method(
        # Labelled 2025, not 2023. The review we originally sourced this row from
        # cited it as a preprint ("recently reported as a preprint at the time of
        # writing"), and we inherited its year. It has since published as eLife
        # 13:RP95566 and is now in Zotero AND the mirror under
        # boocockSinglecellEQTLMapping2025, so the citekey and the plotted label
        # must agree with the published record.
        label="Boocock 2025",
        citation_key="boocockSinglecellEQTLMapping2025",
        year=2025,
        organism="S. cerevisiae",
        platform="10x Chromium 3' v2 + v3",
        isolation="droplet",
        # All three numbers below are now derived from the PRIMARY source --
        # Supplementary file 1, Table S4, retrieved and reduced by
        # boocock_si.py (sha256-pinned; per-dataset rows in
        # results/boocock2025_table_s4.csv). Previously they came from a review's
        # summary table, and the plotted value was 4036: the arithmetic midpoint
        # of 1514 and 6559, a number that appears in no source at all.
        #
        # Table S4 also explains what the review's two figures actually were, and
        # both reconcile exactly:
        #   100,220 = the SUM of n_cells over all seven datasets (4 eQTL + 3 ASE).
        #             The paper's prose says only "over 100,000".
        #   1514-6559 = the MIN and MAX of median_umi_per_cell across those same
        #             seven datasets -- a BETWEEN-EXPERIMENT range, which is why
        #             spread_kind below is not the default.
        cells_profiled=100_220,
        # The cell-weighted mean of the per-experiment medians. Weighted because
        # the x-axis is total cells profiled: an unweighted mean would let the
        # 2,864-cell ASE run count as much as the 44,784-cell eQTL run, and the
        # point would stop describing the 100,220 cells its x-coordinate claims.
        # Derivation lives in boocock_si.py so it is recomputed, not transcribed.
        mrna_umis_per_cell=2809,
        mrna_umis_low=1514,
        mrna_umis_high=6559,
        # Each dataset reports a median; the plotted point is a cell-weighted mean
        # OF those medians. Named precisely because the two are different
        # statistics and the distinction is the whole reason this row was wrong.
        umi_statistic="cell-weighted mean of per-experiment medians",
        # The bar on this row is NOT a spread across cells like the others -- it
        # is a spread across three separate experiments whose depth deliberately
        # increased. Plotting it on the same axis as a cell-to-cell spread without
        # saying so would misread as "this method is noisy" when it means "these
        # runs were different".
        spread_kind="across_experiments",
        value_basis="weighted",
        has_perturbation_readout=True,  # natural variants, single-cell eQTL
        provenance=[
            Provenance(
                citation_key="boocockSinglecellEQTLMapping2025",
                line=46,
                quote=(
                    "used scRNA-seq to obtain the transcriptomes of 7124 cells "
                    "... We captured a median of 1514 unique RNA molecules "
                    "(unique molecular identifiers; henceforth UMIs) ... per cell"
                ),
            ),
            Provenance(
                citation_key="boocockSinglecellEQTLMapping2025",
                line=32,
                quote=(
                    "applied it to over 100,000 single cells from three crosses"
                ),
            ),
            Provenance(
                citation_key="boocockSinglecellEQTLMapping2025",
                line=71,
                quote=(
                    "we leveraged higher yields of UMIs per cell in subsequent "
                    "experiments to ensure better genotyping accuracy"
                ),
            ),
            # Supplementary file 1, Table S4. Quoted as the sheet's own header
            # plus the reduced figures, since the source is a spreadsheet rather
            # than prose and has no line number to cite.
            Provenance(
                citation_key="boocockSinglecellEQTLMapping2025",
                quote=(
                    "Table S4: Summary information for the single-cell "
                    "expression data generated in this study. "
                    "[7 datasets; sum(n_cells)=100,220; "
                    "median_umi_per_cell 1514..6559; "
                    "cell-weighted mean of medians = 2809. "
                    "Retrieved by boocock_si.py from "
                    "https://cdn.elifesciences.org/articles/95566/"
                    "elife-95566-supp1-v1.xlsx, sha256 7a0d8104d053...ee1c555; "
                    "per-dataset rows in results/boocock2025_table_s4.csv]"
                ),
            ),
            # Kept deliberately: this is the row the values USED to come from, and
            # recording that both of its figures reconcile against Table S4 is
            # what lets us trust the review's other rows a little more.
            Provenance(
                citation_key="nadal-ribellesRiseSinglecellTranscriptomics2024",
                line=62,
                quote="100,220 in total ... 1514-6559 UMIs",
            ),
        ],
    ),
    Method(
        # The most consequential row in this table, and it was missing. This is a
        # genome-scale yeast Perturb-seq: >3,500 barcoded deletion mutants (75% of
        # the non-essential genome) read out in one scRNA-seq atlas, with the
        # genotype recovered from the transcriptome itself. Sec. 3.5 was written
        # around the claim that no such system exists in yeast, and it does.
        #
        # It is also the only row here that legitimately reaches 10^6 cells --
        # which is the number the Gaisser row was wrongly asserting from a
        # protocol capacity claim.
        label="Nadal-Ribelles 2025",
        citation_key="nadal-ribellesSinglecellResolvedGenotypephenotype2025",
        year=2025,
        organism="S. cerevisiae",
        platform="Singleron GEXSCOPE HD (microwell)",
        isolation="microwell",
        cells_profiled=1_061_865,
        mrna_umis_per_cell=1200,
        umi_statistic="median",
        genes_per_cell=550,
        has_perturbation_readout=True,
        provenance=[
            Provenance(
                citation_key="nadal-ribellesSinglecellResolvedGenotypephenotype2025",
                quote=(
                    "we used a microwell-based platform for single-cell isolation "
                    "and oligo dT priming to capture polyadenylated RNA for cDNA "
                    "synthesis (Yeast GENEXSCOPE HD, Singleron Biotechnologies). "
                    "We profiled a total of 1.061.865 cells"
                ),
            ),
            Provenance(
                citation_key="nadal-ribellesSinglecellResolvedGenotypephenotype2025",
                quote=(
                    "The resulting dataset contained more than 3500 mutant "
                    "genotypes ... with an average of 93 and 108 cells/genotype "
                    "(median of 77 and 87 respectively) in control and stress "
                    "conditions, respectively, and had a median coverage of 550 "
                    "genes/cell and 1200 molecules/cell."
                ),
            ),
        ],
    ),
    Method(
        label="Brettner 2024 (SPLiT-seq)",
        study="Brettner 2024",
        citation_key="brettnerUltraHighthroughputMassively2024",
        year=2024,
        organism="S. cerevisiae + C. albicans",
        platform="yeast-optimized SPLiT-seq",
        isolation="split_pool",
        cells_profiled=43_388,
        mrna_umis_per_cell=410,
        mrna_umis_low=120,
        mrna_umis_high=700,
        umi_statistic="mean",
        # "~120-700 mRNAs per cell depending on the experiment and sequencing
        # depth" -- explicitly a between-experiment range, and the plotted 410
        # is its midpoint by convention.
        spread_kind="across_experiments",
        value_basis="midpoint",
        genes_per_cell=350,
        genes_low=100,
        genes_high=600,
        mrna_read_fraction=0.0575,
        provenance=[
            Provenance(
                citation_key="brettnerUltraHighthroughputMassively2024",
                line=70,
                quote=(
                    "We recover, on average, between 100 and 600 genes per cell "
                    "(Figure 3c). Of the detected mRNAs, we see an average of "
                    "~120-700 mRNAs per cell depending on the experiment and "
                    "sequencing depth."
                ),
            ),
            Provenance(
                citation_key="brettnerUltraHighthroughputMassively2024",
                line=70,
                quote=(
                    "On average, the transcript proportions we recover are 93.7% "
                    "rRNA, 0.005% tRNA, 0.04% ncRNA, and 5.75% mRNA."
                ),
            ),
        ],
    ),
    # --- bacteria: the split-pool lineage, included because it is where the
    # only published pooled-CRISPR-with-transcriptome screen actually lives.
    Method(
        # This row was "Gaisser 2024, 1,000,000 cells" and it was the rightmost
        # point in the landscape figure -- the anchor for "split-pool reaches a
        # million cells". It does not belong on an axis labeled "cells profiled
        # in the published study", for two reasons found on re-reading the source.
        #
        # First, Gaisser et al. is a Nature Protocols article, and every one of
        # its million-cell statements is a CAPACITY claim, not a measurement:
        # "enables ... profiling of UP TO 1 million bacterial cells", "feasible to
        # barcode UP TO 1 million cells", "POSSIBLE to bring the practical
        # throughput ... to 45,000 x 24", and "designed to work with high cell
        # INPUT numbers (>=1 million cells per sample of STARTING MATERIAL)".
        # Starting material is not profiled cells -- the same page puts cell loss
        # through the protocol at up to ~70%.
        #
        # Second, the depth it was plotted at is not its own. The 397 UMIs/cell is
        # Gaisser quoting the original microSPLiT paper, so the point mixed a
        # capacity ceiling on x with a borrowed measurement on y.
        #
        # The honest data point for this method is the study that produced it.
        # Gaisser remains the source for the protocol content it is actually
        # authoritative on -- collision model, sublibrary sizing, per-run cost.
        label="microSPLiT (B. subtilis)",
        study="Kuchina 2021",
        citation_key="kuchinaMicrobialSinglecellRNA2021",
        year=2021,
        organism="B. subtilis",
        platform="microSPLiT",
        isolation="split_pool",
        cells_profiled=25_214,
        mrna_umis_per_cell=397,
        umi_statistic="median",
        genes_per_cell=230,
        mrna_read_fraction=0.075,
        provenance=[
            Provenance(
                citation_key="kuchinaMicrobialSinglecellRNA2021",
                quote=(
                    "We sampled a median of 235 mRNA transcripts per cell for E. "
                    "coli and 397 for B. subtilis (Fig. 1C), or about 5 to 10% of "
                    "the estimated total mRNA ... the replicates were consistent "
                    "and produced a combined dataset of 25,214 cells"
                ),
            ),
            Provenance(
                citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
                line=98,
                quote=(
                    "the final microSPLiT library still has ~5-10% mRNA because "
                    "of the high inherent rRNA content in bacterial cells"
                ),
            ),
            # Kept so the capacity claim stays on the record as a claim, next to
            # the measurement it was being confused with.
            Provenance(
                citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
                quote=(
                    "The standard plate setup enables single-cell transcriptional "
                    "profiling of up to 1 million bacterial cells and up to 96 "
                    "samples in a single barcoding experiment [a capacity claim "
                    "for the protocol, not a profiled cell count]"
                ),
            ),
        ],
    ),
    # --- bacteria: the droplet lineage, added because the frontier argument in
    # Sec. 4.2 is about isolation principles and was missing the bacterial
    # droplet corner entirely. Both are in the microbe-perturb-seq collection.
    Method(
        label="ProBac-seq (B. subtilis)",
        study="McNulty 2023",
        citation_key="mcnultyProbebasedBacterialSinglecell2023",
        year=2023,
        organism="B. subtilis",
        platform="ProBac-seq (probe-based, 10x)",
        isolation="droplet",
        cells_profiled=2_784,
        mrna_umis_per_cell=325,
        umi_statistic="median",
        genes_per_cell=241,
        provenance=[
            Provenance(
                citation_key="mcnultyProbebasedBacterialSinglecell2023",
                quote=(
                    "In total, 2,784 cells were captured as determined by unique "
                    "barcodes at an average depth of 1,073 reads per cell. We "
                    "detected a median of 325 mRNA transcripts per cell "
                    "(approximately 10-20% of the total mRNA pool) corresponding "
                    "to a median of 241 genes per cell."
                ),
            ),
        ],
    ),
    Method(
        # Genes only. BacDrop reports mRNA GENES per cell throughout and never a
        # per-cell molecule count, so it is tabulated and left off the depth axis
        # rather than having a UMI value invented for it -- the same treatment
        # Dohn and Gasch get, and the reason the two quantities are separate
        # fields on this model.
        label="BacDrop (K. pneumoniae)",
        study="Ma 2023",
        citation_key="maBacterialDropletbasedSinglecell2023",
        year=2023,
        organism="K. pneumoniae",
        platform="BacDrop (pre-indexed droplet)",
        isolation="droplet",
        cells_profiled=44_140,
        mrna_umis_per_cell=None,
        genes_per_cell=127,
        genes_low=30,
        genes_high=127,
        provenance=[
            Provenance(
                citation_key="maBacterialDropletbasedSinglecell2023",
                quote=(
                    "Of 44,140 cells that passed quality control at a sequencing "
                    "depth of ~1,000 reads per cell ... we detected an average of "
                    "30 mRNA genes per cell across all 60,000 cells with at least "
                    "15 mRNA genes detected. However, from this large library, "
                    "the top 3,000 high-quality cells had an average of 127 mRNA "
                    "genes detected per cell"
                ),
            ),
        ],
    ),
    Method(
        label="mapSPLiT (E. coli, CRISPRi/a)",
        study="Brandner 2025",
        citation_key="brandnerPooledSinglecellCRISPRa2025",
        year=2025,
        organism="E. coli",
        platform="mapSPLiT (microSPLiT + GBC)",
        isolation="split_pool",
        cells_profiled=76_068,
        mrna_umis_per_cell=325,
        umi_statistic="median",
        genes_per_cell=161,
        mrna_read_fraction=0.602,
        has_perturbation_readout=True,
        provenance=[
            Provenance(
                citation_key="brandnerPooledSinglecellCRISPRa2025",
                line=89,
                quote=(
                    "We sequenced 365,000 cells and retained 76,068 of them after "
                    "assignment, with a median of 325 transcripts and 161 genes per cell."
                ),
            ),
            Provenance(
                citation_key="brandnerPooledSinglecellCRISPRa2025",
                line=71,
                quote=(
                    "With rRNA depletion, the mRNA fraction increased from 4.6% "
                    "to 60.2% ... and median mRNA transcripts per cell more than "
                    "doubled, from 54 to 115"
                ),
            ),
        ],
    ),
]


# --- Library construction ceiling (Lian et al., MAGIC/sMAGIC) -----------------
# What caps multiplexing before sequencing does. Sourced from our own library's
# paper, which is also the only published yeast precedent for a COMBINATORIAL
# guide library, so it reports both the ceiling and what happens when you hit it.

YEAST_TRANSFORMATION_CLONES_LOW = 1_000_000
YEAST_TRANSFORMATION_CLONES_HIGH = 10_000_000
# Lian et al.'s own construction standard for a genome-scale library, and the
# multiplier that turns a pair count into a required clone count.
LIBRARY_REDUNDANCY_FOLD = 30

LIBRARY_CEILING_PROVENANCE = [
    Provenance(
        citation_key="lianMultifunctionalGenomewideCRISPR2019",
        quote=(
            "due to the limitations in transformation efficiency (~10^6-10^7 for "
            "yeast transformation), only ~0.01% to ~0.1% of the total "
            "combinations (~10^5 x 10^5 = 10^10) were covered in the sMAGIC yeast "
            "library"
        ),
    ),
    Provenance(
        citation_key="lianMultifunctionalGenomewideCRISPR2019",
        quote=(
            "The independent clones for each library should be >10^6, with at "
            "least 30-fold redundancy."
        ),
    ),
    Provenance(
        citation_key="lianMultifunctionalGenomewideCRISPR2019",
        quote=(
            "The Golden-Gate Assembly efficiency was estimated to be nearly 100%"
        ),
    ),
]


def clones_required(n_pairs: int, redundancy: int = LIBRARY_REDUNDANCY_FOLD) -> int:
    """Independent clones needed to cover ``n_pairs`` at the stated redundancy.

    The comparison that matters is this against YEAST_TRANSFORMATION_CLONES_*:
    a constructed library is feasible only when the result fits under the
    ceiling, and for genome-scale pairs it does not, by about two orders of
    magnitude.
    """
    return n_pairs * redundancy


# --- Barcode-collision model (Gaisser Table 1) --------------------------------
# The authoritative published form. B = number of distinct barcode trajectories,
# C = number of cells barcoded together in one sublibrary.

COLLISION_FORMULA = Provenance(
    citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
    line=77,
    quote="100(1-((B-1)/B)^(C-1))",
)

# Published operating point: 5% collision is the accepted ceiling, and it is what
# fixes the cells-per-sublibrary number that the whole cost model hangs on.
COLLISION_CEILING = 0.05
CELLS_PER_SUBLIBRARY_AT_CEILING = 45_000
SUBLIBRARY_PROVENANCE = Provenance(
    citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
    line=161,
    quote=(
        "We prefer not to exceed the barcode collision rate of 5%, corresponding "
        "to ~45,000 cells per single sub-library... Using 24 indexing barcodes "
        "during sub-library preparation, it is possible to bring the practical "
        "throughput of a microSPLiT run to 45,000 x 24 (or ~1 million) cells."
    ),
)


def collision_rate(n_cells: int, barcode_space: int) -> float:
    """Fraction of barcodes shared by >1 cell, per Gaisser et al. Table 1."""
    return 1.0 - ((barcode_space - 1) / barcode_space) ** (n_cells - 1)


def barcode_space(rounds: int = 3, wells: int = 96, sublibraries: int = 1) -> int:
    """Distinct barcode trajectories.

    The sublibrary index is a genuine barcode dimension, not bookkeeping: cells
    are split into sublibraries AFTER the last in-cell ligation, so two cells
    sharing a BC1-BC2-BC3 path are still resolved if they land in different
    sublibraries. Leaving it out of the barcode space is the most common way to
    over-estimate the collision rate of a real run.
    """
    return wells**rounds * sublibraries


# A study with no explicit `study` name is named by its label. Applied here rather
# than in a validator so the default is visible at the point of use.
for _m in METHODS:
    if not _m.study:
        _m.study = _m.label
