# experiments/database/scripts/build_candidate_datasets_table.py
# [[experiments.database.expansion-100]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_candidate_datasets_table
r"""The candidate list for taking the database from 50 supported datasets to 200.

This is the CURATION, held as data. Unlike ``build_supported_datasets_table.py``,
which measures built LMDBs, nothing here can be recomputed from a store: the rows
are a judgment about which *S. cerevisiae* datasets to ingest next, so the
judgment itself is the artifact and it lives in a committed script rather than a
gitignored results file.

Emits, off the same records:
  - notes-tex/database-expansion-100/tables/candidates.tex   (triage table)
  - notes-tex/database-expansion-100/tables/sources.tex      (citation + link + accession)
  - notes-tex/database-expansion-100/tables/perturbseq.tex   (rows bearing on a Perturb-seq)
  - notes-tex/database-expansion-100/tables/synergies.tex    (candidate x partner joins)
  - notes-tex/database-expansion-100/tables/swaps.tex        (rank changes vs the last pass)
  - notes-tex/database-expansion-100/tables/pins.tex         (requested rows pinned above the cut)
  - notes-tex/database-expansion-100/tables/excluded.tex     (what was dropped, and why)
  - notes-tex/database-expansion-100/tables/counts.tex       (per-class and per-band totals)
  - <results>/candidates/candidate_datasets.json             (machine-readable dump)

Run from the repo root:
  python experiments/database/scripts/build_candidate_datasets_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
TEX_DIR = REPO / "notes-tex" / "database-expansion-100" / "tables"
JSON_OUT = RESULTS / "candidates" / "candidate_datasets.json"

SOURCE_LINE = (
    "%% SOURCE: experiments/database/scripts/build_candidate_datasets_table.py"
)

# The database holds 50 schematized + L0-L4-verified datasets. The goal is 200, so
# the long-run recommended set is the first 150 rows; everything after is a ranked
# reserve bench, kept because the cut line moves whenever one of the 150 turns out
# to have no recoverable per-strain data.
BUILT_COUNT = 50
TARGET_COUNT = 200
CUT = TARGET_COUNT - BUILT_COUNT  # 150

# The long-run cut is not a work queue. Ingestion happens in waves, so the list
# carries two nearer lines that are what a build week is planned against:
#   WAVE_1  the next builds, ranked
#   WAVE_2  the bench directly behind them, promoted the moment a wave-1 row
#           turns out to have no recoverable per-strain data
WAVE_1 = 50
WAVE_2 = 70
# Rows rendered in the final table: the recommended fifty plus ten extra, in rank
# order, so a row that proves unreachable has a named replacement.
FINAL = 60

# ---------------------------------------------------------------------------
# Vocabulary. Defined here so it is defined before use in the document, and so a
# typo becomes a validation error rather than a silently novel category.
# ---------------------------------------------------------------------------

Klass = Literal[
    "Natural variation",
    "Tolerance / robustness",
    "CRISPR library screen",
    "Expression / single cell",
    "Metabolite / precursor",
    "Modality / backbone",
    "Regulatory DNA",
    "Deep mutational scan",
    "Combinatorial genome",
]

# How the total genomic content of one strain would be reconstructed. This is the
# hard gate: a row with no route to a sequence cannot train a genotype-to-phenotype
# model and is not a candidate at all (see EXCLUDED).
SeqBasis = Literal[
    "S288C-KO",  # reference minus one cataloged ORF; per-strain WGS exists (Puddu 2019)
    "S288C-KO/het",  # heterozygous diploid deletion -- a dosage edit, still cataloged
    "S288C+guide",  # genome unedited; perturbation is a designed cassette + guide target
    "S288C+designed-edit",  # designed SNV/indel library; designed, not per-strain verified
    "S288C+tag",  # reference plus a designed tag / degron / promoter cassette
    "S288C+ORF-plasmid",  # reference plus a barcoded ORF on a known plasmid
    "S288C+reporter-locus",  # reference plus a designed regulatory sequence driving a reporter
    "isolate-WGS",  # per-isolate assembly or VCF published
    "segregant-WGS",  # per-progeny genotype calls or whole-genome sequence
    "engineered-chassis",  # named production strain + heterologous cassettes
    "reference-only",  # wild type; the environment is the perturbation
]

Basis = Literal["reported", "product", "estimate"]

# Ingestion state. "candidate" means untouched. The others exist because this
# pass initially ranked a blocked dataset first and a half-built one twentieth:
# scanning the built list is not enough, since a dataset can have a loader, or a
# failed retrieval attempt behind it, without appearing there. "built" is kept in
# the list rather than deleted so the previous pass's ranking can be reproduced
# exactly and the row's departure shows up as a recorded move.
Status = Literal["candidate", "blocked", "loader-in-flight", "built"]

# How well a row's numbers and citation were checked. "sourced" means the figures
# trace to a source read this session or to the sourced triage note; "recall" means
# they were written from domain knowledge and the citation, accession and counts all
# need confirming before a loader. Recording this is the difference between a list
# that can be acted on and one that has to be re-checked wholesale.
Confidence = Literal["sourced", "recall"]

# Bearing on a yeast Perturb-seq, on two independent axes:
#   input  -- a HIGH-DIMENSIONAL perturbation space: combinatorial, or perturbation
#             crossed with genetic background. A large single-perturbation library
#             is NOT high-dimensional input; it is one edit per cell, sampled widely.
#   output -- a transcriptome-scale or per-cell distributional readout, rather than
#             a scalar fitness or titer.
# The quadrant that matters is "both", and in yeast it is nearly empty.
PertSeq = Literal["none", "input", "output", "both"]

# Priority band, applied BEFORE tier and scale. Scale is the right default and it
# is the wrong first question when a specific campaign is being planned: a
# 100-million-sequence promoter library is the largest thing in this table and it
# does not tell a Perturb-seq design anything, while a 60,000-guide CRISPRi library
# with a released per-guide matrix does.
#
#   perturb-seq   -- pairs directly with a planned yeast Perturb-seq: a CRISPR
#                    interference or activation library whose per-guide fitness or
#                    expression matrix is released, a barcoded single-cell
#                    genotype set, or an induction series with a transcriptome
#                    readout. These are the rows a campaign is designed against.
#   molecular layers
#                 -- two or more molecular layers joinable on one axis, either
#                    within the row or against a supported dataset. The layers
#                    are transcript, translation, protein, phosphosite,
#                    metabolite, flux and turnover. The join is per STRAIN when
#                    both layers are measured on the same genotypes, and per GENE
#                    when a layer is a single genome-wide coefficient such as a
#                    half-life. This was the "metabolism x expression" band and is
#                    widened: transcript against protein is the same join as
#                    expression against flux, and the rows that explain WHY the
#                    two disagree, translation efficiency and turnover, had no
#                    band to sit in at all.
#   scale         -- everything else, ranked as before.
Band = Literal["perturb-seq", "molecular layers", "scale"]

BAND_ORDER: dict[str, int] = {"perturb-seq": 0, "molecular layers": 1, "scale": 2}


class Synergy(BaseModel):
    """What joining this row to another buys, and on what key.

    A synergy is only real if the two datasets share an addressable axis, so the
    join key is a required field rather than a remark: "same deletion collection",
    "same 1,011-isolate panel", "same segregant genotype class". Without one the
    pair is a theme, not a join.
    """

    partner: str
    partner_status: Literal["supported", "candidate"]
    join: str
    yields: str


class Candidate(BaseModel):
    """One dataset proposed for ingestion."""

    name: str
    citation: str
    url: str
    klass: Klass
    tier: int = Field(ge=1, le=4)
    genotypes_n: int | None
    genotypes: str
    env_n: int | None
    env: str
    instances_n: int | None
    instances_basis: Basis
    phenotype: str
    shape: str
    dim: int = 1  # phenotype vector length; 1 for a scalar label
    seq_basis: SeqBasis
    why: str
    accession: str
    perturbseq: PertSeq = "none"
    requested: bool = (
        False  # named explicitly in the scoping request; pinned above the cut
    )
    status: Status = "candidate"
    confidence: Confidence = "sourced"
    band: Band = "scale"
    band_why: str = ""  # why this row is out of the scale band; required when it is
    synergy: list[Synergy] = Field(default_factory=list)
    added: bool = False  # first appears in this pass, so it has no previous rank
    # The row's time dimension, when it has one: a sampled series, an age axis or a
    # rate derived from a labeling series. Empty for a steady-state or endpoint row.
    # Growth rate in a chemostat is a rate, not a time axis, and is left empty.
    time_axis: str = ""

    @property
    def measurements(self) -> int | None:
        """Instances times phenotype dimensionality.

        Ranking on instances alone silently prefers a scalar-fitness screen over a
        vector-valued omics panel of the same size: 796 isolate proteomes is 796
        instances but roughly 1.6 million numbers. This is the same normalization
        the built table's gzip-signal column exists to make.
        """
        return None if self.instances_n is None else self.instances_n * self.dim

    @property
    def scale_key(self) -> tuple[int, float]:
        """The previous pass's ordering: tier, then measurements descending."""
        return (self.tier, -math.log10(max(self.measurements or 1, 1)))

    @property
    def sort_key(self) -> tuple[int, int, float]:
        """Band first, then the scale key inside it."""
        return (BAND_ORDER[self.band],) + self.scale_key


class Excluded(BaseModel):
    """A dataset considered and dropped, with the rule that dropped it."""

    name: str
    reason: str
    rule: Literal["no-sequence", "off-species", "already-built", "not-a-dataset"]


# ---------------------------------------------------------------------------
# Tier rule, stated once so the ordering is reproducible rather than a matter of
# taste. Within a tier, rows sort by MEASUREMENTS (instances x phenotype
# dimensionality), descending -- see Candidate.measurements for why not instances.
#
#   1  clears all three bars -- >=1e3 genotypes, >=1e4 instances, a clean
#      sequence basis -- and lands in one of the six requested classes.
#   2  clears two of the three; or is the ONLY dataset covering a requested class;
#      or directly de-risks the Perturb-seq proposal.
#   3  a modality or reference backbone the substrate needs, without itself
#      carrying a large genotype axis.
#   4  real, but small n, a coarse label, or heavy extraction overhead.
#
# The bars are applied to what a row actually contains, not to its reputation. A
# 157-strain compound panel fails the genotype bar however large its compound axis
# is, and a meta-aggregation fails tier 1 because its instance count is not net new.
# ---------------------------------------------------------------------------

CANDIDATES: list[Candidate] = [
    # -- Tier 1 -------------------------------------------------------------
    Candidate(
        name="Lee 2014 (HIP-HOP fitness signatures)",
        citation="Lee AY, St Onge RP, Proctor MJ, et al., Giaever G, Nislow C. Science 2014;344:208-211.",
        url="https://doi.org/10.1126/science.1250217",
        klass="Tolerance / robustness",
        tier=1,
        genotypes_n=11000,
        genotypes="~11,000 (het + hom)",
        env_n=3356,
        env="3,356 compounds",
        instances_n=13000000,
        instances_basis="reported",
        phenotype="chemogenomic fitness score",
        shape="scalar",
        seq_basis="S288C-KO",
        why="Largest chemical-genetic matrix in existence, on the exact YKO genotype axis Costanzo and Kuzmin already use. Biggest single instance-count gain available. BLOCKED: WS15 already attempted it and it stands as awaiting author matrices, so the Science SI and the lab portal did not yield per-strain values. Unblocking is an author request, not a loader.",
        accession="Science SI + Nislow/Giaever portal, neither of which yielded per-strain matrices; awaiting author matrices per the WS15 roadmap",
        status="blocked",
    ),
    Candidate(
        name="Turco 2023 (Yeast Phenome)",
        citation="Turco G, Chang C, Wang RY, et al., Boone C, Andrews BJ, Roth FP. Sci Adv 2023;9:eadg5702.",
        url="https://doi.org/10.1126/sciadv.adg5702",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=5000,
        genotypes="~5,000",
        env_n=7536,
        env="7,536 environments",
        instances_n=37680000,
        instances_basis="product",
        phenotype="growth / fitness per environment",
        shape="scalar",
        seq_basis="S288C-KO",
        why="The widest environment axis in yeast, 96 percent chemical. Held out of tier 1 because it is a meta-aggregation: its instance count re-serves primary screens, several already built, so the net-new fraction is unknown until it is de-duplicated against the rest of this table. A loader already exists and retains 49 growth screens, so this is in flight rather than net-new work; Lee 2014 is not among those screens, so YeastPhenome does not backfill row 1.",
        accession="yeastphenome.org (Zenodo 10.5281/zenodo.7714347); loader torchcell/datasets/scerevisiae/yeastphenome.py, 49 screens, not yet in the built set",
        status="loader-in-flight",
    ),
    Candidate(
        name="Piotrowski 2017 (MOSAIC diagnostic panel)",
        citation="Piotrowski JS, Li SC, Deshpande R, et al., Boone C, Myers CL. Nat Chem Biol 2017;13:982-993.",
        url="https://doi.org/10.1038/nchembio.2436",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=157,
        genotypes="~157 diagnostic",
        env_n=13524,
        env="13,524 compounds",
        instances_n=2123268,
        instances_basis="product",
        phenotype="chemogenomic fitness score",
        shape="scalar",
        seq_basis="S288C-KO",
        why="Highest compound count of any yeast screen. Held out of tier 1 on the genotype bar: 157 strains buys condition breadth, not genotype breadth, so it trains a compound embedding rather than a genotype model.",
        accession="MOSAIC portal (mosaic.cs.umn.edu) + Nat Chem Biol SI",
    ),
    Candidate(
        name="Hale 2024 (CRISPRi x natural variation)",
        citation="Hale JJ, Matsui T, Goldstein I, et al., Kruglyak L. Nat Commun 2024;15:4234.",
        url="https://doi.org/10.1038/s41467-024-48626-1",
        klass="Natural variation",
        tier=1,
        genotypes_n=290849,
        genotypes="169 segregants x 1,721 genes",
        env_n=2,
        env="2 induced",
        instances_n=1359774,
        instances_basis="product",
        phenotype="relative fitness (double-barcode seq)",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="Directly measures how a genetic perturbation's effect changes with genetic background. The single best dataset for asking whether a model trained on one background transfers to another.",
        accession="SRA PRJNA986287 + Figshare",
        perturbseq="input",
    ),
    Candidate(
        name="Galardini 2019 (four backgrounds x 38 conditions)",
        citation="Galardini M, Busby BP, Vieitez C, Dunham AS, Typas A, Beltrao P. Mol Syst Biol 2019;15:e8831.",
        url="https://doi.org/10.15252/msb.20198831",
        klass="Natural variation",
        tier=1,
        genotypes_n=15144,
        genotypes="3,786 KOs x 4 backgrounds",
        env_n=38,
        env="38 conditions",
        instances_n=575472,
        instances_basis="product",
        phenotype="colony-size fitness (S-score)",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="The same knockout built independently in four sequenced backgrounds. Quantifies the 18.5 percent of deletion phenotypes that do not transfer, which is the generalization risk the whole substrate is exposed to.",
        accession="GEO GSE123118 + github.com/mgalardini/2018koyeast",
    ),
    Candidate(
        name="Bloom 2019 (16-parent cross)",
        citation="Bloom JS, Boocock J, Treusch S, Sadhu MJ, Day L, Oates-Barker H, Kruglyak L. eLife 2019;8:e49212.",
        url="https://doi.org/10.7554/eLife.49212",
        klass="Natural variation",
        tier=1,
        genotypes_n=14000,
        genotypes="~14,000 segregants",
        env_n=38,
        env="38 traits",
        instances_n=532000,
        instances_basis="product",
        phenotype="quantitative growth traits",
        shape="scalar per trait",
        seq_basis="segregant-WGS",
        why="Largest sequenced recombinant panel in yeast. Sixteen founders means allelic diversity a two-parent cross cannot reach, and the genotypes are sequence, not markers. Built since the previous pass as the 50th supported dataset, so it leaves the list here and enters the supported table.",
        accession="eLife SI + SRA; built as torchcell/datasets/scerevisiae/bloom2019.py",
        status="built",
    ),
    Candidate(
        name="Parsons 2006 (bioactive-compound profiling)",
        citation="Parsons AB, Lopez A, Givoni IE, et al., Boone C. Cell 2006;126:611-625.",
        url="https://doi.org/10.1016/j.cell.2006.06.040",
        klass="Tolerance / robustness",
        tier=1,
        genotypes_n=5000,
        genotypes="~5,000 viable YKO",
        env_n=82,
        env="82 compounds",
        instances_n=410000,
        instances_basis="product",
        phenotype="hypersensitivity score",
        shape="scalar",
        seq_basis="S288C-KO",
        why="Full-collection strain coverage against a modest compound panel, the complement to the diagnostic-panel screens above.",
        accession="Cell SI tables (accession unconfirmed)",
    ),
    Candidate(
        name="Dutta 2026 (barcoded natural-isolate chemical response)",
        citation="Dutta A, Garin M, Loegler V, Brach G, Friedrich A, Yoshimura M, Hirano H, Osada H, Boone C, Yashiroda Y, Hou J, Schacherer J. Nat Commun 2026.",
        url="https://doi.org/10.1038/s41467-026-73532-z",
        klass="Natural variation",
        tier=1,
        genotypes_n=520,
        genotypes="520 natural isolates",
        env_n=600,
        env=">600 compounds",
        instances_n=312000,
        instances_basis="product",
        phenotype="pooled-barcode fitness",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Natural sequence diversity crossed with a chemogenomic panel, on isolates drawn from the sequenced 1,011 collection. The same strains already carry Caudal 2024 transcriptomes and Muenzner 2024 proteomes, so it is a three-modality join on one genotype axis.",
        accession="Nat Commun SI; isolates from the 1,011 panel (ENA)",
    ),
    Candidate(
        name="Peeters 2021 (fermentation-trait QTL atlas)",
        citation="Peeters B, Reijbroek F, Verbaet J, et al., Jarosz DF, Verstrepen KJ. 2021. PMID 34727964.",
        url="https://pubmed.ncbi.nlm.nih.gov/34727964/",
        klass="Natural variation",
        tier=1,
        genotypes_n=1125,
        genotypes="1,125 sequenced segregants",
        env_n=18,
        env="18 traits",
        instances_n=20250,
        instances_basis="product",
        phenotype="ethanol / glycerol / isobutanol titer, stress resistance, aroma",
        shape="scalar per trait",
        seq_basis="segregant-WGS",
        why="The only large sequenced panel with direct isobutanol, ethanol and glycerol titers. Hits the natural-isolate and the isobutanol asks at once.",
        accession="Journal SI; raw reads at SRA/ENA (accession unconfirmed)",
    ),
    Candidate(
        name="Peter 2018 (1,011 isolate genomes + phenome)",
        citation="Peter J, De Chiara M, Friedrich A, et al., Schacherer J. Nature 2018;556:339-344.",
        url="https://doi.org/10.1038/s41586-018-0030-5",
        klass="Natural variation",
        tier=1,
        genotypes_n=1011,
        genotypes="1,011 isolates (971 phenotyped)",
        env_n=36,
        env="36 conditions",
        instances_n=34956,
        instances_basis="product",
        phenotype="growth fitness per condition",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="The genome backbone every other natural-variation row resolves against, and it carries its own phenotype panel. 1,625,809 high-quality SNPs at 232-fold mean coverage.",
        accession="ENA/SRA; phenotype tables via the paper and France Genomique",
    ),
    Candidate(
        name="Cooper 2010 (CE-MS amino-acid metabolome)",
        citation="Cooper SJ, Finney GL, Brown SL, Nelson SI, Hesse J, MacCoss MJ, Fields S. Genome Res 2010;20:1288-1296.",
        url="https://doi.org/10.1101/gr.105825.110",
        klass="Metabolite / precursor",
        tier=1,
        genotypes_n=4700,
        genotypes="~4,700 YKO",
        env_n=1,
        env="1",
        instances_n=4700,
        instances_basis="reported",
        phenotype="free amino-acid pools (capillary electrophoresis)",
        shape="vector (~20)",
        dim=20,
        seq_basis="S288C-KO",
        why="An independent platform measuring the same trait class as the already-built Mulleder 2016 on an overlapping genotype axis. Two independent measurements of one phenotype is the cleanest available test of whether a model has learned biology or a batch.",
        accession="Genome Research SI tables",
    ),
    Candidate(
        name="Chica 2026 (AutoDRY autophagy screen)",
        citation="Chica N, et al. Nat Cell Biol 2026;28:465-479.",
        url="https://doi.org/10.1038/s41556-025-01837-0",
        klass="Modality / backbone",
        tier=1,
        genotypes_n=5919,
        genotypes="4,760 KO + 1,159 DAmP",
        env_n=2,
        env="2 (nitrogen starvation, replete)",
        instances_n=11838,
        instances_basis="product",
        phenotype="autophagy flux (deep-learning image classifier)",
        shape="scalar + per-cell distribution",
        seq_basis="S288C-KO",
        why="Largest single-condition imaging screen with a live Dryad deposit, and it covers essential-adjacent genes via DAmP rather than stopping at the non-essential set.",
        accession="Dryad 10.5061/dryad.cfxpnvxdh",
    ),
    Candidate(
        name="Zhang 2021 (YETI titratable overexpression)",
        citation="Zhang Y, Ho Yee Chow M, Ling Y, et al., Andrews BJ. Mol Syst Biol 2021;17:e10321.",
        url="https://doi.org/10.15252/msb.202010321",
        klass="Metabolite / precursor",
        tier=1,
        genotypes_n=5690,
        genotypes="1,022 essential + 4,668 non-essential",
        env_n=4,
        env="beta-estradiol dose series",
        instances_n=22760,
        instances_basis="estimate",
        phenotype="fitness across an induction gradient",
        shape="dose-response curve",
        seq_basis="S288C+tag",
        why="Native-locus, continuously titratable overexpression. The dosage axis pathway engineering actually uses, and the only gain-of-function library that is not binary.",
        accession="Mol Syst Biol SI (repository accession unconfirmed)",
    ),
    Candidate(
        name="Sopko 2006 (genome-scale overexpression toxicity)",
        citation="Sopko R, Huang D, Preston N, et al., Boone C, Andrews B. Mol Cell 2006;21:319-330.",
        url="https://doi.org/10.1016/j.molcel.2005.12.011",
        klass="Metabolite / precursor",
        tier=1,
        genotypes_n=5280,
        genotypes="~5,280 ORFs",
        env_n=2,
        env="GAL induced vs glucose",
        instances_n=10560,
        instances_basis="product",
        phenotype="growth-rate reduction on induction",
        shape="scalar + phenotype class",
        seq_basis="S288C+ORF-plasmid",
        why="769 genes are growth-inhibitory when overexpressed. Overexpression toxicity is the constraint that caps titer when a pathway enzyme is pushed, and nothing in the built set measures it.",
        accession="Mol Cell SI tables",
    ),
    Candidate(
        name="Kuroda 2019 (isobutanol-specific tolerance)",
        citation="Kuroda K, Hammer SK, Watanabe Y, Montano Lopez J, Fink GR, Stephanopoulos G, Ueda M, Avalos JL. Cell Syst 2019;9:534-547.e5.",
        url="https://doi.org/10.1016/j.cels.2019.10.006",
        klass="Tolerance / robustness",
        tier=1,
        genotypes_n=4800,
        genotypes="~4,800 YKO",
        env_n=2,
        env="isobutanol vs ethanol",
        instances_n=9600,
        instances_basis="product",
        phenotype="isobutanol-specific fitness",
        shape="scalar",
        seq_basis="S288C-KO",
        why="The isobutanol screen, designed against an ethanol comparator so the alcohol-general response subtracts out. Its GLN3 hit raised production 4.9-fold, a rare tolerance-to-titer translation.",
        accession="ArrayExpress E-MTAB-8175 (RNA-seq); screen data in Cell Syst SI",
    ),
    Candidate(
        name="Liu 2021 (tryptophan / isobutanol tolerance)",
        citation="Liu H-L, Wang CH-T, Chiang EP-I, Huang C-C, Li W-H. Biotechnol Biofuels 2021;14:200.",
        url="https://doi.org/10.1186/s13068-021-02048-z",
        klass="Tolerance / robustness",
        tier=1,
        genotypes_n=5006,
        genotypes="5,006 YKO",
        env_n=2,
        env="+/- isobutanol",
        instances_n=10012,
        instances_basis="product",
        phenotype="colony-size fitness (ScreenMill imaging)",
        shape="scalar",
        seq_basis="S288C-KO",
        why="A second, independent full-collection isobutanol screen with a different readout. Paired with Kuroda it gives a same-phenotype, same-genotype, different-method replicate.",
        accession="GEO GSE175794 (companion RNA-seq); screen in BMC SI",
    ),
    Candidate(
        name="Van Leeuwen 2024 (short-chain organic acids)",
        citation="Shared and specific genetic determinants of tolerance to acetic, butyric and octanoic acid. PMC10903034.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10903034/",
        klass="Tolerance / robustness",
        tier=1,
        genotypes_n=4800,
        genotypes="~4,800 YKO",
        env_n=3,
        env="acetic, butyric, octanoic",
        instances_n=14400,
        instances_basis="product",
        phenotype="fitness per acid",
        shape="scalar",
        seq_basis="S288C-KO",
        why="Three chemically homologous acids on one strain panel, so shared and acid-specific mechanisms separate within a single study rather than across incomparable screens.",
        accession="Journal SI (open-access PMC)",
    ),
    Candidate(
        name="Aulakh 2025 (genome-scale ionome)",
        citation="Aulakh SK, et al. Cell Syst 2025;16:101319.",
        url="https://doi.org/10.1016/j.cels.2025.101319",
        klass="Metabolite / precursor",
        tier=1,
        genotypes_n=4800,
        genotypes="~4,800 YKO",
        env_n=1,
        env="1",
        instances_n=4800,
        instances_basis="reported",
        phenotype="intracellular metal-ion content (ICP-MS)",
        shape="vector (multi-element)",
        dim=10,
        seq_basis="S288C-KO",
        why="The only gene-indexed ionome. Metal-cofactor supply gates iron-sulfur and zinc-dependent pathway enzymes, and no built dataset measures it.",
        accession="Cell Systems SI (accession unconfirmed)",
    ),
    Candidate(
        name="Muenzner 2024 (natural-isolate proteome)",
        citation="Muenzner J, et al., Ralser M. Nature 2024;630:149-157.",
        url="https://doi.org/10.1038/s41586-024-07442-9",
        klass="Natural variation",
        tier=1,
        genotypes_n=796,
        genotypes="796 natural isolates",
        env_n=1,
        env="1",
        instances_n=796,
        instances_basis="reported",
        phenotype="protein abundance (DIA-MS)",
        shape="vector",
        dim=2000,
        seq_basis="isolate-WGS",
        why="Proteomes on the sequenced isolate panel whose transcriptomes Caudal 2024 already supplies. That overlap is the paired anchor the RNA-to-protein inference thesis needs.",
        accession="PRIDE PXD048219",
    ),
    Candidate(
        name="Jakobson 2025 (genome-to-proteome map)",
        citation="Jakobson CM, et al. Science 2025;390:eadu3198.",
        url="https://doi.org/10.1126/science.adu3198",
        klass="Natural variation",
        tier=1,
        genotypes_n=800,
        genotypes="800 F6 segregants",
        env_n=1,
        env="1",
        instances_n=800,
        instances_basis="reported",
        phenotype="protein abundance, >6,400 pQTL",
        shape="vector",
        dim=2000,
        seq_basis="segregant-WGS",
        why="Natural alleles mapped to enzyme abundance. A lower-risk engineering lever than a heterologous construct, and the proteome counterpart to the metabolite-QTL panels.",
        accession="Science data-availability statement (accession unconfirmed)",
    ),
    Candidate(
        name="Puddu 2019 (WGS of the deletion collection)",
        citation="Puddu F, et al., Jackson SP. Nature 2019;573:416-420.",
        url="https://doi.org/10.1038/s41586-019-1549-9",
        klass="Modality / backbone",
        tier=1,
        genotypes_n=4800,
        genotypes="~4,800 deletion strains",
        env_n=1,
        env="1",
        instances_n=4800,
        instances_basis="reported",
        phenotype="structural variants, CNV, aneuploidy",
        shape="variant call set",
        seq_basis="S288C-KO",
        why="Whole-genome sequence for every strain in the deletion collection. This is what turns the S288C-KO sequence basis from an assumption into a measurement, and it applies to roughly half the rows in this table at once.",
        accession="ENA/SRA (accession unconfirmed)",
    ),
    # -- Tier 2 -------------------------------------------------------------
    Candidate(
        name="Dong 2021 (MAGIC + SAM biosensor)",
        citation="Dong C, Schultz JC, Liu W, Lian J, Huang L, Xu Z, Zhao H. Metab Eng 2021;66:319-327.",
        url="https://doi.org/10.1016/j.ymben.2021.03.005",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=100000,
        genotypes="~100,000 guides (a/i/d)",
        env_n=1,
        env="1",
        instances_n=100000,
        instances_basis="estimate",
        phenotype="SAM biosensor fluorescence (FACS)",
        shape="scalar",
        seq_basis="S288C+guide",
        why="MAGIC re-run with a metabolite biosensor instead of a growth selection, so the label is product concentration rather than fitness. The pattern to copy for every precursor we care about, and the sibling of the already-built Lian 2019 furfural screen.",
        accession="Metab Eng SI (accession unconfirmed)",
    ),
    Candidate(
        name="Bao 2018 (CHAnGE single-nucleotide library)",
        citation="Bao Z, HamediRad M, Xue P, Xiao H, Tasan I, Chao R, Liang J, Zhao H. Nat Biotechnol 2018;36:505-508.",
        url="https://doi.org/10.1038/nbt.4132",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=60000,
        genotypes="tens of thousands of designed variants",
        env_n=1,
        env="inhibitor selection",
        instances_n=60000,
        instances_basis="estimate",
        phenotype="variant fitness (pooled barcode seq)",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="Single-nucleotide resolution rather than whole-gene nulls. The only genotype axis in this table finer than an ORF, and the one that matches how a real strain-design edit is specified.",
        accession="Nat Biotechnol SI (raw sequencing location unconfirmed)",
    ),
    Candidate(
        name="McGlincy 2021 (genome-scale CRISPRi library)",
        citation="McGlincy NJ, Meacham ZA, Reynaud KK, Muller R, Baum R, Ingolia NT. BMC Genomics 2021;22:205.",
        url="https://doi.org/10.1186/s12864-021-07518-0",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=61094,
        genotypes="61,094 guides, ~10/gene",
        env_n=1,
        env="continuous culture",
        instances_n=61094,
        instances_basis="reported",
        phenotype="per-guide fitness",
        shape="scalar",
        seq_basis="S288C+guide",
        why="Ten guides per gene gives a graded knockdown axis instead of a binary null, and it covers essential genes. This is the library a yeast Perturb-seq would most plausibly be built on.",
        accession="ingolia-lab.org/yeast-crispri + Addgene + BMC SI",
        perturbseq="input",
    ),
    Candidate(
        name="Roy 2018 (multiplexed precision editing)",
        citation="Roy KR, Smith JD, Vonesch SC, et al., St Onge RP. Nat Biotechnol 2018;36:512-520.",
        url="https://doi.org/10.1038/nbt.4137",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=16000,
        genotypes="thousands of designed edits",
        env_n=1,
        env="1",
        instances_n=16000,
        instances_basis="estimate",
        phenotype="variant fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="Genomic rather than plasmid barcodes, which removes the barcode-swapping artifact that quietly corrupts pooled fitness data. The methodological complement to CHAnGE.",
        accession="Nat Biotechnol SI (accession unconfirmed)",
    ),
    Candidate(
        name="Momen-Roknabadi 2020 (inducible CRISPRi library)",
        citation="Momen-Roknabadi A, Oikonomou P, Zegans M, Tavazoie S. Commun Biol 2020;3:723.",
        url="https://doi.org/10.1038/s42003-020-01452-9",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=30000,
        genotypes="genome-wide guide library",
        env_n=3,
        env="nutrient limitations",
        instances_n=90000,
        instances_basis="estimate",
        phenotype="per-guide fitness under induction",
        shape="scalar",
        seq_basis="S288C+guide",
        why="A second CRISPRi library on a different vector and guide-design rule. Whether a model trained on one library transfers to the other is a cheap, decisive test of guide-level overfitting. Not a Perturb-seq row: one edit per cell and a scalar fitness readout is low-dimensional on both axes, however large the library.",
        accession="Addgene + Commun Biol SI",
    ),
    Candidate(
        name="Yoshikawa 2011 (deletion + overexpression phenome)",
        citation="Yoshikawa K, Tanaka T, Furusawa C, Nagahisa K, Hirasawa T, Shimizu H. G3 2011;1:247-267.",
        url="https://doi.org/10.1534/g3.111.000695",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=9600,
        genotypes="~4,800 KO + matched overexpression",
        env_n=6,
        env="carbon sources / stress",
        instances_n=57600,
        instances_basis="estimate",
        phenotype="growth-rate fitness",
        shape="scalar",
        seq_basis="S288C-KO",
        why="The same genes phenotyped in both loss- and gain-of-function side by side. A matched over/under pair on one platform is what a dosage-aware model needs and nothing else here provides.",
        accession="G3 open-access SI",
    ),
    Candidate(
        name="Mukherjee 2021 (CRISPRi essential genes x acetic acid)",
        citation="Mukherjee V, Lind U, St Onge RP, Blomberg A, Nystrom T. mSystems 2021;6:e00410-21.",
        url="https://doi.org/10.1128/mSystems.00410-21",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=1100,
        genotypes="~1,100 essential genes",
        env_n=2,
        env="acetic acid vs control",
        instances_n=2200,
        instances_basis="product",
        phenotype="fitness under acid stress",
        shape="scalar",
        seq_basis="S288C+guide",
        why="Acetic acid is the dominant hydrolysate stressor, and this reaches the essential genes a deletion collection structurally cannot.",
        accession="mSystems open-access SI",
    ),
    Candidate(
        name="Pereira 2014 (wheat-straw hydrolysate)",
        citation="Pereira FB, Guimaraes PMR, Gomes DG, et al., Domingues L. J Ind Microbiol Biotechnol 2014;41:1753-1761.",
        url="https://doi.org/10.1007/s10295-014-1519-z",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=4800,
        genotypes="EUROSCARF deletion collection",
        env_n=2,
        env="hydrolysate vs control",
        instances_n=9600,
        instances_basis="product",
        phenotype="tolerance to industrial hydrolysate",
        shape="scalar",
        seq_basis="S288C-KO",
        why="Real industrial wheat-straw hydrolysate rather than reconstituted single toxins. The recalcitrant-biomass condition as a process actually presents it.",
        accession="Journal SI",
    ),
    Candidate(
        name="Endo 2008 (vanillin tolerance)",
        citation="Endo A, Nakamura T, Ando A, Tokuyasu K, Shima J. Appl Environ Microbiol 2008;74:7175-7185.",
        url="https://doi.org/10.1128/AEM.01541-08",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=4800,
        genotypes="diploid deletion collection",
        env_n=2,
        env="vanillin vs control",
        instances_n=9600,
        instances_basis="product",
        phenotype="vanillin sensitivity",
        shape="scalar",
        seq_basis="S288C-KO",
        why="Vanillin is the one major hydrolysate phenolic with no dedicated screen in the built set; it is currently only folded into broad compound panels.",
        accession="AEM SI (PMC2375868)",
    ),
    Candidate(
        name="Xiao 2014 (genome-wide RNAi furfural tolerance)",
        citation="Xiao H, Zhao H. Biotechnol Biofuels 2014;7:78.",
        url="https://doi.org/10.1186/1754-6834-7-78",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=10000,
        genotypes="genome-scale RNAi library",
        env_n=2,
        env="furfural vs control",
        instances_n=20000,
        instances_basis="estimate",
        phenotype="knockdown enrichment under furfural",
        shape="scalar",
        seq_basis="S288C+guide",
        why="Furfural via a knockdown rather than a deletion axis, so it reaches essential genes and gives a dose-graded rather than binary perturbation on the flagship hydrolysate inhibitor.",
        accession="BMC open-access SI",
    ),
    Candidate(
        name="Crook 2016 (tunable RNAi, isobutanol + 1-butanol)",
        citation="Crook N, Sun J, Morse N, Schmitz A, Alper HS. Appl Microbiol Biotechnol 2016;100:10005-10018. PMID 27654654.",
        url="https://pubmed.ncbi.nlm.nih.gov/27654654/",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=10000,
        genotypes="genome-scale tunable RNAi",
        env_n=2,
        env="isobutanol, 1-butanol",
        instances_n=20000,
        instances_basis="estimate",
        phenotype="dose-graded knockdown fitness under alcohol stress",
        shape="dose-response curve",
        seq_basis="S288C+guide",
        why="Two related alcohols at graded knockdown strength. The dosage axis is the signal a binary screen throws away, and Hsp70 emerged only because of it.",
        accession="AMB SI (DOI is an open verification item)",
    ),
    Candidate(
        name="Si 2017 (RAGE RNAi, isobutanol titer)",
        citation="Si T, Chao R, Min Y, Wu Y, Ren W, Zhao H. Nat Commun 2017;8:15187.",
        url="https://doi.org/10.1038/ncomms15187",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=10000,
        genotypes="genome-scale RNAi, iterative rounds",
        env_n=1,
        env="isobutanol selection",
        instances_n=10000,
        instances_basis="estimate",
        phenotype="construct enrichment; isobutanol titer for top strains",
        shape="scalar",
        seq_basis="S288C+guide",
        why="One of very few library screens whose endpoint is a titer in g/L rather than a fitness proxy, and it is iterative, so it carries combinatorial rather than single-locus effects.",
        accession="Nat Commun SI (accession unconfirmed)",
    ),
    Candidate(
        name="HamediRad 2018 (RAGE xylose utilization)",
        citation="HamediRad M, Lian J, Li H, Zhao H. Biotechnol Bioeng 2018;115:1552-1560.",
        url="https://doi.org/10.1002/bit.26570",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=10000,
        genotypes="genome-scale RNAi",
        env_n=1,
        env="xylose-limited",
        instances_n=10000,
        instances_basis="estimate",
        phenotype="enrichment; xylose consumption rate",
        shape="scalar",
        seq_basis="S288C+guide",
        why="Pentose utilization is the other half of the recalcitrant-biomass problem, and no built dataset covers a xylose selection.",
        accession="Biotechnol Bioeng SI (accession unconfirmed)",
    ),
    Candidate(
        name="Blank 2005 (13C metabolic flux)",
        citation="Blank LM, Kuepfer L, Sauer U. Genome Biol 2005;6:R49.",
        url="https://doi.org/10.1186/gb-2005-6-6-r49",
        klass="Metabolite / precursor",
        tier=2,
        genotypes_n=200,
        genotypes="~200 viable null mutants",
        env_n=1,
        env="glucose minimal",
        instances_n=200,
        instances_basis="estimate",
        phenotype="intracellular flux distribution (13C-MFA)",
        shape="vector (central carbon)",
        dim=30,
        seq_basis="S288C-KO",
        why="The only real flux measurement tied to single-gene deletions. Everything else in the built set is a concentration or a titer, which is a proxy for flux rather than flux.",
        accession="Genome Biology open-access SI",
        requested=True,
    ),
    Candidate(
        name="Zhu 2014 (kinase / phosphatase lipidomics)",
        citation="Zhu Z, Loewen CJR. Mol Biol Cell 2014. PMC4196872.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4196872/",
        klass="Metabolite / precursor",
        tier=2,
        genotypes_n=129,
        genotypes="129 kinase + phosphatase KOs",
        env_n=1,
        env="1",
        instances_n=129,
        instances_basis="reported",
        phenotype="lipid class and species abundance",
        shape="vector",
        dim=200,
        seq_basis="S288C-KO",
        why="A gene-indexed lipidome on a regulatory gene set. Complements the already-built da Silveira 2014 lipidomics, which has no kinase axis, and reads out the malonyl-CoA sink directly.",
        accession="Mol Biol Cell SI Table S4 (Excel)",
    ),
    Candidate(
        name="Trikka 2015 (carotenogenic heterozygous screen)",
        citation="Trikka FA, Nikolaidis A, Athanasakoglou A, et al., Kampranis SC, Makris AM. Microb Cell Fact 2015;14:60.",
        url="https://doi.org/10.1186/s12934-015-0246-0",
        klass="Metabolite / precursor",
        tier=2,
        genotypes_n=4700,
        genotypes="4,700 heterozygous deletions",
        env_n=1,
        env="1",
        instances_n=4700,
        instances_basis="reported",
        phenotype="carotenoid color; sclareol titer for top strains",
        shape="ordinal, then scalar",
        seq_basis="S288C-KO/het",
        why="The second carotenoid deletion-collection screen, and the only haploinsufficiency design here. Halving dosage finds genes a null would kill, which is where the isoprenoid flux control sits. Currently listed as figure-only, so confirm Additional file 1 carries per-strain scores.",
        accession="BMC Additional files 1-2",
        requested=True,
    ),
    Candidate(
        name="Lian 2017 (CRISPR-AID, beta-carotene)",
        citation="Lian J, HamediRad M, Hu S, Zhao H. Nat Commun 2017;8:1688.",
        url="https://doi.org/10.1038/s41467-017-01695-x",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=200,
        genotypes="combinatorial a/i/d triples",
        env_n=1,
        env="1",
        instances_n=200,
        instances_basis="estimate",
        phenotype="beta-carotene titer",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="The tri-functional chassis MAGIC was built on, read out on beta-carotene. Small, but it is a combinatorial genotype with a measured product titer, which is the exact record type inverse strain design needs.",
        accession="Nat Commun SI",
    ),
    Candidate(
        name="Chang 2012 (organic-acid production screen)",
        citation="Chang HJ, Suga H, et al. 2012. PMID 22277779.",
        url="https://pubmed.ncbi.nlm.nih.gov/22277779/",
        klass="Metabolite / precursor",
        tier=2,
        genotypes_n=4800,
        genotypes="~4,800 YKO",
        env_n=1,
        env="1",
        instances_n=4800,
        instances_basis="reported",
        phenotype="acetate / pyruvate / succinate halo (pH indicator)",
        shape="ordinal",
        seq_basis="S288C-KO",
        why="A whole-collection screen whose label is production rather than fitness. Coarse and PDF-only, but organic-acid overproduction at collection scale exists nowhere else.",
        accession="PDF tables only; manual transcription required",
    ),
    Candidate(
        name="Gurvitz (fatty-acid utilization screen)",
        citation="Gurvitz A, et al. Mol Syst Biol (EMBO Press).",
        url="https://www.embopress.org/doi/pdf/10.1038/msb4100051",
        klass="Metabolite / precursor",
        tier=2,
        genotypes_n=4800,
        genotypes="~4,800 viable YKO",
        env_n=3,
        env="oleate, myristate, acetate",
        instances_n=14400,
        instances_basis="product",
        phenotype="fatty-acid utilization competence",
        shape="ordinal",
        seq_basis="S288C-KO",
        why="Whole-collection coverage against two fatty-acid substrates. Relevant to the lipid and malonyl-CoA branch, at the cost of manual extraction from an embedded table.",
        accession="Mol Syst Biol Table I (PDF-embedded)",
    ),
    Candidate(
        name="Ho 2009 (MoBY-ORF barcoded overexpression)",
        citation="Ho CH, Magtanong L, Barker SL, et al., Boone C. Nat Biotechnol 2009;27:369-377.",
        url="https://doi.org/10.1038/nbt.1534",
        klass="Metabolite / precursor",
        tier=2,
        genotypes_n=5100,
        genotypes="~5,100 barcoded ORFs",
        env_n=10,
        env="compound panel",
        instances_n=51000,
        instances_basis="estimate",
        phenotype="fitness per compound",
        shape="scalar",
        seq_basis="S288C+ORF-plasmid",
        why="Native-promoter overexpression against a compound panel, giving a third dosage state alongside deletion and knockdown on the same chemogenomic readout.",
        accession="moby.ccbr.utoronto.ca + Andrews lab portal",
    ),
    Candidate(
        name="Douglas 2012 (barcoded overexpression fitness)",
        citation="Douglas AC, Smith AM, Sharifpoor S, et al., Andrews BJ, Boone C, Nislow C. G3 2012;2:1279-1289.",
        url="https://doi.org/10.1534/g3.112.003400",
        klass="Metabolite / precursor",
        tier=2,
        genotypes_n=5000,
        genotypes="genome-scale barcoded ORFs",
        env_n=4,
        env="baseline + stress",
        instances_n=20000,
        instances_basis="estimate",
        phenotype="pooled competitive fitness on induction",
        shape="scalar",
        seq_basis="S288C+ORF-plasmid",
        why="The same barcode-sequencing pipeline as the already-built Hillenmeyer FitDb, run in the opposite dosage direction. Lowest harmonization cost of any overexpression row.",
        accession="G3 open-access SI",
    ),
    Candidate(
        name="Costello 2020 (bioreactor Bar-seq)",
        citation="Costello Z, Wehrs M, Mukhopadhyay A, et al. Microb Cell Fact 2020;19:167.",
        url="https://doi.org/10.1186/s12934-020-01423-z",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=4800,
        genotypes="pooled YKO library",
        env_n=4,
        env="feed schemes / pH",
        instances_n=19200,
        instances_basis="estimate",
        phenotype="fitness under fed-batch cultivation",
        shape="scalar",
        seq_basis="S288C-KO",
        why="The only fitness measurement at bioreactor scale. Shake-flask ranking is known to diverge from fed-batch, and nothing else here tests that.",
        accession="Microb Cell Fact open-access SI",
    ),
    Candidate(
        name="Ambroset 2014 (metabolite QTL)",
        citation="Ambroset C, et al., Fay JC. PLoS Genet 2014;10:e1004142.",
        url="https://doi.org/10.1371/journal.pgen.1004142",
        klass="Natural variation",
        tier=2,
        genotypes_n=100,
        genotypes="~100 segregants",
        env_n=1,
        env="1",
        instances_n=100,
        instances_basis="reported",
        phenotype="74 metabolite concentrations (LC-MS/MS)",
        shape="vector (74)",
        dim=74,
        seq_basis="segregant-WGS",
        why="The founding metabolite-QTL panel. Pairs with Cooper and Mulleder as the natural-variation view of a phenotype space the deletion collection covers by engineering.",
        accession="PLOS Genetics SI",
    ),
    Candidate(
        name="Gerke 2017 (urea-cycle mQTL)",
        citation="Gerke J, et al., Fay JC. Genetics 2017;206:2199.",
        url="https://doi.org/10.1534/genetics.117.201107",
        klass="Natural variation",
        tier=2,
        genotypes_n=147,
        genotypes="147 diploid segregants",
        env_n=1,
        env="1",
        instances_n=147,
        instances_basis="reported",
        phenotype="untargeted metabolite abundance",
        shape="vector",
        dim=100,
        seq_basis="segregant-WGS",
        why="A second mQTL panel from a different cross (oak x wine), so cross-background generalization of metabolite QTL can be tested rather than assumed.",
        accession="Genetics (GSA) SI",
    ),
    Candidate(
        name="Cubillos 2017 (nitrogen-consumption QTL)",
        citation="Cubillos FA, Brice C, Molinet J, et al., Martinez C. G3 2017;7:1693-1705.",
        url="https://doi.org/10.1534/g3.117.042127",
        klass="Natural variation",
        tier=2,
        genotypes_n=165,
        genotypes="165 sequenced F12 segregants",
        env_n=1,
        env="nitrogen-limited fermentation",
        instances_n=165,
        instances_basis="reported",
        phenotype="nitrogen consumption",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="One of the few QTL panels whose genotype location is confirmed rather than assumed, and nitrogen assimilation sets higher-alcohol and ester formation.",
        accession="BioProject PRJNA379146 (RNA-seq); genotypes in Cubillos 2013 Table S1",
    ),
    Candidate(
        name="Hackett 2016 (SIMMER multi-omic flux)",
        citation="Hackett SR, Zanotelli VRT, Xu W, et al., Rabinowitz JD. Science 2016;354:aaf2786.",
        url="https://doi.org/10.1126/science.aaf2786",
        klass="Expression / single cell",
        tier=2,
        genotypes_n=1,
        genotypes="wild type",
        env_n=25,
        env="25 chemostat states",
        instances_n=25,
        instances_basis="reported",
        phenotype="paired metabolome + proteome + flux",
        shape="multi-omic triplet",
        dim=3000,
        seq_basis="reference-only",
        why="The only jointly measured metabolome, proteome and fluxome. No genotype axis, so it is a mechanism prior rather than training data, but it is the only place the three layers are measured on the same cells.",
        accession="Science SI (accession unconfirmed)",
    ),
    Candidate(
        name="Gasch 2000 (environmental stress response)",
        citation="Gasch AP, Spellman PT, Kao CM, et al., Brown PO. Mol Biol Cell 2000;11:4241-4257.",
        url="https://doi.org/10.1091/mbc.11.12.4241",
        klass="Expression / single cell",
        tier=2,
        genotypes_n=1,
        genotypes="wild type",
        env_n=9,
        env="~9 stress series, multi-timepoint",
        instances_n=150,
        instances_basis="estimate",
        phenotype="mRNA abundance time course",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="The canonical stress-response signature. Every genotype-indexed transcriptome in the built set needs it as the background any generic stress response is subtracted against.",
        accession="SPELL / SGD Expression Connection; Stanford legacy mirror",
    ),
    Candidate(
        name="Leutert 2023 (phosphoproteome x 101 conditions)",
        citation="Leutert M, et al., Villen J. Nat Struct Mol Biol 2023;30:1761-1773.",
        url="https://doi.org/10.1038/s41594-023-01115-3",
        klass="Expression / single cell",
        tier=2,
        genotypes_n=1,
        genotypes="wild type",
        env_n=101,
        env="101 conditions",
        instances_n=101,
        instances_basis="reported",
        phenotype="phosphosite abundance",
        shape="vector (thousands)",
        dim=5000,
        seq_basis="reference-only",
        why="Post-translational enzyme control acts faster than transcription and is invisible to every other layer here. Condition axis only.",
        accession="PRIDE PXD034997",
    ),
    Candidate(
        name="Gameiro 2025 (AID2 degron libraries)",
        citation="Gameiro E, Juarez-Nunez KA, Fung JJ, Shankar S, Luke B, Khmelinskii A. J Cell Biol 2025;224:e202409007.",
        url="https://doi.org/10.1083/jcb.202409007",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=5600,
        genotypes=">5,600 tagged ORFs",
        env_n=2,
        env="baseline + hydroxyurea",
        instances_n=11200,
        instances_basis="product",
        phenotype="degradation efficiency; fitness under stress",
        shape="scalar",
        seq_basis="S288C+tag",
        why="Acute depletion on a minutes-to-hours timescale, covering essential genes without the pre-adaptation a deletion strain has already done.",
        accession="J Cell Biol SI; strains via IMB Mainz / EUROSCARF",
    ),
    Candidate(
        name="Breslow 2008 (DAmP hypomorphic collection)",
        citation="Breslow DK, Cameron DM, Collins SR, et al., Weissman JS. Nat Methods 2008;5:711-718.",
        url="https://doi.org/10.1038/nmeth.1234",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=1812,
        genotypes="842 haploid + 970 diploid essential",
        env_n=1,
        env="1",
        instances_n=1812,
        instances_basis="reported",
        phenotype="competitive fitness",
        shape="scalar",
        seq_basis="S288C+tag",
        why="Partial loss of function for essential genes at the native locus, about 82 percent of the essential genome. The built set has no essential-gene hypomorph axis at all.",
        accession="Nat Methods SI; Horizon YSC5050/5090/5093/5094",
    ),
    Candidate(
        name="Kofoed 2015 (barcoded ts alleles)",
        citation="Kofoed M, Milbury KL, Chiang JH, et al., Hieter P, Stirling PC. G3 2015;5:1879-1887.",
        url="https://doi.org/10.1534/g3.115.019174",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=600,
        genotypes="600+ barcoded ts alleles",
        env_n=2,
        env="permissive / restrictive",
        instances_n=1200,
        instances_basis="product",
        phenotype="ts severity (growth ratio)",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="Conditional essential-gene loss of function as point mutants at native loci, barcoded so it runs through the same pooled pipeline as the deletion collection.",
        accession="G3 open-access SI; strains via Dharmacon/Horizon",
    ),
    Candidate(
        name="Chong 2015 / CYCLoPs (single-cell proteome dynamics)",
        citation="Chong YT, Koh JLY, Friesen H, et al., Andrews BJ, Boone C. Cell 2015;161:1413-1424.",
        url="https://doi.org/10.1016/j.cell.2015.04.051",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=4100,
        genotypes="~4,100 GFP-tagged ORFs",
        env_n=6,
        env="cell-cycle / stress",
        instances_n=24600,
        instances_basis="estimate",
        phenotype="abundance + localization class",
        shape="per-cell distribution",
        dim=17,
        seq_basis="S288C+tag",
        why="Per-cell rather than per-strain-mean labels, over 20 million cells. The distributional readout a Perturb-seq design has to be scored against.",
        accession="thecellvision.org/cyclops (bulk CSV)",
        perturbseq="output",
    ),
    Candidate(
        name="Decourty 2021 (RNA-metabolism genetic interactions)",
        citation="Decourty L, et al., Saveanu C. Nucleic Acids Res 2021;49:8535-8555.",
        url="https://doi.org/10.1093/nar/gkab680",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=700000,
        genotypes="~700,000 double mutants",
        env_n=1,
        env="1",
        instances_n=700000,
        instances_basis="reported",
        phenotype="genetic-interaction score",
        shape="scalar (edge)",
        seq_basis="S288C-KO",
        why="Structurally identical to Costanzo and Kuzmin, so the existing loader pattern applies directly, and it extends the interaction network into RNA processing.",
        accession="NAR open-access SI (accession unconfirmed)",
    ),
    Candidate(
        name="Sharifpoor 2012 (kinome synthetic dosage lethality)",
        citation="Sharifpoor S, van Dyk D, Costanzo M, et al., Boone C, Andrews BJ. Genome Res 2012;22:791-801.",
        url="https://doi.org/10.1101/gr.129213.111",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=8700,
        genotypes="92 kinase queries, genome-wide array",
        env_n=1,
        env="1",
        instances_n=8700,
        instances_basis="reported",
        phenotype="genetic interaction / SDL score",
        shape="scalar (edge)",
        seq_basis="S288C-KO",
        why="The overexpression-by-deletion axis, which neither the built SynLethDB nor Costanzo and Kuzmin carries: SynLethDB models only synthetic lethality and synthetic rescue, both two-loss-of-function, so synthetic dosage lethality is out of its scope by construction and this paper's PMID is absent from both its layers. Kinases are also the regulatory nodes flux redirection actually targets.",
        accession="Genome Research SI",
    ),
    # -- Tier 3 -------------------------------------------------------------
    Candidate(
        name="Brettner 2024 (ultra-high-throughput yeast scRNA-seq)",
        citation="Brettner L, et al. Yeast 2024. doi:10.1002/yea.3927.",
        url="https://doi.org/10.1002/yea.3927",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=None,
        genotypes="method",
        env_n=None,
        env="method",
        instances_n=None,
        instances_basis="estimate",
        phenotype="single-cell transcriptome",
        shape="vector",
        seq_basis="reference-only",
        why="The platform the costing model in experiments/024 is built on. Not a dataset to ingest; the reason it is here is that a yeast Perturb-seq proposal has to name its chemistry.",
        accession="Yeast (Wiley) SI",
        perturbseq="output",
    ),
    Candidate(
        name="Nadal-Ribelles 2019 (sensitive yeast scRNA-seq)",
        citation="Nadal-Ribelles M, Islam S, Wei W, et al., Posas F, Steinmetz LM. Nat Microbiol 2019;4:683-692.",
        url="https://doi.org/10.1038/s41564-018-0346-9",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=1,
        genotypes="clonal wild type",
        env_n=2,
        env="baseline + stress",
        instances_n=2,
        instances_basis="estimate",
        phenotype="single-cell transcriptome",
        shape="vector",
        dim=6000,
        seq_basis="reference-only",
        why="Establishes within-clone transcript correlation, which is the noise floor any per-strain Perturb-seq mean has to clear. Directly sizes the cells-per-guide question.",
        accession="Nat Microbiol SI / GEO",
        perturbseq="output",
    ),
    Candidate(
        name="Gasch 2017 (single-cell stress heterogeneity)",
        citation="Gasch AP, Yu FB, Hose J, et al., Quake SR. PLoS Biol 2017;15:e2004050.",
        url="https://doi.org/10.1371/journal.pbio.2004050",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=2,
        env="stress vs unstressed",
        instances_n=2,
        instances_basis="estimate",
        phenotype="single-cell transcriptome",
        shape="vector",
        dim=6000,
        seq_basis="reference-only",
        why="Separates intrinsic from extrinsic expression heterogeneity under stress. Sets what fraction of Perturb-seq variance is recoverable signal rather than cell-state noise.",
        accession="PLoS Biology SI / GEO",
        perturbseq="output",
    ),
    Candidate(
        name="N'Guessan 2025 (segregant scRNA-seq eQTL)",
        citation="N'Guessan A, Boocock J, Kruglyak L, Albert FW. eLife 2025. PMC12303567.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12303567/",
        klass="Expression / single cell",
        tier=2,
        genotypes_n=4500,
        genotypes="~4,500 profiled segregants",
        env_n=1,
        env="1",
        instances_n=4500,
        instances_basis="reported",
        phenotype="single-cell transcriptome + eQTL",
        shape="vector",
        dim=6000,
        seq_basis="segregant-WGS",
        why="Single-cell transcriptomes indexed by recombinant natural genotype rather than by an engineered knockout. The closest existing analog to the proposed Perturb-seq design, and the only one that already pairs per-cell expression with sequenced genotypes at scale.",
        accession="eLife data availability; segregant panel from Bloom lineage",
        perturbseq="both",
    ),
    Candidate(
        name="Valenti 2025 (SWAT degron + GFP collection)",
        citation="Valenti M, et al. J Cell Biol 2025;224:e202409050.",
        url="https://doi.org/10.1083/jcb.202409050",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=5000,
        genotypes="proteome-wide SWAT library",
        env_n=2,
        env="depleted / not",
        instances_n=10000,
        instances_basis="estimate",
        phenotype="depletion efficiency + abundance/localization",
        shape="scalar + categorical",
        seq_basis="S288C+tag",
        why="Depletion and its abundance consequence measured in the same strain, so enzyme dosage and its phenotype are not joined across two experiments.",
        accession="J Cell Biol SI",
    ),
    Candidate(
        name="Weill 2018 (SWAT libraries)",
        citation="Weill U, Yofe I, Sass E, et al., Schuldiner M. Nat Methods 2018;15:617-622.",
        url="https://doi.org/10.1038/s41592-018-0044-9",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=5500,
        genotypes="~5,500 acceptor-tagged strains",
        env_n=1,
        env="1",
        instances_n=5500,
        instances_basis="reported",
        phenotype="abundance, localization, topology",
        shape="scalar + categorical",
        dim=3,
        seq_basis="S288C+tag",
        why="The library the degron rows are built from. Ingest for its abundance and localization tables; its main value is as the cassette definition those rows' sequences resolve against.",
        accession="Nat Methods SI; strains via EUROSCARF",
    ),
    Candidate(
        name="Huh 2003 (GFP localization atlas)",
        citation="Huh WK, Falvo JV, Gerke LC, et al., O'Shea EK. Nature 2003;425:686-691.",
        url="https://doi.org/10.1038/nature02026",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=4156,
        genotypes="~4,156 GFP-tagged ORFs",
        env_n=1,
        env="1",
        instances_n=4156,
        instances_basis="reported",
        phenotype="subcellular localization (22 classes)",
        shape="categorical",
        seq_basis="S288C+tag",
        why="The wild-type localization baseline every perturbation-imaging dataset measures change against. Compartment identity is a prerequisite for compartmentalized pathway engineering.",
        accession="SGD / YeastGFP mirror",
    ),
    Candidate(
        name="Ghaemmaghami 2003 (absolute protein abundance)",
        citation="Ghaemmaghami S, Huh WK, Bower K, et al., Weissman JS. Nature 2003;425:737-741.",
        url="https://doi.org/10.1038/nature02046",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=4251,
        genotypes="~4,251 TAP-tagged ORFs",
        env_n=1,
        env="1",
        instances_n=4251,
        instances_basis="reported",
        phenotype="absolute abundance (molecules/cell)",
        shape="scalar",
        seq_basis="S288C+tag",
        why="Converts every relative proteomics row here into molecules per cell. Without an absolute anchor, enzyme-cost and flux-capacity arguments have no units.",
        accession="Nature SI (Excel)",
    ),
    Candidate(
        name="Mulleder 2012 (prototrophic deletion collection)",
        citation="Mulleder M, Capuano F, Pir P, et al., Ralser M. Nat Biotechnol 2012;30:1176-1178.",
        url="https://doi.org/10.1038/nbt.2442",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=6024,
        genotypes="5,185 prototrophic + 839 titratable essential",
        env_n=1,
        env="1",
        instances_n=None,
        instances_basis="estimate",
        phenotype="strain resource (no phenotype)",
        shape="n/a",
        seq_basis="S288C-KO",
        why="The background the already-built Mulleder 2016 and Messner 2023 were measured in, and mulleder2016.py already cites it. Listed not as an ingestion target but because that loader records the prototrophy-restoring markers as unmodeled: they are a GeneAddition this schema does not yet carry, and auxotrophy markers distort metabolite pools.",
        accession="EUROSCARF; Addgene kit 40276",
    ),
    Candidate(
        name="Zackrisson 2016 (Scan-o-matic growth curves)",
        citation="Zackrisson M, Hallin J, Ottosson LG, et al., Warringer J, Blomberg A. G3 2016;6:3003-3014.",
        url="https://doi.org/10.1534/g3.116.032342",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=4700,
        genotypes="YKO-compatible platform",
        env_n=None,
        env="panel-dependent",
        instances_n=None,
        instances_basis="estimate",
        phenotype="full growth curve (rate, lag, yield)",
        shape="time series",
        seq_basis="S288C-KO",
        why="Time-series rather than endpoint fitness. A platform, so a specific released condition panel has to be chosen before this becomes a row rather than a plan.",
        accession="Open-source platform; per-study datasets",
    ),
    Candidate(
        name="Michaelis 2023 (protein interactome)",
        citation="Michaelis AC, et al., Mann M. Nature 2023;624:192-200.",
        url="https://doi.org/10.1038/s41586-023-06739-5",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=4000,
        genotypes="proteome-scale AP-MS baits",
        env_n=1,
        env="1",
        instances_n=4000,
        instances_basis="estimate",
        phenotype="protein-protein interaction + interface",
        shape="scalar (edge)",
        seq_basis="S288C+tag",
        why="A physical-interaction layer to constrain which genetic interactions reflect complex membership. A graph prior rather than a phenotype.",
        accession="yeast-interactome.biochem.mpg.de + ProteomeXchange",
    ),
    Candidate(
        name="Braberg 2020 (point-mutant E-MAP)",
        citation="Braberg H, et al., Krogan NJ. Science 2020;370:eaaz4910.",
        url="https://doi.org/10.1126/science.aaz4910",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=350,
        genotypes="350 point mutants x genome-wide array",
        env_n=1,
        env="1",
        instances_n=500000,
        instances_basis="reported",
        phenotype="residue-level genetic interaction",
        shape="scalar (edge)",
        seq_basis="S288C+designed-edit",
        why="Sub-gene resolution genetic interactions. Narrow in gene scope, but it is the only demonstration that the interaction formalism extends below the ORF.",
        accession="Science SI (accession unconfirmed)",
    ),
    Candidate(
        name="Schulte 2023 (mitochondrial complexome)",
        citation="Schulte U, et al. Nature 2023;614:153-159.",
        url="https://doi.org/10.1038/s41586-022-05641-w",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=900,
        genotypes="~900 mitochondrial proteins",
        env_n=1,
        env="1",
        instances_n=900,
        instances_basis="reported",
        phenotype="complex assembly profile",
        shape="vector",
        seq_basis="reference-only",
        why="Mitochondrial compartmentalization is a live strategy for sequestering toxic intermediates, and this maps the import and assembly bottlenecks that would cap it.",
        accession="complexomics.org + ProteomeXchange",
    ),
    Candidate(
        name="Yue 2017 (wild vs domesticated assemblies)",
        citation="Yue JX, Li J, Aigrain L, et al., Liti G. Nat Genet 2017;49:913-924.",
        url="https://doi.org/10.1038/ng.3847",
        klass="Natural variation",
        tier=3,
        genotypes_n=12,
        genotypes="12 long-read assemblies",
        env_n=None,
        env="n/a",
        instances_n=12,
        instances_basis="reported",
        phenotype="genome structure, TE and subtelomere dynamics",
        shape="assembly",
        seq_basis="isolate-WGS",
        why="Structural variation short reads miss. Relevant because industrial strains differ from S288C structurally, not only at SNPs.",
        accession="ENA/GenBank (accession unconfirmed)",
    ),
    Candidate(
        name="Tengolics 2024 (domestication metabolome)",
        citation="Tengolics R, Szappanos B, Mulleder M, et al., Ralser M, Papp B. PNAS 2024;121:e2313354121.",
        url="https://doi.org/10.1073/pnas.2313354121",
        klass="Natural variation",
        tier=3,
        genotypes_n=17,
        genotypes="17 S. cerevisiae populations",
        env_n=1,
        env="1",
        instances_n=17,
        instances_basis="reported",
        phenotype="19 amino acids + 78 other metabolites",
        shape="vector (97)",
        dim=97,
        seq_basis="isolate-WGS",
        why="Shows which metabolic traits already moved under domestication, which is a prior on which are tractable to engineer further. Small n.",
        accession="PNAS Dataset S11",
    ),
    Candidate(
        name="Eder 2020 (flux QTL)",
        citation="Eder M, Nidelet T, Sanchez I, Camarasa C, Legras JL, Dequin S. PMID 32034164.",
        url="https://pubmed.ncbi.nlm.nih.gov/32034164/",
        klass="Natural variation",
        tier=3,
        genotypes_n=100,
        genotypes="segregant panel (n unconfirmed)",
        env_n=1,
        env="wine fermentation",
        instances_n=100,
        instances_basis="estimate",
        phenotype="modeled intracellular flux",
        shape="vector",
        seq_basis="segregant-WGS",
        why="Flux as a QTL phenotype, with PDB1 and VID30 validated. The label is modeled rather than measured, which caps how far it can be trusted.",
        accession="Journal SI (venue and accession unconfirmed)",
    ),
    # -- Tier 4 -------------------------------------------------------------
    Candidate(
        name="Bennett 2001 (ionizing-radiation resistance)",
        citation="Bennett CB, et al. Nat Genet 2001;29:426-434.",
        url="https://doi.org/10.1038/ng778",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=4800,
        genotypes="~4,800 YKO",
        env_n=2,
        env="irradiated vs control",
        instances_n=9600,
        instances_basis="product",
        phenotype="radiation sensitivity",
        shape="ordinal",
        seq_basis="S288C-KO",
        why="Genome-stability tolerance. Only indirectly relevant, through strain robustness over long fed-batch runs where instability erodes titer.",
        accession="Nat Genet SI (likely PDF)",
    ),
    Candidate(
        name="Sugiyama (lactic-acid tolerance)",
        citation="Sugiyama M, Kaneko Y, et al. National Research Institute of Brewing technical report.",
        url="https://www.nisr.or.jp/wp-content/uploads/NISR06sugiyama.pdf",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=4800,
        genotypes="~4,800 YKO",
        env_n=2,
        env="4 percent lactic acid vs control",
        instances_n=9600,
        instances_basis="product",
        phenotype="acid tolerance",
        shape="ordinal",
        seq_basis="S288C-KO",
        why="Lactic acid is a major platform chemical, but this is a non-peer-reviewed technical report with PDF-only tables. Lowest provenance confidence in the table; treat as a hypothesis source.",
        accession="NRIB technical report PDF",
    ),
    # ---- Regulatory DNA: the highest-dimensional perturbation spaces in yeast ----
    Candidate(
        name="de Boer 2020 (100M random promoters)",
        citation="de Boer CG, Vaishnav ED, Sadeh R, Abeyta EL, Friedman N, Regev A. Nat Biotechnol 2020;38:56-65.",
        url="https://doi.org/10.1038/s41587-019-0315-8",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=100000000,
        genotypes=">100M random promoters",
        env_n=2,
        env="complex + defined",
        instances_n=100000000,
        instances_basis="reported",
        phenotype="expression driven (YFP sort-seq)",
        shape="scalar",
        seq_basis="S288C+reporter-locus",
        why="The largest sequence-to-phenotype dataset in yeast by orders of magnitude, and every genotype is an exactly known 80mer. The perturbation space is continuous rather than a gene list, which is what a model of regulatory grammar needs and no deletion collection can supply.",
        accession="GEO (de Boer 2020); DREAM 2022 release of 6,739,258 promoters",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Vaishnav 2022 (regulatory DNA fitness landscape)",
        citation="Vaishnav ED, de Boer CG, Molinet J, Yassour M, Fan L, Adiconis X, Thompson DA, Levin JZ, Cubillos FA, Regev A. Nature 2022;603:455-463.",
        url="https://doi.org/10.1038/s41586-022-04506-6",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=20000000,
        genotypes="~2e7 promoter variants",
        env_n=1,
        env="1",
        instances_n=20000000,
        instances_basis="estimate",
        phenotype="expression + inferred fitness",
        shape="scalar",
        seq_basis="S288C+reporter-locus",
        why="Extends de Boer from measurement to a fitness landscape, and tests native and natural-isolate promoter variants against the random library. The bridge between synthetic sequence space and the natural variation the isolate panels carry.",
        accession="GEO per the Nature data-availability statement (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Renganaath 2020 (natural promoter-variant MPRA)",
        citation="Renganaath K, Cheung R, Day L, Kosuri S, Kruglyak L, Albert FW. eLife 2020;9:e62669.",
        url="https://doi.org/10.7554/eLife.62669",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=5832,
        genotypes="5,832 natural variants, 2,503 promoters",
        env_n=1,
        env="1",
        instances_n=5832,
        instances_basis="reported",
        phenotype="allele-specific promoter activity",
        shape="scalar",
        seq_basis="S288C+reporter-locus",
        why="Native promoter variation measured allele by allele rather than sampled randomly, with 451 variants called causal. The row that joins the synthetic MPRA libraries to the natural-isolate panels: same readout, real alleles.",
        accession="eLife data availability; GEO (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Cuperus 2017 (random 5' UTR library)",
        citation="Cuperus JT, Groves B, Kuchina A, Rosenberg AB, Jojic N, Fields S, Seelig G. Genome Res 2017;27:2015-2024.",
        url="https://doi.org/10.1101/gr.224964.117",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=500000,
        genotypes="~500,000 random 5' UTRs",
        env_n=1,
        env="1",
        instances_n=500000,
        instances_basis="estimate",
        phenotype="protein output (translation)",
        shape="scalar",
        seq_basis="S288C+reporter-locus",
        why="The translational layer of the same idea at half a million exactly known sequences. Expression is set after transcription as well as at it, and nothing else here measures that axis.",
        accession="GEO (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Sharon 2012 (designed promoter library)",
        citation="Sharon E, Kalma Y, Sharp A, et al., Segal E. Nat Biotechnol 2012;30:521-530.",
        url="https://doi.org/10.1038/nbt.2205",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=2000,
        genotypes="~2,000 designed promoters",
        env_n=1,
        env="1",
        instances_n=2000,
        instances_basis="estimate",
        phenotype="expression (fluorescence)",
        shape="scalar",
        seq_basis="S288C+reporter-locus",
        why="Designed rather than random, so site number, affinity, spacing and orientation vary one at a time. Small, but it is the controlled counterpart the random libraries lack.",
        accession="Nat Biotechnol SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Duveau 2021 (TDH3 promoter allele series)",
        citation="Duveau F, Vande Zande P, Metzger BP, et al., Wittkopp PJ. eLife 2021;10:e67806.",
        url="https://doi.org/10.7554/eLife.67806",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=250,
        genotypes="~250 promoter alleles",
        env_n=4,
        env="carbon sources",
        instances_n=1000,
        instances_basis="estimate",
        phenotype="expression level, noise, fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="One native promoter dissected at its own locus, with expression and fitness on the same alleles across environments. The rare case tying a regulatory change to a fitness consequence rather than a reporter number.",
        accession="eLife data availability (unconfirmed)",
        confidence="recall",
    ),
    # ---- Deep mutational scanning: depth instead of one knockout per gene ----
    Candidate(
        name="Li 2016 (tRNA fitness landscape)",
        citation="Li C, Qian W, Maclean CJ, Zhang J. Science 2016;352:837-840.",
        url="https://doi.org/10.1126/science.aae0568",
        klass="Deep mutational scan",
        tier=1,
        genotypes_n=65000,
        genotypes="~65,000 tRNA variants",
        env_n=1,
        env="1",
        instances_n=65000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="A near-complete fitness landscape of one native yeast gene. Every genotype is a known sequence and the sampling is exhaustive rather than one allele per gene, which is the opposite of what a deletion collection provides.",
        accession="Science SI (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Puchta 2016 (snoRNA U3 landscape)",
        citation="Puchta O, Cseke B, Czaja H, Tollervey D, Sanguinetti G, Kudla G. Science 2016;352:840-844.",
        url="https://doi.org/10.1126/science.aaf0965",
        klass="Deep mutational scan",
        tier=2,
        genotypes_n=60000,
        genotypes="~60,000 U3 variants",
        env_n=1,
        env="1",
        instances_n=60000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The companion landscape on a non-coding RNA, published alongside Li 2016. Two exhaustive landscapes on different molecule classes is a stronger test of a sequence model than either alone.",
        accession="Science SI (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Domingo 2018 (tRNA double-mutant landscape)",
        citation="Domingo J, Diss G, Lehner B. Nature 2018;558:117-121.",
        url="https://doi.org/10.1038/s41586-018-0170-7",
        klass="Deep mutational scan",
        tier=1,
        genotypes_n=23000,
        genotypes="~23,000 single + double mutants",
        env_n=1,
        env="1",
        instances_n=23000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="Pairwise epistasis at nucleotide resolution inside one gene. The higher-order structure Costanzo and Kuzmin map between genes, measured within a gene, which is the resolution inverse design actually needs.",
        accession="Nature SI (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Mavor 2016 (ubiquitin DMS across conditions)",
        citation="Mavor D, Barlow K, Thompson S, et al., Bolon DNA, Fraser JS. eLife 2016;5:e15802.",
        url="https://doi.org/10.7554/eLife.15802",
        klass="Deep mutational scan",
        tier=2,
        genotypes_n=1400,
        genotypes="~1,400 ubiquitin variants",
        env_n=5,
        env="chemical stresses",
        instances_n=7000,
        instances_basis="product",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="A complete single-mutant scan of an essential yeast gene, repeated across stresses. Variant effects are conditional, and this is the cleanest demonstration of that in the table.",
        accession="eLife data availability (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Flynn 2020 (Hsp90 DMS across environments)",
        citation="Flynn JM, Rossouw A, Cote-Hammarlof P, et al., Bolon DNA. eLife 2020;9:e53810.",
        url="https://doi.org/10.7554/eLife.53810",
        klass="Deep mutational scan",
        tier=2,
        genotypes_n=1000,
        genotypes="~1,000 Hsp90 variants",
        env_n=6,
        env="stress conditions",
        instances_n=6000,
        instances_basis="product",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="A chaperone hub scanned under multiple stresses, so the environment-dependence of a variant effect is measured rather than assumed. Hsp90 buffers exactly the variation the natural-isolate rows carry.",
        accession="eLife data availability (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Melamed 2013 (PAB1 RRM domain scan)",
        citation="Melamed D, Young DL, Gamble CE, Miller CR, Fields S. RNA 2013;19:1537-1551.",
        url="https://doi.org/10.1261/rna.040709.113",
        klass="Deep mutational scan",
        tier=3,
        genotypes_n=40000,
        genotypes="~40,000 PAB1 variants",
        env_n=1,
        env="1",
        instances_n=40000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="Dense coverage of one RNA-binding domain including double mutants, on a native essential yeast gene.",
        accession="journal SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Kitzman 2015 (GAL4 DMS)",
        citation="Kitzman JO, Starita LM, Lo RS, Fields S, Shendure J. Nat Methods 2015;12:203-206.",
        url="https://doi.org/10.1038/nmeth.3223",
        klass="Deep mutational scan",
        tier=3,
        genotypes_n=1200,
        genotypes="~1,200 GAL4 variants",
        env_n=2,
        env="selective / permissive",
        instances_n=2400,
        instances_basis="product",
        phenotype="transcription-factor activity",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="A transcription factor scanned residue by residue, which pairs with the regulatory-DNA rows: one measures the site, the other the protein that reads it.",
        accession="Nat Methods SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Diss 2018 (Fos-Jun interaction double mutants)",
        citation="Diss G, Lehner B. eLife 2018;7:e32472.",
        url="https://doi.org/10.7554/eLife.32472",
        klass="Deep mutational scan",
        tier=2,
        genotypes_n=120000,
        genotypes="~120,000 double mutants",
        env_n=1,
        env="1",
        instances_n=120000,
        instances_basis="estimate",
        phenotype="protein-protein interaction strength",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="Exhaustive pairwise mutation of an interaction interface, read out in yeast. The genotype is a heterologous domain rather than a yeast gene, so it enters as an engineered chassis, but the epistasis structure is the deepest in the table.",
        accession="eLife data availability (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Faure 2022 (doubledeepPCA)",
        citation="Faure AJ, Domingo J, Schmiedel JM, Hidalgo-Carcedo C, Diss G, Lehner B. Nature 2022;604:175-183.",
        url="https://doi.org/10.1038/s41586-022-04586-4",
        klass="Deep mutational scan",
        tier=2,
        genotypes_n=500000,
        genotypes="~5e5 double mutants, 2 domains",
        env_n=1,
        env="1",
        instances_n=500000,
        instances_basis="estimate",
        phenotype="binding and abundance",
        shape="scalar pair",
        seq_basis="engineered-chassis",
        why="Separates a variant's effect on folding from its effect on binding by measuring both in the same library. That decomposition is what turns a fitness landscape into a mechanistic one.",
        accession="Nature data availability (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    # ---- Sequenced recombinant panels at scale ----
    Candidate(
        name="Nguyen Ba 2022 (barcoded bulk QTL, 100k segregants)",
        citation="Nguyen Ba AN, Lawrence KR, Rego-Costa A, Gopalakrishnan S, Temko D, Michor F, Desai MM. eLife 2022;11:e73983.",
        url="https://doi.org/10.7554/eLife.73983",
        klass="Natural variation",
        tier=1,
        genotypes_n=100000,
        genotypes="~100,000 barcoded segregants",
        env_n=8,
        env="8 conditions",
        instances_n=800000,
        instances_basis="product",
        phenotype="fitness (Bar-seq)",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="The largest genotyped recombinant panel in yeast, each segregant barcoded and low-coverage sequenced against known parents. Two orders of magnitude more genotypes than Bloom 2013 on the same kind of cross, which is what makes small-effect and epistatic loci detectable.",
        accession="eLife data availability; SRA (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Bloom 2013 (1,008 BYxRM segregants)",
        citation="Bloom JS, Ehrenreich IM, Loo WT, Lite TL, Kruglyak L. Nature 2013;494:234-237.",
        url="https://doi.org/10.1038/nature11867",
        klass="Natural variation",
        tier=2,
        genotypes_n=1008,
        genotypes="1,008 segregants, 11,623 markers",
        env_n=46,
        env="46 traits",
        instances_n=46368,
        instances_basis="product",
        phenotype="quantitative growth traits",
        shape="scalar per trait",
        seq_basis="segregant-WGS",
        why="The reference two-parent panel the field calibrates against, and the direct predecessor of Bloom 2019 and Nguyen Ba 2022. Ingesting all three gives one cross at three panel sizes, which is a clean test of how much genotype coverage a model actually needs.",
        accession="Nature SI",
        confidence="recall",
    ),
    Candidate(
        name="Bloom 2025 (global epistasis in a yeast cross)",
        citation="Bloom JS, et al., Kruglyak L. 2025. PMID 40679398.",
        url="https://pubmed.ncbi.nlm.nih.gov/40679398/",
        klass="Natural variation",
        tier=2,
        genotypes_n=100000,
        genotypes="large sequenced segregant panel",
        env_n=10,
        env="conditions",
        instances_n=1000000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="Reports that many natural variants have effects that scale with background fitness, which is a specific, testable claim about how genotype effects compose. Directly relevant to whether a model trained on one background transfers.",
        accession="journal SI (unconfirmed; recent)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Albert 2018 (eQTL in 1,012 segregants)",
        citation="Albert FW, Bloom JS, Siegel J, Day L, Kruglyak L. eLife 2018;7:e35471.",
        url="https://doi.org/10.7554/eLife.35471",
        klass="Natural variation",
        tier=1,
        genotypes_n=1012,
        genotypes="1,012 sequenced segregants",
        env_n=1,
        env="1",
        instances_n=1012,
        instances_basis="reported",
        phenotype="mRNA abundance (RNA-seq)",
        shape="vector (~5,700)",
        dim=5700,
        seq_basis="segregant-WGS",
        why="Transcriptomes on a thousand sequenced recombinants, so genotype maps to a genome-wide expression vector rather than one trait. The single best existing analog to a Perturb-seq readout on a high-dimensional genotype axis.",
        accession="GEO / eLife data availability (unconfirmed)",
        perturbseq="both",
        confidence="recall",
    ),
    Candidate(
        name="Boocock 2025 (single-cell eQTL mapping)",
        citation="Boocock J, Alexander N, Alamo Tapia L, Walter-McNeill L, Patel SP, Munugala C, Bloom JS, Kruglyak L. eLife 2025;13:e95566.",
        url="https://doi.org/10.7554/eLife.95566",
        klass="Expression / single cell",
        tier=1,
        genotypes_n=393,
        genotypes="393 segregants, 27,744 cells",
        env_n=2,
        env="glucose / galactose",
        instances_n=55488,
        instances_basis="product",
        phenotype="single-cell transcriptome + eQTL",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="segregant-WGS",
        why="27,744 single cells drawn from 393 sequenced recombinants in two carbon sources, a median of 17 cells per segregant. High-dimensional on both axes at once, which almost nothing in yeast is, and the closest existing template for the proposed design.",
        accession="mirrored as boocockSinglecellEQTLMapping2025; GEO (unconfirmed)",
        perturbseq="both",
        confidence="sourced",
    ),
    Candidate(
        name="Sardi 2018 (natural variation in hydrolysate tolerance)",
        citation="Sardi M, Paithane V, Place M, Robinson DE, Hose J, Wohlbach DJ, Gasch AP. PLoS Genet 2018;14:e1007217.",
        url="https://doi.org/10.1371/journal.pgen.1007217",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=500,
        genotypes="sequenced isolate / segregant panel",
        env_n=4,
        env="hydrolysate toxins",
        instances_n=2000,
        instances_basis="estimate",
        phenotype="tolerance",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Natural variation in exactly the bioprocess phenotype the deletion screens cover by engineering. Pairs with Vanacloig-Pedros and Pereira to ask whether tolerance loci found by deletion recur as natural alleles.",
        accession="PLoS Genetics SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="De Chiara 2022 (domestication and genome instability)",
        citation="De Chiara M, Barre BP, Persson K, et al., Liti G. Nat Ecol Evol 2022;6:761-773.",
        url="https://doi.org/10.1038/s41559-022-01671-9",
        klass="Natural variation",
        tier=3,
        genotypes_n=1011,
        genotypes="1,011-isolate panel, reanalyzed",
        env_n=36,
        env="conditions",
        instances_n=36396,
        instances_basis="estimate",
        phenotype="growth, aneuploidy, instability",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Ties genome instability and aneuploidy in the isolate panel to phenotype. Industrial strains acquire aneuploidy under selection, so this is the natural-variation view of a failure mode long fermentations create.",
        accession="Nat Ecol Evol SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Teyssonniere 2024 (species-wide trait survey)",
        citation="Teyssonniere EM, Shichino Y, Mito M, Friedrich A, Iwasaki S, Schacherer J. PLoS Genet 2024;20:e1011119.",
        url="https://doi.org/10.1371/journal.pgen.1011119",
        klass="Natural variation",
        tier=2,
        genotypes_n=1011,
        genotypes="1,011 isolates",
        env_n=200,
        env="~200 traits",
        instances_n=202200,
        instances_basis="estimate",
        phenotype="growth traits, expressivity and complexity",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Extends the phenotype axis of the sequenced panel far past Peter 2018's 36 conditions, and classifies traits by how monogenic or polygenic they are. That classification is a prior on which phenotypes a model can be expected to predict at all.",
        accession="PLoS Genetics SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Kinsler 2020 (barcoded pleiotropy across environments)",
        citation="Kinsler G, Geiler-Samerotte K, Petrov DA. eLife 2020;9:e61271.",
        url="https://doi.org/10.7554/eLife.61271",
        klass="Natural variation",
        tier=3,
        genotypes_n=300,
        genotypes="~300 sequenced adaptive clones",
        env_n=45,
        env="45 environments",
        instances_n=13500,
        instances_basis="product",
        phenotype="fitness",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Sequenced adaptive clones assayed across dozens of environments, so one genotype yields a fitness vector rather than a point. Included because the clones were sequenced; the parent barcoding study was not, and is excluded.",
        accession="eLife data availability (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Venkataram 2016 (barcoded adaptive lineages, sequenced)",
        citation="Venkataram S, Dunn B, Li Y, et al., Petrov DA, Kryazhimskiy S, Sherlock G. Cell 2016;166:1585-1596.",
        url="https://doi.org/10.1016/j.cell.2016.08.002",
        klass="Natural variation",
        tier=3,
        genotypes_n=400,
        genotypes="~400 sequenced adaptive clones",
        env_n=1,
        env="glucose-limited",
        instances_n=400,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Only the sequenced subset qualifies. The half-million barcoded lineages behind it do not, because their mutations were never determined, and that boundary is the sequence gate doing its job.",
        accession="Cell SI; SRA (unconfirmed)",
        confidence="recall",
    ),
    # ---- Acetic acid: the production and tolerance pairing ----
    Candidate(
        name="Mira 2010 (acetic acid tolerance, full collection)",
        citation="Mira NP, Palma M, Guerreiro JF, Sa-Correia I. Microb Cell Fact 2010;9:79.",
        url="https://doi.org/10.1186/1475-2859-9-79",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=4800,
        genotypes="EUROSCARF collection",
        env_n=3,
        env="70-110 mM acetic acid, pH 4.5",
        instances_n=14400,
        instances_basis="product",
        phenotype="acetic-acid tolerance",
        shape="scalar",
        seq_basis="S288C-KO",
        why="About 650 tolerance determinants on the full deletion collection at controlled pH. The loss-of-function counterpart to the already-built Mormino biosensor screen and to Mukherjee's essential-gene CRISPRi, so the same phenotype is covered by deletion, knockdown and biosensor.",
        accession="Microb Cell Fact SI (PMC2972246)",
        confidence="recall",
    ),
    Candidate(
        name="Sousa 2013 (acetate resistance overexpression screen)",
        citation="Sousa M, Duarte AM, Fernandes TR, et al. 2013. PMID 23262128.",
        url="https://pubmed.ncbi.nlm.nih.gov/23262128/",
        klass="Tolerance / robustness",
        tier=3,
        genotypes_n=5000,
        genotypes="genome-wide overexpression library",
        env_n=2,
        env="sodium acetate vs control",
        instances_n=10000,
        instances_basis="product",
        phenotype="acetate resistance",
        shape="scalar",
        seq_basis="S288C+ORF-plasmid",
        why="Gain-of-function on the same stress the deletion and CRISPRi rows cover by loss. Over and under the same condition is what makes a dosage-aware model trainable rather than sign-blind.",
        accession="journal SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Sousa 2013 (acetic-acid programmed cell death screen)",
        citation="Sousa M, Duarte AM, Fernandes TR, et al. 2013. PMID 24286259.",
        url="https://pubmed.ncbi.nlm.nih.gov/24286259/",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=4800,
        genotypes="YKO haploid collection",
        env_n=2,
        env="acetic acid vs control",
        instances_n=9600,
        instances_basis="product",
        phenotype="death / survival (regulated cell death)",
        shape="scalar",
        seq_basis="S288C-KO",
        why="Survival rather than growth rate under the same stressor. A different phenotype from fitness, and the one that matters when a fed-batch culture crashes rather than slows.",
        accession="journal SI (unconfirmed)",
        confidence="recall",
    ),
    # ---- Chromatin, transcription factors and the regulatory backbone ----
    Candidate(
        name="Rossi 2021 (ChIP-exo protein architecture)",
        citation="Rossi MJ, Kuntala PK, Lai WKM, et al., Pugh BF. Nature 2021;592:309-314.",
        url="https://doi.org/10.1038/s41586-021-03314-8",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=400,
        genotypes="~400 tagged proteins",
        env_n=1,
        env="1",
        instances_n=400,
        instances_basis="reported",
        phenotype="genome-wide binding location, near-bp resolution",
        shape="genome-wide track",
        dim=6000,
        seq_basis="S288C+tag",
        why="Where roughly 400 regulatory proteins sit on the genome at near-base-pair resolution. This is the wiring prior that makes a regulatory-DNA model mechanistic instead of correlational, and it pairs directly with the MPRA rows.",
        accession="yeastepigenome.org; github.com/CEGRcode/2021-Rossi_Nature",
        confidence="recall",
    ),
    Candidate(
        name="Hu 2007 (TF deletion expression compendium)",
        citation="Hu Z, Killion PJ, Iyer VR. Nat Genet 2007;39:683-687.",
        url="https://doi.org/10.1038/ng2012",
        klass="Expression / single cell",
        tier=2,
        genotypes_n=269,
        genotypes="269 TF deletion strains",
        env_n=1,
        env="1",
        instances_n=269,
        instances_basis="reported",
        phenotype="mRNA abundance",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="S288C-KO",
        why="Transcriptomes for the regulators specifically, predating and complementing the already-built Kemmeren compendium. Deleting a transcription factor is the perturbation whose transcriptome response is most interpretable.",
        accession="GEO (unconfirmed)",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Brogaard 2012 (chemical nucleosome map)",
        citation="Brogaard K, Xi L, Wang JP, Widom J. Nature 2012;486:496-501.",
        url="https://doi.org/10.1038/nature11142",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=1,
        env="1",
        instances_n=1,
        instances_basis="reported",
        phenotype="nucleosome center positions, bp resolution",
        shape="genome-wide track",
        dim=67000,
        seq_basis="reference-only",
        why="Nucleosome positioning at base-pair resolution genome-wide. Promoter accessibility is what the MPRA rows are implicitly measuring, and this is the structural map underneath it.",
        accession="GEO (unconfirmed)",
        confidence="recall",
    ),
    # ---- Proteome, translation and turnover ----
    Candidate(
        name="Ho 2018 (unified absolute protein abundance)",
        citation="Ho B, Baryshnikova A, Brown GW. Cell Syst 2018;6:192-205.",
        url="https://doi.org/10.1016/j.cels.2017.12.004",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=1,
        genotypes="wild type",
        env_n=1,
        env="1",
        instances_n=5858,
        instances_basis="reported",
        phenotype="absolute abundance, molecules/cell",
        shape="scalar per protein",
        seq_basis="reference-only",
        why="Reconciles 21 independent abundance datasets into one unified estimate per protein. A better absolute anchor than any single study, including Ghaemmaghami, and the reference frame every relative proteomics row needs.",
        accession="Cell Systems SI",
        confidence="recall",
    ),
    Candidate(
        name="Lahtvee 2017 (absolute proteome across conditions)",
        citation="Lahtvee PJ, Sanchez BJ, Smialowska A, et al., Nielsen J. Cell Syst 2017;4:495-504.",
        url="https://doi.org/10.1016/j.cels.2017.03.003",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=9,
        env="9 growth conditions",
        instances_n=9,
        instances_basis="reported",
        phenotype="absolute protein abundance + turnover",
        shape="vector",
        dim=2000,
        seq_basis="reference-only",
        why="Absolute enzyme copy numbers across conditions, which is the quantity enzyme-constrained metabolic models consume. Ties this table's proteomics to the Yeast9 modeling scaffold.",
        accession="Cell Systems SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Christiano 2014 (protein turnover)",
        citation="Christiano R, Nagaraj N, Frohlich F, Walther TC. Cell Rep 2014;9:1959-1965.",
        url="https://doi.org/10.1016/j.celrep.2014.10.065",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=1,
        env="1",
        instances_n=3900,
        instances_basis="estimate",
        phenotype="protein half-life",
        shape="scalar per protein",
        seq_basis="reference-only",
        why="Half-lives genome-wide. Abundance is production over degradation, and every other proteomics row here measures only the product, so turnover is the missing half.",
        accession="Cell Reports SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Breker 2013 (GFP library under stress)",
        citation="Breker M, Gymrek M, Schuldiner M. J Cell Biol 2013;200:839-850.",
        url="https://doi.org/10.1083/jcb.201301120",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=4159,
        genotypes="~4,159 GFP-tagged strains",
        env_n=3,
        env="stress conditions",
        instances_n=12477,
        instances_basis="product",
        phenotype="abundance + localization, per cell",
        shape="per-cell distribution",
        dim=17,
        seq_basis="S288C+tag",
        why="The GFP collection re-imaged under stress rather than at baseline, so relocalization is measured as a response. Feeds LoQAtE, and adds the condition axis Huh 2003 lacks.",
        accession="LoQAtE (weizmann.ac.il/molgen/loqate)",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Weinberg 2016 (ribosome profiling, improved)",
        citation="Weinberg DE, Shah P, Eichhorn SW, Hussmann JA, Plotkin JB, Bartel DP. Cell Rep 2016;14:1787-1799.",
        url="https://doi.org/10.1016/j.celrep.2016.01.043",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=1,
        env="1",
        instances_n=1,
        instances_basis="reported",
        phenotype="ribosome occupancy, translation efficiency",
        shape="vector",
        dim=6000,
        seq_basis="reference-only",
        why="The corrected reference translation-efficiency dataset, after earlier profiling runs were shown to carry a drug-induced artifact. Translation efficiency is the step between the transcriptome and proteome layers this table otherwise joins by assumption.",
        accession="GEO (unconfirmed)",
        confidence="recall",
    ),
    # ---- Combinatorial genome engineering: many edits per cell ----
    Candidate(
        name="Zhang 2022 (GCE-SCRaMbLE recombination outcomes)",
        citation="Zhang W, Lazar-Stefanita L, Yamashita H, et al., Boeke JD. Nat Commun 2022;13:5836.",
        url="https://doi.org/10.1038/s41467-022-33606-0",
        klass="Combinatorial genome",
        tier=2,
        genotypes_n=200,
        genotypes="~200 sequenced SCRaMbLE genomes",
        env_n=1,
        env="1",
        instances_n=200,
        instances_basis="estimate",
        phenotype="rearrangement structure + fitness",
        shape="structural variant set",
        seq_basis="isolate-WGS",
        why="Each cell carries tens of simultaneous structural edits, sequenced afterwards. The deepest per-cell perturbation in the table: a genotype here is a rearranged chromosome, not an allele, which is what inverse strain design eventually has to predict on.",
        accession="mirrored as zhangSystematicDissectionKey2022; SRA (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Shen 2016 (SCRaMbLE genome minimization)",
        citation="Shen Y, Stracquadanio G, Wang Y, et al., Boeke JD, Bader JS. Genome Res 2016;26:36-49.",
        url="https://doi.org/10.1101/gr.193433.115",
        klass="Combinatorial genome",
        tier=3,
        genotypes_n=100,
        genotypes="~100 SCRaMbLEd genomes",
        env_n=2,
        env="selective / permissive",
        instances_n=200,
        instances_basis="estimate",
        phenotype="viability and fitness",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Combinatorial deletion and inversion across a synthetic chromosome, with the surviving genotypes sequenced. Maps which gene combinations are jointly dispensable, which single deletions cannot.",
        accession="Genome Research SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Blount 2018 (SCRaMbLE for phenotype improvement)",
        citation="Blount BA, Gowers GF, Ho JCH, Ledesma-Amaro R, Jovicevic D, McKiernan RM, Xie ZX, Li BZ, Yuan YJ, Ellis T. Nat Commun 2018;9:1932.",
        url="https://doi.org/10.1038/s41467-018-03143-w",
        klass="Combinatorial genome",
        tier=3,
        genotypes_n=100,
        genotypes="~100 rearranged genomes",
        env_n=2,
        env="selection conditions",
        instances_n=200,
        instances_basis="estimate",
        phenotype="growth and product phenotype",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="SCRaMbLE driven toward an industrial phenotype rather than toward minimization, so the genotype-to-product link is measured on multiply rearranged genomes.",
        accession="Nat Commun SI (unconfirmed)",
        confidence="recall",
    ),
    # ---- CRISPR screens beyond the ones already listed ----
    Candidate(
        name="Sadhu 2018 (CRISPR-directed mitotic recombination)",
        citation="Sadhu MJ, Bloom JS, Day L, Siegel JJ, Kosuri S, Kruglyak L. eLife 2018;7:e41090.",
        url="https://doi.org/10.7554/eLife.41090",
        klass="CRISPR library screen",
        tier=3,
        genotypes_n=1500,
        genotypes="~1,500 recombinant genotypes",
        env_n=1,
        env="1",
        instances_n=1500,
        instances_basis="estimate",
        phenotype="fitness / trait",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="Uses targeted recombination to build a genotype series with a single locus swapped against a fixed background, isolating causal effects the way a QTL panel cannot.",
        accession="eLife data availability (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Guo 2018 (CRISPR-Cas9 tiling of essential genes)",
        citation="Guo X, Chavez A, Tung A, et al., Church GM, Shalem O. Nat Biotechnol 2018;36:540-546.",
        url="https://doi.org/10.1038/nbt.4147",
        klass="CRISPR library screen",
        tier=3,
        genotypes_n=30000,
        genotypes="~3e4 tiling guides",
        env_n=1,
        env="1",
        instances_n=30000,
        instances_basis="estimate",
        phenotype="per-guide fitness",
        shape="scalar",
        seq_basis="S288C+guide",
        why="Tiles guides across genes rather than sampling a few per gene, so the readout resolves within-gene functional regions. A finer input axis than a gene-level library.",
        accession="Nat Biotechnol SI (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    Candidate(
        name="Jaffe 2019 (multiplexed CRISPR interference epistasis)",
        citation="Jaffe M, Sherlock G, Levy SF. G3 2019;9:2843-2852.",
        url="https://doi.org/10.1534/g3.119.400323",
        klass="CRISPR library screen",
        tier=3,
        genotypes_n=10000,
        genotypes="~1e4 guide pairs",
        env_n=1,
        env="1",
        instances_n=10000,
        instances_basis="estimate",
        phenotype="fitness / genetic interaction",
        shape="scalar",
        seq_basis="S288C+guide",
        why="Two guides per cell, so the perturbation space is combinatorial rather than one edit at a time. This is the shape a Perturb-seq input axis should have, measured here against fitness only.",
        accession="G3 open-access SI (unconfirmed)",
        perturbseq="input",
        confidence="recall",
    ),
    # ---- Metabolic engineering, product and precursor ----
    Candidate(
        name="Jakociunas 2021 (degron-tuned terpene flux)",
        citation="Jakociunas T, Klitgaard AK, Kristensen M, Jensen MK, Keasling JD. 2021.",
        url="https://doi.org/10.1002/bit.27735",
        klass="Metabolite / precursor",
        tier=3,
        genotypes_n=30,
        genotypes="~30 degron strains",
        env_n=3,
        env="induction levels",
        instances_n=90,
        instances_basis="estimate",
        phenotype="farnesol / geraniol / nerolidol titer",
        shape="vector (3)",
        dim=3,
        seq_basis="engineered-chassis",
        why="Erg20p is FPP synthase, so degron-tuning it reports FPP pathway throughput directly. Small, but it is the only row measuring the mevalonate branch, which the precursor accounting lists as unmeasured.",
        accession="journal SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Lian 2016 (combinatorial xylose pathway optimization)",
        citation="Lian J, Chao R, Zhao H. Metab Eng 2016;33:139-148.",
        url="https://doi.org/10.1016/j.ymben.2015.11.001",
        klass="Metabolite / precursor",
        tier=4,
        genotypes_n=100,
        genotypes="combinatorial pathway variants",
        env_n=1,
        env="xylose",
        instances_n=100,
        instances_basis="estimate",
        phenotype="ethanol titer and rate on xylose",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="Combinatorial promoter and copy-number variants of one pathway with titers measured per combination. Small, and it is the record type inverse design produces.",
        accession="Metab Eng SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Deutschbauer 2005 (haploinsufficiency, genome-wide)",
        citation="Deutschbauer AM, Jaramillo DF, Proctor M, Kumm J, Hillenmeyer ME, Davis RW, Nislow C, Giaever G. Genetics 2005;169:1915-1925.",
        url="https://doi.org/10.1534/genetics.104.036871",
        klass="Tolerance / robustness",
        tier=3,
        genotypes_n=5900,
        genotypes="heterozygous diploid collection",
        env_n=2,
        env="rich / minimal",
        instances_n=11800,
        instances_basis="product",
        phenotype="haploinsufficiency fitness",
        shape="scalar",
        seq_basis="S288C-KO/het",
        why="Gene-dosage sensitivity genome-wide, from halving rather than removing. Dosage-sensitive genes are exactly the ones whose overexpression in a pathway will misbehave.",
        accession="Genetics SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Giaever 2004 (chemogenomic drug-target profiling)",
        citation="Giaever G, Flaherty P, Kumm J, et al., Davis RW. PNAS 2004;101:793-798.",
        url="https://doi.org/10.1073/pnas.0307490100",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=5900,
        genotypes="het + hom collections",
        env_n=10,
        env="~10 compounds",
        instances_n=59000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C-KO",
        why="An early HIP profiling set from the group that built the collection. Superseded in scale by Lee and Hoepfner, retained as a low-cost cross-validation panel.",
        accession="PNAS SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Hillenmeyer 2010 (systematic fitness reanalysis)",
        citation="Hillenmeyer ME, Ericson E, Davis RW, Nislow C, Koller D, Giaever G. Genome Biol 2010;11:R30.",
        url="https://doi.org/10.1186/gb-2010-11-3-r30",
        klass="Tolerance / robustness",
        tier=3,
        genotypes_n=5900,
        genotypes="het + hom collections",
        env_n=400,
        env="~400 conditions",
        instances_n=2360000,
        instances_basis="estimate",
        phenotype="fitness-defect score",
        shape="scalar",
        seq_basis="S288C-KO",
        why="The reprocessed and normalized form of the FitDb compendium, with per-condition significance. Ingesting the reanalysis rather than the raw release avoids redoing the normalization the authors already validated.",
        accession="Genome Biology SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Gibney 2013 (osmotic and glucose stress fitness)",
        citation="Gibney PA, Lu C, Caudy AA, Hess DC, Botstein D. PNAS 2013;110:E4393-E4402.",
        url="https://doi.org/10.1073/pnas.1318100110",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=4800,
        genotypes="YKO collection",
        env_n=4,
        env="osmotic / glucose",
        instances_n=19200,
        instances_basis="product",
        phenotype="fitness",
        shape="scalar",
        seq_basis="S288C-KO",
        why="High-gravity fermentation is an osmotic problem, and this covers the osmotic axis on the deletion collection with a growth readout rather than a transcriptome.",
        accession="PNAS SI (unconfirmed)",
        confidence="recall",
    ),
    # ---- Single cell and expression, the Perturb-seq-adjacent tranche ----
    Candidate(
        name="Jackson 2020 (TF-deletion single-cell atlas)",
        citation="Jackson CA, Castro DM, Saldi GA, Bonneau R, Gresham D. eLife 2020;9:e51254.",
        url="https://doi.org/10.7554/eLife.51254",
        klass="Expression / single cell",
        tier=1,
        genotypes_n=12,
        genotypes="12 genotypes (wild type + 11 TF deletions), 72 barcoded strains",
        env_n=11,
        env="11 growth conditions",
        instances_n=132,
        instances_basis="product",
        phenotype="single-cell transcriptome",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="S288C-KO",
        why="The closest thing to a yeast Perturb-seq that exists. A transcribed barcode in the 3' UTR of the marker cassette labels every cell with its genotype, so 72 strains, 12 genotypes at 6 independently constructed replicates, are pooled and read out per cell; every strain appears in every one of the 11 conditions. The perturbations are homozygous diploid whole-ORF deletions in a prototrophic FY4/FY5 background rather than the Giaever collection, which is what permits the minimal and nitrogen-limited media. The DNA-only barcodes of the Giaever collection are the stated reason it could not be used, and that is the design constraint any campaign here inherits.",
        accession="GEO GSE125162; also eLife Source code 2 (103118_SS_Data.tsv.gz). Confirmed from the mirrored paper. The released matrix carries 38,225 cells; the abstract says 38,285 and the discussion 38,255, so the deposit governs. The authoritative 72-strain genotype list is Supplementary file 1 Table S2, an Excel file not held in the mirror.",
        perturbseq="output",
        requested=True,
    ),
    Candidate(
        name="Jackson 2023 (wild-type scRNA-seq time course for RNA kinetics)",
        citation="Jackson CA, Beheler-Amass M, Tjarnberg A, Suresh I, Hickey AS, Bonneau R, Gresham D. bioRxiv 2023. 10.1101/2023.09.21.558277.",
        url="https://doi.org/10.1101/2023.09.21.558277",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=1,
        env="one time course, sampled continuously",
        instances_n=1,
        instances_basis="reported",
        phenotype="single-cell transcriptome (173,361 cells)",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="The second scYeast pre-training set: 173,361 cells from one wild-type strain sampled continuously through a time course without metabolic labeling, so the strain-by-condition count is one and the value is the per-cell distribution of transcriptional states, not a genotype axis. A reference backbone for any single-cell expression head, and the companion of Jackson 2020 on the same platform.",
        accession="GEO GSE242556 (from the scYeast Methods, Sec. 4.1.3)",
        perturbseq="output",
    ),
    Candidate(
        name="Airoldi 2016 (nitrogen-limited steady-state and dynamic transcriptome)",
        citation="Airoldi EM, Miller D, Athanasiadou R, Brandt N, Abdul-Rahman F, Neymotin B, Hashimoto T, Bahmani T, Gresham D. Mol Biol Cell 2016;27:1383-1396.",
        url="https://doi.org/10.1091/mbc.E14-05-1013",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=1,
        genotypes="wild type, prototrophic",
        env_n=20,
        env="nitrogen sources x steady-state growth rates, plus an upshift time course",
        instances_n=20,
        instances_basis="estimate",
        phenotype="bulk transcriptome",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="Chemostat transcriptomes under controlled nitrogen limitation from the same laboratory, and on the same media axis, as Jackson 2020's NLIM-GLN, NLIM-PRO, NLIM-NH4 and NLIM-UREA conditions. That shared axis is what makes the wild-type baseline of the single-cell atlas separable from the deletion effect, and it is the nitrogen counterpart of Brauer 2008's carbon-limited growth-rate series. Counts are from the citation rather than the released series and need confirming.",
        accession="GEO, series not confirmed this pass",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Hackett 2020 (IDEA inducible-TF transcriptome time series)",
        citation="Hackett SR, Baltz EA, Coram M, Wranik BJ, Kim G, Baker A, Fan M, Hendrickson DG, Berndl M, McIsaac RS. Mol Syst Biol 2020;16:e9174.",
        url="https://doi.org/10.15252/msb.20199174",
        klass="Expression / single cell",
        tier=2,
        genotypes_n=200,
        genotypes="over 200 inducible-TF strains",
        env_n=8,
        env="estradiol induction, about eight time points",
        instances_n=1600,
        instances_basis="product",
        phenotype="bulk transcriptome per time point",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="S288C+designed-edit",
        why="Each TF is individually induced from a designed synthetic cassette and the whole transcriptome is followed over time, which is the only yeast dataset where a single-gene perturbation is crossed with a dense time axis. scYeast used it as the target for its time-series perturbation-response task (Fan 2027, Sec. 4.1.7). The instance count treats each strain-by-time-point as a record; per strain it is about 200.",
        accession="Published with the paper (Mol Syst Biol SI); accession unconfirmed from the mirror",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Wang 2022 (single-cell transcriptomes across replicative aging)",
        citation="Wang J, Sang Y, Jin S, Wang X, Azad GK, McCormick MA, Kennedy BK, Li Q, Wang J, Zhang X, et al. Aging Cell 2022;21:e13712.",
        url="https://doi.org/10.1111/acel.13712",
        klass="Expression / single cell",
        tier=4,
        genotypes_n=1,
        genotypes="wild type",
        env_n=3,
        env="2 h, 16 h, 36 h of aging",
        instances_n=3,
        instances_basis="reported",
        phenotype="single-cell transcriptome (125 cells)",
        shape="vector (~2,200 detected)",
        dim=2200,
        seq_basis="reference-only",
        why="Three age groups of one strain, 37, 43 and 45 cells each, about 2,200 genes detected per cell. Small, but it is the only yeast single-cell set with age as the condition axis, and scYeast used it to infer age-specific regulatory relationships (Fan 2027, Sec. 4.1.4).",
        accession="GEO GSE210032 (from the scYeast Methods)",
        confidence="recall",
    ),
    Candidate(
        name="Su 2023 (single-cell transcriptomes under four stresses)",
        citation="Su Y, Xu C, Shea J, DeStephanis D, Su Z. BMC Genomics 2023;24:88.",
        url="https://doi.org/10.1186/s12864-023-09184-w",
        klass="Expression / single cell",
        tier=4,
        genotypes_n=1,
        genotypes="wild type",
        env_n=4,
        env="isosmotic, hypoosmotic, glucose starvation, amino acid starvation",
        instances_n=4,
        instances_basis="reported",
        phenotype="single-cell transcriptome (117 cells, full length)",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="Full-length single-cell RNA-seq of 117 cells across four stress conditions in one strain; scYeast's stress-state classification benchmark (Fan 2027, Sec. 4.1.6). Tiny, but it is a full-length protocol rather than 3' tag counting, which matters for isoform-level heads.",
        accession="GEO GSE201386 (from the scYeast Methods)",
        confidence="recall",
    ),
    Candidate(
        name="Jariani 2020 (yeast scRNA-seq through lag phase)",
        citation="Jariani A, Vermeersch L, Cerulus B, Perez-Samper G, Voordeckers K, Van Brussel T, Thienpont B, Lambrechts D, Verstrepen KJ. eLife 2020;9:e55320.",
        url="https://doi.org/10.7554/eLife.55320",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=1,
        genotypes="isogenic wild type",
        env_n=6,
        env="glucose to maltose shift, time course through lag phase",
        instances_n=6,
        instances_basis="estimate",
        phenotype="single-cell transcriptome",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="A yeast-adapted 10x protocol that resolves the transcriptional states of individual cells during a carbon-source shift, so the readout is the distribution across a population that is not synchronized rather than its mean. One genotype, so the value is the protocol and the per-cell variance structure a campaign has to budget against.",
        accession="GEO GSE144820 and GSE116246; from the collection sweep, not fetched",
        perturbseq="output",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Urbonaite 2021 (yeastDrop-Seq under drug treatment)",
        citation="Urbonaite G, Lee JTH, Liu P, Parras GG, Hemberg M, Acar M. Commun Biol 2021;4:1372.",
        url="https://doi.org/10.1038/s42003-021-02895-4",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=1,
        genotypes="isogenic wild type",
        env_n=4,
        env="mycophenolic acid, guanine, both, untreated",
        instances_n=4,
        instances_basis="reported",
        phenotype="single-cell transcriptome",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="A yeast-optimized Drop-seq with a small chemical condition axis, including a two-compound arm. One genotype, so it enters as a platform reference: the pair of single-drug arms against their combination is a per-cell readout of a two-factor environment, which is the environmental analog of the combinatorial genetic perturbation a Perturb-seq needs.",
        accession="GEO GSE165686; Zenodo 10.5281/zenodo.4767298 and 10.5281/zenodo.4762526; from the collection sweep, not fetched",
        perturbseq="output",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Handfield 2013 (image-based localization change)",
        citation="Handfield LF, Chong YT, Simmons J, Andrews BJ, Moses AM. PLoS Comput Biol 2013;9:e1003085.",
        url="https://doi.org/10.1371/journal.pcbi.1003085",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=4000,
        genotypes="GFP collection",
        env_n=2,
        env="baseline / perturbed",
        instances_n=8000,
        instances_basis="estimate",
        phenotype="localization change score",
        shape="per-cell distribution",
        dim=17,
        seq_basis="S288C+tag",
        why="An unsupervised treatment of the same imaging collection, which matters because the supervised class labels in CYCLoPs constrain what can be found.",
        accession="PLoS Comput Biol SI (unconfirmed)",
        confidence="recall",
    ),
    # ---- Genetic interactions and network structure ----
    Candidate(
        name="Hénault 2023 (hybrid and introgression panel)",
        citation="Henault M, Marsit S, Charron G, Landry CR. Nat Commun / Landry lab hybrid panels.",
        url="https://doi.org/10.1038/s41467-024-46155-5",
        klass="Natural variation",
        tier=4,
        genotypes_n=500,
        genotypes="hybrids and introgression lines",
        env_n=10,
        env="conditions",
        instances_n=5000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Hybrid genomes rather than recombinants of one cross, so the genotype axis includes whole-subgenome combinations. Confirm the exact release before ingestion.",
        accession="journal SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Paulo 2016 (TMT proteome across carbon sources)",
        citation="Paulo JA, O'Connell JD, Gaun A, Gygi SP. Mol Biol Cell 2016;27:2823-2838.",
        url="https://doi.org/10.1091/mbc.E16-03-0184",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=1,
        genotypes="wild type",
        env_n=9,
        env="carbon sources",
        instances_n=9,
        instances_basis="reported",
        phenotype="protein abundance",
        shape="vector",
        dim=2500,
        seq_basis="reference-only",
        why="Proteome remodeling across carbon sources, which is the substrate switch every bioprocess makes. Condition axis only.",
        accession="ProteomeXchange (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Yu 2021 (proteome and metabolome under nitrogen limitation)",
        citation="Yu R, Vorontsov E, Sihlbom C, Nielsen J. eLife 2021;10:e65722.",
        url="https://doi.org/10.7554/eLife.65722",
        klass="Metabolite / precursor",
        tier=4,
        genotypes_n=1,
        genotypes="wild type",
        env_n=10,
        env="nitrogen-limited chemostats",
        instances_n=10,
        instances_basis="estimate",
        phenotype="proteome + metabolome",
        shape="multi-omic",
        dim=2500,
        seq_basis="reference-only",
        why="Nitrogen limitation is the standard trigger for redirecting carbon to product, and this measures both layers under it. Condition axis only, and small.",
        accession="eLife data availability (unconfirmed)",
        confidence="recall",
    ),
    # ---- Expression noise and single-cell distributions ----
    Candidate(
        name="Newman 2006 (protein noise, GFP library)",
        citation="Newman JRS, Ghaemmaghami S, Ihmels J, Breslow DK, Noble M, DeRisi JL, Weissman JS. Nature 2006;441:840-846.",
        url="https://doi.org/10.1038/nature04785",
        klass="Modality / backbone",
        tier=2,
        genotypes_n=2500,
        genotypes="~2,500 GFP-tagged strains",
        env_n=2,
        env="rich / minimal",
        instances_n=5000,
        instances_basis="product",
        phenotype="protein abundance and cell-to-cell noise",
        shape="mean + noise per strain",
        dim=2,
        seq_basis="S288C+tag",
        why="Measures the distribution rather than the mean, per protein, in two media. Noise is the quantity a per-cell readout recovers and a bulk one destroys, so this sets what a single-cell design should expect to see.",
        accession="Nature SI",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Metzger 2015 (promoter mutation effects on expression and noise)",
        citation="Metzger BPH, Yuan DC, Gruber JD, Duveau F, Wittkopp PJ. Nature 2015;521:344-346.",
        url="https://doi.org/10.1038/nature14424",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=235,
        genotypes="~235 promoter mutants",
        env_n=1,
        env="1",
        instances_n=235,
        instances_basis="estimate",
        phenotype="expression mean and noise",
        shape="scalar pair",
        dim=2,
        seq_basis="S288C+designed-edit",
        why="Single mutations in one native promoter, each with expression and noise measured. The mutational spectrum underlying the natural promoter variation the MPRA rows survey.",
        accession="Nature SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Keren 2013 (promoter activity across conditions)",
        citation="Keren L, Zackay O, Lotan-Pompan M, Barenholz U, Dekel E, Sasson V, Aidelberg G, Bren A, Zeevi D, Weinberger A, Alon U, Milo R, Segal E. Mol Syst Biol 2013;9:701.",
        url="https://doi.org/10.1038/msb.2013.59",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=900,
        genotypes="~900 native promoter reporters",
        env_n=10,
        env="10 conditions",
        instances_n=9000,
        instances_basis="product",
        phenotype="promoter activity, time-resolved",
        shape="time series",
        dim=10,
        seq_basis="S288C+reporter-locus",
        why="Native promoter strength measured directly, across conditions and over time, rather than inferred from mRNA. This is the closest published answer to the question of how strong every promoter actually is.",
        accession="Mol Syst Biol SI (unconfirmed)",
        confidence="recall",
    ),
    # ---- Natural isolate genome and phenotype panels ----
    Candidate(
        name="Liti 2009 (population genomics of S. cerevisiae and S. paradoxus)",
        citation="Liti G, Carter DM, Moses AM, et al., Louis EJ. Nature 2009;458:337-341.",
        url="https://doi.org/10.1038/nature07743",
        klass="Natural variation",
        tier=3,
        genotypes_n=70,
        genotypes="~70 sequenced isolates",
        env_n=1,
        env="1",
        instances_n=70,
        instances_basis="estimate",
        phenotype="genome sequence and population structure",
        shape="assembly / variants",
        seq_basis="isolate-WGS",
        why="The founding sequenced isolate panel and the source of the SGRP strains many later crosses use as parents. Historical, and it is where several parental genomes in the segregant rows trace to.",
        accession="SGRP / ENA (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Strope 2015 (100 clinical and natural genomes)",
        citation="Strope PK, Skelly DA, Kozmin SG, Mahadevan G, Stone EA, Magwene PM, Dietrich FS, McCusker JH. Genome Res 2015;25:762-774.",
        url="https://doi.org/10.1101/gr.185538.114",
        klass="Natural variation",
        tier=3,
        genotypes_n=100,
        genotypes="100 sequenced isolates",
        env_n=10,
        env="phenotype panel",
        instances_n=1000,
        instances_basis="estimate",
        phenotype="growth and clinical-relevant traits",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="A sequenced panel enriched for clinical isolates, which sit outside the vineyard and laboratory backgrounds most panels oversample. Diversity of origin is what makes a natural-variation training set generalize.",
        accession="Genome Research SI; SRA (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Warringer 2011 (trait variation across the SGRP panel)",
        citation="Warringer J, Zorgo E, Cubillos FA, et al., Blomberg A, Liti G. PLoS Genet 2011;7:e1002111.",
        url="https://doi.org/10.1371/journal.pgen.1002111",
        klass="Natural variation",
        tier=2,
        genotypes_n=200,
        genotypes="~200 sequenced isolates",
        env_n=200,
        env="~200 conditions",
        instances_n=40000,
        instances_basis="estimate",
        phenotype="growth-curve parameters",
        shape="rate / lag / yield",
        dim=3,
        seq_basis="isolate-WGS",
        why="A very wide condition panel on sequenced isolates, with growth-curve parameters rather than endpoint fitness. Pairs with Scan-o-matic as an actually-released panel rather than a platform.",
        accession="PLoS Genetics SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Skelly 2013 (expression variation across isolates)",
        citation="Skelly DA, Merrihew GE, Riffle M, et al., MacCoss MJ, Akey JM. Genome Res 2013;23:1496-1504.",
        url="https://doi.org/10.1101/gr.155762.113",
        klass="Natural variation",
        tier=3,
        genotypes_n=22,
        genotypes="22 sequenced isolates",
        env_n=1,
        env="1",
        instances_n=22,
        instances_basis="reported",
        phenotype="transcriptome + proteome",
        shape="paired vectors",
        dim=8000,
        seq_basis="isolate-WGS",
        why="Transcriptome and proteome on the same isolates, which is the paired anchor the RNA-to-protein inference argument needs. Few strains, but the pairing is measured rather than joined across studies.",
        accession="Genome Research SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Yvert 2013 (single-cell trait variation across isolates)",
        citation="Yvert G, Ohnuki S, Nogami S, Imanaga Y, Fehrmann S, Schacherer J, Ohya Y. BMC Syst Biol 2013;7:54.",
        url="https://doi.org/10.1186/1752-0509-7-54",
        klass="Natural variation",
        tier=3,
        genotypes_n=37,
        genotypes="37 isolates",
        env_n=1,
        env="1",
        instances_n=37,
        instances_basis="reported",
        phenotype="single-cell morphology (CalMorph)",
        shape="vector (501)",
        dim=501,
        seq_basis="isolate-WGS",
        why="CalMorph morphology on natural isolates rather than deletion strains, in the same feature space as the already-built Ohya and Ohnuki datasets. Joins the morphology layer to the natural-variation axis without harmonization.",
        accession="BMC SI (unconfirmed)",
        perturbseq="output",
        confidence="recall",
    ),
    # ---- Growth physiology and chemostat reference ----
    Candidate(
        name="Brauer 2008 (growth-rate-controlled chemostat transcriptome)",
        citation="Brauer MJ, Huttenhower C, Airoldi EM, et al., Botstein D. Mol Biol Cell 2008;19:352-367.",
        url="https://doi.org/10.1091/mbc.e07-08-0779",
        klass="Expression / single cell",
        tier=2,
        genotypes_n=1,
        genotypes="wild type",
        env_n=36,
        env="36 chemostats, 6 limitations",
        instances_n=36,
        instances_basis="reported",
        phenotype="mRNA abundance",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="Separates growth rate from nutrient identity by design, which no batch-culture dataset can. A large share of any expression response is a growth-rate confound, and this is the reference for removing it.",
        accession="GEO (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Boer 2010 (metabolome across nutrient limitations)",
        citation="Boer VM, Crutchfield CA, Bradley PH, Botstein D, Rabinowitz JD. Mol Biol Cell 2010;21:198-211.",
        url="https://doi.org/10.1091/mbc.e09-07-0597",
        klass="Metabolite / precursor",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=25,
        env="nutrient limitations x growth rates",
        instances_n=25,
        instances_basis="reported",
        phenotype="intracellular metabolite concentrations",
        shape="vector",
        dim=100,
        seq_basis="reference-only",
        why="The metabolite counterpart to Brauer on the same chemostat design, so metabolite pools are separated from growth rate the same way. Covers central-carbon nodes the deletion metabolomics rows measure at low power.",
        accession="MBoC SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Xia 2022 (proteome allocation across conditions)",
        citation="Xia J, Sanchez BJ, Chen Y, Campbell K, Kasvandik S, Nielsen J. Nat Commun 2022;13:2819.",
        url="https://doi.org/10.1038/s41467-022-30513-2",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=30,
        env="chemostat conditions",
        instances_n=30,
        instances_basis="estimate",
        phenotype="absolute proteome, resource allocation",
        shape="vector",
        dim=2000,
        seq_basis="reference-only",
        why="Proteome allocation measured across a designed condition grid and used to parameterize enzyme-constrained models. The bridge between this table's proteomics rows and the Yeast9 modeling scaffold.",
        accession="Nat Commun SI (unconfirmed)",
        confidence="recall",
    ),
    # ---- Segregant and bulk-segregant mapping, remaining ----
    Candidate(
        name="Ehrenreich 2010 (X-QTL bulk segregant mapping)",
        citation="Ehrenreich IM, Torabi N, Jia Y, Kent J, Martis S, Shapiro JA, Gresham D, Caudy AA, Kruglyak L. Nature 2010;464:1039-1042.",
        url="https://doi.org/10.1038/nature08923",
        klass="Natural variation",
        tier=3,
        genotypes_n=1000000,
        genotypes="~1e6 pooled segregants",
        env_n=17,
        env="17 selective conditions",
        instances_n=17,
        instances_basis="reported",
        phenotype="allele-frequency shift (pooled)",
        shape="genome-wide track",
        dim=11000,
        seq_basis="segregant-WGS",
        why="Millions of segregants assayed in bulk rather than individually, so power comes from pool size instead of strain construction. The phenotype is per-pool rather than per-strain, which is a different record type and needs a pooled-genotype schema.",
        accession="Nature SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Cubillos 2013 (F12 recombinant panel construction)",
        citation="Cubillos FA, Parts L, Salinas F, et al., Liti G. Genetics 2013;195:1141-1155.",
        url="https://doi.org/10.1534/genetics.113.155515",
        klass="Natural variation",
        tier=4,
        genotypes_n=192,
        genotypes="192 sequenced F12 segregants",
        env_n=20,
        env="conditions",
        instances_n=3840,
        instances_basis="estimate",
        phenotype="growth traits",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="The panel and genotype table the already-listed Cubillos 2017 nitrogen study resolves against. Ingesting it is what makes that row's genotype axis self-contained.",
        accession="Genetics SI Table S1",
        confidence="recall",
    ),
    Candidate(
        name="She 2018 (natural variant effects across a large isolate panel)",
        citation="She R, Jarosz DF. Cell 2018;172:478-490.",
        url="https://doi.org/10.1016/j.cell.2017.12.015",
        klass="Natural variation",
        tier=2,
        genotypes_n=500,
        genotypes="~500 isolates",
        env_n=40,
        env="~40 conditions",
        instances_n=20000,
        instances_basis="estimate",
        phenotype="growth traits",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="Maps how often a natural variant's effect depends on the strain it sits in, across a wide condition panel. The natural-variation counterpart to Galardini's four-background deletion tensor.",
        accession="Cell SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Sadhu 2016 (CRISPR-directed causal variant mapping)",
        citation="Sadhu MJ, Bloom JS, Day L, Kruglyak L. Science 2016;352:1113-1116.",
        url="https://doi.org/10.1126/science.aaf5124",
        klass="Natural variation",
        tier=3,
        genotypes_n=1000,
        genotypes="~1,000 targeted recombinants",
        env_n=10,
        env="conditions",
        instances_n=10000,
        instances_basis="estimate",
        phenotype="fitness",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="Drives recombination to a chosen point, so a locus can be swapped against a fixed background rather than inherited with everything linked to it. Turns correlational QTL into causal tests.",
        accession="Science SI (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Smith 2008 (gene-environment interaction eQTL)",
        citation="Smith EN, Kruglyak L. PLoS Biol 2008;6:e83.",
        url="https://doi.org/10.1371/journal.pbio.0060083",
        klass="Natural variation",
        tier=3,
        genotypes_n=109,
        genotypes="109 segregants",
        env_n=2,
        env="glucose / ethanol",
        instances_n=218,
        instances_basis="product",
        phenotype="mRNA abundance",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="segregant-WGS",
        why="Transcriptomes on the same segregants in two carbon sources, so an expression QTL can be shown to be condition-dependent. Small, and it is the design Boocock 2025 scales up.",
        accession="GEO (unconfirmed)",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Kessi-Perez 2019 (nitrogen-consumption QTL panel)",
        citation="Kessi-Perez EI, Molinet J, Martinez C, et al. 2019/2020 nitrogen-utilization QTL work.",
        url="https://doi.org/10.1186/s13068-019-1440-9",
        klass="Natural variation",
        tier=4,
        genotypes_n=100,
        genotypes="sequenced segregants / isolates",
        env_n=2,
        env="nitrogen conditions",
        instances_n=200,
        instances_basis="estimate",
        phenotype="nitrogen consumption, fermentation rate",
        shape="scalar",
        seq_basis="segregant-WGS",
        why="A second nitrogen-utilization mapping panel alongside Cubillos 2017. Nitrogen sets higher-alcohol and ester formation, so it is fermentation-relevant, and the exact release needs pinning.",
        accession="journal SI (unconfirmed)",
        confidence="recall",
    ),
    # ---- Wild-type condition-response backbone ----
    Candidate(
        name="Spellman 1998 (cell-cycle transcriptome)",
        citation="Spellman PT, Sherlock G, Zhang MQ, Iyer VR, Anders K, Eisen MB, Brown PO, Botstein D, Futcher B. Mol Biol Cell 1998;9:3273-3297.",
        url="https://doi.org/10.1091/mbc.9.12.3273",
        klass="Expression / single cell",
        tier=4,
        genotypes_n=1,
        genotypes="wild type, synchronized",
        env_n=4,
        env="4 synchronization methods",
        instances_n=77,
        instances_basis="reported",
        phenotype="mRNA abundance time course",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="The cell-cycle expression reference. O'Duibhir showed a large share of deletion expression response is a cell-cycle-distribution artifact, and this is the signature used to detect and remove it.",
        accession="SGD / SPELL",
        confidence="recall",
    ),
    Candidate(
        name="Gasch 2001 (DNA damage response transcriptome)",
        citation="Gasch AP, Huang M, Metzner S, Botstein D, Elledge SJ, Brown PO. Mol Biol Cell 2001;12:2987-3003.",
        url="https://doi.org/10.1091/mbc.12.10.2987",
        klass="Expression / single cell",
        tier=4,
        genotypes_n=4,
        genotypes="wild type + checkpoint mutants",
        env_n=6,
        env="damaging agents",
        instances_n=24,
        instances_basis="estimate",
        phenotype="mRNA abundance",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="S288C-KO",
        why="Extends the stress-response backbone to genotoxic stress, and includes checkpoint mutants, so it carries a small genotype axis rather than being wild-type only.",
        accession="SGD / SPELL",
        confidence="recall",
    ),
    Candidate(
        name="Causton 2001 (environmental change transcriptome)",
        citation="Causton HC, Ren B, Koh SS, et al., Young RA. Mol Biol Cell 2001;12:323-337.",
        url="https://doi.org/10.1091/mbc.12.2.323",
        klass="Expression / single cell",
        tier=4,
        genotypes_n=1,
        genotypes="wild type",
        env_n=8,
        env="environmental changes",
        instances_n=40,
        instances_basis="estimate",
        phenotype="mRNA abundance",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="reference-only",
        why="An independently generated environmental-response compendium contemporaneous with Gasch 2000. Two independent measurements of the same response class is a platform-effect control.",
        accession="SGD / SPELL",
        confidence="recall",
    ),
    Candidate(
        name="Tkach 2012 (protein relocalization under DNA damage)",
        citation="Tkach JM, Yimit A, Lee AY, et al., Andrews BJ, Brown GW. Nat Cell Biol 2012;14:966-976.",
        url="https://doi.org/10.1038/ncb2549",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=4000,
        genotypes="GFP collection",
        env_n=3,
        env="DNA-damaging agents",
        instances_n=12000,
        instances_basis="product",
        phenotype="localization and abundance change",
        shape="per-cell distribution",
        dim=17,
        seq_basis="S288C+tag",
        why="The GFP collection imaged under damage, giving a second condition axis on the imaging layer alongside Breker's stress panel. Relocalization is a response no abundance measurement captures.",
        accession="TheCellVision / Nat Cell Biol SI (unconfirmed)",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Denervaud 2013 (microfluidic GFP dynamics)",
        citation="Denervaud N, Becker J, Delgado-Gonzalo R, et al., Maerkl SJ. PNAS 2013;110:15842-15847.",
        url="https://doi.org/10.1073/pnas.1308265110",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1000,
        genotypes="~1,000 GFP strains",
        env_n=4,
        env="dynamic environments",
        instances_n=4000,
        instances_basis="product",
        phenotype="time-resolved single-cell abundance",
        shape="time series per cell",
        dim=17,
        seq_basis="S288C+tag",
        why="Single-cell protein dynamics under controlled environmental switches, rather than endpoint snapshots. The temporal axis every other imaging row lacks.",
        accession="PNAS SI (unconfirmed)",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Lenstra 2011 (chromatin regulator deletion expression)",
        citation="Lenstra TL, Benschop JJ, Kim T, et al., Holstege FCP. Mol Cell 2011;42:536-549.",
        url="https://doi.org/10.1016/j.molcel.2011.03.026",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=165,
        genotypes="165 chromatin-regulator deletions",
        env_n=1,
        env="1",
        instances_n=165,
        instances_basis="reported",
        phenotype="mRNA abundance",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="S288C-KO",
        why="Transcriptomes for the chromatin machinery specifically, from the group behind the built Kemmeren compendium and in the same format. Chromatin regulators are the class whose deletion most reshapes the expression landscape.",
        accession="GEO (unconfirmed)",
        perturbseq="output",
        confidence="recall",
    ),
    Candidate(
        name="Pelechano 2013 (transcript isoform landscape)",
        citation="Pelechano V, Wei W, Steinmetz LM. Nature 2013;497:127-131.",
        url="https://doi.org/10.1038/nature12121",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1,
        genotypes="wild type",
        env_n=2,
        env="glucose / galactose",
        instances_n=2,
        instances_basis="reported",
        phenotype="transcript isoform boundaries and abundance",
        shape="isoform set",
        dim=6000,
        seq_basis="reference-only",
        why="Shows a single gene produces many transcript isoforms with distinct 5' and 3' ends. Every expression row in this table collapses that to one number per gene, and this is the map of what that collapse discards.",
        accession="ArrayExpress / GEO (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Nagalakshmi 2008 (RNA-seq transcriptome annotation)",
        citation="Nagalakshmi U, Wang Z, Waern K, Shou C, Raha D, Gerstein M, Snyder M. Science 2008;320:1344-1349.",
        url="https://doi.org/10.1126/science.1158441",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=1,
        genotypes="wild type",
        env_n=1,
        env="1",
        instances_n=1,
        instances_basis="reported",
        phenotype="transcript structure and abundance",
        shape="genome-wide track",
        dim=6000,
        seq_basis="reference-only",
        why="The first RNA-seq transcriptome, and the annotation correction underneath the microarray-era expression rows. Historical, and it is why gene models used by later datasets differ.",
        accession="SRA / GEO (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Kaplan 2009 (nucleosome sequence preferences)",
        citation="Kaplan N, Moore IK, Fondufe-Mittendorf Y, et al., Segal E. Nature 2009;458:362-366.",
        url="https://doi.org/10.1038/nature07667",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=1,
        genotypes="wild type (in vitro + in vivo)",
        env_n=2,
        env="in vitro / in vivo",
        instances_n=2,
        instances_basis="reported",
        phenotype="nucleosome occupancy",
        shape="genome-wide track",
        dim=67000,
        seq_basis="reference-only",
        why="Separates the sequence-intrinsic component of nucleosome positioning from the cellular one by reconstituting in vitro. That separation is what lets a promoter model attribute an effect to sequence rather than to context.",
        accession="GEO (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Yofe 2016 (SWAT N-terminal library)",
        citation="Yofe I, Weill U, Meurer M, et al., Knop M, Schuldiner M. Nat Methods 2016;13:371-378.",
        url="https://doi.org/10.1038/nmeth.3795",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=1800,
        genotypes="~1,800 N-terminally tagged strains",
        env_n=1,
        env="1",
        instances_n=1800,
        instances_basis="estimate",
        phenotype="localization and abundance",
        shape="categorical + scalar",
        dim=3,
        seq_basis="S288C+tag",
        why="N-terminal tagging, which preserves C-terminal signals the GFP collection destroys. Complements Huh 2003 on exactly the proteins that collection mislocalizes.",
        accession="Nat Methods SI; EUROSCARF (unconfirmed)",
        confidence="recall",
    ),
    Candidate(
        name="Meurer 2018 (genome-wide GFP library, rebuilt)",
        citation="Meurer M, Duan Y, Sass E, Kats I, Herbst K, Buchmuller BC, Dederer V, Huber F, Kirrmaier D, Stefl M, Van Laer K, Dick TP, Lemberg MK, Khmelinskii A, Levy ED, Knop M. Nat Methods 2018;15:617-620.",
        url="https://doi.org/10.1038/s41592-018-0044-9",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=4000,
        genotypes="~4,000 tagged strains",
        env_n=1,
        env="1",
        instances_n=4000,
        instances_basis="estimate",
        phenotype="abundance and localization",
        shape="categorical + scalar",
        dim=3,
        seq_basis="S288C+tag",
        why="A rebuilt and sequence-verified tagging collection, correcting errors accumulated in the original. Verify the DOI before ingestion; it collides with the Weill 2018 SWAT row in this table.",
        accession="Nat Methods SI (DOI collision with Weill 2018 UNRESOLVED)",
        confidence="recall",
    ),
    Candidate(
        name="Levy 2008 (morphology and phenotypic capacitance)",
        citation="Levy SF, Siegal ML. PLoS Biol 2008;6:e264.",
        url="https://doi.org/10.1371/journal.pbio.0060264",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=1700,
        genotypes="~1,700 deletion strains",
        env_n=1,
        env="1",
        instances_n=1700,
        instances_basis="estimate",
        phenotype="single-cell morphology variability",
        shape="per-cell distribution",
        dim=200,
        seq_basis="S288C-KO",
        why="Scores how much a deletion increases cell-to-cell variability rather than shifting the mean, identifying capacitor genes. Variance as the phenotype is a readout only per-cell data supports, which is the argument the Perturb-seq proposal rests on.",
        accession="PLoS Biology SI (unconfirmed)",
        confidence="recall",
        perturbseq="output",
    ),
    # ---- Transcript against protein, and the layers in between ------------
    # Added because the list ranked steady-state transcript and steady-state
    # protein well and had no row for what sits between them. Transcript level
    # and protein level disagree, and the disagreement is not noise: it is
    # translation efficiency and it is turnover. Without a row for each, a model
    # asked to infer protein from RNA has no term for why the two differ.
    # Citations, counts and venues below were verified against PubMed on
    # 2026-09-13; none of these papers is in the mirror.
    Candidate(
        name="Grossbach 2022 (BY x RM transcriptome, proteome and phosphoproteome)",
        citation="Grossbach J, Gillet L, Clement-Ziza M, Schmalohr CL, Schubert OT, Schutter M, Mawer JSP, Barnes CA, Bludau I, Weith M, Tessarz P, Graef M, Aebersold R, Beyer A. Mol Syst Biol 2022;18:e10712.",
        url="https://doi.org/10.15252/msb.202110712",
        klass="Natural variation",
        tier=2,
        genotypes_n=112,
        genotypes="112 genomically defined strains (BY x RM panel)",
        env_n=1,
        env="1",
        instances_n=112,
        instances_basis="reported",
        phenotype="transcript, protein and phosphosite abundance",
        shape="three vectors",
        dim=9978,
        seq_basis="segregant-WGS",
        why="The only row that measures transcript, protein and protein phosphorylation on one set of sequenced strains, 1,862 proteins and 2,116 phosphopeptides over 988 proteins. The pairing it yields is per strain rather than per gene: for each of the 112 genotypes the mRNA level and the protein level of the same gene are both observed, which is what turns the RNA-to-protein question from a population correlation into a within-strain residual. It also reports that genetic variants changing a protein's phosphorylation are mostly different from those changing its abundance, so the phosphosite layer is not a proxy for the protein layer.",
        accession="Mol Syst Biol SI + PRIDE, identifier not fetched this pass",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Teyssonniere 2024 (species-wide proteome against transcriptome)",
        citation="Teyssonniere EM, Trebulle P, Muenzner J, Loegler V, Ludwig D, Amari F, Mulleder M, Friedrich A, Hou J, Ralser M, Schacherer J. Proc Natl Acad Sci USA 2024;121:e2319211121.",
        url="https://doi.org/10.1073/pnas.2319211121",
        klass="Natural variation",
        tier=2,
        genotypes_n=942,
        genotypes="942 natural isolates",
        env_n=1,
        env="1",
        instances_n=942,
        instances_basis="reported",
        phenotype="protein abundance, paired with transcript abundance",
        shape="vector",
        dim=2000,
        seq_basis="isolate-WGS",
        why="Quantitative proteomes for 942 sequenced isolates, the same panel whose transcriptomes the built Caudal 2024 supplies, so the strain-level pairing is one to one rather than an overlap. It is also the published answer to what that pairing shows: mRNA and protein correlate weakly at the population gene level, and the variants associated with protein levels overlap those associated with transcript levels by 3 percent. That makes it the calibration for any RNA-to-protein inference rather than only another proteome. DE-DUPLICATION OPEN: it shares authors and a panel with Muenzner 2024 at 796 isolates, and whether the two are independent acquisitions or one re-served is not established here and must be settled before both are built.",
        accession="PNAS SI; PMC11087752. PRIDE identifier not fetched this pass",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Foss 2007 (BY x RM segregant proteome)",
        citation="Foss EJ, Radulovic D, Shaffer SA, Ruderfer DM, Bedalov A, Goodlett DR, Kruglyak L. Nat Genet 2007;39:1369-1375.",
        url="https://doi.org/10.1038/ng.2007.22",
        klass="Natural variation",
        tier=3,
        genotypes_n=None,
        genotypes="BY x RM segregants; the count is not stated in the abstract and was not confirmed",
        env_n=1,
        env="1",
        instances_n=None,
        instances_basis="estimate",
        phenotype="protein abundance (label-free MS)",
        shape="vector",
        seq_basis="segregant-WGS",
        why="The first segregant proteome, and the independent replicate of Grossbach 2022 on the same cross fifteen years earlier and on a different platform. Its finding is the one this whole group turns on: the loci that influence protein abundance differ from those that influence transcript levels. Ranked below Grossbach because Grossbach measures the transcript layer on the same strains and this does not, and because neither the segregant count nor the proteome depth was confirmed this pass.",
        accession="Nat Genet supplementary tables; not fetched",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="McManus 2014 (ribosome profiling, allele-specific)",
        citation="McManus CJ, May GE, Spealman P, Shteyman A. Genome Res 2014;24:422-430.",
        url="https://doi.org/10.1101/gr.164996.113",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=3,
        genotypes="S. cerevisiae, S. paradoxus and their F1 hybrid",
        env_n=1,
        env="1",
        instances_n=3,
        instances_basis="reported",
        phenotype="ribosome occupancy paired with mRNA abundance",
        shape="two vectors (5,474 orthologs)",
        dim=10948,
        seq_basis="reference-only",
        why="Translation efficiency measured gene by gene alongside the mRNA abundance of the same gene, which is the first of the two terms that make protein level differ from transcript level. Its result is the reason the term is needed: ribosome occupancy is more conserved than transcript abundance, so translation buffers divergence in mRNA rather than tracking it. The S. paradoxus arm is out of species and would be stored separately or dropped; the cerevisiae arm and the hybrid's cerevisiae alleles are in scope.",
        accession="GEO, series not fetched this pass",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Martin-Perez 2017 (protein half-lives)",
        citation="Martin-Perez M, Villen J. Cell Syst 2017;5:283-294.e5.",
        url="https://doi.org/10.1016/j.cels.2017.08.008",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=1,
        genotypes="wild type, exponential growth",
        env_n=1,
        env="1",
        instances_n=1,
        instances_basis="reported",
        phenotype="protein turnover rate",
        shape="vector (3,160 proteins)",
        dim=3160,
        seq_basis="reference-only",
        why="Turnover rates for 3,160 proteins, the second term that makes protein level differ from transcript level. Half-lives span three orders of magnitude and are not normally distributed, so a single global scaling from mRNA to protein cannot be right, and this is the per-gene coefficient that replaces it. The paper also reports that localization, complex membership and connectivity predict turnover better than sequence features do, which names the covariates such a model needs.",
        accession="Cell Systems SI; PRIDE identifier not fetched this pass",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Sun 2013 (mRNA synthesis and decay rates across deletion strains)",
        citation="Sun M, Schwalb B, Pirkl N, Maier KC, Schenk A, Failmezger H, Tresch A, Cramer P. Mol Cell 2013;52:52-62.",
        url="https://doi.org/10.1016/j.molcel.2013.09.010",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=46,
        genotypes="46 deletion strains of mRNA degradation and metabolism genes",
        env_n=1,
        env="1",
        instances_n=46,
        instances_basis="reported",
        phenotype="mRNA level, synthesis rate and decay rate",
        shape="three vectors",
        dim=18000,
        seq_basis="S288C-KO",
        why="Comparative dynamic transcriptome analysis decomposes a transcript level into the synthesis rate and the decay rate that produce it, across 46 single deletions. That is the transcript-side counterpart of Martin-Perez 2017's protein half-lives, so the two together give both turnover terms. Its finding also warns what a steady-state compendium hides: a change in degradation rate is generally compensated by a change in synthesis rate, so mRNA level is buffered and two strains with the same level can have different kinetics. Overlap with the other deletion-strain expression sets is on the strain axis, not the measurement: the built Kemmeren 2014 (1,484 deletions), Hughes 2000, Hu 2007 and Lenstra 2011 all store a steady-state level per gene, and none stores a rate, so this row is a second layer on shared strains rather than a second copy of one. The 46 strains are single deletions of mRNA degradation and metabolism genes and are expected to be inside the Kemmeren set by gene identity; the exact shared-strain count needs the GEO strain list, which was not fetched, and the abstract does not say the strains came from that collection.",
        accession="GEO, series not fetched this pass",
        confidence="recall",
        added=True,
    ),
    Candidate(
        name="Hughes 2000 (compendium of expression profiles)",
        citation="Hughes TR, Marton MJ, Jones AR, Roberts CJ, Stoughton R, et al., Friend SH. Cell 2000;102:109-126.",
        url="https://doi.org/10.1016/S0092-8674(00)00015-5",
        klass="Expression / single cell",
        tier=3,
        genotypes_n=300,
        genotypes="300 mutations and chemical treatments",
        env_n=1,
        env="1",
        instances_n=300,
        instances_basis="reported",
        phenotype="mRNA log ratio (two-color array)",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="S288C-KO",
        why="The original deletion expression compendium, and the oldest independent measurement of the phenotype the built Kemmeren 2014 measures. Ingesting it makes cross-laboratory replication of the deletion transcriptome testable on gene-by-gene profiles fourteen years apart and on different array platforms, which is a check the built set cannot run against itself. The 300 profiles mix deletions, titratable alleles and compound treatments, so the compound arm carries a reference genotype and the mutant arm an S288C-KO one; the split was not confirmed this pass.",
        accession="Rosetta compendium, distributed with the paper; not fetched",
        confidence="recall",
        added=True,
    ),
]

# Why a row left the ranked list. Keyed by name so the move table can state it
# without the reason being buried in the row it no longer occupies.
REMOVAL_REASON: dict[str, str] = {
    "Bloom 2019 (16-parent cross)": (
        "Built since the previous pass as the 50th supported dataset, 13,950 "
        "segregants over 38 traits, L0-L4 verified. It is now a join partner "
        "rather than a candidate."
    )
}

EXCLUDED: list[Excluded] = [
    Excluded(
        name="Snoek 2015 robot-assisted genome shuffling (ethanol tolerance)",
        reason="Shuffled progeny are recombinants of unrecorded parental segments with no per-strain sequencing, so the genotype cannot be reconstructed.",
        rule="no-sequence",
    ),
    Excluded(
        name="Adaptive laboratory evolution / EMS mutagenesis tolerance panels (e.g. farnesol-FPP selection)",
        reason="Mutations are unmapped, so a strain's genome is unknown even in principle without whole-genome resequencing the study did not do.",
        rule="no-sequence",
    ),
    Excluded(
        name="Avalos lab isobutanol-biosensor deletion screen",
        reason="Described only in a PhD thesis and a DOE report. Track for publication; not citable as a dataset.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Kitamoto/Kaneko ester and higher-alcohol screen",
        reason="Citation could not be pinned to a specific paper. A placeholder, not a located dataset.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Fuhrer/Zamboni flow-injection metabolomics",
        reason="The validated genome-scale application is in E. coli. No yeast deletion-collection version was located.",
        rule="off-species",
    ),
    Excluded(
        name="Anglada-Girotto 2022 (CRISPRi + metabolomics)",
        reason="352 CRISPRi genes against 1,342 drug-induced metabolic changes, but in E. coli. Verified from the mirrored PDF.",
        rule="off-species",
    ),
    Excluded(
        name="Wildenhain 2016 (Sci Data chemical-genetic matrix)",
        reason="A data descriptor, not a second experiment. Its own Data Citation 1 is NCBI PubChem BioAssay AID 1159580, which is exactly what torchcell/datasets/scerevisiae/wildenhain2015.py already ingests: 242 strains, 5,518 compounds, 492,126 interaction tests.",
        rule="already-built",
    ),
    Excluded(
        name="O'Duibhir 2014 expression compendium",
        reason="The paper's expression data IS the Kemmeren 2014 compendium plus PCA transforms, already built, as oduibhir2014.py states; only its growth-rate readout was new and that is built too. Caught by check_candidate_overlap.py rather than by inspection, which is the first time the script found a duplicate before a person did.",
        rule="already-built",
    ),
    Excluded(
        name="Ozturk 2022 proteome-effects screen",
        reason="3,308 deletions in S. pombe. Out of species scope.",
        rule="off-species",
    ),
    Excluded(
        name="Chen 2024 K. phaffii CRISPR fitness screen; Y1000+ multi-species panel",
        reason="Not S. cerevisiae. Revisit if the generalization axis widens to other hosts.",
        rule="off-species",
    ),
    Excluded(
        name="iIsor850 genome-scale model (Issatchenkia orientalis)",
        reason="A reconstruction, not an experiment, and off species. No genome-wide I. orientalis phenotype screen was located; the CABBI knockout library is unpublished.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="CABBI 13C-MFA kinetic-model repository",
        reason="16 knockout strains total, and the released fluxes are K-FIT model fits recapitulating 75-77 percent of measurements, not raw measurement. Blank 2005 supersedes it on both counts.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="YEASTRACT+ (Teixeira 2023; Monteiro 2020)",
        reason="A literature-curated knowledge base of 304,547 documented TF-target associations, each tagged with the evidence type (DNA binding or expression) and with the environmental condition in which it was observed, so a condition-scoped regulatory graph is recoverable from it. It is a prior over the genome, not a genotype-by-environment measurement, and it belongs in the graph layer beside GO and the metabolic model rather than in this table. Both papers are mirrored; the flat files are shared on request by the maintainers and are not yet held, and scYeast (Fan 2027) built its attention prior from the documented-targets subset (about 170,000 pairs).",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Sc2.0 / Sc3.0 synthetic-genome design",
        reason="Genome-engineering programs, not gene-indexed perturbation-phenotype data.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Lee 2014 / Hoepfner 2014 / Vanacloig-Pedros 2022 / Messner 2023 / Mulleder 2016 / Lian 2019 / Mormino 2022 / Nadal-Ribelles 2025 and 12 others",
        reason="Named as top candidates by the 2026-07 triage pass and BUILT since. Only Lee 2014 remains outstanding and is kept as row 4.",
        rule="already-built",
    ),
    Excluded(
        name="Bloom 2019 (16-parent cross)",
        reason="Ranked 12 in the previous pass and built since as the 50th supported dataset: 13,950 sequenced segregants over 38 traits, L0-L4 verified. Its place in this pass is as a join partner, not a candidate.",
        rule="already-built",
    ),
    # -- Cited by Jackson 2020 as inputs, and none of them is a dataset in this
    # table's sense. Recorded rather than dropped silently, because "Jackson uses
    # four priors" reads like four candidate rows until the shape of each is
    # written down.
    Excluded(
        name="TF-target prior networks used by Jackson 2020 (YEASTRACT, ATAC-motif of Castro 2019, Bussemaker affinity of Ward 2008, Tchourine 2018 gold standard)",
        reason="Four regulator-by-gene matrices: 11,486 unsigned YEASTRACT edges over 3,912 genes and 152 TFs, 71,865 signed ATAC-motif edges over 5,551 genes and 138 TFs, a dense 6,516 by 123 affinity matrix, and the 1,403-edge signed gold standard. Each is a prior over the genome with no genotype and no environment, so each belongs in the graph layer beside GO and YEASTRACT+, not in a genotype-by-environment table. All four ship as TSVs in the eLife Source code 1 archive.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Tchourine 2018 bulk expression compendium (2,577 observations)",
        reason="The benchmark Jackson 2020 compares its single-cell network against. It is a re-aggregation of public bulk series, and the strain behind an observation lives in each source series' metadata rather than in the released matrix, so admitting it is a de-duplication and re-curation task against Kemmeren 2014, Gasch 2000 and Brauer 2008 rather than one loader. Revisit if the per-sample genotype table is located.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Albert 2014 (X-pQTL, single-cell protein abundance in large populations)",
        reason="Albert FW, Treusch S, Shockley AH, Bloom JS, Kruglyak L. Nature 2014;506:494-497. Considered as the BY x RM segregant proteome and it is not one: protein level is read from a green fluorescent protein fusion for one gene at a time in a large unsequenced pool, and the mapping comes from sorting that pool rather than from genotyped individuals. There is no strain-by-protein matrix to store, so it cannot enter as a proteome row. The segregant proteome rows here are Grossbach 2022 and Foss 2007.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Scholes 2019 bulk RNA-seq control (GEO GSE135430)",
        reason="Jackson 2020's external bulk reference point. One wild-type BY4741 genotype in one condition, and the study's own variable is the RNA isolation protocol, so there is neither a genotype nor an environment axis to record.",
        rule="not-a-dataset",
    ),
    # -- From the microbe-perturb-seq collection, 42 items read this pass. Seven
    # carry a real genotype axis in a single-cell bioproduction host; three of
    # those are already built, three are ranked here, and the seventh is below.
    # The rest divide cleanly into the four rows that follow.
    Excluded(
        name="Brandner 2025 (mapSPLiT, CRISPRa and CRISPRi single-cell transcriptomes in E. coli and P. putida)",
        reason="The only microbial dataset found that reaches both Perturb-seq axes on purpose: 52 transcription factors targeted by 118 guides, pooled, with multi-guide combinations included to resolve genetic interactions, read out as a single-cell transcriptome. Pseudomonas putida is a single-cell bioproduction host and so is in scope for the generalization axis. It is a preprint and no deposited accession was found, so there is no scriptable data route and it is excluded rather than ranked. It is the closest existing design template for the campaign and should be revisited on publication.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Mammalian Perturb-seq (Replogle 2022, Zhu 2026, Dixit 2016, Datlinger 2017, Yao 2024)",
        reason="The format references, and off species. Replogle perturbs every expressed gene by CRISPR interference across more than 2.5 million cells; Zhu 2026 crosses a genome-scale perturbation with several stimulation time points across about 22 million cells, which is the perturbation-by-context tensor a yeast campaign is an analog of; Yao 2024 pools random multi-perturbation composites over 598 genes and decompresses them, which is the one published route to second-order genetic interactions at a tractable cell count. None supplies yeast data; all four inform the design.",
        rule="off-species",
    ),
    Excluded(
        name="Bacterial single-cell transcriptomics (Wang 2023 M3-seq, Ma 2023 BacDrop, McNulty 2023 ProBac-seq, Kuchina 2021 microSPLiT)",
        reason="Off species, and the axis is environmental rather than genetic: hundreds of thousands of cells over stress, antibiotic and growth-stage conditions in isogenic backgrounds. No perturbation-by-readout matrix to ingest.",
        rule="off-species",
    ),
    Excluded(
        name="Fulcher 2024 (nanoSPLITS), Taniguchi 2010, Baronas 2026, Leonaviciene 2023 and 2020, Datlinger 2021, Macosko 2015, Rosenberg 2018",
        reason="Platform and paired-modality papers, all off species. nanoSPLITS measures a transcriptome and a mass-spectrometry proteome from the same single cell, which is the precedent for inferring an expensive layer from a cheap one on paired anchors; Taniguchi 2010 is the reference for protein and mRNA copy-number noise in a microbial host. Neither releases a perturbation matrix.",
        rule="off-species",
    ),
    Excluded(
        name="Reviews, protocols and single-cell statistics in the collection (Nadal-Ribelles 2024, Sun 2023, Larson 2013, Zun 2023, Gaisser 2024, Squair 2021, Zhang 2020, Svensson 2020, Grun 2014, Robinson 2008, McCarthy 2012, Hart 2013, Wang 2026)",
        reason="Thirteen items with no perturbation data of their own. They are the sizing and false-discovery discipline for the campaign, not rows: pseudobulk replicate variation, sequencing depth per cell, dispersion estimation, and whether droplet counts are zero-inflated.",
        rule="not-a-dataset",
    ),
]


# ---------------------------------------------------------------------------
# Band assignment and synergies, held here rather than on the rows.
#
# Both are cross-row judgments: a band says how a row compares to every other
# row, and a synergy names a second dataset. Spread across 180 row literals they
# could not be read as a single decision, and a partner could be renamed in one
# place and not the other. Keyed by name, checked against the rows at import, so
# a typo is a startup failure rather than a silently dropped pairing.
# ---------------------------------------------------------------------------

BANDS: dict[str, tuple[Band, str]] = {
    # -- perturb-seq: rows a yeast Perturb-seq campaign is designed against ----
    "Boocock 2025 (single-cell eQTL mapping)": (
        "perturb-seq",
        "Genotype recovered per cell from the cell's own reads, crossed with a "
        "per-cell transcriptome. The largest existing yeast dataset in the "
        "quadrant the campaign targets.",
    ),
    "N'Guessan 2025 (segregant scRNA-seq eQTL)": (
        "perturb-seq",
        "About 4,500 sequenced segregants profiled by single-cell RNA-seq, so "
        "genotype and transcriptome are measured in the same cell.",
    ),
    "Hale 2024 (CRISPRi x natural variation)": (
        "perturb-seq",
        "The one released yeast library that crosses a CRISPR interference guide "
        "set with a sequenced genetic background, which is the input axis a "
        "campaign has to plan against.",
    ),
    "Jackson 2020 (TF-deletion single-cell atlas)": (
        "perturb-seq",
        "Transcribed genotype barcodes with a per-cell transcriptome readout, "
        "fully crossed over 11 conditions. The design a yeast Perturb-seq is "
        "specified against.",
    ),
    "Hackett 2020 (IDEA inducible-TF transcriptome time series)": (
        "perturb-seq",
        "Induction rather than deletion, with a transcriptome followed over time. "
        "Jackson 2020 names transient induction as the perturbation modality most "
        "likely to produce a detectable expression response.",
    ),
    "Dong 2021 (MAGIC + SAM biosensor)": (
        "perturb-seq",
        "A genome-wide CRISPR activation, interference and deletion library sorted "
        "on a metabolite biosensor, so the label is product concentration rather "
        "than fitness. The pattern a sorted Perturb-seq would reuse.",
    ),
    "Momen-Roknabadi 2020 (inducible CRISPRi library)": (
        "perturb-seq",
        "A genome-scale inducible CRISPR interference library with a released "
        "per-guide matrix. Induction control is what lets a knockdown be applied "
        "after a cell is captured rather than during outgrowth.",
    ),
    "McGlincy 2021 (genome-scale CRISPRi library)": (
        "perturb-seq",
        "The second independently designed genome-scale yeast CRISPR interference "
        "library with a released per-guide matrix, so guide design and gene effect "
        "can be separated.",
    ),
    "Bao 2018 (CHAnGE single-nucleotide library)": (
        "perturb-seq",
        "A guide-indexed genome-wide library whose edits sit below the open "
        "reading frame, so the perturbation is an allele rather than a null.",
    ),
    "Roy 2018 (multiplexed precision editing)": (
        "perturb-seq",
        "Multiplexed designed edits per cell, which is combinatorial input in the "
        "sense the Perturb-seq specification requires.",
    ),
    "Crook 2016 (tunable RNAi, isobutanol + 1-butanol)": (
        "perturb-seq",
        "A dose-graded knockdown axis. Graded rather than binary perturbation is "
        "what turns a per-cell readout into a dose-response curve.",
    ),
    "Mukherjee 2021 (CRISPRi essential genes x acetic acid)": (
        "perturb-seq",
        "Knockdown reaches the essential genes a deletion collection cannot hold, "
        "which is a third of the genome a deletion-based campaign would miss.",
    ),
    "Lian 2017 (CRISPR-AID, beta-carotene)": (
        "perturb-seq",
        "Three perturbation modalities in one cell, activation, interference and "
        "deletion, which is combinatorial input by modality rather than by gene.",
    ),
    "Guo 2018 (CRISPR-Cas9 tiling of essential genes)": (
        "perturb-seq",
        "Guide tiling produces a graded allelic series across one gene, so guide "
        "position rather than gene identity carries the effect.",
    ),
    "Jaffe 2019 (multiplexed CRISPR interference epistasis)": (
        "perturb-seq",
        "Two knockdowns in the same cell with a measured interaction, which is the "
        "only combinatorial CRISPR interference design located in yeast.",
    ),
    "Sun 2013 (mRNA synthesis and decay rates across deletion strains)": (
        "perturb-seq",
        "A designed single-gene perturbation crossed with a transcriptome, and the "
        "only one that resolves the level into a synthesis rate and a decay rate. "
        "Rates are what a per-cell snapshot cannot recover on its own.",
    ),
    "Hughes 2000 (compendium of expression profiles)": (
        "perturb-seq",
        "The original designed-perturbation expression compendium, and the "
        "independent measurement that makes cross-laboratory replication of the "
        "deletion transcriptome testable against the built Kemmeren 2014.",
    ),
    "Hu 2007 (TF deletion expression compendium)": (
        "perturb-seq",
        "A designed single-gene perturbation crossed with a transcriptome over 269 "
        "transcription-factor deletions, which is the bulk form of what the "
        "campaign measures per cell, and the widest regulator coverage available.",
    ),
    "Lenstra 2011 (chromatin regulator deletion expression)": (
        "perturb-seq",
        "The same design over chromatin regulators rather than sequence-specific "
        "factors, so the two together cover both halves of transcriptional "
        "control by deletion.",
    ),
    "Newman 2006 (protein noise, GFP library)": (
        "perturb-seq",
        "Per-cell protein abundance distributions across a genome-scale tagged "
        "collection. Expression noise per gene is what sets how large a "
        "perturbation effect has to be before a per-cell readout can resolve it.",
    ),
    "Jackson 2023 (wild-type scRNA-seq time course for RNA kinetics)": (
        "perturb-seq",
        "The platform companion of Jackson 2020: 173,361 cells of one strain, "
        "which is what sets the per-cell variance a perturbation has to exceed.",
    ),
    "Nadal-Ribelles 2019 (sensitive yeast scRNA-seq)": (
        "perturb-seq",
        "The high-sensitivity yeast protocol, strand and isoform aware. Sensitivity "
        "per cell is the constraint on how small a perturbation effect a campaign "
        "can resolve.",
    ),
    "Gasch 2017 (single-cell stress heterogeneity)": (
        "perturb-seq",
        "Separates intrinsic from extrinsic per-cell variation in an isogenic "
        "population under stress, which is the null a perturbation effect is "
        "measured against.",
    ),
    "Brettner 2024 (ultra-high-throughput yeast scRNA-seq)": (
        "perturb-seq",
        "Combinatorial barcoding in yeast at 96 to 384 multiplexed genotypes or "
        "environments per run. The throughput route by which a campaign becomes "
        "affordable.",
    ),
    "Jariani 2020 (yeast scRNA-seq through lag phase)": (
        "perturb-seq",
        "Per-cell states through a carbon-source shift, so the readout is the "
        "distribution across an unsynchronized population rather than its mean.",
    ),
    "Urbonaite 2021 (yeastDrop-Seq under drug treatment)": (
        "perturb-seq",
        "Single-drug arms against their combination with a per-cell readout, the "
        "environmental analog of a combinatorial genetic perturbation.",
    ),
    "Su 2023 (single-cell transcriptomes under four stresses)": (
        "perturb-seq",
        "Full-length rather than three-prime tag counting, so isoform-level "
        "readout is on the table for a campaign that needs it.",
    ),
    "Wang 2022 (single-cell transcriptomes across replicative aging)": (
        "perturb-seq",
        "The only yeast single-cell set with age as the condition axis, which is a "
        "covariate any pooled campaign carries whether or not it measures it.",
    ),
    "Puddu 2019 (WGS of the deletion collection)": (
        "perturb-seq",
        "Whole-genome sequence for every strain in the deletion collection. A "
        "pooled campaign reads a barcode and infers a genotype; this is the "
        "measurement of how often that inference is wrong, and Jackson 2020 built "
        "its own strains rather than use the collection for related reasons.",
    ),
    "Mulleder 2012 (prototrophic deletion collection)": (
        "perturb-seq",
        "The prototrophic deletion collection. Jackson 2020 used a prototrophic "
        "background because auxotrophy blocks minimal and nitrogen-limited media, "
        "so this is the strain resource a campaign in defined media needs.",
    ),
    # -- molecular layers ------------------------------------------------------
    "Grossbach 2022 (BY x RM transcriptome, proteome and phosphoproteome)": (
        "molecular layers",
        "Three layers on one set of 112 sequenced strains, so the transcript and "
        "the protein of a gene are observed in the same genotype and the RNA to "
        "protein question becomes a within-strain residual.",
    ),
    "Teyssonniere 2024 (species-wide proteome against transcriptome)": (
        "molecular layers",
        "Proteomes for 942 isolates, the panel whose transcriptomes the built "
        "Caudal 2024 supplies, so the strain-level pairing is one to one.",
    ),
    "Foss 2007 (BY x RM segregant proteome)": (
        "molecular layers",
        "The independent replicate of Grossbach 2022 on the same cross, and the "
        "first measurement that protein-level loci differ from transcript-level "
        "loci.",
    ),
    "McManus 2014 (ribosome profiling, allele-specific)": (
        "molecular layers",
        "Translation efficiency per gene alongside the mRNA abundance of that "
        "gene, which is the first of the two terms that separate protein level "
        "from transcript level.",
    ),
    "Martin-Perez 2017 (protein half-lives)": (
        "molecular layers",
        "Protein turnover per gene, the second of those two terms, spanning three "
        "orders of magnitude and therefore not replaceable by a global constant.",
    ),
    "Jakobson 2025 (genome-to-proteome map)": (
        "molecular layers",
        "Protein abundance across the same sequenced segregant panel Albert 2018 "
        "profiles by transcriptome, so protein and transcript quantitative trait "
        "loci are measurable on one genotype set.",
    ),
    "Muenzner 2024 (natural-isolate proteome)": (
        "molecular layers",
        "796 proteomes drawn from the sequenced 1,011-isolate panel that the "
        "supported Caudal 2024 transcriptomes also come from.",
    ),
    "Albert 2018 (eQTL in 1,012 segregants)": (
        "molecular layers",
        "The transcriptome half of the segregant panel that Jakobson 2025, "
        "Gerke 2017 and Eder 2020 measure protein, metabolite and flux on.",
    ),
    "Cooper 2010 (CE-MS amino-acid metabolome)": (
        "molecular layers",
        "Amino-acid pools on the deletion collection by capillary electrophoresis, "
        "the same trait class the supported Mulleder 2016 measures by mass "
        "spectrometry and the same strains Kemmeren 2014 profiles.",
    ),
    "Aulakh 2025 (genome-scale ionome)": (
        "molecular layers",
        "A metabolic readout on the whole deletion collection, which is the "
        "genotype axis the supported expression compendium already covers.",
    ),
    "Blank 2005 (13C metabolic flux)": (
        "molecular layers",
        "The only row measuring flux rather than a concentration, on deletion "
        "mutants that also have a transcriptome in the supported set.",
    ),
    "Zhu 2014 (kinase / phosphatase lipidomics)": (
        "molecular layers",
        "A lipidome over 129 signaling deletions, most of which carry an "
        "expression profile in the supported compendium.",
    ),
    "Hackett 2016 (SIMMER multi-omic flux)": (
        "molecular layers",
        "Flux, metabolite and transcript measured in one study under the same "
        "nutrient limitations, so the join is internal rather than across papers.",
    ),
    "Boer 2010 (metabolome across nutrient limitations)": (
        "molecular layers",
        "The metabolite half of the chemostat nutrient-limitation series whose "
        "transcriptome half is Brauer 2008, same laboratory and same conditions.",
    ),
    "Brauer 2008 (growth-rate-controlled chemostat transcriptome)": (
        "molecular layers",
        "Expression at controlled growth rate under each nutrient limitation, "
        "which is the condition axis Boer 2010 and Hackett 2016 measure "
        "metabolites and flux on.",
    ),
    "Airoldi 2016 (nitrogen-limited steady-state and dynamic transcriptome)": (
        "molecular layers",
        "Expression under the exact nitrogen-limited conditions that Jackson 2020 "
        "profiles single cells in, so the deletion effect separates from the "
        "medium effect.",
    ),
    "Leutert 2023 (phosphoproteome x 101 conditions)": (
        "molecular layers",
        "Enzyme regulation acts faster than transcription, so a phosphosite layer "
        "over 101 conditions is what explains flux changes an expression table "
        "cannot.",
    ),
    "Gerke 2017 (urea-cycle mQTL)": (
        "molecular layers",
        "Metabolite quantitative trait loci on a segregant cross that also has an "
        "expression map, so a metabolite locus can be read through its transcript.",
    ),
    "Ambroset 2014 (metabolite QTL)": (
        "molecular layers",
        "A 74-metabolite panel on a sequenced segregant panel, the widest "
        "metabolite vector available on a recombinant genotype axis.",
    ),
    "Eder 2020 (flux QTL)": (
        "molecular layers",
        "Flux mapped to segregant genotypes, which is the flux counterpart of the "
        "expression and protein maps on the same panel type.",
    ),
    "Tengolics 2024 (domestication metabolome)": (
        "molecular layers",
        "Metabolite levels across sequenced isolates, the panel the supported "
        "Caudal 2024 transcriptomes and Muenzner 2024 proteomes also sit on.",
    ),
    "Yu 2021 (proteome and metabolome under nitrogen limitation)": (
        "molecular layers",
        "Both layers measured in one study under nitrogen limitation, so it "
        "calibrates the protein-to-metabolite step that cross-study joins assume.",
    ),
    "Skelly 2013 (expression variation across isolates)": (
        "molecular layers",
        "Transcriptome and proteome on the same 22 strains, which is the only row "
        "where the cheap and expensive layers are paired within one experiment "
        "rather than joined across two.",
    ),
}


def _syn(
    partner: str, status: Literal["supported", "candidate"], join: str, yields: str
) -> Synergy:
    return Synergy(partner=partner, partner_status=status, join=join, yields=yields)


# Partner names in the "supported" column are the names used by
# experiments/database/scripts/build_supported_datasets_table.py, so a reader can
# find the partner in the supported table without translating.
SYNERGIES: dict[str, list[Synergy]] = {
    "Boocock 2025 (single-cell eQTL mapping)": [
        _syn(
            "Nadal-Ribelles 2025 (Perturb-seq)",
            "supported",
            "single-cell transcriptome readout, one genotype per cell",
            "One dataset carries genotypes made by recombination and the other "
            "genotypes made by deletion, on the same readout, which is the test "
            "of whether a per-cell expression model transfers between the two.",
        ),
        _syn(
            "Albert 2018 (eQTL in 1,012 segregants)",
            "candidate",
            "same BY by RM cross, bulk against single cell",
            "The same expression quantitative trait loci measured in bulk and per "
            "cell, so the cell-to-cell variance a bulk average hides is "
            "recoverable.",
        ),
        _syn(
            "Jakobson 2025 (genome-to-proteome map)",
            "candidate",
            "segregant genotype class",
            "Transcript and protein variation on one recombinant panel.",
        ),
    ],
    "N'Guessan 2025 (segregant scRNA-seq eQTL)": [
        _syn(
            "Boocock 2025 (single-cell eQTL mapping)",
            "candidate",
            "segregant genotype class, single-cell transcriptome",
            "Two independent single-cell expression quantitative trait loci maps, "
            "which is the replication neither has on its own.",
        ),
        _syn(
            "Bloom 2019 (16-parent cross)",
            "supported",
            "segregant genotype class, 16 founders against 2",
            "Whether an expression effect measured in a two-parent cross holds "
            "across sixteen founders.",
        ),
    ],
    "Hale 2024 (CRISPRi x natural variation)": [
        _syn(
            "Smith 2016 (CRISPRi chem-genetic)",
            "supported",
            "CRISPR interference knockdown in S288C",
            "One knockdown effect measured in the reference background and across "
            "169 segregants, which separates a gene effect from a background "
            "effect.",
        ),
        _syn(
            "McGlincy 2021 (genome-scale CRISPRi library)",
            "candidate",
            "guide library design",
            "Whether a knockdown effect is a property of the gene or of the guide.",
        ),
        _syn(
            "Bloom 2019 (16-parent cross)",
            "supported",
            "segregant genotype class, growth traits",
            "Knockdown effect against natural allele effect on comparable panels.",
        ),
    ],
    "Jackson 2020 (TF-deletion single-cell atlas)": [
        _syn(
            "Kemmeren 2014",
            "supported",
            "single deletions of the same transcription factors, bulk against "
            "single cell",
            "The same knockout read out as a population average and as a per-cell "
            "distribution, which is the only available calibration of what "
            "pseudobulk loses.",
        ),
        _syn(
            "Nadal-Ribelles 2025 (Perturb-seq)",
            "supported",
            "deletion perturbation with a single-cell transcriptome readout",
            "Twelve genotypes over eleven conditions against about 3,500 genotypes "
            "over two, so genotype breadth and condition breadth are separable for "
            "once.",
        ),
        _syn(
            "Airoldi 2016 (nitrogen-limited steady-state and dynamic transcriptome)",
            "candidate",
            "the same nitrogen-limited media formulations",
            "A wild-type expression baseline in the same medium, which is what "
            "makes the deletion effect a contrast rather than an absolute.",
        ),
        _syn(
            "Hackett 2020 (IDEA inducible-TF transcriptome time series)",
            "candidate",
            "the same transcription factors, deletion against induction",
            "Loss of function against gain of function on one regulator set.",
        ),
    ],
    "Hackett 2020 (IDEA inducible-TF transcriptome time series)": [
        _syn(
            "Kemmeren 2014",
            "supported",
            "the same transcription factors, induction against deletion",
            "Whether a regulator's targets are the same set whether it is removed "
            "or over-induced.",
        ),
        _syn(
            "Sameith 2015 dm",
            "supported",
            "transcription-factor pairs",
            "Induction dynamics for the single factors whose double deletions "
            "carry a measured expression epistasis.",
        ),
    ],
    "Dong 2021 (MAGIC + SAM biosensor)": [
        _syn(
            "Lian 2019 (MAGIC CRISPR-AID)",
            "supported",
            "the same tri-functional library, fitness against biosensor readout",
            "One library, two labels: growth selection and product concentration, "
            "which is how a tolerance screen is converted into a production "
            "screen.",
        ),
        _syn(
            "Cachera 2023 (CRI-SPA betaxanthin)",
            "supported",
            "metabolite biosensor or colorimetric product readout on a "
            "genome-scale library",
            "Two product-labeled genome-scale screens on different chemistry.",
        ),
    ],
    "McGlincy 2021 (genome-scale CRISPRi library)": [
        _syn(
            "Momen-Roknabadi 2020 (inducible CRISPRi library)",
            "candidate",
            "independently designed guide libraries over the same genes",
            "Library-to-library transfer, which is the cheapest decisive test of "
            "whether a model learned gene function or guide-design idiosyncrasy.",
        ),
        _syn(
            "Smith 2016 (CRISPRi chem-genetic)",
            "supported",
            "CRISPR interference over the same gene set",
            "A knockdown fitness prior for the chemical-genetic conditions already "
            "built.",
        ),
    ],
    "Momen-Roknabadi 2020 (inducible CRISPRi library)": [
        _syn(
            "McGlincy 2021 (genome-scale CRISPRi library)",
            "candidate",
            "independently designed guide libraries over the same genes",
            "The paired half of the library-transfer test.",
        ),
        _syn(
            "SGD essentiality",
            "supported",
            "essential genes, knockdown against deletion",
            "A graded phenotype where a deletion collection records only absence.",
        ),
    ],
    "Mukherjee 2021 (CRISPRi essential genes x acetic acid)": [
        _syn(
            "Mormino 2022 (CRISPRi acetic-acid)",
            "supported",
            "acetic acid, CRISPR interference",
            "The genome-scale extension of a twelve-gene screen already built.",
        ),
        _syn(
            "Mota 2024 (weak-acid screen)",
            "supported",
            "weak-acid stress on deletion strains",
            "Essential-gene coverage for a phenotype the deletion collection can "
            "only sample from the non-essential side.",
        ),
    ],
    "Bao 2018 (CHAnGE single-nucleotide library)": [
        _syn(
            "Lian 2019 (MAGIC CRISPR-AID)",
            "supported",
            "guide-indexed genome-wide library from the same laboratory",
            "Sub-gene alleles against whole-gene activation, interference and "
            "deletion on one platform lineage.",
        ),
        _syn(
            "Li 2016 (tRNA fitness landscape)",
            "candidate",
            "designed single-nucleotide variants with a fitness label",
            "Genome-wide shallow variant coverage against single-gene exhaustive "
            "coverage.",
        ),
    ],
    "Guo 2018 (CRISPR-Cas9 tiling of essential genes)": [
        _syn(
            "SGD essentiality",
            "supported",
            "essential genes",
            "A graded allelic series where the built record is binary.",
        )
    ],
    "Jaffe 2019 (multiplexed CRISPR interference epistasis)": [
        _syn(
            "Costanzo 2016 dmi",
            "supported",
            "gene pairs, knockdown against deletion",
            "Whether digenic interaction measured between two nulls is recovered "
            "between two knockdowns, which decides if the built interaction data "
            "can supervise a knockdown campaign.",
        ),
        _syn(
            "Kuzmin 2018 tmi",
            "supported",
            "higher-order gene combinations",
            "The knockdown analog of trigenic interaction.",
        ),
    ],
    "Roy 2018 (multiplexed precision editing)": [
        _syn(
            "Bao 2018 (CHAnGE single-nucleotide library)",
            "candidate",
            "designed edits, single against multiplexed",
            "Whether multiplexed edit effects are the sum of their single-edit "
            "effects.",
        )
    ],
    "Crook 2016 (tunable RNAi, isobutanol + 1-butanol)": [
        _syn(
            "Lopez 2024 (isobutanol screen, private)",
            "supported",
            "isobutanol, knockdown against deletion",
            "A graded knockdown axis over the phenotype the built biosensor screen "
            "measures as a titer proxy.",
        ),
        _syn(
            "Kuroda 2019 (isobutanol-specific tolerance)",
            "candidate",
            "isobutanol tolerance on the same collection",
            "Dose-graded knockdown against whole-gene deletion for one phenotype.",
        ),
    ],
    "Lian 2017 (CRISPR-AID, beta-carotene)": [
        _syn(
            "Ozaydin 2013 (beta-carotene screen)",
            "supported",
            "beta-carotene titer",
            "Three perturbation modalities against a whole-collection deletion "
            "screen on the same product.",
        )
    ],
    "Sun 2013 (mRNA synthesis and decay rates across deletion strains)": [
        _syn(
            "Kemmeren 2014",
            "supported",
            "single deletion strains, gene by gene",
            "A steady-state transcript level beside the synthesis and decay rates "
            "that produce it, so two strains with the same level but different "
            "kinetics stop looking identical.",
        ),
        _syn(
            "Martin-Perez 2017 (protein half-lives)",
            "candidate",
            "turnover, transcript side against protein side, gene by gene",
            "Both half-life terms in one place, which is what an RNA-to-protein "
            "model needs in order to have a reason for the two to disagree.",
        ),
    ],
    "Hughes 2000 (compendium of expression profiles)": [
        _syn(
            "Kemmeren 2014",
            "supported",
            "deletion strains, gene by gene, fourteen years and two array "
            "platforms apart",
            "Cross-laboratory replication of the deletion transcriptome, which "
            "the built set cannot test against itself and which Nadal-Ribelles "
            "2025 failed cross-batch.",
        ),
        _syn(
            "Hu 2007 (TF deletion expression compendium)",
            "candidate",
            "deletion strains with a bulk expression readout",
            "A third independent compendium, so agreement can be measured across "
            "three laboratories rather than asserted from two.",
        ),
    ],
    "Hu 2007 (TF deletion expression compendium)": [
        _syn(
            "Kemmeren 2014",
            "supported",
            "single deletions with a bulk expression readout",
            "Two independently produced deletion expression compendia over "
            "overlapping regulators, which is the reproducibility check neither "
            "has alone and the largest such pair in yeast.",
        ),
        _syn(
            "Jackson 2020 (TF-deletion single-cell atlas)",
            "candidate",
            "transcription-factor deletions",
            "269 regulators in bulk against 11 per cell, so the regulators the "
            "single-cell atlas cannot reach still carry a profile.",
        ),
    ],
    "Lenstra 2011 (chromatin regulator deletion expression)": [
        _syn(
            "Kemmeren 2014",
            "supported",
            "single deletions with a bulk expression readout, different gene class",
            "Chromatin regulators beside sequence-specific factors on one "
            "expression readout.",
        ),
        _syn(
            "Sameith 2015 dm",
            "supported",
            "regulator deletions, single against double",
            "Whether chromatin-regulator effects combine the way "
            "transcription-factor effects do.",
        ),
    ],
    "Newman 2006 (protein noise, GFP library)": [
        _syn(
            "Messner 2023 (proteome)",
            "supported",
            "protein abundance across the genome, per cell against population",
            "Which proteins have a population mean that no single cell is near, "
            "which decides where a mean-level model is the wrong object.",
        ),
        _syn(
            "Gasch 2017 (single-cell stress heterogeneity)",
            "candidate",
            "per-cell variation in an isogenic population",
            "Noise at the protein and transcript layers on the same question.",
        ),
    ],
    "Jackson 2023 (wild-type scRNA-seq time course for RNA kinetics)": [
        _syn(
            "Jackson 2020 (TF-deletion single-cell atlas)",
            "candidate",
            "same platform and laboratory, wild type against deletions",
            "The unperturbed per-cell variance the perturbed atlas is read against.",
        )
    ],
    "Nadal-Ribelles 2019 (sensitive yeast scRNA-seq)": [
        _syn(
            "Nadal-Ribelles 2025 (Perturb-seq)",
            "supported",
            "same laboratory, protocol against application",
            "The sensitivity ceiling of the platform the built genome-scale "
            "single-cell dataset was produced on.",
        ),
        _syn(
            "Gasch 2017 (single-cell stress heterogeneity)",
            "candidate",
            "isogenic per-cell variation in BY4741",
            "Two independent measurements of how much an unperturbed yeast "
            "population varies, which is the floor a perturbation must clear.",
        ),
    ],
    "Gasch 2017 (single-cell stress heterogeneity)": [
        _syn(
            "Gasch 2000 (environmental stress response)",
            "candidate",
            "the same stress conditions, bulk against single cell",
            "How much of the canonical stress response is a population average of "
            "cells that are individually in different states.",
        )
    ],
    "Brettner 2024 (ultra-high-throughput yeast scRNA-seq)": [
        _syn(
            "Jackson 2020 (TF-deletion single-cell atlas)",
            "candidate",
            "multiplexed genotypes per single-cell run",
            "The throughput ceiling for a barcoded-genotype design, measured "
            "rather than assumed.",
        )
    ],
    "Jariani 2020 (yeast scRNA-seq through lag phase)": [
        _syn(
            "Jackson 2020 (TF-deletion single-cell atlas)",
            "candidate",
            "carbon-source shift with a per-cell readout",
            "A transition sampled densely in time where the atlas samples eleven "
            "steady states.",
        )
    ],
    "Urbonaite 2021 (yeastDrop-Seq under drug treatment)": [
        _syn(
            "Hoepfner 2014 (HIP/HOP atlas)",
            "supported",
            "chemical treatment of yeast",
            "A per-cell readout for compound response where the built atlas has a "
            "pooled fitness score.",
        )
    ],
    "Su 2023 (single-cell transcriptomes under four stresses)": [
        _syn(
            "Gasch 2000 (environmental stress response)",
            "candidate",
            "osmotic and starvation stress",
            "Full-length per-cell transcripts against the bulk stress-response "
            "reference.",
        )
    ],
    "Wang 2022 (single-cell transcriptomes across replicative aging)": [
        _syn(
            "Jackson 2023 (wild-type scRNA-seq time course for RNA kinetics)",
            "candidate",
            "wild-type per-cell states over time",
            "Replicative age as a covariate on a platform that does not measure it.",
        )
    ],
    "Puddu 2019 (WGS of the deletion collection)": [
        _syn(
            "Costanzo 2016 dmf",
            "supported",
            "the same deletion collection",
            "Converts the sequence basis of every S288C-KO row from an assumption "
            "into a measurement, including the largest built datasets.",
        ),
        _syn(
            "Nadal-Ribelles 2025 (Perturb-seq)",
            "supported",
            "pooled deletion strains identified by barcode",
            "Which pooled strains are not what their barcode says, which bears "
            "directly on the reported genotype-assignment impurity.",
        ),
    ],
    "Mulleder 2012 (prototrophic deletion collection)": [
        _syn(
            "Mulleder 2016 (amino-acid metabolome)",
            "supported",
            "the same prototrophic deletion strains",
            "The strain resource the built amino-acid metabolome was measured on, "
            "which is what makes defined-medium phenotyping interpretable.",
        ),
        _syn(
            "Jackson 2020 (TF-deletion single-cell atlas)",
            "candidate",
            "prototrophy as a precondition for minimal media",
            "A ready-made prototrophic library for a campaign that needs nitrogen "
            "or carbon limitation.",
        ),
    ],
    # -- molecular layers ------------------------------------------------------
    "Grossbach 2022 (BY x RM transcriptome, proteome and phosphoproteome)": [
        _syn(
            "Albert 2018 (eQTL in 1,012 segregants)",
            "candidate",
            "BY x RM segregants, transcript layer",
            "112 strains with transcript and protein measured together against "
            "1,012 with transcript alone, so the within-strain RNA-to-protein "
            "residual fitted on the small panel can be applied to the large one.",
        ),
        _syn(
            "Jakobson 2025 (genome-to-proteome map)",
            "candidate",
            "segregant genotype class, protein layer",
            "Two segregant proteomes on different panels and platforms, which is "
            "what separates a protein quantitative trait locus from a batch.",
        ),
        _syn(
            "Leutert 2023 (phosphoproteome x 101 conditions)",
            "candidate",
            "phosphosite identity",
            "Phosphorylation driven by genotype against phosphorylation driven by "
            "condition, on one site vocabulary.",
        ),
    ],
    "Teyssonniere 2024 (species-wide proteome against transcriptome)": [
        _syn(
            "Caudal 2024 (pan-transcriptome)",
            "supported",
            "the same sequenced isolates, 942 against 943",
            "Transcript and protein for the same strain, one to one rather than "
            "by overlap. The published result on this pair is that the two "
            "correlate weakly and share 3 percent of their associated variants, "
            "so it is the calibration an inference model is scored against.",
        ),
        _syn(
            "Muenzner 2024 (natural-isolate proteome)",
            "candidate",
            "the 1,011-isolate panel, protein layer",
            "De-duplication before either is built: the two share authors and a "
            "panel, and whether they are independent acquisitions is unsettled.",
        ),
    ],
    "Foss 2007 (BY x RM segregant proteome)": [
        _syn(
            "Grossbach 2022 (BY x RM transcriptome, proteome and phosphoproteome)",
            "candidate",
            "the same cross, protein layer, fifteen years and two platforms apart",
            "Cross-laboratory replication of segregant protein abundance, which "
            "neither measurement can establish alone.",
        )
    ],
    "McManus 2014 (ribosome profiling, allele-specific)": [
        _syn(
            "Messner 2023 (proteome)",
            "supported",
            "gene identity, translation rate against protein abundance",
            "A per-gene translation term to put against measured protein level, "
            "which is the coefficient a model otherwise has to learn blind.",
        ),
        _syn(
            "Martin-Perez 2017 (protein half-lives)",
            "candidate",
            "gene identity, synthesis against degradation",
            "Both halves of protein turnover, so steady-state abundance can be "
            "decomposed rather than only predicted.",
        ),
    ],
    "Martin-Perez 2017 (protein half-lives)": [
        _syn(
            "Messner 2023 (proteome)",
            "supported",
            "gene identity, half-life against abundance",
            "Which proteins are abundant because they are made fast and which "
            "because they are destroyed slowly, a distinction abundance alone "
            "cannot make.",
        ),
        _syn(
            "Kemmeren 2014",
            "supported",
            "gene identity, protein half-life against transcript response",
            "Whether a transcript change reaches the protein layer at all, which "
            "for a long-lived protein it largely does not.",
        ),
    ],
    "Jakobson 2025 (genome-to-proteome map)": [
        _syn(
            "Albert 2018 (eQTL in 1,012 segregants)",
            "candidate",
            "the same sequenced segregant panel",
            "Protein and transcript quantitative trait loci on one genotype set, "
            "which is the direct measurement of how much of protein variation "
            "transcript variation explains.",
        ),
        _syn(
            "Messner 2023 (proteome)",
            "supported",
            "protein abundance, natural variation against deletion",
            "Whether the proteome response to a deleted gene resembles the "
            "response to a natural allele of it.",
        ),
    ],
    "Muenzner 2024 (natural-isolate proteome)": [
        _syn(
            "Caudal 2024 (pan-transcriptome)",
            "supported",
            "the sequenced 1,011-isolate panel",
            "Transcriptome and proteome on overlapping isolates, which is the "
            "cheap-layer against expensive-layer transfer question at natural "
            "genetic distance.",
        ),
        _syn(
            "Dutta 2026 (barcoded natural-isolate chemical response)",
            "candidate",
            "the same 1,011-isolate panel",
            "Genome, transcriptome, proteome and a chemogenomic response surface "
            "on one genotype axis.",
        ),
        _syn(
            "Teyssonniere 2024 (species-wide proteome against transcriptome)",
            "candidate",
            "the 1,011-isolate panel, protein layer",
            "De-duplication before either is built: 796 isolates here against 942 "
            "there, shared authors and one panel, and no evidence yet on whether "
            "the two acquisitions are independent.",
        ),
    ],
    "Albert 2018 (eQTL in 1,012 segregants)": [
        _syn(
            "Jakobson 2025 (genome-to-proteome map)",
            "candidate",
            "the same segregant panel",
            "The transcript half of a paired transcript and protein map.",
        ),
        _syn(
            "Gerke 2017 (urea-cycle mQTL)",
            "candidate",
            "segregant genotype class",
            "A metabolite locus read through the transcripts of the pathway that "
            "produces it.",
        ),
    ],
    "Cooper 2010 (CE-MS amino-acid metabolome)": [
        _syn(
            "Mulleder 2016 (amino-acid metabolome)",
            "supported",
            "the same amino acids on an overlapping deletion set",
            "Two platforms measuring one trait class, which is the cleanest "
            "available test of whether a model learned biology or a batch effect.",
        ),
        _syn(
            "Kemmeren 2014",
            "supported",
            "the same deletion collection",
            "Amino-acid pools paired with the expression profile of the same knockout.",
        ),
    ],
    "Aulakh 2025 (genome-scale ionome)": [
        _syn(
            "Mulleder 2016 (amino-acid metabolome)",
            "supported",
            "the whole deletion collection, metabolite readout",
            "Two orthogonal metabolic panels on one genotype axis.",
        ),
        _syn(
            "Kemmeren 2014",
            "supported",
            "the same deletion collection",
            "Element levels against the transcriptional response of the same knockout.",
        ),
    ],
    "Blank 2005 (13C metabolic flux)": [
        _syn(
            "Kemmeren 2014",
            "supported",
            "deletion strains present in both",
            "Flux against expression for the same knockout, which is the only "
            "place the transcript-to-flux step can be fitted rather than assumed.",
        ),
        _syn(
            "Mulleder 2016 (amino-acid metabolome)",
            "supported",
            "deletion strains, flux against pool size",
            "Whether a changed pool reflects changed flux or changed demand.",
        ),
    ],
    "Zhu 2014 (kinase / phosphatase lipidomics)": [
        _syn(
            "da Silveira 2014 (lipidomics)",
            "supported",
            "lipid species on deletion strains",
            "A second lipidome on a signaling-focused genotype set.",
        ),
        _syn(
            "Kemmeren 2014",
            "supported",
            "kinase and phosphatase deletions",
            "Lipid composition against the transcriptional response of the same "
            "signaling mutant.",
        ),
        _syn(
            "Xue 2025 (free fatty acids, private)",
            "supported",
            "fatty-acid metabolism",
            "Lipid class distribution against free fatty-acid titer, the two sides "
            "of the malonyl-CoA sink.",
        ),
    ],
    "Hackett 2016 (SIMMER multi-omic flux)": [
        _syn(
            "Boer 2010 (metabolome across nutrient limitations)",
            "candidate",
            "the same nutrient-limited chemostats",
            "Metabolite concentration and flux under one condition set, which is "
            "what a kinetic model needs and neither supplies alone.",
        ),
        _syn(
            "Brauer 2008 (growth-rate-controlled chemostat transcriptome)",
            "candidate",
            "nutrient limitation at controlled growth rate",
            "Expression, metabolite and flux over one condition axis.",
        ),
    ],
    "Boer 2010 (metabolome across nutrient limitations)": [
        _syn(
            "Brauer 2008 (growth-rate-controlled chemostat transcriptome)",
            "candidate",
            "the same chemostat limitations and growth rates",
            "The metabolite and transcript halves of one experimental series.",
        ),
        _syn(
            "Zelezniak 2018 (metabolome)",
            "supported",
            "metabolite panel",
            "Condition-driven against genotype-driven metabolite variation.",
        ),
    ],
    "Brauer 2008 (growth-rate-controlled chemostat transcriptome)": [
        _syn(
            "Boer 2010 (metabolome across nutrient limitations)",
            "candidate",
            "the same chemostat limitations",
            "The transcript half of a paired transcript and metabolite series.",
        ),
        _syn(
            "Airoldi 2016 (nitrogen-limited steady-state and dynamic transcriptome)",
            "candidate",
            "growth rate under limitation, carbon against nitrogen",
            "Whether the growth-rate expression program is nutrient-general.",
        ),
    ],
    "Airoldi 2016 (nitrogen-limited steady-state and dynamic transcriptome)": [
        _syn(
            "Jackson 2020 (TF-deletion single-cell atlas)",
            "candidate",
            "the same nitrogen-limited media",
            "A wild-type bulk baseline for the medium the single-cell atlas "
            "perturbs in.",
        ),
        _syn(
            "Yu 2021 (proteome and metabolome under nitrogen limitation)",
            "candidate",
            "nitrogen limitation",
            "Transcript, protein and metabolite under one limitation.",
        ),
    ],
    "Leutert 2023 (phosphoproteome x 101 conditions)": [
        _syn(
            "Gasch 2000 (environmental stress response)",
            "candidate",
            "overlapping stress conditions",
            "Post-translational regulation against transcriptional regulation for "
            "one condition set, on the layer that acts first.",
        ),
        _syn(
            "Messner 2023 (proteome)",
            "supported",
            "protein identity",
            "Modification state against abundance, which abundance alone cannot "
            "separate.",
        ),
    ],
    "Gerke 2017 (urea-cycle mQTL)": [
        _syn(
            "Albert 2018 (eQTL in 1,012 segregants)",
            "candidate",
            "segregant genotype class",
            "A metabolite locus and the expression loci in the same interval.",
        ),
        _syn(
            "Mulleder 2016 (amino-acid metabolome)",
            "supported",
            "nitrogen and amino-acid metabolites",
            "Natural-allele against deletion effects on one metabolic module.",
        ),
    ],
    "Ambroset 2014 (metabolite QTL)": [
        _syn(
            "Peeters 2021 (fermentation-trait QTL atlas)",
            "candidate",
            "segregant panels with fermentation traits",
            "74 metabolites against 18 mapped traits, so a trait locus can be read "
            "as a metabolite change.",
        ),
        _syn(
            "Zelezniak 2018 (metabolome)",
            "supported",
            "metabolite panel",
            "Recombinant against deletion genotypes on a metabolite readout.",
        ),
    ],
    "Eder 2020 (flux QTL)": [
        _syn(
            "Blank 2005 (13C metabolic flux)",
            "candidate",
            "flux measurement, natural variation against deletion",
            "Whether flux control points found by deletion are the loci natural "
            "variation actually moves.",
        ),
        _syn(
            "Albert 2018 (eQTL in 1,012 segregants)",
            "candidate",
            "segregant genotype class",
            "Flux loci against expression loci on comparable panels.",
        ),
    ],
    "Tengolics 2024 (domestication metabolome)": [
        _syn(
            "Caudal 2024 (pan-transcriptome)",
            "supported",
            "sequenced natural isolates",
            "Metabolite against transcript variation across the same population.",
        )
    ],
    "Yu 2021 (proteome and metabolome under nitrogen limitation)": [
        _syn(
            "Airoldi 2016 (nitrogen-limited steady-state and dynamic transcriptome)",
            "candidate",
            "nitrogen limitation",
            "The transcript layer for the protein and metabolite layers measured here.",
        )
    ],
    "Skelly 2013 (expression variation across isolates)": [
        _syn(
            "Muenzner 2024 (natural-isolate proteome)",
            "candidate",
            "natural isolates, paired against joined modalities",
            "A within-experiment paired transcript and protein measurement to "
            "calibrate the cross-study join.",
        )
    ],
    # -- scale rows whose join value is worth naming --------------------------
    "de Boer 2020 (100M random promoters)": [
        _syn(
            "Vaishnav 2022 (regulatory DNA fitness landscape)",
            "candidate",
            "the same random promoter library, expression against fitness",
            "Two labels on one sequence set, which turns a sequence-to-expression "
            "model into a sequence-to-fitness model without new data.",
        ),
        _syn(
            "Renganaath 2020 (natural promoter-variant MPRA)",
            "candidate",
            "promoter sequence with a measured expression readout",
            "Whether a model trained on random sequence predicts the effect of a "
            "real allele, the cheapest decisive transfer test in the table.",
        ),
    ],
    "Vaishnav 2022 (regulatory DNA fitness landscape)": [
        _syn(
            "de Boer 2020 (100M random promoters)",
            "candidate",
            "the same promoter library",
            "Fitness label for sequences that already carry an expression label.",
        )
    ],
    "Lee 2014 (HIP-HOP fitness signatures)": [
        _syn(
            "Hoepfner 2014 (HIP/HOP atlas)",
            "supported",
            "the same heterozygous and homozygous collections, compound response",
            "Two chemical-genetic matrices on one genotype axis, which is enough "
            "compound overlap to measure cross-screen reproducibility directly.",
        ),
        _syn(
            "Hillenmeyer 2008 het (FitDb HIP)",
            "supported",
            "the same collections and readout",
            "A third independent screen of the same design.",
        ),
    ],
    "Nguyen Ba 2022 (barcoded bulk QTL, 100k segregants)": [
        _syn(
            "Bloom 2019 (16-parent cross)",
            "supported",
            "segregant genotype class, panel size",
            "One cross at 100,000 progeny against sixteen founders at 14,000, "
            "which separates panel size from allelic diversity.",
        )
    ],
    "Galardini 2019 (four backgrounds x 38 conditions)": [
        _syn(
            "Costanzo 2021 (condition-SGA)",
            "supported",
            "deletion strains across conditions",
            "The same conditional fitness question asked in four backgrounds "
            "rather than one.",
        ),
        _syn(
            "Hale 2024 (CRISPRi x natural variation)",
            "candidate",
            "perturbation held fixed, background varied",
            "Deletion against knockdown for the background-dependence question.",
        ),
    ],
    "Parsons 2006 (bioactive-compound profiling)": [
        _syn(
            "Hoepfner 2014 (HIP/HOP atlas)",
            "supported",
            "compound response on deletion strains",
            "Compound overlap across screens run on different platforms.",
        )
    ],
    "Dutta 2026 (barcoded natural-isolate chemical response)": [
        _syn(
            "Caudal 2024 (pan-transcriptome)",
            "supported",
            "the sequenced 1,011-isolate panel",
            "Chemical response and transcriptome on one isolate set.",
        ),
        _syn(
            "Hoepfner 2014 (HIP/HOP atlas)",
            "supported",
            "compound response, isolates against deletions",
            "Whether compound sensitivity found by gene deletion predicts "
            "sensitivity across natural genetic backgrounds.",
        ),
    ],
    "Peter 2018 (1,011 isolate genomes + phenome)": [
        _syn(
            "Caudal 2024 (pan-transcriptome)",
            "supported",
            "the same 1,011 isolates",
            "The reference genome set that every isolate-WGS row in this table "
            "resolves its genotypes against.",
        )
    ],
    "Li 2016 (tRNA fitness landscape)": [
        _syn(
            "Domingo 2018 (tRNA double-mutant landscape)",
            "candidate",
            "the same tRNA gene, single against double mutants",
            "Whether a single-mutant landscape predicts the double-mutant one, "
            "which is the within-gene form of the epistasis question.",
        )
    ],
    "Turco 2023 (Yeast Phenome)": [
        _syn(
            "Hoepfner 2014 (HIP/HOP atlas)",
            "supported",
            "aggregated growth screens against a primary screen",
            "De-duplication, which is the precondition for knowing what the "
            "aggregate adds.",
        )
    ],
    "Liu 2021 (tryptophan / isobutanol tolerance)": [
        _syn(
            "Kuroda 2019 (isobutanol-specific tolerance)",
            "candidate",
            "isobutanol tolerance on the same deletion collection",
            "Same genotypes, same phenotype, different readout, which is a "
            "method-effect control no single screen provides.",
        )
    ],
    "Kuroda 2019 (isobutanol-specific tolerance)": [
        _syn(
            "Lopez 2024 (isobutanol screen, private)",
            "supported",
            "isobutanol on the deletion collection",
            "Tolerance against production for one product, on one genotype axis.",
        )
    ],
    "Trikka 2015 (carotenogenic heterozygous screen)": [
        _syn(
            "Ozaydin 2013 (beta-carotene screen)",
            "supported",
            "carotenoid readout, heterozygous against homozygous deletion",
            "Halved dosage against full deletion, which finds flux control points "
            "a null removes entirely.",
        )
    ],
}


# Rows in the final sixty that carry a time dimension, stated per row so the
# summary can count them and a reader can see which rows would exercise a time
# field on the record. Steady-state chemostat rows (Brauer 2008, Boer 2010,
# Hackett 2016, Yu 2021) vary dilution rate, which is a rate, not a time axis.
TIME_AXES: dict[str, str] = {
    "Hackett 2020 (IDEA inducible-TF transcriptome time series)": (
        "about eight time points after induction; one record per strain and time point"
    ),
    "Sun 2013 (mRNA synthesis and decay rates across deletion strains)": (
        "rates derived from metabolic labeling; the rates are stored, not the series"
    ),
    "Jariani 2020 (yeast scRNA-seq through lag phase)": (
        "time course through the lag phase of a glucose-to-maltose shift"
    ),
    "Jackson 2023 (wild-type scRNA-seq time course for RNA kinetics)": (
        "one time course, sampled continuously"
    ),
    "Wang 2022 (single-cell transcriptomes across replicative aging)": (
        "three ages: 2 h, 16 h and 36 h"
    ),
    "Airoldi 2016 (nitrogen-limited steady-state and dynamic transcriptome)": (
        "an upshift time course beside the steady states"
    ),
    "Martin-Perez 2017 (protein half-lives)": (
        "half-lives derived from a labeling time course; the rates are stored"
    ),
}


def _apply_curation() -> None:
    """Attach bands and synergies to the rows, failing loudly on a stale name.

    Every check here exists because the alternative is a silent defect: a band
    with no reason, a synergy naming a partner that was renamed or never ranked,
    or a partner claimed as built that is not in the supported table, which would
    have the document promise a join nobody can run.
    """
    by_name = {c.name: c for c in CANDIDATES}
    if len(by_name) != len(CANDIDATES):
        raise SystemExit("duplicate candidate name")
    for table, label in (
        (BANDS, "BANDS"),
        (SYNERGIES, "SYNERGIES"),
        (TIME_AXES, "TIME_AXES"),
    ):
        missing = sorted(set(table) - set(by_name))
        if missing:
            raise SystemExit(f"{label} names absent from CANDIDATES: {missing}")
    for name, syns in SYNERGIES.items():
        for s in syns:
            pool = by_name if s.partner_status == "candidate" else SUPPORTED_PARTNERS
            if s.partner not in pool:
                raise SystemExit(
                    f"{name}: partner {s.partner!r} is not a known "
                    f"{s.partner_status} dataset"
                )
    for name, (band, why) in BANDS.items():
        by_name[name].band = band
        by_name[name].band_why = why
    for name, syns in SYNERGIES.items():
        by_name[name].synergy = syns
    for name, axis in TIME_AXES.items():
        by_name[name].time_axis = axis
    for c in CANDIDATES:
        if c.band != "scale" and not c.band_why:
            raise SystemExit(f"{c.name}: banded out of scale with no reason")


# Names as they appear in build_supported_datasets_table.py, with Greek letters
# spelled out. Checked rather than trusted: a partner that is not actually built
# would make the synergy table promise a join that cannot be run.
SUPPORTED_PARTNERS: set[str] = {
    "Costanzo 2016 smf",
    "Costanzo 2016 dmf",
    "Costanzo 2016 dmi",
    "Kuzmin 2018 smf",
    "Kuzmin 2018 dmf",
    "Kuzmin 2018 tmf",
    "Kuzmin 2018 dmi",
    "Kuzmin 2018 tmi",
    "Kuzmin 2020 smf",
    "Kuzmin 2020 dmf",
    "Kuzmin 2020 tmf",
    "Kuzmin 2020 dmi",
    "Kuzmin 2020 tmi",
    "Baryshnikova 2010 (smf)",
    "O'Duibhir 2014 (smf)",
    "Auesukaree 2009 (stress screen)",
    "Mota 2024 (weak-acid screen)",
    "Vanacloig-Pedros 2022",
    "Costanzo 2021 (condition-SGA)",
    "Hillenmeyer 2008 het (FitDb HIP)",
    "Hillenmeyer 2008 hom (FitDb HOP)",
    "Wildenhain 2015 (drug tolerance)",
    "Hoepfner 2014 (HIP/HOP atlas)",
    "Smith 2006 (chemogenomic)",
    "Lian 2019 (MAGIC CRISPR-AID)",
    "Mormino 2022 (CRISPRi acetic-acid)",
    "Smith 2016 (CRISPRi chem-genetic)",
    "SGD essentiality",
    "SynLethDB (lethal)",
    "SynLethDB (rescue)",
    "Ohya 2005 (SCMD CalMorph)",
    "Ohnuki 2018 (SCMD CalMorph)",
    "Ohnuki 2022 (SCMD CalMorph)",
    "Kemmeren 2014",
    "Sameith 2015 sm",
    "Sameith 2015 dm",
    "Caudal 2024 (pan-transcriptome)",
    "Nadal-Ribelles 2025 (Perturb-seq)",
    "Bloom 2019 (16-parent cross)",
    "Cachera 2023 (CRI-SPA betaxanthin)",
    "Mulleder 2016 (amino-acid metabolome)",
    "Zelezniak 2018 (metabolome)",
    "Zelezniak 2018 (SWATH proteome)",
    "Messner 2023 (proteome)",
    "Ozaydin 2013 (beta-carotene screen)",
    "da Silveira 2014 (lipidomics)",
    "Yoshida 2012 (organic acids)",
    "Xue 2025 (free fatty acids, private)",
    "Lopez 2024 (isobutanol screen, private)",
    "Lopez 2024 (isobutanol validated, private)",
}

_apply_curation()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def link_tex(url: str) -> str:
    """A clickable, short display form of a source URL.

    The full URL is too wide for a table column and mostly boilerplate, so the
    common prefixes collapse to a label while the href keeps the real target.
    Break points go after every slash, since a bare host path is one unbreakable
    token to TeX.
    """
    shown = url
    for prefix, label in (
        ("https://doi.org/", "doi:"),
        ("https://pubmed.ncbi.nlm.nih.gov/", "PMID "),
        ("https://pmc.ncbi.nlm.nih.gov/articles/", ""),
        ("https://www.ncbi.nlm.nih.gov/pmc/articles/", ""),
        ("https://www.", ""),
        ("https://", ""),
    ):
        if shown.startswith(prefix):
            shown = label + shown[len(prefix) :]
            break
    shown = shown.rstrip("/")
    body = tex_escape(shown).replace("/", r"/\allowbreak ")
    return r"\href{" + url + r"}{" + body + r"}"


def status_tex(status: str) -> str:
    """Marker for a row that is not an untouched candidate."""
    return {
        "candidate": "",
        "blocked": r"\,\textsuperscript{\textbf{B}}",
        "loader-in-flight": r"\,\textsuperscript{\textbf{L}}",
    }[status]


def seq_tex(basis: str) -> str:
    """Sequence-basis label with break points at + and -.

    "S288C+designed-edit" is one unbreakable token to TeX, so a narrow column
    cannot wrap it and the row runs off the text block. Hyphenation does not help:
    the string is not a word.
    """
    return (
        tex_escape(basis).replace("+", r"+\allowbreak ").replace("-", r"-\allowbreak ")
    )


def sci(n: int | None) -> str:
    """Instances as an order-of-magnitude figure, the way the built table reports it."""
    if n is None:
        return "--"
    if n < 1000:
        return f"{n:,}"
    exp = int(math.floor(math.log10(n)))
    mant = n / 10**exp
    return f"${mant:.1f}\\times 10^{{{exp}}}$"


class Move(BaseModel):
    """One row's change of position between the previous pass and this one."""

    name: str
    old: int | None  # None when the row is new this pass
    new: int | None  # None when the row left the list
    reason: str


def previous_ranked() -> list[Candidate]:
    """The previous pass's order: no bands, and the rows it held.

    Reproduced rather than transcribed, which is why a row built since then keeps
    its record here under ``status="built"`` instead of being deleted. Deleting it
    would shift every rank below it by one and turn a single removal into a
    hundred spurious moves.
    """
    return sorted((c for c in CANDIDATES if not c.added), key=lambda c: c.scale_key)


def moves(rows: list[Candidate]) -> list[Move]:
    """Every position change worth reporting, with the reason for it.

    Bounded to what a reader acts on: the rows in either pass's working set, the
    rows added this pass, and the rows that left. A row that moved inside the
    reserve moved because the rows above it did, and reporting all of those buries
    the ones that were moved on purpose.
    """
    old_rank = {c.name: i for i, c in enumerate(previous_ranked(), 1)}
    new_rank = {c.name: i for i, c in enumerate(rows, 1)}
    out: list[Move] = []
    for c in CANDIDATES:
        o, n = old_rank.get(c.name), new_rank.get(c.name)
        if n is None:
            out.append(
                Move(name=c.name, old=o, new=None, reason=REMOVAL_REASON[c.name])
            )
            continue
        if o is None:
            out.append(
                Move(
                    name=c.name,
                    old=None,
                    new=n,
                    reason="New this pass. " + c.band_why
                    if c.band_why
                    else "New this pass.",
                )
            )
            continue
        # Reported when EITHER position is inside the working set, so a row that
        # climbed in from the reserve is named as well as one that fell out of it.
        if min(o, n) > WAVE_2 or o == n:
            continue
        out.append(
            Move(
                name=c.name,
                old=o,
                new=n,
                reason=c.band_why
                or (
                    f"Unchanged criteria; moved by {abs(o - n)} as rows around it were "
                    "promoted into a band."
                ),
            )
        )
    return sorted(out, key=lambda m: (m.new is None, m.new or 10**6))


def ranked() -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Rank by band, then by the tier rule, then pin requested rows above the cut.

    Scale ranking is the right default and a stakeholder priority is a real
    criterion, so both are applied and the pin is reported rather than absorbed
    into the score. Returns the ordered rows and the list of (pinned, displaced)
    swaps so the document can name every one.
    """
    rows = sorted(
        (c for c in CANDIDATES if c.status != "built"), key=lambda c: c.sort_key
    )
    swaps: list[tuple[str, str]] = []
    while True:
        below = [c for c in rows[CUT:] if c.requested]
        if not below:
            return rows, swaps
        promote = below[0]
        # Displace the weakest row in the recommended set that was not itself
        # requested: worst tier first, then fewest measurements. Ranking on
        # measurements alone would evict a tier-1 row with a small but dense
        # label -- it picked Puddu 2019, the whole-collection sequencing that
        # every S288C-KO row's sequence basis rests on -- to make room for a
        # tier-2 one.
        droppable = [c for c in rows[:CUT] if not c.requested]
        demote = min(droppable, key=lambda c: (-c.tier, c.measurements or 0))
        rows.remove(promote)
        rows.insert(rows.index(demote), promote)
        rows.remove(demote)
        rows.insert(CUT, demote)
        swaps.append((promote.name, demote.name))


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "%% GENERATED FILE -- do not hand-edit.\n" + SOURCE_LINE + "\n" + body
    )
    print(f"Wrote {path.relative_to(REPO)}")


def render_candidates(rows: list[Candidate]) -> str:
    # Landscape. Portrait gives 182 mm, which forced nine columns so narrow that
    # every free-text cell hyphenated two or three times and the table stopped
    # being scannable. lscape swaps \textwidth for \textheight, so the block is
    # 261 mm here, and every free-text column is ragged-right (L) rather than
    # justified: justification in a 30 mm column is what produced the stretched,
    # over-hyphenated lines.
    cols = (
        r"@{}r@{\hspace{4pt}} L{42mm} L{21mm} L{26mm} L{20mm} r@{\hspace{5pt}} "
        r"r@{\hspace{5pt}} L{30mm} L{23mm} L{34mm}@{}"
    )
    hdr = (
        r"\textbf{\#} & \textbf{Dataset} & \textbf{Class} & \textbf{Genotypes} & "
        r"\textbf{Env} & \textbf{Instances} & \textbf{Meas.} & \textbf{Phenotype} "
        r"& \textbf{Sequence basis} & \textbf{Link} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{Ranked candidates to take the database from """
        + str(BUILT_COUNT)
        + r""" supported datasets to """
        + str(TARGET_COUNT)
        + r""".
\emph{Genotypes} and \emph{Env} are the perturbation and condition axes. \emph{Instances}
is the number of genotype$\times$environment records, marked $\dagger$ where it is the
product of the two axes rather than a figure the paper reports, and $\ddagger$ where it is
an order-of-magnitude estimate. \emph{Meas.} is instances times phenotype dimensionality,
which is what a vector-valued panel actually contributes and what rows are ranked on.
\emph{Sequence basis} is how the total genomic content of one strain would be
reconstructed; a row with no such route is excluded rather than ranked
(Table~\ref{tab:excluded}). Tier is defined in Sec.~\ref{sec:rule}.
\emph{Link} resolves to the source and is clickable; the full citation and the data
location are in Table~\ref{tab:sources}. A $\bullet$ marks a row bearing on the
Perturb-seq proposal (Table~\ref{tab:perturbseq}). A superscript \textbf{B} marks a row already
attempted and blocked on data access, and \textbf{L} one that already has a loader in
flight; neither is an untouched candidate. Rows
1--"""
        + str(CUT)
        + r""" are the recommended set.}
\label{tab:candidates}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{10}{@{}l}{\footnotesize\emph{Table~\ref{tab:candidates}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    dividers = {
        WAVE_1 + 1: (
            r"End of wave 1. Rows 1--" + str(WAVE_1) + r" are the recommended next "
            r"builds."
        ),
        WAVE_2 + 1: (
            r"End of wave 2. Rows " + str(WAVE_1 + 1) + r"--" + str(WAVE_2) + r" are "
            r"the bench, promoted as soon as a wave-1 row proves unreachable."
        ),
        CUT + 1: (
            r"Long-run cut. Rows above are the "
            + str(CUT)
            + r" that reach "
            + str(TARGET_COUNT)
            + r"; rows below are the ranked reserve."
        ),
    }
    for i, c in enumerate(rows, start=1):
        if i in dividers:
            lines.append(
                r"\midrule \multicolumn{10}{@{}l}{\textbf{"
                + dividers[i]
                + r"}}\\ \midrule"
            )
        mark = {"reported": "", "product": r"$\dagger$", "estimate": r"$\ddagger$"}[
            c.instances_basis
        ]
        star = "" if c.perturbseq == "none" else r"\,$\bullet$"
        lines.append(
            " & ".join(
                [
                    str(i),
                    r"\textbf{"
                    + tex_escape(c.name)
                    + r"}"
                    + star
                    + status_tex(c.status),
                    tex_escape(c.klass),
                    tex_escape(c.genotypes),
                    tex_escape(c.env),
                    sci(c.instances_n) + mark,
                    sci(c.measurements),
                    tex_escape(c.phenotype),
                    seq_tex(c.seq_basis),
                    link_tex(c.url),
                ]
            )
            + r" \\"
        )
        # Same 5 pt gap the perturb-seq glossary uses, and for the same reason:
        # every row here is several lines of wrapped text, so without it adjacent
        # rows run together and the eye cannot find where one ends.
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_perturbseq(rows: list[Candidate]) -> str:
    """Rows bearing on a Perturb-seq, split by which axis they are high on.

    Scale ranking buries several of these, and rank is the wrong ordering anyway:
    what matters is whether a row is high-dimensional on the perturbation axis, on
    the readout axis, or on both.
    """
    order = [
        (
            "both",
            "High on BOTH axes: combinatorial or background-crossed perturbation "
            "AND a transcriptome-scale readout",
        ),
        ("input", "High-dimensional perturbation space, scalar readout"),
        (
            "output",
            "Transcriptome-scale or per-cell readout, low-dimensional perturbation",
        ),
    ]
    head = r"""\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{46mm} L{112mm}@{}}
\caption[]{Rows bearing on a yeast Perturb-seq, grouped by axis rather than by rank.
\emph{Rank} is the row's position in Table~\ref{tab:candidates}. A large library of
single perturbations read out by fitness is not high-dimensional input: it is one edit per
cell, sampled widely.}
\label{tab:perturbseq}\\
\toprule
\textbf{Rank} & \textbf{Dataset} & \textbf{What it supplies} \\
\midrule
\endfirsthead
\toprule
\textbf{Rank} & \textbf{Dataset} & \textbf{What it supplies} \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for key, label in order:
        members = [(i, c) for i, c in enumerate(rows, 1) if c.perturbseq == key]
        if not members:
            continue
        lines.append(r"\multicolumn{3}{@{}l}{\textbf{" + tex_escape(label) + r"}}\\")
        lines.append(r"\addlinespace[2pt]")
        for i, c in members:
            lines.append(
                f"{i} & \\textbf{{{tex_escape(c.name)}}} & {tex_escape(c.why)} \\\\"
            )
            lines.append(r"\addlinespace[5pt]")
        lines.append(r"\addlinespace[4pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_sources(rows: list[Candidate]) -> str:
    hdr = (
        r"\textbf{\#} & \textbf{Dataset, citation and link} & "
        r"\textbf{Why it is ordered here} & \textbf{Data} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{86mm} L{92mm} L{62mm}@{}}
\caption[]{Sources for Table~\ref{tab:final}, in the same order. \emph{Why it is
ordered here} is the band reason for a perturb-seq or molecular-layers row, and the tier
and measurement rule for a scale row; what the row buys is the \emph{Why} column of
Table~\ref{tab:final}. \emph{Data} is where the per-record values live; entries marked
unconfirmed were not fetched live and must be checked before a loader is written. Every
link is clickable.}
\label{tab:sources}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{4}{@{}l}{\footnotesize\emph{Table~\ref{tab:sources}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for i, c in enumerate(rows, start=1):
        cite = (
            r"\textbf{"
            + tex_escape(c.name)
            + r"}\newline "
            + tex_escape(c.citation)
            + r"\newline "
            + link_tex(c.url)
        )
        # Accessions carry bare host paths, which TeX treats as one unbreakable
        # token; allow a break after each slash so the column can wrap.
        acc = tex_escape(c.accession).replace("/", r"/\allowbreak ")
        lines.append(" & ".join([str(i), cite, priority_tex(c), acc]) + r" \\")
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_excluded() -> str:
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{62mm} L{22mm} L{85mm}@{}}
\caption[]{Considered and dropped. \emph{no-sequence} is the hard gate: without a route to
the strain's genomic content there is no genotype to map a phenotype from.}
\label{tab:excluded}\\
\toprule
Dataset or group & Rule & Reason \\
\midrule
\endfirsthead
\toprule
Dataset or group & Rule & Reason \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for e in EXCLUDED:
        lines.append(
            " & ".join([tex_escape(e.name), tex_escape(e.rule), tex_escape(e.reason)])
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_synergies(rows: list[Candidate]) -> str:
    """Every named join, candidate by candidate, in rank order.

    One row per pair rather than one per candidate: a pair is what gets run, and
    collapsing three pairs into one cell makes the join keys unreadable. The
    candidate name is printed once per group so the eye can still find it.
    """
    hdr = (
        r"\textbf{\#} & \textbf{Candidate} & \textbf{Partner} & "
        r"\textbf{Join key} & \textbf{What the join yields} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{42mm} L{44mm} L{48mm} L{96mm}@{}}
\caption[]{Named joins between a candidate and either a supported dataset or
another candidate, for the sixty rows of Table~\ref{tab:final}. \emph{\#} is the
candidate's rank there. A partner in \textbf{bold} is already built, so that
row's join becomes runnable the moment the candidate is ingested; an unbolded
partner is itself a candidate, so the join costs two ingestions. \emph{Join key}
is the shared axis that makes the pair addressable; a pair with no such axis is a
theme and is not listed. None of these transfers has been measured.}
\label{tab:synergies}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{5}{@{}l}{\footnotesize\emph{Table~\ref{tab:synergies}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for i, c in enumerate(rows, start=1):
        if not c.synergy:
            continue
        for j, s in enumerate(c.synergy):
            partner = tex_escape(s.partner)
            if s.partner_status == "supported":
                partner = r"\textbf{" + partner + r"}"
            lines.append(
                " & ".join(
                    [
                        str(i) if j == 0 else "",
                        tex_escape(c.name) if j == 0 else "",
                        partner,
                        tex_escape(s.join),
                        tex_escape(s.yields),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_moves(ms: list[Move]) -> str:
    """Every position change in either pass's working set, with its reason."""
    hdr = r"\textbf{Dataset} & \textbf{Was} & \textbf{Now} & \textbf{Reason} \\"
    head = (
        r"""\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}L{46mm} r@{\hspace{6pt}} r@{\hspace{6pt}} L{104mm}@{}}
\caption[]{Rank changes between the previous pass and this one, for every row in
either pass's first """
        + str(WAVE_2)
        + r""", every row added, and every row that left.
\emph{Was} is recomputed under the previous rule, tier then measurements, over the
rows that pass held. A dash means the row is new or has gone.}
\label{tab:swaps}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for m in ms:
        lines.append(
            " & ".join(
                [
                    r"\textbf{" + tex_escape(m.name) + r"}",
                    "--" if m.old is None else str(m.old),
                    "--" if m.new is None else str(m.new),
                    tex_escape(m.reason),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_swaps(swaps: list[tuple[str, str]]) -> str:
    """Every pin, and what it cost, so the recommended set is auditable."""
    if not swaps:
        return "%% no pins were needed\n"
    head = r"""\begin{table}[H]\centering
\small
\caption[]{Rows pinned into the recommended set because they were named explicitly in
the scoping request, and the row each displaced. Displacement picks the weakest
non-requested row by tier first and measurement count second, so a pin costs the least
it can and cannot evict a tier-1 row to make room for a lower one.}
\label{tab:pins}
\begin{tabular}{@{}L{78mm} L{78mm}@{}}
\toprule
Pinned in & Displaced to the reserve \\
\midrule
"""
    lines = [f"{tex_escape(a)} & {tex_escape(b)} \\\\" for a, b in swaps]
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def render_counts(rows: list[Candidate]) -> str:
    """Class by wave, and band by wave, off the same ordering.

    Two blocks in one table because the question they answer is the same one:
    what the next fifty builds are made of. Class says what phenotype arrives;
    band says why those rows are first.
    """
    klasses = [
        "Natural variation",
        "Tolerance / robustness",
        "CRISPR library screen",
        "Expression / single cell",
        "Metabolite / precursor",
        "Modality / backbone",
        "Regulatory DNA",
        "Deep mutational scan",
        "Combinatorial genome",
    ]
    head = r"""\begin{table}[H]\centering
\small
\caption[]{Candidates by class and by band, split at the two wave lines.
\emph{Genotypes} and \emph{Instances} sum the per-row axes over all waves; rows with
no count contribute nothing, so both totals are lower bounds.}
\label{tab:counts}
\begin{tabular}{@{}l r r r r r@{}}
\toprule
 & Wave 1 & Wave 2 & Reserve & Genotypes & Instances \\
\midrule
"""

    def block(key: str, values: list[str]) -> list[str]:
        out = []
        for v in values:
            members = [(i, c) for i, c in enumerate(rows, 1) if getattr(c, key) == v]
            w1 = [c for i, c in members if i <= WAVE_1]
            w2 = [c for i, c in members if WAVE_1 < i <= WAVE_2]
            rest = [c for i, c in members if i > WAVE_2]
            g = sum(c.genotypes_n or 0 for _i, c in members)
            n = sum(c.instances_n or 0 for _i, c in members)
            out.append(
                f"{tex_escape(v)} & {len(w1)} & {len(w2)} & {len(rest)} & "
                f"{g:,} & {sci(n)} \\\\"
            )
        return out

    lines = block("klass", klasses)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{@{}l}{\emph{The same rows, by band}}\\")
    lines += block("band", list(BAND_ORDER))
    g_all = sum(c.genotypes_n or 0 for c in rows)
    n_all = sum(c.instances_n or 0 for c in rows)
    lines.append(r"\midrule")
    lines.append(
        f"Total & {WAVE_1} & {WAVE_2 - WAVE_1} & {len(rows) - WAVE_2} & "
        f"{g_all:,} & {sci(n_all)} \\\\"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def priority_tex(c: Candidate) -> str:
    """Why the row sits where it does: the band reason, or the scale rule."""
    if c.band != "scale":
        return tex_escape(c.band_why)
    return tex_escape(
        f"Scale band: tier {c.tier}, ordered by measurements ({c.measurements:,})."
        if c.measurements
        else f"Scale band: tier {c.tier}."
    )


def render_final(rows: list[Candidate]) -> str:
    """The final table: the recommended fifty and ten extra, every stat and the reason.

    One table rather than a stats table plus a reasons table, because the reader
    asked for the reason beside the numbers. The dataset cell carries the class
    and the phenotype on a second line so the column count stays at ten and the
    reason column keeps 81 mm.
    """
    final = rows[:FINAL]
    cols = (
        r"@{}r@{\hspace{3pt}} L{40mm} L{11mm} L{20mm} L{18mm} r@{\hspace{4pt}} "
        r"r@{\hspace{4pt}} L{19mm} L{14mm} L{81mm}@{}"
    )
    hdr = (
        r"\textbf{\#} & \textbf{Dataset (class; phenotype)} & \textbf{Band, tier} & "
        r"\textbf{Genotypes} & \textbf{Env} & \textbf{Inst.} & \textbf{Meas.} & "
        r"\textbf{Sequence basis} & \textbf{Time} & \textbf{Why} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{The final fifty, then ten extra in rank order in case a row above proves
unreachable. Every stat the ranking used is here beside the reason. \emph{Band, tier}:
\emph{P} is the perturb-seq band, \emph{M} molecular layers, \emph{S} scale, applied before
tier (Sec.~\ref{sec:rule}); within a band the tier rule orders, then measurements.
\emph{Genotypes} and \emph{Env} are the perturbation and condition axes. \emph{Inst.} is
genotype$\times$environment records, $\dagger$ where it is the product of the two axes
rather than a reported count and $\ddagger$ where it is an order-of-magnitude estimate.
\emph{Meas.} is instances times phenotype dimensionality, the quantity rows are ranked on.
\emph{Sequence basis} is the route to each strain's total genomic content; a row with no
route is excluded (Table~\ref{tab:excluded}). \emph{Time} names the row's time dimension
where it has one; a dash means steady state or endpoint. A $\bullet$ marks a row high on a
Perturb-seq axis; superscript \textbf{B} marks a row blocked on data access, \textbf{L} one
with a loader in flight. Citations, links and data locations are in
Table~\ref{tab:sources}; joins in Table~\ref{tab:synergies}.}
\label{tab:final}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{10}{@{}l}{\footnotesize\emph{Table~\ref{tab:final}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    band_letter = {"perturb-seq": "P", "molecular layers": "M", "scale": "S"}
    lines = []
    for i, c in enumerate(final, start=1):
        if i == WAVE_1 + 1:
            lines.append(
                r"\midrule \multicolumn{10}{@{}l}{\textbf{Ten extra, in rank order: "
                r"rows "
                + str(WAVE_1 + 1)
                + r"--"
                + str(FINAL)
                + r" replace a row above "
                r"that proves unreachable.}}\\ \midrule"
            )
        mark = {"reported": "", "product": r"$\dagger$", "estimate": r"$\ddagger$"}[
            c.instances_basis
        ]
        star = "" if c.perturbseq == "none" else r"\,$\bullet$"
        dataset = (
            r"\textbf{"
            + tex_escape(c.name)
            + r"}"
            + star
            + status_tex(c.status)
            + r"\newline {\scriptsize "
            + tex_escape(c.klass)
            + "; "
            + tex_escape(c.phenotype)
            + "}"
        )
        lines.append(
            " & ".join(
                [
                    str(i),
                    dataset,
                    band_letter[c.band] + ", " + str(c.tier),
                    tex_escape(c.genotypes),
                    tex_escape(c.env),
                    sci(c.instances_n) + mark,
                    sci(c.measurements),
                    seq_tex(c.seq_basis),
                    tex_escape(c.time_axis) if c.time_axis else "--",
                    tex_escape(c.why),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_summary(rows: list[Candidate]) -> str:
    """Summary statistics of the final sixty, split at the fifty line.

    Every figure here is computed from the same rows the final table prints, so
    the two cannot disagree. Genotype and instance sums are lower bounds: a row
    with no count contributes nothing.
    """
    final = rows[:FINAL]
    top = final[:WAVE_1]
    extra = final[WAVE_1:]

    def count(pred: Any) -> str:
        a = sum(1 for c in top if pred(c))
        b = sum(1 for c in extra if pred(c))
        g = sum(c.genotypes_n or 0 for c in final if pred(c))
        n = sum(c.instances_n or 0 for c in final if pred(c))
        m = sum(c.measurements or 0 for c in final if pred(c))
        return f"{a} & {b} & {g:,} & {sci(n)} & {sci(m)} \\\\"

    def block(title: str, items: list[tuple[str, Any]]) -> list[str]:
        out = [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\"]
        for label, pred in items:
            if sum(1 for c in final if pred(c)) == 0:
                continue
            out.append(tex_escape(label) + " & " + count(pred))
        return out

    klasses = sorted({c.klass for c in final})
    bases = sorted({c.seq_basis for c in final})
    lines: list[str] = []
    lines += block("By band", [(b, (lambda c, b=b: c.band == b)) for b in BAND_ORDER])
    lines += block(
        "By tier", [(f"tier {t}", (lambda c, t=t: c.tier == t)) for t in (1, 2, 3, 4)]
    )
    lines += block("By class", [(k, (lambda c, k=k: c.klass == k)) for k in klasses])
    lines += block(
        "By sequence basis", [(b, (lambda c, b=b: c.seq_basis == b)) for b in bases]
    )
    lines += block(
        "Perturb-seq axis",
        [
            ("high on both axes", lambda c: c.perturbseq == "both"),
            ("input axis only", lambda c: c.perturbseq == "input"),
            ("output axis only", lambda c: c.perturbseq == "output"),
            ("neither axis", lambda c: c.perturbseq == "none"),
        ],
    )
    lines += block(
        "Other attributes",
        [
            ("carries a time axis", lambda c: bool(c.time_axis)),
            (
                "has a join to a built dataset",
                lambda c: any(s.partner_status == "supported" for s in c.synergy),
            ),
            (
                "has a join to another candidate only",
                lambda c: (
                    bool(c.synergy)
                    and not any(s.partner_status == "supported" for s in c.synergy)
                ),
            ),
            ("no named join", lambda c: not c.synergy),
            ("figures sourced this pass", lambda c: c.confidence == "sourced"),
            ("figures from recall, to confirm", lambda c: c.confidence == "recall"),
            ("blocked on data access", lambda c: c.status == "blocked"),
            ("loader in flight", lambda c: c.status == "loader-in-flight"),
            ("instances a reported count", lambda c: c.instances_basis == "reported"),
            (
                "instances a product of the axes",
                lambda c: c.instances_basis == "product",
            ),
            ("instances an estimate", lambda c: c.instances_basis == "estimate"),
        ],
    )
    lines.append(r"\midrule")
    lines.append("Total & " + count(lambda c: True))
    n_joins = sum(len(c.synergy) for c in final)
    n_built_joins = sum(
        1 for c in final for s in c.synergy if s.partner_status == "supported"
    )
    head = (
        r"""\begin{table}[H]\centering
\small
\caption[]{Summary of the final sixty, split at the fifty line. \emph{Top 50} and
\emph{Extra} count rows; \emph{Genotypes}, \emph{Instances} and \emph{Meas.} sum the row
axes over all sixty and are lower bounds, since a row with no count contributes nothing.
The sixty rows name """
        + str(n_joins)
        + r""" joins, """
        + str(n_built_joins)
        + r""" of them to a dataset already built.}
\label{tab:summary}
\begin{tabular}{@{}l r r r r r@{}}
\toprule
 & Top 50 & Extra & Genotypes & Instances & Meas. \\
"""
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def main() -> None:
    rows, swaps = ranked()
    if len(rows) < CUT:
        raise SystemExit(f"only {len(rows)} candidates; need at least {CUT}")
    if len(rows) > 250:
        raise SystemExit(f"{len(rows)} candidates exceeds the 250-row ceiling")
    ms = moves(rows)

    write(TEX_DIR / "candidates.tex", render_candidates(rows))
    write(TEX_DIR / "final.tex", render_final(rows))
    write(TEX_DIR / "summary.tex", render_summary(rows))
    write(TEX_DIR / "sources.tex", render_sources(rows[:FINAL]))
    write(TEX_DIR / "perturbseq.tex", render_perturbseq(rows))
    write(TEX_DIR / "synergies.tex", render_synergies(rows[:FINAL]))
    write(TEX_DIR / "excluded.tex", render_excluded())
    write(TEX_DIR / "counts.tex", render_counts(rows))
    write(TEX_DIR / "swaps.tex", render_moves(ms))
    write(TEX_DIR / "pins.tex", render_swaps(swaps))

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(
            {
                "built_count": BUILT_COUNT,
                "target_count": TARGET_COUNT,
                "cut": CUT,
                "wave_1": WAVE_1,
                "wave_2": WAVE_2,
                "final": FINAL,
                "n_candidates": len(rows),
                "pinned_swaps": swaps,
                "moves": [m.model_dump() for m in ms],
                "candidates": [c.model_dump() for c in rows],
                "excluded": [e.model_dump() for e in EXCLUDED],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {JSON_OUT.relative_to(REPO)}")
    for a, b in swaps:
        print(f"pinned {a!r} above the cut, displacing {b!r}")
    n_syn = sum(len(c.synergy) for c in rows)
    n_band = sum(c.band != "scale" for c in rows)
    print(
        f"{len(rows)} candidates; wave 1 = {WAVE_1}, wave 2 ends at {WAVE_2}, "
        f"top {CUT} reach {TARGET_COUNT}"
    )
    print(f"{n_band} rows banded out of scale; {n_syn} named joins; {len(ms)} moves")


if __name__ == "__main__":
    main()
