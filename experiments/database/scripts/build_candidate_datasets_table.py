# experiments/database/scripts/build_candidate_datasets_table.py
# [[experiments.database.expansion-100]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_candidate_datasets_table
r"""The candidate list for taking the database from 49 supported datasets to 100.

This is the CURATION, held as data. Unlike ``build_supported_datasets_table.py``,
which measures built LMDBs, nothing here can be recomputed from a store: the rows
are a judgment about which *S. cerevisiae* datasets to ingest next, so the
judgment itself is the artifact and it lives in a committed script rather than a
gitignored results file.

Emits, off the same records:
  - notes-tex/database-expansion-100/tables/candidates.tex   (triage table)
  - notes-tex/database-expansion-100/tables/sources.tex      (citation + link + accession)
  - notes-tex/database-expansion-100/tables/excluded.tex     (what was dropped, and why)
  - notes-tex/database-expansion-100/tables/counts.tex       (per-class totals)
  - <results>/candidates/candidate_datasets.json             (machine-readable dump)

Run from the repo root:
  python experiments/database/scripts/build_candidate_datasets_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
TEX_DIR = REPO / "notes-tex" / "database-expansion-100" / "tables"
JSON_OUT = RESULTS / "candidates" / "candidate_datasets.json"

SOURCE_LINE = (
    "%% SOURCE: experiments/database/scripts/build_candidate_datasets_table.py"
)

# The database currently holds 49 schematized + L0-L4-verified datasets. The goal
# is 200, so the recommended set is exactly the first 151 rows; everything after is
# a ranked reserve bench, kept because the cut line moves whenever one of the 151
# turns out to have no recoverable per-strain data.
BUILT_COUNT = 49
TARGET_COUNT = 200
CUT = TARGET_COUNT - BUILT_COUNT  # 151

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

# Ingestion state. "candidate" means untouched. The other two exist because this
# pass initially ranked a blocked dataset first and a half-built one twentieth:
# scanning the built list is not enough, since a dataset can have a loader, or a
# failed retrieval attempt behind it, without appearing there.
Status = Literal["candidate", "blocked", "loader-in-flight"]

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
    requested: bool = False  # named explicitly in the scoping request; pinned above the cut
    status: Status = "candidate"
    confidence: Confidence = "sourced"

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
    def sort_key(self) -> tuple[int, float]:
        return (self.tier, -math.log10(max(self.measurements or 1, 1)))


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
        why="Largest sequenced recombinant panel in yeast. Sixteen founders means allelic diversity a two-parent cross cannot reach, and the genotypes are sequence, not markers.",
        accession="eLife SI + SRA (confirm accession before ingestion)",
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
        citation="Boocock J, Alexander N, Tsai L, Sadhu M, Day L, Kruglyak L. 2025.",
        url="https://doi.org/10.1038/s41586-025-09628-1",
        klass="Expression / single cell",
        tier=1,
        genotypes_n=27000,
        genotypes="~2.7e4 profiled segregants",
        env_n=2,
        env="glucose / galactose",
        instances_n=54000,
        instances_basis="estimate",
        phenotype="single-cell transcriptome + eQTL",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="segregant-WGS",
        why="Single-cell transcriptomes across tens of thousands of sequenced recombinants in two carbon sources. High-dimensional on both axes at once, which almost nothing in yeast is, and the closest existing template for the proposed design.",
        accession="mirrored as boocockSinglecellEQTLMapping2025; GEO (unconfirmed)",
        perturbseq="both",
        confidence="recall",
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
        genotypes_n=72,
        genotypes="72 TF deletions + wild type",
        env_n=2,
        env="YPD / minimal",
        instances_n=144,
        instances_basis="product",
        phenotype="single-cell transcriptome",
        shape="vector (~6,000)",
        dim=6000,
        seq_basis="S288C-KO",
        why="Single-cell transcriptomes across a panel of regulator deletions in two media, used to learn a regulatory network. Few genotypes, but it is one of the only yeast datasets crossing a genetic perturbation with a per-cell transcriptome, which is the quadrant the Perturb-seq proposal targets.",
        accession="GEO GSE125162 / GSE125163 (unconfirmed)",
        perturbseq="output",
        confidence="recall",
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
]

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
        name="Sc2.0 / Sc3.0 synthetic-genome design",
        reason="Genome-engineering programs, not gene-indexed perturbation-phenotype data.",
        rule="not-a-dataset",
    ),
    Excluded(
        name="Lee 2014 / Hoepfner 2014 / Vanacloig-Pedros 2022 / Messner 2023 / Mulleder 2016 / Lian 2019 / Mormino 2022 / Nadal-Ribelles 2025 and 12 others",
        reason="Named as top candidates by the 2026-07 triage pass and BUILT since. Only Lee 2014 remains outstanding and is kept as row 1.",
        rule="already-built",
    ),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    for a, b in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
    ]:
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
    return tex_escape(basis).replace("+", r"+\allowbreak ").replace("-", r"-\allowbreak ")


def sci(n: int | None) -> str:
    """Instances as an order-of-magnitude figure, the way the built table reports it."""
    if n is None:
        return "--"
    if n < 1000:
        return f"{n:,}"
    exp = int(math.floor(math.log10(n)))
    mant = n / 10**exp
    return f"${mant:.1f}\\times 10^{{{exp}}}$"


def ranked() -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Rank by the tier rule, then pin explicitly requested rows above the cut.

    Scale ranking is the right default and a stakeholder priority is a real
    criterion, so both are applied and the pin is reported rather than absorbed
    into the score. Returns the ordered rows and the list of (pinned, displaced)
    swaps so the document can name every one.
    """
    rows = sorted(CANDIDATES, key=lambda c: c.sort_key)
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
    path.write_text("%% GENERATED FILE -- do not hand-edit.\n" + SOURCE_LINE + "\n" + body)
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
    for i, c in enumerate(rows, start=1):
        if i == CUT + 1:
            lines.append(
                r"\midrule \multicolumn{10}{@{}l}{\textbf{Cut line. Rows above are the "
                + str(CUT)
                + r" that reach "
                + str(TARGET_COUNT)
                + r"; rows below are the ranked reserve.}}\\ \midrule"
            )
        mark = {"reported": "", "product": r"$\dagger$", "estimate": r"$\ddagger$"}[
            c.instances_basis
        ]
        star = "" if c.perturbseq == "none" else r"\,$\bullet$"
        lines.append(
            " & ".join(
                [
                    str(i),
                    r"\textbf{" + tex_escape(c.name) + r"}" + star + status_tex(c.status),
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
        head
        + "\n".join(lines)
        + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_perturbseq(rows: list[Candidate]) -> str:
    """Rows bearing on a Perturb-seq, split by which axis they are high on.

    Scale ranking buries several of these, and rank is the wrong ordering anyway:
    what matters is whether a row is high-dimensional on the perturbation axis, on
    the readout axis, or on both.
    """
    order = [
        ("both", "High on BOTH axes: combinatorial or background-crossed perturbation "
                 "AND a transcriptome-scale readout"),
        ("input", "High-dimensional perturbation space, scalar readout"),
        ("output", "Transcriptome-scale or per-cell readout, low-dimensional perturbation"),
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
        r"\textbf{\#} & \textbf{Dataset, citation and link} & \textbf{Why} & "
        r"\textbf{Data} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{86mm} L{92mm} L{62mm}@{}}
\caption[]{Sources for Table~\ref{tab:candidates}, in the same order. \emph{Why} states
what the row buys that the built set does not. \emph{Data} is where the per-record values
live; entries marked unconfirmed were not fetched live and must be checked before a loader
is written. Every link is clickable.}
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
        lines.append(" & ".join([str(i), cite, tex_escape(c.why), acc]) + r" \\")
        lines.append(r"\addlinespace[5pt]")
    return (
        head
        + "\n".join(lines)
        + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
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
\label{tab:swaps}
\begin{tabular}{@{}L{78mm} L{78mm}@{}}
\toprule
Pinned in & Displaced to the reserve \\
\midrule
"""
    lines = [f"{tex_escape(a)} & {tex_escape(b)} \\\\" for a, b in swaps]
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def render_counts(rows: list[Candidate]) -> str:
    order = [
        "Natural variation",
        "Tolerance / robustness",
        "CRISPR library screen",
        "Expression / single cell",
        "Metabolite / precursor",
        "Modality / backbone",
    ]
    head = r"""\begin{table}[H]\centering
\small
\caption[]{Candidates by class, split at the cut line. \emph{Genotypes} and \emph{Instances}
sum the per-row axes; rows with no count contribute nothing, so both totals are lower bounds.}
\label{tab:counts}
\begin{tabular}{@{}l r r r r@{}}
\toprule
Class & In top """ + str(CUT) + r""" & Reserve & Genotypes & Instances \\
\midrule
"""
    lines = []
    for k in order:
        top = [c for i, c in enumerate(rows, 1) if c.klass == k and i <= CUT]
        rest = [c for i, c in enumerate(rows, 1) if c.klass == k and i > CUT]
        g = sum(c.genotypes_n or 0 for c in top + rest)
        n = sum(c.instances_n or 0 for c in top + rest)
        lines.append(
            f"{tex_escape(k)} & {len(top)} & {len(rest)} & {g:,} & {sci(n)} \\\\"
        )
    g_all = sum(c.genotypes_n or 0 for c in rows)
    n_all = sum(c.instances_n or 0 for c in rows)
    lines.append(r"\midrule")
    lines.append(
        f"Total & {CUT} & {len(rows) - CUT} & {g_all:,} & {sci(n_all)} \\\\"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def main() -> None:
    rows, swaps = ranked()
    if len(rows) < CUT:
        raise SystemExit(f"only {len(rows)} candidates; need at least {CUT}")
    if len(rows) > 250:
        raise SystemExit(f"{len(rows)} candidates exceeds the 250-row ceiling")

    write(TEX_DIR / "candidates.tex", render_candidates(rows))
    write(TEX_DIR / "sources.tex", render_sources(rows))
    write(TEX_DIR / "perturbseq.tex", render_perturbseq(rows))
    write(TEX_DIR / "excluded.tex", render_excluded())
    write(TEX_DIR / "counts.tex", render_counts(rows))
    write(TEX_DIR / "swaps.tex", render_swaps(swaps))

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(
            {
                "built_count": BUILT_COUNT,
                "target_count": TARGET_COUNT,
                "cut": CUT,
                "n_candidates": len(rows),
                "pinned_swaps": swaps,
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
    print(f"{len(rows)} candidates; top {CUT} reach {TARGET_COUNT}")


if __name__ == "__main__":
    main()
