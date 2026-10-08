# experiments/database/scripts/build_bacteria_candidate_datasets_table.py
# [[experiments.database.expansion-bacteria]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_bacteria_candidate_datasets_table
r"""The bacterial candidate list: *E. coli* and *P. putida* datasets to ingest next.

The sibling of ``build_candidate_datasets_table.py``, which ranks the next fifty
*S. cerevisiae* datasets. This is the CURATION for the two bacterial hosts, held as
data: nothing here can be recomputed from a store, so the judgment itself is the
artifact and it lives in a committed script.

ORDERING RULE, and it differs from the yeast list on purpose. The yeast list bands
by what a row supplies to a planned Perturb-seq campaign, then ranks inside a band.
Here the requested rule is scale and density directly: rows sort by MEASUREMENTS,
meaning instances times phenotype dimensionality, descending. That single quantity
is what "most instances and highest density" resolves to, because a scalar screen
of n records contributes n numbers while a proteome panel of n records contributes
n times the panel depth. Instances break a measurement tie and tier breaks that,
so the rule is total and reproducible.

STAGING. The build is planned in two tranches, 20 rows then 30:
  tranche 1  the first 20, with a per-host FLOOR of six rows (``QUOTA_1``), so
             neither host waits behind the other's larger screens
  tranche 2  the next 30 by the same global rule
The floor is applied AFTER ranking and is reported row by row, so its effect on the
order is auditable rather than folded into a score.

THE ISOPRENOL PIN, and it overrides the ordering rule on purpose. Isoprenol
(3-methyl-3-buten-1-ol) is the product the strain-design work targets, and the
measurement rule above buries it: a titer campaign of fifty strains yields tens to
thousands of numbers while a barcoded fitness screen yields 10^6 to 10^8, so every
isoprenol paper sorts below every screen no matter how central it is. Ranking alone
put two of the four known *P. putida* isoprenol campaigns in the reserve, outside the
fifty recommended builds. ``PIN_DIRECT`` therefore lifts rows that measure isoprenol
itself into tranche 1 and rows that measure another isoprenoid above the cut, and
every lift is reported as a move with its before and after rank. A pin is the honest
mechanism because the alternative is a hand-tuned weight that would hide the same
decision inside a score.

Emits, off the same records:
  - notes-tex/database/database-expansion-bacteria/tables/final.tex     (the ranked list)
  - notes-tex/database/database-expansion-bacteria/tables/sources.tex   (citation, link, data; the fifty)
  - notes-tex/database/database-expansion-bacteria/tables/counts.tex    (per-class, per-organism)
  - notes-tex/database/database-expansion-bacteria/tables/summary.tex   (summary statistics)
  - notes-tex/database/database-expansion-bacteria/tables/analogs.tex   (yeast analog; the fifty)
  - notes-tex/database/database-expansion-bacteria/tables/schema.tex    (what the schema needs)
  - notes-tex/database/database-expansion-bacteria/tables/excluded.tex  (considered and dropped)
  - notes-tex/database/database-expansion-bacteria/tables/pins.tex      (isoprenol rows lifted)
  - <results>/candidates/bacteria_candidate_datasets.json                (machine-readable dump)

The 300-publication sweep behind this list is a separate artifact with a separate
script, ``build_bacteria_discovery_queue.py``: its rows carry no verified counts and
must not be confused with these. That script imports ``ranked()`` from here, so the
fifty it excludes from its row table are always the fifty this script prints.

Run from the repo root:
  python experiments/database/scripts/build_bacteria_candidate_datasets_table.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
TEX_DIR = REPO / "notes-tex" / "database" / "database-expansion-bacteria" / "tables"
JSON_OUT = RESULTS / "candidates" / "bacteria_candidate_datasets.json"

SOURCE_LINE = (
    "%% SOURCE: experiments/database/scripts/build_bacteria_candidate_datasets_table.py"
)

# The database holds 51 schematized and L0-L4-verified S. cerevisiae datasets and no
# bacterial dataset at all. This list is the bacterial tranche: a target of 50 rows,
# built in two waves.
YEAST_BUILT = 51
BACTERIA_BUILT = 0
TARGET_COUNT = 50

# The two tranches the build week is planned against.
TRANCHE_1 = 20
TRANCHE_2 = 50
# Per-host FLOOR inside tranche 1, not a split. The point is that neither host is
# absent from the first tranche, because the identifier namespace and the reference
# release are per-host costs that are only paid once a row for that host is built.
# A floor rather than an even split because an even split would fight the ordering
# rule: the two literatures are not the same size, and forcing ten rows per host
# promotes rows of a few hundred measurements over rows of a few hundred thousand.
# Six is a quarter of the tranche, enough that a host's per-host cost is paid early.
# It is reported whether or not it binds. On the current rows it does NOT bind: the
# unaided first twenty already holds six P. putida rows, so the printed order is the
# measurement order throughout and the floor is a guard that was not needed. An even
# ten-and-ten split WOULD have bound, promoting four rows of a few hundred
# measurements over rows of a few hundred thousand, which is what the floor avoids.
QUOTA_1 = {"E. coli": 6, "P. putida": 6}

# The isoprenol pin. A row that measures isoprenol itself is lifted into tranche 1; a
# row that measures another isoprenoid, so carries the precursor pathway without the
# product, is lifted above the cut at TRANCHE_2. Both are reported as moves.
PIN_DIRECT = TRANCHE_1
PIN_PRECURSOR = TRANCHE_2

# The two row groups Sec. 1.2 of the document compares: rows that connect an
# engineering intervention to a production phenotype, and the functional-genomics
# screens that outrank them on measurements.
ENGINEERING_CLASSES = ("Production campaign", "Combinatorial design", "Tolerance / robustness")
SCREEN_CLASSES = ("Transposon fitness", "Fitness / chemical genomics")

# ---------------------------------------------------------------------------
# Vocabulary. Defined here so it is defined before use in the document, and so a
# typo becomes a validation error rather than a silently novel category.
# ---------------------------------------------------------------------------

Organism = Literal["E. coli", "P. putida"]

Klass = Literal[
    "Fitness / chemical genomics",
    "Transposon fitness",
    "CRISPR library screen",
    "Genetic interaction",
    "Production campaign",
    "Combinatorial design",
    "Tolerance / robustness",
    "Transcriptome",
    "Proteome",
    "Metabolome / flux",
    "Translation / turnover",
    "Multi-omics campaign",
    "Modality / backbone",
    "Aggregation / support",
    # Single-cell shape under perturbation, the bacterial counterpart of the built
    # yeast CalMorph data. Separate from Transcriptome because the measured unit is a
    # segmented cell rather than a culture.
    "Morphology / imaging",
]

# How the total genomic content of one strain would be reconstructed. The same hard
# gate the yeast list applies: a row with no route to a sequence cannot train a
# genotype-to-phenotype model and is excluded rather than ranked low. The bases are
# named per host because the reference differs, and because a b-number and a PP_
# locus tag are different namespaces that must not be silently merged.
# The E. coli bases say "K-12" rather than naming one strain, and that is a deliberate
# admission of what this pass did not resolve. The Keio deletion collection is in
# BW25113, the CRISPRi libraries are mostly MG1655, and the two are different genomes:
# BW25113 carries defined lac, ara and rha lesions among others. Calling a Keio row
# "MG1655-KO" would be a silent sequence-level error in a substrate whose whole claim is
# sequence-level genotype fidelity, so the base names the K-12 lineage and the exact
# background is a per-row provenance item recorded when the loader is written.
SeqBasis = Literal[
    "K-12-KO",  # cataloged single deletion in a sequenced K-12 background
    "K-12-KO x KO",  # a constructed double mutant; both loci cataloged
    "K-12+transposon",  # mapped insertion site, barcoded
    "K-12+guide",  # genome unedited; the perturbation is a cassette plus a guide
    # A cataloged deletion carrying a guide plasmid, which is the basis a measured
    # gene PAIR sits on when one member is deleted and the other knocked down. The
    # two halves resolve to different namespaces, a Keio allele and a 20-nt spacer,
    # and the pair is only addressable because both are cataloged.
    "K-12-KO+guide",
    # A cataloged single-ORF overexpression clone, the ASKA collection. The genome is
    # unedited and the perturbation is extra copies of one native ORF on a plasmid,
    # which is the opposite sign to a deletion and must not be merged with one.
    "K-12+ORF-plasmid",
    # The reference genome plus a low-copy plasmid carrying one native promoter driving
    # a fluorescent protein. Nothing is perturbed: the plasmid reports rather than acts,
    # so the readout is expression and not fitness. It is still a distinct basis from
    # "reference-only" because the plasmid is present in the cell, and a genotype here
    # is the total genomic content, which includes the plasmids that are there.
    "K-12+promoter-reporter",
    "KT2440-KO",
    "KT2440+transposon",
    "KT2440+guide",
    "KT2440+promoter",  # reference plus a designed promoter cassette
    "engineered-chassis",  # named production strain plus heterologous cassettes
    "engineered-chassis+RBS",  # the chassis plus a designed ribosome binding site
    "engineered-chassis+promoter",
    # The chassis plus a guide array, which is the basis a CRISPRi production campaign
    # sits on: the pathway cassettes are heterologous and integrated or plasmid-borne,
    # and the perturbation on top of them leaves the genome unedited. Distinct from
    # "K-12+guide", where the host is the bare reference.
    "engineered-chassis+guide",
    "evolved-WGS",  # a resequenced evolved clone; unsequenced populations are excluded
    "reference-only",  # wild type; the environment carries the perturbation
]

Basis = Literal["reported", "product", "estimate"]

# Where a row sits on the isoprenol axis, which is what the pin reads. The division is
# by the ROLE the molecule plays in the experiment, because that is what decides whether
# a row teaches production or teaches survival, and the two are different labels.
#   direct      isoprenol or isopentenol is the measured PRODUCT, meaning a titer. The
#               two names are the same molecule, 3-methyl-3-buten-1-ol; "isopentenol"
#               is the older usage in the JBEI papers and "isoprenol" the current one,
#               so a row is direct under either word. Lifted into tranche 1.
#   tolerance   isoprenol is the STRESSOR and the readout is growth, with no titer. A
#               row here still measures the molecule of interest, so it is lifted into
#               tranche 1 alongside the production rows. A row reporting both a titer
#               and a tolerance assay is "direct", because the titer is the stronger
#               label and a row carries one value.
#   precursor   the measured product is a different isoprenoid or terpenoid, so the row
#               carries the mevalonate or MEP pathway that feeds isoprenol without
#               carrying the product. Lycopene, pinene, mevalonate and geranic acid are
#               here. Lifted above the cut, not into tranche 1.
#   analog      the stressor is a short-chain alcohol or a biomass inhibitor standing in
#               for isoprenol rather than isoprenol itself. Reported on the axis and NOT
#               lifted: the substitution is an assumption about cross-tolerance, and a
#               row should not be promoted on an assumption.
#   none        no isoprenol relevance; the row is in the table on scale alone.
Isoprenol = Literal["direct", "tolerance", "precursor", "analog", "none"]

# Ingestion state. Every row here starts as a candidate because no bacterial dataset
# is built, but two other states matter: a corpus that re-serves other papers is an
# "aggregation" and its records are not net new until it is split by source, and a
# row whose per-record values are not released is "blocked".
Status = Literal["candidate", "blocked", "aggregation"]

# How well a row's numbers and citation were checked. "sourced" means every headline
# figure traces to a source fetched this pass; "recall" means the dataset is real and
# the description sound, but the counts and accession need confirming before a loader.
Confidence = Literal["sourced", "recall"]


class Synergy(BaseModel):
    """What joining this row to another buys, and on what key.

    A synergy is only real if the two datasets share an addressable axis, so the
    join key is a required field rather than a remark. For this list the keys are
    the Keio collection, a named transposon library, a guide library over one gene
    set, a condition panel, a product pathway, or a single gene.
    """

    partner: str
    partner_status: Literal["supported", "candidate"]
    join: str
    yields: str


class Analog(BaseModel):
    """The already-built S. cerevisiae dataset a bacterial row mirrors.

    Recorded per row because the request was for bacterial data that matches the
    schema already in use. A row with a named yeast analog needs no new phenotype
    class: the loader writes the same record type against a different reference
    genome. A row with no analog is naming a phenotype the substrate has never
    held, which is a larger piece of work and is flagged as such.
    """

    dataset: str
    why: str


class Candidate(BaseModel):
    """One bacterial dataset proposed for ingestion."""

    name: str
    organism: Organism
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
    dim: int = 1  # phenotype vector length; 1 for a scalar label
    dim_basis: Basis = "reported"
    seq_basis: SeqBasis
    modality: str
    why: str
    accession: str
    accession_confirmed: bool = False
    status: Status = "candidate"
    confidence: Confidence = "sourced"
    # The isoprenol axis, which the pin in ``ranked`` reads. Default "none" so a row
    # is only on the axis when the judgment was made deliberately.
    isoprenol: Isoprenol = "none"
    # What the row measures, when it is on the isoprenol axis: the product name as the
    # paper words it, so "isopentenol" stays "isopentenol". Empty off the axis.
    product: str = ""
    analog: Analog | None = None
    synergy: list[Synergy] = Field(default_factory=list)
    time_axis: str = ""
    # What the schema must gain before this row can be written, empty when the
    # existing classes already cover it apart from the shared bacterial blockers.
    schema_need: str = ""

    @property
    def measurements(self) -> int | None:
        """Instances times phenotype dimensionality, the quantity rows are ranked on.

        Ranking on instances alone silently prefers a scalar fitness screen over a
        vector-valued omics panel of the same size: a 22-condition proteome is 22
        instances but roughly 50,000 numbers. Density is the second half of the
        requested ordering and this is where it enters.
        """
        return None if self.instances_n is None else self.instances_n * self.dim

    @property
    def sort_key(self) -> tuple[float, float, int]:
        """Measurements descending, then instances descending, then tier.

        Total and reproducible: no two rows can tie on all three unless they are
        the same size in every respect, and the log keeps a 10^8 row from
        overflowing the comparison.
        """
        return (
            -math.log10(max(self.measurements or 1, 1)),
            -math.log10(max(self.instances_n or 1, 1)),
            self.tier,
        )


class Excluded(BaseModel):
    """A dataset considered and dropped, with the rule that dropped it."""

    name: str
    reason: str
    rule: Literal["no-sequence", "off-host", "not-a-dataset", "no-per-record-data"]


class Acquisition(BaseModel):
    """A resource worth having that is NOT a row of the ranked table.

    A ranked row carries a verified instance count and a confirmed accession. A row
    here carries neither: it is a lead whose scale is asserted somewhere we cannot
    check, so it cannot be ranked against rows whose numbers were read off a file.
    Keeping it in its own table is what stops an unverifiable number from being
    compared with a measured one, and printing it first is what stops it from being
    forgotten, which is the only way a lead like this is actually lost.

    ``claimed`` is reproduced as the source states it and is never presented as
    measured. ``verification`` records what was actually done to check it, including
    a search that found nothing, because "we looked and did not find it" is a result
    and "we never looked" is not.
    """

    name: str
    organism: Organism
    modality: str
    claimed: str
    claim_source: str
    verification: str
    why: str
    blocking_action: str


class SchemaNeed(BaseModel):
    """One change the schema needs before any bacterial row can be written.

    Separated from the per-row ``schema_need`` because these are shared: every row
    in the table waits on them, so they are the critical path rather than a
    per-dataset cost. ``evidence`` is how the blocker was established, and for the
    two hard ones it is a command that was actually run.
    """

    what: str
    where: str
    blocks: str
    evidence: str
    additive: bool  # True when the change adds a class and moves no served closure


# ---------------------------------------------------------------------------
# The shared blockers. Established by running the validators, not by reading them:
# see the dated section of notes/experiments.database.expansion-bacteria.md.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Acquisition targets: leads that are not ranked rows. Printed FIRST in the
# document, before the fifty, because an unlocated deposit is the one kind of row
# that disappears from a backlog without anyone deciding to drop it.
# ---------------------------------------------------------------------------

ACQUISITION: list[Acquisition] = [
    Acquisition(
        name="Genome-wide KT2440 CRISPRi guide library, pooled growth selection",
        organism="P. putida",
        modality="CRISPRi repression, multiple guides per gene, pooled selection "
        "read out by guide enrichment and depletion",
        claimed="78,932 library members at 10 to 15 guides per gene, plus 798 "
        "non-targeting controls; selections in glucose, in acetate, and in "
        "p-coumaric acid at 10 and 50 mM. About 316,000 guide by selection slots "
        "if every member were quantified in all four selections, which is a "
        "product of the design and not a count of released values.",
        claim_source="A US Department of Energy project abstract, supplied by the "
        "owner on 2026-10-07. Not a peer-reviewed dataset release, and no "
        "accession accompanied the figures.",
        verification="UNVERIFIED. Searched 2026-10-07 and found no publication or "
        "deposit carrying these figures. The nearest confirmed genome-wide KT2440 "
        "CRISPRi resource is the dCpf1 library of row 20 (Menasalvas 2025), "
        "reported at about 16,500 guide targets, so the claimed scale is roughly "
        "five times that and is not the same library. Nothing here was measured "
        "from a file.",
        why="It is the only candidate whose perturbation modality matches the "
        "Carruthers 2025 target of row 6: repression rather than transposon "
        "disruption, several guides per gene, and a pooled selection readout. A "
        "transposon compendium transfers across a modality boundary; this would "
        "not have to.",
        blocking_action="Locate the deposit before any loader work. Until a file "
        "exists with a sha256, this row cannot be ranked, costed or scheduled, and "
        "the claimed slot count must not be added to any instance total.",
    )
]

SCHEMA_NEEDS: list[SchemaNeed] = [
    SchemaNeed(
        what=(
            "A bacterial gene-identifier namespace on GenePerturbation, so a "
            "b-number or a PP_ locus tag validates as a systematic gene name"
        ),
        where="torchcell/datamodels/schema.py, GenePerturbation.validate_sys_gene_name",
        blocks="every row in the table that perturbs a named gene",
        evidence=(
            "The validator admits only Y[A-P][LR]NNN[WC], QNNNN and YNC[A-Q]NNNN[WC]. "
            "Constructing a DeletionPerturbation with b0002, PP_0001, ECK0002 or thrA "
            "raises 'Invalid systematic gene name format'; YAL001C is accepted"
        ),
        additive=True,
    ),
    SchemaNeed(
        what=(
            "A genomes-tier assembly set per host, so a strain's sequence resolves "
            "the way a yeast strain's does"
        ),
        where="torchcell/sequence/genome/registry.py and torchcell/sequence/genome/",
        blocks="the sequence basis of every row, and so the sequence-reconstruction gate",
        evidence=(
            "The tier defines two assembly sets, both S. cerevisiae "
            "(sgd_S288C_R64-4-1_20230830 and peter2018_1011_assemblies), and the "
            "genome package holds only a scerevisiae subpackage"
        ),
        additive=True,
    ),
    SchemaNeed(
        what="Nothing on the reference side",
        where="torchcell/datamodels/schema.py, ReferenceGenome",
        blocks="nothing",
        evidence=(
            "species and strain are free strings, so ReferenceGenome("
            "species='Escherichia coli', strain='MG1655') and the KT2440 equivalent "
            "both validate unchanged"
        ),
        additive=True,
    ),
]


# ---------------------------------------------------------------------------
# Candidates. Figures marked confidence="sourced" trace to a source fetched this
# pass; "recall" rows carry counts that need confirming before a loader is written.
# ---------------------------------------------------------------------------

CANDIDATES: list[Candidate] = [
    Candidate(
        name="Nichols 2011",
        organism="E. coli",
        citation="Nichols RJ, Sen S, Choo YJ, Beltrao P, Zietek M, Chaba R, Lee "
        "S, Kazmierczak KM, Lee KJ, Wong A, Shales M, Lovett S, Winkler "
        "ME, Krogan NJ, Typas A, Gross CA. Phenotypic landscape of a "
        "bacterial cell. Cell 2011",
        url="https://doi.org/10.1016/j.cell.2010.11.052",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=3979,
        genotypes="3,979 deletions",
        env_n=324,
        env="324 conditions",
        instances_n=1289196,
        instances_basis="product",
        phenotype="colony size drug-gene fitness score",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="The paper states 3979 mutant strains passing quality control, 324 "
        "conditions covering 114 unique stresses, and a total of 13497 "
        "significant phenotypes at 5 percent FDR, which is about 1 percent of "
        "all condition-gene pairs tested. The URL printed in the paper, "
        "http://ecoliwiki.net/tools/chemgen/, now returns HTTP 410 Gone; the "
        "same portal is live at the ecoliwiki.org host and serves the full "
        "per-strain by per-condition flat file. The Keio background is "
        "BW25113, not MG1655 itself.",
        accession="https://ecoliwiki.org/tools/chemgen/ (coli_FinalData2.txt, 12.3 Mb; nichols.tgz, 61.5 Mb; conditions.txt); Cell Table S2 via PMC3060659",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="deletion collection across many chemicals",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Keio collection",
                yields="deletion collection across many chemicals",
            )
        ],
    ),
    Candidate(
        name="Baba 2006",
        organism="E. coli",
        citation="Baba T, Ara T, Hasegawa M, Takai Y, Okumura Y, Baba M, "
        "Datsenko KA, Tomita M, Wanner BL, Mori H. Construction of "
        "Escherichia coli K-12 in-frame, single-gene knockout mutants: "
        "the Keio collection. Molecular Systems Biology 2006",
        url="https://doi.org/10.1038/msb4100050",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=3985,
        genotypes="3,985 deletions",
        env_n=1,
        env="1 condition",
        instances_n=3985,
        instances_basis="reported",
        phenotype="deletion viability (in-frame knockout obtained or not)",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="Of 4288 genes targeted, mutants were obtained for 3985, held in "
        "duplicate for 7970 strains; 303 genes including 37 of unknown "
        "function could not be disrupted and are the paper's essential-gene "
        "candidates. Strain background is E. coli K-12 BW25113. The "
        "distribution URL given in the paper (GenoBase, "
        "http://ecoli.aist-nara.ac.jp/) is stale; NBRP E. coli currently "
        "lists 3909 Keio entries ready to distribute. This is the strain "
        "resource and the join axis for Nichols 2011, Tong 2020, Campos 2018, "
        "Shiver 2016 and the conjugation-based interaction screens, not a "
        "phenotype screen itself.",
        accession="https://shigen.nig.ac.jp/ecoli/strain/resource/keioCollection/list (NBRP E. coli, National Institute of Genetics)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="SGD essentiality",
            why="reference deletion set defines essentiality",
        ),
        synergy=[
            Synergy(
                partner="SGD essentiality",
                partner_status="supported",
                join="Keio collection",
                yields="reference deletion set defines essentiality",
            )
        ],
    ),
    Candidate(
        name="Wetmore 2015",
        organism="E. coli",
        citation="Wetmore KM, Price MN, Waters RJ, Lamson JS, He J, Hoover CA, "
        "Blow MJ, Bristow J, Butland G, Arkin AP, Deutschbauer A. Rapid "
        "quantification of mutant fitness in diverse bacteria by "
        "sequencing randomly bar-coded transposons. mBio 2015",
        url="https://doi.org/10.1128/mBio.00306-15",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=152018,
        genotypes="152,018 insertion mutants",
        env_n=89,
        env="89 conditions",
        instances_n=13529602,
        instances_basis="product",
        phenotype="barcode competitive fitness (BarSeq)",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="Table 1 gives 152018 strains with unique bar codes for the E. coli "
        "library KEIO_ML9 and fitness estimates for 3471 genes (84 percent). "
        "The E. coli lane included 96 samples with 4 time-zero samples across "
        "47 different carbon or nitrogen sources, of which 89 passed quality "
        "metrics. Aggregated to gene level the usable record count is 3471 by "
        "89, about 308919. Across all five bacteria the paper reports 387 "
        "successful genome-wide assays. No SRA accession is stated in the "
        "text; the LBL supplemental URL was taken from the paper and was not "
        "fetched.",
        accession="http://genomics.lbl.gov/supplemental/rbarseq/ ; code at https://bitbucket.org/berkeleylab/feba",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled barcoded competitive fitness assay",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="KEIO_ML9 RB-TnSeq library",
                yields="pooled barcoded competitive fitness assay",
            )
        ],
    ),
    Candidate(
        name="Price 2018",
        organism="E. coli",
        citation="Price MN, Wetmore KM, Waters RJ, Callaghan M, Ray J, Liu H, "
        "Kuehl JV, Melnyk RA, Lamson JS, Suh Y, Carlson HK, Esquivel Z, "
        "Sadeeshkumar H, Chakraborty R, Zane GM, Rubin BE, Wall JD, "
        "Visel A, Bristow J, Blow MJ, Arkin AP, Deutschbauer AM. Mutant "
        "phenotypes for thousands of bacterial genes of unknown "
        "function. Nature 2018",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=3789,
        genotypes="3,789 insertion mutants",
        env_n=162,
        env="162 conditions",
        instances_n=613818,
        instances_basis="product",
        phenotype="barcode competitive fitness (BarSeq log ratio)",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="The E. coli subset is separable: the published per-organism page for "
        "orgId Keio (Escherichia coli BW25113) states 207 condition samples "
        "with 162 successful, and its fit_logratios_good.tab was downloaded "
        "and counted at 162 experiment columns by 3789 gene rows. Whole-study "
        "totals are 32 bacteria, 4870 genome-wide fitness experiments meeting "
        "consistency criteria, and 11779 poorly annotated protein-coding "
        "genes with phenotypes. A 2021 data correction removes the sucrose "
        "and D-mannitol experiments for this organism. The live Fitness "
        "Browser sits behind a Cloudflare challenge and could not be read, so "
        "its current counts may exceed these.",
        accession="https://genomics.lbl.gov/supplemental/bigfit/html/Keio/ (fit_logratios_good.tab, fit_t.tab, strain_fit.tab, all.poolcount); figshare 10.6084/m9.figshare.5134840 and 10.6084/m9.figshare.5134837; Fitness Browser https://fit.genomics.lbl.gov",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled barcoded fitness across many conditions",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="KEIO_ML9 RB-TnSeq library",
                yields="pooled barcoded fitness across many conditions",
            )
        ],
    ),
    Candidate(
        name="Rousset 2018",
        organism="E. coli",
        citation="Rousset F, Cui L, Siouve E, Becavin C, Depardieu F, Bikard D. "
        "Genome-wide CRISPR-dCas9 screens in E. coli identify essential "
        "genes and phage host factors. PLoS Genetics 2018",
        url="https://doi.org/10.1371/journal.pgen.1007749",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=59000,
        genotypes="59,000 guides",
        env_n=4,
        env="4 conditions",
        instances_n=236000,
        instances_basis="product",
        phenotype="guide abundance log2 fold change (relative sgRNA fitness)",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+guide",
        modality="CRISPRi",
        why="The screen starts from a pool of about 92000 sgRNAs targeting random "
        "chromosomal positions and is filtered to about 59000 guides. "
        "Conditions are growth in rich medium over 17 generations for "
        "essentiality plus infection by phages lambda, T4 and 186 at MOI 1, "
        "all in triplicate; the paper screens no antibiotics, so any title "
        "citing antibiotic exposure for this paper is a conflation. 379 genes "
        "were selected for follow-up, including 235 annotated as essential. "
        "The ENA accession was verified directly at EBI.",
        accession="ENA PRJEB28256 (secondary study ERP110440); supplementary tables S1 to S10",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Smith 2016 CRISPRi", why="pooled CRISPRi guide fitness screen"
        ),
        synergy=[
            Synergy(
                partner="Smith 2016 CRISPRi",
                partner_status="supported",
                join="MG1655 gene set",
                yields="pooled CRISPRi guide fitness screen",
            )
        ],
    ),
    Candidate(
        name="Cui 2018",
        organism="E. coli",
        citation="Cui L, Vigouroux A, Rousset F, Varet H, Khanna V, Bikard D. A "
        "CRISPRi screen in E. coli reveals sequence-specific toxicity "
        "of dCas9. Nature Communications 2018",
        url="https://doi.org/10.1038/s41467-018-04209-5",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=92000,
        genotypes="92,000 guides",
        env_n=2,
        env="2 conditions",
        instances_n=184000,
        instances_basis="product",
        phenotype="guide abundance log2 fold change over 17 generations",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+guide",
        modality="CRISPRi",
        why="The library is about 92000 unique guide RNAs targeting random "
        "positions along the MG1655 genome with an NGG PAM requirement, "
        "averaging 19 targets per gene; an exact targeted-gene count is not "
        "stated. The two genome-wide screens differ only in dCas9 expression "
        "level (strains LC-E18 and LC-E75), both in rich medium with "
        "anhydrotetracycline in triplicate, with no stress panel. Data "
        "availability names Supplementary Data 4 for the screen results and "
        "otherwise defers to the corresponding author on request, so there is "
        "no SRA, ENA or GEO accession for this paper.",
        accession="Supplementary Data 4 of https://www.nature.com/articles/s41467-018-04209-5 (no repository accession)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Smith 2016 CRISPRi", why="pooled CRISPRi guide fitness screen"
        ),
        synergy=[
            Synergy(
                partner="Smith 2016 CRISPRi",
                partner_status="supported",
                join="MG1655 gene set",
                yields="pooled CRISPRi guide fitness screen",
            )
        ],
    ),
    Candidate(
        name="Wang 2018",
        organism="E. coli",
        citation="Wang T, Guan C, Guo J, Liu B, Wu Y, Xie Z, Zhang C, Xing XH. "
        "Pooled CRISPR interference screening enables genome-scale "
        "functional genomics study in bacteria with superior "
        "performance. Nature Communications 2018",
        url="https://doi.org/10.1038/s41467-018-04899-x",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=56071,
        genotypes="56,071 guides",
        env_n=5,
        env="5 conditions",
        instances_n=280355,
        instances_basis="product",
        phenotype="sgRNA abundance by Illumina sequencing, gene and sgRNA fitness",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+guide",
        modality="CRISPRi",
        why="Two corrections to the commonly cited form of this row. The venue is "
        "Nature Communications 9:2475, not Science; no Science 2018 "
        "genome-wide E. coli CRISPRi paper could be found. The library is "
        "55671 targeting sgRNAs plus 400 negative controls, not about 92000; "
        "the 92000-guide figure belongs to the Bikard-lab libraries of Cui "
        "2018 and Rousset 2018. Coverage is at least one sgRNA for 98.6 "
        "percent of 4140 protein-coding genes and 79.8 percent of 178 "
        "RNA-coding genes. The five screened phenotypes are essentiality in "
        "LB, auxotrophy in MOPS versus LB, L-tryptophan biosynthesis, "
        "furfural tolerance and isobutanol tolerance. Processed per-gene and "
        "per-sgRNA fitness are released as supplementary data; no "
        "raw-sequencing accession could be confirmed.",
        accession="Supplementary Data 05 to 10 of https://www.nature.com/articles/s41467-018-04899-x ; code at https://github.com/zhangchonglab/CRISPRi-functional-genomics-in-prokaryotes ; library at Addgene (Chong Zhang E. coli Genome-wide Inhibition Library)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Smith 2016 CRISPRi", why="pooled CRISPRi guide fitness screen"
        ),
        synergy=[
            Synergy(
                partner="Smith 2016 CRISPRi",
                partner_status="supported",
                join="MG1655 gene set",
                yields="pooled CRISPRi guide fitness screen",
            )
        ],
    ),
    Candidate(
        name="Choe 2025",
        organism="E. coli",
        citation="Choe D, Lee E, Kim K, Hwang S, Jeong KJ, Palsson BO, Cho BK, "
        "Cho S. Rapid identification of key antibiotic resistance genes "
        "in E. coli using high-resolution genome-scale CRISPRi "
        "screening. iScience 2025",
        url="https://doi.org/10.1016/j.isci.2025.112435",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=39591,
        genotypes="39,591 guides",
        env_n=12,
        env="12 conditions",
        instances_n=475092,
        instances_basis="product",
        phenotype="sgRNA abundance by next-generation sequencing",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+guide",
        modality="CRISPRi",
        why="The initial library holds 39591 sgRNA sequences at 99.96 percent "
        "coverage, targeting 4198 coding sequences. Twelve chemical "
        "conditions were evaluated: CCCP, polymyxin B, pyocyanin, rifampicin, "
        "sulfamethizole, verapamil, puromycin, erythromycin, phleomycin, "
        "mitomycin C, MMS and novobiocin. This is the genuine "
        "antibiotic-panel CRISPRi screen in E. coli and is the correct source "
        "for that claim rather than Rousset 2018. The ENA project was fetched "
        "and confirmed, released December 2024, submitted by KAIST.",
        accession="ENA PRJEB33267 (E. coli genome-wide CRISPRi under multiple chemical treatments)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Smith 2016 CRISPRi", why="pooled CRISPRi knockdown across drugs"
        ),
        synergy=[
            Synergy(
                partner="Smith 2016 CRISPRi",
                partner_status="supported",
                join="MG1655 gene set",
                yields="pooled CRISPRi knockdown across drugs",
            )
        ],
    ),
    Candidate(
        name="Mutalik 2020",
        organism="E. coli",
        citation="Mutalik VK, Adler BA, Rishi HS, Piya D, Zhong C, Koskella B, "
        "Kutter EM, Calendar R, Novichkov PS, Price MN, Deutschbauer "
        "AM, Arkin AP. High-throughput mapping of the phage resistance "
        "landscape in E. coli. PLoS Biology 2020",
        url="https://doi.org/10.1371/journal.pbio.3000877",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=152018,
        genotypes="152,018 insertion mutants",
        env_n=77,
        env="77 conditions",
        instances_n=11705386,
        instances_basis="product",
        phenotype="barcode competitive fitness (RB-TnSeq BarSeq)",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="The K-12 BW25113 RB-TnSeq library carries 152018 barcoded insertions "
        "in 3728 genes and was assayed against 14 diverse dsDNA phages in 68 "
        "RB-TnSeq assays plus 9 no-phage controls, giving the 77 conditions "
        "used here. A parallel BL21 library of about 97000 mutants was "
        "assayed against 12 phages in 53 pooled fitness experiments plus 9 "
        "controls. The same study also ran CRISPRi and Dub-seq arms on the "
        "same phage panel, which makes it a three-modality join on one "
        "condition axis. Accessions are quoted from the article's data "
        "availability statement; the SRA and figshare pages themselves were "
        "not fetched. The related Mutalik 2019 Nature Communications Dub-seq "
        "paper (doi 10.1038/s41467-018-08177-8, PMID 30659179) is a barcoded "
        "overexpression library of 30558 barcode pairs assayed in 155 fitness "
        "experiments over 52 chemicals with SRA PRJNA512427 and figshare "
        "10.6084/m9.figshare.6752753.v1; it is omitted as its own row only "
        "because plasmid overexpression has no matching modality value here.",
        accession="SRA BioProject PRJNA645443; RB-TnSeq data at figshare 10.6084/m9.figshare.11413128; CRISPRi data at 10.6084/m9.figshare.11859216.v1; Dub-seq data at 10.6084/m9.figshare.11838879.v2",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled barcoded mutants across selective conditions",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="KEIO_ML9 RB-TnSeq library",
                yields="pooled barcoded mutants across selective conditions",
            )
        ],
    ),
    Candidate(
        name="Girgis 2009",
        organism="E. coli",
        citation="Girgis HS, Hottes AK, Tavazoie S. Genetic architecture of "
        "intrinsic antibiotic susceptibility. PLoS ONE 2009",
        url="https://doi.org/10.1371/journal.pone.0005629",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=500000,
        genotypes="500,000 insertion mutants",
        env_n=17,
        env="17 conditions",
        instances_n=73100,
        instances_basis="estimate",
        phenotype="microarray genetic footprinting z-score per locus",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="The library is about 5 by 10^5 mutants each with a single transposon "
        "insertion, selected in 17 antibiotics, read out by hybridizing "
        "transposon-junction DNA against genomic DNA on spotted microarrays. "
        "The released data are per-locus z-scores, with Dataset S5 the "
        "combined z-scores across all loci, so usable records are locus by "
        "drug rather than strain by drug; the number of array loci is not "
        "stated, so the instance count here is an estimate from an assumed "
        "roughly 4300 loci and is labeled as such. No GEO or ArrayExpress "
        "accession exists for this dataset; a GEO search returned only the "
        "unrelated expression series GSE10855. The companion motility screen "
        "is Girgis HS, Liu Y, Ryu WS, Tavazoie S 2007 PLoS Genetics "
        "3(9):e154, doi 10.1371/journal.pgen.0030154, PMID 17941710, same "
        "library size, four selection regimes released as 13 supplementary "
        "microarray datasets with no accession.",
        accession="PLoS ONE Supporting Information Datasets S1 to S5 at https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0005629 (no repository accession)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant library across drug panel",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="MG1655 gene set",
                yields="pooled mutant library across drug panel",
            )
        ],
    ),
    Candidate(
        name="Typas 2008",
        organism="E. coli",
        citation="Typas A, Nichols RJ, Siegele DA, Shales M, Collins SR, Lim B, "
        "Braberg H, Yamamoto N, Takeuchi R, Wanner BL, Mori H, Weissman "
        "JS, Krogan NJ, Gross CA. High-throughput, quantitative "
        "analyses of genetic interactions in E. coli. Nature Methods "
        "2008",
        url="https://doi.org/10.1038/nmeth.1240",
        klass="Genetic interaction",
        tier=1,
        genotypes_n=91655,
        genotypes="91,655 double mutants",
        env_n=2,
        env="2 conditions",
        instances_n=91655,
        instances_basis="product",
        phenotype="colony pixel count interaction score",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO x KO",
        modality="double deletion",
        why="This is a method paper (GIANT-coli). Its released per-pair data "
        "cover only two demonstration screens: a 12 by 12 cross yielding 66 "
        "distinct pairwise double mutants plus 12 self-matings, and the pal "
        "and yraP screens in Supplementary Tables 2A and 2B. The genome-wide "
        "work is reported only as a linkage curve from 9 by 3985 crosses in "
        "LB and 14 by 3985 in M9, which is the 91655 figure above and is a "
        "product, not a released matrix. Status is blocked because no "
        "genome-wide double-mutant score matrix was released. The toolkit "
        "couples the Keio kanamycin deletion library with the ASKA "
        "chloramphenicol library, about 4000 single-gene deletions each. "
        "RE-MEASURED 2026-10-08 (experiments/036-dataset-fixes-before-kg-build/"
        "scripts/typas2008_release_loadability.py): the ACCESSION claim was STALE and "
        "the BLOCKING claim was not. The SI IS mirrored, fetched by the scriptable "
        "pmc_cloud route on 2026-10-07 (si/si1.pdf sha256 bd531ec2ee865506, OCR'd to "
        "si/si1.md), and enumerating the PMC Article Datasets bucket for PMC2700713 "
        "returns exactly ONE supplementary object, which is that file, so the deposit "
        "is mirrored in full and nothing sits behind a paywall. Supplementary Tables "
        "2A and 2B are readable in those bytes and they carry NO SCORE: 2A lists 23 "
        "pal partners as the terms ``neg (sick)``, ``neg (lethal)`` and ``pos``, and "
        "2B lists 15 yraP suppressors over the columns gene name, ECK number, location "
        "and function, with no value column at all. Of the 42 released pairs exactly 4 "
        "carry a number, and that statistic is ``% Co-inheritance of both markers`` "
        "from Supplementary Table 1's co-transduction check, not a colony-size "
        "interaction score. GeneInteractionPhenotype.gene_interaction is a REQUIRED "
        "float with no categorical mode, so the term refuses with float_parsing and "
        "omitting it refuses with Field required. The 12 by 12 cross's 66 distinct "
        "doubles exist only as the four colour-cell heat maps of Supplementary Figure "
        "4; its 12 axis genes are recoverable (surA, ybaY, ycbS, ompC, yraI, cpxR, "
        "degP, pal, ompA, yfgL, yraP, basR) and its numbers are not. NOT SUBSUMED, "
        "which is the opposite of what Butland 2008 turned out to be: 0 of the 38 "
        "screen pairs are in Babu 2014 Table S2, neither pal nor yraP is among Babu's "
        "163 donors, and the only 2 of 42 pairs that overlap are the verification "
        "pairs degP/surA and pal/ompA.",
        accession="Supplementary Tables 2A and 2B via PMC2700713, mirrored 2026-10-07 by pmc_cloud (NIHMS95293-supplement-Supp_Info.pdf, the bucket's only supplementary object for this PMCID); author manuscript at https://escholarship.org/content/qt0b4526ns/qt0b4526ns.pdf",
        accession_confirmed=True,
        status="blocked",
        confidence="sourced",
        analog=Analog(
            dataset="Costanzo 2016 SGA", why="conjugation analog of yeast SGA"
        ),
        synergy=[
            Synergy(
                partner="Costanzo 2016 SGA",
                partner_status="supported",
                join="Keio collection",
                yields="conjugation analog of yeast SGA",
            )
        ],
    ),
    Candidate(
        name="Butland 2008",
        organism="E. coli",
        citation="Butland G, Babu M, Diaz-Mejia JJ, Bohdana F, Phanse S, Gold B, "
        "Yang W, Li J, Gagarinova AG, Pogoutse O, Mori H, Wanner BL, Lo "
        "H, Wasniewski J, Christopolous C, Ali M, Venn P, Safavi-Naini "
        "A, Sourour N, Caron S, Choi JY, Laigle L, Nazarians-Armavil A, "
        "Deshpande A, Joe S, Datsenko KA, Yamamoto N, Andrews BJ, Boone "
        "C, Ding H, Sheikh B, Moreno-Hagelsieb G, Greenblatt JF, Emili "
        "A. eSGA: E. coli synthetic genetic array analysis. Nature "
        "Methods 2008",
        url="https://doi.org/10.1038/nmeth.1239",
        klass="Genetic interaction",
        tier=1,
        genotypes_n=314847,
        genotypes="314,847 double mutants",
        env_n=1,
        env="1 condition",
        instances_n=314847,
        instances_basis="reported",
        phenotype="colony size interaction (S) score",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO x KO",
        modality="double deletion",
        why="MEASURED 2026-10-08 and NO LONGER BLOCKED. The retrieval blocker is "
        "gone: the paper and five SI files are in the mirror, so every count "
        "this row used to call an unverified estimate is now read from pinned "
        "bytes by experiments/036-dataset-fixes-before-kg-build/scripts/"
        "bacteria_subsumed_rows.py (issue #794). Supplementary Table 2 lists "
        "exactly 39 query strains. Supplementary Table 1 lists the recipient "
        "array as 7,924 Keio deletion strains, two isolates of 3,968 genes, "
        "plus 149 SPA-tag essential hypomorphs: 8,073 rows over 4,117 "
        "distinct genes. Supplementary Table 4 then releases the whole 39 by "
        "8,073 matrix in four layers, raw colony sizes, normalized median "
        "colony sizes, absolute Z scores and S scores, and says it does so "
        "without any filtering parameters: all 314,847 cells are populated in "
        "every layer, which collapses to 160,563 distinct gene pairs. "
        "Supplementary Table 3's high-confidence set is 1,288 distinct "
        "ordered pairs, of which 799 are non-essential and split 730 "
        "aggravating and 69 alleviating exactly as the Results state, plus "
        "489 whose recipient is a SPA-tag hypomorph. THE SUBSUMPTION IS "
        "MEASURED AND IT IS SMALL, which is the opposite of the triage this "
        "row inherited: the served GeneInteractionBabu2014Dataset carries 727 "
        "records tagged screen_id Butland et al., 0.23 percent of the "
        "released matrix, because Babu's Table S2 is the high-confidence tail "
        "of a re-analysis and not a superset of this release. Babu "
        "re-publishes 1,129 rows over these same 39 donors and omits 490 of "
        "this paper's own 1,270 high-confidence gene pairs, 321 of them "
        "non-essential; where the two releases share a pair the number is the "
        "same number, 793 of 873 same-orientation rows identical, so loading "
        "this release adds records rather than a second score definition. A "
        "loader is therefore warranted and the row stays on the schedule. The "
        "489 SPA-tag pairs stay blocked on the hypomorph perturbation leaf "
        "the Babu loader already filed, and the raw and normalized colony "
        "sizes need a colony-size phenotype class that does not exist. "
        "Provenance record: $DATA_ROOT/torchcell-raw/"
        "butlandESGAColiSynthetic2008/subsumption_record.json.",
        accession="https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1239/MediaObjects/",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Costanzo 2016 SGA", why="conjugation analog of yeast SGA"
        ),
        synergy=[
            Synergy(
                partner="Costanzo 2016 SGA",
                partner_status="supported",
                join="Keio collection",
                yields="conjugation analog of yeast SGA",
            )
        ],
    ),
    Candidate(
        name="Babu 2014",
        organism="E. coli",
        citation="Babu M, Arnold R, Bundalovic-Torma C, Gagarinova A, Wong KS, "
        "Kumar A, Stewart G, Samanfar B, Aoki H, Wagih O, Vlasblom J, "
        "Phanse S, Lad K, Yeou Hsiung Yu A, Graham C, Jin K, Brown E, "
        "Golshani A, Kim P, Moreno-Hagelsieb G, Greenblatt J, Houry WA, "
        "Parkinson J, Emili A. Quantitative genome-wide genetic "
        "interaction screens reveal global epistatic relationships of "
        "protein complexes in Escherichia coli. PLoS Genetics 2014",
        url="https://doi.org/10.1371/journal.pgen.1004120",
        klass="Genetic interaction",
        tier=1,
        genotypes_n=671071,
        genotypes="671,071 double mutants",
        env_n=1,
        env="1 condition",
        instances_n=42705,
        instances_basis="reported",
        phenotype="colony size epistasis (S) score",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO x KO",
        modality="double deletion",
        why="The largest E. coli digenic screen that could be verified. It "
        "reports over 600000 digenic mutant combinations from 163 query donor "
        "genes crossed into an array of 3968 non-essential deletions plus 149 "
        "hypomorphic strains, which is 671071 nominal pairs, with donors "
        "transferred by conjugation from an Hfr-Cavalli "
        "chloramphenicol-marked background. Only the high-confidence subset "
        "is released, 25239 aggravating plus 17466 alleviating for 42705 "
        "pairs in Table S2, so the full matrix is not recoverable and the "
        "instance count reflects the released subset. The dedicated portal "
        "http://ecoli.med.utoronto.ca/esga no longer resolves to the Emili "
        "lab (the host presents a certificate for lymnaea.org), and BioGRID "
        "has no entry for PMID 24586182. The earlier Babu 2011 PLoS Genetics "
        "map (doi 10.1371/journal.pgen.1002377, PMID 22125496) is smaller at "
        "over 235000 combinations and likewise released only significant "
        "pairs. Kumar 2016 Cell Reports (doi 10.1016/j.celrep.2015.12.060, "
        "PMID 26774489) covers 102255 viable digenic mutants and is the one "
        "E. coli interaction set BioGRID serves, at 31644 curated "
        "interactions.",
        accession="Table S2 (XLS, high-confidence epistatic gene pairs) at https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004120",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Costanzo 2016 SGA",
            why="conjugation digenic epistasis colony scores",
        ),
        synergy=[
            Synergy(
                partner="Costanzo 2016 SGA",
                partner_status="supported",
                join="Keio collection",
                yields="conjugation digenic epistasis colony scores",
            )
        ],
    ),
    Candidate(
        name="Tong 2020",
        organism="E. coli",
        citation="Tong M, French S, El Zahed SS, Ong WK, Karp PD, Brown ED. Gene "
        "dispensability in Escherichia coli grown in thirty different "
        "carbon environments. mBio 2020",
        url="https://doi.org/10.1128/mBio.02259-20",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=3796,
        genotypes="3,796 deletions",
        env_n=30,
        env="30 conditions",
        instances_n=113880,
        instances_basis="reported",
        phenotype="24 hour solid-agar colony growth curve",
        dim=72,
        dim_basis="estimate",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="Kinetic growth measurements for 3796 genes under 30 carbon source "
        "conditions in MOPS minimal medium, 1536 colonies per plate on solid "
        "agar, giving 113880 individual growth curves and over 8 million data "
        "points. Plates were scanned every 20 minutes over 24 hours, so each "
        "instance is a time series of roughly 72 points; the 72 figure is "
        "derived from the stated interval and duration rather than stated "
        "directly. 51 poorly annotated genes showed a low growth phenotype in "
        "at least one carbon source. This is the cleanest released Keio by "
        "carbon-source matrix and the only row here carrying a real time "
        "axis.",
        accession="Supplementary Tables S1 to S6 via PMC7527729; CarPE web application at https://edbrownlab.shinyapps.io/CarPE/",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Smith 2006 chemogenomic",
            why="colony growth per deletion per condition",
        ),
        synergy=[
            Synergy(
                partner="Smith 2006 chemogenomic",
                partner_status="supported",
                join="Keio collection",
                yields="colony growth per deletion per condition",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Campos 2018",
        organism="E. coli",
        citation="Campos M, Govers SK, Irnov I, Dobihal GS, Cornet F, "
        "Jacobs-Wagner C. Genomewide phenotypic analysis of growth, "
        "cell morphogenesis, and cell cycle events in Escherichia coli. "
        "Molecular Systems Biology 2018",
        url="https://doi.org/10.15252/msb.20177573",
        klass="Fitness / chemical genomics",
        tier=2,
        genotypes_n=4227,
        genotypes="4,227 deletions",
        env_n=1,
        env="1 condition",
        instances_n=4227,
        instances_basis="product",
        phenotype="single-cell microscopy morphology and cell cycle features",
        dim=26,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="4227 strains of the Keio collection, covering 98 percent of the "
        "non-essential genome and 87 percent of the complete genome, imaged "
        "in one condition (M9 with 0.1 percent casamino acids and 0.2 percent "
        "glucose at 30 degrees C). 26 quantitative features per strain: 19 "
        "morphological, 5 cell cycle, and 2 growth-related. On average about "
        "360 cells were imaged per strain, roughly 1.3 million cells after "
        "filtering. This is the highest-dimensionality E. coli deletion "
        "phenotype release found, and the natural counterpart to a yeast "
        "morphology or transcriptome profile rather than to a scalar fitness "
        "score.",
        accession="Dataset EV2 (corrected and normalized scores) with the article at https://www.embopress.org/doi/full/10.15252/msb.20177573",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Kemmeren 2014 deletion transcriptome",
            why="high-dimensional profile per deletion strain",
        ),
        synergy=[
            Synergy(
                partner="Kemmeren 2014 deletion transcriptome",
                partner_status="supported",
                join="Keio collection",
                yields="high-dimensional profile per deletion strain",
            )
        ],
    ),
    Candidate(
        name="Shiver 2016",
        organism="E. coli",
        citation="Shiver AL, Osadnik H, Kritikos G, Li B, Krogan N, Typas A, "
        "Gross CA. A chemical-genomic screen of neglected antibiotics "
        "reveals illicit transport of kasugamycin and blasticidin S. "
        "PLoS Genetics 2016",
        url="https://doi.org/10.1371/journal.pgen.1006124",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=3975,
        genotypes="3,975 deletions",
        env_n=57,
        env="57 conditions",
        instances_n=226575,
        instances_basis="product",
        phenotype="colony opacity fitness score (Iris)",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="3975 Keio deletion mutants against 26 new stresses for 57 conditions "
        "in total, scored from colony opacity with the Iris image analysis "
        "software, and designed as an extension of the Nichols 2011 condition "
        "set. The release is raw plate images plus Iris output and ranked "
        "score files rather than a clean gene by condition matrix, so "
        "building the matrix requires rerunning the published scoring "
        "scripts. The Zenodo mirror of the Dryad deposit was fetched and its "
        "file list confirmed.",
        accession="Dryad 10.5061/dryad.f3kc0, mirrored at https://zenodo.org/records/5085832 (imagefile.tar 10.5 GB, irisfile.tar 65.1 MB, ImageKeyFinal.xls, InputFiles.tar); scripts at https://github.com/AnthonyShiverMicrobes/fitness_score",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Wildenhain 2015 drug tolerance",
            why="neglected-compound panel against deletion collection",
        ),
        synergy=[
            Synergy(
                partner="Wildenhain 2015 drug tolerance",
                partner_status="supported",
                join="Keio collection",
                yields="neglected-compound panel against deletion collection",
            )
        ],
    ),
    Candidate(
        name="Goodall 2018",
        organism="E. coli",
        citation="Goodall ECA, Robinson A, Johnston IG, Jabbari S, Turner KA, "
        "Cunningham AF, Lund PA, Cole JA, Henderson IR. The essential "
        "genome of Escherichia coli K-12. mBio 2018",
        url="https://doi.org/10.1128/mBio.02096-17",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=901383,
        genotypes="901,383 insertion mutants",
        env_n=1,
        env="1 condition",
        instances_n=901383,
        instances_basis="product",
        phenotype="TraDIS insertion-site read density, essentiality call",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="A transposon mutant library estimated at approximately 3.7 million "
        "mutants, sequenced to 8279309 mapped reads over 901383 unique "
        "insertion sites, from which 358 genes were identified as essential. "
        "Only one condition was used, Luria broth at 37 degrees C to OD600 of "
        "1.0, in two independent DNA extracts TL1 and TL2. 248 genes (59.5 "
        "percent) of the essential calls were common to this study, the Keio "
        "collection and PEC. The ENA accession is quoted from the paper's "
        "data availability statement and the ENA page itself was not fetched.",
        accession="ENA PRJEB24436",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="SGD essentiality", why="genome-wide essential gene call set"
        ),
        synergy=[
            Synergy(
                partner="SGD essentiality",
                partner_status="supported",
                join="MG1655 gene set",
                yields="genome-wide essential gene call set",
            )
        ],
    ),
    Candidate(
        name="Gerdes 2003",
        organism="E. coli",
        citation="Gerdes SY, Scholle MD, Campbell JW, Balazsi G, Ravasz E, "
        "Daugherty MD, Somera AL, Kyrpides NC, Anderson I, Gelfand MS, "
        "Bhattacharya A, Kapatral V, D'Souza M, Baev MV, Grechkin Y, "
        "Mseeh F, Fonstein MY, Overbeek R, Barabasi AL, Oltvai ZN, "
        "Osterman AL. Experimental determination and system level "
        "analysis of essential genes in Escherichia coli MG1655. "
        "Journal of Bacteriology 2003",
        url="https://doi.org/10.1128/jb.185.19.5673-5684.2003",
        klass="Transposon fitness",
        tier=2,
        genotypes_n=3746,
        genotypes="3,746 insertion mutants",
        env_n=1,
        env="1 condition",
        instances_n=3746,
        instances_basis="reported",
        phenotype="genetic footprinting essentiality assertion",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="Unambiguous essentiality assessments were made for 3746 (87 percent) "
        "of E. coli protein-encoding genes or ORFs, of which 620 (14 percent) "
        "were asserted essential and 3126 (73 percent) non-essential; no "
        "assertion was possible for 327 genes on technical grounds and "
        "evidence was insufficient for 218. Method is Tn5-based genetic "
        "footprinting in rich aerobic media. The per-gene Table S1 is still "
        "served live at the Wisconsin mirror, which makes this the cleanest "
        "hash-pinnable per-gene essentiality table for MG1655. Note that the "
        "Database of Essential Genes lists 609 essential genes for this study "
        "and 296 for Baba 2006, both of which disagree with the source papers "
        "(620 and 303); use the paper tables.",
        accession="https://www.genome.wisc.edu/Gerdes2003/ (Table S1 in HTML, Excel, PDF and TXT)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="SGD essentiality", why="genome-wide essential gene call set"
        ),
        synergy=[
            Synergy(
                partner="SGD essentiality",
                partner_status="supported",
                join="MG1655 gene set",
                yields="genome-wide essential gene call set",
            )
        ],
    ),
    Candidate(
        name="D2Cell 2026",
        organism="E. coli",
        citation="Li X, et al. Leveraging large language models for metabolic "
        "engineering design. Trends in Biotechnology (preprint: "
        "bioRxiv) 2026",
        url="https://doi.org/10.1101/2024.09.09.612023",
        klass="Aggregation / support",
        tier=3,
        genotypes_n=8134,
        genotypes="8,134 designs",
        env_n=1,
        env="1 condition",
        instances_n=8134,
        instances_basis="reported",
        phenotype="literature-reported production metric per gene-modification record",
        dim=1,
        dim_basis="estimate",
        seq_basis="reference-only",
        modality="mixed",
        why="Confirmed verbatim in the preprint: 8134 experimentally reported "
        "single- and double-gene modification records for 73 products in E. "
        "coli, merged for training with 11643 single-gene-modification rows "
        "for the same 73 products simulated by the genome-scale model "
        "(FSEOF-style), so the simulated rows originate from a separate "
        "generation step and are separable in principle, but the preprint "
        "states no explicit flag distinguishing them in the released file; "
        "the whole database is 29006 entries, 1210 products, 751 organisms. "
        "These are counts of literature records, not measured "
        "strain-condition observations, and must be resolved to primary "
        "source DOIs before they count as independent datasets.",
        accession="https://zenodo.org/records/18240770 (Cell_factory_dataset_Qwen_110b.xlsx, 11.4 MB); code https://github.com/LiLabTsinghua/D2Cell; web server https://digitallifethu.com/d2cell",
        accession_confirmed=True,
        status="aggregation",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene screen",
            why="both score a small-molecule product per engineered strain over a "
            "genome-scale perturbation panel",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene screen",
                partner_status="supported",
                join="E. coli gene set x product titer",
                yields="both score a small-molecule product per engineered strain "
                "over a genome-scale perturbation panel",
            )
        ],
    ),
    Candidate(
        name="Oyetunde 2019",
        organism="E. coli",
        citation="Oyetunde T, et al. Machine learning framework for assessment "
        "of microbial factory performance. PLOS ONE 2019",
        url="https://doi.org/10.1371/journal.pone.0210558",
        klass="Aggregation / support",
        tier=3,
        genotypes_n=1200,
        genotypes="1,200 designs",
        env_n=1,
        env="1 condition",
        instances_n=1200,
        instances_basis="reported",
        phenotype="titer (g/L), rate (g/L/h), yield (g/g)",
        dim=3,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="mixed",
        why="About 1200 experimentally realized E. coli cell factories "
        "hand-curated from about 100 papers covering more than 20 products, "
        "with features in six categories (carbon source, bioprocess "
        "conditions, genetic modifications, product properties, production "
        "metrics, undocumented factors) and TRY targets; data availability "
        "statement says all data are in the supplementary files. These are "
        "literature records, not measured strain-condition observations, so "
        "they must be resolved to primary source DOIs before counting as "
        "independent datasets.",
        accession="https://doi.org/10.1371/journal.pone.0210558 (S1 File Excel: paper list plus extracted per-factory data)",
        accession_confirmed=True,
        status="aggregation",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene screen",
            why="both score a small-molecule product per engineered strain over a "
            "genome-scale perturbation panel",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene screen",
                partner_status="supported",
                join="E. coli gene set x product TRY",
                yields="both score a small-molecule product per engineered strain "
                "over a genome-scale perturbation panel",
            )
        ],
    ),
    Candidate(
        name="Fang 2025 FFA CRISPRi-FACS",
        organism="E. coli",
        citation="Fang L, Hao X, Fan J, Liu X, Chen Y, Wang L, Huang X, Song H, "
        "Cao Y. Genome-scale CRISPRi screen identifies pcnB repression "
        "conferring improved physiology for overproduction of free "
        "fatty acids in Escherichia coli. Nature Communications 2025",
        url="https://doi.org/10.1038/s41467-025-58368-3",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=55671,
        genotypes="55,671 guides",
        env_n=1,
        env="1 condition",
        instances_n=55671,
        instances_basis="reported",
        phenotype="sgRNA enrichment (FACS sort plus NGS), then titer (g/L) on validated strains",
        dim=1,
        dim_basis="estimate",
        seq_basis="K-12+guide",
        modality="CRISPRi",
        why="The screen reuses a previously published plasmid library of 55671 "
        "sgRNAs, sorted with a fluorescent free-fatty-acid biosensor and read "
        "out by NGS; the engineered pcnBi-acrDi-fadR+ strain reached 35.1 g/L "
        "FFAs in fed-batch. Pooled enrichment gives a guide-level score "
        "rather than a per-design titer, so only the handful of individually "
        "reconstructed strains carry phenotype values; no sequencing "
        "accession was confirmed.",
        accession="https://www.nature.com/articles/s41467-025-58368-3",
        accession_confirmed=False,
        status="blocked",
        confidence="sourced",
        analog=Analog(
            dataset="Xue 2025 free fatty acids",
            why="both read free fatty acid output as the label on a perturbed strain",
        ),
        synergy=[
            Synergy(
                partner="Xue 2025 free fatty acids",
                partner_status="supported",
                join="MG1655 gene set",
                yields="both read free fatty acid output as the label on a perturbed "
                "strain",
            )
        ],
    ),
    Candidate(
        name="Fang 2021 FFA CRISPRi",
        organism="E. coli",
        citation="Fang L, Fan J, Luo S, Chen Y, Wang C, Cao Y, Song H. "
        "Genome-scale target identification in Escherichia coli for "
        "high-titer production of free fatty acids. Nature "
        "Communications 2021",
        url="https://doi.org/10.1038/s41467-021-25243-w",
        klass="CRISPR library screen",
        tier=4,
        genotypes_n=108,
        genotypes="108 guides",
        env_n=1,
        env="1 condition",
        instances_n=324,
        instances_basis="estimate",
        phenotype="free fatty acid titer (g/L) by GC",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+guide",
        modality="CRISPRi",
        why="Arrayed rather than pooled: 108 synthetic sgRNAs repressing 108 "
        "chromosomal genes across five metabolic modules, each strain "
        "measured for FFA titer by GC in three biological replicates, giving "
        "recoverable per-design phenotypes; 30 beneficial genes from the "
        "screen plus 26 more from omics, 56 total, and 30.0 g/L in fed-batch. "
        "Accessions are named in the paper but were not fetched.",
        accession="GEO GSE146162 (transcriptomics); ProteomeXchange/iProX PXD017890 (proteomics); GenBank MZ567118-MZ567123 (plasmids)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Xue 2025 free fatty acids",
            why="both read free fatty acid output as the label on a perturbed strain",
        ),
        synergy=[
            Synergy(
                partner="Xue 2025 free fatty acids",
                partner_status="supported",
                join="MG1655 gene set",
                yields="both read free fatty acid output as the label on a perturbed "
                "strain",
            )
        ],
    ),
    Candidate(
        name="Opgenorth 2019",
        organism="E. coli",
        citation="Opgenorth P, Costello Z, Okada T, Goyal G, Chen Y, Gin J, "
        "Benites V, de Raad M, Northen TR, Deng K, Deutsch S, Baidoo "
        "EEK, Petzold CJ, Hillson NJ, Garcia Martin H, Beller HR. "
        "Lessons from Two Design-Build-Test-Learn Cycles of Dodecanol "
        "Production in Escherichia coli Aided by Machine Learning. ACS "
        "Synthetic Biology 2019",
        url="https://doi.org/10.1021/acssynbio.9b00020",
        klass="Multi-omics campaign",
        tier=4,
        genotypes_n=60,
        genotypes="60 RBS variants",
        env_n=2,
        env="2 conditions",
        instances_n=180,
        instances_basis="estimate",
        phenotype="1-dodecanol titer (g/L) plus targeted proteomics of the engineered pathway proteins",
        dim=5,
        dim_basis="estimate",
        seq_basis="engineered-chassis+RBS",
        modality="RBS variant",
        why="Two DBTL cycles over 60 engineered E. coli MG1655 strains; cycle 1 "
        "tested 36 strains varying ribosome-binding sites and reductase "
        "enzymes in one pathway operon, and cycle 2 designs from the "
        "machine-learning models raised titer 21% to 0.83 g/L. The abstract "
        "states dodecanol plus all engineered-pathway protein concentrations "
        "were measured; the exact proteomics panel size and replicate count "
        "were not verified because the full text is paywalled, so "
        "dimensionality 5 and instances 180 are estimates. Conditions counted "
        "as 2 for the two DBTL cycles.",
        accession="https://acs.figshare.com/articles/dataset/Lessons_from_Two_Design_Build_Test_Learn_Cycles_of_Dodecanol_Production_in_i_Escherichia_coli_i_Aided_by_Machine_Learning/8171012",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol screen",
            why="both carry an alcohol titer per perturbed strain, which is the "
            "same record shape with a different reference genome",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol screen",
                partner_status="supported",
                join="fatty alcohol pathway",
                yields="both carry an alcohol titer per perturbed strain, which is "
                "the same record shape with a different reference genome",
            )
        ],
    ),
    Candidate(
        name="Jervis 2019",
        organism="E. coli",
        citation="Jervis AJ, Carbonell P, Vinaixa M, Dunstan MS, Hollywood KA, "
        "Robinson CJ, Rattray NJW, Yan C, Swainston N, Currin A, Sung "
        "R, Toogood H, Taylor S, Faulon JL, Breitling R, Takano E, "
        "Scrutton NS. Machine Learning of Designed Translational "
        "Control Allows Predictive Pathway Optimization in Escherichia "
        "coli. ACS Synthetic Biology 2019",
        url="https://doi.org/10.1021/acssynbio.8b00398",
        klass="Combinatorial design",
        tier=4,
        genotypes_n=88,
        genotypes="88 RBS variants",
        env_n=1,
        env="1 condition",
        instances_n=264,
        instances_basis="estimate",
        phenotype="limonene titer (mg per L organic phase)",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis+RBS",
        modality="RBS variant",
        why="Two RBS libraries in E. coli DH10-beta: pGLlib pairs 12 designed RBS "
        "variants each for trAg-gpps and trMs-limS, 144 possible "
        "combinations, from which 360 colonies were screened in deep-well "
        "plates and 64 clones sequenced, covering 56 of the 144 combinations; "
        "a second 5184-member pMVA2 library had 156 colonies screened and a "
        "designed 32-combination reduced library built and tested. Genotypes "
        "88 counts only sequence-resolved designs (56 plus 32), since "
        "unsequenced colonies have titers without a recoverable genotype; "
        "instances assumes triplicate.",
        accession="https://doi.org/10.1021/acssynbio.8b00398 (Supporting Information Tables S1-S2, RBS sequences plus titers)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="both are designed multi-part libraries read out by product or "
            "growth selection rather than by a per-gene assay",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="mevalonate pathway",
                yields="both are designed multi-part libraries read out by product "
                "or growth selection rather than by a per-gene assay",
            )
        ],
    ),
    Candidate(
        name="Brunk 2016",
        organism="E. coli",
        citation="Brunk E, George KW, Alonso-Gutierrez J, Thompson M, Baidoo E, "
        "Wang G, Petzold CJ, McCloskey D, Monk J, Yang L, O'Brien EJ, "
        "Batth TS, Martin HG, Feist A, Adams PD, Keasling JD, Palsson "
        "BO, Lee TS. Characterizing Strain Variation in Engineered E. "
        "coli Using a Multi-Omics-Based Workflow. Cell Systems 2016",
        url="https://doi.org/10.1016/j.cels.2016.04.004",
        klass="Multi-omics campaign",
        tier=4,
        genotypes_n=9,
        genotypes="9 pathway designs",
        env_n=9,
        env="9 conditions",
        instances_n=81,
        instances_basis="product",
        phenotype="86 metabolites plus 55 protein complexes, with product titers (isopentenol, limonene, bisabolene)",
        dim=141,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="pathway plasmid",
        why="Nine strains (eight engineered plus wild-type E. coli DH1) carrying "
        "three versions of a heterologous mevalonate pathway, sampled across "
        "a 0 to 72 hour batch fermentation: the joint panel is 86 metabolites "
        "and 55 protein complexes at 9 time points, while the aggregate "
        "metabolomics set is described as 9 strains x 13 time points x 86 "
        "metabolites, so instances 81 is the product for the joint panel and "
        "the metabolome alone is larger. Variances come from triplicate "
        "measurements where available. No omics repository accession was "
        "confirmed.",
        accession="https://escholarship.org/uc/item/2k69b9zr (accepted manuscript plus supplementary files; analysis as iPython notebooks)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Zelezniak 2018 metabolome",
            why="both put metabolite concentrations and a second molecular layer "
            "on one strain axis",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 metabolome",
                partner_status="supported",
                join="mevalonate pathway",
                yields="both put metabolite concentrations and a second molecular "
                "layer on one strain axis",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Jones 2015 ePathOptimize",
        organism="E. coli",
        citation="Jones JA, Vernacchio VR, Lachance DM, Lebovich M, Fu L, Shirke "
        "AN, Schultz VL, Cress B, Linhardt RJ, Koffas MAG. "
        "ePathOptimize: A Combinatorial Approach for Transcriptional "
        "Balancing of Metabolic Pathways. Scientific Reports 2015",
        url="https://doi.org/10.1038/srep11301",
        klass="Combinatorial design",
        tier=4,
        genotypes_n=307,
        genotypes="307 promoter variants",
        env_n=1,
        env="1 condition",
        instances_n=307,
        instances_basis="reported",
        phenotype="violacein titer (mg/L) by HPLC",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis+promoter",
        modality="promoter variant",
        why="Five IPTG-inducible mutant T7 promoters of graded strength were "
        "placed on each of the five violacein pathway genes vioABCDE in E. "
        "coli BL21star(DE3), a theoretical 3125-member library; 107 colonies "
        "were screened first (3.4% of the library) and a second enriched "
        "library of 200 mutants was screened, giving 307 genotype-linked "
        "titers. Best screen hit 238 mg/L, 63-fold over control, rising to "
        "1829 plus or minus 46 mg/L after fermentation optimization. "
        "Deoxyviolacein and pathway intermediates were also quantified for "
        "selected strains, so the per-instance panel may exceed 1.",
        accession="https://doi.org/10.1038/srep11301 (Supplementary Information, screened mutants plus sequences)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene screen",
            why="both score a small-molecule product per engineered strain over a "
            "genome-scale perturbation panel",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene screen",
                partner_status="supported",
                join="violacein pathway",
                yields="both score a small-molecule product per engineered strain "
                "over a genome-scale perturbation panel",
            )
        ],
    ),
    Candidate(
        name="Rapp 2026 CRISPRi metabolome",
        organism="E. coli",
        citation="Rapp J, Verhulsdonk A, Garcke A, Stadelmann A, Farke N, "
        "Trossmann F, Kronenberger T, Alvarado A, Petras D, Link H. The "
        "metabolome of an E. coli CRISPRi library identifies benefits "
        "of minimal metabolite levels and targets for engineering. Cell "
        "Systems 2026",
        url="https://doi.org/10.1016/j.cels.2025.101518",
        klass="CRISPR library screen",
        tier=3,
        genotypes_n=1515,
        genotypes="1,515 guides",
        env_n=1,
        env="1 condition",
        instances_n=1515,
        instances_basis="reported",
        phenotype="flow-injection MS metabolome (ion intensities), plus carotenoid titer in the engineering follow-up",
        dim=300,
        dim_basis="estimate",
        seq_basis="K-12+guide",
        modality="CRISPRi",
        why="Arrayed CRISPRi library covering all 1515 metabolic genes of "
        "iML1515, each strain profiled by fast flow-injection MS, so "
        "per-design metabolome vectors are recoverable; 36% of "
        "iML1515-predicted metabolites accumulated specifically in some "
        "knockdown, and LC-MS/MS spectra were generated for 102 previously "
        "uncharacterized metabolites. The exact number of annotated ions per "
        "strain was not verified from the full text, so dimensionality 300 is "
        "an estimate. The MassIVE record was fetched and confirmed.",
        accession="MassIVE MSV000095534 ('A metabolism-wide CRISPRi library expands the measurable E. coli metabolome - FI-MS dataset', submitted by H. Link, 2024-08-07); Zenodo https://zenodo.org/records/15640293",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Zelezniak 2018 metabolome",
            why="both put metabolite concentrations and a second molecular layer "
            "on one strain axis",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 metabolome",
                partner_status="supported",
                join="iML1515 metabolic gene set",
                yields="both put metabolite concentrations and a second molecular "
                "layer on one strain axis",
            )
        ],
    ),
    Candidate(
        name="ActiveOpt valine (Kumar 2021)",
        organism="E. coli",
        citation="Kumar P, Adamczyk PA, Zhang X, Andrade RB, Romero PA, "
        "Ramanathan P, Reed JL. Active and machine learning-based "
        "approaches to rapidly enhance microbial chemical production. "
        "Metabolic Engineering 2021",
        url="https://doi.org/10.1016/j.ymben.2021.06.009",
        klass="Combinatorial design",
        tier=4,
        genotypes_n=93,
        genotypes="93 RBS variants",
        env_n=1,
        env="1 condition",
        instances_n=186,
        instances_basis="estimate",
        phenotype="valine elemental carbon yield (% of maximum theoretical yield from glucose plus acetate)",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis+RBS",
        modality="RBS variant",
        why="New experimental dataset reported in this paper: 39 plasmids "
        "expressing ilvBN*DE and ilvIH*C/C*-ygaZH with varied RBS strengths, "
        "tested as 89 pairwise plasmid combinations in strain PYR003a "
        "(BW25113 delta-aceE delta-gdhA delta-poxB delta-ldhA delta-recA), "
        "plus 4 new combinations chosen by ActiveOpt, so 93 genotype-linked "
        "yields; best strain 45% elemental carbon yield, 54.7% of maximum "
        "theoretical. A minimum of two biological replicate colonies per "
        "experiment, 48 h shake flask endpoint, no time course.",
        accession="https://doi.org/10.1016/j.ymben.2021.06.009 (Supplementary Table S1: plasmid and RBS details; supplementary information lists all 89 experiments)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Yoshida 2012 organic acids",
            why="both carry organic acid output per strain as a quantitative label",
        ),
        synergy=[
            Synergy(
                partner="Yoshida 2012 organic acids",
                partner_status="supported",
                join="valine biosynthesis operon RBS space",
                yields="both carry organic acid output per strain as a quantitative "
                "label",
            )
        ],
    ),
    Candidate(
        name="Farasat 2014 neurosporene (ActiveOpt case 2)",
        organism="E. coli",
        citation="Farasat I, Kushwaha M, Collens J, Easterbrook M, Guido M, "
        "Salis HM. Efficient search, mapping, and optimization of "
        "multi-protein genetic systems in diverse bacteria. Molecular "
        "Systems Biology 2014",
        url="https://doi.org/10.15252/msb.20134955",
        klass="Combinatorial design",
        tier=4,
        genotypes_n=101,
        genotypes="101 RBS variants",
        env_n=1,
        env="1 condition",
        instances_n=101,
        instances_basis="reported",
        phenotype="specific neurosporene productivity (microgram per gCDW per hour)",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis+RBS",
        modality="RBS variant",
        why="Kept as a row separate from the valine campaign because the design "
        "space is independent (crtEBI RBS library, different host and "
        "readout). The neurosporene measurements are Farasat et al. 2014's, "
        "not new work: 73 designed crtEBI expression constructs measured as "
        "exploration experiments plus 28 kinetic-model-designed extrapolation "
        "constructs, productivity rising from 196.3 to 286 microgram per gCDW "
        "per hour. ActiveOpt (Kumar 2021, doi 10.1016/j.ymben.2021.06.009) is "
        "a retrospective reanalysis of that dataset and generated no new "
        "neurosporene strains, so the primary citation is Farasat 2014.",
        accession="https://doi.org/10.15252/msb.20134955 (supplementary datasets; 646 characterized genetic variants across six hosts)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene screen",
            why="both score a small-molecule product per engineered strain over a "
            "genome-scale perturbation panel",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene screen",
                partner_status="supported",
                join="carotenoid (crtEBI) pathway",
                yields="both score a small-molecule product per engineered strain "
                "over a genome-scale perturbation panel",
            )
        ],
    ),
    Candidate(
        name="Wang 2023 BATCH biosensor CRISPRi",
        organism="E. coli",
        citation="Wang J, Li C, Jiang T, Yan Y. Biosensor-assisted titratable "
        "CRISPRi high-throughput (BATCH) screening for over-production "
        "phenotypes. Metabolic Engineering 2023",
        url="https://doi.org/10.1016/j.ymben.2022.11.004",
        klass="CRISPR library screen",
        tier=4,
        genotypes_n=20,
        genotypes="20 guides",
        env_n=2,
        env="2 conditions",
        instances_n=40,
        instances_basis="estimate",
        phenotype="biosensor fluorescence sort, then titer (mg/L) on selected sgRNA variants",
        dim=1,
        dim_basis="estimate",
        seq_basis="engineered-chassis",
        modality="CRISPRi",
        why="Doubly mismatched sgRNA pools give graded knockdown of 20 "
        "central-carbon-metabolism genes, screened with a PadR-based "
        "p-coumaric acid biosensor and an HpdR-based butyrate biosensor: pfkA "
        "plus ptsI variants raised p-coumarate 40.6% to 1308.6 mg/L from "
        "glycerol, and sucA or ldhA variants raised butyrate 19.0% and 25.2%. "
        "The total number of mismatched sgRNA variants in the pools was not "
        "confirmed, and the pooled sort yields no per-variant phenotype "
        "except for the few reconstructed winners, hence blocked. Conditions "
        "2 counts the two product-biosensor campaigns.",
        accession="https://doi.org/10.1016/j.ymben.2022.11.004",
        accession_confirmed=False,
        status="blocked",
        confidence="sourced",
        analog=Analog(
            dataset="Mormino 2022 CRISPRi acetic acid",
            why="both read a biosensor response under graded knockdown rather "
            "than a growth score",
        ),
        synergy=[
            Synergy(
                partner="Mormino 2022 CRISPRi acetic acid",
                partner_status="supported",
                join="central carbon metabolism gene set",
                yields="both read a biosensor response under graded knockdown rather "
                "than a growth score",
            )
        ],
    ),
    Candidate(
        name="Wang 2018 phosphatase CRISPRi",
        organism="E. coli",
        citation="Wang T, Guo J, Liu Y, Xue Z, Zhang C, Xing XH. Genome-wide "
        "screening identifies promiscuous phosphatases impairing "
        "terpenoid biosynthesis in Escherichia coli. Applied "
        "Microbiology and Biotechnology 2018",
        url="https://doi.org/10.1007/s00253-018-9330-9",
        klass="CRISPR library screen",
        tier=4,
        genotypes_n=56,
        genotypes="56 guides",
        env_n=1,
        env="1 condition",
        instances_n=168,
        instances_basis="estimate",
        phenotype="lycopene content (and beta-carotene in follow-up strains)",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="CRISPRi",
        why="A CRISPRi knockdown library over 56 phosphatase-encoding genes built "
        "in a lycopene overproducer; 28 of the 56 knockdowns impaired "
        "lycopene synthesis, and combinatorial knockdowns plus knockouts were "
        "then tested in lycopene and beta-carotene overproducers. The gene "
        "set is small and fully named, so per-design values should be "
        "recoverable from the figures, but replicate counts were not verified "
        "so instances is an estimate.",
        accession="https://doi.org/10.1007/s00253-018-9330-9",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene screen",
            why="both score a small-molecule product per engineered strain over a "
            "genome-scale perturbation panel",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene screen",
                partner_status="supported",
                join="E. coli phosphatase gene set",
                yields="both score a small-molecule product per engineered strain "
                "over a genome-scale perturbation panel",
            )
        ],
    ),
    Candidate(
        name="Alper 2005 lycopene",
        organism="E. coli",
        citation="Alper H, Miyaoku K, Stephanopoulos G. Construction of "
        "lycopene-overproducing E. coli strains by combining systematic "
        "and combinatorial gene knockout targets. Nature Biotechnology "
        "2005",
        url="https://doi.org/10.1038/nbt1083",
        klass="Combinatorial design",
        tier=4,
        genotypes_n=64,
        genotypes="64 designs",
        env_n=1,
        env="1 condition",
        instances_n=192,
        instances_basis="estimate",
        phenotype="lycopene content per cell dry weight",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="mixed",
        why="Two independent target-discovery routes, a genome-scale-model "
        "systematic search and a transposon-insertion combinatorial search, "
        "produced two disjoint gene sets whose exhaustive combination gave a "
        "defined set of 64 knockout strains, the best 8.5-fold over "
        "recombinant K-12 wild type and 2-fold over the engineered parent. "
        "The deletion set is in a K-12 background that also carries the "
        "heterologous lycopene pathway; transposon insertions were used for "
        "discovery, not as the final genotypes. Replicate counts were not "
        "verified.",
        accession="https://doi.org/10.1038/nbt1083",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene screen",
            why="both score a small-molecule product per engineered strain over a "
            "genome-scale perturbation panel",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene screen",
                partner_status="supported",
                join="K-12 gene set",
                yields="both score a small-molecule product per engineered strain "
                "over a genome-scale perturbation panel",
            )
        ],
    ),
    Candidate(
        name="Fong 2005 growth-coupled lactate",
        organism="E. coli",
        citation="Fong SS, Burgard AP, Herring CD, Knight EM, Blattner FR, "
        "Maranas CD, Palsson BO. In silico design and adaptive "
        "evolution of Escherichia coli for production of lactic acid. "
        "Biotechnology and Bioengineering 2005",
        url="https://doi.org/10.1002/bit.20542",
        klass="Production campaign",
        tier=4,
        genotypes_n=3,
        genotypes="3 deletions",
        env_n=1,
        env="1 condition",
        instances_n=11,
        instances_basis="reported",
        phenotype="lactate titer (g/L), growth rate, lactate secretion rate",
        dim=3,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="Three OptKnock-designed knockout strains were built and adaptively "
        "evolved to yield 11 evolved lactate-producing strains, growing on 2 "
        "g/L glucose at 37 C with titers 0.87 to 1.75 g/L and secretion rates "
        "coupled to growth rate; computational post-evolution growth-rate "
        "predictions were within 10% in 38 of 50 cases. Only 3 distinct "
        "engineered genotypes exist, so this is a small validation campaign "
        "rather than a library.",
        accession="https://doi.org/10.1002/bit.20542",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Yoshida 2012 organic acids",
            why="both carry organic acid output per strain as a quantitative label",
        ),
        synergy=[
            Synergy(
                partner="Yoshida 2012 organic acids",
                partner_status="supported",
                join="MG1655 gene set",
                yields="both carry organic acid output per strain as a quantitative "
                "label",
            )
        ],
    ),
    Candidate(
        name="Jain 2015 1,2-propanediol",
        organism="E. coli",
        citation="Jain R, Sun X, Yuan Q, Yan Y. Systematically engineering "
        "Escherichia coli for enhanced production of 1,2-propanediol "
        "and 1-propanol. ACS Synthetic Biology 2015",
        url="https://doi.org/10.1021/sb500345t",
        klass="Production campaign",
        tier=4,
        genotypes_n=20,
        genotypes="20 designs",
        env_n=1,
        env="1 condition",
        instances_n=60,
        instances_basis="estimate",
        phenotype="1,2-propanediol titer (g/L) and yield (g/g glucose)",
        dim=2,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="mixed",
        why="I could not confirm the existence of a distinct growth-coupled "
        "1,2-propanediol design-validation campaign: repeated searches "
        "surfaced only computational growth-coupling frameworks (OptKnock, "
        "minimal cut sets, gcFront) plus separate 1,2-PDO engineering papers, "
        "so the 1,2-propanediol row is filled by the best-documented "
        "reconstructable E. coli campaign, which reached 5.13 g/L and 0.48 "
        "g/g glucose via an optimal minimal enzyme set, flux channeling, "
        "raised NADH availability and improved anaerobic growth. The citation "
        "is sourced; the genotype and instance counts are unverified "
        "estimates, hence confidence recall.",
        accession="https://doi.org/10.1021/sb500345t",
        accession_confirmed=False,
        status="candidate",
        confidence="recall",
        analog=Analog(
            dataset="Lopez 2024 isobutanol screen",
            why="both carry an alcohol titer per perturbed strain, which is the "
            "same record shape with a different reference genome",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol screen",
                partner_status="supported",
                join="methylglyoxal / glycolysis gene set",
                yields="both carry an alcohol titer per perturbed strain, which is "
                "the same record shape with a different reference genome",
            )
        ],
    ),
    Candidate(
        name="MCF2Chem 2023",
        organism="E. coli",
        citation="Cai P, Liu S, Zhang D, Hu QN. MCF2Chem: A manually curated "
        "knowledge base of biosynthetic compound production. "
        "Biotechnology for Biofuels and Bioproducts 2023",
        url="https://doi.org/10.1186/s13068-023-02419-8",
        klass="Aggregation / support",
        tier=3,
        genotypes_n=8888,
        genotypes="8,888 designs",
        env_n=1,
        env="1 condition",
        instances_n=8888,
        instances_basis="reported",
        phenotype="titer, yield, productivity, content per literature record",
        dim=4,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="mixed",
        why="8888 production records covering 1231 compounds in 590 microbial "
        "cell factories, with titer, yield, productivity and content plus "
        "strain culture information; bacteria are 60% of species, and E. "
        "coli, S. cerevisiae, Y. lipolytica and C. glutamicum together "
        "account for 78% of the compounds, with E. coli alone producing about "
        "a quarter of the compounds among the top 20 species. No exact E. "
        "coli record count is published, so the E. coli subset size could not "
        "be determined. These are counts of literature records, not measured "
        "strain-condition observations, and must be resolved to primary "
        "source DOIs before they count as independent datasets.",
        accession="https://mcf.lifesynther.com",
        accession_confirmed=False,
        status="aggregation",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene screen",
            why="both score a small-molecule product per engineered strain over a "
            "genome-scale perturbation panel",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene screen",
                partner_status="supported",
                join="host x compound x titer",
                yields="both score a small-molecule product per engineered strain "
                "over a genome-scale perturbation panel",
            )
        ],
    ),
    Candidate(
        name="Carruthers 2025",
        organism="P. putida",
        citation="Carruthers DN, Kinnunen PC, Li Y, Chen Y, Gin JW, Yunus IS, et "
        "al. Automation and machine learning drive rapid optimization "
        "of isoprenol production in Pseudomonas putida. Nature "
        "Communications 2025",
        url="https://doi.org/10.1038/s41467-025-66304-8",
        klass="Production campaign",
        tier=3,
        genotypes_n=472,
        genotypes="472 guides",
        env_n=1,
        env="1 condition",
        instances_n=1416,
        instances_basis="product",
        phenotype="isoprenol titer by GC-FID plus paired DIA shotgun proteome",
        dim=1500,
        dim_basis="reported",
        seq_basis="KT2440+guide",
        modality="CRISPRi",
        why="The full text states 472 unique strains (125 single perturbations "
        "and 347 combinations) over six DBTL cycles in triplicate, matching "
        "the recalled numbers exactly; machine learning selected roughly 400 "
        "constructs out of 800,000 possibilities for a fivefold titer gain. "
        "ONE study with eight linked deposits: Dryad 10.5061/dryad.gtht76hzh "
        "(proteomic metadata plus a 29.7 MB Top3 peptide quantification "
        "table, file list enumerated), a Zenodo and GitHub release of "
        "per-cycle titer tables (10.5281/zenodo.17178684, "
        "JBEI/Isoprenol_CRISPRi), and seven per-cycle PRIDE accessions "
        "PXD063733 (DBTL0), PXD063737, PXD063738, PXD063740, PXD063743, "
        "PXD063744, PXD063746 (DBTL6). An ingestion that expects one PXD per "
        "paper silently drops six sevenths of the proteomics.",
        accession="https://doi.org/10.5061/dryad.gtht76hzh",
        accession_confirmed=True,
        status="candidate",
        isoprenol="direct",
        product="isoprenol",
        confidence="sourced",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="multiplexed guide combinations scored by product",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="isoprenol pathway (IPP-bypass) in KT2440",
                yields="multiplexed guide combinations scored by product",
            )
        ],
    ),
    Candidate(
        name="Banerjee 2024",
        organism="P. putida",
        citation="Banerjee D, Yunus IS, Wang X, Kim J, Srinivasan A, Menchavez "
        "R, et al. Genome-scale and pathway engineering for the "
        "sustainable aviation fuel precursor isoprenol production in "
        "Pseudomonas putida. Metabolic Engineering 2024",
        url="https://doi.org/10.1016/j.ymben.2024.02.004",
        klass="Production campaign",
        tier=4,
        genotypes_n=19,
        genotypes="19 deletions",
        env_n=3,
        env="3 conditions",
        instances_n=171,
        instances_basis="estimate",
        phenotype="isoprenol titer plus growth and glucose consumption, shake flask and fed-batch",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440-KO",
        modality="gene deletion",
        why="The recalled 3.5 g/L fed-batch figure is confirmed in the abstract, "
        "against 1.1 g/L in batch and over tenfold above the starting strain. "
        "Eight deletion targets came from bilevel optimization and "
        "constrained minimal cut sets on iJN1462, and the best strain carries "
        "five deletions. The strain count is an estimate read from the "
        "preprint because the published version is paywalled. Proteomics here "
        "is targeted MRM of pathway enzymes only, not a discovery panel, so "
        "dimensionality stays 1. Linked assets: PXD039868 on PanoramaPublic "
        "plus six Experimental Data Depot studies at edd.jbei.org, which were "
        "not opened.",
        accession="https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD039868",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="model-guided deletion panel, alcohol titer",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="isoprenol pathway (IPP-bypass) in KT2440",
                yields="model-guided deletion panel, alcohol titer",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Menasalvas 2025",
        organism="P. putida",
        citation="Menasalvas J, Kulakowski S, Chen Y, Gin JW, Akyuz Turumtay E, "
        "Baral NR, et al. Biosensor-driven strain engineering reveals "
        "key cellular processes for maximizing isoprenol production in "
        "Pseudomonas putida. Science Advances 2025",
        url="https://doi.org/10.1126/sciadv.ady2677",
        klass="Production campaign",
        tier=3,
        genotypes_n=165,
        genotypes="165 deletions",
        env_n=2,
        env="2 conditions",
        instances_n=1320,
        instances_basis="product",
        phenotype="isoprenol titer by GC-FID after biosensor-coupled pyrF growth selection",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="gene deletion",
        why="ATTRIBUTION CORRECTION: no author named Teng is on this paper. The "
        "Dryad DOI belongs to Menasalvas et al. with Thomas Eng as "
        "co-corresponding author, so the recalled Teng is almost certainly "
        "Eng. Over 165 deletion and overexpression strains across 70 gene "
        "loci, quadruplicate, titers at 24 and 48 h, reaching about 900 mg/L "
        "(36-fold). ONE study, four linked deposits, all confirmed: Dryad "
        "10.5061/dryad.sbcc2frjq (2 files, 78.98 MB), PRIDE PXD061547 (An "
        "Isoprenol Biosensor for Combinatorial High Throughput Strain "
        "Engineering in Pseudomonas putida, 97 raw files with 8, 24 and 48 h "
        "timepoints), BioProject PRJNA1226229 (3 WGS runs), Zenodo "
        "10.5281/zenodo.17155686 (AlphaFold output only). IMPORTANT on the "
        "CRISPRi library question: the paper does contain a pooled dCpf1 "
        "library of about 16,500 guides (3 per gene over 5,591 coding "
        "sequences), but there is NO released per-guide fitness or abundance "
        "matrix. Dryad holds only baseline per-guide read counts, a "
        "lost-guide list, per-round enriched-guide hit lists, and "
        "representative rather than per-replicate ONT reads, so a "
        "guide-by-sample matrix cannot be reconstructed. Proteomics covers "
        "about 2,500 to 3,000 proteins on a five-context subset, not every "
        "strain, so dimensionality is kept at 1.",
        accession="https://doi.org/10.5061/dryad.sbcc2frjq",
        accession_confirmed=True,
        status="candidate",
        isoprenol="direct",
        product="isoprenol",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="iterative engineered panel, alcohol titer",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="isoprenol pathway (IPP-bypass) in KT2440",
                yields="iterative engineered panel, alcohol titer",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Lim 2025 isoprenol TALE",
        organism="P. putida",
        citation="Lim HG, Srinivasan A, Menchavez R, Yunus IS, Noh MH, White M, "
        "et al. Evolution-guided tolerance engineering of Pseudomonas "
        "putida KT2440 for production of the aviation fuel precursor "
        "isoprenol. Metabolic Engineering 2025",
        url="https://doi.org/10.1016/j.ymben.2025.05.007",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=46,
        genotypes="46 resequenced clones",
        env_n=2,
        env="2 conditions",
        instances_n=114,
        instances_basis="product",
        phenotype="maximum specific growth rate under isoprenol challenge plus isoprenol titer with the pIY670 plasmid",
        dim=1,
        dim_basis="reported",
        seq_basis="evolved-WGS",
        modality="laboratory evolution",
        why="The recalled design is correct and reported: WT, IPL300 and IPL400 "
        "each evolved in four independent lineages (12 isoprenol lineages), "
        "plus four extra WT lineages tolerized against formaldehyde as "
        "controls, so 16 lineages total, isoprenol ramped 4 to 8.5 g/L over "
        "143 to 353 generations. Only the 46 whole-genome resequenced strains "
        "(16 endpoint clones plus about 30 intermediates) have a "
        "reconstructable genotype; the evolved populations themselves do not "
        "and are blocked at the genotype level. 158 unique mutations across "
        "73 regions; proteomics quantified an average of 2,375 of 5,565 "
        "putative proteins. Linked assets: PXD054609 (confirmed, citing this "
        "PMID), GEO GSE281392, and ALEdb project Pputida_isoprenol_TALE. "
        "IMPORTANT: the stated resequencing accession PRJNA1187681 does NOT "
        "resolve at NCBI BioProject, NCBI SRA or ENA as of 2026-09-24, so the "
        "raw reads are not retrievable; the mutation calls survive only in "
        "Supplementary Dataset 1.",
        accession="https://www.ebi.ac.uk/pride/archive/projects/PXD054609",
        accession_confirmed=True,
        status="candidate",
        isoprenol="direct",
        product="isoprenol",
        confidence="sourced",
        analog=Analog(
            dataset="Mormino 2022 CRISPRi acetic acid",
            why="inhibitor tolerance scored by growth",
        ),
        synergy=[
            Synergy(
                partner="Mormino 2022 CRISPRi acetic acid",
                partner_status="supported",
                join="isoprenol pathway (IPP-bypass) in KT2440",
                yields="inhibitor tolerance scored by growth",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="de Siqueira 2025",
        organism="P. putida",
        citation="de Siqueira GMV, Srinivasan A, Chen Y, Gin JW, Petzold CJ, Lee "
        "TS, Guazzaroni ME, Eng T, Mukhopadhyay A. Alternate routes to "
        "acetate tolerance lead to varied isoprenol production from "
        "mixed carbon sources in Pseudomonas putida. Applied and "
        "Environmental Microbiology 2025",
        url="https://doi.org/10.1128/aem.02123-24",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=7,
        genotypes="7 resequenced clones",
        env_n=3,
        env="3 conditions",
        instances_n=63,
        instances_basis="product",
        phenotype="growth rate on glucose-acetate mixtures plus isoprenol titer by GC-FID",
        dim=1,
        dim_basis="estimate",
        seq_basis="evolved-WGS",
        modality="laboratory evolution",
        why="Both recalled accessions verified by fetching the repository "
        "records. PXD055153 is titled Restoration of growth and isoprenol "
        "production in dual carbon source media by an acetate-sensitive "
        "Pseudomonas putida strain, submitted 2024-08-23, linked to this DOI. "
        "PRJNA1153078 (Pseudomonas putida KT2440 tolerization on acetate and "
        "isoprenol) holds exactly 5 SRA experiments and 5 BioSamples, "
        "matching the 5 resequenced isolates, four tolerized Sigma-class "
        "strains plus the parental producer; 2 of the 7 characterized strains "
        "were not resequenced and are blocked at the genotype level. ONE "
        "study with linked assets PXD055153 plus PRJNA1153078. Proteome panel "
        "size is never stated numerically, so dimensionality is kept at 1 "
        "rather than guessed.",
        accession="https://www.ebi.ac.uk/pride/archive/projects/PXD055153",
        accession_confirmed=True,
        status="candidate",
        isoprenol="direct",
        product="isoprenol",
        confidence="sourced",
        analog=Analog(
            dataset="Mormino 2022 CRISPRi acetic acid",
            why="acetate tolerance with a product readout",
        ),
        synergy=[
            Synergy(
                partner="Mormino 2022 CRISPRi acetic acid",
                partner_status="supported",
                join="isoprenol pathway (IPP-bypass) in KT2440",
                yields="acetate tolerance with a product readout",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Borchert 2024 fModules",
        organism="P. putida",
        citation="Borchert AJ, Bleem AC, Lim HG, Rychel K, Dooley KD, Kellermyer "
        "ZA, Hodges TL, Palsson BO, Beckham GT. Machine learning "
        "analysis of RB-TnSeq fitness data predicts functional gene "
        "modules in Pseudomonas putida KT2440. mSystems 2024",
        url="https://doi.org/10.1128/msystems.00942-23",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=4732,
        genotypes="4,732 insertion mutants",
        env_n=179,
        env="179 conditions",
        instances_n=1571024,
        instances_basis="reported",
        phenotype="gene-level RB-TnSeq relative fitness, log2 barcode ratio against the time-zero baseline",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="transposon insertion",
        why="Every recalled number checks out, and the 179 versus 183 discrepancy "
        "is real rather than an error: Methods say 183 unique growth "
        "conditions were collected, the Abstract says 179 were analyzed, and "
        "the released workbook's classifier column has exactly 179 distinct "
        "values, so 183 is the collection and 179 is the analyzed matrix. "
        "Fitness for 4,732 of 5,564 protein-coding genes. The instance count "
        "is VERIFIED, not a product: fModule_Metadata.xlsx (23.3 MB) was "
        "opened and its fitness sheet counted at 4,732 rows by 332 value "
        "columns with zero nulls, alongside a t-statistic sheet of identical "
        "shape. This row is an AGGREGATION and a superset of the other "
        "transposon rows: 254 of 332 samples were pulled from the Fitness "
        "Browser and 78 are the Beckham-group sets (Putida_ML5_set100 and "
        "set101) deposited as SRA PRJNA809672 (15 runs), PRJNA856070 (57 "
        "runs) and PRJNA1011287 (39 runs), all counted. Replicate structure: "
        "194 samples from duplicate conditions, 78 from triplicate, 60 "
        "singletons. The environment axis is two-slot (condition_1 plus "
        "condition_2 with units), which is what keeps glucose-plus-stressor "
        "tolerance samples distinguishable from sole-carbon catabolism "
        "samples; collapsing it to one string destroys that contrast. Caveat: "
        "genes lacking a value in any sample were dropped, so "
        "condition-specific genes reported by the primary papers are silently "
        "absent here.",
        accession="https://github.com/beckham-lab/fModule",
        accession_confirmed=True,
        status="aggregation",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant fitness across hundreds of conditions",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Putida_ML5 / JBEI-1 RB-TnSeq library",
                yields="pooled mutant fitness across hundreds of conditions",
            )
        ],
    ),
    Candidate(
        name="Lim 2022 putidaPRECISE321",
        organism="P. putida",
        citation="Lim HG, Rychel K, Sastry AV, Bentley GJ, Mueller J, Schindel "
        "HS, et al. Machine-learning from Pseudomonas putida KT2440 "
        "transcriptomes reveals its transcriptional regulatory network. "
        "Metabolic Engineering 2022",
        url="https://doi.org/10.1016/j.ymben.2022.04.004",
        klass="Transcriptome",
        tier=4,
        genotypes_n=17,
        genotypes="17 wild type",
        env_n=118,
        env="118 conditions",
        instances_n=321,
        instances_basis="reported",
        phenotype="batch-normalized log2 TPM RNA-seq expression, decomposed into 84 iModulons",
        dim=5564,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="environment-only",
        why="All recalled numbers confirmed twice over, from the paper and by "
        "recomputation on the released matrix: 321 samples, 118 unique "
        "condition groups, 21 projects, a 5,564 gene by 321 sample "
        "log_tpm_norm matrix, 84 iModulons explaining 75.7 percent of "
        "variance with 1,265 member genes. The live iModulonDB API returns "
        "the dataset under organism p_putida, dataset precise321. Replicate "
        "structure recomputed as 40 conditions at n=2, 71 at n=3, 7 at n=4, "
        "summing to 321 exactly. Gene-level values total 1,786,044. This row "
        "is an AGGREGATION of 21 projects: 541 samples were collected and 321 "
        "passed QC, of which the paper says 305 were previously published and "
        "16 newly generated. The genotype count of 17 distinct genotype "
        "tuples (10 with an annotated perturbation such as relA, finR, crc, "
        "fleQ) is an UNDERCOUNT, because engineered muconate producers and "
        "ALE clones are encoded in condition strings rather than genotype "
        "fields. There is no umbrella GEO accession: the sample table carries "
        "47 distinct SRA or BioProject accessions covering 216 of 321 samples "
        "and 9 GEO Series, and 105 samples carry no accession at all, so "
        "per-sample raw-data provenance is incomplete by construction.",
        accession="https://github.com/SBRG/modulome_ppu",
        accession_confirmed=True,
        status="aggregation",
        confidence="sourced",
        analog=Analog(
            dataset="Caudal 2024 pan-transcriptome",
            why="aggregated RNA-seq compendium across many backgrounds",
        ),
        synergy=[
            Synergy(
                partner="Caudal 2024 pan-transcriptome",
                partner_status="supported",
                join="KT2440 gene set",
                yields="aggregated RNA-seq compendium across many backgrounds",
            )
        ],
    ),
    Candidate(
        name="Banerjee 2020 indigoidine",
        organism="P. putida",
        citation="Banerjee D, Eng T, Lau AK, Sasaki Y, Wang B, Chen Y, et al. "
        "Genome-scale metabolic rewiring improves titers rates and "
        "yields of the non-native product indigoidine at scale. Nature "
        "Communications 2020",
        url="https://doi.org/10.1038/s41467-020-19171-4",
        klass="Production campaign",
        tier=4,
        genotypes_n=2,
        genotypes="2 guides",
        env_n=8,
        env="8 conditions",
        instances_n=48,
        instances_basis="estimate",
        phenotype="indigoidine titer, rate and yield plus growth, scaled from deep-well plate to 2 L bioreactor",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+guide",
        modality="CRISPRi",
        why="This is the growth-coupled indigoidine CRISPRi campaign, and it "
        "needs a CORRECTION to the recalled framing. The counts are right but "
        "the object is not: 63 constrained-minimal-cut-set solutions were "
        "analyzed, one feasible cut set hit 14 metabolic reactions, which "
        "mapped through GPRs to 16 single-copy genes, of which 2 (mqo-I and "
        "cynT) were dropped as essential by genome-wide RB-TnSeq, leaving 14 "
        "genes knocked down. Those 14 guides sit in ONE multiplexed dCpf1 "
        "gRNA array in a SINGLE strain. There are not 14 CRISPRi strains, so "
        "the genotype axis here is 2 (engineered versus empty-vector "
        "control), not 14, and anything that ingests this as a 14-strain "
        "panel is wrong. Conditions are 2 carbon sources crossed with 4 "
        "culture formats. RNA-seq is genome-wide (about 5,500 genes, "
        "estimated from the annotation, not stated); the proteomics is "
        "targeted SRM, so it is a selected reaction list rather than a "
        "discovery panel. BioProjects PRJNA580539 through PRJNA580574 (both "
        "endpoints resolve; the paper typos the upper bound); proteomics at "
        "PanoramaWeb, not opened.",
        accession="https://www.ncbi.nlm.nih.gov/bioproject/PRJNA580539",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="multiplexed knockdowns boosting a pigment product",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="KT2440 gene set",
                yields="multiplexed knockdowns boosting a pigment product",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Czajka 2022",
        organism="P. putida",
        citation="Czajka JJ, Banerjee D, Eng T, Menasalvas J, Yan C, Munoz NM, "
        "et al. Tuning a high performing multiplexed-CRISPRi "
        "Pseudomonas putida strain to further enhance indigoidine "
        "production. Metabolic Engineering Communications 2022",
        url="https://doi.org/10.1016/j.mec.2022.e00206",
        klass="Production campaign",
        tier=4,
        genotypes_n=6,
        genotypes="6 guides",
        env_n=1,
        env="1 condition",
        instances_n=36,
        instances_basis="product",
        phenotype="indigoidine titer by A612 in DMSO plus growth rate",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+guide",
        modality="CRISPRi",
        why="The direct follow-up to Banerjee 2020 on the same 14-target "
        "multiplexed CRISPRi strain, adding an optimized Cpf1 ribosome "
        "binding site for a 1.6-fold gain, PHA-operon deletion strains, "
        "RNA-seq, targeted proteomics and 13C metabolic flux analysis. Six "
        "strains carry titers in one condition (M9, 10 g/L glucose, arabinose "
        "plus IPTG), n at least 6 for production profiling, sampled at 24, "
        "48, 72 and 120 h. Kept as its own row rather than folded into "
        "Banerjee 2020 because the genotypes and the readout differ. Data "
        "availability names only public-registry.jbei.org with no folder "
        "number and states explicitly that everything else is in the "
        "manuscript and supplementary files, so there is no GEO, SRA, PRIDE "
        "or MassIVE deposit and nothing is machine-retrievable.",
        accession="https://pmc.ncbi.nlm.nih.gov/articles/PMC9494242/",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="knockdown tuning scored by pigment titer",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="KT2440 gene set",
                yields="knockdown tuning scored by pigment titer",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Banerjee 2025",
        organism="P. putida",
        citation="Banerjee D, Menasalvas J, Chen Y, Gin JW, Baidoo EEK, Petzold "
        "CJ, Eng T, Mukhopadhyay A. Addressing genome scale design "
        "tradeoffs in Pseudomonas putida for bioconversion of an "
        "aromatic carbon source. npj Systems Biology and Applications "
        "2025",
        url="https://doi.org/10.1038/s41540-024-00480-z",
        klass="Production campaign",
        tier=4,
        genotypes_n=12,
        genotypes="12 promoter variants",
        env_n=3,
        env="3 conditions",
        instances_n=36,
        instances_basis="estimate",
        phenotype="indigoidine titer from p-coumarate plus growth rate, with a global DIA proteome",
        dim=2000,
        dim_basis="reported",
        seq_basis="KT2440+promoter",
        modality="promoter variant",
        why="This is the recalled genome-scale design-tradeoff campaign, and it "
        "is the cleanest promoter-variant genotype axis I found for KT2440: a "
        "four-gene growth-coupling cutset for p-coumarate to glutamine to "
        "indigoidine, whose fourth gene PP_0897 (a fumarate hydratase isomer) "
        "proved multifunctional and rate-limiting, so its native promoter was "
        "swapped for the Anderson-collection pJ23109 or the weak PP_0415 "
        "promoter. PXD050285 verified two ways (ProteomeCentral and the PRIDE "
        "v3 API) and its files enumerated: 26 raw MS runs, Orbitrap Exploris "
        "480, DIA searched library-free with DIA-NN at 1 percent global FDR, "
        "comprising an 8-run promoter-variant arm (pJ and 0415 driving "
        "PP_0897, 4 replicates each) and an 18-run cross-feeding arm (strains "
        "2370 and 2487 across M9 alanine, M9 alanine malate, M9 p-CA alanine "
        "malate, n=3). The about 2,000 quantified proteins is REPORTED "
        "verbatim in the paper, not estimated. A separate 190-substrate "
        "BIOLOG grid in the same paper is a much larger environment axis if "
        "it is ingested. Note the PRIDE submission title differs from the "
        "published title; both are correct for their object.",
        accession="https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD050285",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Zelezniak 2018 proteome",
            why="engineered panel paired with quantitative proteomes",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 proteome",
                partner_status="supported",
                join="KT2440 gene set",
                yields="engineered panel paired with quantitative proteomes",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Yunus 2026",
        organism="P. putida",
        citation="Yunus IS, Carruthers DN, Chen Y, Gin JW, Baidoo EEK, Petzold "
        "CJ, Garcia Martin H, Adams PD, Mukhopadhyay A, Lee TS. "
        "Predictive CRISPR-mediated gene downregulation for enhanced "
        "production of sustainable aviation fuel precursor in "
        "Pseudomonas putida. Metabolic Engineering 2026",
        url="https://doi.org/10.1016/j.ymben.2025.11.007",
        klass="Production campaign",
        tier=4,
        genotypes_n=20,
        genotypes="20 guides",
        env_n=1,
        env="1 condition",
        instances_n=60,
        instances_basis="estimate",
        phenotype="isoprenol titer under predicted CRISPRi knockdowns",
        dim=1,
        dim_basis="estimate",
        seq_basis="KT2440+guide",
        modality="CRISPRi",
        why="Found while sweeping for CRISPRi isoprenol work and it clearly "
        "belongs: FluxRETAP and VAMMPIRE were used to pick downregulation "
        "targets and to assemble arrays of up to five sgRNAs, reaching nearly "
        "1.5 g/L isoprenol by knocking down PP_4118 (alpha-ketoglutarate "
        "dehydrogenase). Bibliography verified through NCBI eutils. Strain "
        "and instance counts are estimates: the abstract does not state them "
        "and the full text is paywalled. No accession is given in the indexed "
        "metadata, so data_location is deliberately empty rather than "
        "guessed. Shares authors and the pIY670-era pathway with Carruthers "
        "2025 and Banerjee 2024, so genotype deduplication against those rows "
        "is required.",
        accession="",
        accession_confirmed=False,
        status="candidate",
        isoprenol="direct",
        product="isoprenol",
        confidence="sourced",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="predicted multiplexed knockdowns scored by titer",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="isoprenol pathway (IPP-bypass) in KT2440",
                yields="predicted multiplexed knockdowns scored by titer",
            )
        ],
    ),
    Candidate(
        name="Yu 2016 PHBA",
        organism="P. putida",
        citation="Yu S, Plan MR, Winter G, Kromer JO. Metabolic Engineering of "
        "Pseudomonas putida KT2440 for the Production of para-Hydroxy "
        "Benzoic Acid. Frontiers in Bioengineering and Biotechnology "
        "2016",
        url="https://doi.org/10.3389/fbioe.2016.00090",
        klass="Production campaign",
        tier=4,
        genotypes_n=7,
        genotypes="7 deletions",
        env_n=1,
        env="1 condition",
        instances_n=21,
        instances_basis="product",
        phenotype="para-hydroxybenzoic acid titer from glucose via chorismate, plus carbon yield",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440-KO",
        modality="gene deletion",
        why="The recalled six strains plus wild type is exact: S0 wild-type "
        "control and S1 through S6, built from markerless deletions of pobA, "
        "pheA, trpE and hexR combined with plasmid-borne E. coli ubiC and "
        "feedback-resistant aroG-D146N; S6 reached 1.73 g/L at 18.1 percent "
        "C-mol per C-mol. One defined medium, one 44 h endpoint, biological "
        "triplicates, which makes it the cheapest row here to ingest. A "
        "single S6 fed-batch run adds a short time series. Up to eight "
        "co-measured scalars exist per instance (PHBA, biomass, glucose, "
        "acetate, alpha-ketogluconate, pyruvate, lactate, succinate) if each "
        "counts as a phenotype. No proteomics, no transcriptomics, no "
        "repository deposit of any kind, so accession_confirmed is false "
        "because no accession exists to confirm.",
        accession="https://pmc.ncbi.nlm.nih.gov/articles/PMC5124731/",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene",
            why="small rational deletion series scored by product",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene",
                partner_status="supported",
                join="KT2440 gene set",
                yields="small rational deletion series scored by product",
            )
        ],
    ),
    Candidate(
        name="Thompson 2019 valerolactam",
        organism="P. putida",
        citation="Thompson MG, Valencia LE, Blake-Hedges JM, Cruz-Morales P, "
        "Velasquez AE, Pearson AN, et al. Omics-driven identification "
        "and elimination of valerolactam catabolism in Pseudomonas "
        "putida KT2440 for increased product titer. Metabolic "
        "Engineering Communications 2019",
        url="https://doi.org/10.1016/j.mec.2019.e00098",
        klass="Transposon fitness",
        tier=2,
        genotypes_n=4778,
        genotypes="4,778 insertion mutants",
        env_n=2,
        env="2 conditions",
        instances_n=9556,
        instances_basis="product",
        phenotype="gene-level RB-TnSeq fitness on lactam carbon sources, then valerolactam titer in deletion strains",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="transposon insertion",
        why="Two selective RB-TnSeq conditions, 10 mM valerolactam and 10 mM "
        "5-aminovalerate in MOPS minimal medium, with three time-zero "
        "aliquots; the two 500 uL aliquots per condition are pooling, not "
        "replication, so the paper reports no replicate count. RB-TnSeq is "
        "the small half of this paper; the payload worth ingesting is the "
        "engineering ladder it produced, wild type 0 mg/L to delta-oplBA 9.27 "
        "to delta-oplBA delta-davT 85.19 to delta-oplBA delta-davT delta-alr "
        "91.97 mg/L valerolactam at 48 h, with strains at "
        "public-registry.jbei.org folder 456. Fitness lives only in the "
        "Fitness Browser with no per-experiment accession, and that site is "
        "behind a Cloudflare managed challenge to both WebFetch and curl, so "
        "accession_confirmed is false. The two valerolactam carbon-source "
        "samples are however confirmed to exist, appearing by name "
        "(set6IT064, set7IT045) in the Borchert 2024 compendium workbook.",
        accession="https://fit.genomics.lbl.gov/cgi-bin/org.cgi?orgId=Putida",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant fitness on defined carbon sources",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Putida_ML5 / JBEI-1 RB-TnSeq library",
                yields="pooled mutant fitness on defined carbon sources",
            )
        ],
    ),
    Candidate(
        name="Thompson 2019 lysine",
        organism="P. putida",
        citation="Thompson MG, Blake-Hedges JM, Cruz-Morales P, Barajas JF, "
        "Curran SC, Eiben CB, et al. Massively Parallel Fitness "
        "Profiling Reveals Multiple Novel Enzymes in Pseudomonas putida "
        "Lysine Metabolism. mBio 2019",
        url="https://doi.org/10.1128/mbio.02577-18",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=4778,
        genotypes="4,778 insertion mutants",
        env_n=4,
        env="4 conditions",
        instances_n=19112,
        instances_basis="product",
        phenotype="gene-level RB-TnSeq fitness on glucose, 5-aminovalerate, D-lysine and L-lysine",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="transposon insertion",
        why="This is where the name JBEI-1 is defined, verbatim: the JBEI-1 "
        "library was created by diluting a 1 mL aliquot of the previously "
        "described P. putida RB-TnSeq library (Rand 2017) into 500 mL of LB "
        "plus kanamycin. So JBEI-1 is a regrown working stock of Putida_ML5, "
        "not a separate library, which is why every transposon row here "
        "shares one join key. Four sole-carbon conditions at 10 mM in MOPS "
        "minimal medium, 48 h, with three time-zero aliquots; Methods state "
        "no per-condition replicate count. The paper reports 39 genes with "
        "fitness below -2 at absolute t above 4. Fitness data are said to be "
        "publicly available at fit.genomics.lbl.gov with no per-experiment "
        "accession, and that site returns a Cloudflare challenge, so "
        "unconfirmed. Strains and plasmids at public-registry.jbei.org folder "
        "391.",
        accession="https://fit.genomics.lbl.gov/cgi-bin/org.cgi?orgId=Putida",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled deletion fitness across nutrient conditions",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Putida_ML5 / JBEI-1 RB-TnSeq library",
                yields="pooled deletion fitness across nutrient conditions",
            )
        ],
    ),
    Candidate(
        name="Thompson 2020 fatty acid and alcohol",
        organism="P. putida",
        citation="Thompson MG, Incha MR, Pearson AN, Schmidt M, Sharpless WA, "
        "Eiben CB, et al. Fatty Acid and Alcohol Metabolism in "
        "Pseudomonas putida: Functional Analysis Using Random Barcode "
        "Transposon Sequencing. Applied and Environmental Microbiology "
        "2020",
        url="https://doi.org/10.1128/aem.01665-20",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=4778,
        genotypes="4,778 insertion mutants",
        env_n=23,
        env="23 conditions",
        instances_n=219788,
        instances_basis="product",
        phenotype="gene-level RB-TnSeq fitness on fatty acid and alcohol carbon sources",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="transposon insertion",
        why="The recalled 13 fatty acids and 10 alcohols is confirmed verbatim in "
        "the abstract and the compounds were enumerated: straight-chain C3 to "
        "C10 plus C12 and C14, the esters Tween 20 and butyl stearate, oleic "
        "acid; and ethanol, butanol, pentanol, 1,2-propanediol, "
        "1,3-butanediol, 1,4-butanediol, 1,5-pentanediol, isopentanol, "
        "isoprenol, 2-methyl-1-butanol. Biological duplicate throughout, "
        "corroborated by the compendium metadata showing exactly 2 samples "
        "per compound, hence 23 x 2 x 4,778; the condition-collapsed count is "
        "about 109,900. Directly relevant to the isoprenol rows: this paper "
        "postulates the catabolic routes for isoprenol and isopentanol "
        "through leucine metabolism after oxidation and CoA activation, which "
        "is the product-loss pathway the isoprenol producers delete. A "
        "published correction exists (AEM 87(8):e00177-21, DOI "
        "10.1128/aem.00177-21). MEASURED SUBSUMPTION 2026-10-08 and it is "
        "TOTAL, so this row is a provenance record and NOT a loader "
        "(experiments/036-dataset-fixes-before-kg-build/scripts/"
        "bacteria_subsumed_rows.py). Each of the 23 enumerated carbon sources "
        "is a condition of the Borchert 2024 compendium's carbon-source group, "
        "which holds 47 samples at them, 22 of the 23 in exactly the "
        "biological duplicate the Methods state; the 23rd, butyrate, has a "
        "third sample from an earlier library set and date. The served "
        "RbTnseqBorchert2024Dataset carries all 47 of those samples as 222,404 "
        "records over 4,732 loci each, so a loader for this row would store "
        "every one of those values a second time. The earlier note that not "
        "every published condition survived the compendium's completeness "
        "filter was WRONG: that filter eliminated LOCI, not conditions, and no "
        "condition this paper reports is missing. Nothing is left to load "
        "either, because the paper released no per-gene data file: its "
        "supplemental material is one figure PDF, it references no Table S, "
        "and its only release is the Fitness Browser, which answered HTTP 403 "
        "when probed on 2026-10-08. Provenance record: $DATA_ROOT/"
        "torchcell-raw/thompsonFattyAcidAlcohol2020/subsumption_record.json.",
        accession="https://fit.genomics.lbl.gov/cgi-bin/org.cgi?orgId=Putida",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant fitness across a carbon-source panel",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Putida_ML5 / JBEI-1 RB-TnSeq library",
                yields="pooled mutant fitness across a carbon-source panel",
            )
        ],
    ),
    Candidate(
        name="Schmidt 2022 nitrogen",
        organism="P. putida",
        citation="Schmidt M, Pearson AN, Incha MR, Thompson MG, Baidoo EEK, "
        "Kakumanu R, et al. Nitrogen Metabolism in Pseudomonas putida: "
        "Functional Analysis Using Random Barcode Transposon "
        "Sequencing. Applied and Environmental Microbiology 2022",
        url="https://doi.org/10.1128/aem.02430-21",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=4778,
        genotypes="4,778 insertion mutants",
        env_n=71,
        env="71 conditions",
        instances_n=339238,
        instances_basis="product",
        phenotype="gene-level RB-TnSeq fitness across nitrogen-containing sole nitrogen sources",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="transposon insertion",
        why="Not on the requested list but it clearly belongs, and it is the "
        "largest single-study condition axis for this organism: assimilation "
        "of 52 different nitrogen-containing compounds across 71 tested "
        "conditions, with significant fitness phenotypes in 672 genes "
        "including 100 transcriptional regulators and 112 transport proteins. "
        "Same JBEI-1 library and same Deutschbauer and Keasling pipeline as "
        "the lysine, valerolactam and fatty-acid rows. MEASURED SUBSUMPTION "
        "2026-10-08, and at the condition level it is total, so this row is a "
        "provenance record and NOT a loader (experiments/"
        "036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py). "
        "Table 1 enumerates the 71 conditions as 52 sole-nitrogen compounds "
        "plus 19 amino-acid drop-out conditions. The Borchert 2024 "
        "compendium's nitrogen-source group holds exactly 52 conditions in 104 "
        "of its 332 samples, and the mapping between the two lists is a "
        "BIJECTION: every compound of Table 1 is one of those conditions and "
        "every one of those conditions is one of those compounds, two samples "
        "each, which is the biological duplicate the Methods state. The served "
        "RbTnseqBorchert2024Dataset carries all 104 of those samples as "
        "492,128 records over 4,732 loci each, so a loader for this row would "
        "store every one of those values a second time. WHAT IS MISSING IS "
        "ALSO MEASURED, and it is not loadable: the 19 drop-out conditions "
        "have no condition in the compendium at all, and this paper's own "
        "t-SNE Methods name 129 sole-nitrogen growth assays over 51 conditions "
        "against the 102 samples the compendium carries over those same "
        "conditions, so 27 replicate assays are absent too. Neither slice can "
        "be recovered: the paper released no per-gene data file, its "
        "supplemental material is a single figure PDF, and its only release is "
        "the Fitness Browser, which answered HTTP 403 when probed on "
        "2026-10-08. Provenance record: $DATA_ROOT/torchcell-raw/"
        "schmidtNitrogenMetabolismPseudomonas2022/subsumption_record.json.",
        accession="https://fit.genomics.lbl.gov/cgi-bin/org.cgi?orgId=Putida",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant fitness across nitrogen sources",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Putida_ML5 / JBEI-1 RB-TnSeq library",
                yields="pooled mutant fitness across nitrogen sources",
            )
        ],
    ),
    Candidate(
        name="Borchert 2023 lignin tolerance",
        organism="P. putida",
        citation="Borchert AJ, Bleem A, Beckham GT. RB-TnSeq identifies genetic "
        "targets for improved tolerance of Pseudomonas putida towards "
        "compounds relevant to lignin conversion. Metabolic Engineering "
        "2023",
        url="https://doi.org/10.1016/j.ymben.2023.04.007",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=4732,
        genotypes="4,732 insertion mutants",
        env_n=16,
        env="16 conditions",
        instances_n=212940,
        instances_basis="product",
        phenotype="gene-level RB-TnSeq fitness under lignin-stream inhibitors on a fixed glucose background",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="transposon insertion",
        why="This is the lignin-stream tolerance row with reconstructed "
        "engineering strategies, and it is a tolerance rather than a "
        "catabolism screen: every stressor is overlaid on M9 plus 20 mM "
        "glucose. The 15 stressors were enumerated from the SRA sample titles "
        "(ferulate, 4-coumarate, levulinate, beta-ketoadipate, "
        "protocatechuate, cis,cis-muconate, glycolate, lactate, acetate, "
        "Na2SO4, NaCl, vanillate, vanillin, 4-hydroxybenzoate, "
        "4-hydroxybenzaldehyde) plus a glucose-only control, in triplicate: "
        "the 57 SRA runs decompose exactly as 15 x 3 selective, plus 3 plus 3 "
        "glucose-control enrichments, plus 3 plus 3 time-zero samples. "
        "PRJNA856070 confirmed by fetching the BioProject page (57 SRA "
        "experiments, 57 BioSamples). Reconstructed strategies from the "
        "fitness hits: delta-gacAS, delta-fleQ, delta-lapAB, delta-ttgR with "
        "Ptac-ttgABC, Ptac-PP_1150-PP_1152, delta-relA, delta-PP_1430, "
        "several of which also improved growth in a complex lignin-stream "
        "mimic. These samples are NOT in the Fitness Browser, which is "
        "precisely why the Borchert 2024 compendium reached into SRA for "
        "them; the 39 samples of Putida_ML5_set100 are recoverable at full "
        "precision from that workbook. MEASURED SUBSUMPTION, and it is near "
        "total: the 13 pairwise comparison sheets of Supplementary File 1 "
        "carry 42 distinct (experiment, replicate) fitness columns, and each "
        "one equals exactly one Borchert 2024 sample column on all 4,732 "
        "shared loci to a maximum absolute difference of 0.0005, the "
        "half-unit of the compendium's three-decimal rounding, with the "
        "runner-up column never nearer than 1.78. So 198,744 of the released "
        "values are the served compendium records. What IS new: the "
        "compendium eliminated 832 protein-coding genes for lacking a value "
        "in some sample, and 271 of them carry full triplicate fitness here, "
        "which is the 10,824 records RbTnseqBorchert2023Dataset stores. Its "
        "t, p, q and adjusted-q columns (64,853 rows x 4) have no field on "
        "any phenotype class and are recorded on issue #776 rather than "
        "stored; the 186,957 barcode-level rows are raw read counts and the "
        "growth curves are back-scattered light with no MeasurementType "
        "member.",
        accession="https://www.ncbi.nlm.nih.gov/bioproject/PRJNA856070",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant fitness across chemical stresses",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Putida_ML5 / JBEI-1 RB-TnSeq library",
                yields="pooled mutant fitness across chemical stresses",
            )
        ],
    ),
    Candidate(
        name="Rand 2017 Putida_ML5 library",
        organism="P. putida",
        citation="Rand JM, Pisithkul T, Clark RL, Thiede JM, Mehrer CR, Agnew "
        "DE, et al. A metabolic pathway for catabolizing levulinic acid "
        "in bacteria. Nature Microbiology 2017",
        url="https://doi.org/10.1038/s41564-017-0028-z",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=185401,
        genotypes="185,401 insertion mutants",
        env_n=4,
        env="4 conditions",
        instances_n=19112,
        instances_basis="estimate",
        phenotype="defines the barcoded insertion pool every fitness row is scored on; itself assayed on four carbon sources",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="transposon insertion",
        why="CORRECTION to the recalled library definition on two points. First, "
        "the library is named Putida_ML5 and was built here in the "
        "Deutschbauer and Arkin lab from the pKMW3 mariner vector library of "
        "Wetmore 2015; JBEI-1 is not a separate library but a regrown, "
        "re-aliquoted working stock of Putida_ML5 named later in Thompson "
        "2019 mBio. The compendium metadata's mutantLibrary column carries "
        "exactly two values, Putida_ML5 (25 samples) and Putida_ML5_JBEI (307 "
        "samples). Second, the recalled sizes of about 100,000 insertion "
        "mutants and about 4,800 nonessential genes are BOTH wrong. The only "
        "quantitative description in the literature is in Borchert 2022 (ACS "
        "Synth Biol 11:2015-2021, DOI 10.1021/acssynbio.2c00119, PMID "
        "35657709) citing this paper: 185,401 uniquely barcoded transposon "
        "insertions, of which 32,591 are intergenic and 152,810 map to 5,213 "
        "of 5,661 genes. The about 4,800 figure is a FITNESS-coverage number, "
        "not library coverage: the Fitness Browser states fitness data for "
        "4,778 of 5,661 genes, and Borchert 2024 reports 4,732 of 5,564 after "
        "filtering. Rand 2017 itself says only thousands of "
        "kanamycin-resistant colonies. Its own screen covers 4 carbon sources "
        "in duplicate over two days. Join on barcode to mapped insertion to "
        "PP_ locus tag against AE015451 (6,181,873 nt, taxid 160488); no "
        "study releases per-barcode fitness, all publish gene-level "
        "aggregates. The Fitness Browser organism page is orgId=Putida and "
        "reported 314 experiments across 54 carbon sources, 52 nitrogen "
        "sources, 2 stress compounds and 27 other conditions, but it is "
        "behind a Cloudflare managed JavaScript challenge today, so those "
        "figures were read from Internet Archive snapshots (2025-03-10 and "
        "2025-04-24) and must NOT be recorded as live 2026 values. Hence "
        "accession_confirmed false.",
        accession="https://fit.genomics.lbl.gov/cgi-bin/org.cgi?orgId=Putida",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="the pooled barcoded mutant collection itself",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="Putida_ML5 / JBEI-1 RB-TnSeq library",
                yields="the pooled barcoded mutant collection itself",
            )
        ],
    ),
    Candidate(
        name="Schmidt 2025 3-hydroxyacid PKS",
        organism="P. putida",
        citation="Schmidt M, Vilchez AA, Lee N, Keiser LS, Pearson AN, Thompson "
        "MG, et al. Engineering Pseudomonas putida for production of "
        "3-hydroxyacids using hybrid type I polyketide synthases. "
        "Metabolic Engineering Communications 2025",
        url="https://doi.org/10.1016/j.mec.2025.e00261",
        klass="Production campaign",
        tier=4,
        genotypes_n=12,
        genotypes="12 pathway designs",
        env_n=2,
        env="2 conditions",
        instances_n=72,
        instances_basis="estimate",
        phenotype="LC-MS titer of three 3-hydroxyacids in supernatant at 48 h",
        dim=3,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="pathway plasmid",
        why="This is the modular type I PKS row. JBEI public registry folder 887 "
        "(Schmidt et al 2025) was confirmed through the REST endpoint: 58 "
        "STRAIN entries with public read access, which is an upper bound on "
        "strains built and the reason the genotype count is a figure-level "
        "estimate rather than a stated total. Three products quantified "
        "(3H24DMPA, 3H4MPA at 18.2 mg/L, 3H4MHA at 6.4 mg/L), n=3 for titers, "
        "up to seven media variants that are not fully crossed, single 48 h "
        "harvest. The genotype classes here exceed the usual four and need "
        "extra encoding: alongside deletions and plasmid PKS expression there "
        "are mini-Tn7 and BxB1 chromosomal integrations, C-terminal ssrA "
        "degron tags on FabD (LAA active, DAS intermediate, LDD inert), and "
        "acyltransferase-domain swaps in the LipPKS1 loading module. "
        "Accompanying RB-TnSeq is said to be at fit.genomics.lbl.gov, "
        "unconfirmed. A sibling deposit from the same group, PXD069956 "
        "(Engineering an Extremely Hybrid PKS for Adipic Acid Production, "
        "Klass et al., ACS Synth Biol, DOI 10.1021/acssynbio.5c00972, PMID "
        "42339614), covers the same PKS platform in P. putida and E. coli if "
        "the modality is worth widening.",
        accession="https://public-registry.jbei.org/rest/folders/887/entries",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="engineered pathway panel with a single-endpoint titer",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="KT2440 gene set",
                yields="engineered pathway panel with a single-endpoint titer",
            )
        ],
    ),
    Candidate(
        name="Ling 2022 muconate",
        organism="P. putida",
        citation="Ling C, Peabody GL, Salvachua D, Kim YM, Kneucker CM, Calvey "
        "CH, et al. Muconic acid production from glucose and xylose in "
        "Pseudomonas putida via evolution and metabolic engineering. "
        "Nature Communications 2022",
        url="https://doi.org/10.1038/s41467-022-32296-y",
        klass="Production campaign",
        tier=4,
        genotypes_n=11,
        genotypes="11 deletions",
        env_n=4,
        env="4 conditions",
        instances_n=66,
        instances_basis="estimate",
        phenotype="cis,cis-muconate titer, molar yield and volumetric rate on glucose and xylose",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="gene deletion",
        why="The best documented muconate campaign with per-strain titers: 11 "
        "strains carry titer or growth measurements and Table 1 lists 24 "
        "strains, across four carbon conditions in flask, plate reader and "
        "0.5 L fed-batch, reaching 33.7 g/L at 0.18 g/L/h and 46 percent "
        "molar yield. PRJNA783062 confirmed by fetching the BioProject page, "
        "titled Sequencing of 5 ALE isolates on xylose (NREL, 2021-11-23), so "
        "only the genome resequencing is deposited and there is no proteomics "
        "accession. Genotype encoding is unusually mixed and does not fit a "
        "single modality: rational deletions and plasmid pathways sit "
        "alongside ALE-derived promoter mutations in the PP_2569 promoter, "
        "xylE point mutations (A62V, A455V) and a roughly 227.8 kb genome "
        "duplication from PP_5050 to PP_5242, none of which are designed "
        "edits.",
        accession="https://www.ncbi.nlm.nih.gov/bioproject/PRJNA783062",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="engineered strain ladder with fermentation titers",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="KT2440 gene set",
                yields="engineered strain ladder with fermentation titers",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Werner 2023 beta-ketoadipate",
        organism="P. putida",
        citation="Werner AZ, Cordell WT, Lahive CW, Klein BC, Singer CA, Tan "
        "ECD, et al. Lignin conversion to beta-ketoadipic acid by "
        "Pseudomonas putida via metabolic engineering and bioprocess "
        "development. Science Advances 2023",
        url="https://doi.org/10.1126/sciadv.adj0053",
        klass="Production campaign",
        tier=4,
        genotypes_n=8,
        genotypes="8 deletions",
        env_n=4,
        env="4 conditions",
        instances_n=64,
        instances_basis="estimate",
        phenotype="beta-ketoadipate titer, productivity and molar yield from lignin-derived aromatics",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440-KO",
        modality="gene deletion",
        why="Eight strains with titers across four aromatic feeds (p-coumarate, "
        "ferulate, a 3:1 molar mixture, and corn stover alkaline pretreated "
        "liquor extractives), reaching 44.5 g/L at 1.15 g/L/h from model "
        "compounds and 25 g/L at 0.66 g/L/h from real corn stover aromatics; "
        "genotypes are deletions of pcaIJ, crc, pobAR, lvaE, gacA and gacS "
        "plus chromosomal overexpression at the fpvA locus. It has the best "
        "per-strain numerical table of this cluster, with every main-text "
        "data point tabulated in data S1. But there is no repository "
        "accession at all and strains are available only from NREL under a "
        "material transfer agreement, so accession_confirmed is false because "
        "no accession exists. One arm is informative as a negative: the gacA "
        "variants underperformed.",
        accession="https://pmc.ncbi.nlm.nih.gov/articles/PMC10482344/",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol", why="deletion panel with bioreactor titers"
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="KT2440 gene set",
                yields="deletion panel with bioreactor titers",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Valencia 2022 free fatty acids",
        organism="P. putida",
        citation="Valencia LE, Incha MR, Schmidt M, Pearson AN, Thompson MG, "
        "Roberts JB, et al. Engineering Pseudomonas putida KT2440 for "
        "chain length tailored free fatty acid and oleochemical "
        "production. Communications Biology 2022",
        url="https://doi.org/10.1038/s42003-022-04336-2",
        klass="Production campaign",
        tier=4,
        genotypes_n=48,
        genotypes="48 pathway designs",
        env_n=3,
        env="3 conditions",
        instances_n=144,
        instances_basis="estimate",
        phenotype="free fatty acid and fatty acid methyl ester chain-length distribution at 48 h",
        dim=6,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="pathway plasmid",
        why="Included because its readout is genuinely multivariate: every sample "
        "yields a chain-length vector (C8, C10, C12:1, C14, C16, C16:1) "
        "rather than a scalar titer, so encoding it at dimensionality 1 would "
        "discard the study. Eight host backgrounds from deletions of three "
        "fatty acyl-CoA ligases, crossed with about six plasmid constructs "
        "carrying protein-engineered TesA thioesterase variants or MmFAMT, "
        "across three media including fourfold-diluted sorghum hydrolysate, "
        "n=3; maxima 670.9 mg/L total free fatty acid, 253.6 mg/L C8, 302.4 "
        "mg/L total methyl ester, 561.1 mg/L on hydrolysate. The genotype "
        "count is an estimate because the host by plasmid design is not fully "
        "crossed. Source data for Figures 2 to 6 sit in Supplementary Data 1, "
        "but data availability cites only the bare public-registry.jbei.org "
        "root with no folder number, so nothing is retrievable by accession.",
        accession="https://pmc.ncbi.nlm.nih.gov/articles/PMC9744835/",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="host background crossed with pathway variants",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="KT2440 gene set",
                yields="host background crossed with pathway variants",
            )
        ],
    ),
    Candidate(
        name="Kang 2026 isoprenyl acetate",
        organism="P. putida",
        citation="Kang CW, Carruthers DN, McCauley J, Chen Y, Gin JW, Petzold "
        "CJ, Simmons BA, Lee TS. Multi-layered metabolic remodeling of "
        "Pseudomonas putida for efficient conversion of lignocellulosic "
        "sugars to the precursors of advanced aviation fuel. Metabolic "
        "Engineering Communications 2026",
        url="https://doi.org/10.1016/j.mec.2026.e00274",
        klass="Production campaign",
        tier=4,
        genotypes_n=25,
        genotypes="25 pathway designs",
        env_n=3,
        env="3 conditions",
        instances_n=225,
        instances_basis="estimate",
        phenotype="isoprenyl acetate and isoprenol titer from mixed glucose and xylose",
        dim=1,
        dim_basis="estimate",
        seq_basis="engineered-chassis",
        modality="pathway plasmid",
        why="Found in a ProteomeXchange sweep rather than the recalled list, and "
        "it belongs only if the downstream ester is admitted: the product is "
        "isoprenyl acetate, an ATF1-esterified derivative of isoprenol, at "
        "1.9 g/L in fed-batch with a yield of 0.067 g per g total sugar. "
        "PXD067010 confirmed by fetching its PRIDE record, which names this "
        "DOI and PMID and lists Petzold as lab head. Engineering combines a "
        "heterologous alcohol acetyltransferase, esterase deletions to stop "
        "product degradation, glucose-xylose co-utilization, and reinforced "
        "acetyl-CoA flux, across tube, flask and 2 L fed-batch in biological "
        "triplicate. Strain and instance counts are estimates; the deposit "
        "does not state a run count. Shares Carruthers and Lee with the "
        "flagship isoprenol rows, so deduplicate genotypes.",
        accession="https://www.ebi.ac.uk/pride/archive/projects/PXD067010",
        accession_confirmed=True,
        status="candidate",
        isoprenol="direct",
        product="isoprenyl acetate, isoprenol",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="engineered strain series producing an alcohol ester",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="isoprenol pathway (IPP-bypass) in KT2440",
                yields="engineered strain series producing an alcohol ester",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Mohamed 2020 hydroxycinnamic TALE",
        organism="P. putida",
        citation="Mohamed ET, Werner AZ, Salvachua D, Singer CA, Szostkiewicz K, "
        "Rafael Jimenez-Diaz M, et al. Adaptive laboratory evolution of "
        "Pseudomonas putida KT2440 improves p-coumaric and ferulic acid "
        "catabolism and tolerance. Metabolic Engineering Communications "
        "2020",
        url="https://doi.org/10.1016/j.mec.2020.e00143",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=105,
        genotypes="105 resequenced clones",
        env_n=4,
        env="4 conditions",
        instances_n=420,
        instances_basis="estimate",
        phenotype="growth rate and lag phase on p-coumarate and ferulate at increasing concentrations",
        dim=1,
        dim_basis="reported",
        seq_basis="evolved-WGS",
        modality="laboratory evolution",
        why="Not on the recalled list but it is the best deposited P. putida "
        "tolerance evolution campaign. Six independent parallel replicates "
        "each for p-coumarate, ferulate and an equal-mass mixture, plus four "
        "replicates for a glucose control ALE, ramped to the solubility limit "
        "(40 g/L ferulate) or a 67-day cap; outcomes include a 37 h decrease "
        "in lag phase at 20 g/L p-coumarate and a 2.4-fold growth-rate "
        "increase at 30 g/L ferulate. PRJNA624660 confirmed by fetching the "
        "BioProject page: 102 SRA experiments and 105 BioSamples, with the "
        "ALEdb project named P. putida Hydroxycinnamic TALE. CAVEAT on "
        "genotype reconstruction: the deposit mixes endpoint populations with "
        "clonal isolates (1 to 2 intermediate isolates, the endpoint "
        "population, and a single endpoint isolate per condition), and only "
        "the isolates have a reconstructable single genotype; the population "
        "samples are blocked at the genotype level even though their reads "
        "are public.",
        accession="https://www.ncbi.nlm.nih.gov/bioproject/PRJNA624660",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Mormino 2022 CRISPRi acetic acid",
            why="weak-acid tolerance scored by growth",
        ),
        synergy=[
            Synergy(
                partner="Mormino 2022 CRISPRi acetic acid",
                partner_status="supported",
                join="KT2440 gene set",
                yields="weak-acid tolerance scored by growth",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Lim 2020 ionic liquid ALE",
        organism="P. putida",
        citation="Lim HG, Fong B, Alarcon G, Magurudeniya HD, Eng T, Szubin R, "
        "et al. Generation of ionic liquid tolerant Pseudomonas putida "
        "KT2440 strains via adaptive laboratory evolution. Green "
        "Chemistry 2020",
        url="https://doi.org/10.1039/D0GC01663B",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=10,
        genotypes="10 resequenced clones",
        env_n=3,
        env="3 conditions",
        instances_n=90,
        instances_basis="estimate",
        phenotype="growth performance at high protic ionic liquid concentrations",
        dim=1,
        dim_basis="estimate",
        seq_basis="evolved-WGS",
        modality="laboratory evolution",
        why="Bibliography confirmed through CrossRef (Green Chemistry "
        "22(17):5677-5690); the journal is not indexed in PubMed, so the PMID "
        "is deliberately left empty rather than guessed. Evolved strains grow "
        "at up to 4 percent triethanolammonium acetate and 8 percent "
        "triethylammonium hydrogen sulfate where the wild type cannot, with "
        "mutations in relA, gacS, oprB, fleQ, tktA and uvrY for minimal media "
        "and PP_5350, emrE, oprD and PP_5324 for the ionic-liquid-specific "
        "arms; PP_5350 (an RpiR-family regulator upregulating the glyoxylate "
        "cycle) and the emrE efflux pump were validated by reverse "
        "engineering. Marked BLOCKED because no sequencing deposit could be "
        "located: the publisher page returns HTTP 403, OSTI dropped the "
        "connection, and no BioProject or SRA accession surfaced anywhere, so "
        "the evolved genotypes are not retrievable even though the mutations "
        "are named in the text. Strain and instance counts are estimates.",
        accession="https://doi.org/10.1039/D0GC01663B",
        accession_confirmed=False,
        status="blocked",
        confidence="sourced",
        analog=Analog(
            dataset="Mormino 2022 CRISPRi acetic acid",
            why="solvent and acid tolerance by growth",
        ),
        synergy=[
            Synergy(
                partner="Mormino 2022 CRISPRi acetic acid",
                partner_status="supported",
                join="KT2440 gene set",
                yields="solvent and acid tolerance by growth",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Schmidt 2016",
        organism="E. coli",
        citation="Schmidt A, Kochanowski K, Vedelaar S, Ahrne E, Volkmer B, "
        "Callipo L, Knoops K, Bauer M, Aebersold R, Heinemann M. The "
        "quantitative and condition-dependent Escherichia coli "
        "proteome. Nature Biotechnology 2016",
        url="https://doi.org/10.1038/nbt.3418",
        klass="Proteome",
        tier=4,
        genotypes_n=3,
        genotypes="3 wild type",
        env_n=22,
        env="22 conditions",
        instances_n=66,
        instances_basis="product",
        phenotype="absolute protein copies per cell and per cell volume",
        dim=2359,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="environment-only",
        why="2,359 proteins quantified across all 22 conditions, about 55 percent "
        "of predicted ORFs, calibrated to copies per cell with flow cytometry "
        "and synthetic peptide standards. Primary strain BW25113, with MG1655 "
        "and NCM3722 at selected conditions; biological triplicates for the "
        "main dataset, so 66 is a product not a reported sample count.",
        accession="PXD000498 (ProteomeXchange/PRIDE, DOI 10.6019/PXD000498)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Messner 2023 proteome",
            why="both give panel-wide absolute proteome depth",
        ),
        synergy=[
            Synergy(
                partner="Messner 2023 proteome",
                partner_status="supported",
                join="growth condition panel",
                yields="both give panel-wide absolute proteome depth",
            )
        ],
    ),
    Candidate(
        name="PRECISE-1K",
        organism="E. coli",
        citation="Lamoureux CR, Decker KT, Sastry AV, Rychel K, Gao Y, McConn "
        "JL, Zielinski DC, Palsson BO. A multi-scale expression and "
        "regulation knowledge base for Escherichia coli. Nucleic Acids "
        "Research 2023",
        url="https://doi.org/10.1093/nar/gkad750",
        klass="Transcriptome",
        tier=3,
        genotypes_n=76,
        genotypes="76 designs",
        env_n=116,
        env="116 conditions",
        instances_n=1035,
        instances_basis="reported",
        phenotype="RNA-seq expression compendium plus 201 iModulon activities",
        dim=4257,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="mixed",
        why="1,035 samples over 4,257 genes, 5 strain backgrounds, 76 unique gene "
        "knockouts, 45 distinct projects, yielding 201 iModulons; a combined "
        "public K-12 set reaches 2,710 samples and 194 iModulons. The paper "
        "reports no single unique-condition total, so 116 is the sum of its "
        "itemized condition variables (9 base media, 18 carbon sources, 38 "
        "supplements, 42 heterologous proteins, 4 temperatures, 5 pH values). "
        "No GEO SuperSeries was found; Zenodo is the stated repository.",
        accession="https://imodulondb.org (dataset E. coli PRECISE-1K); https://doi.org/10.5281/zenodo.8284223; https://github.com/SBRG/precise1k",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Kemmeren 2014 deletion transcriptome",
            why="deletion-and-condition expression compendium",
        ),
        synergy=[
            Synergy(
                partner="Kemmeren 2014 deletion transcriptome",
                partner_status="supported",
                join="MG1655 gene set",
                yields="deletion-and-condition expression compendium",
            )
        ],
    ),
    Candidate(
        name="Ishii 2007",
        organism="E. coli",
        citation="Ishii N, Nakahigashi K, Baba T, Robert M, Soga T, Kanai A, "
        "Hirasawa T, Naba M, Hirai K, Hoque A, Ho PY, Kakazu Y, "
        "Sugawara K, Igarashi S, Harada S, Masuda T, Sugiyama N, "
        "Togashi T, Hasegawa M, Takai Y, Yugi K, Arakawa K, Iwata N, "
        "Toya Y, Nakayama Y, Nishioka T, Shimizu K, Mori H, Tomita M. "
        "Multiple high-throughput analyses monitor the response of E. "
        "coli to perturbations. Science 2007",
        url="https://doi.org/10.1126/science.1132067",
        klass="Multi-omics campaign",
        tier=4,
        genotypes_n=25,
        genotypes="25 designs",
        env_n=5,
        env="5 conditions",
        instances_n=29,
        instances_basis="reported",
        phenotype="paired transcriptome, proteome, metabolome and 13C flux in glucose-limited chemostats",
        dim=579,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="mixed",
        why="Design and per-layer depth both MEASURED off the released workbook "
        "(ishii2007_release_inventory.py, 2026-10-08): 24 single-gene disruptants at a "
        "fixed dilution rate of 0.2 per hour plus wild type at 5 dilution rates. Per "
        "layer, targets and servable records: metabolite 579 and 24, mRNA 85 and 24, "
        "protein 67 and 24, flux 50 and 24. The earlier 4,300 was a genome-wide-array "
        "estimate and is replaced by 579, the largest targeted panel; the genome-wide "
        "DNA-array arm is a separate 4,213-oligo ratio release. The data is NOT in the "
        "paywalled supplement: the paper's reference 21 is a project web site that "
        "serves the whole release over plain HTTP, and science.org answers HTTP 403 "
        "with a Cloudflare challenge for every route. Three of the four layers are "
        "loaded (metabolome, proteome, flux, 24 records each); the qRT-PCR mRNA layer "
        "is an absolute transcript abundance no phenotype holds.",
        accession="Keio project web site (the paper's reference 21), "
        "http://ecoli.iab.keio.ac.jp/; no repository accession, and the Science "
        "supporting online material is Cloudflare-gated",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Zelezniak 2018 proteome and metabolome",
            why="several layers, one strain panel",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 proteome and metabolome",
                partner_status="supported",
                join="Keio collection",
                yields="several layers, one strain panel",
            )
        ],
    ),
    Candidate(
        name="Li 2014",
        organism="E. coli",
        citation="Li GW, Burkhardt D, Gross C, Weissman JS. Quantifying absolute "
        "protein synthesis rates reveals principles underlying "
        "allocation of cellular resources. Cell 2014",
        url="https://doi.org/10.1016/j.cell.2014.02.033",
        klass="Translation / turnover",
        tier=3,
        genotypes_n=1,
        genotypes="1 wild type",
        env_n=2,
        env="2 conditions",
        instances_n=2,
        instances_basis="product",
        phenotype="absolute protein synthesis rates from ribosome profiling",
        dim=3041,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="environment-only",
        why="3,041 genes accounting for more than 96 percent of total protein "
        "synthesized in rich defined medium, with a similar number in glucose "
        "minimal medium, at 90 million fragments per sample. Method is "
        "ribosome profiling calibrated to total protein per doubling; the "
        "quantitative mass spectrometry comparison is to published external "
        "datasets, not generated here. Strain designation is unconfirmed "
        "(main-text methods do not name it).",
        accession="GSE53767 (GEO); browsable table at http://ecoliwiki.net/tools/proteome/",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Messner 2023 proteome",
            why="genome-wide absolute protein-level quantification",
        ),
        synergy=[
            Synergy(
                partner="Messner 2023 proteome",
                partner_status="supported",
                join="MG1655 gene set",
                yields="genome-wide absolute protein-level quantification",
            )
        ],
    ),
    Candidate(
        name="Gupta 2024 turnover",
        organism="E. coli",
        citation="Gupta M, Johnson ANT, Cruz ER, Costa EJ, Guest RL, Li SHJ, "
        "Hart EM, Nguyen T, Stadlmeier M, Bratton BP, Silhavy TJ, "
        "Wingreen NS, Gitai Z, Wuhr M. Global protein turnover "
        "quantification in Escherichia coli reveals cytoplasmic "
        "recycling under nitrogen limitation. Nature Communications "
        "2024",
        url="https://doi.org/10.1038/s41467-024-49920-8",
        klass="Translation / turnover",
        tier=3,
        genotypes_n=1,
        genotypes="1 wild type",
        env_n=13,
        env="13 conditions",
        instances_n=13,
        instances_basis="estimate",
        phenotype="per-protein degradation and turnover rates",
        dim=3200,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="environment-only",
        why="About 3,200 proteins, 77 percent of all genes, with turnover rates "
        "across 13 growth conditions in strain NCM3722, by heavy 15N dynamic "
        "labeling with TMTproC complement-reporter mass spectrometry. "
        "Replicates exist but the per-condition replicate count is "
        "unconfirmed, so instances is a floor. Cite the 2024 Nature "
        "Communications version, not the 2022 bioRxiv preprint.",
        accession="PXD042444 (ProteomeXchange/PRIDE); code at https://github.com/wuhrlab/ProteinTurnoverEcoli",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Messner 2023 proteome",
            why="proteome-wide per-protein quantitative depth",
        ),
        synergy=[
            Synergy(
                partner="Messner 2023 proteome",
                partner_status="supported",
                join="MG1655 gene set",
                yields="proteome-wide per-protein quantitative depth",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Haverkorn van Rijsewijk 2011",
        organism="E. coli",
        citation="Haverkorn van Rijsewijk BRB, Nanchen A, Nallet S, Kleijn RJ, "
        "Sauer U. Large-scale 13C-flux analysis reveals distinct "
        "transcriptional control of respiratory and fermentative "
        "metabolism in Escherichia coli. Molecular Systems Biology 2011",
        url="https://doi.org/10.1038/msb.2011.9",
        klass="Metabolome / flux",
        tier=4,
        genotypes_n=91,
        genotypes="91 deletions",
        env_n=2,
        env="2 conditions",
        instances_n=182,
        instances_basis="product",
        phenotype="13C metabolic flux ratios in central carbon metabolism",
        dim=8,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="91 regulator deletion mutants (81 transcription factors plus 10 "
        "sigma and anti-sigma factors) on glucose and galactose, giving 8 "
        "metabolic flux ratios per strain by GC-MS of proteinogenic amino "
        "acids with FiatFlux. Dimensionality is genuinely small; the value of "
        "this row is the 91-strain deletion axis, not depth. Do not confuse "
        "with the Nielsen commentary (DOI 10.1038/msb.2011.10, PMID "
        "21451588).",
        accession="Supplementary Table 2 (glucose) and Supplementary Table 3 (galactose); no external accession",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Mulleder 2016 amino-acid metabolome",
            why="deletion panel, central-metabolic readout",
        ),
        synergy=[
            Synergy(
                partner="Mulleder 2016 amino-acid metabolome",
                partner_status="supported",
                join="Keio collection",
                yields="deletion panel, central-metabolic readout",
            )
        ],
    ),
    Candidate(
        name="CeCaFDB flux compendium",
        organism="E. coli",
        citation="Zhang Z, Shen T, Rui B, et al. CeCaFDB: a curated database for "
        "the documentation, visualization and comparative analysis of "
        "central carbon metabolic flux distributions explored by "
        "13C-fluxomics. Nucleic Acids Research 2015",
        url="https://doi.org/10.1093/nar/gku1137",
        klass="Aggregation / support",
        tier=3,
        genotypes_n=None,
        genotypes="mixed",
        env_n=None,
        env="not stated",
        instances_n=297,
        instances_basis="reported",
        phenotype="curated published 13C flux distributions on a normalized central-carbon network",
        dim=76,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="mixed",
        why="581 flux distributions across 36 organisms from 118 references (1995 "
        "to 2013); E. coli alone contributes 297 flux distributions from 32 "
        "references, the largest single-organism set (S. cerevisiae is next "
        "at 76). The normalized comparison network has 66 metabolites and 76 "
        "reactions, which is the per-map dimensionality bound, not "
        "necessarily each source study's own network size. The site returned "
        "an expired-TLS error on 2026-09-24, so current operation is "
        "unconfirmed.",
        accession="http://www.cecafdb.org",
        accession_confirmed=False,
        status="aggregation",
        confidence="sourced",
        synergy=[
            Synergy(
                partner="another candidate in this table",
                partner_status="candidate",
                join="central carbon metabolism reaction set",
                yields="a shared-axis comparison",
            )
        ],
    ),
    Candidate(
        name="Fuhrer 2017",
        organism="E. coli",
        citation="Fuhrer T, Zampieri M, Sevin DC, Sauer U, Zamboni N. Genomewide "
        "landscape of gene-metabolome associations in Escherichia coli. "
        "Molecular Systems Biology 2017",
        url="https://doi.org/10.15252/msb.20167150",
        klass="Metabolome / flux",
        tier=1,
        genotypes_n=3807,
        genotypes="3,807 deletions",
        env_n=1,
        env="1 condition",
        instances_n=34000,
        instances_basis="reported",
        phenotype="non-targeted flow-injection TOF-MS metabolite ion intensities",
        dim=7534,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        why="3,807 Keio single-gene deletion mutants (from 4,320 after excluding "
        "21 very sick strains and 16 injection failures), two independent "
        "clones each in technical duplicate, more than 34,000 "
        "mass-spectrometric injections. 7,534 distinct m/z features (3,169 "
        "negative plus 4,365 positive mode); 3,130 of them were putatively "
        "matched to 1,432 of 2,028 chemical formulas, so annotated depth is "
        "far below raw feature depth. Deposited in MassIVE and BioStudies, "
        "not MetaboLights.",
        accession="MassIVE MSV000078963; EBI BioStudies S-BSST5; Datasets EV1-EV4",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Mulleder 2016 amino-acid metabolome",
            why="whole deletion collection metabolite profiling",
        ),
        synergy=[
            Synergy(
                partner="Mulleder 2016 amino-acid metabolome",
                partner_status="supported",
                join="Keio collection",
                yields="whole deletion collection metabolite profiling",
            )
        ],
    ),
    Candidate(
        name="Caglar 2017",
        organism="E. coli",
        citation="Caglar MU, Houser JR, Barnhart CS, Boutz DR, Carroll SM, "
        "Dasgupta A, Lenoir WF, Smith BL, Sridhara V, Sydykova DK, "
        "Vander Wood D, Marx CJ, Marcotte EM, Barrick JE, Wilke CO. The "
        "E. coli molecular phenotype under different growth conditions. "
        "Scientific Reports 2017",
        url="https://doi.org/10.1038/srep45303",
        klass="Multi-omics campaign",
        tier=4,
        genotypes_n=1,
        genotypes="1 wild type",
        env_n=34,
        env="34 conditions",
        instances_n=322,
        instances_basis="reported",
        phenotype="matched RNA-seq and shotgun proteomics plus central carbon fluxes",
        dim=4196,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="environment-only",
        why="34 growth conditions in E. coli B REL606 with 152 RNA-seq samples, "
        "105 proteomics samples and 65 flux measurements (322 total), "
        "spanning exponential and stationary phase plus two time courses from "
        "3 hours to 2 weeks. The paper reports 4,196 distinct mRNAs and "
        "proteins as a joint figure, so treat 4,196 as the matched "
        "transcript-and-protein gene axis rather than as two independent "
        "depths.",
        accession="GSE67402 and GSE94117 (GEO); PXD002140 and PXD005721 (PRIDE); flux data doi:10.18738/T8/UG3TUR",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Zelezniak 2018 proteome and metabolome",
            why="two layers on identical samples",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 proteome and metabolome",
                partner_status="supported",
                join="growth condition panel",
                yields="two layers on identical samples",
            )
        ],
        time_axis="sampled series",
    ),
    Candidate(
        name="Potel 2018 phosphoproteome",
        organism="E. coli",
        citation="Potel CM, Lin MH, Heck AJR, Lemeer S. Widespread bacterial "
        "protein histidine phosphorylation revealed by mass "
        "spectrometry-based proteomics. Nature Methods 2018",
        url="https://doi.org/10.1038/nmeth.4580",
        klass="Proteome",
        tier=4,
        genotypes_n=1,
        genotypes="1 wild type",
        env_n=2,
        env="2 conditions",
        instances_n=2,
        instances_basis="estimate",
        phenotype="phosphosite identifications including phosphohistidine",
        dim=2129,
        dim_basis="estimate",
        seq_basis="reference-only",
        modality="environment-only",
        why="The deepest E. coli phosphoproteome found, but with almost no "
        "breadth: one wild-type MG1655 strain on glycerol, exponential versus "
        "stationary phase. The 2,129 total phosphosites and 246 "
        "phosphohistidine sites on 173 proteins come from a secondary search "
        "snippet, not from a fetched primary text; the abstract confirms only "
        "that about 10 percent of sites are phosphohistidine, so "
        "dimensionality needs primary confirmation before ranking.",
        accession="PXD008369 (ProteomeXchange, title 'Widespread protein histidine phosphorylation in bacteria')",
        accession_confirmed=True,
        status="candidate",
        confidence="recall",
        analog=Analog(
            dataset="Messner 2023 proteome",
            why="deep single-organism protein-level mass spectrometry",
        ),
        synergy=[
            Synergy(
                partner="Messner 2023 proteome",
                partner_status="supported",
                join="MG1655 gene set",
                yields="deep single-organism protein-level mass spectrometry",
            )
        ],
    ),
    Candidate(
        name="Schastnaya 2021",
        organism="E. coli",
        citation="Schastnaya E, Raguz Nakic Z, Gruber CH, Doubleday PF, Krishnan "
        "A, Johns NI, Park J, Wang HH, Sauer U. Extensive regulation of "
        "enzyme activity by phosphorylation in Escherichia coli. Nature "
        "Communications 2021",
        url="https://doi.org/10.1038/s41467-021-25988-4",
        klass="Metabolome / flux",
        tier=4,
        genotypes_n=89,
        genotypes="89 designs",
        env_n=5,
        env="5 conditions",
        instances_n=445,
        instances_basis="product",
        phenotype="metabolome of phosphosite point mutants and matched deletion mutants",
        dim=460,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="mixed",
        why="52 known phosphosites on 23 central metabolic enzymes were mutated "
        "in E. coli MG1655 delta-mutS, giving 89 phosphomutants profiled on 5 "
        "carbon sources against matched Keio deletion strains, with 460 to "
        "500 annotated metabolites by flow-injection TOF-MS. This is the best "
        "genotype-by-condition phospho-regulation panel, but it generated no "
        "phosphoproteome of its own; its phosphosite list is drawn from "
        "earlier studies such as Potel 2018.",
        accession="PXD027243 (PRIDE); MSV000087795 (MassIVE metabolomics)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Mulleder 2016 amino-acid metabolome",
            why="point-mutant panel, metabolite readout",
        ),
        synergy=[
            Synergy(
                partner="Mulleder 2016 amino-acid metabolome",
                partner_status="supported",
                join="Keio collection",
                yields="point-mutant panel, metabolite readout",
            )
        ],
    ),
    Candidate(
        name="Taniguchi 2010",
        organism="E. coli",
        citation="Taniguchi Y, Choi PJ, Li GW, Chen H, Babu M, Hearn J, Emili A, "
        "Xie XS. Quantifying E. coli proteome and transcriptome with "
        "single-molecule sensitivity in single cells. Science 2010",
        url="https://doi.org/10.1126/science.1188308",
        klass="Multi-omics campaign",
        tier=3,
        genotypes_n=1018,
        genotypes="1,018 designs",
        env_n=1,
        env="1 condition",
        instances_n=1018,
        instances_basis="reported",
        phenotype="single-cell protein copy number distribution and mRNA copy number per tagged gene",
        dim=2,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="mixed",
        why="1,018 confirmed strains of 1,400 attempted, each carrying a "
        "chromosomal C-terminal YFP fusion to one native gene at its native "
        "locus, imaged at about 4,000 cells per strain (96 strains per "
        "microfluidic device, 25 seconds each), with mRNA by RNA-seq and "
        "single-molecule FISH (137 genes had matched FISH). Per-strain "
        "dimensionality is only 2 summary values, so its value is the "
        "1,018-gene tagged panel and the single-cell distributions, not depth "
        "per sample. The CGSC availability claim is unverified.",
        accession="Supplementary Table S1 (strain list); strains listed at the Coli Genetic Stock Center",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Nadal-Ribelles 2025 Perturb-seq",
            why="single-cell readout across strain library",
        ),
        synergy=[
            Synergy(
                partner="Nadal-Ribelles 2025 Perturb-seq",
                partner_status="supported",
                join="MG1655 gene set",
                yields="single-cell readout across strain library",
            )
        ],
    ),
    Candidate(
        name="Balakrishnan 2022",
        organism="E. coli",
        citation="Balakrishnan R, Mori M, Segota I, Zhang Z, Aebersold R, Ludwig "
        "C, Hwa T. Principles of gene regulation quantitatively connect "
        "DNA to RNA and proteins in bacteria. Science 2022",
        url="https://doi.org/10.1126/science.abk2066",
        klass="Multi-omics campaign",
        tier=4,
        genotypes_n=2,
        genotypes="2 designs",
        env_n=4,
        env="4 conditions",
        instances_n=8,
        instances_basis="product",
        phenotype="matched mRNA and protein mass fractions plus promoter activity and mRNA degradation rates",
        dim=1900,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="mixed",
        why="The cleanest transcript-and-protein join on identical samples: more "
        "than 1,900 proteins with matched mRNA, plus about 2,700 genome-wide "
        "mRNA degradation rates, in E. coli K-12 NCM3722 (and a delta-rsd "
        "mutant) across a reference glucose minimal medium plus carbon "
        "limitation, anabolic limitation and translational inhibition, growth "
        "rates 0.3 to 0.9 per hour. Replicate counts vary by assay and are "
        "not uniformly reported, so instances is a product.",
        accession="PXD014948 (PRIDE); GSE205717 (GEO RNA-seq); PASS01421 (SWATHAtlas)",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Zelezniak 2018 proteome and metabolome",
            why="two layers, same samples, joinable",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 proteome and metabolome",
                partner_status="supported",
                join="MG1655 gene set",
                yields="two layers, same samples, joinable",
            )
        ],
    ),
    Candidate(
        name="Mori 2021",
        organism="E. coli",
        citation="Mori M, Zhang Z, Banaei-Esfahani A, Lalanne JB, Okano H, "
        "Collins BC, Schmidt A, Schubert OT, Lee DS, Li GW, Aebersold "
        "R, Hwa T, Ludwig C. From coarse to fine: the absolute "
        "Escherichia coli proteome under diverse growth conditions. "
        "Molecular Systems Biology 2021",
        url="https://doi.org/10.15252/msb.20209536",
        klass="Proteome",
        tier=4,
        genotypes_n=3,
        genotypes="3 wild type",
        env_n=60,
        env="60 conditions",
        instances_n=66,
        instances_basis="reported",
        phenotype="absolute protein mass fractions by DIA/SWATH with xTop inference",
        dim=2335,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="environment-only",
        why="2,335 proteins with absolute mass fractions over about 60 growth "
        "conditions in 66 samples, covering carbon, nitrogen, phosphate and "
        "oxygen limitation, non-metabolic stresses and biofilm states, in "
        "K-12 NCM3722 and derivatives plus MG1655 sub-strain EQ353 and E. "
        "coli Nissle 1917. No ProteomeXchange or PRIDE accession was stated "
        "in the text examined; the deposit found is the SWATHAtlas spectral "
        "library. An author correction was published in October 2024 and "
        "should be read with the paper.",
        accession="PASS01421 (SWATHAtlas spectral library); Datasets EV1-EV12",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Messner 2023 proteome",
            why="absolute proteome across a large condition panel",
        ),
        synergy=[
            Synergy(
                partner="Messner 2023 proteome",
                partner_status="supported",
                join="growth condition panel",
                yields="absolute proteome across a large condition panel",
            )
        ],
    ),
    Candidate(
        name="Kochanowski 2017",
        organism="E. coli",
        citation="Kochanowski K, Gerosa L, Brunner SF, Christodoulou D, Nikolaev "
        "YV, Sauer U. Few regulatory metabolites coordinate expression "
        "of central metabolic genes in Escherichia coli. Molecular "
        "Systems Biology 2017",
        url="https://doi.org/10.15252/msb.20167402",
        klass="Multi-omics campaign",
        tier=4,
        genotypes_n=2,
        genotypes="2 wild type",
        env_n=26,
        env="26 conditions",
        instances_n=26,
        instances_basis="reported",
        phenotype="promoter activities and absolute central metabolite concentrations on shared conditions",
        dim=142,
        dim_basis="reported",
        seq_basis="reference-only",
        modality="environment-only",
        why="95 fluorescent promoter reporters (an existing library expanded by "
        "28, with 31 inactive promoters discarded) measured at steady state "
        "across 26 conditions spanning carbon sources, amino-acid "
        "supplementation and sub-lethal chloramphenicol, growth rates 0.1 to "
        "1.5 per hour, in BW25113 plus a Crp deletion mutant. 47 central "
        "metabolites were quantified absolutely by ion-pairing UPLC-MS/MS in "
        "23 of the 26 conditions, so dimensionality 142 is the union of 95 "
        "promoter activities and 47 metabolites, not a per-sample vector "
        "measured in every condition. No fluxes were measured.",
        accession="Tables EV1-EV8 (no external accession)",
        accession_confirmed=False,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Zelezniak 2018 proteome and metabolome",
            why="expression plus metabolome, shared conditions",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 proteome and metabolome",
                partner_status="supported",
                join="growth condition panel",
                yields="expression plus metabolome, shared conditions",
            )
        ],
    ),
    # ---------------------------------------------------------------------
    # The two E. coli isoprenol rows. Added after a 300-publication sweep
    # (Sec. sec:sweep) found them absent, and they close a gap that mattered: every
    # other isoprenol row here is P. putida, so before these the product existed in
    # one host only and no cross-host question could be posed.
    # ---------------------------------------------------------------------
    Candidate(
        name="Tian 2019 isopentenol CRISPRi",
        organism="E. coli",
        citation="Tian T, Kang JW, Kang A, Lee TS. Redirecting metabolic flux "
        "via combinatorial multiplex CRISPRi-mediated repression for "
        "isopentenol production in Escherichia coli. ACS Synthetic "
        "Biology 2019;8:391-402",
        url="https://doi.org/10.1021/acssynbio.8b00429",
        klass="Combinatorial design",
        tier=4,
        genotypes_n=24,
        genotypes="24: 1 base, 18 single-guide, 3 two-guide, 2 three-guide",
        env_n=3,
        env="3 induction levels",
        instances_n=84,
        instances_basis="product",
        phenotype="isopentenol titer by GC plus OD600, replicate count not stated",
        dim=2,
        dim_basis="reported",
        seq_basis="engineered-chassis+guide",
        modality="multiplex CRISPRi knockdown",
        isoprenol="direct",
        product="isopentenol",
        why="The only isoprenol production campaign in either host with a nested "
        "combinatorial perturbation axis, and the reason this row is here "
        "despite releasing nothing. Three genes, asnA, gldA and prpE, are "
        "realized as 3 singles, all 3 pairs and the full triple, each assayed "
        "across repression levels of 0, 5 and 10 nM anhydrotetracycline, so the "
        "design scores an interaction and a dose rather than a ranking. A second "
        "3-gene array over poxB, ackA and pta is matched against a triple "
        "knockout of the same genes, which is a knockdown-versus-deletion "
        "comparison on one gene set. The host is DH1, not BW25113 or MG1655, "
        "carrying the KG1R10 mevalonate pathway as two plasmids: atoB, HMGS, "
        "HMGR, MK and PMK under Ptrc, then nudB and PMD under PlacUV5. Every "
        "construct has a JBEI registry number. The paper's own target count is "
        "self-inconsistent, 21 genes in the introduction against 18 guides in "
        "the results and 15 in the abstract, and one printed symbol, arcC, is "
        "not a K-12 gene name. RE-MEASURED 2026-10-08 "
        "(experiments/036-dataset-fixes-before-kg-build/scripts/"
        "tian2019_release_loadability.py): STILL BLOCKED, and the reason is now exact. "
        "The paper has NO Supporting Information at all, which is a property of the "
        "release and not a gap in the mirror: the library manifest records "
        "si_expected as empty with provenance_complete true, and the PDF carries no "
        "ACS ASSOCIATED CONTENT block and no occurrence of the string ``Supporting "
        "Information``. The release holds 3 OCR'd table blocks, all of them Table 1 and "
        "its two continuations, and across their 452 cells exactly ONE is a bare number, "
        "the bibliography reference on the KG1R10 row, so the measurement count is "
        "zero: every other digit belongs to an identifier. The 86 absolute isopentenol "
        "values in mg/L are bar heights in Figures 6B and 7, with 60 matching OD600 "
        "values and 10 acetate values beside them, and no figure caption says what its "
        "error bars represent. Exactly 4 (strain, condition) pairs carry a figure-free "
        "number, every one of them a PERCENTAGE IMPROVEMENT over the base strain rather "
        "than a titer, with no unit, no uncertainty and, for 2 of the 4, no stated "
        "timepoint; admitting the shared 18 to 24 percent interval as three per-strain "
        "intervals raises that to 7. The replicate count is genuinely absent, read "
        "twice from the OCR and from the PDF text layer. Zero gRNA spacer sequences are "
        "published anywhere, which is why arcC cannot be resolved rather than merely "
        "looking odd. Three corrections to this row's own description: the 3 induction "
        "levels apply to only 4 of the 24 genotypes, since the 18 singles and the "
        "poxB-ackA-pta triple ran at 10 nM alone; the 24 and 48 h sampling is Figure 7 "
        "only, against 24/48/72 h in the Results and 24/48/76 h in the Methods; and the "
        "15/18/21 target counts are three different SETS that reconcile as 15 + 3 = 18 "
        "and 18 + 3 = 21, so the defensible finding is narrower, that the abstract and "
        "conclusion call the delivered library 15 when 18 single-guide strains were "
        "built and measured. An OCR caveat for any later reader: paper.md's Table 1 "
        "misassigns JPUB numbers for five rows and drops JBEI-18656's plasmid row, so "
        "that table must be read from the PDF text layer.",
        accession="JBEI public registry (https://public-registry.jbei.org), JPUB numbers per plasmid in Table 1; no sequence-data or value accession. The paper's whole availability text is one Table 1 footnote, quoted 2026-10-08: ``Strains with JBEI plasmid are available at the JBEI public registry (https://public-registry.jbei.org) and searchable using the JPUB number.``",
        accession_confirmed=False,
        status="blocked",
        confidence="sourced",
        time_axis="sampled at 24 and 48 h, with the methods also stating 76 h against 72 h in the results",
        schema_need="a guide-array perturbation carrying more than one target at a "
        "graded induction level, and a titer measured on a chassis whose pathway is "
        "plasmid-borne rather than integrated",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="multiplexed guide arrays over one gene set",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="multiplex guide array",
                yields="multiplexed guide arrays over one gene set",
            ),
            Synergy(
                partner="Wang 2015 isoprenol tolerance",
                partner_status="candidate",
                join="isoprenol",
                yields="production and tolerance for one product in one host",
            ),
        ],
    ),
    Candidate(
        name="Wang 2015 isoprenol tolerance",
        organism="E. coli",
        citation="Wang C, Yang L, Shah AA, Choi ES, Kim SW. Dynamic interplay of "
        "multidrug transporters with TolC for isoprenol tolerance in "
        "Escherichia coli. Scientific Reports 2015;5:16505",
        url="https://doi.org/10.1038/srep16505",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=47,
        genotypes="47 released: wild type plus 46 deletions",
        env_n=2,
        env="2 isoprenol doses",
        instances_n=94,
        instances_basis="product",
        phenotype="OD600 at 12 h with and without isoprenol, sample SD over 2 biological replicates",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="gene deletion",
        isoprenol="tolerance",
        product="isoprenol",
        why="The only isoprenol tolerance screen in either host over a cataloged "
        "deletion collection, which is what earns tier 2 on one cleared bar. "
        "Supplementary Table S3 releases OD600 with standard deviation for the "
        "wild type and 46 Keio single deletions at 0 and 0.5 percent isoprenol by "
        "volume, in 2YT at 30 degrees, and Table S4 adds a 9-gene transporter "
        "transcript panel. Keio JW numbers are given per strain, so the deletion "
        "alleles resolve to Baba 2006. Two multi-gene strains exist and are the "
        "paper's own epistasis test, acrA with acrB and that double with tolC, "
        "both reported as non-additive against their singles with a plasmid "
        "complementation control; neither appears in any released table, so the "
        "combinatorial arm is a provenance gap rather than a record. The 0.75 "
        "percent dose is figure-only for the same reason. The paper's mutant "
        "count is self-inconsistent, 44 in the results and 45 in the abstract "
        "against 46 rows in Tables S2 and S3; the table count is what is ingested.",
        accession="Springer ESM 41598_2015_BFsrep16505_MOESM1_ESM.pdf (Tables S2, S3, S4); strains are Keio JW numbers, NBRP National Institute of Genetics",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="deletion collection scored for an alcohol phenotype",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="deletion collection",
                yields="deletion collection scored for an alcohol phenotype",
            ),
            Synergy(
                partner="Nichols 2011",
                partner_status="candidate",
                join="Keio collection",
                yields="one more chemical on a shared deletion axis",
            ),
        ],
    ),
    Candidate(
        name="Foo 2014 isopentenol tolerance",
        organism="E. coli",
        citation="Foo JL, Jensen HM, Dahl RH, George K, Keasling JD, Lee TS, "
        "Leong S, Mukhopadhyay A. Improving microbial biogasoline "
        "production in Escherichia coli using tolerance engineering. "
        "mBio 2014;5:e01932-14",
        url="https://doi.org/10.1128/mBio.01932-14",
        klass="Production campaign",
        tier=2,
        genotypes_n=9,
        genotypes="9 released: 8 tolerance genes plus the control",
        env_n=1,
        env="1 production condition",
        instances_n=9,
        instances_basis="reported",
        phenotype="isopentenol titer in mg per liter at 48 h, mean and SD over 3 replicates",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="single-gene overexpression on a production chassis",
        isoprenol="direct",
        product="isopentenol",
        why="The word biogasoline in the title is isoprenol: the paper opens "
        "``Isopentenol (3-methyl-3-buten-1-ol) is an important target "
        "compound'', and isopentenol and isoprenol are the same molecule, so "
        "this is an isoprenol row under the older name. It is the only "
        "E. coli isoprenol campaign that releases numeric titers. Table 1 "
        "gives mg per liter with SD for the 8 tolerance genes that reduced the "
        "isopentenol growth lag, metR, ibpA, nrdH, soxS, mdlB, fpr, gidB and "
        "yqhD, against an 834 plus or minus 5 mg per liter control, which is "
        "what earns tier 2 on one cleared bar as the only released E. coli "
        "isoprenol production data. A second layer is genuinely deposited and "
        "is larger than the titers: GEO GSE53138 holds 6 arrays, 3 with and 3 "
        "without 0.2 percent isopentenol, on a 4,254-probe platform, which is "
        "an isoprenol-stress transcriptome rather than a genotype axis because "
        "all 6 are one strain. The host is DH1 and the tolerance alleles were "
        "amplified from MG1655, so the two backgrounds are mixed within one "
        "strain. The 40-gene candidate screen behind the 8 winners is "
        "figure-only, and the methods state sampling at 150 and 390 minutes "
        "while only the 150-minute arrays were deposited.",
        accession="GEO GSE53138 (GSM1282891-GSM1282896, platform GPL14649); mBio Table 1 and Tables S2-S4",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        time_axis="titers sampled at 24, 48 and 72 h; Table 1 releases 48 h",
        schema_need="a transcriptome measured against a dosed inhibitor on an "
        "unperturbed genotype, which is an environment-response record rather than a "
        "gene-perturbation record",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="alcohol tolerance genes scored on a production chassis",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="alcohol tolerance",
                yields="alcohol tolerance genes scored on a production chassis",
            ),
            Synergy(
                partner="Wang 2015 isoprenol tolerance",
                partner_status="candidate",
                join="isoprenol",
                yields="tolerance genes by overexpression against tolerance by deletion",
            ),
        ],
    ),
    Candidate(
        name="Wang 2022 P. putida isoprenoids",
        organism="P. putida",
        citation="Wang X, Baidoo EEK, Kakumanu R, Xie S, Mukhopadhyay A, Lee "
        "TS. Engineering isoprenoids production in metabolically "
        "versatile microbial host Pseudomonas putida. Biotechnology for "
        "Biofuels and Bioproducts 2022;15:137",
        url="https://doi.org/10.1186/s13068-022-02235-6",
        klass="Production campaign",
        tier=4,
        genotypes_n=18,
        genotypes="18 registry-resolved, 16 multi-gene",
        env_n=20,
        env="20 medium and supplement conditions",
        instances_n=380,
        instances_basis="product",
        phenotype="isoprenol, epi-isozizaene, limonene and cineole titers in mg per liter, plus growth and glucose",
        dim=1,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="heterologous pathway plus chromosomal deletion",
        isoprenol="direct",
        product="isoprenol, epi-isozizaene",
        why="The first isoprenol production in this host, and the row with the "
        "most multi-gene genotypes of any isoprenol campaign: 16 of its 18 "
        "registry-resolved strains carry either the 3-gene phaABC deletion at "
        "PP_5003 to PP_5005, a 5-to-8-gene heterologous operon, or both. Best "
        "titer 104 mg per liter from 2 percent glucose at 48 h, and 25 mg per "
        "liter from p-coumarate, which makes it the only isoprenol row with an "
        "aromatic carbon source. Two further deletions are informative for "
        "cross-referencing: crc at PP_5292, and PP_2675, the isoprenol "
        "catabolism gene, which sits immediately beside the PP_2674 ethanol "
        "dehydrogenase that Yang 2019 deletes in the same host. Every per-strain "
        "titer is figure-only, which is why this is blocked rather than "
        "buildable: the data statement points at JBEI's Experiment Data Depot "
        "with no study identifier, and that host returned HTTP 502 when checked. "
        "Four further assayed genotypes, two single and one double deletion of "
        "ppc and pycAB plus a crc overexpression strain, carry no strain "
        "identifier and no locus tags at all. The authors also record that "
        "``the polyploid property nature of $P .$ putida may increase the "
        "instability of using a high-copy plasmid for gene expression [48], and "
        "consistent with this we observed signifcant variations among colonies when "
        "screening for productions.'' RE-MEASURED 2026-10-08 "
        "(experiments/036-dataset-fixes-before-kg-build/scripts/"
        "wang2022_release_loadability.py): STILL BLOCKED, with the 502 now diagnosed "
        "rather than merely observed. The transient-status hypothesis is dead: "
        "https://edd.jbei.org/ and public-edd.jbei.org both answer 502 after about 30 "
        "seconds with Cloudflare's own ``origin_bad_gateway`` body, so the EDD origin "
        "is down behind a live edge; a 30-second client timeout returns a read timeout "
        "instead and must not be read as the status. The registry INDEX is up, 200 with "
        "the ICE single-page shell, and it exposes nothing: /entry/019914 is 404 "
        "because JPUB_ is a part id and not the numeric ICE id, every /rest/parts/ path "
        "is 403, /rest/config is 401, and ICE is a parts registry by design, so even an "
        "authorized read would serve plasmid sequence and part metadata, never "
        "measurements. The figure-only claim holds: si1.docx carries exactly TWO "
        "tables, an 8-row amino-acid recipe (whose 8 numeric cells are medium inputs, "
        "not measurements) and a 5-row strain and plasmid inventory, with no embedded "
        "workbooks, no chart parts and no textboxes to hide a figure's source data in; "
        "the two tables OCR'd from paper.md are a 34-row strain and plasmid inventory "
        "and Figure 3's uncaptioned strain-configuration grid, and neither holds a "
        "single bare-number cell. Zero numeric records come from any table. 19 "
        "(strain, condition) titers are recoverable from PROSE, 14 from the Results and "
        "5 cross-referenced to an SI figure caption, and all 19 are ``highest`` or "
        "``up to`` order statistics over a time course rather than per-timepoint "
        "values; 0 of the 19 carry an uncertainty, because every SD in this paper is an "
        "error bar inside a raster. Replication is stated nine times identically in the "
        "SI and seven in the paper, ``Error bars indicate one standard deviation of "
        "triplicates``, so n = 3 with a sample SD; ``replicate``, ``independent``, "
        "``standard error``, ``duplicate`` and ``n =`` occur zero times in either "
        "artifact, and every hit on ``biological`` is an affiliation or funding-office "
        "string, so whether those triplicates are biological or technical is "
        "unresolved and must not be guessed. Two corrections to this row: the "
        "crc-overexpression genotype IS "
        "identified, by the Fig. S6 caption as JPUB_019964 plus plasmid JPUB_019949, so "
        "the unidentified set is THREE and not four (the ppc single, the pycAB single "
        "and their double); and 380 instances does not close, since 18 genotypes times "
        "20 conditions is 360. The headline 104 mg/L is also self-contradicted, a "
        "culture tube in the Results against a batch flask in the abstract.",
        accession="JBEI registry JPUB_019914 to JPUB_019988 (index 200, every data endpoint 401 or 403, and a parts registry carries no measurements); JBEI Experiment Data Depot with no study id, https://edd.jbei.org/ re-tested 2026-10-08 and still HTTP 502 with a Cloudflare origin_bad_gateway body",
        accession_confirmed=False,
        status="blocked",
        confidence="sourced",
        time_axis="sampled at 0, 24, 48 h, and to 72 h for the sesquiterpene",
        schema_need="a titer for several products measured on one genotype, so the "
        "phenotype is a vector over product identity rather than one label",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene",
            why="heterologous isoprenoid pathway plus host deletions",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene",
                partner_status="supported",
                join="isoprenoid pathway",
                yields="heterologous isoprenoid pathway plus host deletions",
            ),
            Synergy(
                partner="Yang 2019 mevalonate from ethanol",
                partner_status="candidate",
                join="PP_2674 and PP_2675 neighborhood",
                yields="two alcohol-catabolism deletions in adjacent loci",
            ),
        ],
    ),
    Candidate(
        name="Rachwalski 2024 mobile CRISPRi",
        organism="E. coli",
        citation="Rachwalski K, Tu MM, Madden SJ, French S, Hansen DM, Brown "
        "ED. A mobile CRISPRi collection enables genetic interaction "
        "studies for the essential genes of Escherichia coli. Cell "
        "Reports Methods 2024;4:100693",
        url="https://doi.org/10.1016/j.crmeth.2023.100693",
        klass="Genetic interaction",
        tier=1,
        genotypes_n=12404,
        genotypes="12,404 gene pairs plus 357 knockdowns",
        env_n=12,
        env="2 media at 6 inducer levels",
        instances_n=63552,
        instances_basis="reported",
        phenotype="normalized colony growth, mean of 2 replicates, no dispersion released",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12-KO+guide",
        modality="essential-gene knockdown crossed with a cataloged deletion",
        why="The only measured pairwise genetic-interaction axis at scale in "
        "either host, and it is released in full rather than as hits. Three "
        "knockdown constructs, of lolA, pssA and mreD, were conjugated into the "
        "whole Keio collection, giving 4,017 distinct deletions crossed against "
        "each, and the entire 356-guide collection was separately crossed into a "
        "lpp deletion. Supplementary Tables S2A, S3 and S4A carry a normalized "
        "growth value for every pair at every condition with no gaps, so the "
        "negative results are present and not only the 68 suppressors and 9 "
        "enhancers of lolA knockdown. The pairing is what makes it valuable here: "
        "an essential gene reachable only by knockdown, crossed with a "
        "non-essential gene reachable only by deletion, is a combination no "
        "deletion collection can produce. Three cautions for the loader. The "
        "methods list five inducer levels and omit 100 nanograms per milliliter, "
        "which the table headers do carry, so trusting the prose drops a sixth of "
        "the dose response. Table S4A holds 4,542 rows for 4,017 unique "
        "deletions, with 485 labels repeated up to five times and no plate or "
        "well key to separate them. And replicate TYPE is contradictory, called "
        "technical in the methods and biological in three figure captions.",
        accession="Zenodo 10.5281/zenodo.10214517 (1.4 GB and 364 MB image archives plus analysis code); Tables S1-S4 as xlsx; vector pFD152 is Addgene 125546",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        schema_need="a genotype combining a cataloged deletion with a guide-borne "
        "knockdown, so one perturbation is an edit to the genome and the other is not",
        analog=Analog(
            dataset="Costanzo 2016 SGA",
            why="pairwise gene-gene fitness across a cataloged array",
        ),
        synergy=[
            Synergy(
                partner="Costanzo 2016 SGA",
                partner_status="supported",
                join="pairwise gene pairs",
                yields="pairwise gene-gene fitness across a cataloged array",
            ),
            Synergy(
                partner="Baba 2006",
                partner_status="candidate",
                join="Keio collection",
                yields="the deletion half of every measured pair",
            ),
        ],
    ),
    Candidate(
        name="Royet 2025 KT2440 metal Tn-seq",
        organism="P. putida",
        citation="Royet K, Kergoat L, Lutz S, Oriol C, Parisot N, Schori C, "
        "Ahrens CH, Rodrigue A, Gueguen E. High-throughput Tn-seq "
        "screens identify both known and novel Pseudomonas putida "
        "KT2440 genes involved in metal tolerance. Environmental "
        "Microbiology 2025;27:e70095",
        url="https://doi.org/10.1111/1462-2920.70095",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=5729,
        genotypes="5,729 genes, 600 essential in LB",
        env_n=4,
        env="4 metals at one dose each",
        instances_n=22916,
        instances_basis="product",
        phenotype="gene-level log2 fold change of insertion reads, metal against LB, with a permutation q",
        dim=1,
        dim_basis="reported",
        seq_basis="KT2440+transposon",
        modality="mariner transposon insertion",
        why="A complete released matrix, which is rarer than a large one: "
        "Supplementary Table S5 carries a log2 fold change and a "
        "Benjamini-Hochberg q for all 5,729 genes under each of cobalt, copper, "
        "zinc and cadmium, with no missing values, so the 22,916 records include "
        "every null and not only the 25 retained hits. Table S4 adds a "
        "four-level essentiality call per gene from a hidden Markov model, 600 "
        "essential and 4,458 non-essential, on 105,349 within-gene TA sites "
        "against 129,002 genome-wide. Sequence provenance is unusually good: the "
        "laboratory isolate was itself resequenced for this work and deposited as "
        "CP036494 at 100 percent average nucleotide identity to the canonical "
        "AE015451.2, and the 12 raw runs map one-to-one onto the six pools by an "
        "SRA isolate attribute whose read counts match Table S3 exactly. Two "
        "limits are structural rather than fixable. The design is non-barcoded "
        "mariner Tn-seq, so the only addressable genotype unit is the GENE and no "
        "per-mutant record can ever be built from it, which is the opposite of a "
        "barcoded library. And there is no multi-gene axis: every screened "
        "genotype carries one insertion, with only a roxS roxR double and a "
        "pvdMNOE operon deletion built by hand for validation. All supplementary "
        "numbers are stored as text strings under merged headers.",
        accession="SRA PRJNA1175559 (SRR31057760-SRR31057771); figshare 10.6084/m9.figshare.28676777; genome CP036494",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        schema_need="a fitness record whose genotype is a gene rather than a strain, "
        "because a non-barcoded insertion pool is never resolvable to a clone",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="genome-wide fitness across dosed chemical stress",
        ),
        synergy=[
            Synergy(
                partner="Hillenmeyer 2008 HIP/HOP",
                partner_status="supported",
                join="condition panel",
                yields="genome-wide fitness across dosed chemical stress",
            ),
            Synergy(
                partner="Thompson 2020 fatty acid and alcohol",
                partner_status="candidate",
                join="KT2440 gene set",
                yields="metal stress beside carbon-source fitness in one host",
            ),
        ],
    ),
    Candidate(
        name="Hawkins 2020 mismatch-CRISPRi",
        organism="E. coli",
        citation="Hawkins JS, Silvis MR, Koo BM, Peters JM, Osadnik H, Jost M, "
        "Hearne CC, Weissman JS, Todor H, Gross CA. Mismatch-CRISPRi "
        "reveals the co-varying expression-fitness relationships of "
        "essential genes in Escherichia coli and Bacillus subtilis. "
        "Cell Systems 2020;11:523-535.e9",
        url="https://doi.org/10.1016/j.cels.2020.09.009",
        klass="CRISPR library screen",
        tier=1,
        genotypes_n=27000,
        genotypes="about 27,000 guides over 270 essential genes",
        env_n=1,
        env="1 medium",
        instances_n=27000,
        instances_basis="product",
        phenotype="relative fitness in doublings against wild type, mean and SD over 4 replicates",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+guide",
        modality="graded CRISPRi knockdown by guide mismatch",
        why="A dose axis on an essential gene, which no deletion collection can "
        "carry. Each essential gene is targeted by 100 guides, 10 fully matched "
        "plus 9 singly mismatched variants of each, so knockdown is titrated "
        "rather than switched, and the paper reports a per-gene "
        "expression-fitness curve over 17 sliding activity bins for all 270 "
        "E. coli essential genes. Table S3 releases per-guide fitness with SD "
        "and the 20-nucleotide spacer, which is the element that distinguishes "
        "the genotypes, and 1,000 non-targeting controls give a measured noise "
        "floor of 0.0825. The guide count is arithmetic over two sourced "
        "figures, 270 analyzed genes times 100 guides: the paper states only "
        "that the libraries exceed 30,000 elements and never splits that between "
        "the two species, so the exact E. coli library size is unstated. Three "
        "cautions. The knockdown LEVEL is a linear-model prediction, not a "
        "measurement, at a cross-validated R-squared of 0.56, and a preliminary "
        "version of that model was used to design the libraries, so the dose "
        "must be typed as an imputed covariate. Guides whose fully matched "
        "member proved non-functional had their whole series excluded before the "
        "released curves, so Table S3 and the curve tables are different "
        "populations. And the article is not open access, so the supplementary "
        "workbooks need a manual retrieval before they can be mirrored.",
        accession="SRA PRJNA574461 (86 runs, 38 E. coli; experiment titles carry the condition); Tables S1-S12 as xlsx behind a paywall",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        time_axis="sampled before and after about 10 doublings, with a third point at 15 for 2 of 4 replicates",
        schema_need="a graded knockdown whose level is a model prediction rather than a "
        "measured quantity, so the dose carries its own uncertainty",
        analog=Analog(
            dataset="Smith 2016 CRISPRi",
            why="essential-gene knockdown reachable only by interference",
        ),
        synergy=[
            Synergy(
                partner="Smith 2016 CRISPRi",
                partner_status="supported",
                join="guide library",
                yields="essential-gene knockdown reachable only by interference",
            ),
            Synergy(
                partner="Rachwalski 2024 mobile CRISPRi",
                partner_status="candidate",
                join="essential gene set",
                yields="a knockdown dose axis beside a knockdown by deletion cross",
            ),
        ],
    ),
    Candidate(
        name="Silvis 2021 CRISPRi morphology",
        organism="E. coli",
        citation="Silvis MR, Rajendram M, Shi H, Osadnik H, Gray AN, Cesar S, "
        "Peters JM, Hearne CC, Kumar P, Todor H, Huang KC, Gross CA. "
        "Morphological and transcriptional responses to CRISPRi "
        "knockdown of essential genes in Escherichia coli. mBio "
        "2021;12:e02561-21",
        url="https://doi.org/10.1128/mBio.02561-21",
        klass="Morphology / imaging",
        tier=2,
        genotypes_n=548,
        genotypes="548 targeting guides over 522 genes",
        env_n=2,
        env="2 induction levels",
        instances_n=585,
        instances_basis="reported",
        phenotype="median cell length and width with their coefficients of variation",
        dim=4,
        dim_basis="reported",
        seq_basis="K-12+guide",
        modality="CRISPRi knockdown",
        why="The only route to a bacterial cell-shape phenotype under gene "
        "perturbation, which is the one modality the built yeast morphology data "
        "has no bacterial counterpart for, and that is what earns tier 2 on one "
        "cleared bar. Table S3 releases median length, median width and a robust "
        "coefficient of variation for each, over 346 uninduced and 239 induced "
        "strain-conditions, summarizing 2,646,096 segmented cells of which "
        "2,257,926 passed a stated width filter. Table S2 adds pooled relative "
        "fitness with 4 replicate columns and individually measured growth rate "
        "and lag. Correcting a common assumption about this paper, there is NO "
        "RNA-seq in it: the transcriptional half of the title is a 4-gene "
        "quantitative PCR panel plus promoter reporters, both figure-only, so no "
        "expression vector can be taken from it. Three further limits. None of "
        "the images, contours or per-cell measurements were deposited anywhere, "
        "so the provenance chain for 2.6 million measured cells terminates at a "
        "65-kilobyte workbook. The imaging replicate count is never stated. And "
        "induction, timepoint and plate format are confounded, uninduced read at "
        "3.5 hours in 96-well plates with a plate-median correction and induced "
        "at 5.5 hours in 384-well plates, so the two are not a dose axis. 163 of "
        "548 strains are flagged contaminated and the flag must travel with the "
        "record.",
        accession="SRA PRJNA669343 (fitness) and PRJNA728203 (strain purity); Tables S1-S3 as xlsx; no image repository",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        schema_need="a shape phenotype whose measured unit is a segmented cell, "
        "summarized per strain as a location and a dispersion for each dimension",
        analog=Analog(
            dataset="Ohya 2005 morphology",
            why="single-cell shape features under gene perturbation",
        ),
        synergy=[
            Synergy(
                partner="Ohya 2005 morphology",
                partner_status="supported",
                join="morphology features",
                yields="single-cell shape features under gene perturbation",
            ),
            Synergy(
                partner="Hawkins 2020 mismatch-CRISPRi",
                partner_status="candidate",
                join="essential gene set",
                yields="shape and fitness on a shared guide-reachable gene set",
            ),
        ],
    ),
    Candidate(
        name="Choe 2019 genome-reduced ALE",
        organism="E. coli",
        citation="Choe D, Lee JH, Yoo M, Hwang S, Sung BH, Cho S, Palsson B, "
        "Kim SC, Cho BK. Adaptive laboratory evolution of a "
        "genome-reduced Escherichia coli. Nature Communications "
        "2019;10:935",
        url="https://doi.org/10.1038/s41467-019-08888-6",
        klass="Multi-omics campaign",
        tier=3,
        genotypes_n=3,
        genotypes="3 strains, 31 resequenced populations",
        env_n=1,
        env="1 evolving condition",
        instances_n=10,
        instances_basis="reported",
        phenotype="matched transcriptome and ribosome-profiling RPKM over 3,457 genes",
        dim=3457,
        dim_basis="reported",
        seq_basis="evolved-WGS",
        modality="adaptive laboratory evolution on a reduced genome",
        why="The only matched transcriptome and translatome in either host, and "
        "the densest per-instance vector in the table: Supplementary Data 6 "
        "releases RPKM for all 3,457 genes across 6 RNA-seq samples and "
        "Supplementary Data 7 the ribosome-protected-fragment RPKM plus "
        "translational efficiency over the same gene set, so expression and "
        "translation are measured on one strain pair. Supplementary Data 5 adds "
        "839 sigma-70 binding peaks. The genotype side is the problem and it is "
        "not a small one: all 31 resequenced samples are POPULATIONS, every "
        "variant table gives allele frequency against a timepoint, and the "
        "flagship evolved strain has no released variant list of its own, so a "
        "per-clone genotype would have to be produced by thresholding day-62 "
        "frequencies, a call the authors never made. The reduced parent MS56 has "
        "no sequence accession, only a laboratory URL behind a bot wall, so "
        "total genomic content is reachable for the MG1655 lineage and not for "
        "the reduced one. The evolution environment is non-stationary by design, "
        "lysogeny broth supplement falling from 0.1 percent to zero over 62 days, "
        "and the released trajectory is cell density rather than growth rate, so "
        "early and late passages are not the same condition. Gene identifiers in "
        "both expression tables are bare gene names with no locus tag.",
        accession="ENA PRJEB21199 (51 runs: 31 WGS, 6 RNA-seq, 8 Ribo-seq, 6 ChIP-seq); Supplementary Data 1-7 as xlsx; MG1655 is NC_000913.3, MS56 has no accession",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        time_axis="populations resequenced at 20 timepoints over 62 days, 807 generations",
        schema_need="a genotype that is a population allele-frequency vector rather "
        "than one strain, and a reference that is itself a reduced derivative with no "
        "deposited assembly",
        analog=Analog(
            dataset="Caudal 2024 pan-transcriptome",
            why="expression across genetically distinct backgrounds",
        ),
        synergy=[
            Synergy(
                partner="Caudal 2024 pan-transcriptome",
                partner_status="supported",
                join="gene set",
                yields="expression across genetically distinct backgrounds",
            ),
            Synergy(
                partner="PRECISE-1K",
                partner_status="candidate",
                join="MG1655 gene set",
                yields="an evolved reduced genome against a large expression compendium",
            ),
        ],
    ),
    Candidate(
        name="Yang 2019 mevalonate from ethanol",
        organism="P. putida",
        citation="Yang J, Son JH, Kim H, Cho S, Na JG, Yeon YJ, Lee J. "
        "Mevalonate production from ethanol by direct conversion "
        "through acetyl-CoA using recombinant Pseudomonas putida, a "
        "novel biocatalyst for terpenoid production. Microbial Cell "
        "Factories 2019;18:168",
        url="https://doi.org/10.1186/s12934-019-1213-y",
        klass="Production campaign",
        tier=4,
        genotypes_n=11,
        genotypes="11 strains, 9 multi-gene, up to 9 perturbed loci",
        env_n=6,
        env="6 conditions including 3 pH levels",
        instances_n=63,
        instances_basis="product",
        phenotype="mevalonate, ethanol and acetate titers with growth and medium pH",
        dim=5,
        dim_basis="reported",
        seq_basis="engineered-chassis",
        modality="stacked chromosomal deletion plus heterologous pathway",
        isoprenol="precursor",
        product="mevalonate",
        why="The deepest deletion stacking of any row in either host and the "
        "cleanest sequence definition, which is the opposite of what its data "
        "statement suggests. Five chromosomal deletions accumulate in order, "
        "endA, endX, qedH-I, qedH-II and phaG, each identified by a GenBank gene "
        "identifier and each with its upstream and downstream homology-arm "
        "primers released, so the deletion BOUNDARIES are pinned rather than "
        "implied; the method is markerless with sucrose counter-selection, so no "
        "marker is left unaccounted for. All six heterologous genes are released "
        "as literal codon-optimized nucleotide sequences totaling 9,283 bases "
        "including the ribosome-binding regions. The deepest strain carries 5 "
        "deletions plus 4 heterologous genes in one genotype. Two cautions. The "
        "data statement reads ``Not applicable'' in full, so every titer is "
        "figure-only and only a handful of values with standard deviations "
        "survive in prose, which is why this is blocked. And the locus tags are "
        "absent from the paper: PP_3375, PP_2451, PP_2674, PP_2679 and PP_1408 "
        "were DERIVED from the stated gene identifiers rather than quoted, and "
        "must be recorded as derived. PP_2674 sits beside the PP_2675 isoprenol "
        "catabolism gene that Wang 2022 deletes in the same host. RE-MEASURED "
        "2026-10-08 (experiments/036-dataset-fixes-before-kg-build/scripts/"
        "yang2019_release_loadability.py): STILL BLOCKED, and every structural claim "
        "above is now verified. The data statement is ``Availability of data and "
        "materials Not applicable.`` with nothing else in the section, and a sweep for "
        "deposit, repository, accession, supplementary data, raw data and dataset "
        "returns zero hits. The 9,283 bases are confirmed EXACTLY by summing the six "
        "released sequences: mvaE 2,430, mvaS 1,185, atoB 1,216, acs 1,994, eutE 1,432 "
        "and nphT7 1,026, which is 181 leader bases plus 9,102 coding bases, every CDS "
        "in frame and stop-terminated and every leader carrying a Shine-Dalgarno motif. "
        "The five GenBank gene identifiers are quoted in one sentence: endA 1047019, "
        "endX 1045620, qedH-I 1046117, qedH-II 1046129, phaG 1046114. The locus tags "
        "are confirmed ABSENT, zero occurrences of PP_ in paper.md, the layout JSON or "
        "either SI file, so the five PP_ tags in this row stay derived and unverifiable "
        "against the mirror. Neither SI file carries a measurement: Additional file 1 "
        "is a 65-row primer table and Additional file 2 is six unbroken sequence "
        "paragraphs with no table at all, and paper.md's single table is a strain and "
        "plasmid inventory. Exactly SIX (strain, condition) mevalonate titers are "
        "recoverable without a figure, four with an SD from triplicate flasks (ELPP010 "
        "1.70 +/- 0.55, ELPP110 2.43 +/- 1.34, ELPP111 2.88 +/- 1.16, ELPP211 4.07 "
        "+/- 0.29 g/L at 27 h) and two from the fermenter with none (ELPP311 4.18 g/L "
        "at pH 7.0, 4.60 g/L at pH 6.75), plus two ethanol values and ZERO acetate "
        "values; all 60 time-course readouts are figure-only. Replication IS stated, "
        "``All strains were cultured under the above conditions and in triplet for "
        "reproducibility confrmation`` and ``All the experiments were performed in "
        "triplicates and standard deviations of triplet culture were shown in the form "
        "of error bars`` in six legends, so the flask values are n = 3 with a sample "
        "SD; Figure 9's three batch runs carry no replicate statement, which is "
        "consistent with their two titers being the only ones without an SD. A FOURTH "
        "titer-family refusal for want of a RELEASED reference titer: ELPP000, the base "
        "strain, has no figure-free number at all, so "
        "ProductTiterExperimentReference.phenotype_reference cannot be built. The ten "
        "homology-arm primer pairs do pin each deletion to the base in sequence space, "
        "since the two inner primers cross-prime for 5 of 5 loci, but the interval's "
        "coordinates need the KT2440 reference and the paper gives only ``About 500 - "
        "2000 bp``; F-qedH-I-upstream and F-qedH-II-upstream are also byte-identical, "
        "so that one arm boundary should not be trusted without alignment. Two "
        "corrections: the deepest genotype is a THREE-way tie at 9 perturbed loci "
        "(ELPP311 5+4, ELPP213 4+5, ELPP221 4+5), and 63 instances is not derivable "
        "from the bytes by any stated rule.",
        accession="GenBank gene ids for all five deleted loci and accessions for all six heterologous genes; Additional file 2 holds the codon-optimized sequences; no data deposit",
        accession_confirmed=True,
        status="blocked",
        confidence="sourced",
        time_axis="metabolic time courses to 27 h in flask, 3 days in the fermenter",
        schema_need="a genotype stacking five markerless deletions with four "
        "heterologous genes, where the deletion boundary is given by primer rather "
        "than by coordinate",
        analog=Analog(
            dataset="Ozaydin 2013 beta-carotene",
            why="stacked host deletions feeding a heterologous isoprenoid route",
        ),
        synergy=[
            Synergy(
                partner="Ozaydin 2013 beta-carotene",
                partner_status="supported",
                join="mevalonate pathway",
                yields="stacked host deletions feeding a heterologous isoprenoid route",
            ),
            Synergy(
                partner="Wang 2022 P. putida isoprenoids",
                partner_status="candidate",
                join="PP_2674 and PP_2675 neighborhood",
                yields="two alcohol-catabolism deletions in adjacent loci",
            ),
        ],
    ),
    Candidate(
        name="Niu 2019 pinene evolved",
        organism="E. coli",
        citation="Niu FX, Huang YB, Ji LN, Liu JZ. Genomic and transcriptional "
        "changes in response to pinene tolerance and overproduction in "
        "evolved Escherichia coli. Synthetic and Systems Biotechnology "
        "2019;4:113-119",
        url="https://doi.org/10.1016/j.synbio.2019.05.001",
        klass="Combinatorial design",
        tier=4,
        genotypes_n=80,
        genotypes="77 single-target strains plus 2 six-target combinations",
        env_n=1,
        env="1 pinene dose",
        instances_n=77,
        instances_basis="reported",
        phenotype="ratio of growth and of pinene titer against the unactivated parent, 3 replicates",
        dim=2,
        dim_basis="reported",
        seq_basis="evolved-WGS",
        modality="CRISPR activation and interference on an evolved host",
        isoprenol="precursor",
        product="pinene",
        why="Two properties worth the row. First, a complete released "
        "perturbation result rather than a hit list: Supplementary Tables 3 and "
        "4 carry both a growth ratio and a pinene ratio for all 57 activation "
        "and all 20 interference targets, with every guide sequence in Table 1, "
        "and two six-target combination strains cross activation of flgFGH, "
        "sufBCDS, dusB, rpoA, yehA and hslU in one strain and interference of "
        "ydiJ, yjbQ, prpR, marR, fabR and cedA in another. Second, a "
        "374-variant table that is the only evolved-strain genotype in either "
        "host carrying b-numbers and absolute coordinates with reference and "
        "alternate bases, all at frequency 1.00. That table comes with a caveat "
        "that must travel with it: reads were aligned to MG1655 while the parent "
        "is BW25113, with no stated parental subtraction, so an unknown and "
        "probably large share of the 374 calls are background differences "
        "between the two K-12 strains rather than evolution-acquired mutations. "
        "It is a genotype description against MG1655, not a validated mutation "
        "list. The reads themselves were never deposited. No absolute pinene "
        "titer appears anywhere, only ratios, and the 182-gene quantitative PCR "
        "panel is figure-only. A 2020 erratum attaches to the citation and "
        "corrects only a competing-interest statement.",
        accession="Supplementary mmc1.docx holds the 374-variant table with b-numbers, all guide sequences, and both ratio tables; no sequence-data accession",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        schema_need="an evolved genotype released as a variant list against a "
        "reference that is not its own parent, so the call set needs a provenance gap "
        "rather than a clean edit list",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="activation and interference over one gene set",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="activation and interference",
                yields="activation and interference over one gene set",
            ),
            Synergy(
                partner="Tian 2019 isopentenol CRISPRi",
                partner_status="candidate",
                join="multiplex guide array",
                yields="two multiplex guide campaigns toward isoprenoid products",
            ),
        ],
    ),
    Candidate(
        name="Reyes 2011 n-butanol library",
        organism="E. coli",
        citation="Reyes LH, Almario MP, Kao KC. Genomic library screens for "
        "genes involved in n-butanol tolerance in Escherichia coli. "
        "PLoS ONE 2011;6:e17678",
        url="https://doi.org/10.1371/journal.pone.0017678",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=14,
        genotypes="14 released of 140 assayed",
        env_n=2,
        env="2 butanol levels",
        instances_n=14,
        instances_basis="reported",
        phenotype="two growth-rate statistics per gene with a t-test p, 4 biological replicates",
        dim=2,
        dim_basis="reported",
        seq_basis="K-12+ORF-plasmid",
        modality="cataloged single-ORF overexpression and single deletion",
        isoprenol="analog",
        product="n-butanol",
        why="The only row that scores overexpression and deletion of the same "
        "trait side by side from cataloged collections: 55 ASKA overexpression "
        "clones and 84 Keio deletions were assayed individually after a pooled "
        "enrichment of a roughly 14,000-clone genomic library at sevenfold "
        "coverage. Both halves resolve to public collections, so the sequence "
        "basis is clean on both signs of perturbation. What limits it badly is "
        "release: only the 14 winners carry values, 11 overexpression and 3 "
        "deletion, and the 126 other assayed genotypes are not figure-locked but "
        "unrecoverable, never plotted per gene. The array-CGH enrichment layer is "
        "deposited as GEO GSE26223, 16 two-color arrays on a 4,062-row platform, "
        "but its replicates are technical rather than biological and the mapping "
        "from the 193 enriched genes to the 55 that were picked is not released. "
        "The paper disagrees with itself on the candidate count, 193 in one place "
        "and 194 in another, and on the first enrichment rung, 0.5 percent in the "
        "results against 0 percent in the methods; both readings must be "
        "recorded. One of the 3 deletion hits, rph, is self-flagged as an "
        "artifact because the host background already carries an inactivating "
        "frameshift there, so that record needs a reliability flag. The "
        "overexpression induction level is never stated.",
        accession="GEO GSE26223 (GSM643831-GSM643846, platform GPL8984); Tables 3 and 5 hold the 14 released genotypes; ASKA and Keio collections for the strains",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        schema_need="a matched over- and under-expression pair of the same gene on one "
        "trait, which is what a dosage-aware model needs and no built row provides",
        analog=Analog(
            dataset="Lopez 2024 isobutanol",
            why="alcohol tolerance over a cataloged collection",
        ),
        synergy=[
            Synergy(
                partner="Lopez 2024 isobutanol",
                partner_status="supported",
                join="alcohol tolerance",
                yields="alcohol tolerance over a cataloged collection",
            ),
            Synergy(
                partner="Foo 2014 isopentenol tolerance",
                partner_status="candidate",
                join="alcohol tolerance by overexpression",
                yields="two overexpression tolerance screens on different alcohols",
            ),
        ],
    ),
    Candidate(
        name="Zou 2022 P. putida furfural",
        organism="P. putida",
        citation="Zou L, Jin X, Tao Y, Zheng Z, Ouyang J. Unraveling the "
        "mechanism of furfural tolerance in engineered Pseudomonas "
        "putida by genomics. Frontiers in Microbiology 2022;13:1035263",
        url="https://doi.org/10.3389/fmicb.2022.1035263",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=20,
        genotypes="20 strains, a 5-gene deletion parent and an evolved derivative",
        env_n=6,
        env="6 inhibitor conditions",
        instances_n=18,
        instances_basis="reported",
        phenotype="growth plus furfural, HMF and acetate concentrations, 2 replicates",
        dim=4,
        dim_basis="reported",
        seq_basis="evolved-WGS",
        modality="adaptive evolution plus single-gene overexpression",
        isoprenol="analog",
        product="furfural",
        why="A cross-host counterpart to the built yeast furfural screen, which "
        "is the reason to want it: furfural tolerance measured in a bacterial "
        "host on a defined genotype series, against the same inhibitor a yeast "
        "CRISPR screen already covers. Supplementary Tables S3 to S6 release the "
        "mock-hydrolysate time courses as numbers, with optical density, furfural, "
        "HMF and acetate per timepoint for the parent and the evolved strain. The "
        "evolved reads are deposited as SRR19970103, which makes this the only "
        "evolved row here whose variant calls can be reproduced. The parent is "
        "itself a 5-gene deletion, of gcd plus the four gtsABCD genes, and the "
        "evolved derivative carries 37 further mutations after four months of "
        "escalating furfural. Three limits. Only 11 of the 37 mutations are "
        "individually listed; the other 26 are counted by class in Table S7 and "
        "never named, so the evolved genotype is incomplete as released and "
        "recoverable only by re-calling the deposited reads. The parent was never "
        "sequenced and variants were called against the public reference, so the "
        "37 calls conflate furfural selection with the deletion construction and "
        "with lab drift, which the paper does not acknowledge. Every recombinant "
        "strain phenotype is figure-only, and every uncertainty in the paper is a "
        "two-point standard deviation.",
        accession="SRA PRJNA855347 (SRR19970103, SAMN29490017); Data_Sheet_1.docx holds Tables S1-S7; reference NC_002947.4",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        time_axis="hydrolysate courses sampled to 72 h",
        schema_need="an evolved genotype whose released variant list is a partial "
        "enumeration plus per-class counts, with no parental resequencing to subtract",
        analog=Analog(
            dataset="Lian 2019 CRISPR-AID",
            why="furfural tolerance on a defined perturbation series",
        ),
        synergy=[
            Synergy(
                partner="Lian 2019 CRISPR-AID",
                partner_status="supported",
                join="furfural",
                yields="furfural tolerance on a defined perturbation series",
            ),
            Synergy(
                partner="Borchert 2023 lignin tolerance",
                partner_status="candidate",
                join="KT2440 inhibitor panel",
                yields="furfural beside the lignin-stream inhibitors in one host",
            ),
        ],
    ),
    Candidate(
        name="Li 2021 mevalonate flux",
        organism="E. coli",
        citation="Li Y, Xian H, Xu Y, Zhu Y, Sun Z, Wang Q, Qi Q. Fine tuning "
        "the glycolytic flux ratio of EP-bifido pathway for mevalonate "
        "production by enhancing glucose-6-phosphate dehydrogenase and "
        "CRISPRi suppressing 6-phosphofructose kinase in Escherichia "
        "coli. Microbial Cell Factories 2021;20:32",
        url="https://doi.org/10.1186/s12934-021-01526-1",
        klass="Metabolome / flux",
        tier=4,
        genotypes_n=3,
        genotypes="3 strains with a released flux map",
        env_n=1,
        env="1 labeling condition",
        instances_n=3,
        instances_basis="reported",
        phenotype="fitted net flux for 198 reactions with 90 percent confidence bounds",
        dim=198,
        dim_basis="reported",
        seq_basis="K-12-KO",
        modality="promoter replacement plus CRISPRi knockdown",
        isoprenol="precursor",
        product="mevalonate",
        why="Worth the row for its flux layer alone, which is released in full "
        "where the titers are not: Additional file 2 carries 198 reaction rows "
        "per strain with a best fit plus lower and upper 90 percent bounds, in "
        "both absolute and normalized units, with the fit diagnostics printed, "
        "so a carbon-flux phenotype vector can be ingested directly. The "
        "perturbation design pairs a 5-level synthetic promoter series over zwf, "
        "whose relative strengths are released, with CRISPRi of pfkA in a "
        "pfkA-positive host, because deleting pfkA in this background caused a "
        "severe growth defect. Serious defects to record. The paper states NO "
        "replicate count, has no statistics section and reports no error "
        "statistic anywhere, so nothing from it can carry an uncertainty. Its "
        "CRISPRi axis is three guide positions, not a strength series, and the "
        "repression depth was explicitly never measured, so there is no dose "
        "coordinate. Mevalonate titers are figure-only for about 22 of its 27 "
        "strains. And the flux supplement covers a different strain set than the "
        "text claims: the control's map is absent and a fourth strain's map is "
        "present but never discussed, so the released fluxes cannot be matched to "
        "the published figure.",
        accession="Additional file 2 (xlsx) holds the 198-reaction flux maps; promoters are iGEM Anderson registry parts; no data accession",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        schema_need="a fitted flux distribution as the phenotype, where each value "
        "carries an interval from the fit rather than a replicate standard deviation",
        analog=Analog(
            dataset="Zelezniak 2018 metabolome",
            why="internal flux state under targeted perturbation",
        ),
        synergy=[
            Synergy(
                partner="Zelezniak 2018 metabolome",
                partner_status="supported",
                join="central carbon reactions",
                yields="internal flux state under targeted perturbation",
            ),
            Synergy(
                partner="Ishii 2007",
                partner_status="candidate",
                join="central carbon flux",
                yields="two fitted flux sets on a shared reaction network",
            ),
        ],
    ),
    # -----------------------------------------------------------------------
    # The second pass over the 300-row discovery queue. Selection ran the
    # requested order: scale times density first, then overlap with a modality
    # the store already serves for E. coli, then the isoprenol axis for
    # P. putida. Every count below was read off the released file, not off the
    # abstract, which is what separated these three from the five that read as
    # genome-wide and release a hit list.
    # -----------------------------------------------------------------------
    Candidate(
        name="Teteneva 2024 lake water",
        organism="E. coli",
        citation="Teteneva N, Sanches-Medeiros A, Sourjik V. Genome-wide screen "
        "of genetic determinants that govern Escherichia coli growth and "
        "persistence in lake water. The ISME Journal 2024;18(1):wrae096",
        url="https://doi.org/10.1093/ismejo/wrae096",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=3691,
        genotypes="3,691 genes carrying insertions",
        env_n=18,
        env="3 lake water samples x filtered or not x 3 days",
        instances_n=66162,
        instances_basis="reported",
        phenotype="gene-level RB-TnSeq fitness (Wetmore normalization)",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="Counted off the file rather than the abstract: Supplementary Table S4 "
        "holds 11,027 gene by water-sample rows and 66,162 non-empty fitness "
        "values across its six columns, filtered and non-filtered water at days "
        "2, 4 and 8, for 3,691 distinct genes. The library is 430,849 unique "
        "insertions in 3,833 genes of E. coli K-12 W3110 RpoS+, built by the "
        "Wetmore protocol, and fitness is averaged over three technical "
        "replicates per sample. Two things make it worth the slot. The host is "
        "W3110 rather than the BW25113 that Price 2018, Shiver 2016 and Tong "
        "2020 all sit on, so joining it to them tests whether a gene-level "
        "fitness effect survives a change of K-12 background, which no pair of "
        "rows already on the list can ask. And the environment is an "
        "oligotrophic natural water rather than a defined medium or a drug, "
        "which is the only such condition in either host here. The rpoS "
        "direction is the opposite sign to the usual stress result, since rpoS "
        "mutants gain fitness in filtered water and lose viability, so the row "
        "carries a documented growth-versus-survival split that a single "
        "fitness number compresses.",
        accession="Supplementary Table S4 (xlsx, gene fitness); NCBI BioProject PRJNA1043681 (lake water) and PRJNA1073534 (Tn5 library annotation); code at github.com/NataliyaTeteneva/Tn5_library_analysis",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant fitness under an environmental stress",
        ),
        synergy=[
            Synergy(
                partner="Price 2018",
                partner_status="candidate",
                join="E. coli gene identifier",
                yields="the same fitness statistic in a second K-12 background",
            ),
            Synergy(
                partner="Mutalik 2020",
                partner_status="candidate",
                join="RB-TnSeq barcode fitness",
                yields="a natural-water condition beside the phage panel",
            ),
        ],
    ),
    Candidate(
        name="Wang 2024 rifampicin Tn-seq",
        organism="E. coli",
        citation="Wang Y, Fu H, Shi X-J, Zhao G-P, Lyu L-D. Genome-wide screen "
        "reveals cellular functions that counteract rifampicin lethality "
        "in Escherichia coli. Microbiology Spectrum 2024;12(1):e0289523",
        url="https://doi.org/10.1128/spectrum.02895-23",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=4419,
        genotypes="4,419 genes",
        env_n=6,
        env="3 rifampicin multiples of MIC x 2 exposure times",
        instances_n=26514,
        instances_basis="reported",
        phenotype="TRANSIT resampling log2 fold change, with p and adjusted p",
        dim=1,
        dim_basis="reported",
        seq_basis="K-12+transposon",
        modality="transposon insertion",
        why="Table S2 is the whole matrix and not the hits: six sheets, one per "
        "condition, each carrying all 4,419 genes with insertion-site count, "
        "mean control and experimental read depth, log2 fold change, raw p and "
        "adjusted p, for 26,514 gene by condition rows. The separate hit table "
        "S3 holds 911 genes, so the released negatives outnumber the reported "
        "positives by about twenty-nine to one, which is the ratio that decides "
        "whether a row can train anything. Host is K-12 MG1655 with a Tn5 "
        "library at 71.4 insertions per kilobase, two biological replicates, "
        "and the design is a dose by time grid, 0.25x, 4x and 20x MIC at 1 and "
        "3 hours, which is a shape no other row here has: every other drug "
        "screen on the list is one dose per drug. It joins Choe 2025 and Shiver "
        "2016 on the drug axis and Girgis 2009 on the modality, so the same "
        "gene under the same antibiotic can be read as knockdown, deletion and "
        "insertion fitness.",
        accession="Table S2 spectrum.02895-23-s0002.xlsx (raw resampling, all genes); Table S3 identified genes; Table S4 pathway enrichment",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        analog=Analog(
            dataset="Hillenmeyer 2008 HIP/HOP",
            why="pooled mutant fitness across a drug dose series",
        ),
        synergy=[
            Synergy(
                partner="Choe 2025",
                partner_status="candidate",
                join="antibiotic panel",
                yields="knockdown and insertion fitness for one gene under one drug",
            ),
            Synergy(
                partner="Girgis 2009",
                partner_status="candidate",
                join="transposon insertion fitness under antibiotics",
                yields="a dose and time grid where the earlier row has one dose",
            ),
        ],
    ),
    Candidate(
        name="Mohiuddin 2022 promoter library",
        organism="E. coli",
        citation="Mohiuddin SG, Massahi A, Orman MA. High-Throughput Screening "
        "of a Promoter Library Reveals New Persister Mechanisms in "
        "Escherichia coli. Microbiology Spectrum 2022;10(1):e0225321",
        url="https://doi.org/10.1128/spectrum.02253-21",
        klass="Transcriptome",
        tier=1,
        genotypes_n=1930,
        genotypes="1,930 promoter-GFP reporter strains, 1,809 distinct promoters",
        env_n=4,
        env="untreated plus ampicillin, ofloxacin and gentamicin",
        instances_n=7720,
        instances_basis="product",
        phenotype="GFP fluorescence time course over 9 reads, and fold change against untreated",
        dim=9,
        dim_basis="reported",
        seq_basis="K-12+promoter-reporter",
        modality="promoter-GFP reporter",
        why="Supplemental Data Set 1 releases the full grid twice over: 69,480 raw "
        "GFP readings, 1,930 reporter strains at nine times under four "
        "conditions, and 52,110 fold changes against the untreated arm. The "
        "library is the E. coli K-12 MG1655 collection of more than 1,900 "
        "native promoters fused to a fast-folding GFP on a low-copy plasmid, "
        "treated in early stationary phase with ampicillin at 200, ofloxacin at "
        "5 and gentamicin at 50 micrograms per milliliter for five hours. "
        "Nothing in the genome is perturbed, which is why it sits beside the "
        "screens rather than among them, and it is the reason to take it: the "
        "store serves drug-response FITNESS for E. coli from four rows and no "
        "drug-response EXPRESSION at all, so this is the half that is missing. "
        "Two of its three drugs are also in the Choe 2025 CRISPRi panel, so a "
        "gene can be read as both a knockdown fitness cost and an induced "
        "promoter under the same compound. The time axis is the second reason: "
        "nine reads across the exposure, where the RNA-seq compendia are "
        "endpoint.",
        accession="Supplemental File 2 spectrum02253-21_supp_2_seq5.xlsx, sheets Raw Data and Fold Change",
        accession_confirmed=True,
        status="candidate",
        confidence="sourced",
        time_axis="9 reads across a 5 hour exposure",
        schema_need="a genotype that is the reference genome plus a reporter plasmid "
        "which is present but not perturbing, so the record is expression under an "
        "environment rather than a perturbation phenotype",
        analog=Analog(
            dataset="Nadal-Ribelles 2019 single-cell",
            why="per-gene expression readout under a condition panel",
        ),
        synergy=[
            Synergy(
                partner="Choe 2025",
                partner_status="candidate",
                join="antibiotic panel",
                yields="induced expression beside knockdown fitness for one gene",
            ),
            Synergy(
                partner="PRECISE-1K",
                partner_status="candidate",
                join="E. coli gene identifier",
                yields="a reporter time course against an endpoint RNA-seq compendium",
            ),
        ],
    ),
]

EXCLUDED: list[Excluded] = [
    # Five isoprenoid papers from the 300-publication sweep, each dropped on a rule
    # rather than on low value. Recorded so the verification is not repeated.
    Excluded(
        name="Mi 2014, geranic acid in P. putida DSM 12264",
        rule="no-sequence",
        reason=(
            "The host is DSM 12264, a solvent-tolerant strain that is not KT2440, and "
            "no genome accession for it appears anywhere in the paper. PP_ locus tags "
            "therefore do not apply and the chassis genome is unidentified, so there "
            "is no route to the total genomic content of any strain. The enzyme "
            "responsible for the key endogenous geraniol oxidation is also unidentified, "
            "stated by the authors as a hypothesis. Two comparisons made in KT2440, the "
            "host actually of interest, are reported as data not shown"
        ),
    ),
    Excluded(
        name="Chen 2013, lycopene by chromosomal evolution in E. coli",
        rule="no-sequence",
        reason=(
            "The perturbation is triclosan-driven tandem gene amplification, and its "
            "extent is quantified only as about 30 copies by quantitative PCR of a "
            "single gene against one reference gene. The amplicon boundaries were never "
            "sequenced or mapped and no strain was resequenced, so an array of unstated "
            "extent cannot be written as a genome sequence, and mutations hitchhiking "
            "through six rounds of escalating selection are unknown. The released "
            "phenotype tables are good, with mean and SD for 20 strains, but the "
            "perturbation axis itself, triclosan dose against copy number, is "
            "figure-only"
        ),
    ),
    Excluded(
        name="Glebes 2014, SCALEs furfural dosage map in E. coli",
        rule="no-per-record-data",
        reason=(
            "The headline is the largest in the whole queue and the release is the "
            "smallest part of it. More than ten to the fifth genomic fragments were "
            "assayed by microarray at about 125 nucleotide resolution across the "
            "genome, under furfural at 0.75 grams per liter against a control, and "
            "the only table deposited is Table S2, 268 enriched genes with one "
            "fitness score each. The microarray data are not in GEO or anywhere "
            "else named in the paper, so the per-clone frequencies the method is "
            "built on are gone, and the surviving 268 rows are the enriched tail "
            "with every neutral and depleted fragment discarded. Ranking this by "
            "its abstract would have placed it near the top of the list on a "
            "measurement count that no longer exists"
        ),
    ),
    Excluded(
        name="Casanova-Hampton 2021, copper stress across the Keio collection",
        rule="no-per-record-data",
        reason=(
            "The whole Keio collection was stamped onto LB agar at 0, 3 and 6 "
            "millimolar copper sulfate, which would be about twelve thousand "
            "mutant by dose observations, but the paper states that phenotypic "
            "response was determined qualitatively relative to on-plate controls "
            "at 24, 48 and 72 hours. There is no colony size, no growth index and "
            "no score of any kind, and what is released is the call: 43 sensitive "
            "and 25 resistant mutants in Tables S4 and S5, plus a 36-mutant "
            "targeted rescreen of iron genes. A qualitative call carries no "
            "uncertainty and cannot be compared against the quantitative colony "
            "scores the store already holds for the same collection"
        ),
    ),
    Excluded(
        name="Mei 2026, ciprofloxacin tolerance across the Keio collection",
        rule="no-per-record-data",
        reason=(
            "A two-step genome-wide screen of the Keio library, stationary-phase "
            "starvation then dilution into antibiotic, and the data availability "
            "statement says all data are in the Supporting Information, which is a "
            "single PDF. Nothing machine-readable is deposited, and what the PDF "
            "carries is the outcome, 37 ciprofloxacin-sensitive and 11 tolerant "
            "mutants out of the roughly four thousand screened. The 48 named "
            "mutants are a hit list, and the survival values behind the other "
            "mutants are what would have made it a dataset"
        ),
    ),
    Excluded(
        name="Huang 2022, pinene by shotgun library in E. coli MG1655",
        rule="no-per-record-data",
        reason=(
            "The data availability statement reads Not applicable, there is no results "
            "table of any kind, and only two absolute titers plus three percentages are "
            "recoverable from prose. The library screen also did not work as a screen: a "
            "library of more than ten to the fifth transformants yielded four surviving "
            "colonies, which the authors attribute to loss during construction or "
            "pinene toxicity, so there is no genotype-phenotype matrix to ingest. "
            "Insert boundaries for the four clones and insertion coordinates for the "
            "three promoter replacements are both unstated"
        ),
    ),
    Excluded(
        name="Mireles 2026, long-chain alcohol metabolism in P. putida",
        rule="no-per-record-data",
        reason=(
            "Sequencing data are available from the corresponding author upon "
            "reasonable request, with no deposit and no accession, and the evolved-clone "
            "phenotype is figure-only, so genotype and phenotype cannot both be taken as "
            "released values. The released variant table also disagrees with the text, "
            "84 per-clone calls against 91 claimed, and labels one mutation PP_1802 "
            "where the text argues from PP_1801. Separately it is off this table's "
            "phenotype axis: the alcohols are C16 and C20 used as sole carbon sources "
            "and the paper states that toxicity is absent, so it measures catabolism "
            "rather than tolerance"
        ),
    ),
    Excluded(
        name="Hernandez-Arranz 2019, isoprenoid precursors in P. putida",
        rule="no-per-record-data",
        reason=(
            "Lycopene is reported in two incompatible ways that cannot be joined. The "
            "one table of measurements gives nanograms per milliliter by chromatography "
            "for four strains, all of them the lowest producers, while every other "
            "result is an absorbance normalized to cell density and expressed as a fold "
            "change against a control. The headline fifty-fold strain has no absolute "
            "value, so ten of fourteen genotypes carry only a ratio. There is no strain "
            "table, no supplementary file, and no accession of any kind, and the "
            "perturbed native genes are never given a locus tag"
        ),
    ),
    Excluded(
        name="Multi-host cell-factory capacity resource (modeled yields)",
        rule="not-a-dataset",
        reason=(
            "Genome-scale model predictions of production capacity across host and "
            "product combinations. Every value is computed rather than measured, so it "
            "belongs in a model-derived partition and must not enter an experimental "
            "instance or dataset count. Useful for negative sampling and as a feature "
            "source, which is a different role from training data"
        ),
    ),
    Excluded(
        name="Simulated FSEOF rows inside the curated engineering corpora",
        rule="not-a-dataset",
        reason=(
            "The larger metabolic-engineering corpora carry both experimentally "
            "reported modifications and rows generated by flux-scanning enforced "
            "objective flux. The simulated rows are predictions of what an "
            "intervention would do and are excluded or separately typed; mixing them "
            "into an experimental count is the specific error these corpora invite"
        ),
    ),
    Excluded(
        name="Adaptive laboratory evolution without resequencing",
        rule="no-sequence",
        reason=(
            "An evolved population differs from its parent at unmapped positions, so "
            "there is no genotype to map a phenotype from. Only resequenced clones and "
            "explicitly reconstructed mutations from such a campaign are admitted, "
            "which is why the tolerance-evolution rows enter on their sequenced "
            "isolates rather than on their lineages"
        ),
    ),
    Excluded(
        name="Genome shuffling, chemical and UV mutagenesis strain panels",
        rule="no-sequence",
        reason=(
            "Same gate as unsequenced evolution, and it bites harder: the selected "
            "isolate carries an unknown number of edits at unknown positions. No "
            "amount of phenotypic value substitutes for a missing genotype"
        ),
    ),
    Excluded(
        name="Other bacterial hosts (B. subtilis, C. glutamicum, Z. mobilis, cyanobacteria)",
        rule="off-host",
        reason=(
            "Real and in several cases large, but outside the two hosts this list "
            "covers. The generalization axis is single-cell bioproduction hosts, so "
            "these become candidates once either bacterial host is actually served and "
            "the per-host schema and genome-tier cost has been paid once"
        ),
    ),
    Excluded(
        name="EcoCyc and the curated annotation databases",
        rule="not-a-dataset",
        reason=(
            "EcoCyc releases per-class flat files covering 4,688 genes, 479 pathways "
            "and 6,124 regulatory interactions for the reference strain (BioCyc v30.0, "
            "checked live). Those are annotations of one genome rather than measured "
            "genotype-by-environment records, so under a ranking on instances times "
            "dimensionality the curation would outrank real screens. It belongs in the "
            "reference layer the graph already has, alongside the gene ontology, not in "
            "a candidate list of experiments"
        ),
    ),
    Excluded(
        name="Review and perspective tables of reported titers",
        rule="not-a-dataset",
        reason=(
            "A titer quoted in a review is a second-hand number without the strain "
            "construction, the medium or the replicate structure that makes it a "
            "record. The primary paper is the candidate; the review is a discovery "
            "index, and the curated corpora already serve that role better"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Rendering. The same helpers and table shapes the yeast document uses, so the two
# read as one family and a reader who knows one table can read the other.
# ---------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    """Escape every TeX special in free text.

    The backslash goes FIRST, or it would double-escape the backslashes the later
    replacements introduce. ``^`` and ``~`` are here because they are not merely
    special but silently fatal in text mode: a source note reading "about 5 by 10^5
    mutants" halts the engine with "Missing $ inserted", which is how this was found.
    """
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
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
    return r"\href{" + url + r"}{" + breakable(shown) + r"}"


def breakable(s: str) -> str:
    r"""Escape a path or accession and give TeX somewhere to break it.

    Breaking only at slashes is not enough. A figshare URL carries the paper title as
    one underscore-joined segment, which is a single 120-character token to TeX and
    ran 112 mm past the text block. Underscores, hyphens and dots are break points
    too, and \\allowbreak permits rather than forces each one.
    """
    out = tex_escape(s)
    for tok in ("/", r"\_", "-", "."):
        out = out.replace(tok, tok + r"\allowbreak ")
    return out


def status_tex(status: str) -> str:
    """Marker for a row that is not a plain candidate."""
    return {
        "candidate": "",
        "blocked": r"\,\textsuperscript{\textbf{B}}",
        "aggregation": r"\,\textsuperscript{\textbf{A}}",
    }[status]


def seq_tex(basis: str) -> str:
    """Sequence-basis label with break points at + and -.

    "engineered-chassis+RBS" is one unbreakable token to TeX, so a narrow column
    cannot wrap it and the row runs off the text block.
    """
    return (
        tex_escape(basis).replace("+", r"+\allowbreak ").replace("-", r"-\allowbreak ")
    )


def org_tex(organism: str) -> str:
    """Host abbreviation, italic per house style for an organism name."""
    return r"\org{" + tex_escape(organism) + r"}"


def sci(n: int | None) -> str:
    """A count as an order-of-magnitude figure, the way the built table reports it."""
    if n is None:
        return "--"
    if n < 1000:
        return f"{n:,}"
    exp = int(math.floor(math.log10(n)))
    mant = n / 10**exp
    return f"${mant:.1f}\\times 10^{{{exp}}}$"


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "%% GENERATED FILE -- do not hand-edit.\n" + SOURCE_LINE + "\n" + body
    )
    print(f"Wrote {path.relative_to(REPO)}")


def is_pinned(c: Candidate) -> bool:
    """Whether the isoprenol pin protects this row from being displaced.

    Read by every lift, including the per-host floor, so that no mechanism can undo
    another's promotion. ``analog`` rows are deliberately NOT protected: standing in a
    different alcohol for isoprenol is an assumption about cross-tolerance, and the
    table does not promote a row on an assumption.
    """
    return c.isoprenol in ("direct", "tolerance", "precursor")


def _lift(
    rows: list[Candidate],
    boundary: int,
    wanted: Callable[[Candidate], bool],
    reason: str,
    moves: list[tuple[str, str, int, int]],
) -> None:
    """Move every row satisfying ``wanted`` inside the first ``boundary`` rows.

    One lift per iteration, strongest row below the boundary first, displacing the
    weakest row inside it that no pin protects. Every move is appended with its
    before and after rank, so the printed order can be reconciled against the pure
    measurement order row by row.
    """
    while True:
        below = [c for c in rows[boundary:] if wanted(c)]
        if not below:
            return
        promote = below[0]
        inside = [c for c in rows[:boundary] if not is_pinned(c)]
        if not inside:
            return  # the boundary is entirely pinned; nothing left to displace
        demote = max(inside, key=lambda c: c.sort_key)
        old = rows.index(promote) + 1
        rows.remove(promote)
        rows.insert(rows.index(demote), promote)
        rows.remove(demote)
        rows.insert(boundary, demote)
        moves.append((promote.name, reason, old, rows.index(promote) + 1))


def ranked() -> tuple[list[Candidate], list[tuple[str, str, int, int]]]:
    """Rank by measurements, then apply the isoprenol pin and the per-host floor.

    Returns the ordered rows and the moves as (name, reason, from_rank, to_rank), so
    the document can report every row that was lifted rather than hiding the
    reordering inside a score. Three mechanisms run in priority order: isoprenol
    first because it is the product the work targets, then its precursor pathway,
    then the per-host floor, which on the current rows no longer binds.
    """
    rows = sorted(CANDIDATES, key=lambda c: c.sort_key)
    if not rows:
        return rows, []

    moves: list[tuple[str, str, int, int]] = []

    # The isoprenol pin, and it is the whole reason this function is not just a sort.
    # Measurements rank a fifty-strain titer campaign below every barcoded screen,
    # which on the unaided order left two of the four P. putida isoprenol campaigns
    # in the reserve. Direct rows reach tranche 1; precursor rows reach the cut.
    _lift(
        rows,
        PIN_DIRECT,
        lambda c: c.isoprenol in ("direct", "tolerance"),
        "isoprenol",
        moves,
    )
    _lift(
        rows, PIN_PRECURSOR, lambda c: c.isoprenol == "precursor", "precursor", moves
    )

    # Tranche 1 by the global rule alone, then corrected to the floor. A host whose
    # screens are all smaller would otherwise not appear in the first tranche at
    # all, and the request was to start both hosts together.
    for organism, quota in QUOTA_1.items():
        while True:
            have = [c for c in rows[:TRANCHE_1] if c.organism == organism]
            if len(have) >= quota:
                break
            below = [c for c in rows[TRANCHE_1:] if c.organism == organism]
            if not below:
                break
            promote = below[0]
            # Displace the weakest row of the OTHER host that is over its own quota
            # and that no pin protects, so a promotion never costs a row another
            # mechanism is holding in place.
            others = [
                c
                for c in rows[:TRANCHE_1]
                if c.organism != organism
                and not is_pinned(c)
                and len([d for d in rows[:TRANCHE_1] if d.organism == c.organism])
                > QUOTA_1.get(c.organism, 0)
            ]
            if not others:
                break
            demote = max(others, key=lambda c: c.sort_key)
            old = rows.index(promote) + 1
            rows.remove(promote)
            rows.insert(rows.index(demote), promote)
            rows.remove(demote)
            rows.insert(TRANCHE_1, demote)
            moves.append((promote.name, "host floor", old, rows.index(promote) + 1))
    return rows, moves


def render_final(rows: list[Candidate]) -> str:
    """The ranked list, every statistic the ranking used beside the reason."""
    cols = (
        r"@{}r@{\hspace{3pt}} L{38mm} L{13mm} L{10mm} L{19mm} L{17mm} r@{\hspace{4pt}} "
        r"r@{\hspace{4pt}} L{19mm} L{74mm}@{}"
    )
    hdr = (
        r"\textbf{\#} & \textbf{Dataset (class; phenotype)} & \textbf{Host} & "
        r"\textbf{Tier} & \textbf{Genotypes} & \textbf{Env} & \textbf{Inst.} & "
        r"\textbf{Meas.} & \textbf{Sequence basis} & \textbf{Why} \\"
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
\caption[]{The bacterial candidates, ranked by \emph{Meas.} descending, which is
instances times phenotype dimensionality (Sec.~\ref{sec:rule}), with the isoprenol rows
then lifted out of that order (Table~\ref{tab:bpins}). \emph{Genotypes} and
\emph{Env} are the perturbation and condition axes. \emph{Inst.} is
genotype$\times$environment records, $\dagger$ where it is the product of the two axes
rather than a reported count and $\ddagger$ where it is an order-of-magnitude estimate.
\emph{Sequence basis} is the route to each strain's total genomic content; a row with no
route is excluded (Table~\ref{tab:bexcluded}). Superscript \textbf{B} marks a row whose
per-record values are not released, \textbf{A} a corpus that re-serves other papers and
is not net new until split by source. For the fifty recommended builds, rows 1--"""
        + str(TRANCHE_2)
        + r""", citations and data locations
are in Table~\ref{tab:bsources} and the yeast dataset each row mirrors is in
Table~\ref{tab:banalogs}; the reserve rows below the cut appear in neither.}
\label{tab:bfinal}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{10}{@{}l}{\footnotesize\emph{Table~\ref{tab:bfinal}, continued}}\\
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
        TRANCHE_1 + 1: (
            r"End of tranche 1. Rows 1--"
            + str(TRANCHE_1)
            + r" are the first builds, "
            + str(QUOTA_1["E. coli"])
            + r" per host."
        ),
        TRANCHE_2 + 1: (
            r"End of tranche 2. Rows "
            + str(TRANCHE_1 + 1)
            + r"--"
            + str(TRANCHE_2)
            + r" complete the fifty; rows below are the ranked reserve."
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
        dataset = (
            r"\textbf{"
            + tex_escape(c.name)
            + r"}"
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
                    org_tex(c.organism),
                    str(c.tier),
                    tex_escape(c.genotypes),
                    tex_escape(c.env),
                    sci(c.instances_n) + mark,
                    sci(c.measurements),
                    seq_tex(c.seq_basis),
                    tex_escape(c.why),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_sources(rows: list[Candidate]) -> str:
    hdr = (
        r"\textbf{\#} & \textbf{Dataset, citation and link} & "
        r"\textbf{Modality and readout} & \textbf{Data} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{86mm} L{74mm} L{78mm}@{}}
\caption[]{Sources for the fifty recommended builds, rows 1--"""
        + str(TRANCHE_2)
        + r""" of
Table~\ref{tab:bfinal}, in the same order; the ranked reserve below the cut is not listed.
\emph{Data} is where the per-record values live; an entry marked unconfirmed was not
fetched live and must be checked before a loader is written. Every link is clickable.}
\label{tab:bsources}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{4}{@{}l}{\footnotesize\emph{Table~\ref{tab:bsources}, continued}}\\
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
        modality = (
            tex_escape(c.modality)
            + r"\newline {\scriptsize "
            + (tex_escape(c.time_axis) if c.time_axis else "endpoint or steady state")
            + "}"
        )
        acc = breakable(c.accession)
        if not c.accession_confirmed:
            acc += r"\newline {\scriptsize (unconfirmed)}"
        lines.append(" & ".join([str(i), cite, modality, acc]) + r" \\")
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_analogs(rows: list[Candidate]) -> str:
    """The built yeast dataset each bacterial row mirrors, and what the schema needs.

    The column that answers whether a row fits the existing record types: a row
    with a named analog writes the same record against a different reference, and
    a row without one is naming a phenotype the substrate has never held.
    """
    hdr = (
        r"\textbf{\#} & \textbf{Bacterial row} & \textbf{Built yeast analog} & "
        r"\textbf{Why it is the analog} & \textbf{What the schema still needs} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{40mm} L{40mm} L{74mm} L{74mm}@{}}
\caption[]{What each of the fifty recommended builds, rows 1--"""
        + str(TRANCHE_2)
        + r""" of
Table~\ref{tab:bfinal}, maps onto in the supported set; the ranked reserve below the cut is
not listed. A named analog means
the loader writes a record type the schema already holds, against a different reference
genome, so the work is the retrieval and the provenance rather than a new phenotype
class. \emph{What the schema still needs} is per-row and excludes the two blockers every
row shares (Table~\ref{tab:bschema}); a dash means those two are the whole cost.}
\label{tab:banalogs}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{5}{@{}l}{\footnotesize\emph{Table~\ref{tab:banalogs}, continued}}\\
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
        analog = r"\textbf{" + tex_escape(c.analog.dataset) + r"}" if c.analog else "--"
        why = tex_escape(c.analog.why) if c.analog else "No built row measures this."
        lines.append(
            " & ".join(
                [
                    str(i),
                    tex_escape(c.name) + " " + org_tex(c.organism),
                    analog,
                    why,
                    tex_escape(c.schema_need) if c.schema_need else "--",
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_acquisition() -> str:
    """The leads that are not ranked rows, printed before the fifty.

    One block per target rather than a wide grid: every field is a sentence, and a
    6-column landscape table would wrap each of them to three lines.
    """
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{30mm} L{140mm}@{}}
\caption[]{Acquisition targets, which are NOT among the fifty and carry no verified
instance count. A ranked row in Table~\ref{tab:bfinal} has a confirmed accession and a
number read off a file; a row here has neither, so the two cannot be compared and this
table is deliberately kept out of every total. \emph{Claimed} reproduces the source's
own figures and is not a measurement. \emph{Verification} is what was done to check
them, and a search that found nothing is recorded as such.}
\label{tab:bacquisition}\\
\toprule
Field & Statement \\
\midrule
\endfirsthead
\toprule
Field & Statement \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for a in ACQUISITION:
        lines.append(
            r"\multicolumn{2}{@{}L{172mm}@{}}{\textbf{"
            + tex_escape(a.name)
            + r"} \quad \org{"
            + tex_escape(a.organism)
            + r"}} \\"
        )
        lines.append(r"\addlinespace[3pt]")
        for label, value in (
            ("Modality", a.modality),
            ("Claimed", a.claimed),
            ("Claim source", a.claim_source),
            ("Verification", a.verification),
            ("Why it matters", a.why),
            ("Blocking action", a.blocking_action),
        ):
            lines.append(label + " & " + tex_escape(value) + r" \\")
        lines.append(r"\addlinespace[6pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_schema() -> str:
    """The blockers every row shares, with how each was established."""
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{52mm} L{46mm} L{72mm}@{}}
\caption[]{What the schema needs before any bacterial row can be written, and how each
was established. The first two are the critical path: every row in
Table~\ref{tab:bfinal} waits on them, and both are additive, so they add classes and
move no served closure. \emph{Evidence} is the result of running the validator, not a
reading of it.}
\label{tab:bschema}\\
\toprule
Change & Where & Evidence \\
\midrule
\endfirsthead
\toprule
Change & Where & Evidence \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for s in SCHEMA_NEEDS:
        lines.append(
            " & ".join(
                [
                    r"\textbf{" + tex_escape(s.what) + r"}",
                    # \file is a \DeclareUrlCommand, so its argument is read with
                    # URL catcodes and must NOT be tex_escape'd: an escaped
                    # underscore renders as a literal backslash plus underscore.
                    r"\file{" + s.where + r"}",
                    tex_escape(s.evidence) + ". Blocks: " + tex_escape(s.blocks) + ".",
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_excluded() -> str:
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{58mm} L{28mm} L{83mm}@{}}
\caption[]{Considered and dropped. \emph{no-sequence} is the hard gate: without a route
to the strain's genomic content there is no genotype to map a phenotype from.
\emph{no-per-record-data} is a real experiment whose released form is a figure or a
summary rather than per-strain values.}
\label{tab:bexcluded}\\
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


def render_counts(rows: list[Candidate]) -> str:
    """Class by tranche and host by tranche, off the same ordering."""
    head = r"""\begin{table}[H]\centering
\small
\caption[]{Candidates by class and by host, split at the two tranche lines.
\emph{Genotypes}, \emph{Instances} and \emph{Meas.} sum the per-row axes over all
tranches; a row with no count contributes nothing, so every total is a lower bound.}
\label{tab:bcounts}
\begin{tabular}{@{}l r r r r r r@{}}
\toprule
 & Tr. 1 & Tr. 2 & Reserve & Genotypes & Instances & Meas. \\
\midrule
"""

    def block(key: str, values: list[str]) -> list[str]:
        out = []
        for v in values:
            members = [(i, c) for i, c in enumerate(rows, 1) if getattr(c, key) == v]
            if not members:
                continue
            t1 = [c for i, c in members if i <= TRANCHE_1]
            t2 = [c for i, c in members if TRANCHE_1 < i <= TRANCHE_2]
            rest = [c for i, c in members if i > TRANCHE_2]
            g = sum(c.genotypes_n or 0 for _i, c in members)
            n = sum(c.instances_n or 0 for _i, c in members)
            m = sum(c.measurements or 0 for _i, c in members)
            out.append(
                f"{tex_escape(v)} & {len(t1)} & {len(t2)} & {len(rest)} & "
                f"{g:,} & {sci(n)} & {sci(m)} \\\\"
            )
        return out

    klasses = sorted({c.klass for c in rows})
    lines = block("klass", klasses)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{@{}l}{\emph{The same rows, by host}}\\")
    lines += block("organism", ["E. coli", "P. putida"])
    g_all = sum(c.genotypes_n or 0 for c in rows)
    n_all = sum(c.instances_n or 0 for c in rows)
    m_all = sum(c.measurements or 0 for c in rows)
    lines.append(r"\midrule")
    lines.append(
        f"Total & {min(TRANCHE_1, len(rows))} & "
        f"{max(min(TRANCHE_2, len(rows)) - TRANCHE_1, 0)} & "
        f"{max(len(rows) - TRANCHE_2, 0)} & "
        f"{g_all:,} & {sci(n_all)} & {sci(m_all)} \\\\"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def render_summary(rows: list[Candidate]) -> str:
    """Summary statistics over the fifty, split at the tranche-1 line."""
    fifty = rows[:TRANCHE_2]
    t1 = fifty[:TRANCHE_1]
    t2 = fifty[TRANCHE_1:]

    def count(pred: Any) -> str:
        a = sum(1 for c in t1 if pred(c))
        b = sum(1 for c in t2 if pred(c))
        g = sum(c.genotypes_n or 0 for c in fifty if pred(c))
        n = sum(c.instances_n or 0 for c in fifty if pred(c))
        m = sum(c.measurements or 0 for c in fifty if pred(c))
        return f"{a} & {b} & {g:,} & {sci(n)} & {sci(m)} \\\\"

    def block(title: str, items: list[tuple[str, Any]]) -> list[str]:
        out = [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\"]
        for label, pred in items:
            if sum(1 for c in fifty if pred(c)) == 0:
                continue
            out.append(tex_escape(label) + " & " + count(pred))
        return out

    klasses = sorted({c.klass for c in fifty})
    bases = sorted({c.seq_basis for c in fifty})
    lines: list[str] = []
    lines += block(
        "By host",
        [(o, (lambda c, o=o: c.organism == o)) for o in ("E. coli", "P. putida")],
    )
    lines += block(
        "By tier", [(f"tier {t}", (lambda c, t=t: c.tier == t)) for t in (1, 2, 3, 4)]
    )
    lines += block("By class", [(k, (lambda c, k=k: c.klass == k)) for k in klasses])
    lines += block(
        "By sequence basis", [(b, (lambda c, b=b: c.seq_basis == b)) for b in bases]
    )
    lines += block(
        "Other attributes",
        [
            ("carries a time axis", lambda c: bool(c.time_axis)),
            ("mirrors a built yeast dataset", lambda c: c.analog is not None),
            ("needs a phenotype class the substrate lacks", lambda c: c.analog is None),
            (
                "has a join to a built dataset",
                lambda c: any(s.partner_status == "supported" for s in c.synergy),
            ),
            ("figures sourced this pass", lambda c: c.confidence == "sourced"),
            ("figures from recall, to confirm", lambda c: c.confidence == "recall"),
            ("data location confirmed live", lambda c: c.accession_confirmed),
            ("per-record values not released", lambda c: c.status == "blocked"),
            ("a corpus, not net new until split", lambda c: c.status == "aggregation"),
            ("instances a reported count", lambda c: c.instances_basis == "reported"),
            (
                "instances a product of the axes",
                lambda c: c.instances_basis == "product",
            ),
            ("instances an estimate", lambda c: c.instances_basis == "estimate"),
            (
                "dimensionality reported, not estimated",
                lambda c: c.dim_basis == "reported",
            ),
        ],
    )
    lines.append(r"\midrule")
    lines.append("Total & " + count(lambda c: True))
    n_joins = sum(len(c.synergy) for c in fifty)
    n_built = sum(
        1 for c in fifty for s in c.synergy if s.partner_status == "supported"
    )
    head = (
        r"""\begin{table}[H]\centering
\small
\caption[]{Summary of the fifty, split at the tranche-1 line. \emph{Tr. 1} and
\emph{Tr. 2} count rows; \emph{Genotypes}, \emph{Instances} and \emph{Meas.} sum the row
axes over all fifty and are lower bounds, since a row with no count contributes nothing.
The fifty name """
        + str(n_joins)
        + r""" joins, """
        + str(n_built)
        + r""" of them to a dataset already built.}
\label{tab:bsummary}
\begin{tabular}{@{}l r r r r r@{}}
\toprule
 & Tr. 1 & Tr. 2 & Genotypes & Instances & Meas. \\
"""
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def render_pins(
    rows: list[Candidate], moves: list[tuple[str, str, int, int]]
) -> str:
    """Every row on the isoprenol axis, and every lift the pin performed.

    Printed in full rather than counted, because the pin overrides the ordering rule
    and a reader has to be able to undo it: the measurement rank is what the rule
    alone gives, the final rank is what the table prints.
    """
    axis = [c for c in rows if c.isoprenol != "none"]
    pure = sorted(CANDIDATES, key=lambda c: c.sort_key)
    pure_rank = {c.name: i for i, c in enumerate(pure, start=1)}
    final_rank = {c.name: i for i, c in enumerate(rows, start=1)}
    reason = {name: why for name, why, _, _ in moves}

    body = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{longtable}{@{}L{40mm} L{13mm} L{17mm} L{32mm} r r L{20mm}@{}}",
        r"\caption[]{The isoprenol axis, divided by the role the molecule plays. "
        r"\emph{Rule} is the rank the measurement ordering gives on its own and "
        r"\emph{Final} the rank this document prints, so the pin can be undone row by "
        r"row. \emph{Direct} measures isoprenol or isopentenol as a product, the same "
        r"molecule under two names; \emph{tolerance} measures growth against it as a "
        r"stressor; both are lifted into tranche 1. \emph{Precursor} measures another "
        r"isoprenoid and is lifted above the cut. \emph{Analog} substitutes a different "
        r"alcohol or inhibitor and is reported but not lifted, because the substitution "
        r"assumes cross-tolerance.}",
        r"\label{tab:bpins}\\",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Host} & \textbf{Axis} & \textbf{Product} & "
        r"\textbf{Rule} & \textbf{Final} & \textbf{Lifted by} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{@{}l}{\footnotesize\emph{Table~\ref{tab:bpins}, continued}}\\",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Host} & \textbf{Axis} & \textbf{Product} & "
        r"\textbf{Rule} & \textbf{Final} & \textbf{Lifted by} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    order = {"direct": 0, "tolerance": 1, "precursor": 2, "analog": 3, "none": 4}
    for c in sorted(axis, key=lambda c: (order[c.isoprenol], final_rank[c.name])):
        body.append(
            f"{tex_escape(c.name)} & \\emph{{{tex_escape(c.organism)}}} & {c.isoprenol} & "
            f"{tex_escape(c.product) if c.product else '--'} & "
            f"{pure_rank[c.name]} & {final_rank[c.name]} & "
            f"{tex_escape(reason.get(c.name, '--'))} \\\\"
        )
    body += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(body) + "\n"


def main() -> None:
    rows, quota_moves = ranked()
    if len(rows) < TARGET_COUNT:
        raise SystemExit(
            f"only {len(rows)} candidates; the target is {TARGET_COUNT} rows"
        )

    write(TEX_DIR / "acquisition.tex", render_acquisition())
    write(TEX_DIR / "final.tex", render_final(rows))
    write(TEX_DIR / "sources.tex", render_sources(rows[:TRANCHE_2]))
    write(TEX_DIR / "analogs.tex", render_analogs(rows[:TRANCHE_2]))
    write(TEX_DIR / "schema.tex", render_schema())
    write(TEX_DIR / "counts.tex", render_counts(rows))
    write(TEX_DIR / "summary.tex", render_summary(rows))
    write(TEX_DIR / "excluded.tex", render_excluded())
    write(TEX_DIR / "pins.tex", render_pins(rows, quota_moves))

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(
            {
                "yeast_built": YEAST_BUILT,
                "bacteria_built": BACTERIA_BUILT,
                "target_count": TARGET_COUNT,
                "tranche_1": TRANCHE_1,
                "tranche_2": TRANCHE_2,
                "quota_1": QUOTA_1,
                "n_candidates": len(rows),
                "quota_moves": quota_moves,
                "acquisition": [a.model_dump() for a in ACQUISITION],
                "schema_needs": [s.model_dump() for s in SCHEMA_NEEDS],
                "candidates": [c.model_dump() for c in rows],
                "excluded": [e.model_dump() for e in EXCLUDED],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {JSON_OUT.relative_to(REPO)}")

    for name, why, old, new in quota_moves:
        print(f"{why} lifted {name!r} from rank {old} to {new}")
    n_ec = sum(c.organism == "E. coli" for c in rows[:TRANCHE_1])
    n_pp = sum(c.organism == "P. putida" for c in rows[:TRANCHE_1])
    print(
        f"{len(rows)} candidates; tranche 1 = {TRANCHE_1} "
        f"({n_ec} E. coli, {n_pp} P. putida), tranche 2 ends at {TRANCHE_2}"
    )
    n_analog = sum(c.analog is not None for c in rows[:TRANCHE_2])
    n_sourced = sum(c.confidence == "sourced" for c in rows[:TRANCHE_2])
    print(
        f"of the fifty: {n_analog} mirror a built yeast dataset, "
        f"{n_sourced} carry figures sourced this pass"
    )
    axis = {
        k: [c for c in rows if c.isoprenol == k]
        for k in ("direct", "tolerance", "precursor", "analog")
    }
    inside = {k: sum(rows.index(c) < TRANCHE_2 for c in v) for k, v in axis.items()}
    print(
        "isoprenol axis: "
        + ", ".join(
            f"{len(v)} {k} ({inside[k]} inside the fifty)" for k, v in axis.items()
        )
    )

    # The figures sections/rule.tex and the abstract quote. Printed here so the
    # prose traces to this script: rule rank is the pure measurement order, final
    # rank is the printed order after the pins.
    rule = sorted(rows, key=lambda c: c.sort_key)
    for label, classes in (("engineering", ENGINEERING_CLASSES), ("screens", SCREEN_CLASSES)):
        group = [c for c in rows if c.klass in classes]
        print(
            f"{label}: {len(group)} rows, "
            f"median measurements {statistics.median(c.measurements or 0 for c in group):,.0f}, "
            f"median rule rank {statistics.median(rule.index(c) + 1 for c in group)}, "
            f"inside the fifty {sum(rule.index(c) < TRANCHE_2 for c in group)} on the rule alone "
            f"and {sum(rows.index(c) < TRANCHE_2 for c in group)} after the pins, "
            f"{sum(rows.index(c) >= TRANCHE_2 for c in group)} of the "
            f"{len(rows) - TRANCHE_2} reserve rows"
        )
    for label, cut in (("fifty", TRANCHE_2), ("tranche 1", TRANCHE_1)):
        for order_name, order in (("rule alone", rule), ("after the pins", rows)):
            n_ec = sum(c.organism == "E. coli" for c in order[:cut])
            print(f"{label}, {order_name}: {n_ec} E. coli, {cut - n_ec} P. putida")
    for c in rows:
        if c.name.startswith(("Rachwalski 2024", "Silvis 2021")):
            print(
                f"{c.name}: rule rank {rule.index(c) + 1}, "
                f"final rank {rows.index(c) + 1}, measurements {c.measurements:,}"
            )


if __name__ == "__main__":
    main()
