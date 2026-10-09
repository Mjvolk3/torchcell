# torchcell/datasets/ecoli/teteneva2024
# [[torchcell.datasets.ecoli.teteneva2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/teteneva2024
# Test file: tests/torchcell/datasets/ecoli/test_teteneva2024.py
r"""Teteneva 2024 lake water: the W3110 host gate is open, the loader is blocked.

Teteneva, Sanches-Medeiros and Sourjik 2024 (The ISME Journal 18(1):wrae096,
doi:10.1093/ismejo/wrae096, PMC11188689) is row 41 of the fifty bacterial rows of
``notes-tex/database/database-expansion-bacteria/tables/final.tex``: RB-TnSeq in
*E. coli* K-12 **W3110** in lake water. The row was counted off the deposited workbook
by ``experiments/database/scripts/build_bacteria_candidate_datasets_table.py`` (row
``Teteneva 2024 lake water``): 11,027 gene by water-sample rows and 66,162 non-empty
fitness values in Supplementary Table S4. Those two counts are that script's
measurement, not this module's.

TWO GATES, and this module closes the first and refuses the second.

**Gate 1, the host: CLOSED.** W3110 is a K-12 derivative distinct from the BW25113 and
MG1655 assembly sets the genomes tier carried, so the row needs its own set. It now has
one, deposited on 2026-10-09 by ``scripts/provision_bacterial_genomes.py --set
ecoli_K12_W3110_ASM1024v1``: ``ECOLI_K12_W3110`` in the genomes registry,
GCA_000010245.1 / GCF_000010245.2 (ASM1024v1), replicon AP009048.1 / NC_007779.1,
4,646,332 bp, ten members, every md5 matched against its NCBI directory's own
``md5checksums.txt`` and every sha256 re-hashed by ``deposit_assembly_set``
(:data:`W3110_TIER_DEPOSIT`).

**Gate 1a, the annotation route: OPEN, and NOT decided here.** The deposited set is not
yet readable by :class:`~torchcell.sequence.genome.bacterial.BacterialGenome`, which
reads GenBank first. The 2006 DDBJ/NIG annotation of AP009048.1 carries 4,444 gene
features and NOT ONE ``locus_tag``: it keys genes by symbol and carries the
``ECK:JW:b`` crosswalk as a ``/note`` on 3,730 CDS features. ``read_genbank`` therefore
refuses it by name, measured in :func:`annotation_routes`. The RefSeq member parses,
4,531 loci with ``Y75_RS`` tags and ``Y75_p`` ``old_locus_tag`` values on 4,254 of them,
so a RefSeq-primary route exists. Which route this strain is read through decides the
``BacterialGeneNamespace`` member, and that choice belongs with the identifiers
Supplementary Table S4 actually reports, which cannot be read (gate 2). So no genome
class, no ``BacterialReferenceStrain`` member and no schema vocabulary names this set
yet: nothing served changes.

**Gate 2, the paper: BLOCKED on a curation decision, nothing was fetched.** The paper
is in neither mirror: ``$DATA_ROOT/torchcell-library/`` holds no Teteneva key and
``$DATA_ROOT/torchcell-raw/`` holds none either (checked 2026-10-09). The library
mirror is populated FROM Zotero by ``scripts/lit_sync.py``, and filing a paper in
Zotero is a curation decision that needs an explicit instruction each time, which the
2026.10.08 entry of ``notes/experiments.database.expansion-bacteria.md`` records as
still open for this row. So every schema value a loader needs is a typed absence here
(:data:`LOADER_GAPS`), not a guess: the replicate count, the fitness statistic and its
scale, time zero, the medium (an oligotrophic natural water, which would be a new
``Media`` entry that must be defined from the paper), and the identifier namespace of
Table S4. ``SourcedValue`` cannot even be constructed for this row, because it requires
a ``citation_key`` and the ``sha256`` of a mirrored artifact.

What makes the refusal cheap to reverse: the whole deposit is in the PMC open-access
bucket, listed on 2026-10-09 (:data:`PMC_OA_DEPOSIT`), including
``supplementary_table_s4_wrae096.xlsx``, the workbook the row was counted off. Once the
paper is filed in Zotero, ``lit_sync`` plus ``lit_capture_si`` mirror it through the
existing ``pmc_cloud`` retriever with no by-hand step. :data:`EDITS_NEEDED` is the rest
of the work, in order.
"""

from __future__ import annotations

import gzip
import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from torchcell.sequence.genome.bacterial import read_genbank
from torchcell.sequence.genome.registry import ECOLI_K12_W3110, resolve
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

# --------------------------------------------------------------------------- #
# Paper identity
# --------------------------------------------------------------------------- #
PAPER_DOI: Final = "10.1093/ismejo/wrae096"
PAPER_PMCID: Final = "PMC11188689"
PAPER_TITLE: Final = (
    "Genome-wide screen of genetic determinants that govern Escherichia coli growth "
    "and persistence in lake water"
)
#: Title, authors, journal and DOI as NCBI's ``esummary`` returned them on 2026-10-09;
#: the DOI resolved to this one PMC id through ``esearch`` on the same day.
PAPER_AUTHORS: Final = ("Teteneva N", "Sanches-Medeiros A", "Sourjik V")
PAPER_JOURNAL: Final = "The ISME journal 2024 Jan 8, volume 18, issue 1, wrae096"
#: The citation key Better BibTeX would be EXPECTED to emit, by the pattern of the
#: mirrored keys (first author, three title words with the hyphen collapsed, year). A
#: PREDICTION, not a measurement: the paper is not in the library, so nothing has
#: exported a key for it. ``torchcell.literature.citation_keys.generate_citation_key``
#: emits a longer five-word variant that does not reproduce the mirrored keys either.
EXPECTED_CITATION_KEY: Final = "tetenevaGenomewideScreenGenetic2024"

#: The row as the schedule generator recorded it, with the file that holds it. Both
#: counts are that script's measurement off the workbook with ``openpyxl``.
SCHEDULE_ROW: Final = 41
SCHEDULE_SOURCE: Final = (
    "experiments/database/scripts/build_bacteria_candidate_datasets_table.py"
)
SCHEDULE_FITNESS_VALUES: Final = 66_162
SCHEDULE_GENE_BY_SAMPLE_ROWS: Final = 11_027


class PmcDepositFile(BaseModel):
    """One object of this article's PMC open-access prefix, as the bucket listed it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(description="bucket key, passed straight to pmc_cloud_object")
    role: str = Field(description="paper | si_data | si_pdf | figure | metadata")


_OA_PREFIX: Final = f"{PAPER_PMCID}.1"

#: Every object under ``pmc-oa-opendata/PMC11188689.1/``, listed on 2026-10-09 with
#: ``GET /?list-type=2&prefix=PMC11188689``. NOTHING here was downloaded: the listing is
#: evidence that the retrieval is scriptable the moment the paper is curated, and
#: ``supplementary_table_s4_wrae096.xlsx`` is the workbook the row was counted off.
PMC_OA_DEPOSIT: Final = (
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.pdf", role="paper"),
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.xml", role="paper"),
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.txt", role="paper"),
    PmcDepositFile(key=f"{_OA_PREFIX}/{_OA_PREFIX}.json", role="metadata"),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s1_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s2_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s3_wrae096.xlsx", role="si_data"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_table_s4_wrae096.xlsx", role="si_data"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s1_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s2_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s3_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(
        key=f"{_OA_PREFIX}/supplementary_figure_s4_wrae096.pdf", role="si_pdf"
    ),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f1.jpg", role="figure"),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f2.jpg", role="figure"),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f3.jpg", role="figure"),
    PmcDepositFile(key=f"{_OA_PREFIX}/wrae096f4.jpg", role="figure"),
)
#: The one object a loader would read its values out of.
TABLE_S4_OA_KEY: Final = f"{_OA_PREFIX}/supplementary_table_s4_wrae096.xlsx"


# --------------------------------------------------------------------------- #
# Gate 1: the deposited assembly set
# --------------------------------------------------------------------------- #
class TierMember(BaseModel):
    """One member of the W3110 set, as fetched, md5-matched and deposited."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    role: str = Field(description="genomes-tier role: annotation | sequence | index")
    bytes: int
    md5: str = Field(description="NCBI's md5checksums.txt value, matched on retrieval")
    sha256: str


class DepositedAssemblySet(BaseModel):
    """The W3110 assembly set as it was deposited into the genomes tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    organism: str = Field(description="'# Organism name' of the assembly report")
    strain: str
    assembly_name: str
    genbank_accession: str
    refseq_accession: str
    genbank_replicon: str
    refseq_replicon: str
    replicon_length_bp: int
    genbank_url: str
    refseq_url: str
    md5_checksum_files: dict[str, str] = Field(
        description="URL of each directory's md5checksums.txt -> sha256 of the copy used"
    )
    members: tuple[TierMember, ...]
    deposited_at: str


_NCBI: Final = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
_GCA_DIR: Final = f"{_NCBI}/GCA/000/010/245/GCA_000010245.1_ASM1024v1"
_GCF_DIR: Final = f"{_NCBI}/GCF/000/010/245/GCF_000010245.2_ASM1024v1"

#: The GCA ``_assembly_report.txt`` of this set still names the 2006 RefSeq release,
#: ``GCF_000010245.1`` with RefSeq-Accn ``AC_000091.1``, while the current RefSeq
#: release is ``GCF_000010245.2`` with RefSeq-Accn ``NC_007779.1``. The accession PAIR
#: is therefore only readable off the GCF report, which is why that file is a member of
#: this set and of no other.
STALE_GCA_REFSEQ_ACCESSION: Final = "GCF_000010245.1"
STALE_GCA_REFSEQ_REPLICON: Final = "AC_000091.1"

W3110_TIER_DEPOSIT: Final = DepositedAssemblySet(
    assembly_set=ECOLI_K12_W3110,
    organism="Escherichia coli str. K-12 substr. W3110 (E. coli)",
    strain="K-12 W3110",
    assembly_name="ASM1024v1",
    genbank_accession="GCA_000010245.1",
    refseq_accession="GCF_000010245.2",
    genbank_replicon="AP009048.1",
    refseq_replicon="NC_007779.1",
    replicon_length_bp=4_646_332,
    genbank_url=f"{_GCA_DIR}/",
    refseq_url=f"{_GCF_DIR}/",
    md5_checksum_files={
        f"{_GCA_DIR}/md5checksums.txt": (
            "6674b4dec2356b88516648a3947cc8a6c7085b1c20b1af2982edd884a9e70321"
        ),
        f"{_GCF_DIR}/md5checksums.txt": (
            "71a7b12e78101e72ddbca9c018bcc933c81c0a8b14646ac423c588948b71dbee"
        ),
    },
    members=(
        TierMember(
            path="GCA_000010245.1_ASM1024v1_genomic.gbff.gz",
            role="annotation",
            bytes=3286152,
            md5="7c5e9f6b10f282bc9dc82ddcdc0bcc8d",
            sha256="7e22368bc1783fe3b56b07a7196dd0abe3104ef88cd795ad0970ab546418a157",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_genomic.fna.gz",
            role="sequence",
            bytes=1381336,
            md5="7cc36bddd25647f60941a20c76571bac",
            sha256="57709b8e4bf6a66951db1779bb61a75fbca2db416aa3a1ba245365fdebff93a0",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_genomic.gff.gz",
            role="annotation",
            bytes=287541,
            md5="f73b4df02eb07ccd0ab437d2f728d221",
            sha256="8aa0069b3715239d16103653b7becd27c5e59676f711d20854f43882c033bb6e",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_protein.faa.gz",
            role="sequence",
            bytes=892689,
            md5="cc6b6bdc016bb04306214642419a8306",
            sha256="dfb59618a7c2c993859ad6b05c9c613a9ae1f7e2605cc7298ac286652ad33ff5",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_feature_table.txt.gz",
            role="index",
            bytes=167217,
            md5="6f3ab693a5a99c1a76085a347a2b58ed",
            sha256="21b04994e343fa3bf892492f70180bc215110f4016ad463a47acce6a24112ec8",
        ),
        TierMember(
            path="GCA_000010245.1_ASM1024v1_assembly_report.txt",
            role="index",
            bytes=1283,
            md5="aeb8a80d69d84f7f69f752d2a6996d12",
            sha256="d3f6d0fc9bd9e8ef9479c9554239e8618a7d0bc464c362304b25385e646ce267",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_genomic.gbff.gz",
            role="annotation",
            bytes=3452629,
            md5="526fb39d6930215f7c627b977536bfbd",
            sha256="3edf6662a5aac6a28f3550622f66328e88ae775da117cc0004eedb1f066b991e",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_genomic.gff.gz",
            role="annotation",
            bytes=435483,
            md5="fa616f6df51b2db6d00514b8762306b6",
            sha256="1501e3787567e224ccd03e81b3467f00e1bad7db719b1e41db5200d46edce4f3",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_gene_ontology.gaf.gz",
            role="annotation",
            bytes=156751,
            md5="508c8d4e53593b0986d5dc4a38cfd0a2",
            sha256="cd8de975cf31e31145c1254f6bf4f393ec275f56e1dba706f850bf1714b611a8",
        ),
        TierMember(
            path="GCF_000010245.2_ASM1024v1_assembly_report.txt",
            role="index",
            bytes=1200,
            md5="58588ab7b196dd61a04cdeb76d7836b6",
            sha256="98015272fe4acd3171174109f666137bd310885f580100e11c2ed1ae132b8c1c",
        ),
    ),
    deposited_at="2026-10-09",
)


# --------------------------------------------------------------------------- #
# Gate 1a: which member can be read, measured
# --------------------------------------------------------------------------- #
class AnnotationRoute(BaseModel):
    """What one annotation member of the set offers a GenBank-first ingest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member: str
    gene_features: int = Field(description="'gene' features in the flat file")
    gene_features_with_locus_tag: int
    read_genbank_error: str | None = Field(
        description="read_genbank's message, or None when the member parses"
    )
    loci: int | None = Field(
        description="loci read, or None when the member is refused"
    )
    replicon: str | None


#: The ``ECK:JW:b`` crosswalk the GenBank CDS features carry instead of a locus tag.
ECK_JW_B_NOTE = re.compile(r'/note="(ECK\d+):(JW\d+):(b\d+)"')

#: Measured on the deposited bytes on 2026-10-09 by :func:`annotation_routes`. The
#: GenBank member is refused by name; the RefSeq member parses. The GenBank CDS notes
#: carry 3,730 ``ECK:JW:b`` triples, 3,730 distinct JW numbers and 3,726 distinct
#: b-numbers, so the file crosswalks W3110 to MG1655 without naming a W3110 locus tag.
GENBANK_GENE_FEATURES: Final = 4_444
GENBANK_GENE_FEATURES_WITH_LOCUS_TAG: Final = 0
GENBANK_READ_ERROR: Final = (
    "GCA_000010245.1_ASM1024v1_genomic.gbff.gz: a gene feature at [189:255](+) has no "
    "locus_tag"
)
ECK_JW_B_TRIPLES: Final = 3_730
ECK_JW_B_DISTINCT_JW: Final = 3_730
ECK_JW_B_DISTINCT_BNUMBER: Final = 3_726
REFSEQ_GENE_FEATURES: Final = 4_531
REFSEQ_LOCI: Final = 4_531
REFSEQ_LOCUS_TAG_PATTERN: Final = r"Y75_RS\d{5}"
REFSEQ_OLD_LOCUS_TAG_PATTERN: Final = r"Y75_p\d{4}"
REFSEQ_OLD_LOCUS_TAGS: Final = 4_254
REFSEQ_ONTOLOGY_TERM_ROWS: Final = 2_298

GENBANK_MEMBER: Final = "GCA_000010245.1_ASM1024v1_genomic.gbff.gz"
REFSEQ_MEMBER: Final = "GCF_000010245.2_ASM1024v1_genomic.gbff.gz"


def _gene_feature_counts(path: str) -> tuple[int, int]:
    """``(gene features, gene features carrying a locus_tag)`` of a flat file."""
    genes = tagged = 0
    in_features = in_gene = False
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN"):
                in_features = False
            if not in_features:
                continue
            if re.match(r"^     gene +", line):
                genes += 1
                in_gene = True
                continue
            if re.match(r"^     \S", line):
                in_gene = False
            if in_gene and "/locus_tag=" in line:
                tagged += 1
    return genes, tagged


def annotation_routes(data_root: str | None = None) -> tuple[AnnotationRoute, ...]:
    """Measure both annotation members of the deposited set, GenBank first.

    Returns one :class:`AnnotationRoute` per member in deposit order. The GenBank
    member's ``read_genbank_error`` is the parser's own message, so the finding that
    the tier's GenBank-first ingest does not reach W3110 is re-derived rather than
    asserted.
    """
    routes = []
    for member in (GENBANK_MEMBER, REFSEQ_MEMBER):
        path = resolve(ECOLI_K12_W3110, member, data_root=data_root)
        genes, tagged = _gene_feature_counts(path)
        try:
            annotation, _ = read_genbank(path, member)
        except ValueError as error:
            routes.append(
                AnnotationRoute(
                    member=member,
                    gene_features=genes,
                    gene_features_with_locus_tag=tagged,
                    read_genbank_error=str(error),
                    loci=None,
                    replicon=None,
                )
            )
            continue
        routes.append(
            AnnotationRoute(
                member=member,
                gene_features=genes,
                gene_features_with_locus_tag=tagged,
                read_genbank_error=None,
                loci=len(annotation.loci),
                replicon=annotation.replicons[0].accession,
            )
        )
    return tuple(routes)


def eck_jw_b_notes(data_root: str | None = None) -> tuple[tuple[str, str, str], ...]:
    """Every ``(ECK, JW, b-number)`` triple the GenBank member's CDS notes carry.

    This is the only W3110-to-MG1655 crosswalk the GenBank deposit publishes, and it is
    a ``/note``, not an identifier: no feature of the file carries a W3110 locus tag.
    """
    path = resolve(ECOLI_K12_W3110, GENBANK_MEMBER, data_root=data_root)
    with gzip.open(path, "rt") as handle:
        return tuple(ECK_JW_B_NOTE.findall(handle.read()))


# --------------------------------------------------------------------------- #
# Gate 2: the refusal
# --------------------------------------------------------------------------- #
#: Neither mirror holds this paper, checked on 2026-10-09 by listing both roots.
LIBRARY_MIRROR_KEYS_MATCHING_TETENEVA: Final = 0
RAW_MIRROR_KEYS_MATCHING_TETENEVA: Final = 0

_RESOLVE_WITH_PAPER = Provenance(
    source_uri=f"https://doi.org/{PAPER_DOI}",
    page="Materials and methods",
    method="not retrieved: the paper is in neither mirror and Zotero curation is the "
    "owner's decision",
)
_RESOLVE_WITH_TABLE_S4 = Provenance(
    source_uri=f"https://pmc-oa-opendata.s3.amazonaws.com/{TABLE_S4_OA_KEY}",
    page="Supplementary Table S4",
    method="not retrieved: see above",
)


def _gap(field: str, note: str, *, in_table: bool = False) -> ProvenanceGap:
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.deferred_pending_source_review,
        resolve_with=_RESOLVE_WITH_TABLE_S4 if in_table else _RESOLVE_WITH_PAPER,
        note=note,
    )


#: Every schema value a loader for this row needs and that NOTHING in our own
#: documentation can source today. Each is ``deferred_pending_source_review``, the one
#: recoverable reason: the artifact exists and is open access, it is just not mirrored.
LOADER_GAPS: Final = (
    _gap(
        "n_samples",
        "the replicate count behind one fitness value; the schedule row records "
        "'fitness is averaged over three technical replicates per sample' from the "
        "same earlier pass that counted the workbook, which is OUR note and not a "
        "quote of the paper, so it is not a source",
    ),
    _gap(
        "phenotype_statistic",
        "which statistic Table S4's columns hold and on what scale (the row says "
        "'gene-level RB-TnSeq fitness (Wetmore normalization)', a log2 ratio against "
        "a time-zero baseline, which decides EnvironmentResponsePhenotype against "
        "FitnessPhenotype; FitnessPhenotype clamps non-positive values, so a signed "
        "log2 ratio cannot be stored as fitness)",
        in_table=True,
    ),
    _gap(
        "time_zero",
        "what the fitness ratio is taken against: the inoculum, a day-0 sample, or a "
        "per-sample reference",
    ),
    _gap(
        "media",
        "the medium is an oligotrophic natural lake water, filtered and non-filtered, "
        "which is a NEW Media entry and must be defined from the paper's own "
        "description of the water (source, filtration, nutrient content); a "
        "placeholder medium would be a fabricated environment",
    ),
    _gap(
        "gene_namespace",
        "which identifiers Table S4 names its 3,691 genes by. This also decides gate "
        "1a: a b-number or JW number is reachable through the GenBank ECK:JW:b notes, "
        "a Y75_RS tag through the RefSeq member, and a gene symbol through neither "
        "uniquely",
        in_table=True,
    ),
    _gap(
        "genotype",
        "the library's construction and background (the row records 430,849 unique "
        "insertions in 3,833 genes of an RpoS+ W3110, again from our own note), and "
        "whether a stored genotype is the insertion mutant or the gene",
    ),
)

#: The work left, in order. Everything after the first item waits on the first.
EDITS_NEEDED: Final = (
    "file the paper in Zotero (owner's curation decision, group library "
    "database/Escherichia-coli, matched by DOI 10.1093/ismejo/wrae096), then "
    "scripts/lit_sync.py and scripts/lit_capture_si.py mirror it; every object is in "
    "the PMC open-access bucket under PMC11188689.1, so the pmc_cloud retriever needs "
    "no by-hand step",
    "deposit Supplementary Table S4 in the raw mirror "
    "($DATA_ROOT/torchcell-raw/<key>/) with its retrieval record and sha256, and "
    "re-count its 11,027 rows and 66,162 non-empty values off the pinned bytes",
    "decide the annotation route for ecoli_K12_W3110_ASM1024v1 against the "
    "identifiers Table S4 reports: GenBank-first does not apply (the deposit carries "
    "no locus_tag), so either the set is read RefSeq-primary with Y75_RS tags or the "
    "ingest gains a symbol-plus-ECK:JW:b route",
    "torchcell/sequence/genome/ecoli/: a W3110 genome class beside EcoliK12Genome, "
    "reading whichever member the route decides",
    "torchcell/datamodels/schema.py: 'W3110' in BacterialReferenceStrain; "
    "BACTERIAL_ASSEMBLY_SETS['W3110'] = 'ecoli_K12_W3110_ASM1024v1'; the set id in "
    "BacterialAssemblySet; ASSEMBLY_SET_ACCESSIONS[set] = ('GCA_000010245.1', "
    "'GCF_000010245.2'), the pair read off the GCF assembly report because the GCA one "
    "is stale; the namespace in BacterialGeneNamespace with its pattern in "
    "BACTERIAL_LOCUS_TAG_PATTERNS (disjoint from the four existing patterns)",
    "torchcell/datasets/bacteria_common.py: W3110 in HOST_STRAINS['ecoli'], "
    "STRAIN_GENE_NAMESPACES and BACTERIAL_GENOME_CLASSES",
    "torchcell/verification/runners.py: a W3110 gene universe beside the K-12 one",
    "the loader itself: signed fitness as EnvironmentResponsePhenotype log2 ratio per "
    "the wave-1 rule, with the adapter, its conf, the map entry and the four adapter "
    "pin places, and a measured record count against the 66,162 estimate with every "
    "drop explained",
)
