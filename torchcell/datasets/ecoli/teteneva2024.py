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

**Gate 2, the paper: the SI is deposited, the paper text is not yet mirrored.** The
owner filed the paper in Zotero on 2026-10-12 (group item ``PUKZFNFZ`` in
``database/Escherichia-coli``, #910), and ``scripts/lit_sync.py`` mirrors it into
``$DATA_ROOT/torchcell-library/`` on its next nightly run; until then the library
mirror holds no Teteneva key. The same day the owner deposited five supplementary files
by hand into ``$DATA_ROOT/torchcell-raw/tetenevaGenomewideScreenGenetic2024/si/``
(:data:`SI_DEPOSIT`, ``manual_browser``), and every one of them is byte-identical to the
PMC open-access bucket object of the same name (fetched and compared on 2026-10-12), so
a rebuild needs no person. Table S4 is measured off the pinned bytes
(:func:`table_s4_inventory`): 11,027 gene-by-water-sample rows, 66,162 non-empty
fitness values (six per row: filtered and non-filtered water at days 2, 4 and 8), three
water samples, and 3,691 distinct genes named ONLY by a gene symbol (column
``Gene name``). That answers the identifier half of gate 1a without deciding it: a
symbol route is needed whichever annotation member is read, and
:data:`SYMBOL_ROUTES` counts how the symbols land on each. Every schema value a loader
needs is still a typed absence (:data:`LOADER_GAPS`), because the values that decide
them (the statistic, time zero, the water) are in the paper text, which is not yet in
the library mirror, and the annotation route is the owner's decision (#910).
:data:`EDITS_NEEDED` is the rest of the work, in order.
"""

from __future__ import annotations

import gzip
import io
import re
from collections import Counter
from pathlib import Path
from typing import Final

from Bio import SeqIO
from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import ROLE_SI_DATA, ROLE_SI_PDF
from torchcell.literature.manual_deposit import (
    DepositedFile,
    ManualDeposit,
    ScriptedEquivalent,
)
from torchcell.literature.retrieve import pmc_cloud_url
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


def symbol_index(member: str, data_root: str | None = None) -> dict[str, set[str]]:
    """``/gene`` symbol -> the gene features carrying it, for one annotation member.

    A feature is keyed by its ``locus_tag`` where it has one (RefSeq) and by its
    location otherwise (the GenBank member has none), so two features sharing a symbol
    count as two.
    """
    path = resolve(ECOLI_K12_W3110, member, data_root=data_root)
    index: dict[str, set[str]] = {}
    with gzip.open(path, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            for feature in record.features:
                if feature.type != "gene":
                    continue
                key = feature.qualifiers.get("locus_tag", [str(feature.location)])[0]
                for symbol in feature.qualifiers.get("gene", []):
                    index.setdefault(symbol, set()).add(key)
    return index


def symbol_routes(
    symbols: tuple[str, ...], data_root: str | None = None
) -> dict[str, dict[str, int]]:
    """How Table S4's symbols land on each annotation member, by ``/gene`` only.

    ``unique``: one gene feature carries the symbol; ``ambiguous``: two or more do;
    ``none``: no gene feature's ``/gene`` is the symbol (a synonym route is not
    counted, because which synonyms a route may follow is part of the decision).
    """
    out: dict[str, dict[str, int]] = {}
    for member in (GENBANK_MEMBER, REFSEQ_MEMBER):
        index = symbol_index(member, data_root)
        counts = Counter(
            "none"
            if symbol not in index
            else "unique"
            if len(index[symbol]) == 1
            else "ambiguous"
            for symbol in symbols
        )
        out[member] = {k: counts.get(k, 0) for k in ("unique", "ambiguous", "none")}
    return out


#: Measured on 2026-10-12 by :func:`symbol_routes` over the 3,691 Table S4 symbols.
SYMBOL_ROUTES: Final = {
    GENBANK_MEMBER: {"unique": 3_036, "ambiguous": 0, "none": 655},
    REFSEQ_MEMBER: {"unique": 3_607, "ambiguous": 44, "none": 40},
}


# --------------------------------------------------------------------------- #
# Gate 2: the refusal
# --------------------------------------------------------------------------- #
#: The library mirror holds no Teteneva key (the paper was filed in Zotero on
#: 2026-10-12 and lit_sync has not run since); the raw mirror holds the SI deposit's
#: key, checked on 2026-10-12 by listing both roots.
LIBRARY_MIRROR_KEYS_MATCHING_TETENEVA: Final = 0
RAW_MIRROR_KEYS_MATCHING_TETENEVA: Final = 1
#: The Zotero item the owner filed on 2026-10-12 (#910), and its PDF attachment.
ZOTERO_ITEM: Final = "PUKZFNFZ"
ZOTERO_PDF_ATTACHMENT: Final = "DFFQ6A6W"

# --------------------------------------------------------------------------- #
# Gate 2: the owner's SI deposit (#910, 2026-10-12)
# --------------------------------------------------------------------------- #
SI_PAGE: Final = f"https://doi.org/{PAPER_DOI}"


def _si_file(
    name: str, role: str, sha256: str, size: int, arrived: str, holds: str
) -> DepositedFile:
    """One deposited SI file, with the PMC bucket object measured to equal it."""
    key = f"{_OA_PREFIX}/{name}"
    return DepositedFile(
        path=f"si/{name}",
        role=role,
        sha256=sha256,
        bytes=size,
        original_filename=name,
        arrived_at=f"2026-10-12T03:40:{arrived}-05:00",
        description=holds,
        scripted_equivalent=ScriptedEquivalent(
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            url=pmc_cloud_url(key),
            checked_at="2026-10-12",
        ),
    )


SI_DEPOSIT: Final = ManualDeposit(
    citation_key=EXPECTED_CITATION_KEY,
    doi=PAPER_DOI,
    title=PAPER_TITLE,
    source_url=SI_PAGE,
    retrieval_command=(
        f"open {SI_PAGE} (The ISME Journal) in a browser, open the article's "
        "Supplementary data, download the five files, and copy them unchanged to si/; "
        "or, measured equivalent, fetch each from the PMC Article Datasets bucket "
        f"under {_OA_PREFIX}/ with torchcell.literature.retrieve.pmc_cloud_object"
    ),
    retrieved_by="the owner (mjvolk3), browser download",
    deposited_at="2026-10-12",
    issues=(910,),
    note="five of the article's supplementary files; the PMC bucket also lists "
    "Supplementary Table S1 and Figures S3 and S4, which were not downloaded. The "
    "DOI 10.1093/ismejo/wrae096 is the one NCBI esummary returned for PMC11188689 and "
    f"the one Zotero item {ZOTERO_ITEM} was filed under (#910); every file name "
    "carries the article id 'wrae096'",
    files=(
        _si_file(
            "supplementary_figure_s1_wrae096.pdf",
            ROLE_SI_PDF,
            "65bcac7a61e938a92b90838401e7a153e6ff5ec33ae57b5840950fb11efd5d82",
            354566,
            "57",
            "Supplementary Figure S1, hierarchical clustering of gene fitness, 1 page",
        ),
        _si_file(
            "supplementary_figure_s2_wrae096.pdf",
            ROLE_SI_PDF,
            "d88f962bd32bb756a0d16659e5b3208ad84b7dc620e21d747f0ff549d369f356",
            251076,
            "57",
            "Supplementary Figure S2, PCA of gene fitness per water sample, 1 page",
        ),
        _si_file(
            "supplementary_table_s2_wrae096.pdf",
            ROLE_SI_PDF,
            "c41e8b17ed4ecd8ddf420332e4d32a33aae13ae7b029feb275632a5baf5ab396",
            36151,
            "57",
            "Supplementary Table S2, date, pH, air and water temperature of five lake "
            "water samples, 1 page",
        ),
        _si_file(
            "supplementary_table_s3_wrae096.xlsx",
            ROLE_SI_DATA,
            "5cec5dd2f2b3fbc46a0012154b1864dda812326f12b1622391081e96995e323b",
            14056,
            "57",
            "Supplementary Table S3, sequencing QC of 63 samples (FastQC/MultiQC)",
        ),
        _si_file(
            "supplementary_table_s4_wrae096.xlsx",
            ROLE_SI_DATA,
            "678bbe6cecc2a11517fc11cbeb4d5a5ff88bca3e4aea1264bad336d5fccad0dc",
            1809613,
            "58",
            "Supplementary Table S4, gene fitness values, principal components and "
            "Euclidean distances per water sample",
        ),
    ),
)
TABLE_S4_RELPATH: Final = "si/supplementary_table_s4_wrae096.xlsx"
TABLE_S3_RELPATH: Final = "si/supplementary_table_s3_wrae096.xlsx"


class TableS4Inventory(BaseModel):
    """What Supplementary Table S4's DATA sheet holds, measured off the bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    header_rows: int
    gene_by_sample_rows: int
    rows_per_water_sample: dict[int, int]
    fitness_columns: tuple[str, ...]
    fitness_values: int
    identifier_column: str
    distinct_symbols: tuple[str, ...]
    other_columns: tuple[str, ...]


#: Table S4's DATA sheet: three header rows, then one row per (water sample, gene).
_S4_HEADER_ROWS: Final = 3
_S4_FITNESS: Final = slice(1, 7)
_S4_SYMBOL: Final = 7


def table_s4_inventory(path: str | Path) -> TableS4Inventory:
    """Measure Table S4: rows, water samples, the six fitness columns, the identifiers.

    The three header rows name the block ("gene fitness values", then "filtered" and
    "non-filtered", then "day 2/4/8"), so a fitness column is named
    ``<water>, <day>`` by carrying the merged cells to the right. Raises ``ValueError``
    when the headers are not that layout or a row carries no gene symbol.
    """
    workbook = load_workbook(io.BytesIO(Path(path).read_bytes()), read_only=True)
    rows = list(workbook["DATA"].iter_rows(values_only=True))
    workbook.close()
    top, water, day = rows[:_S4_HEADER_ROWS]
    if (top[0], top[1], top[_S4_SYMBOL]) != (
        "Lake water sample",
        "gene fitness values",
        "Gene name",
    ):
        raise ValueError(f"Table S4 header is {top}")
    columns: list[str] = []
    current = ""
    for w, d in zip(water[_S4_FITNESS], day[_S4_FITNESS], strict=True):
        current = str(w) if w is not None else current
        columns.append(f"{current}, {d}")
    body = rows[_S4_HEADER_ROWS:]
    if any(row[_S4_SYMBOL] is None for row in body):
        raise ValueError("a Table S4 row carries no gene symbol")
    per_sample = Counter(int(row[0]) for row in body)
    return TableS4Inventory(
        header_rows=_S4_HEADER_ROWS,
        gene_by_sample_rows=len(body),
        rows_per_water_sample=dict(sorted(per_sample.items())),
        fitness_columns=tuple(columns),
        fitness_values=sum(1 for r in body for c in r[_S4_FITNESS] if c is not None),
        identifier_column=str(top[_S4_SYMBOL]),
        distinct_symbols=tuple(sorted({str(row[_S4_SYMBOL]) for row in body})),
        other_columns=tuple(str(c) for c in top[_S4_SYMBOL + 1 :] if c is not None),
    )


#: Measured on the deposited bytes on 2026-10-12 by :func:`table_s4_inventory`.
TABLE_S4_GENE_BY_SAMPLE_ROWS: Final = 11_027
TABLE_S4_FITNESS_VALUES: Final = 66_162
TABLE_S4_ROWS_PER_WATER_SAMPLE: Final = {1: 3_675, 2: 3_679, 3: 3_673}
TABLE_S4_DISTINCT_SYMBOLS: Final = 3_691
TABLE_S4_FITNESS_COLUMNS: Final = (
    "filtered, day 2",
    "filtered, day 4",
    "filtered, day 8",
    "non-filtered, day 2",
    "non-filtered, day 4",
    "non-filtered, day 8",
)

_RESOLVE_WITH_PAPER = Provenance(
    source_uri=f"https://doi.org/{PAPER_DOI}",
    page="Materials and methods",
    method="not yet mirrored: filed in Zotero on 2026-10-12 (PUKZFNFZ), mirrored by "
    "the next lit_sync run",
)
_RESOLVE_WITH_TABLE_S4 = Provenance(
    source_uri=f"https://pmc-oa-opendata.s3.amazonaws.com/{TABLE_S4_OA_KEY}",
    page="Supplementary Table S4",
    method="deposited 2026-10-12 as torchcell-raw/<key>/si/"
    "supplementary_table_s4_wrae096.xlsx (byte-identical to this bucket object); its "
    "legend does not state the statistic, so the paper text decides",
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
        "Table S4 names its 3,691 genes by gene symbol only (column 'Gene name'; "
        "measured 2026-10-12, SYMBOL_ROUTES). By /gene alone 3,036 land uniquely on "
        "the GenBank member and 655 on none; 3,607 land uniquely on the RefSeq member, "
        "44 on two or more features and 40 on none. Which member, and which synonym "
        "rule closes the remainder, is the owner's annotation-route decision (#910)",
        in_table=True,
    ),
    _gap(
        "genotype",
        "the library's construction and background (the row records 430,849 unique "
        "insertions in 3,833 genes of an RpoS+ W3110, again from our own note), and "
        "whether a stored genotype is the insertion mutant or the gene",
    ),
)

#: The work left, in order. Filing the paper (PUKZFNFZ) and depositing Tables S3 and
#: S4 were done on 2026-10-12; the route decision is the owner's.
EDITS_NEEDED: Final = (
    "let scripts/lit_sync.py mirror the filed paper (Zotero PUKZFNFZ, filed "
    "2026-10-12) into the library mirror and OCR it; its Methods source the "
    "statistic, time zero, the water and the replicate design",
    "decide the annotation route for ecoli_K12_W3110_ASM1024v1 against the "
    "identifiers Table S4 reports, which are gene symbols only (SYMBOL_ROUTES): "
    "GenBank-first does not apply (the deposit carries no locus_tag), so either the "
    "set is read RefSeq-primary with Y75_RS tags and a symbol route, or the ingest "
    "gains a symbol-plus-ECK:JW:b route",
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
