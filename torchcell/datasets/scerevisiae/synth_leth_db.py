# torchcell/datasets/scerevisiae/synth_leth_db
# [[torchcell.datasets.scerevisiae.synth_leth_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/synth_leth_db
# Test file: tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
"""SynLethDB 2.0: a provenance record of an aggregation that states no growth condition.

Wang et al. 2022 (Nucleic Acids Res. 51:D1, citation key
``wangSynLethDB20Webbased2022``) releases ``Yeast_SL.csv`` (14,000 synthetic-lethal gene
pairs) and ``Yeast_SR.csv`` (6,948 synthetic-rescue pairs), each row an aggregated pair
with a PubMed id naming the paper it came from. They were served as
``SynthLethalityYeastSynthLethDbDataset`` and ``SynthRescueYeastSynthLethDbDataset``
until 2026-10-11.

DECISION: NOT LOADED. This module registers no dataset, in the MCF2Chem 2023
(:mod:`torchcell.datasets.ecoli.cai2023`) and CeCaFDB 2015
(:mod:`torchcell.datasets.ecoli.zhang2015`) pattern. ``Environment.media`` is required,
because a synthetic-lethal call is a growth measurement and a growth measurement
happened in some medium; a source that does not name the medium cannot produce a
record, and the schema is not weakened to admit one.

**1. No field of either file states a growth condition.** Both headers are the same
eight columns, measured by :func:`release_inventory` on the pinned bytes: ``n1.name``,
``n1.identifier``, ``n2.name``, ``n2.identifier``, ``r.cell_line``, ``r.pubmed_id``,
``r.source``, ``r.statistic_score``. None names a medium, a temperature, a carbon
source or a culture state, and ``r.cell_line``, the one column that could carry a
growth context, is empty on all 20,948 rows. The paper states the same thing of its own
data model (:data:`SYNLETHDB_ENTRY_FIELDS_QUOTE`): species, PubMed references,
supporting evidence and cell lines, and nothing about how the cells were grown. So the
storable-record count is **0 of 14,000 SL** and **0 of 6,948 SR**.

**2. The served stores carried a medium nobody stated.** Until #622 they emitted an
unsourced ``Media(name="YEPD", state="solid")``; #622 replaced it with a
composition-deferred stand-in whose own name said the medium was not carried. Both are
the same error in different words: a stand-in joins no shared medium, and a query that
groups by medium reads it as a medium somebody stated.

**3. The medium is per source study, so the source studies are the loadable path.**
Under the 2026-10-10 decomposition policy
([[torchcell.candidates.aggregation-decomposition]]) a served compendium is a
processing reference, not the source of record, and each original study inside it is
loaded from its own release. Measured by :func:`source_decomposition`: the SL file
cites 1,741 distinct PubMed ids and the SR file 1,918, 3,308 distinct across both, and
every one of them is an original study whose own paper states its medium. The SL file
further labels each row's evidence class (``r.source``), 7,721 rows High Throughput,
4,944 Low Throughput, 1,248 both and 87 Text Mining; the whole SR file is labeled
BioGRID, so every SR row is a database re-export rather than a reading of the paper its
PubMed id names.

**4. The SL confidence score is defined, the SR score is not.** The paper defines a
normalized confidence score for SL pairs only
(:data:`SYNLETHDB_SL_SCORE_QUOTE`) and says of SR pairs only that they were collected
(:data:`SYNLETHDB_SR_COLLECTION_QUOTE`). All 14,000 SL rows carry a score, and it is a
function of the evidence label rather than of the pair: 0.5 for High Throughput, 0.8
for Low Throughput, 0.9 for both, and one of four values for Text Mining
(:func:`score_by_source`). 74 SR rows carry a score (-1.0 to -0.2, all
``r.source`` BioGRID, all PubMed 24470217) that no mirrored document defines; the
BioGRID 2019 update it cites (:data:`BIOGRID_2019_REFERENCE`) is not in the mirror, so
those 74 numbers have no sourced meaning and are recorded here rather than stored.

**5. The genes resolve, and the old resolution was wrong** (#597). Each row names both
genes twice, by name and by NCBI Entrez Gene id, and the id resolves through the pinned
RefSeq R64 annotation (:func:`load_entrez_to_orf`) while the name is a cross-check
(``SCerevisiaeGenome.resolve_gene_name``). The resolution that the served stores were
built with was a name to ORF dict in which a later gene's alias overwrote an earlier
gene's standard name, and which turned ``IMP2'`` into ``IMP2`` by ``rstrip("'")``: it
put 872 of 14,000 SL and 119 of 6,948 SR records under the wrong ORF, and produced 9
duplicate unordered ORF pairs (7 SL, 2 SR) that the Entrez resolution does not. This
machinery is kept because a per-study loader needs the same map, not because anything
here builds a store.

Script and committed results:
``experiments/036-dataset-fixes-before-kg-build/scripts/sgd_synlethdb_findings.py``,
``results/sgd_synlethdb_findings.json``.
Finding: [[torchcell.datasets.scerevisiae.synth_leth_db]]. Issues #902, #903, #622.
"""

from __future__ import annotations

import os.path as osp
from collections import Counter
from collections.abc import Mapping
from typing import Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from torchcell.data import verify_sha256
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
#: The raw SynLethDB release files, pinned to the bytes the served builds consumed
#: (``$DATA_ROOT/data/torchcell/synth_{lethality,rescue}_yeast_synth_leth_db/raw/``,
#: 14,000 and 6,948 data rows).
SL_CSV_NAME: Final = "Yeast_SL.csv"
SL_CSV_SHA256: Final = (
    "091e04dbab80044324970feea4a5fc933df0ab7c887042fd87419173129c56ca"
)
SR_CSV_NAME: Final = "Yeast_SR.csv"
SR_CSV_SHA256: Final = (
    "d84fba780cfa55d59f4844324f5c203f85f09bcfe50bfff6de06169ade2a18f4"
)

#: The eight columns of both files, verbatim and in order. Not one names a growth
#: condition, which is clause 1 of the decision.
SYNLETHDB_COLUMNS: Final[tuple[str, ...]] = (
    "n1.name",
    "n1.identifier",
    "n2.name",
    "n2.identifier",
    "r.cell_line",
    "r.pubmed_id",
    "r.source",
    "r.statistic_score",
)

#: The RefSeq annotation of the R64 assembly (header ``#!genome-build-accession
#: NCBI_Assembly:GCF_000146045.2``, ``#!annotation-source SGD R64-4-1``) that maps an
#: Entrez Gene id to its ORF. It sits beside the SGD release files under the genome's
#: ``genome_root``; it is NOT part of the genomes-tier set ``sgd_S288C_R64-4-1_20230830``
#: (scripts/migrate_genomes_tier.py), and NCBI now serves an R64-5-1 annotation under
#: the same accession, so these bytes cannot be re-fetched; the pin is the record.
NCBI_GFF_RELPATH: Final = "S288C_reference_genome_R64-4-1_20230830/ncbi_genomic.gff"
NCBI_GFF_SHA256: Final = (
    "8200def54936659721e3f7b4484d725667ed4e23492edefbbca667b8430209c4"
)

#: SynLethDB 2.0 (Wang 2022, NAR), the release paper of the aggregated CSVs (issue #622).
#: Its mirror OCR is the only document that says what one SL entry carries.
SYNLETHDB_CITATION_KEY: Final = "wangSynLethDB20Webbased2022"
SYNLETHDB_PAPER_SHA256: Final = (
    "acaee2c7e16480fb39656cdf6a5e7ac621c34f08b41433dcab2cebe7daf1f20a"
)
SYNLETHDB_PAPER: Final = Provenance(
    source_uri="paper.md",
    citation_key=SYNLETHDB_CITATION_KEY,
    sha256=SYNLETHDB_PAPER_SHA256,
)

#: paper.md line 49: the fields an SL entry carries. No growth medium and no growth
#: temperature, and the released CSV header agrees (:data:`SYNLETHDB_COLUMNS`).
SYNLETHDB_ENTRY_FIELDS_QUOTE: Final = (
    "The species, references to PubMed, supporting evidence, cell lines and other "
    "relevant information about an SL entry are stored as properties of the edge"
)

#: paper.md line 38: SynLethDB's second source is BioGRID, reference 22 (the BioGRID
#: 2019 update, Nucleic Acids Res. 47, D529-D541, DOI 10.1093/nar/gky1079), which is
#: not in the mirror.
SYNLETHDB_BIOGRID_SOURCE_QUOTE: Final = (
    "The second source is public databases containing SL data such as GenomeRNAi (21) "
    "and BioGRID (22)."
)

#: paper.md line 38: what SynLethDB says about its SR pairs. No SR score is defined
#: anywhere in the paper; the confidence score it defines is for SL pairs only
#: (:data:`SYNLETHDB_SL_SCORE_QUOTE`).
SYNLETHDB_SR_COLLECTION_QUOTE: Final = (
    "We collected 16 207 SR gene pairs and 5798 non-synthetic lethal (non-SL) gene "
    "pairs from the above three sources"
)

#: paper.md line 75 (OCR drops the 'fi' ligature, kept verbatim).
SYNLETHDB_SL_SCORE_QUOTE: Final = (
    "we integrated the scores of different types of sources into a normalized "
    "confdence score for every SL pair"
)

#: Every SynLethDB 2.0 sentence this module rests on, as audited ``SourcedValue``s
#: (each is a verbatim substring of the pinned ``paper.md``).
SYNLETHDB_SOURCED_QUOTES: Final[tuple[SourcedValue, ...]] = (
    SourcedValue(
        value="an SL entry carries species, PubMed, evidence and cell line; no medium "
        "and no temperature",
        provenance=SYNLETHDB_PAPER,
        quote=SYNLETHDB_ENTRY_FIELDS_QUOTE,
        note="line 49; the sentence that settles clause 1",
    ),
    SourcedValue(
        value="BioGRID is a source, cited as reference 22",
        provenance=SYNLETHDB_PAPER,
        quote=SYNLETHDB_BIOGRID_SOURCE_QUOTE,
        note="line 38",
    ),
    SourcedValue(
        value="SR pairs were collected from the same three sources; no SR score is "
        "defined",
        provenance=SYNLETHDB_PAPER,
        quote=SYNLETHDB_SR_COLLECTION_QUOTE,
        note="line 38",
    ),
    SourcedValue(
        value="the confidence score is defined for SL pairs",
        provenance=SYNLETHDB_PAPER,
        quote=SYNLETHDB_SL_SCORE_QUOTE,
        note="line 75; OCR drops the 'fi' ligature",
    ),
)

#: The BioGRID 2019 update, SynLethDB 2.0's reference 22 (paper.pdf reference list,
#: "The BioGRID interaction database: update. Nucleic Acids Res., 47, D529-D541"). Not
#: mirrored; reading it (and each row's own primary paper) is what would define the 74
#: SR scores of clause 4.
BIOGRID_2019_REFERENCE: Final = Provenance(
    source_uri="https://doi.org/10.1093/nar/gky1079",
    method="SynLethDB 2.0 reference 22 (Oughtred et al. 2019, The BioGRID interaction "
    "database: 2019 update, Nucleic Acids Res. 47, D529-D541); not in the mirror",
)

#: GFF feature types whose ``Dbxref`` ``GeneID`` names a locus (``locus_tag``).
_NCBI_LOCUS_TYPES: Final = frozenset({"gene", "pseudogene"})

#: What the served stores held under the old name-based resolution (#597), recorded so
#: the finding does not leave with the stores.
OLD_RESOLUTION_WRONG_ORF: Final[dict[str, int]] = {
    "Yeast_SL.csv": 872,
    "Yeast_SR.csv": 119,
}
#: Duplicate unordered ORF pairs the old name map produced, which Entrez resolution does
#: not: 7 in SL and 2 in SR.
OLD_RESOLUTION_DUPLICATE_PAIRS: Final[dict[str, int]] = {
    "Yeast_SL.csv": 7,
    "Yeast_SR.csv": 2,
}


class BlankPubmedIdError(ValueError):
    """A SynLethDB row with a blank or whitespace-only ``r.pubmed_id``.

    Every row cites its PMID, so a blank one is refused by name. The pinned
    ``Yeast_SL.csv`` (14,000 rows) and ``Yeast_SR.csv`` (6,948 rows) have none
    (0 empty and 0 whitespace-only cells).
    """


class EntrezGeneIdConflictError(ValueError):
    """One ``GeneID`` in the NCBI GFF carries two different ``locus_tag`` values."""


class ReleaseInventory(BaseModel):
    """One released file, and whether it states anything a record would need."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    columns: list[str] = Field(description="the header, verbatim and in order")
    n_rows: int
    n_condition_columns: int = Field(
        description="columns naming a medium, temperature, carbon source or culture "
        "state; the count clause 1 rests on"
    )
    n_cell_line_stated: int = Field(
        description="rows with a non-empty r.cell_line, the one column that could "
        "carry a growth context"
    )
    n_scored: int = Field(description="rows with a non-null r.statistic_score")
    n_storable: int = Field(
        description="rows that could be stored with a sourced Media; 0 while no column "
        "states a medium"
    )


class SourceDecomposition(BaseModel):
    """The original studies inside one released file (the decomposition policy)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    n_rows: int
    n_source_pubmed_ids: int = Field(
        description="distinct r.pubmed_id values, i.e. original studies to load from "
        "their own releases"
    )
    n_multi_pubmed_rows: int = Field(
        description="rows citing more than one PubMed id (';'-joined)"
    )
    rows_by_evidence: dict[str, int] = Field(
        description="row count per r.source evidence label"
    )
    pubmed_ids_by_evidence: dict[str, int] = Field(
        description="distinct PubMed ids per r.source evidence label"
    )


#: Header terms that would make a column a growth-condition column. Deliberately
#: generous: the point is that even a generous rule finds none.
CONDITION_HEADER_TERMS: Final[tuple[str, ...]] = (
    "media",
    "medium",
    "temperature",
    "carbon",
    "nitrogen",
    "glucose",
    "state",
    "solid",
    "liquid",
    "condition",
)


def read_synlethdb_csv(path: str) -> pd.DataFrame:
    """Read a SynLethDB CSV with ``r.pubmed_id`` as text, refusing a blank PMID.

    A cell that is empty or only whitespace counts as blank.

    Reading the PMID as text keeps it verbatim (``"18676811"``, or
    ``"24125552;19918932"`` for a row citing two papers); read as a number, one blank
    cell would turn the whole column float and every PMID into ``"111.0"``. The Entrez
    ids are read as integers; a blank or non-integer id raises from pandas.
    """
    df = pd.read_csv(
        path,
        dtype={
            "r.pubmed_id": "str",
            "n1.identifier": "int64",
            "n2.identifier": "int64",
        },
    )
    blank = df["r.pubmed_id"].fillna("").str.strip().eq("")
    if blank.any():
        raise BlankPubmedIdError(
            f"{path}: {int(blank.sum())} row(s) with a blank r.pubmed_id "
            f"(first at row {int(blank.to_numpy().nonzero()[0][0])})"
        )
    return df


class ConditionColumnFound(ValueError):
    """A released file carries a column the decision measured absent.

    Clause 1 reads 0 storable rows because no column states a growth condition. A
    release that grew one invalidates the measurement rather than being absorbed by it,
    so it raises here instead of being counted.
    """


def release_inventory(path: str, sha256: str) -> ReleaseInventory:
    """Clause 1, measured: what one released file states, on verified bytes."""
    verify_sha256(path, sha256)
    df = read_synlethdb_csv(path)
    columns = [str(column).lstrip("﻿") for column in df.columns]
    condition_columns = [
        column
        for column in columns
        if any(term in column.casefold() for term in CONDITION_HEADER_TERMS)
    ]
    if condition_columns:
        raise ConditionColumnFound(
            f"{path} carries {condition_columns}, which the decision measured absent; "
            "re-read clause 1 before anything is stored from these bytes"
        )
    cell_line = df["r.cell_line"].fillna("").astype(str).str.strip()
    return ReleaseInventory(
        name=osp.basename(path),
        sha256=sha256,
        columns=columns,
        n_rows=len(df),
        n_condition_columns=0,
        n_cell_line_stated=int(cell_line.ne("").sum()),
        n_scored=int(df["r.statistic_score"].notna().sum()),
        n_storable=0,
    )


def source_decomposition(path: str, sha256: str) -> SourceDecomposition:
    """Clause 3, measured: the original studies one released file aggregates."""
    verify_sha256(path, sha256)
    df = read_synlethdb_csv(path)
    pubmed = df["r.pubmed_id"].astype(str)
    ids_by_evidence: dict[str, set[str]] = {}
    for evidence, value in zip(df["r.source"].astype(str), pubmed, strict=True):
        ids_by_evidence.setdefault(evidence, set()).update(value.split(";"))
    distinct = {one for value in pubmed for one in value.split(";")}
    return SourceDecomposition(
        name=osp.basename(path),
        n_rows=len(df),
        n_source_pubmed_ids=len(distinct),
        n_multi_pubmed_rows=int(pubmed.str.contains(";").sum()),
        rows_by_evidence=dict(sorted(Counter(df["r.source"].astype(str)).items())),
        pubmed_ids_by_evidence={
            evidence: len(ids) for evidence, ids in sorted(ids_by_evidence.items())
        },
    )


def score_by_source(path: str, sha256: str) -> dict[str, list[float]]:
    """Clause 4, measured: the distinct scores each evidence label carries."""
    verify_sha256(path, sha256)
    df = read_synlethdb_csv(path)
    out: dict[str, set[float]] = {}
    for evidence, score in zip(
        df["r.source"].astype(str), df["r.statistic_score"], strict=True
    ):
        if pd.notna(score):
            out.setdefault(evidence, set()).add(float(score))
    return {evidence: sorted(scores) for evidence, scores in sorted(out.items())}


def ncbi_gff_path(genome: SCerevisiaeGenome) -> str:
    """Where the genome's directory keeps the pinned ``ncbi_genomic.gff``."""
    return osp.join(genome.genome_root, NCBI_GFF_RELPATH)


def load_entrez_to_orf(gff_path: str) -> dict[int, str]:
    """Map every Entrez Gene id in the pinned NCBI GFF to its ORF (``locus_tag``).

    The file is verified against :data:`NCBI_GFF_SHA256` before it is read. Only
    ``gene`` and ``pseudogene`` features count; a ``GeneID`` that names two different
    locus tags raises :class:`EntrezGeneIdConflictError`. This is the map a per-study
    loader of any SynLethDB source paper resolves its gene pairs with (clause 5).
    """
    verify_sha256(gff_path, NCBI_GFF_SHA256)
    mapping: dict[int, str] = {}
    with open(gff_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in _NCBI_LOCUS_TYPES:
                continue
            attributes = dict(kv.split("=", 1) for kv in fields[8].split(";") if kv)
            gene_ids = [
                ref[len("GeneID:") :]
                for ref in attributes["Dbxref"].split(",")
                if ref.startswith("GeneID:")
            ]
            if len(gene_ids) != 1:
                raise EntrezGeneIdConflictError(
                    f"{gff_path}: feature {attributes['ID']} carries {len(gene_ids)} "
                    "GeneID cross-references"
                )
            entrez = int(gene_ids[0])
            locus_tag = attributes["locus_tag"]
            if mapping.setdefault(entrez, locus_tag) != locus_tag:
                raise EntrezGeneIdConflictError(
                    f"{gff_path}: GeneID {entrez} names both {mapping[entrez]} and "
                    f"{locus_tag}"
                )
    return mapping


def resolved_orf_pairs(
    df: pd.DataFrame, entrez_to_orf: Mapping[int, str]
) -> list[tuple[str, str] | None]:
    """Each row's ORF pair by Entrez id alone, or None when a side is not in the GFF.

    The gene-name cross-check needs a genome, so it is not done here; this is the id
    half of clause 5, which is what a per-study loader starts from.
    """
    return [
        None
        if (orf1 := entrez_to_orf.get(int(e1))) is None
        or (orf2 := entrez_to_orf.get(int(e2))) is None
        else (orf1, orf2)
        for e1, e2 in zip(df["n1.identifier"], df["n2.identifier"], strict=True)
    ]


def main() -> None:
    """Re-measure every clause of the decision from the pinned release files."""
    from dotenv import load_dotenv

    from torchcell.graph.sgd import data_root

    load_dotenv()
    root = data_root()
    for relroot, name, sha in (
        ("data/torchcell/syn_leth_db_yeast", SL_CSV_NAME, SL_CSV_SHA256),
        ("data/torchcell/syn_rescue_db_yeast", SR_CSV_NAME, SR_CSV_SHA256),
    ):
        path = osp.join(root, relroot, "raw", name)
        print(release_inventory(path, sha).model_dump_json(indent=2))
        print(source_decomposition(path, sha).model_dump_json(indent=2))
        print(score_by_source(path, sha))


if __name__ == "__main__":
    main()
