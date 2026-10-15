"""SynLethDB-derived yeast synthetic lethality and synthetic rescue datasets.

Gene identity (#1045). Every row of ``Yeast_SL.csv`` and ``Yeast_SR.csv`` names both
genes twice: a gene name (``n1.name``, ``n2.name``) and an NCBI Entrez Gene id
(``n1.identifier``, ``n2.identifier``). Each side is stored exactly as released on
``ReleasedIdentifierPerturbation``: the name verbatim in ``systematic_gene_name`` and
``released_gene_name``, the Entrez id verbatim in ``released_gene_id`` (namespace
``ncbi_entrez_gene``), and ``released_kind`` a typed gap, because SynLethDB states the
pair and its source, not the perturbation that made it. No genome or GFF is consulted;
a row naming one gene on both sides is stored as released (the pair's two sides are
one perturbation), and the verifier is told how many such rows the file holds. The
only refusals are a blank PMID and a repeated unordered (name, id) pair, neither of
which the pinned files carry. Every released row is a record: 14,000 SL and 6,948 SR.

Before #1045 the loader resolved each side by its Entrez id through the pinned NCBI GFF
and dropped 4 SL and 6 SR rows under named rules (an id the GFF lacks, a self pair).
"""

# torchcell/datasets/scerevisiae/synth_leth_db
# [[torchcell.datasets.scerevisiae.synth_leth_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/synth_leth_db
# Test file: tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py

import logging
import os
import os.path as osp
import pickle
from collections import Counter
from collections.abc import Callable
from typing import Any

import lmdb
import pandas as pd
import requests
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    Media,
    Publication,
    ReferenceGenome,
    ReleasedGeneIdNamespace,
    ReleasedIdentifierPerturbation,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
    SyntheticLethalityPhenotype,
    SyntheticRescueExperiment,
    SyntheticRescueExperimentReference,
    SyntheticRescuePhenotype,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#: The raw SynLethDB release files, pinned to the bytes the served build consumed
#: (``$DATA_ROOT/data/torchcell/synth_{lethality,rescue}_yeast_synth_leth_db/raw/``,
#: 14,000 and 6,948 data rows).
SL_CSV_NAME = "Yeast_SL.csv"
SL_CSV_SHA256 = "091e04dbab80044324970feea4a5fc933df0ab7c887042fd87419173129c56ca"
SR_CSV_NAME = "Yeast_SR.csv"
SR_CSV_SHA256 = "d84fba780cfa55d59f4844324f5c203f85f09bcfe50bfff6de06169ade2a18f4"

#: The RefSeq annotation of the R64 assembly (header ``#!genome-build-accession
#: NCBI_Assembly:GCF_000146045.2``, ``#!annotation-source SGD R64-4-1``) that maps an
#: Entrez Gene id to its ORF. It sits beside the SGD release files under the genome's
#: ``genome_root``; it is NOT part of the genomes-tier set ``sgd_S288C_R64-4-1_20230830``
#: (scripts/migrate_genomes_tier.py), and NCBI now serves an R64-5-1 annotation under
#: the same accession, so these bytes cannot be re-fetched; the pin is the record.
NCBI_GFF_RELPATH = "S288C_reference_genome_R64-4-1_20230830/ncbi_genomic.gff"
NCBI_GFF_SHA256 = "8200def54936659721e3f7b4484d725667ed4e23492edefbbca667b8430209c4"

#: SynLethDB 2.0 (Wang 2022, NAR), the release paper of the aggregated CSVs (issue #622).
#: Its mirror OCR is the only document that says what one SL entry carries.
SYNLETHDB_CITATION_KEY = "wangSynLethDB20Webbased2022"
SYNLETHDB_PAPER_SHA256 = (
    "acaee2c7e16480fb39656cdf6a5e7ac621c34f08b41433dcab2cebe7daf1f20a"
)
_SYNLETHDB_PAPER = Provenance(
    source_uri="paper.md",
    citation_key=SYNLETHDB_CITATION_KEY,
    sha256=SYNLETHDB_PAPER_SHA256,
)
#: paper.md line 49: the fields an SL entry carries. No growth medium and no growth
#: temperature, and the released CSV header agrees (``n1.name, n1.identifier, n2.name,
#: n2.identifier, r.cell_line, r.pubmed_id, r.source, r.statistic_score``).
SYNLETHDB_ENTRY_FIELDS_QUOTE = (
    "The species, references to PubMed, supporting evidence, cell lines and other "
    "relevant information about an SL entry are stored as properties of the edge"
)

#: paper.md line 38: SynLethDB's second source is BioGRID, reference 22 (the BioGRID
#: 2019 update, Nucleic Acids Res. 47, D529-D541, DOI 10.1093/nar/gky1079), which is
#: not in the mirror.
SYNLETHDB_BIOGRID_SOURCE_QUOTE = (
    "The second source is public databases containing SL data such as GenomeRNAi (21) "
    "and BioGRID (22)."
)
#: paper.md line 38: what SynLethDB says about its SR pairs. No SR score is defined
#: anywhere in the paper; the confidence score it defines is for SL pairs only
#: (``SYNLETHDB_SL_SCORE_QUOTE``).
SYNLETHDB_SR_COLLECTION_QUOTE = (
    "We collected 16 207 SR gene pairs and 5798 non-synthetic lethal (non-SL) gene "
    "pairs from the above three sources"
)
#: paper.md line 75 (OCR drops the 'fi' ligature, kept verbatim).
SYNLETHDB_SL_SCORE_QUOTE = (
    "we integrated the scores of different types of sources into a normalized "
    "confdence score for every SL pair"
)

#: Every SynLethDB 2.0 sentence this module rests on, as audited ``SourcedValue``s
#: (each is a verbatim substring of the pinned ``paper.md``).
SYNLETHDB_SOURCED_QUOTES: tuple[SourcedValue, ...] = (
    SourcedValue(
        value="an SL entry carries species, PubMed, evidence and cell line; no medium "
        "and no temperature",
        provenance=_SYNLETHDB_PAPER,
        quote=SYNLETHDB_ENTRY_FIELDS_QUOTE,
        note="line 49",
    ),
    SourcedValue(
        value="BioGRID is a source, cited as reference 22",
        provenance=_SYNLETHDB_PAPER,
        quote=SYNLETHDB_BIOGRID_SOURCE_QUOTE,
        note="line 38",
    ),
    SourcedValue(
        value="SR pairs were collected from the same three sources; no SR score is "
        "defined",
        provenance=_SYNLETHDB_PAPER,
        quote=SYNLETHDB_SR_COLLECTION_QUOTE,
        note="line 38",
    ),
    SourcedValue(
        value="the confidence score is defined for SL pairs",
        provenance=_SYNLETHDB_PAPER,
        quote=SYNLETHDB_SL_SCORE_QUOTE,
        note="line 75; OCR drops the 'fi' ligature",
    ),
)

# --- begin #902/#903: a typed gap on the medium -------------------------------- #
#: What every SynLethDB medium gap quotes: the paper's own list of what an SL entry
#: stores, and the released header. Neither carries a condition field, and
#: ``r.cell_line``, the one column that could carry a growth context, is empty on all
#: 14,000 SL and 6,948 SR rows (``n_cell_line_stated`` 0 in
#: ``experiments/036-dataset-fixes-before-kg-build/results/sgd_synlethdb_findings.json``).
SYNLETHDB_MEDIUM_SILENCE = (
    "SynLethDB carries no growth medium: paper.md line 49 lists what an SL entry stores "
    "('" + SYNLETHDB_ENTRY_FIELDS_QUOTE + "') and both released CSVs carry only "
    "n1.name, n1.identifier, n2.name, n2.identifier, r.cell_line, r.pubmed_id, "
    "r.source, r.statistic_score, with r.cell_line empty on every row. The medium, if "
    "stated at all, is in each record's own Publication.pubmed_id. Replaces the #622 "
    "composition-deferred stand-in medium, which joined no shared medium"
)
SYNLETHDB_MEDIUM = Media(
    name=None,
    state=None,
    is_synthetic=None,
    provenance_gaps=[
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_carried_by_curation,
            looked_in=_SYNLETHDB_PAPER,
            note=SYNLETHDB_MEDIUM_SILENCE,
        )
        for field in ("name", "state", "is_synthetic")
    ],
)
"""The medium of every SL and SR record: a ``Media`` whose name, state and class are
typed gaps quoting the release's silence, never a guessed YPD or a free-text stand-in."""
# --- end #902/#903 ------------------------------------------------------------- #

#: The temperature is absent for the same reason (``not_carried_by_curation``).
SYNLETHDB_TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_carried_by_curation,
    looked_in=_SYNLETHDB_PAPER,
    note="SynLethDB carries no growth temperature (paper.md line 49 and the CSV "
    "header); the loader formerly stored an unsourced 30 C",
)

SYNLETHDB_ENVIRONMENT = Environment(
    media=SYNLETHDB_MEDIUM,
    temperature=None,
    provenance_gaps=[SYNLETHDB_TEMPERATURE_GAP],
)
"""Every SL and SR record's environment: neither medium nor temperature is carried."""

# --- begin #903: the SR file's statistic score has no sourced definition -------- #
#: The BioGRID 2019 update, SynLethDB 2.0's reference 22 (paper.pdf reference list,
#: "The BioGRID interaction database: update. Nucleic Acids Res., 47, D529–D541"). Not
#: mirrored; reading it (and the row's own primary paper) is what would define the
#: SR scores, so it is the gap's ``resolve_with``.
BIOGRID_2019_REFERENCE = Provenance(
    source_uri="https://doi.org/10.1093/nar/gky1079",
    method="SynLethDB 2.0 reference 22 (Oughtred et al. 2019, The BioGRID interaction "
    "database: 2019 update, Nucleic Acids Res. 47, D529-D541); not in the mirror",
)


def sr_score_gap(score: float, source: str, pubmed_id: str) -> ProvenanceGap:
    """The typed absence stored in place of one released SR ``r.statistic_score``.

    ``Yeast_SR.csv`` states a score on 74 of its 6,948 rows (all ``r.source`` BioGRID,
    all PubMed 24470217, -1.0 to -0.2). SynLethDB 2.0 defines a normalized confidence
    score for SL pairs only (``SYNLETHDB_SL_SCORE_QUOTE``) and says of SR pairs only
    that they were collected (``SYNLETHDB_SR_COLLECTION_QUOTE``); the BioGRID
    documentation it cites is not mirrored. A number with no sourced meaning is not
    stored as SynLethDB's statistic score: the record keeps ``None`` and this gap,
    whose note carries the released value verbatim so nothing is lost.
    """
    return ProvenanceGap(
        field="synthetic_rescue_statistic_score",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        looked_in=_SYNLETHDB_PAPER,
        resolve_with=BIOGRID_2019_REFERENCE,
        note=f"Yeast_SR.csv states r.statistic_score {score!r} (r.source {source}, "
        f"r.pubmed_id {pubmed_id}); no mirrored source defines it: SynLethDB 2.0 "
        "defines its confidence score for SL pairs only and cites BioGRID "
        "(reference 22, not mirrored) for its database-sourced pairs",
    )


# --- end #903 ------------------------------------------------------------------ #

#: GFF feature types whose ``Dbxref`` ``GeneID`` names a locus (``locus_tag``).
_NCBI_LOCUS_TYPES = frozenset({"gene", "pseudogene"})

#: Retention rule (#1045): every released row is a record.
RETENTION_RULE = (
    "one record per released row, each side stored on ReleasedIdentifierPerturbation "
    "under its released name and Entrez id verbatim; no genome or GFF is consulted and "
    "no row is dropped, a row naming one gene on both sides included"
)

#: What the release says about the perturbation behind a pair: nothing. The paper
#: lists what an SL entry stores (``SYNLETHDB_ENTRY_FIELDS_QUOTE``) and the released
#: header carries the two genes, the cell line, the PubMed id, the source and the
#: score; how either gene was perturbed in the cited study is not a field.
SYNLETHDB_KIND_GAP = ProvenanceGap(
    field="released_kind",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri="paper.md",
        citation_key=SYNLETHDB_CITATION_KEY,
        sha256=SYNLETHDB_PAPER_SHA256,
    ),
    note=(
        "SynLethDB names the two genes of a pair (n1.name, n1.identifier, n2.name, "
        "n2.identifier) and the pair's source, PubMed id and score; it does not say "
        "how either gene was perturbed in the cited study ('"
        + SYNLETHDB_ENTRY_FIELDS_QUOTE
        + "'). The perturbation, if stated, is in each record's own "
        "Publication.pubmed_id"
    ),
)


class BlankPubmedIdError(ValueError):
    """A SynLethDB row with a blank or whitespace-only ``r.pubmed_id``.

    Every record cites its PMID, so a blank one is refused by name. The pinned
    ``Yeast_SL.csv`` (14,000 rows) and ``Yeast_SR.csv`` (6,948 rows) have none
    (0 empty and 0 whitespace-only cells).
    """


class EntrezGeneIdConflictError(ValueError):
    """One ``GeneID`` in the NCBI GFF carries two different ``locus_tag`` values."""


class DuplicatePairError(ValueError):
    """Two released rows name one unordered (name, Entrez id) pair.

    The pinned files have none; the nine the old name map stored (7 SL, 2 SR) were
    all artifacts of its alias overwrite.
    """


class DropLog(BaseModel):
    """The build's accounting: every released row is a record, so nothing is dropped.

    ``rules`` is empty and kept for the verifier, which reads the ledger's drop rules
    to compare the store with the release; ``n_self_pairs`` counts the rows that name
    one gene on both sides, which the verifier is told to expect.
    """

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rule: str
    n_self_pairs: int
    rules: list[dict[str, Any]]


def _read_synlethdb_csv(path: str) -> pd.DataFrame:
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


# --- retained utilities: the Entrez -> ORF map of the pinned NCBI GFF, which the
# loader no longer consults (#1045) and experiments/036 still reads ------------------ #
def ncbi_gff_path(genome: SCerevisiaeGenome) -> str:
    """Where the genome's directory keeps the pinned ``ncbi_genomic.gff``."""
    return osp.join(genome.genome_root, NCBI_GFF_RELPATH)


def load_entrez_to_orf(gff_path: str) -> dict[int, str]:
    """Map every Entrez Gene id in the pinned NCBI GFF to its ORF (``locus_tag``).

    The file is verified against ``NCBI_GFF_SHA256`` before it is read. Only ``gene``
    and ``pseudogene`` features count; a ``GeneID`` that names two different locus
    tags raises ``EntrezGeneIdConflictError``.
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


def released_side(name: str, entrez: int) -> ReleasedIdentifierPerturbation:
    """One side of a pair, exactly as the release names it (#1045)."""
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=name,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=name,
        identifier_kind="gene name (the n1.name / n2.name column), as released",
        released_gene_id=str(entrez),
        released_gene_id_namespace=ReleasedGeneIdNamespace.ncbi_entrez_gene,
        released_kind=None,
        provenance_gaps=[SYNLETHDB_KIND_GAP],
    )


def pair_keys(df: pd.DataFrame) -> list[frozenset[tuple[str, int]]]:
    """The unordered (name, Entrez id) pair of every row, in row order."""
    return [
        frozenset(((str(n1), int(e1)), (str(n2), int(e2))))
        for n1, e1, n2, e2 in zip(
            df["n1.name"],
            df["n1.identifier"],
            df["n2.name"],
            df["n2.identifier"],
            strict=True,
        )
    ]


def refuse_duplicate_pairs(df: pd.DataFrame) -> None:
    """Raise ``DuplicatePairError`` when two rows name one unordered (name, id) pair."""
    pairs = pair_keys(df)
    repeated = {pair for pair, n in Counter(pairs).items() if n > 1}
    if repeated:
        rows = [
            int(index)
            for index, pair in zip(df.index, pairs, strict=True)
            if pair in repeated
        ]
        raise DuplicatePairError(
            f"{len(repeated)} released pair(s) repeat across rows {rows}: "
            f"{sorted(sorted(p) for p in repeated)}"
        )


def drop_log(dataset_name: str, df: pd.DataFrame) -> DropLog:
    """The ledger of a build: every row a record, the self pairs counted."""
    return DropLog(
        dataset=dataset_name,
        source_records=int(len(df)),
        kept_records=int(len(df)),
        dropped_records=0,
        rule=RETENTION_RULE,
        n_self_pairs=sum(1 for pair in pair_keys(df) if len(pair) == 1),
        rules=[],
    )


def _write_synlethdb_lmdb(
    dataset: "SynthLethalityYeastSynthLethDbDataset | SynthRescueYeastSynthLethDbDataset",
    csv_path: str,
) -> None:
    """Read, ledger and write one SynLethDB file (shared by both datasets)."""
    df = dataset.preprocess_raw(_read_synlethdb_csv(csv_path))
    refuse_duplicate_pairs(df)
    os.makedirs(dataset.processed_dir, exist_ok=True)
    os.makedirs(dataset.preprocess_dir, exist_ok=True)
    ledger = drop_log(dataset.name, df)
    with open(osp.join(dataset.preprocess_dir, "dropped_records.json"), "w") as f:
        f.write(ledger.model_dump_json(indent=2))

    env = lmdb.open(osp.join(dataset.processed_dir, "lmdb"), map_size=int(1e12))
    with env.begin(write=True) as txn:
        for key, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            experiment, reference, publication = dataset.create_experiment(
                dataset.name, row
            )
            serialized_data = pickle.dumps(
                {
                    "experiment": experiment.model_dump(),
                    "reference": reference.model_dump(),
                    "publication": publication.model_dump(),
                }
            )
            txn.put(f"{key}".encode(), serialized_data)
    env.close()
    log.info(
        "%s: wrote %d of %d rows as released (%d name one gene on both sides)",
        dataset.name,
        ledger.kept_records,
        ledger.source_records,
        ledger.n_self_pairs,
    )


@register_dataset
class SynthLethalityYeastSynthLethDbDataset(ExperimentDataset):
    """Yeast synthetic lethality gene-pair experiments from SynLethDB."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/syn_leth_db_yeast",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the dataset; every released row is stored as released."""
        super().__init__(root, io_workers, transform, pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw synthetic-lethality CSV filename."""
        return [SL_CSV_NAME]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed LMDB directory name."""
        return ["lmdb"]

    @property
    def experiment_class(self) -> type[SyntheticLethalityExperiment]:
        """Return the synthetic-lethality experiment schema class."""
        return SyntheticLethalityExperiment

    @property
    def reference_class(self) -> type[SyntheticLethalityExperimentReference]:
        """Return the synthetic-lethality experiment-reference schema class."""
        return SyntheticLethalityExperimentReference

    def download(self) -> None:
        """Download the synthetic-lethality CSV from Google Drive."""
        url = "https://drive.google.com/uc?export=download&id=1_56ebyBatapNml8S5HlJW7Dz1l0DZZIq"
        download_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        os.makedirs(self.raw_dir, exist_ok=True)
        log.info(f"Downloading {url} to {download_path}")

        session = requests.Session()
        response = session.get(url, stream=True)

        if "download_warning" in response.cookies:
            params = {"confirm": response.cookies["download_warning"]}
            response = session.get(url, params=params, stream=True)

        response.raise_for_status()

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info("Download completed successfully.")

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Return the released rows unchanged; nothing is resolved (#1045)."""
        return df

    @post_process
    def process(self) -> None:
        """Verify the raw CSV, ledger its rows, and write the LMDB."""
        verify_raw_files(self.raw_dir, {SL_CSV_NAME: SL_CSV_SHA256})
        log.info("Processing Synthetic Lethality Yeast Data...")
        _write_synlethdb_lmdb(self, osp.join(self.raw_dir, SL_CSV_NAME))

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        SyntheticLethalityExperiment, SyntheticLethalityExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects for one row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        genotype = Genotype(
            perturbations=[
                released_side(str(row["n1.name"]), int(row["n1.identifier"])),
                released_side(str(row["n2.name"]), int(row["n2.identifier"])),
            ]
        )

        environment = SYNLETHDB_ENVIRONMENT

        phenotype = SyntheticLethalityPhenotype(
            is_synthetic_lethal=True,
            synthetic_lethality_statistic_score=float(row["r.statistic_score"]),
        )

        phenotype_reference = SyntheticLethalityPhenotype(
            is_synthetic_lethal=False, synthetic_lethality_statistic_score=None
        )

        experiment = SyntheticLethalityExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        reference = SyntheticLethalityExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment,
            phenotype_reference=phenotype_reference,
        )

        publication = Publication(
            pubmed_id=str(row["r.pubmed_id"]),
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{row['r.pubmed_id']}/",
            doi=None,
            doi_url=None,
        )
        return experiment, reference, publication


@register_dataset
class SynthRescueYeastSynthLethDbDataset(ExperimentDataset):
    """Yeast synthetic rescue gene-pair experiments from SynLethDB."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/syn_rescue_db_yeast",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the dataset; every released row is stored as released."""
        super().__init__(root, io_workers, transform, pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw synthetic-rescue CSV filename."""
        return [SR_CSV_NAME]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed LMDB directory name."""
        return ["lmdb"]

    @property
    def experiment_class(self) -> type[SyntheticRescueExperiment]:
        """Return the synthetic-rescue experiment schema class."""
        return SyntheticRescueExperiment

    @property
    def reference_class(self) -> type[SyntheticRescueExperimentReference]:
        """Return the synthetic-rescue experiment-reference schema class."""
        return SyntheticRescueExperimentReference

    def download(self) -> None:
        """Download the synthetic-rescue CSV from Google Drive."""
        url = "https://drive.google.com/uc?export=download&id=1lBaApm70E05JnkrE1Hwmn8gT1cV5Bzlt"
        download_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        os.makedirs(self.raw_dir, exist_ok=True)
        log.info(f"Downloading {url} to {download_path}")

        session = requests.Session()
        response = session.get(url, stream=True)

        if "download_warning" in response.cookies:
            params = {"confirm": response.cookies["download_warning"]}
            response = session.get(url, params=params, stream=True)

        response.raise_for_status()

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info("Download completed successfully.")

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Return the released rows unchanged; nothing is resolved (#1045)."""
        return df

    @post_process
    def process(self) -> None:
        """Verify the raw CSV, ledger its rows, and write the LMDB."""
        verify_raw_files(self.raw_dir, {SR_CSV_NAME: SR_CSV_SHA256})
        log.info("Processing Synthetic Rescue Yeast Data...")
        _write_synlethdb_lmdb(self, osp.join(self.raw_dir, SR_CSV_NAME))

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        SyntheticRescueExperiment, SyntheticRescueExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects for one row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        genotype = Genotype(
            perturbations=[
                released_side(str(row["n1.name"]), int(row["n1.identifier"])),
                released_side(str(row["n2.name"]), int(row["n2.identifier"])),
            ]
        )

        environment = SYNLETHDB_ENVIRONMENT

        # #903: a released SR score has no sourced definition, so it is a typed gap
        score = row["r.statistic_score"]
        phenotype = SyntheticRescuePhenotype(
            is_synthetic_rescue=True,
            synthetic_rescue_statistic_score=None,
            provenance_gaps=(
                []
                if pd.isna(score)
                else [
                    sr_score_gap(
                        float(score), str(row["r.source"]), str(row["r.pubmed_id"])
                    )
                ]
            ),
        )

        phenotype_reference = SyntheticRescuePhenotype(
            is_synthetic_rescue=False, synthetic_rescue_statistic_score=None
        )

        experiment = SyntheticRescueExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        reference = SyntheticRescueExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment,
            phenotype_reference=phenotype_reference,
        )

        publication = Publication(
            pubmed_id=str(row["r.pubmed_id"]),
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{row['r.pubmed_id']}/",
            doi=None,
            doi_url=None,
        )
        return experiment, reference, publication


def main() -> None:
    """Build and inspect both SynLethDB datasets under ``$DATA_ROOT``.

    The roots are the dev-tree directories the knowledge-graph configs read
    (``torchcell/knowledge_graphs/conf/synth_*_yeast_synth_leth_db.yaml``).
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.environ["DATA_ROOT"]

    lethality_dataset = SynthLethalityYeastSynthLethDbDataset(
        root=osp.join(DATA_ROOT, "data/torchcell/synth_lethality_yeast_synth_leth_db")
    )
    print(lethality_dataset)

    rescue_dataset = SynthRescueYeastSynthLethDbDataset(
        root=osp.join(DATA_ROOT, "data/torchcell/synth_rescue_yeast_synth_leth_db")
    )
    print(rescue_dataset)


if __name__ == "__main__":
    main()
