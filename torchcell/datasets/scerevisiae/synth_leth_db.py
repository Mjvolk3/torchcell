"""SynLethDB-derived yeast synthetic lethality and synthetic rescue datasets.

Gene resolution (issue #597). Every row of ``Yeast_SL.csv`` and ``Yeast_SR.csv`` names
both genes twice: a gene name (``n1.name``, ``n2.name``) and an NCBI Entrez Gene id
(``n1.identifier``, ``n2.identifier``). Each side is resolved by its Entrez id through the
RefSeq R64 annotation ``ncbi_genomic.gff`` (``GeneID`` -> ``locus_tag``), pinned by
sha256 and read from ``<genome_root>/S288C_reference_genome_R64-4-1_20230830/``. The gene
name is a cross-check: ``SCerevisiaeGenome.resolve_gene_name`` (standard name before
alias, the prime of ``IMP2'`` kept) must give the same current gene, or the row is
dropped under a named rule in ``preprocess/dropped_records.json``. A row naming one gene
on both sides is dropped too, since the release does not say what such a pair means.
After resolution no unordered ORF pair may repeat (``DuplicateOrfPairError``).

The old resolution (a name -> ORF dict in which a later gene's alias overwrote an
earlier gene's standard name, and ``rstrip("'")`` turned ``IMP2'`` into ``IMP2``) stored
872 of 14,000 SL and 119 of 6,948 SR records under the wrong ORF.
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
from collections.abc import Callable, Mapping
from typing import Any

import lmdb
import pandas as pd
import requests
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
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
    SgaKanMxDeletionPerturbation,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
    SyntheticLethalityPhenotype,
    SyntheticRescueExperiment,
    SyntheticRescueExperimentReference,
    SyntheticRescuePhenotype,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

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

#: GFF feature types whose ``Dbxref`` ``GeneID`` names a locus (``locus_tag``).
_NCBI_LOCUS_TYPES = frozenset({"gene", "pseudogene"})

#: Drop rules, in the order a row is tested against them.
RULE_ENTREZ_NOT_IN_GFF = "entrez_id_not_in_ncbi_gff"
RULE_NAME_DISAGREES = "gene_name_disagrees_with_entrez_id"
RULE_SELF_PAIR = "same_gene_on_both_sides"

_RULE_DESCRIPTIONS = {
    RULE_ENTREZ_NOT_IN_GFF: (
        "an Entrez id on the row has no gene or pseudogene feature in the pinned "
        "ncbi_genomic.gff, so the side cannot be resolved by its id; the name alone is "
        "not used in its place"
    ),
    RULE_NAME_DISAGREES: (
        "the gene name on the row does not resolve (standard name before alias, through "
        "SCerevisiaeGenome.resolve_gene_name) to the current gene its Entrez id maps to, "
        "so the two identifiers the release gives for one side contradict each other"
    ),
    RULE_SELF_PAIR: (
        "both sides resolve to one gene (the release gives the same name and the same "
        "Entrez id on both sides); the release does not say what a gene paired with "
        "itself means, so no meaning is assigned and the row is not stored"
    ),
}


class BlankPubmedIdError(ValueError):
    """A SynLethDB row with a blank or whitespace-only ``r.pubmed_id``.

    Every record cites its PMID, so a blank one is refused by name. The pinned
    ``Yeast_SL.csv`` (14,000 rows) and ``Yeast_SR.csv`` (6,948 rows) have none
    (0 empty and 0 whitespace-only cells).
    """


class EntrezGeneIdConflictError(ValueError):
    """One ``GeneID`` in the NCBI GFF carries two different ``locus_tag`` values."""


class DuplicateOrfPairError(ValueError):
    """Two kept rows resolve to the same unordered ORF pair.

    With Entrez resolution the pinned files have none; the nine the old name map
    stored (7 SL, 2 SR) were all artifacts of its alias overwrite.
    """


class MissingGenomeError(ValueError):
    """A build was started without the genome the gene cross-check needs."""


class DroppedRow(BaseModel):
    """One source row a drop rule removed, with both sides as the release gives them."""

    source_row: int
    n1_name: str
    n1_entrez: int
    n2_name: str
    n2_entrez: int
    detail: str


class DropRule(BaseModel):
    """One drop rule and the source rows it removed."""

    rule: str
    description: str
    n_records: int
    rows: list[DroppedRow]


class DropLog(BaseModel):
    """Every drop rule applied to a build, and the files the resolution read."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    entrez_source_path: str
    entrez_source_sha256: str
    rules: list[DropRule]


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
        dtype={"r.pubmed_id": str, "n1.identifier": "int64", "n2.identifier": "int64"},
    )
    blank = df["r.pubmed_id"].fillna("").str.strip().eq("")
    if blank.any():
        raise BlankPubmedIdError(
            f"{path}: {int(blank.sum())} row(s) with a blank r.pubmed_id "
            f"(first at row {int(blank.to_numpy().nonzero()[0][0])})"
        )
    return df


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


def _resolve_side(
    name: str, entrez: int, genome: SCerevisiaeGenome, entrez_to_orf: Mapping[int, str]
) -> tuple[str | None, str | None, str]:
    """``(orf, None, detail)`` when the side resolves, else ``(None, rule, detail)``."""
    orf = entrez_to_orf.get(entrez)
    if orf is None:
        return None, RULE_ENTREZ_NOT_IN_GFF, f"{name} (Entrez {entrez}): not in the GFF"
    resolution = genome.resolve_gene_name(name)
    if resolution.is_current_gene and resolution.systematic_name == orf:
        return orf, None, f"{name} (Entrez {entrez}) -> {orf}"
    return (
        None,
        RULE_NAME_DISAGREES,
        f"{name} (Entrez {entrez}) -> {orf} by id, but the name resolves "
        f"{resolution.status.value} to {resolution.systematic_name} "
        f"(candidates {resolution.candidates})",
    )


def resolve_pairs(
    df: pd.DataFrame, genome: SCerevisiaeGenome, entrez_to_orf: Mapping[int, str]
) -> pd.DataFrame:
    """Add ``n{1,2}.systematic_name``, ``drop_rule`` and ``drop_detail`` to every row.

    A side resolves to the ORF its Entrez id maps to, and only when its gene name
    resolves to that same current gene. A row is dropped by the first rule it meets:
    a side's Entrez id absent from the GFF, a side's name disagreeing with its id
    (``n1`` is tested before ``n2``), then both sides being one gene. Kept rows have
    ``drop_rule`` None.
    """
    n1_orfs: list[str | None] = []
    n2_orfs: list[str | None] = []
    rules: list[str | None] = []
    details: list[str] = []
    for n1, e1, n2, e2 in zip(
        df["n1.name"],
        df["n1.identifier"],
        df["n2.name"],
        df["n2.identifier"],
        strict=True,
    ):
        orf1, rule1, detail1 = _resolve_side(n1, int(e1), genome, entrez_to_orf)
        orf2, rule2, detail2 = _resolve_side(n2, int(e2), genome, entrez_to_orf)
        rule = rule1 or rule2
        if rule is None and orf1 == orf2:
            rule = RULE_SELF_PAIR
        n1_orfs.append(orf1)
        n2_orfs.append(orf2)
        rules.append(rule)
        details.append(f"n1 {detail1}; n2 {detail2}")
    out = df.copy()
    out["n1.systematic_name"] = n1_orfs
    out["n2.systematic_name"] = n2_orfs
    out["drop_rule"] = rules
    out["drop_detail"] = details
    return out


def drop_log(dataset_name: str, resolved: pd.DataFrame, gff_path: str) -> DropLog:
    """Build the ledger of a resolved frame (from :func:`resolve_pairs`)."""
    rules = []
    for rule in (RULE_ENTREZ_NOT_IN_GFF, RULE_NAME_DISAGREES, RULE_SELF_PAIR):
        hit = resolved[resolved["drop_rule"] == rule]
        rules.append(
            DropRule(
                rule=rule,
                description=_RULE_DESCRIPTIONS[rule],
                n_records=len(hit),
                rows=[
                    DroppedRow(
                        source_row=source_row,
                        n1_name=str(row["n1.name"]),
                        n1_entrez=int(row["n1.identifier"]),
                        n2_name=str(row["n2.name"]),
                        n2_entrez=int(row["n2.identifier"]),
                        detail=str(row["drop_detail"]),
                    )
                    for source_row, row in zip(
                        hit.index.tolist(), hit.to_dict("records"), strict=True
                    )
                ],
            )
        )
    kept = int(resolved["drop_rule"].isna().sum())
    return DropLog(
        dataset=dataset_name,
        source_records=len(resolved),
        kept_records=kept,
        dropped_records=len(resolved) - kept,
        entrez_source_path=gff_path,
        entrez_source_sha256=NCBI_GFF_SHA256,
        rules=rules,
    )


def refuse_duplicate_pairs(kept: pd.DataFrame) -> None:
    """Raise ``DuplicateOrfPairError`` when two kept rows name one unordered ORF pair."""
    pairs = [
        frozenset((a, b))
        for a, b in zip(
            kept["n1.systematic_name"], kept["n2.systematic_name"], strict=True
        )
    ]
    repeated = {pair for pair, n in Counter(pairs).items() if n > 1}
    if repeated:
        rows = [
            int(index)
            for index, pair in zip(kept.index, pairs, strict=True)
            if pair in repeated
        ]
        raise DuplicateOrfPairError(
            f"{len(repeated)} ORF pair(s) repeat across kept rows {rows}: "
            f"{sorted(sorted(p) for p in repeated)}"
        )


def _write_synlethdb_lmdb(
    dataset: "SynthLethalityYeastSynthLethDbDataset | SynthRescueYeastSynthLethDbDataset",
    csv_path: str,
) -> None:
    """Resolve, ledger and write one SynLethDB file (shared by both datasets)."""
    genome = _require_genome(dataset.genome)
    df = _read_synlethdb_csv(csv_path)
    resolved = dataset.preprocess_raw(df)
    os.makedirs(dataset.processed_dir, exist_ok=True)
    os.makedirs(dataset.preprocess_dir, exist_ok=True)
    ledger = drop_log(dataset.name, resolved, ncbi_gff_path(genome))
    with open(osp.join(dataset.preprocess_dir, "dropped_records.json"), "w") as f:
        f.write(ledger.model_dump_json(indent=2))
    kept = resolved[resolved["drop_rule"].isna()]
    refuse_duplicate_pairs(kept)
    if sum(rule.n_records for rule in ledger.rules) != ledger.dropped_records:
        raise RuntimeError("SynLethDB drop accounting does not add up")

    env = lmdb.open(osp.join(dataset.processed_dir, "lmdb"), map_size=int(1e12))
    with env.begin(write=True) as txn:
        for key, (_, row) in enumerate(tqdm(kept.iterrows(), total=len(kept))):
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
        "%s: wrote %d of %d rows (%s)",
        dataset.name,
        ledger.kept_records,
        ledger.source_records,
        ", ".join(f"{r.rule} {r.n_records}" for r in ledger.rules),
    )


def _require_genome(genome: SCerevisiaeGenome | None) -> SCerevisiaeGenome:
    if genome is None:
        raise MissingGenomeError(
            "SynLethDB gene resolution needs the SCerevisiaeGenome passed as genome="
        )
    return genome


def _preprocess(genome: SCerevisiaeGenome | None, df: pd.DataFrame) -> pd.DataFrame:
    resolved_genome = _require_genome(genome)
    return resolve_pairs(
        df, resolved_genome, load_entrez_to_orf(ncbi_gff_path(resolved_genome))
    )


@register_dataset
class SynthLethalityYeastSynthLethDbDataset(ExperimentDataset):
    """Yeast synthetic lethality gene-pair experiments from SynLethDB."""

    def __init__(
        self,
        root: str = "data/torchcell/syn_leth_db_yeast",
        genome: SCerevisiaeGenome | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Keep the genome for a build, then release it once the dataset is ready."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform)
        del self.genome

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
        """Resolve both sides of every row by Entrez id (see :func:`resolve_pairs`)."""
        return _preprocess(self.genome, df)

    @post_process
    def process(self) -> None:
        """Verify the raw CSV, resolve and ledger its rows, and write the LMDB."""
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
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        genotype = Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n1.systematic_name"],
                    perturbed_gene_name=row["n1.name"],
                    strain_id="S288C",
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n2.systematic_name"],
                    perturbed_gene_name=row["n2.name"],
                    strain_id="S288C",
                ),
            ]
        )

        environment = Environment(
            media=Media(name="YEPD", state="solid", is_synthetic=False),
            temperature=Temperature(value=30),
        )

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

    def __init__(
        self,
        root: str = "data/torchcell/syn_rescue_db_yeast",
        genome: SCerevisiaeGenome | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Keep the genome for a build, then release it once the dataset is ready."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform)
        del self.genome

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
        """Resolve both sides of every row by Entrez id (see :func:`resolve_pairs`)."""
        return _preprocess(self.genome, df)

    @post_process
    def process(self) -> None:
        """Verify the raw CSV, resolve and ledger its rows, and write the LMDB."""
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
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        genotype = Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n1.systematic_name"],
                    perturbed_gene_name=row["n1.name"],
                    strain_id="S288C",
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n2.systematic_name"],
                    perturbed_gene_name=row["n2.name"],
                    strain_id="S288C",
                ),
            ]
        )

        environment = Environment(
            media=Media(name="YEPD", state="solid", is_synthetic=False),
            temperature=Temperature(value=30),
        )

        phenotype = SyntheticRescuePhenotype(
            is_synthetic_rescue=True,
            synthetic_rescue_statistic_score=(
                float(row["r.statistic_score"])
                if pd.notna(row["r.statistic_score"])
                else None
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

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )

    lethality_dataset = SynthLethalityYeastSynthLethDbDataset(
        root=osp.join(DATA_ROOT, "data/torchcell/synth_lethality_yeast_synth_leth_db"),
        genome=genome,
    )
    print(lethality_dataset)

    rescue_dataset = SynthRescueYeastSynthLethDbDataset(
        root=osp.join(DATA_ROOT, "data/torchcell/synth_rescue_yeast_synth_leth_db"),
        genome=genome,
    )
    print(rescue_dataset)


if __name__ == "__main__":
    main()
