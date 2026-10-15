# torchcell/datasets/scerevisiae/kuzmin2018.py
# [[torchcell.datasets.scerevisiae.kuzmin2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2018.py
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2018.py
"""Kuzmin 2018 fitness and genetic-interaction datasets (single/double/triple mutants)."""

import logging
import os
import os.path as osp
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from torch_geometric.data import download_url
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    copy_verified,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.media import SGA_TM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    GenePerturbationType,
    Genotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaAllelePerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Sample size for combined (double/triple) mutant fitness measurements.
# The reported "Combined mutant fitness standard deviation" (Additional Data S1
# col. 12) is a SAMPLE SD over colony replicates. Kuzmin's SI (si1.md line 59)
# does not restate its type and defers to Baryshnikova 2010 (ref 8), whose Eq. 14
# defines it as the colony s.d. sigma_Iij over N_ij colonies -> we record it as
# sample_sd over colonies (SE = SD/sqrt(n), auto-derived).
#
# n = 4. The exact per-record colony count is NOT in the released data (12 cols,
# no count/SE column), so it is fixed by three converging lines:
#  (1) Empirical back-solve against the reported P-value (the "other provided
#      statistic"): the single-term normal model 2*Phi(-|eps|/(sd/sqrt(n))) over
#      410k digenic records matches the reported P-value median (0.358) best at
#      n=4 (0.377); n=8 overshoots badly (0.211). Spearman(p_pred,p_reported)=0.985
#      confirms this SD column drives the p-value ranking. Exact recovery is
#      precluded by the unpublished pooled-background term in the Baryshnikova
#      p-value, so this fixes the central estimate, not an exact per-record n.
#  (2) Conservative lower-end of the Baryshnikova range (typically 4/screen, 4-8).
#  (3) Consistency with Costanzo 2016 DMF (also n=4).
# NB: the "12-24 colony measurements" in si1.md line 59 is the QUERY fitness
# (col 9, bootstrap), a DIFFERENT column whose std the loader does not store.
N_SAMPLES_COMBINED_MUTANT = 4

CITATION_KEY = "kuzminSystematicAnalysisComplex2018"
RAW_TABLE_NAME = "aao1729_data_s1.tsv"
#: Additional Data S1 as every Kuzmin 2018 loader reads it; the hosted zip and the
#: library mirror's ``si/si_data/Data File S1_Raw genetic interaction dataset.tsv``
#: hold the same bytes (measured 2026-10-12). Verified before any record is read.
RAW_TABLE_SHA256 = "c82b60f98590071fcb07b74b672958a5863b58e03b78631e6235cb664ee82afb"

# --- Additional Data S4: the query strains' fitness standard deviation ------------- #
# The SD of a query strain's fitness is released in Additional Data S4 ("This file
# contains the fitness standard for single and double mutant query strains.", si1.md
# line 462), a file the hosted zip of Data S1 does not carry. It is read from the raw
# mirror, where it was deposited from the library mirror's copy (deposit_raw_mirror).
# Verbatim from si/si1.md line 59 (sha256 2ec80d05...): "The quantitative scoring
# method employed for single and double mutant fitness estimation was described
# previously (8), with the exception that bootstrapped means, instead of medians,
# across replicates were used in variance estimation and final fitness values. Each
# high-density array was screened in triplicate for a total of 6 replicates. Since the
# arrays were in 1536-colony format, there are 4 technical replicates of mutants on the
# array resulting in a total of between 12 and 24 colony measurements for each fitness
# estimate." So the released "St.dev." is a BOOTSTRAP quantity, already an SE of the
# estimator (never divided by sqrt(n)), and the colony count is the conservative lower
# end of the stated range, as kuzmin2020.py does for its Table S5.
S4_NAME = (
    "Data File S4_Fitness standard for single and double mutant query strains.xlsx"
)
S4_SHA256 = "2c9ee77dc6f0a08900158f23cecbcad41db6dc4a2555fbc7100fbca455869d51"
_S4_LIBRARY_RELPATH = f"si/si_data/{S4_NAME}"
_S4_RAW_RELPATH = f"data/{S4_NAME}"
N_SAMPLES_QUERY_STRAIN_FITNESS = 12
# Data S1 rounds the query fitness to 4 decimals; S4 carries more. A matched pair must
# agree to within the rounding (half a unit of the 4th decimal; the largest measured
# difference over the 504 matched strains is 5.0e-05).
_QUERY_FITNESS_ROUNDING_TOL = 5e-5 + 1e-9
#: A query strain Additional Data S4 lists with a blank Fitness and St.dev.: the release
#: states no uncertainty for it (19 Smf and 7 Dmf records).
S4_BLANK_GAP = ProvenanceGap(
    field="fitness_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=f"tc://raw/{CITATION_KEY}/{_S4_RAW_RELPATH}",
        citation_key=CITATION_KEY,
        sha256=S4_SHA256,
        page="the row of this query strain's tm number",
    ),
    note="Additional Data S4 lists this query strain with a blank Fitness and "
    "St.dev.; no other file of the release states its fitness uncertainty",
)


def raw_mirror_dir(data_root: str | None = None) -> str:
    """``$DATA_ROOT/torchcell-raw/kuzminSystematicAnalysisComplex2018``."""
    return osp.join(data_root or os.environ["DATA_ROOT"], "torchcell-raw", CITATION_KEY)


def deposit_raw_mirror(*, retrieved_at: str, data_root: str | None = None) -> str:
    """Copy Additional Data S4 from the library mirror into the raw mirror; write its
    ``manifest.json``.

    Idempotent by sha256: an existing deposit with the pinned hash is kept, a different
    one raises. The library mirror's own record of this file carries no upstream
    retrieval; the SI names where the files were released (the Boone-lab supplement
    page and DRYAD doi:10.5061/dryad.tt367), recorded as ``si_data_sources``.
    """
    root = data_root or os.environ["DATA_ROOT"]
    source = osp.join(root, "torchcell-library", CITATION_KEY, _S4_LIBRARY_RELPATH)
    dest = Path(raw_mirror_dir(root)) / _S4_RAW_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        verify_raw_files(str(dest.parent), {S4_NAME: S4_SHA256})
    else:
        copy_verified(source, dest, S4_SHA256)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi="10.1126/science.aao1729",
        title="Systematic analysis of complex genetic interactions",
        library_id="6582362",
        zotero_item_key="F7BPKWES",
        files=[
            ArtifactRecord(
                path=_S4_RAW_RELPATH,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=S4_SHA256,
                source=source,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.local_archive,
                    source_url=source,
                    retriever="torchcell.literature.retrieve.local_archive",
                    params={"path": source, "sha256": S4_SHA256},
                    sha256=S4_SHA256,
                    retrieved_at=retrieved_at,
                ),
            )
        ],
        si_data_sources=[
            "http://boonelab.ccbr.utoronto.ca/supplement/kuzmin2018/supplement.html",
            "https://doi.org/10.5061/dryad.tt367",
        ],
        si_expected=[
            "Additional Data S4 (Data File S4_Fitness standard for single and double "
            "mutant query strains.xlsx): deposited, read by SmfKuzmin2018Dataset and "
            "DmfKuzmin2018Dataset for the query strains' fitness SD",
            "Additional Data S1 (aao1729_data_s1.tsv): NOT deposited here; the loaders "
            "read it from the hosted zip, and the library mirror holds the same bytes "
            "(sha256 c82b60f98590071fcb07b74b672958a5863b58e03b78631e6235cb664ee82afb)",
            "the library mirror's manifest records no upstream retrieval for its copy "
            "of Data S4; the SI names the two locations in si_data_sources",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (Path(raw_mirror_dir(root)) / "manifest.json").write_text(
        manifest.model_dump_json(indent=2)
    )
    return raw_mirror_dir(root)


def _link_s4(raw_dir: str) -> None:
    """Symlink Additional Data S4 from the raw mirror into ``raw_dir`` (verified)."""
    mirror = osp.join(raw_mirror_dir(), _S4_RAW_RELPATH)
    if not osp.exists(mirror):
        raise RuntimeError(
            f"{mirror} is missing: deposit Additional Data S4 into the raw mirror with "
            "kuzmin2018.deposit_raw_mirror(), then rebuild (sha256 verified)."
        )
    link_verified(mirror, osp.join(raw_dir, S4_NAME), S4_SHA256)


def _query_strain_uncertainty(std_val: Any) -> dict[str, Any]:
    """Ontology fields for a query-strain fitness SD (Additional Data S4).

    A blank S4 SD is a typed gap (``S4_BLANK_GAP``); a released one is a bootstrap SE
    over colonies, used as-is (see ``N_SAMPLES_QUERY_STRAIN_FITNESS``).
    """
    if pd.isna(std_val):
        return {"provenance_gaps": [S4_BLANK_GAP]}
    return {
        "fitness_uncertainty": std_val,
        "fitness_uncertainty_type": UncertaintyType.bootstrap_se,
        "n_samples": N_SAMPLES_QUERY_STRAIN_FITNESS,
        "sample_unit": SampleUnit.colony,
    }


def _attach_query_strain_sd(
    df: pd.DataFrame, df_s4: pd.DataFrame, is_query: pd.Series
) -> pd.DataFrame:
    """Add ``query_strain_sd`` (Additional Data S4 "St.dev.") to the query rows.

    S4 keys a strain by its bare tm number ("tm3180"); Data S1 by the full strain id
    ("YAR002W+YDL227C_tm3180"). Every query row must find its S4 row (all 546 do);
    a missing one, or an S4 fitness that differs from Data S1's 4-decimal value by more
    than the rounding, is refused rather than joined. Non-query rows get NaN.
    """
    repeated = df_s4["Query Strain ID"][df_s4["Query Strain ID"].duplicated()]
    if len(repeated):
        raise ValueError(
            f"Additional Data S4 lists query strain(s) {sorted(repeated)} more than once"
        )
    s4 = df_s4.set_index("Query Strain ID")
    tm = df.loc[is_query, "Query strain ID"].str.rsplit("_", n=1).str[-1]
    missing = sorted(set(tm) - set(s4.index))
    if missing:
        raise ValueError(
            f"{len(missing)} query strain(s) are absent from Additional Data S4: "
            f"{missing[:10]}"
        )
    s4_fitness = tm.map(s4["Fitness"])
    s4_sd = tm.map(s4["St.dev."])
    if (s4_fitness.isna() != s4_sd.isna()).any():
        raise ValueError(
            "Additional Data S4 gives a Fitness without a St.dev. (or the reverse) "
            "for a query strain"
        )
    gap = (s4_fitness - df.loc[is_query, "Query single/double mutant fitness"]).abs()
    if (gap > _QUERY_FITNESS_ROUNDING_TOL).any():
        raise ValueError(
            "Additional Data S4 and Data S1 disagree beyond rounding on "
            f"{int((gap > _QUERY_FITNESS_ROUNDING_TOL).sum())} query strain(s) "
            f"(max |diff| {gap.max():.6f}); the SD would describe another estimate"
        )
    df = df.copy()
    df["query_strain_sd"] = float("nan")
    df.loc[is_query, "query_strain_sd"] = s4_sd
    return df


# Record kinds emitted by DmfKuzmin2018Dataset. Two physically different
# double-mutant measurements live in one raw table:
#  - digenic_array_cross: one row per digenic query x array cross ("Combined mutant
#    fitness"), the records this loader has always produced.
#  - double_mutant_query_strain: the double-mutant QUERY strain of a trigenic screen,
#    measured on its own ("Query single/double mutant fitness"). The raw table repeats
#    that value on every trigenic row the strain appears on (~500 rows), so the
#    measurement is per query strain, not per row.
# create_experiment branches on this column, NOT on "Combined mutant type": the
# digenic records must stay byte-identical to what is already served, since their
# content-addressed ids are what an incremental import matches on.
RECORD_KIND_DIGENIC_ARRAY_CROSS = "digenic_array_cross"
RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN = "double_mutant_query_strain"


def _array_perturbation_type(array_strain_id: str) -> str:
    """The perturbation class of an SGA array strain, read off its strain id.

    ``tsa`` strains are temperature-sensitive alleles and ``dma`` strains KanMX
    deletions, the two arrays of Data S1. Any other strain id is refused rather than
    guessed: nothing records its perturbation class (0 of the 501,510 Data S1 rows
    has another array strain).
    """
    if "tsa" in array_strain_id:
        return "temperature_sensitive"
    if "dma" in array_strain_id:
        return "KanMX_deletion"
    raise ValueError(
        f"array strain {array_strain_id!r} is neither a 'tsa' (temperature-sensitive "
        "allele) nor a 'dma' (KanMX deletion) strain; its perturbation type is unknown"
    )


def _reported_sd(std_val: Any) -> Any:
    """A fitness SD as stored: None for a blank cell, the released value otherwise.

    The SI defines no meaning for a blank SD cell, so a blank is stored as "no value
    reported" (None, matching the empty uncertainty fields) rather than as a float NaN
    (0 of the 501,510 Data S1 rows has a blank SD).
    """
    return None if pd.isna(std_val) else std_val


def _refuse_repeated_crosses(df_digenic: pd.DataFrame) -> None:
    """Refuse a digenic (query strain, array strain) cross listed more than once.

    Data S1 lists every digenic cross once (0 repeats among its 410,399 digenic rows).
    A repeat would be stored as two records of one cross, and nothing records which
    value is right, so it is refused instead of being stored twice or merged.
    """
    repeated = df_digenic[
        df_digenic.duplicated(subset=["Query strain ID", "Array strain ID"])
    ]
    if len(repeated):
        crosses = list(
            zip(repeated["Query strain ID"], repeated["Array strain ID"], strict=True)
        )
        raise ValueError(
            f"Kuzmin 2018 Data S1 lists {len(crosses)} digenic cross(es) more than "
            f"once (query strain, array strain): {crosses}"
        )


def _combined_mutant_uncertainty(std_val: Any) -> dict[str, Any]:
    """Ontology fields for a combined-mutant-fitness sample SD over colonies.

    Returns empty uncertainty when the SD is unreported (None/NaN); otherwise
    fitness_se is auto-derived by FitnessPhenotype as SD/sqrt(n_samples).

    A released SD of exactly 0 (4 digenic and 5 trigenic rows of Data S1) is stored
    as released, with the SE of 0 that arithmetic gives, never dropped or replaced by
    a gap (#894, #895; owner decision 2026-10-12: the uncertainty a source releases
    is stored as released). The SI defines the column only as "12. Combined mutant
    fitness standard deviation" (si/si1.md line 427) and states nothing else about
    these rows.
    """
    if pd.isna(std_val):
        return {"fitness_uncertainty": None, "fitness_uncertainty_type": None}
    return {
        "fitness_uncertainty": std_val,
        "fitness_uncertainty_type": UncertaintyType.sample_sd,
        "n_samples": N_SAMPLES_COMBINED_MUTANT,
        "sample_unit": SampleUnit.colony,
    }


def _double_mutant_query_strain_rows(
    df_trigenic: pd.DataFrame, columns: pd.Index
) -> pd.DataFrame:
    """One row per DISTINCT double-mutant query strain of the trigenic screens.

    The trigenic rows cross a double-mutant QUERY strain against a single-mutant
    array and report that query strain's own fitness in "Query single/double mutant
    fitness", constant across all of the strain's rows -- so it is deduplicated to
    one record per strain. Strains with no reported value are dropped. Columns are
    reindexed to the digenic frame's so the two kinds concatenate cleanly; the
    digenic-only columns (array perturbation type, the "no ho" query columns) are
    absent for these rows and never read for them.
    """
    rows = (
        df_trigenic[df_trigenic["Query single/double mutant fitness"].notna()]
        .drop_duplicates(subset=["Query strain ID"])
        .copy()
    )
    rows["query_perturbation_type_1"] = rows["Query allele name_1"].apply(
        lambda x: "KanMX_deletion" if "Δ" in x else "allele"
    )
    rows["query_perturbation_type_2"] = rows["Query allele name_2"].apply(
        lambda x: "KanMX_deletion" if "Δ" in x else "allele"
    )
    rows["record_kind"] = RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN
    return rows.reindex(columns=columns)


def _query_pair_perturbations(row: pd.Series) -> list[GenePerturbationType]:
    """The two perturbations of a double-mutant query strain (no array gene).

    Both carry the FULL query strain ID: the pair is one physical strain.
    """
    perturbations: list[GenePerturbationType] = []
    for idx in ("1", "2"):
        if row[f"query_perturbation_type_{idx}"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row[f"Query systematic name_{idx}"],
                    perturbed_gene_name=row[f"Query allele name_{idx}"],
                    strain_id=row["Query strain ID"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row[f"Query systematic name_{idx}"],
                    perturbed_gene_name=row[f"Query allele name_{idx}"],
                    strain_id=row["Query strain ID"],
                )
            )
    return perturbations


# Fitness
@register_dataset
class SmfKuzmin2018Dataset(ExperimentDataset):
    """Single-mutant fitness experiments from Kuzmin 2018."""

    organism = Organism.scerevisiae

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/smf_kuzmin2018",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset at ``root`` and trigger download/processing."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment model class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Data S1 (from the hosted zip) and Additional Data S4 (from the raw mirror)."""
        return [RAW_TABLE_NAME, S4_NAME]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed LMDB directory."""
        return "lmdb"  # type: ignore[return-value]  # single processed dir as str, base/PyG accept str

    def download(self) -> None:
        """Fetch Data S1 from the hosted zip unless it is in ``raw/``; link S4 in.

        Data S1 is fetched only when absent so a store whose Data S1 is already in
        ``raw/`` gains Additional Data S4 without a network call.
        """
        if not osp.exists(osp.join(self.raw_dir, RAW_TABLE_NAME)):
            path = download_url(self.url, self.raw_dir)
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
            os.remove(path)
        _link_s4(self.raw_dir)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        verify_raw_files(
            self.raw_dir, {RAW_TABLE_NAME: RAW_TABLE_SHA256, S4_NAME: S4_SHA256}
        )
        df = pd.read_csv(osp.join(self.raw_dir, RAW_TABLE_NAME), sep="\t")
        df_s4 = pd.read_excel(osp.join(self.raw_dir, S4_NAME))

        df = self.preprocess_raw(df, df_s4)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(  # type: ignore[override]  # dataset-specific signature
        self, df: pd.DataFrame, df_s4: pd.DataFrame
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows.

        The query-strain rows gain ``query_strain_sd`` from Additional Data S4
        (``_attach_query_strain_sd``).
        """
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query allele name no ho"] = (
            df["Query allele name"].str.replace("hoΔ", "").str.replace("+", "")
        )
        df["Query systematic name"] = df["Query strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query systematic name no ho"] = (
            df["Query systematic name"].str.replace("YDL227C", "").str.replace("+", "")
        )

        df["query_perturbation_type"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            _array_perturbation_type
        )
        self.phenotype_reference_std = df[
            "Combined mutant fitness standard deviation"
        ].mean()

        # array single mutants
        unique_array_allele_names = df["Array allele name"].drop_duplicates()
        df_array = df[
            df["Array allele name"].isin(unique_array_allele_names)
        ].drop_duplicates(subset=["Array allele name"])
        df_array["smf_type"] = "array_smf"
        # query single mutants, trigenic is not smf
        digenic_df = df[df["Combined mutant type"] == "digenic"]

        # Get unique 'Query allele name' and find first instances
        unique_query_allele_names = digenic_df[
            "Query allele name no ho"
        ].drop_duplicates()
        df_query = digenic_df[
            digenic_df["Query allele name no ho"].isin(unique_query_allele_names)
        ].drop_duplicates(subset=["Query allele name"])
        df_query["smf_type"] = "query_smf"
        df = pd.concat([df_array, df_query], axis=0)
        df = df.reset_index(drop=True)
        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = (
            df[~df["Query single/double mutant fitness"].isna()]
            .copy()
            .reset_index(drop=True)
        )
        return _attach_query_strain_sd(df, df_s4, df["smf_type"] == "query_smf")

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series, phenotype_reference_std: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from a data row."""
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        # genotype
        if row["smf_type"] == "query_smf":
            # Query
            if "KanMX_deletion" in row["query_perturbation_type"]:
                genotype = Genotype(
                    perturbations=[
                        SgaKanMxDeletionPerturbation(
                            systematic_gene_name=row["Query systematic name no ho"],
                            perturbed_gene_name=row["Query allele name no ho"],
                            strain_id=row["Query strain ID"],
                        )
                    ]
                )

            else:  # "allele", the only other query type preprocess_raw assigns
                genotype = Genotype(
                    perturbations=[
                        SgaAllelePerturbation(
                            systematic_gene_name=row["Query systematic name no ho"],
                            perturbed_gene_name=row["Query allele name no ho"],
                            strain_id=row["Query strain ID"],
                        )
                    ]
                )

        elif row["smf_type"] == "array_smf":
            # Array
            if "KanMX_deletion" in row["array_perturbation_type"]:
                genotype = Genotype(
                    perturbations=[
                        SgaKanMxDeletionPerturbation(
                            systematic_gene_name=row["Array systematic name"],
                            perturbed_gene_name=row["Array allele name"],
                            strain_id=row["Array strain ID"],
                        )
                    ]
                )

            # Only array has ts; _array_perturbation_type admits no third type.
            else:
                genotype = Genotype(
                    perturbations=[
                        SgaTsAllelePerturbation(
                            systematic_gene_name=row["Array systematic name"],
                            perturbed_gene_name=row["Array allele name"],
                            strain_id=row["Array strain ID"],
                        )
                    ]
                )

        # genotype
        environment = Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()
        # Phenotype
        if row["smf_type"] == "query_smf":
            smf_key = "Query single/double mutant fitness"
        elif row["smf_type"] == "array_smf":
            smf_key = "Array single mutant fitness"

        # A query single mutant's SD is released in Additional Data S4 (a bootstrap
        # SE, or a typed gap where S4 leaves it blank); an array single mutant's
        # fitness comes with no SD in this release, so it carries none.
        if row["smf_type"] == "query_smf":
            phenotype = FitnessPhenotype(
                fitness=row[smf_key],
                fitness_std=_reported_sd(row["query_strain_sd"]),
                **_query_strain_uncertainty(row["query_strain_sd"]),
            )
        else:
            phenotype = FitnessPhenotype(fitness=row[smf_key], fitness_std=None)

        # Reference noise = mean combined-mutant SD -> sample_sd over colonies.
        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            **_combined_mutant_uncertainty(phenotype_reference_std),
        )

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


@register_dataset
class DmfKuzmin2018Dataset(ExperimentDataset):
    """Double-mutant fitness experiments from Kuzmin 2018.

    Two record kinds (``record_kind``): the digenic query x array crosses, and the
    double-mutant QUERY strains of the trigenic screens, which the raw table also
    measures ("Query single/double mutant fitness"), one record per distinct strain.
    """

    organism = Organism.scerevisiae

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmf_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment model class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Data S1 (from the hosted zip) and Additional Data S4 (from the raw mirror)."""
        return [RAW_TABLE_NAME, S4_NAME]

    def download(self) -> None:
        """Fetch Data S1 from the hosted zip unless it is in ``raw/``; link S4 in.

        Data S1 is fetched only when absent so a store whose Data S1 is already in
        ``raw/`` gains Additional Data S4 without a network call.
        """
        if not osp.exists(osp.join(self.raw_dir, RAW_TABLE_NAME)):
            path = download_url(self.url, self.raw_dir)
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
            os.remove(path)
        _link_s4(self.raw_dir)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        verify_raw_files(
            self.raw_dir, {RAW_TABLE_NAME: RAW_TABLE_SHA256, S4_NAME: S4_SHA256}
        )
        df = pd.read_csv(osp.join(self.raw_dir, RAW_TABLE_NAME), sep="\t")
        df_s4 = pd.read_excel(osp.join(self.raw_dir, S4_NAME))

        df = self.preprocess_raw(df, df_s4)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(  # type: ignore[override]  # dataset-specific signature
        self, df: pd.DataFrame, df_s4: pd.DataFrame
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows.

        The query-strain rows gain ``query_strain_sd`` from Additional Data S4
        (``_attach_query_strain_sd``).
        """
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]
        # The trigenic rows' query strains are themselves double mutants, measured
        # on their own -- kept aside before the digenic filter drops those rows.
        df_trigenic = df[df["Combined mutant type"] == "trigenic"].copy()
        # Select doubles only
        df = df[df["Combined mutant type"] == "digenic"].copy()
        _refuse_repeated_crosses(df)

        df["Query allele name no ho"] = (
            df["Query allele name"].str.replace("hoΔ", "").str.replace("+", "")
        )
        df["Query systematic name"] = df["Query strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query systematic name no ho"] = (
            df["Query systematic name"].str.replace("YDL227C", "").str.replace("+", "")
        )

        df["query_perturbation_type_no_ho"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            _array_perturbation_type
        )
        # Reference noise stays the DIGENIC combined-mutant mean, as served.
        self.phenotype_reference_std = df[
            "Combined mutant fitness standard deviation"
        ].mean()
        df["record_kind"] = RECORD_KIND_DIGENIC_ARRAY_CROSS
        # Query-strain records go LAST so the digenic rows keep their positions
        # (the LMDB key is the row index).
        df = pd.concat(
            [df, _double_mutant_query_strain_rows(df_trigenic, df.columns)], axis=0
        )
        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return _attach_query_strain_sd(
            df, df_s4, df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN
        )

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series, phenotype_reference_std: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from a data row."""
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        # genotype
        perturbations: list[GenePerturbationType] = []
        if row["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN:
            # The query PAIR is the genotype; the array gene of the trigenic row this
            # value was read off is NOT part of this strain.
            perturbations = _query_pair_perturbations(row)
        else:
            # Query...
            if "KanMX_deletion" in row["query_perturbation_type_no_ho"]:
                perturbations.append(
                    SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Query systematic name no ho"],
                        perturbed_gene_name=row["Query allele name no ho"],
                        strain_id=row["Query strain ID"],
                    )
                )
            elif "allele" in row["query_perturbation_type_no_ho"]:
                perturbations.append(
                    SgaAllelePerturbation(
                        systematic_gene_name=row["Query systematic name no ho"],
                        perturbed_gene_name=row["Query allele name no ho"],
                        strain_id=row["Query strain ID"],
                    )
                )

            # Array - only array has ts
            if "temperature_sensitive" in row["array_perturbation_type"]:
                perturbations.append(
                    SgaTsAllelePerturbation(
                        systematic_gene_name=row["Array systematic name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array strain ID"],
                    )
                )
            elif "KanMX_deletion" in row["array_perturbation_type"]:
                perturbations.append(
                    SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Array systematic name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array strain ID"],
                    )
                )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 2, "Genotype must have 2 perturbations."
        # genotype
        environment = Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()
        # Phenotype
        if row["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN:
            # The SD of a query strain's fitness is released in Additional Data S4:
            # a bootstrap SE over colonies, or a typed gap where S4 leaves it blank.
            fitness = row["Query single/double mutant fitness"]
            fitness_std = _reported_sd(row["query_strain_sd"])
            uncertainty: dict[str, Any] = _query_strain_uncertainty(
                row["query_strain_sd"]
            )
        else:
            fitness = row["Combined mutant fitness"]
            fitness_std = _reported_sd(
                row["Combined mutant fitness standard deviation"]
            )
            uncertainty = _combined_mutant_uncertainty(fitness_std)
        phenotype = FitnessPhenotype(
            fitness=fitness, fitness_std=fitness_std, **uncertainty
        )

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            **_combined_mutant_uncertainty(phenotype_reference_std),
        )

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


@register_dataset
class TmfKuzmin2018Dataset(ExperimentDataset):
    """Triple-mutant fitness experiments from Kuzmin 2018."""

    organism = Organism.scerevisiae

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmf_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment model class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Additional Data S1, from the hosted zip."""
        return [RAW_TABLE_NAME]

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        verify_raw_files(self.raw_dir, {RAW_TABLE_NAME: RAW_TABLE_SHA256})
        df = pd.read_csv(osp.join(self.raw_dir, RAW_TABLE_NAME), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]
        # Select doubles only
        df = df[df["Combined mutant type"] == "trigenic"].copy()

        df["Query allele name no ho"] = (
            df["Query allele name"].str.replace("hoΔ", "").str.replace("+", "")
        )
        df["Query systematic name"] = df["Query strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query systematic name no ho"] = (
            df["Query systematic name"].str.replace("YDL227C", "").str.replace("+", "")
        )

        df["query_perturbation_type"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            _array_perturbation_type
        )
        self.phenotype_reference_std = df[
            "Combined mutant fitness standard deviation"
        ].mean()
        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series, phenotype_reference_std: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from a data row."""
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        # genotype
        perturbations: list[GenePerturbationType] = []
        # Query
        # Query 1
        if "KanMX_deletion" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Query 2
        if "KanMX_deletion" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Array - only array has ts
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 3, "Genotype must have 3 perturbations."
        # genotype
        environment = Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()
        # Phenotype based on temperature
        tmf_key = "Combined mutant fitness"
        tmf_std_key = "Combined mutant fitness standard deviation"
        tmf_std = _reported_sd(row[tmf_std_key])
        phenotype = FitnessPhenotype(
            fitness=row[tmf_key],
            fitness_std=tmf_std,
            **_combined_mutant_uncertainty(tmf_std),
        )

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            **_combined_mutant_uncertainty(phenotype_reference_std),
        )

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


# Interactions
@register_dataset
class DmiKuzmin2018Dataset(ExperimentDataset):
    """Double-mutant genetic-interaction experiments from Kuzmin 2018."""

    organism = Organism.scerevisiae

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmi_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[GeneInteractionExperiment]:
        """Return the experiment model class for this dataset."""
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Additional Data S1, from the hosted zip."""
        return [RAW_TABLE_NAME]

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        verify_raw_files(self.raw_dir, {RAW_TABLE_NAME: RAW_TABLE_SHA256})
        df = pd.read_csv(osp.join(self.raw_dir, RAW_TABLE_NAME), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmi Files...")

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]
        # Select doubles only
        df = df[df["Combined mutant type"] == "digenic"].copy()
        _refuse_repeated_crosses(df)

        df["Query allele name no ho"] = (
            df["Query allele name"].str.replace("hoΔ", "").str.replace("+", "")
        )
        df["Query systematic name"] = df["Query strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query systematic name no ho"] = (
            df["Query systematic name"].str.replace("YDL227C", "").str.replace("+", "")
        )

        df["query_perturbation_type_no_ho"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            _array_perturbation_type
        )

        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        GeneInteractionExperiment, GeneInteractionExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects from a data row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        perturbations: list[GenePerturbationType] = []
        # Query...
        if "KanMX_deletion" in row["query_perturbation_type_no_ho"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_no_ho"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"],
                    strain_id=row["Query strain ID"],
                )
            )

        # Array - only array has ts
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 2, "Genotype must have 2 perturbations."

        environment = Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        # Digenic interaction: relates two gene nodes -> an edge (not a hyperedge,
        # which is >=3 genes). GeneInteractionPhenotype defaults to "hyperedge"
        # (the trigenic case), so digenic loaders override it explicitly.
        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            gene_interaction_p_value=row["P-value"],
            graph_level="edge",
        )

        # By definition in paper interaction would be 0.
        phenotype_reference = GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level="edge"
        )

        reference = GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


@register_dataset
class TmiKuzmin2018Dataset(ExperimentDataset):
    """Triple-mutant genetic-interaction experiments from Kuzmin 2018."""

    organism = Organism.scerevisiae

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmi_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[GeneInteractionExperiment]:
        """Return the experiment model class for this dataset."""
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Additional Data S1, from the hosted zip."""
        return [RAW_TABLE_NAME]

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        verify_raw_files(self.raw_dir, {RAW_TABLE_NAME: RAW_TABLE_SHA256})
        df = pd.read_csv(osp.join(self.raw_dir, RAW_TABLE_NAME), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Tmi Files...")

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]
        # Select triples only
        df = df[df["Combined mutant type"] == "trigenic"].copy()

        df["Query allele name no ho"] = (
            df["Query allele name"].str.replace("hoΔ", "").str.replace("+", "")
        )
        df["Query systematic name"] = df["Query strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query systematic name no ho"] = (
            df["Query systematic name"].str.replace("YDL227C", "").str.replace("+", "")
        )

        df["query_perturbation_type"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            _array_perturbation_type
        )

        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        GeneInteractionExperiment, GeneInteractionExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects from a data row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        perturbations: list[GenePerturbationType] = []
        # Query 1
        if "KanMX_deletion" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Query 2
        if "KanMX_deletion" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Array
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 3, "Genotype must have 3 perturbations."

        environment = Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            gene_interaction_p_value=row["P-value"],
        )

        # By definition in paper interaction would be 0.
        phenotype_reference = GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None
        )

        reference = GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


if __name__ == "__main__":
    # Fitness
    print("Fitness")
    dataset = SmfKuzmin2018Dataset()
    print(dataset[0])
    print(len(dataset))
    dataset = DmfKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    dataset = TmfKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    print()
    print("Interactions")
    # Interactions
    dataset = DmiKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    dataset = TmiKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
