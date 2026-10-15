# torchcell/datasets/scerevisiae/lopez2024
# [[torchcell.datasets.scerevisiae.lopez2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/lopez2024
# Test file: tests/torchcell/datasets/scerevisiae/test_lopez2024.py
"""Montaño López 2024 isobutanol-biosensor YKO screen (biosensor-fluorescence proxy).

José de Jesús Montaño López's 2024 Princeton dissertation ("Systems metabolic
engineering of isobutanol production in Saccharomyces cerevisiae") integrated a
genetically-encoded isobutanol/branched-chain-amino-acid biosensor -- yeast-enhanced GFP
under the Leu3p-regulated ``LEU1`` promoter, which reports endogenous alpha-ketoisovalerate/
isobutanol-pathway flux -- into EVERY strain of the BY4741 (``ura3Δ0``) yeast gene-knockout
(YKO) collection, then read each strain's median GFP fluorescence by flow cytometry. The
biosensor is a CONSTANT reporter background present in every strain (documented on the
``ReferenceGenome``; it is NOT modeled as a per-record perturbation).

Two screens, two datasets (separate roots so the same gene+environment does not collide on
L1 uniqueness across the two):

- ``IsobutanolScreenLopez2024Dataset`` (Table S2): the FIRST genome-wide screen. Median
  fluorescence was measured ONCE per strain. Every released row is ONE record
  (``n_replicates = 1``, ``metabolite_level_se = None``): the 4805 rows include 186
  genes reported twice (the strain measured on two plates), and those are two records of
  one deletion, two plate measurements, not a replicate pair to average (2026.10.12,
  #1045; the previous build averaged them into 4554 records). The metabolite verifier
  runs this store with ``allow_duplicate_orfs``.
- ``IsobutanolValidatedLopez2024Dataset`` (Table S3): the VALIDATED re-screen of the strong
  hits (66 up, FC>=2; 161 down, FC<=0.5). Re-screened in TRIPLICATE (``n_replicates = 3``);
  the released ``STD`` is the sample SD, so SE = STD / sqrt(3). YBL071W-A appears in BOTH the
  up and down blocks (FC 3.469 and 0.0757); both rows are stored as released and the
  contradiction is flagged in the dataset note -> 227 records (the previous build dropped
  both rows).

FOLD CHANGE (verbatim, dissertation Methods, sha256 525e03b4...): "the ratio of the median
fluorescence value of the deletion strain over the median fluorescence value of the
wild-type control in the same plate." Hence the reference (WT control) level is FC = 1.0.

ENVIRONMENT (dissertation Methods): synthetic complete (SC) LIQUID medium; aerobic; NO
inducer (the biosensor reports endogenous pathway flux). Temperature is not stated
explicitly; 30 C is the standard yeast growth temperature and is recorded as such (flagged
standard-not-explicit).

Maps to ``MetabolitePhenotype`` (WS4), following Cachera 2023: the readout is a
biosensor-fluorescence PROXY for isobutanol, stored as
``metabolite_level = {"isobutanol": fold_change}``. ``MetabolitePhenotype.measurement_type``
is a free-form string (no enum), set here to ``biosensor_gfp_fluorescence_fold_change`` to
document exactly what the number is (a median-GFP fold change vs the same-plate WT control,
NOT an absolute abundance). ``target_metabolite_ids`` is left ``None`` (isobutanol is a
heterologous-flux product, not modeled against a Yeast9 ``s_NNNN`` id here). Gene ids in the
source are SYSTEMATIC ORF names and are stored AS RELEASED (upper-cased), as both
``systematic_gene_name`` and ``perturbed_gene_name``; no genome is consulted (#1045,
owner rule 2026-10-12). A released name the current annotation renames (e.g. YAL043C-A)
or no longer carries (e.g. YAR037W) keeps its own name and its record. One Table S2 row,
``YLR287-A``, is not ORF-shaped at all (the release dropped the strand letter); it is
stored on ``ReleasedIdentifierPerturbation`` under that name verbatim (kind ``deletion``,
marker KanMX), so every released row is a record and
``preprocess/dropped_records.json`` lists none.

PROVENANCE / PUBLICATION NOTE: the genome-wide biosensor YKO SCREEN DATA is unpublished
except in the 2024 dissertation, which has no discoverable DOI/PMID (checked Crossref +
Princeton DataSpace + catalog). ``Publication`` requires a resolvable identifier, so the
citation points to the peer-reviewed paper by the same authors/lab that describes the exact
biosensor construct + methodology used in the screen: Montaño-López, Duran & Avalos,
"Biosensor for branched-chain amino acid metabolism in yeast and applications in isobutanol
and isopentanol production," Nature Communications 2022 (DOI 10.1038/s41467-021-27852-x,
PMID 35022416). The DATA source is the sha256-pinned dissertation supplementary tables in
the library mirror. Flagged for review.

DATA SOURCE: ``supplementary_tables.xlsx`` in the library mirror
(``lopezSystemsMetabolicEngineering2024/data/``), sha256-pinned below. Sheet ``Table S2`` =
first screen (header on 0-based row 1; cols ``Gene-knockout strain`` | ``Fold change``);
sheet ``Table S3`` = validated re-screen (header on 0-based row 2; a FC>=2 UP block in cols
A-C and a FC<=0.5 DOWN block in cols E-G, each ``Gene knockout-strain`` | ``FC (average)`` |
``STD``).
"""

import logging
import math
import os
import os.path as osp
import pickle
import re
from collections.abc import Callable
from typing import Any

import lmdb
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    copy_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Peer-reviewed anchor for the biosensor construct/methodology (see module docstring note).
DOI = "10.1038/s41467-021-27852-x"
PMID = "35022416"

# Canonical DATA source: sha256-pinned dissertation supplementary tables in the library
# mirror. The mirror is the reproducible source (dissertation SI is not scriptable).
_LIBRARY_CITATION_KEY = "lopezSystemsMetabolicEngineering2024"
_XLSX_FILENAME = "supplementary_tables.xlsx"
_XLSX_SHA256 = "f97cf13c2d40c2a0a374475001dcf4dc8493429314661c6960d5f019ce23889b"

# MetabolitePhenotype.measurement_type is a free-form string (no enum member exists for a
# biosensor fold change); this documents exactly what the number is.
MEASUREMENT_TYPE = "biosensor_gfp_fluorescence_fold_change"
TARGET_METABOLITE = "isobutanol"

_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


RETENTION_RULE = "none"
RETENTION_DESCRIPTION = (
    "every released row is a record under its released strain name, upper-cased: an "
    "ORF-shaped name on KanMxDeletionPerturbation with both names the ORF, any other "
    "name (YLR287-A: the release dropped the strand letter) on "
    "ReleasedIdentifierPerturbation verbatim; no genome is consulted and no row is dropped"
)


class DropRule(BaseModel):
    """One rule that leaves released rows without a record, and the rows it names."""

    rule: str
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """The build's accounting: every released row is either a record or listed here."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


def _write_drop_log(
    preprocess_dir: str, dataset: str, source: int, kept: int, released: list[str]
) -> None:
    """Write ``preprocess/dropped_records.json``: nothing is dropped; the names stored
    on the released-identifier leaf are listed under the retention rule.
    """
    drop_log = DropLog(
        dataset=dataset,
        source_records=source,
        kept_records=kept,
        dropped_records=source - kept,
        rules=[
            DropRule(
                rule=RETENTION_RULE,
                description=RETENTION_DESCRIPTION,
                n_records=0,
                items=sorted(released),
            )
        ],
    )
    if drop_log.dropped_records != 0:
        raise RuntimeError(
            f"drop accounting mismatch: {drop_log.dropped_records} rows without a record"
        )
    os.makedirs(preprocess_dir, exist_ok=True)
    with open(osp.join(preprocess_dir, "dropped_records.json"), "w") as handle:
        handle.write(drop_log.model_dump_json(indent=2))


class _IsobutanolLopez2024Base(ExperimentDataset):
    """Shared base for the two Montaño López 2024 isobutanol-biosensor YKO datasets."""

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: every strain is stored as released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The mirrored supplementary tables .xlsx (both screens share it)."""
        return [_XLSX_FILENAME]

    def download(self) -> None:
        """Copy the pinned library .xlsx into raw_dir and verify its sha256.

        The canonical DATA source is the sha256-pinned supplementary tables in the local
        library mirror (dissertation SI; not reliably scriptable from any live URL).
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, _XLSX_FILENAME)
        if not osp.exists(dest):
            data_root = os.environ["DATA_ROOT"]
            src = osp.join(
                data_root,
                "torchcell-library",
                _LIBRARY_CITATION_KEY,
                "data",
                _XLSX_FILENAME,
            )
            if not osp.exists(src):
                raise RuntimeError(
                    f"library mirror data file not found: {src}. This dataset's source is "
                    f"the sha256-pinned supplementary tables in the torchcell-library mirror."
                )
            copy_verified(src, dest, _XLSX_SHA256)
            log.info("Verified %s (sha256 %s)", dest, _XLSX_SHA256)

    @staticmethod
    def _released_deletion(
        name: str,
    ) -> KanMxDeletionPerturbation | ReleasedIdentifierPerturbation:
        """The deletion under the released strain name, upper-cased (#1045).

        No genome is consulted. An ORF-shaped name goes on the placed KanMX leaf with
        both names the ORF; the one Table S2 cell that is not (``YLR287-A``) goes on the
        released-identifier leaf verbatim.
        """
        upper = name.upper().strip()
        if _SYSTEMATIC_RE.match(upper):
            return KanMxDeletionPerturbation(
                systematic_gene_name=upper, perturbed_gene_name=upper
            )
        return ReleasedIdentifierPerturbation(
            systematic_gene_name=upper,
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name=upper,
            identifier_kind="Table S2 strain name, upper-cased",
            released_kind=ReleasedKind.deletion,
            cassette="KanMX",
        )

    def _environment(self) -> Environment:
        """Aerobic SC-liquid medium; 30 C (standard yeast temp, not explicit in the source)."""
        return Environment(
            media=Media(name="SC", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30.0),
            aerobicity="aerobic",
        )

    def _publication(self) -> Publication:
        """Peer-reviewed biosensor-methodology anchor (see module docstring note)."""
        return Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )

    def _reference_dump(self, environment: Environment, n: int) -> dict[str, Any]:
        """Build the WT-control reference (FC = 1.0, same-plate control) as a dict."""
        phenotype_reference = MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: 1.0},
            metabolite_level_se=None,
            n_replicates={TARGET_METABOLITE: n},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        return reference.model_dump()

    def _experiment(
        self,
        *,
        name: str,
        fold_change: float,
        se: float | None,
        n: int,
        environment: Environment,
    ) -> MetaboliteExperiment:
        """Build one biosensor-fold-change experiment for a single deletion strain."""
        genotype = Genotype(perturbations=[self._released_deletion(name)])
        phenotype = MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: fold_change},
            metabolite_level_se=(None if se is None else {TARGET_METABOLITE: se}),
            n_replicates={TARGET_METABOLITE: n},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        )
        return MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for these datasets."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for these datasets."""
        raise NotImplementedError


@register_dataset
class IsobutanolScreenLopez2024Dataset(_IsobutanolLopez2024Base):
    """Montaño López 2024 FIRST genome-wide isobutanol-biosensor YKO screen (Table S2, n=1)."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/isobutanol_screen_lopez2024",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: every strain is stored as released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @post_process
    def process(self) -> None:
        """Parse Table S2 into one MetaboliteExperiment per released row; write LMDB.

        Every row is a record (``n_replicates = 1``, no SE): a gene the table reports on
        two plates is two records of one deletion, told apart by their values, not an
        average. The one row whose strain name is not ORF-shaped (``YLR287-A``) is
        stored on the released-identifier leaf (#1045).
        """
        verify_raw_files(self.raw_dir, {_XLSX_FILENAME: _XLSX_SHA256})
        xlsx = osp.join(self.raw_dir, _XLSX_FILENAME)
        # header on 0-based row 1: "Gene-knockout strain" | "Fold change".
        df = pd.read_excel(xlsx, sheet_name="Table S2", header=1)
        gene_col, fc_col = "Gene-knockout strain", "Fold change"

        environment = self._environment()
        pub_dump = self._publication().model_dump()
        ref_dump = self._reference_dump(environment, n=1)

        rows: list[tuple[str, float]] = []
        released_names: list[str] = []
        for _, row in df.iterrows():
            released = str(row[gene_col]).strip().upper()
            if not _SYSTEMATIC_RE.match(released):
                released_names.append(released)
            rows.append((released, float(row[fc_col])))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for name, fold_change in tqdm(rows, desc="lopez2024-S2"):
                experiment = self._experiment(
                    name=name,
                    fold_change=fold_change,
                    se=None,
                    n=1,
                    environment=environment,
                )
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": ref_dump,
                            "publication": pub_dump,
                        }
                    ),
                )
                idx += 1
        env.close()
        _write_drop_log(self.preprocess_dir, self.name, len(df), idx, released_names)
        log.info(
            "Lopez2024 S2: wrote %d row experiments as released (%d on the "
            "released-identifier leaf: %s)",
            idx,
            len(released_names),
            sorted(released_names),
        )


@register_dataset
class IsobutanolValidatedLopez2024Dataset(_IsobutanolLopez2024Base):
    """Montaño López 2024 VALIDATED isobutanol-biosensor re-screen (Table S3, triplicate)."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/isobutanol_validated_lopez2024",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: every strain is stored as released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    def _read_blocks(self) -> list[tuple[str, float, float]]:
        """Read Table S3 UP (FC>=2) + DOWN (FC<=0.5) blocks -> [(gene, fc_avg, std), ...].

        YBL071W-A appears in BOTH the UP block (FC 3.469) and the DOWN block (FC 0.0757)
        of the released table. Both rows are stored as released (2026.10.12, #1045); the
        contradiction is flagged in the dataset note, not resolved by dropping either.
        """
        xlsx = osp.join(self.raw_dir, _XLSX_FILENAME)
        # header on 0-based row 2; UP block cols A-C, DOWN block cols E-G.
        df = pd.read_excel(xlsx, sheet_name="Table S3", header=2)
        blocks: list[tuple[str, float, float]] = []
        for gcol, fcol, scol in (
            ("Gene knockout-strain", "FC (average)", "STD"),
            ("Gene knockout-strain.1", "FC (average).1", "STD.1"),
        ):
            sub = df[[gcol, fcol, scol]].dropna(subset=[gcol])
            for _, row in sub.iterrows():
                gene = str(row[gcol]).strip()
                blocks.append((gene, float(row[fcol]), float(row[scol])))
        return blocks

    @post_process
    def process(self) -> None:
        """Parse Table S3 (both blocks) into one triplicate record per released row; write LMDB.

        Every row is a record under its released ORF, including both YBL071W-A rows; a
        name that is not ORF-shaped goes on the released-identifier leaf (0 of 227 rows).
        """
        verify_raw_files(self.raw_dir, {_XLSX_FILENAME: _XLSX_SHA256})
        records = self._read_blocks()
        environment = self._environment()
        ref_dump = self._reference_dump(environment, n=3)
        pub_dump = self._publication().model_dump()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        released_names: list[str] = []
        with env.begin(write=True) as txn:
            for released, fc_avg, std in tqdm(records, desc="lopez2024-S3"):
                name = released.strip().upper()
                if not _SYSTEMATIC_RE.match(name):
                    released_names.append(name)
                # Triplicate re-screen: released STD is the sample SD -> SE = STD / sqrt(3).
                se = std / math.sqrt(3)
                experiment = self._experiment(
                    name=name, fold_change=fc_avg, se=se, n=3, environment=environment
                )
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": ref_dump,
                            "publication": pub_dump,
                        }
                    ),
                )
                idx += 1
        env.close()
        _write_drop_log(
            self.preprocess_dir, self.name, len(records), idx, released_names
        )
        log.info(
            "Lopez2024 S3: wrote %d validated experiments as released (%d on the "
            "released-identifier leaf: %s)",
            idx,
            len(released_names),
            sorted(released_names),
        )


def main() -> None:
    """Build/load both datasets for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    screen = IsobutanolScreenLopez2024Dataset(
        root=osp.join(data_root, "data/torchcell/isobutanol_screen_lopez2024")
    )
    print(f"screen len = {len(screen)}")
    print(screen[0])
    validated = IsobutanolValidatedLopez2024Dataset(
        root=osp.join(data_root, "data/torchcell/isobutanol_validated_lopez2024")
    )
    print(f"validated len = {len(validated)}")
    print(validated[0])


if __name__ == "__main__":
    main()
