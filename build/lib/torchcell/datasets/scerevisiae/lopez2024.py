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
  fluorescence was measured ONCE per strain. The 4805 released rows include ~190 genes
  reported twice (the strain measured on two plates); rows are AGGREGATED to one record per
  resolved ORF (metabolite L1 orf_uniqueness): ``metabolite_level`` = mean fold change,
  ``n_replicates`` = row count, ``metabolite_level_se`` = sample-SD / sqrt(n) when n >= 2
  (else ``None``).
- ``IsobutanolValidatedLopez2024Dataset`` (Table S3): the VALIDATED re-screen of the strong
  hits (66 up, FC>=2; 161 down, FC<=0.5). Re-screened in TRIPLICATE (``n_replicates = 3``);
  the released ``STD`` is the sample SD, so SE = STD / sqrt(3). YBL071W-A appears in BOTH the
  up and down blocks (contradictory) and is DROPPED entirely -> 225 records.

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
source are SYSTEMATIC ORF names, resolved to current SGD R64 ORFs via an injected genome
(collision-aware, as in Smith 2006).

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

import hashlib
import logging
import math
import os
import os.path as osp
import pickle
import re
import shutil
import statistics
from collections.abc import Callable
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
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
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

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


class _IsobutanolLopez2024Base(ExperimentDataset):
    """Shared base for the two Montaño López 2024 isobutanol-biosensor YKO datasets."""

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED for systematic-name -> ORF mapping."""
        self.genome = genome
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
            shutil.copyfile(src, dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _XLSX_SHA256:
            raise RuntimeError(
                f"{_XLSX_FILENAME} sha256 mismatch: got {digest}, expected {_XLSX_SHA256}"
            )
        log.info("Verified %s (sha256 %s)", dest, _XLSX_SHA256)

    def _require_genome(self) -> SCerevisiaeGenome:
        """Return the injected genome or raise (systematic-name resolution requires it)."""
        if self.genome is None:
            raise RuntimeError(
                f"{type(self).__name__} requires an injected SCerevisiaeGenome to resolve "
                f"systematic gene names to current R64 ORF ids; inject SCerevisiaeGenome(...)"
            )
        return self.genome

    def _resolver(self, systematic_names: list[str]) -> Callable[[str], str | None]:
        """Build a collision-aware systematic-name -> current-R64-ORF resolver.

        A name already a current R64 ID resolves to itself; an old systematic name follows
        the genome alias table to its current ID ONLY when that target is not already a
        directly-present ORF (so a dubious ORF absorbed into a verified neighbor is dropped,
        not mislabeled -- same rule as Smith 2006).
        """
        genome = self._require_genome()
        ids = set(genome.gene_attribute_table["ID"])
        alias_map = genome.alias_to_systematic
        direct_orfs = {n for n in systematic_names if n in ids}

        def resolve(name: str) -> str | None:
            if name in ids:
                return name
            if _SYSTEMATIC_RE.match(name):
                candidates = alias_map.get(name, [])
                if (
                    candidates
                    and candidates[0] in ids
                    and candidates[0] not in direct_orfs
                ):
                    return candidates[0]
            return None

        return resolve

    def _standard_name_map(self) -> dict[str, str]:
        """Map current-R64 ORF id -> standard/common gene name (systematic if unnamed)."""
        genome = self._require_genome()
        df = genome.gene_attribute_table
        out: dict[str, str] = {}
        for orf, gene in zip(df["ID"], df["gene"]):
            out[str(orf)] = str(gene) if pd.notna(gene) else str(orf)
        return out

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
        orf: str,
        standard_name: str,
        fold_change: float,
        se: float | None,
        n: int,
        environment: Environment,
    ) -> MetaboliteExperiment:
        """Build one biosensor-fold-change experiment for a single deletion strain."""
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=standard_name
                )
            ]
        )
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

    def __init__(
        self,
        root: str = "data/torchcell/isobutanol_screen_lopez2024",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; a genome is REQUIRED for systematic-name -> ORF mapping."""
        super().__init__(root, io_workers, genome, transform, pre_transform, **kwargs)

    @post_process
    def process(self) -> None:
        """Parse Table S2 into one MetaboliteExperiment per resolved ORF; write LMDB.

        The released table reports ~190 genes twice with a different fold change (the strain
        was measured on two plates). To satisfy the metabolite L1 orf_uniqueness invariant
        (exactly one record per resolved ORF), rows are AGGREGATED by resolved current-R64
        ORF: ``metabolite_level`` = mean fold change across that gene's rows; ``n_replicates``
        = row count; ``metabolite_level_se`` = sample-SD / sqrt(n) when the gene has >= 2 rows
        (each row was itself an ``n=1`` median-fluorescence measurement), else ``None``.
        """
        xlsx = osp.join(self.raw_dir, _XLSX_FILENAME)
        # header on 0-based row 1: "Gene-knockout strain" | "Fold change".
        df = pd.read_excel(xlsx, sheet_name="Table S2", header=1)
        gene_col, fc_col = "Gene-knockout strain", "Fold change"

        systematic_names = [str(v).upper().strip() for v in df[gene_col].tolist()]
        resolve = self._resolver(systematic_names)
        std_map = self._standard_name_map()
        environment = self._environment()
        pub_dump = self._publication().model_dump()

        # Aggregate fold changes by resolved ORF (multiple rows -> one record per ORF).
        by_orf: dict[str, list[float]] = {}
        n_unresolved = 0
        unresolved_examples: set[str] = set()
        for _, row in df.iterrows():
            sysname = str(row[gene_col]).upper().strip()
            orf = resolve(sysname)
            if orf is None:
                n_unresolved += 1
                unresolved_examples.add(sysname)
                continue
            by_orf.setdefault(orf, []).append(float(row[fc_col]))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_aggregated = 0
        with env.begin(write=True) as txn:
            for orf, fcs in tqdm(by_orf.items(), desc="lopez2024-S2"):
                n = len(fcs)
                mean_fc = sum(fcs) / n
                if n >= 2:
                    n_aggregated += 1
                    sd = statistics.stdev(fcs)  # sample SD across the gene's rows
                    se: float | None = sd / math.sqrt(n)
                else:
                    se = None
                # n=1 reference per replicate; the record n mirrors the aggregated row count.
                ref_dump = self._reference_dump(environment, n=n)
                experiment = self._experiment(
                    orf=orf,
                    standard_name=std_map.get(orf, orf),
                    fold_change=mean_fc,
                    se=se,
                    n=n,
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
        log.info(
            "Lopez2024 S2: wrote %d ORF experiments (%d aggregated from >=2 rows; "
            "%d rows dropped unresolved, e.g. %s)",
            idx,
            n_aggregated,
            n_unresolved,
            sorted(unresolved_examples)[:5],
        )


@register_dataset
class IsobutanolValidatedLopez2024Dataset(_IsobutanolLopez2024Base):
    """Montaño López 2024 VALIDATED isobutanol-biosensor re-screen (Table S3, triplicate)."""

    def __init__(
        self,
        root: str = "data/torchcell/isobutanol_validated_lopez2024",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; a genome is REQUIRED for systematic-name -> ORF mapping."""
        super().__init__(root, io_workers, genome, transform, pre_transform, **kwargs)

    def _read_blocks(self) -> list[tuple[str, float, float]]:
        """Read Table S3 UP (FC>=2) + DOWN (FC<=0.5) blocks -> [(gene, fc_avg, std), ...].

        CONTRADICTION DROP: YBL071W-A appears in BOTH the UP block (FC 3.469) and the DOWN
        block (FC 0.0757) of the released table -- a single strain cannot be both a strong
        up- and down-hit, so no single value can be asserted. Both of its rows are DROPPED
        entirely (not kept, not merged). FLAGGED FOR REVIEW.
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
                gene = str(row[gcol]).upper().strip()
                if gene == "YBL071W-A":  # contradictory up/down hit -> drop entirely
                    continue
                blocks.append((gene, float(row[fcol]), float(row[scol])))
        return blocks

    @post_process
    def process(self) -> None:
        """Parse Table S3 (both blocks) into one triplicate record per resolved ORF; write LMDB.

        YBL071W-A is dropped upstream (contradictory up/down hit). After resolution every
        remaining ORF must be unique (metabolite L1 orf_uniqueness); a collision would signal
        a second contradictory strain and raises rather than silently overwriting.
        """
        records = self._read_blocks()
        systematic_names = [g for g, _, _ in records]
        resolve = self._resolver(systematic_names)
        std_map = self._standard_name_map()
        environment = self._environment()
        ref_dump = self._reference_dump(environment, n=3)
        pub_dump = self._publication().model_dump()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_unresolved = 0
        unresolved_examples: set[str] = set()
        seen_orf: set[str] = set()
        with env.begin(write=True) as txn:
            for sysname, fc_avg, std in tqdm(records, desc="lopez2024-S3"):
                orf = resolve(sysname)
                if orf is None:
                    n_unresolved += 1
                    unresolved_examples.add(sysname)
                    continue
                if orf in seen_orf:
                    raise RuntimeError(
                        f"Lopez2024 S3: ORF {orf} (from {sysname}) appears twice after "
                        f"resolution -- unexpected contradiction; review the source table."
                    )
                seen_orf.add(orf)
                # Triplicate re-screen: released STD is the sample SD -> SE = STD / sqrt(3).
                se = std / math.sqrt(3)
                experiment = self._experiment(
                    orf=orf,
                    standard_name=std_map.get(orf, orf),
                    fold_change=fc_avg,
                    se=se,
                    n=3,
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
        log.info(
            "Lopez2024 S3: wrote %d validated experiments (%d dropped unresolved, e.g. %s)",
            idx,
            n_unresolved,
            sorted(unresolved_examples)[:5],
        )


def main() -> None:
    """Build/load both datasets for interactive debugging (a genome is REQUIRED)."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    screen = IsobutanolScreenLopez2024Dataset(
        root=osp.join(data_root, "data/torchcell/isobutanol_screen_lopez2024"),
        genome=genome,
    )
    print(f"screen len = {len(screen)}")
    print(screen[0])
    validated = IsobutanolValidatedLopez2024Dataset(
        root=osp.join(data_root, "data/torchcell/isobutanol_validated_lopez2024"),
        genome=genome,
    )
    print(f"validated len = {len(validated)}")
    print(validated[0])


if __name__ == "__main__":
    main()
