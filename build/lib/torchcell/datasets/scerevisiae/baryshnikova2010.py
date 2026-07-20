# torchcell/datasets/scerevisiae/baryshnikova2010
# [[torchcell.datasets.scerevisiae.baryshnikova2010]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/baryshnikova2010
# Test file: tests/torchcell/datasets/scerevisiae/test_baryshnikova2010.py
"""Baryshnikova 2010 genome-scale single-mutant fitness (SMF) dataset.

Baryshnikova A et al. 2010 (Boone/Andrews lab), "Quantitative analysis of fitness and
genetic interactions in yeast on a genome scale", Nat Methods 7(12):1017-1024
(doi:10.1038/nmeth.1534). This is the SGA-scoring methods paper that Costanzo 2016 and
Kuzmin 2018 DEFER to for the fitness/interaction formulas, but it ALSO releases a NEW
genome-scale single-mutant fitness catalog (Supplementary Data 1 / ``S1_SMF_standard``) --
the data this loader builds.

WHAT THE FILE CONTAINS (Supplementary Data 1; tab-separated; NO header; 6023 rows, 3 cols):
  - col0: SGA strain/allele id -- a plain systematic ORF for a nonessential deletion
    (e.g. YML062C), an essential-gene DAmP allele (``<ORF>_damp``, e.g. YAL001C_damp), or a
    temperature-sensitive allele (``<ORF>_tsq<NNN>``, e.g. YBR156C_tsq236).
  - col1: single-mutant fitness, WT-normalized so the fitness-distribution MODE == 1.0 (SI
    Note 1: "setting the mode of the fitness distribution to 1"). Range 0.063 .. 1.161.
  - col2: uncertainty = a BOOTSTRAP standard error of the median fitness estimate (SI Note 1:
    "Variance in single mutant fitness estimates was estimated from bootstrap sampling of the
    median"), NOT a sample SD. Range 0.0006 .. 0.1141.

COMPOSITION (6023 = 4635 + 1082 + 306; verified by suffix parse; ALL kept):
  - 4635 nonessential gene deletions (kanMX-marked array; SI Suppl. Table 2 "every array
    kanMX-marked mutant") -> SgaKanMxDeletionPerturbation.
  - 1082 essential-gene DAmP hypomorphs (``_damp``) -> SgaDampPerturbation.
  - 306 temperature-sensitive essential alleles (``_tsq``) -> SgaTsAllelePerturbation.
  The full raw id is preserved on ``strain_id`` so an allelic series is distinct: 58 genes
  carry >1 TS allele (e.g. YAL041W x4), which the fitness L1 signature separates via
  strain_id (see torchcell/verification/fitness.py ``_genotype_signature``).

REPLICATE DESIGN / n_samples (SI Suppl. Table 2, verbatim: "Fitness measurements were based
  on 80 control screens."): n_samples = 80, sample_unit = screen (the control screen is the
  independent replicate unit, as in Costanzo). The bootstrap SE is used as the ML-facing
  ``fitness_se`` as-is (UncertaintyType.bootstrap_se converts without n).

ENVIRONMENT: the genome-wide SGA final-selection growth, represented consistently with the
  sibling Costanzo 2016 SGA dataset -- Media(YEPD, solid), 30 C (the standard Boone-lab SGA
  growth temperature; the exact selection-medium recipe defers to the SGA protocol in
  Baryshnikova 2010 Methods Enzymol 470:145-179). This is a representative aligned with
  Costanzo, not re-derived from this paper's methods.

GENOTYPE / REFERENCE: each strain is one gene perturbation in the BY4741 SGA background (SI
  border control "isogenic to BY4741"). The file releases only systematic allele ids (no
  common names), so ``perturbed_gene_name`` is set to the resolved systematic ORF. Reference
  = wild-type, fitness == 1.0.

DATA SOURCE (sha256-pinned mirror): Supplementary Data 1 text export
  (``data/S1_SMF_standard_100209.txt``, sha256 c8114f88...) in the library mirror; the loader
  copies it into raw_dir and verifies the hash. The near-identical ``.xls`` export differs by
  <=0.031 -- the ``.txt`` is canonical. No live URL is a dependency.
"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path as osp
import shutil
from collections.abc import Callable
from typing import Any

import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1038/nmeth.1534"

_LIBRARY_CITATION_KEY = "baryshnikovaQuantitativeAnalysisFitness2010"
_MIRROR_REL_PATH = osp.join("data", "S1_SMF_standard_100209.txt")
_RAW_FILENAME = "baryshnikova2010_smf_standard.txt"
_SMF_SHA256 = "c8114f88c96f3b605dc5837c8958de30c34e0077558fd78f26440465a19f6b5b"

# SI Suppl. Table 2: "Fitness measurements were based on 80 control screens."
_N_SAMPLES = 80

# Row-count + composition self-checksum (verified by suffix parse of the released file).
_EXPECTED_ROWS = 6023
_EXPECTED_COMPOSITION = {"deletion": 4635, "damp": 1082, "ts": 306}


def _parse_allele(allele: str) -> tuple[str, str]:
    """Split a raw SGA allele id into (systematic-ORF token, perturbation kind)."""
    if allele.endswith("_damp"):
        return allele[: -len("_damp")], "damp"
    if "_tsq" in allele:
        return allele.split("_tsq")[0], "ts"
    return allele, "deletion"


@register_dataset
class SmfBaryshnikova2010Dataset(ExperimentDataset):
    """Baryshnikova 2010 genome-scale single-mutant fitness dataset (6023 alleles)."""

    def __init__(
        self,
        root: str = "data/torchcell/smf_baryshnikova2010",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED to resolve ORFs to current R64."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The pinned Supplementary Data 1 SMF text file."""
        return [_RAW_FILENAME]

    def download(self) -> None:
        """Copy the pinned SMF file into raw_dir and verify its sha256.

        Publisher retrieval (Springer ESM) is scriptable, but the canonical source is the
        sha256-pinned file already staged in the library mirror; the pinned file (not any
        live URL) is the source of record.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, _RAW_FILENAME)
        if not osp.exists(dest):
            data_root = os.environ["DATA_ROOT"]
            src = osp.join(
                data_root, "torchcell-library", _LIBRARY_CITATION_KEY, _MIRROR_REL_PATH
            )
            if not osp.exists(src):
                raise RuntimeError(
                    f"library mirror SMF file not found: {src}. This dataset's source is the "
                    f"sha256-pinned Supplementary Data 1 in the torchcell-library mirror."
                )
            shutil.copyfile(src, dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _SMF_SHA256:
            raise RuntimeError(
                f"{_RAW_FILENAME} sha256 mismatch: got {digest}, expected {_SMF_SHA256}"
            )
        log.info("Verified %s (sha256 %s)", dest, _SMF_SHA256)

    def _resolver(self) -> Callable[[str], str | None]:
        """Build an ORF -> current-R64-systematic-name resolver from the genome."""
        if self.genome is None:
            raise RuntimeError(
                "SmfBaryshnikova2010Dataset requires a genome for ORF resolution; "
                "inject SCerevisiaeGenome(...)"
            )
        genome = self.genome
        df = genome.gene_attribute_table
        ids = set(df["ID"])
        alias_map = genome.alias_to_systematic

        def resolve(token: str) -> str | None:
            gene = token.upper()
            if gene in ids:
                return gene
            candidates = alias_map.get(gene, [])
            if candidates and candidates[0] in ids:
                return candidates[0]
            return None

        return resolve

    def _read_smf(self) -> pd.DataFrame:
        """Read + validate the SMF file (no header; 3 columns; composition self-checksum)."""
        path = osp.join(self.raw_dir, _RAW_FILENAME)
        df = pd.read_csv(path, sep="\t", header=None, names=["allele", "fitness", "se"])
        if len(df) != _EXPECTED_ROWS:
            raise RuntimeError(
                f"SMF row-count self-checksum failed: {len(df)} != {_EXPECTED_ROWS}"
            )
        kinds = df["allele"].map(lambda a: _parse_allele(str(a))[1])
        comp = kinds.value_counts().to_dict()
        if comp != _EXPECTED_COMPOSITION:
            raise RuntimeError(
                f"SMF composition self-checksum failed: {comp} != {_EXPECTED_COMPOSITION}"
            )
        return df

    @staticmethod
    def _perturbation(
        kind: str, orf: str, strain_id: str
    ) -> SgaKanMxDeletionPerturbation | SgaDampPerturbation | SgaTsAllelePerturbation:
        """Build the SGA perturbation for one allele, preserving the raw id on strain_id."""
        if kind == "deletion":
            return SgaKanMxDeletionPerturbation(
                systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=strain_id
            )
        if kind == "damp":
            return SgaDampPerturbation(
                systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=strain_id
            )
        return SgaTsAllelePerturbation(
            systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=strain_id
        )

    def _experiment(
        self, *, orf: str, kind: str, strain_id: str, fitness: float, se: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the fitness experiment/reference/publication for one SGA allele."""
        genotype = Genotype(perturbations=[self._perturbation(kind, orf, strain_id)])
        environment = Environment(
            media=SGA_DM_SELECTION, temperature=Temperature(value=30)
        )
        phenotype = FitnessPhenotype(
            fitness=fitness,
            fitness_uncertainty=se,
            fitness_uncertainty_type=UncertaintyType.bootstrap_se,
            n_samples=_N_SAMPLES,
            sample_unit=SampleUnit.screen,
        )
        phenotype_reference = FitnessPhenotype(
            fitness=1.0, n_samples=_N_SAMPLES, sample_unit=SampleUnit.screen
        )
        reference = FitnessExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        experiment = FitnessExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")
        return experiment, reference, publication

    @post_process
    def process(self) -> None:
        """Convert each SMF row into a fitness record; write LMDB."""
        resolve = self._resolver()
        df = self._read_smf()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # A few raw allele ids repeat in the source with DIFFERENT fitness -- YDL227C (HO, the
        # neutral SGA marker locus) appears twice (1.000155 vs 1.031582). Both source rows are
        # kept as distinct records; a positional suffix on strain_id disambiguates them so the
        # L1 (strain, environment) signature stays unique (never guess-merge two measurements).
        dup_ids = set(df["allele"][df["allele"].duplicated(keep=False)].astype(str))
        occ: dict[str, int] = {}

        dropped: list[str] = []
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="baryshnikova2010"):
                raw_allele = str(row["allele"])
                orf_token, kind = _parse_allele(raw_allele)
                orf = resolve(orf_token)
                if orf is None:
                    dropped.append(raw_allele)
                    continue
                if raw_allele in dup_ids:
                    occ[raw_allele] = occ.get(raw_allele, 0) + 1
                    strain_id = f"{raw_allele}.{occ[raw_allele]}"
                else:
                    strain_id = raw_allele
                experiment, reference, publication = self._experiment(
                    orf=orf,
                    kind=kind,
                    strain_id=strain_id,
                    fitness=float(row["fitness"]),
                    se=float(row["se"]),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                idx += 1
        env.close()
        interned_env.close()
        log.info(
            "Wrote %d Baryshnikova2010 fitness experiments to LMDB (%d dropped: %s)",
            idx,
            len(dropped),
            sorted(dropped),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/smf_baryshnikova2010")
    dataset = SmfBaryshnikova2010Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
