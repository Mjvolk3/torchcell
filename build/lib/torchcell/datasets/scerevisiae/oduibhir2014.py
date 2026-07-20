# torchcell/datasets/scerevisiae/oduibhir2014
# [[torchcell.datasets.scerevisiae.oduibhir2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/oduibhir2014
# Test file: tests/torchcell/datasets/scerevisiae/test_oduibhir2014.py
"""O'Duibhir 2014 per-deletion relative growth rate -> single-mutant fitness dataset.

O'Duibhir E et al. 2014 (Holstege lab), "Cell cycle population effects in perturbation
studies", Mol Syst Biol 10:732 (doi:10.15252/msb.20145172, PMID 24952590). This is a
GROWTH-RATE study whose only new per-strain data is Supplementary Dataset S2: the relative
doubling time of each of 1312 gene-deletion strains (the paper's expression data are the
Kemmeren 2014 compendium + PCA transforms, already ingested as the Kemmeren microarray
dataset, so ONLY the fitness readout is built here).

WHAT DATASET S2 CONTAINS (born-digital, tab-separated; column descriptions from the file
README "Supplementary_data2.txt"):
  - col0 (unnamed index): systematic ORF (e.g. YAL009W).
  - ``commonName``: standard gene name.
  - ``log2relT`` = log2(doubling_time_mutant / doubling_time_wildtype). The paper's growth
    axis (Fig 1D, Fig 6C legend): "log2 (doubling time in mutant/doubling time wild-type)".
  - ``similarity`` = projection of the deletion expression profile onto the slow-growth
    signature (NOT ingested here -- it is an expression-derived covariance, not fitness).

FITNESS DIRECTION (verified against the codebase convention, NOT guessed):
  growth_rate is inversely proportional to doubling_time, so
      2^(log2relT) = doubling_mut / doubling_wt = growth_wt / growth_mut,
  and the mutant's growth RELATIVE to wild-type is its reciprocal:
      fitness = growth_mut / growth_wt = 2^(-log2relT).
  The built Costanzo 2016 SMF dataset stores fitness on exactly this convention -- WT == 1.0
  and sick, slow-growing deletions BELOW 1 (e.g. the essential-gene TS allele ded1-f144c is
  stored as 0.1137). O'Duibhir is matched to that: fitness = 2^(-log2relT), so a slow grower
  (log2relT > 0, up to +1.97 -> ~3.9x slower) lands at fitness < 1 (min ~0.255) and a faster
  grower (log2relT < 0, down to -0.24) at fitness > 1 (max ~1.18). The FitnessPhenotype
  field description reads "ko_growth_rate/wt_growth_rate" (corrected 2026-07-13 from the
  earlier inverted "wt_growth_rate/ko_growth_rate"); the stored Costanzo convention is ko/wt
  relative growth (WT=1, sick<1) and this loader matches it.

REPLICATE STRUCTURE / n_samples (Methods, "Growth rate calculations", verbatim):
  "The doubling times of the deletion strains were calculated from the slope of the
  log2(OD600) by taking the linear part of the growth curve just prior to harvest. ... The
  doubling times from two biological replicate cultures were averaged, and the ratio versus
  wild type (Supplementary Dataset S2) determined from wild-type cultures grown in parallel
  to each batch of mutants." -> n_samples = 2, sample_unit = biological_replicate. NO
  per-strain SD/SE is released in Dataset S2, so fitness_std / fitness_uncertainty /
  fitness_uncertainty_type / fitness_se are all None.

ENVIRONMENT (Methods; same cultures as the Kemmeren 2014 deletion compendium):
  "Deletion strain data were generated as ... two biological replicates harvested in early
  mid-log phase in synthetic complete medium with 2% glucose" and (for this study's growth)
  "wild-type and deletion strains were grown for two cell doublings in synthetic complete
  (SC) media containing 2% glucose." The doubling times are OD600 slopes of these same SC
  liquid cultures; the Holstege/Kemmeren standard growth temperature is 30 C (the value the
  in-repo Kemmeren 2014 loader records for the identical culture setup). -> Environment =
  SC liquid, 30 C.

GENOTYPE / REFERENCE:
  Each strain is one KanMX single-gene deletion in the S. cerevisiae nonessential deletion
  collection (Holstege compendium). genotype = one KanMxDeletionPerturbation
  (systematic_gene_name = ORF, perturbed_gene_name = commonName). Reference = the wild-type
  parental strain, fitness == 1.0. The Kemmeren compendium used BOTH MATa (BY4741) and MATa
  (BY4742) reference pools, but Dataset S2 does NOT resolve mating type per strain, so the
  representative haploid deletion background BY4741 is recorded on ``ReferenceGenome``
  (flagged: mating type is not recoverable from the released table).

DATA SOURCE (sha256-pinned mirror):
  Supplementary Dataset S2 ("data set 2.txt") extracted from the publisher ESM zip
  (msb0010-0732-sd11.zip, sha256 4843ec36...) into the library mirror; the loader copies the
  extracted file into raw_dir and verifies its sha256 (37ef19ee...). No live URL is a
  dependency -- the pinned mirror file is canonical.
"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path as osp
import pickle
import shutil
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
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    Publication,
    ReferenceGenome,
    SampleUnit,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.15252/msb.20145172"
PMID = "24952590"

# Canonical sha256-pinned source: Supplementary Dataset S2 extracted from the publisher ESM
# zip into the library mirror. The loader copies this file into raw_dir and verifies its
# hash; the pinned file (not any live URL) is the source of record.
_LIBRARY_CITATION_KEY = "oduibhirCellCyclePopulation2014"
_MIRROR_REL_PATH = osp.join(
    "data", "ds2_extract", "msb145172-sup-0011-DatasetS2", "dataset 2", "data set 2.txt"
)
_RAW_FILENAME = "oduibhir2014_datasetS2.txt"
_DATASET_S2_SHA256 = "37ef19ee249c64c0557c84870e59b2fd7a8bbaf14371fd355775e650f2a39f1c"

# "The doubling times from two biological replicate cultures were averaged" (Methods).
_N_SAMPLES = 2

# Row count declared in the file's header comment ("... 1312 deletion strains ...").
_EXPECTED_ROWS = 1312


@register_dataset
class SmfODuibhir2014Dataset(ExperimentDataset):
    """O'Duibhir 2014 single-deletion relative-growth-rate -> fitness dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/smf_oduibhir2014",
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
        """The pinned Supplementary Dataset S2 text file (single-mutant relative growth)."""
        return [_RAW_FILENAME]

    def download(self) -> None:
        """Copy the pinned Dataset S2 file into raw_dir and verify its sha256.

        Publisher-side retrieval (Wiley/EMBO ESM, PMC) is not reliably scriptable, so the
        canonical source is the sha256-pinned file already staged in the library mirror.
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
                    f"library mirror Dataset S2 not found: {src}. This dataset's source is "
                    f"the sha256-pinned Supplementary Dataset S2 in the torchcell-library "
                    f"mirror."
                )
            shutil.copyfile(src, dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _DATASET_S2_SHA256:
            raise RuntimeError(
                f"{_RAW_FILENAME} sha256 mismatch: got {digest}, "
                f"expected {_DATASET_S2_SHA256}"
            )
        log.info("Verified %s (sha256 %s)", dest, _DATASET_S2_SHA256)

    def _resolver(self) -> Callable[[str], str | None]:
        """Build an ORF -> current-R64-systematic-name resolver from the genome."""
        if self.genome is None:
            raise RuntimeError(
                "SmfODuibhir2014Dataset requires a genome for ORF resolution; "
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

    def _read_dataset_s2(self) -> pd.DataFrame:
        """Read + validate Dataset S2 (skip the leading '#...' comment line)."""
        path = osp.join(self.raw_dir, _RAW_FILENAME)
        df = pd.read_csv(path, sep="\t", skiprows=1)
        df = df.rename(columns={df.columns[0]: "orf"})
        expected_cols = ["orf", "commonName", "log2relT", "similarity"]
        if list(df.columns) != expected_cols:
            raise RuntimeError(
                f"unexpected Dataset S2 columns: {list(df.columns)} != {expected_cols}"
            )
        if len(df) != _EXPECTED_ROWS:
            raise RuntimeError(
                f"Dataset S2 row-count self-checksum failed: {len(df)} != {_EXPECTED_ROWS}"
            )
        return df

    @staticmethod
    def _experiment(
        *, dataset_name: str, orf: str, common_name: str, log2relt: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the fitness experiment/reference/publication for one deletion strain."""
        # growth_mut / growth_wt = doubling_wt / doubling_mut = 2^(-log2relT). WT == 1.0;
        # slow growers (log2relT > 0) land < 1, matching the stored Costanzo SMF convention.
        fitness = float(2.0 ** (-log2relt))

        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=common_name
                )
            ]
        )
        environment = Environment(
            media=Media(name="SC", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        # No per-strain uncertainty is released in Dataset S2 -> all uncertainty fields None.
        phenotype = FitnessPhenotype(
            fitness=fitness,
            n_samples=_N_SAMPLES,
            sample_unit=SampleUnit.biological_replicate,
        )
        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            n_samples=_N_SAMPLES,
            sample_unit=SampleUnit.biological_replicate,
        )
        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        return experiment, reference, publication

    @post_process
    def process(self) -> None:
        """Convert each Dataset S2 row into a fitness record; write LMDB."""
        resolve = self._resolver()
        df = self._read_dataset_s2()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        dropped: list[str] = []
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="oduibhir2014"):
                token = str(row["orf"])
                orf = resolve(token)
                if orf is None:
                    dropped.append(token)
                    continue
                experiment, reference, publication = self._experiment(
                    dataset_name=self.name,
                    orf=orf,
                    common_name=str(row["commonName"]),
                    log2relt=float(row["log2relT"]),
                )
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
                idx += 1
        env.close()
        log.info(
            "Wrote %d ODuibhir2014 fitness experiments to LMDB (%d ORFs dropped: %s)",
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
    root = osp.join(data_root, "data/torchcell/smf_oduibhir2014")
    dataset = SmfODuibhir2014Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
