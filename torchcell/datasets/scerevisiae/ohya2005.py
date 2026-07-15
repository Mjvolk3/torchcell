"""CalMorph morphology dataset for S. cerevisiae haploid deletions (Ohya 2005, SCMD).

Ohya et al. 2005, "High-dimensional and large-scale phenotyping of yeast mutants",
Proc Natl Acad Sci USA 102:19015-19020 (doi:10.1073/pnas.0509436102; PMID 16365294).
CalMorph high-dimensional single-cell morphology of the haploid ``BY4741`` non-essential
single-gene deletion collection (kanMX4 marker). This is the haploid full-knockout screen;
its diploid essential-gene dosage counterpart is
:class:`~torchcell.datasets.scerevisiae.ohnuki2018.ScmdOhnuki2018Dataset` and its
drug-hypersensitive quadruple-deletion counterpart is
:class:`~torchcell.datasets.scerevisiae.ohnuki2022.ScmdOhnuki2022Dataset`.

Values are RAW per-strain CalMorph population averages, one 501-length vector per strain:
281 base parameters (``CALMORPH_LABELS``: 220 mean + 61 ratio) + 220 coefficient-of-
variation / noise parameters (``CALMORPH_STATISTICS``, prefixed CCV/ACV/DCV/TCV). The
122-row wildtype matrix (122 independent ``his3`` WT replicate averages) is aggregated
per-feature into a single mean-WT reference phenotype.

DATA PROVENANCE -- the 501-trait CalMorph matrices are Ohya 2005's OWN published data,
distributed via the SCMD (Saccharomyces cerevisiae Morphological Database) portal. The
later Suzuki et al. 2018 "Global study of holistic morphological effectors" (BMC Genomics
19:149; PMID 29458326; doi 10.1186/s12864-018-4526-z) merely REUSED this same dataset (its
reference [21]); it did not generate the values. Hence the publication recorded per record
is Ohya 2005.

SOURCING / SHA256 PINS (loader reads the library-mirror ``data/`` files and verifies both
hashes, never the live URL):
- ``mt4718data.tsv`` (mutant matrix, 4718 strains x 501 features; ID column ``ORF``)
  sha256 ``c4ba1e84b4ea6273f0162ef9230e15634933c8c0c4910dd7546a21c6293e0fc0``.
- ``wt122data.tsv`` (122 WT ``his3`` replicate averages; ID column ``NAME``; feature
  columns byte-compatible with the mutant file) sha256
  ``ab2c31b5150b2a33c15b5d22f1bef8687719975223559a740ea233c1f67b27c3``.
- Historical retrieval URLs (SCMD portal): ``http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/
  download.php?path=mt4718data.tsv`` and ``...=wt122data.tsv`` (Box mirror fallbacks in
  the manifest). See ``$DATA_ROOT/torchcell-library/
  ohyaHighdimensionalLargescalePhenotyping2005a/manifest.json``.

ENVIRONMENT -- Ohya 2005 Methods, verbatim: "Each strain was grown in yeast extract/
peptone/dextrose medium, and logarithmic-phase cells were fixed." So media = YPD, state =
liquid (logarithmic-phase liquid culture, then fixed for imaging). Temperature is NOT
stated in Ohya 2005; it is resolved via the Ohya-lab CalMorph standard (25 C), corroborated
by Suzuki 2018 (which reuses this data, "grown at 25 C") and by the same-lab Ohnuki
2018/2022 CalMorph loaders (both 25 C, sourced verbatim from their Methods). Recording
25 C here follows the deferral convention (defer method detail to the cited/related mirrored
paper); FLAG: pin the verbatim Ohya-2005 temperature once that paper is mirrored.

ORFs are 2005-annotation SGD systematic names, reconciled to the current R64-4-1 universe:
(1) four legacy names that SGD renamed/merged with NO measured twin are remapped to their
current systematic name (``_LEGACY_ORF_RENAMES``, sourced from the R64 GFF ``Alias`` field);
(2) all remaining names are validated against the S288C R64 gene universe, and any that do
not resolve are logged + dropped -- 17 ORFs retired from SGD since 2005 plus 6 legacy names
whose R64 target is an already-measured gene (an SGD merge; dropping the legacy strain avoids
duplicating that gene's morphology). Rows with any missing CalMorph value are dropped whole
-- values are NEVER imputed. Net built records: 4718 raw -> 4695 (4 renamed in place, 23
dropped).
"""

# torchcell/datasets/scerevisiae/ohya2005
# [[torchcell.datasets.scerevisiae.ohya2005]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/ohya2005
# Test file: tests/torchcell/datasets/scerevisiae/test_Ohya2005.py

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
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# CV parameters are prefixed CCV/ACV/DCV/TCV; everything else is a base parameter.
_CV_PREFIXES = ("CCV", "ACV", "DCV", "TCV")

# sha256-pinned raw matrices in the library mirror ``data/`` directory.
_RAW_FILES: dict[str, dict[str, str]] = {
    "mt4718data.tsv": {
        "sha256": "c4ba1e84b4ea6273f0162ef9230e15634933c8c0c4910dd7546a21c6293e0fc0",
        "id_column": "ORF",
    },
    "wt122data.tsv": {
        "sha256": "ab2c31b5150b2a33c15b5d22f1bef8687719975223559a740ea233c1f67b27c3",
        "id_column": "NAME",
    },
}

# Library mirror holding the sha256-pinned raw matrices (relative to DATA_ROOT).
_MIRROR_DIR = "torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/data"

# Legacy 2005-annotation ORFs that SGD has since renamed/merged, mapped to their current
# R64-4-1 systematic name. Sourced from the R64-4-1 GFF, where each legacy name is listed
# as an ``Alias`` of the target feature. ONLY the four whose target is NOT itself measured
# elsewhere in this screen are remapped here (a clean 1:1 rename that recovers a real
# strain). The other six aliases (YDL038C->YDL039C, YDL134C-A->YDL133C-A, YER108C->YER109C,
# YIL168W->YIL167W, YIR044C->YIR043C, YML033W->YML034W) COLLIDE with a gene that already has
# its own strain record -- an SGD merge of two 2005 ORFs -- so remapping them would duplicate
# that gene's morphology; those legacy strains are instead dropped (the canonical target
# strain is retained). See module docstring.
_LEGACY_ORF_RENAMES: dict[str, str] = {
    "YGR272C": "YGR271C-A",  # EFG1
    "YIL015C-A": "YIL014C-A",
    "YLR391W": "YLR390W-A",  # CCW14
    "YMR158C-B": "YMR158C-A",
}

# S288C R64 gene universe (systematic ORF + RNA-coding names) for R64 validation.
_SGD_GENE_FASTAS = (
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "orf_coding_all_R64-4-1_20230830.fasta",
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "rna_coding_R64-4-1_20230830.fasta",
)


def _load_sgd_genes(data_root: str) -> set[str]:
    """S288C R64 systematic-name universe from the ORF + RNA-coding FASTA headers."""
    genes: set[str] = set()
    for rel in _SGD_GENE_FASTAS:
        with open(osp.join(data_root, rel)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


@register_dataset
class ScmdOhya2005Dataset(ExperimentDataset):
    """CalMorph morphology for haploid non-essential single-gene deletions (Ohya 2005).

    Haploid ``BY4741`` full-deletion CalMorph screen; the haploid full-knockout base of the
    SCMD CalMorph family (Ohnuki 2018 = diploid essential-gene heterozygotes; Ohnuki 2022 =
    drug-hypersensitive quadruple deletions).
    """

    def __init__(
        self,
        root: str = "data/torchcell/scmd_ohya2005",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset rooted at ``root`` with optional transforms."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment schema class for this dataset."""
        return CalMorphExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference schema class for this dataset."""
        return CalMorphExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw mutant and wildtype TSV filenames."""
        return list(_RAW_FILES)

    def download(self) -> None:
        """Copy both matrices from the sha256-pinned library mirror and verify hashes."""
        data_root = os.environ["DATA_ROOT"]
        mirror = osp.join(data_root, _MIRROR_DIR)
        os.makedirs(self.raw_dir, exist_ok=True)
        for filename, spec in _RAW_FILES.items():
            dest = osp.join(self.raw_dir, filename)
            if not osp.exists(dest):
                src = osp.join(mirror, filename)
                if not osp.exists(src):
                    raise RuntimeError(
                        f"{filename} not found in the library mirror {mirror}. The SCMD "
                        "portal is the historical source; recover the file and deposit it, "
                        "then rebuild (sha256 verified)."
                    )
                shutil.copyfile(src, dest)
            digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
            if digest != spec["sha256"]:
                raise RuntimeError(
                    f"{filename} sha256 mismatch: got {digest}, "
                    f"expected {spec['sha256']}"
                )

    @post_process
    def process(self) -> None:
        """Load raw TSVs, build CalMorph experiments, and write the LMDB store."""
        df_mutant = pd.read_csv(osp.join(self.raw_dir, "mt4718data.tsv"), sep="\t")
        df_wt = pd.read_csv(osp.join(self.raw_dir, "wt122data.tsv"), sep="\t")

        df = self.preprocess_calmorph_data(df_mutant)

        # Save preprocessed data
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        # Aggregate the 122 WT replicate averages into a single mean-WT reference.
        self.wt_reference_phenotype = self._calculate_wt_reference(df_wt)

        log.info("Processing Ohya 2005 CalMorph morphology data...")

        # Initialize LMDB environment
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_calmorph_experiment(
                    self.name, row, wt_reference_phenotype=self.wt_reference_phenotype
                )

                # Serialize the Pydantic objects
                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for CalMorph this is handled in process()."""
        # For this dataset, preprocessing happens in process()
        # This method is required by base class but not used in our flow
        return df

    def preprocess_calmorph_data(self, df_mutant: pd.DataFrame) -> pd.DataFrame:
        """Clean the mutant matrix, drop incomplete rows, and validate ORFs vs R64."""
        df_mutant = df_mutant.copy()
        # The ORF column carries the systematic gene name; there is no common-name column.
        df_mutant["systematic_gene_name"] = df_mutant["ORF"].str.strip().str.upper()

        # Reconcile legacy 2005-annotation ORF names to their current R64 systematic name
        # (SGD alias-based renames; see _LEGACY_ORF_RENAMES). Renaming happens BEFORE R64
        # validation so the recovered strains survive; collisions/retired names do not.
        renamed = df_mutant["systematic_gene_name"].isin(_LEGACY_ORF_RENAMES)
        if renamed.any():
            log.info(
                "Ohya 2005: remapping %d legacy ORF name(s) to R64: %s",
                int(renamed.sum()),
                {
                    o: _LEGACY_ORF_RENAMES[o]
                    for o in df_mutant.loc[renamed, "systematic_gene_name"].unique()
                },
            )
            df_mutant["systematic_gene_name"] = df_mutant[
                "systematic_gene_name"
            ].replace(_LEGACY_ORF_RENAMES)
        df_mutant["perturbed_gene_name"] = df_mutant["systematic_gene_name"]

        df_mutant = df_mutant[df_mutant["systematic_gene_name"].notna()]
        df_mutant = df_mutant[df_mutant["systematic_gene_name"] != ""]

        # Drop any strain with a missing CalMorph value -- CalMorph completeness requires
        # the full 501-trait vocabulary per record; values are NEVER imputed (drop whole).
        feature_cols = [
            c
            for c in df_mutant.columns
            if c not in ("ORF", "systematic_gene_name", "perturbed_gene_name")
        ]
        has_all = df_mutant[feature_cols].notna().all(axis=1)
        n_nan = int((~has_all).sum())
        if n_nan:
            log.warning(
                "Ohya 2005: dropping %d mutant row(s) with missing CalMorph values: %s",
                n_nan,
                df_mutant.loc[~has_all, "systematic_gene_name"].tolist()[:20],
            )
        df_mutant = df_mutant[has_all]

        # Validate against the SGD R64 universe; log + drop any non-resolving ORF.
        data_root = os.environ["DATA_ROOT"]
        r64_genes = _load_sgd_genes(data_root)
        in_r64 = df_mutant["systematic_gene_name"].isin(r64_genes)
        dropped = df_mutant.loc[~in_r64, "systematic_gene_name"].tolist()
        if dropped:
            log.warning(
                "Ohya 2005: dropping %d ORF(s) not resolving to SGD R64 after rename "
                "reconciliation (retired 2005 ORFs + legacy names that collide with an "
                "already-measured gene): %s",
                len(dropped),
                sorted(dropped)[:25],
            )
        df_mutant = df_mutant[in_r64]

        log.info(
            "Ohya 2005: %d non-essential deletion strains after completeness + R64 "
            "validation",
            len(df_mutant),
        )
        return df_mutant.reset_index(drop=True)

    def _calculate_wt_reference(self, df_wt: pd.DataFrame) -> dict[str, Any]:
        """Aggregate the WT replicate averages into a per-feature mean reference."""
        info_columns = [c for c in ("NAME", "ORF") if c in df_wt.columns]
        morphology_columns = [c for c in df_wt.columns if c not in info_columns]
        wt_means: dict[str, Any] = {}
        for col in morphology_columns:
            numeric_values = pd.to_numeric(df_wt[col], errors="coerce")
            wt_means[col] = numeric_values.mean()
        return wt_means

    def create_experiment(self) -> None:
        """Required by base class but not used - see create_calmorph_experiment."""
        pass

    @staticmethod
    def create_calmorph_experiment(
        dataset_name: str, row: pd.Series, wt_reference_phenotype: dict[str, Any]
    ) -> tuple[CalMorphExperiment, CalMorphExperimentReference, Publication]:
        """Build a CalMorph experiment and reference pair from one data row."""
        # Genome reference - haploid BY4741 (MATa his3D1 leu2D0 lys2D0 ura3D0).
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )

        # Create genotype for the non-essential single-gene deletion mutant (kanMX4).
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["systematic_gene_name"],
                    perturbed_gene_name=row["perturbed_gene_name"],
                )
            ]
        )

        # Environment -- Ohya 2005 Methods: YPD, logarithmic-phase (liquid) culture.
        # Temperature 25 C from the Ohya-lab CalMorph standard (see module docstring).
        environment = Environment(
            media=Media(name="YPD", state="liquid", is_synthetic=False),
            temperature=Temperature(value=25),
        )
        environment_reference = environment.model_copy()

        # Extract morphology measurements and separate base from CV parameters. Rows are
        # guaranteed complete + finite (incomplete rows dropped in preprocessing).
        info_columns = {"ORF", "systematic_gene_name", "perturbed_gene_name"}
        base_measurements: dict[str, float] = {}
        cv_measurements: dict[str, float] = {}
        for col in row.index:
            if col in info_columns:
                continue
            float_value = float(row[col])
            if col.startswith(_CV_PREFIXES):
                cv_measurements[col] = float_value
            else:
                base_measurements[col] = float_value

        # Create phenotype with separated base and CV measurements
        phenotype = CalMorphPhenotype(
            calmorph=base_measurements,
            calmorph_coefficient_of_variation=(
                cv_measurements if cv_measurements else None
            ),
        )

        # Reference phenotype from the aggregated WT, split the same way.
        wt_base = {
            k: v
            for k, v in wt_reference_phenotype.items()
            if not k.startswith(_CV_PREFIXES)
        }
        wt_cv = {
            k: v
            for k, v in wt_reference_phenotype.items()
            if k.startswith(_CV_PREFIXES)
        }
        phenotype_reference = CalMorphPhenotype(
            calmorph=wt_base, calmorph_coefficient_of_variation=wt_cv if wt_cv else None
        )

        # Create reference
        reference = CalMorphExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        # Create experiment
        experiment = CalMorphExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        # Publication -- Ohya 2005 PNAS (the data producer; see module docstring).
        publication = Publication(
            pubmed_id="16365294",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/16365294/",
            doi="10.1073/pnas.0509436102",
            doi_url="https://www.pnas.org/doi/10.1073/pnas.0509436102",
        )

        return experiment, reference, publication


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    dataset = ScmdOhya2005Dataset(
        root=osp.join(data_root, "data/torchcell/scmd_ohya2005")
    )
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(dataset[0])
