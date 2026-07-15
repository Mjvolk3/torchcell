"""CalMorph morphology dataset for essential-gene het diploids (Ohnuki & Ohya 2018).

Ohnuki S & Ohya Y 2018, "High-dimensional single-cell phenotyping reveals extensive
haploinsufficiency", PLoS Biol 16(5):e2005130 (doi:10.1371/journal.pbio.2005130;
PMID 29768403). CalMorph high-dimensional morphology of a EUROSCARF collection of
HETEROZYGOUS diploid deletion strains, each dropping ONE essential gene from 2 -> 1
copies in the ``BY4743`` diploid wild type (a HIP-style heterozygous / 50%-dosage
deletion, NOT a full knockout). This is the diploid, essential-gene, dosage counterpart
of :class:`~torchcell.datasets.scerevisiae.ohya2005.ScmdOhya2005Dataset` (which is a
haploid ``BY4741`` full-deletion screen).

Scope: this loader builds ONLY the OPTIMAL (nutrient-rich) arm of the paper -- the 1,112
essential-gene heterozygotes phenotyped in liquid YPD at 25 C. The severe arm (50
heterozygotes in nutrient-poor SD medium at 37 C) has no raw CalMorph matrix released and
is ignored.

Growth conditions sourced verbatim from the Methods ("Yeast strains, media, and growth
conditions"): "The yeast diploid strain BY4743 was used as the wild type. Strains
heterozygous for 1,112 essential genes ... were cultured under optimal growth conditions
at 25 C in nutrient-rich yeast extract peptone dextrose (YPD) medium containing 1% (w/v)
Bacto yeast extract, 2% (w/v) Bacto peptone, and 2% (w/v) glucose". Phenotyped cells were
harvested from early-log-phase LIQUID YPD culture (media state ``"liquid"``). Heterozygous
gene-deletion mutants were purchased from EUROSCARF (kanMX4 marker); essential genes were
defined per the paper's ref [34].

Values are RAW per-strain CalMorph population averages (same semantics as Ohya 2005), one
501-length vector per strain: 281 base parameters (``CALMORPH_LABELS``: 220 mean + 61
ratio) + 220 coefficient-of-variation / noise parameters (``CALMORPH_STATISTICS``,
prefixed CCV/ACV/DCV/TCV). The 114-row wildtype matrix (114 independent BY4743 WT
replicate averages) is aggregated per-feature into a single mean-WT reference phenotype.

PROVENANCE / SOURCING (sha256-pinned; loader reads the library-mirror ``data/`` files and
verifies both hashes, never the live URL):
- ``ess1112data.tsv`` (mutant matrix, 1,112 strains x 501 features; ID column ``ORF``)
  sha256 ``2d168bd1c436c7edae0ab3eb07e99e4c41f3b12f69607a7f3a00092eed7c4b03``.
- ``wt114data.tsv`` (114 WT replicate averages; ID column ``NAME``; feature columns
  byte-identical to the mutant file) sha256
  ``f48d42da2c727854b83b70e8768cfb42ada0e5118f6074765698d3045862804e``.
- Historical retrieval URLs (SCMD2 portal, per the paper's Data Availability statement):
  ``http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=ess1112data.tsv`` and
  ``...=wt114data.tsv``.

ORFs are 2018-annotation SGD systematic names, reconciled to the current R64-4-1
annotation by the shared genome resolver
(:meth:`SCerevisiaeGenome.resolve_gene_name`): live names are kept, SGD renames/merges
and valid non-"gene" loci (pseudogene / blocked_reading_frame) are remapped to their
current systematic id (collision-safe), and retired names are retained verbatim. NO
record is dropped for a naming reason -- every strain is a real measured perturbation.
"""

# torchcell/datasets/scerevisiae/ohnuki2018
# [[torchcell.datasets.scerevisiae.ohnuki2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/ohnuki2018
# Test file: tests/torchcell/datasets/scerevisiae/test_ohnuki2018.py

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
    EngineeredCopyNumberPerturbation,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import GeneNameStatus, SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# CV parameters are prefixed CCV/ACV/DCV/TCV; everything else is a base parameter.
_CV_PREFIXES = ("CCV", "ACV", "DCV", "TCV")

# Statuses whose resolved systematic id is a valid R64 identifier we prefer over the
# source name (a live gene, an SGD rename, or a valid non-"gene" locus such as a
# blocked_reading_frame / pseudogene).
_REMAP_STATUSES = (
    GeneNameStatus.CURRENT,
    GeneNameStatus.RENAMED,
    GeneNameStatus.NON_GENE_FEATURE,
)

# sha256-pinned raw matrices in the library mirror ``data/`` directory.
_RAW_FILES: dict[str, dict[str, str]] = {
    "ess1112data.tsv": {
        "sha256": "2d168bd1c436c7edae0ab3eb07e99e4c41f3b12f69607a7f3a00092eed7c4b03",
        "id_column": "ORF",
    },
    "wt114data.tsv": {
        "sha256": "f48d42da2c727854b83b70e8768cfb42ada0e5118f6074765698d3045862804e",
        "id_column": "NAME",
    },
}

# Library mirror holding the sha256-pinned raw matrices (relative to DATA_ROOT).
_MIRROR_DIR = "torchcell-library/ohnukiHighdimensionalSinglecellPhenotyping2018/data"


@register_dataset
class ScmdOhnuki2018Dataset(ExperimentDataset):
    """CalMorph morphology of essential-gene heterozygous diploid deletions (Ohnuki 2018).

    Diploid ``BY4743`` HIP-style heterozygous (50%-dosage) deletions of 1,112 essential
    genes, phenotyped in liquid YPD at 25 C; the diploid essential-gene counterpart of the
    haploid Ohya 2005 full-deletion CalMorph screen.
    """

    def __init__(
        self,
        root: str = "data/torchcell/scmd_ohnuki2018",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        genome: SCerevisiaeGenome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; an optional genome reconciles ORF names to R64.

        ORFs are 2018-annotation systematic names; the genome's layered resolver maps
        them to the current R64-4-1 annotation. If ``genome`` is not supplied one is
        constructed from ``DATA_ROOT`` during processing.
        """
        self.genome = genome
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
                        f"{filename} not found in the library mirror {mirror}. The SCMD2 "
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
        df_mutant = pd.read_csv(osp.join(self.raw_dir, "ess1112data.tsv"), sep="\t")
        df_wt = pd.read_csv(osp.join(self.raw_dir, "wt114data.tsv"), sep="\t")

        df = self.preprocess_calmorph_data(df_mutant)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        # Aggregate the 114 WT replicate averages into a single mean-WT reference.
        self.wt_reference_phenotype = self._calculate_wt_reference(df_wt)

        log.info("Processing Ohnuki 2018 CalMorph morphology data...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_calmorph_experiment(
                    self.name, row, wt_reference_phenotype=self.wt_reference_phenotype
                )
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
        return df

    def preprocess_calmorph_data(self, df_mutant: pd.DataFrame) -> pd.DataFrame:
        """Clean the mutant matrix and reconcile ORF names to the current R64 annotation.

        Names are reconciled with the genome's layered resolver
        (:meth:`SCerevisiaeGenome.resolve_gene_name`); NO record is dropped for a naming
        reason -- every strain is a real measured perturbation. A resolved current
        identifier (live gene, SGD rename, or valid non-"gene" feature) replaces the 2018
        name unless it would collide with another strain (an SGD merge), in which case the
        original name is kept so the strains stay distinct. Retired/ambiguous names are
        kept verbatim.
        """
        df_mutant = df_mutant.copy()
        # The ORF column carries the systematic gene name; there is no common-name column.
        df_mutant["systematic_gene_name"] = df_mutant["ORF"].str.strip().str.upper()
        df_mutant = df_mutant[df_mutant["systematic_gene_name"].notna()]
        df_mutant = df_mutant[df_mutant["systematic_gene_name"] != ""].reset_index(
            drop=True
        )

        df_mutant["systematic_gene_name"] = self._reconcile_orf_names(
            df_mutant["systematic_gene_name"]
        )
        df_mutant["perturbed_gene_name"] = df_mutant["systematic_gene_name"]

        log.info(
            "Ohnuki 2018: %d essential-gene heterozygote strains (0 dropped for naming)",
            len(df_mutant),
        )
        return df_mutant

    def _reconcile_orf_names(self, names: pd.Series) -> pd.Series:
        """Map 2018 ORF names to current R64 ids via the genome resolver (retain-all).

        Collision-safe: a remap is applied only when its target is not already claimed by
        another strain in the screen; otherwise the original name is kept so merged ORFs
        stay distinct records. Nothing is dropped.
        """
        from collections import Counter

        if self.genome is None:
            data_root = os.environ["DATA_ROOT"]
            self.genome = SCerevisiaeGenome(
                genome_root=osp.join(data_root, "data/sgd/genome"),
                go_root=osp.join(data_root, "data/go"),
                overwrite=False,
            )
        genome = self.genome

        resolutions = {name: genome.resolve_gene_name(name) for name in names.unique()}
        proposed = {
            name: (
                res.systematic_name
                if res.status in _REMAP_STATUSES and res.systematic_name is not None
                else name
            )
            for name, res in resolutions.items()
        }
        proposed_counts = Counter(proposed.values())
        final = {
            name: (name if proposed_counts[prop] > 1 else prop)
            for name, prop in proposed.items()
        }

        remapped = {n: f for n, f in final.items() if f != n}
        by_status: dict[str, int] = {}
        for res in resolutions.values():
            by_status[res.status.value] = by_status.get(res.status.value, 0) + 1
        collided = sorted(
            n for n, prop in proposed.items() if proposed_counts[prop] > 1
        )
        retired = sorted(
            n for n, r in resolutions.items() if r.status == GeneNameStatus.RETIRED
        )
        log.info(
            "Ohnuki 2018 ORF reconciliation: %d unique names %s; %d remapped to current "
            "R64 ids; %d kept as legacy names on merge-collision %s; %d retained as "
            "retired legacy names %s",
            len(resolutions),
            by_status,
            len(remapped),
            len(collided),
            collided,
            len(retired),
            retired,
        )
        return names.map(final)

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
        # Reference genome - diploid BY4743 wild type (Methods).
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
        )

        # HIP-style heterozygous deletion: one essential-gene copy dropped 2 -> 1 in the
        # diploid (50% dosage), kanMX4 marker (EUROSCARF). NOT a full knockout.
        genotype = Genotype(
            perturbations=[
                EngineeredCopyNumberPerturbation(
                    systematic_gene_name=row["systematic_gene_name"],
                    perturbed_gene_name=row["perturbed_gene_name"],
                    copy_number=1,
                    reference_copy_number=2,
                    marker="KanMX",
                )
            ]
        )

        # Optimal growth conditions: nutrient-rich liquid YPD at 25 C (Methods).
        environment = Environment(
            media=Media(name="YPD", state="liquid", is_synthetic=False),
            temperature=Temperature(value=25),
        )
        environment_reference = environment.model_copy()

        # Separate the 501-length feature vector into 281 base + 220 CV parameters.
        info_columns = {"ORF", "systematic_gene_name", "perturbed_gene_name"}
        base_measurements: dict[str, float] = {}
        cv_measurements: dict[str, float] = {}
        for col in row.index:
            if col in info_columns:
                continue
            value = row[col]
            float_value = float(value) if pd.notna(value) else 0.0
            if col.startswith(_CV_PREFIXES):
                cv_measurements[col] = float_value
            else:
                base_measurements[col] = float_value

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

        reference = CalMorphExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )
        experiment = CalMorphExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29768403",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29768403/",
            doi="10.1371/journal.pbio.2005130",
            doi_url="https://doi.org/10.1371/journal.pbio.2005130",
        )
        return experiment, reference, publication


if __name__ == "__main__":
    pass
