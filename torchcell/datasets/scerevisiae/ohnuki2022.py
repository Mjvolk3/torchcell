# torchcell/datasets/scerevisiae/ohnuki2022
# [[torchcell.datasets.scerevisiae.ohnuki2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/ohnuki2022
# Test file: tests/torchcell/datasets/scerevisiae/test_ohnuki2022.py
"""Ohnuki 2022 high-throughput CalMorph morphology of drug-hypersensitive deletions.

Ohnuki et al. 2022 (npj Syst Biol Appl 8:3, doi:10.1038/s41540-022-00212-1; PMID
35087094) built a diagnostic set of 1982 haploid QUADRUPLE-deletion strains for
morphological profiling: each target gene (selected from the 2378 morphology-important
non-essential genes) was individually deleted in a fixed DRUG-HYPERSENSITIVE
triple-deletion background 3Delta (``pdr1Delta pdr3Delta snq2Delta``, strain Y13206;
MATalpha snq2Delta::KlLEU2 pdr3Delta::KlURA3 pdr1Delta::NATMX ... his3Delta1 leu2Delta0
ura3Delta0 met15Delta LYS2, derived from S288C). Cells were cultured in liquid YPD (1%
yeast extract, 2% peptone, 2% glucose) at 25 C, triple-stained (cell wall / actin /
nuclear DNA), imaged on an automatic HT microscope, and quantified with CalMorph v1.2
into the SAME 501-trait vocabulary (281 base + 220 CV) as Ohya 2005. Values are RAW
per-strain CalMorph averages (n=1 measurement per mutant), NOT z-scores.

The REFERENCE is the 3Delta PARENT (not true wild type): ``wt749data.tsv`` holds 749
replicate CalMorph averages of the 3Delta strain, aggregated here into a single mean
reference phenotype carrying the full 501-trait vocabulary.

This loader builds ONLY the gene-deletion morphology matrix. The compound side of the
paper (7 compounds) is not a released per-strain morphology matrix and is not buildable.

GENOTYPE FIDELITY -- 3Delta QUADRUPLE BACKGROUND (design decision, FLAGGED for review).
Every strain is modeled with FOUR deletion perturbations at full genotype fidelity:
  * target gene -> ``KanMxDeletionPerturbation`` (the target deletions are the BY4741
    KanMX YKO alleles crossed into the 3Delta query; the paper's his3Delta control is
    ``BY4741 ... YOR202wDelta::kanMX4`` from EUROSCARF);
  * the constant 3Delta background (systematic names verified against SGD R64):
      - PDR1 = YGL013C -> ``NatMxDeletionPerturbation`` (pdr1Delta::NATMX)
      - PDR3 = YBL005W -> ``MarkerDeletionPerturbation`` marker ``KlURA3``
      - SNQ2 = YDR011W -> ``MarkerDeletionPerturbation`` marker ``KlLEU2``
  This reuses the exact 3DeltaAlpha representation established by the Vanacloig 2022
  loader (same drug-sensitized background).

  The reference (``ExperimentReference``) structurally carries only a
  ``ReferenceGenome`` (species + strain) -- it CANNOT encode the 3Delta background
  genotype. So the reference genome is recorded as ``BY4741`` (per project convention)
  while the ACTUAL reference strain is the 3Delta parent Y13206 (S288C-derived), whose
  MEASURED averages populate ``phenotype_reference`` from ``wt749data.tsv``. This
  asymmetry (mutants carry the explicit 3Delta background; the reference genome is a
  WT-only placeholder) is the design decision flagged for human review.

DROPPED STRAINS (3 of 1982 -> 1979 records):
  * ``YGL141W`` (HRD1): the single mutant row with missing values (2 CV traits
    ``ACV103_A1B``, ``ACV103_C`` are NaN). CalMorph completeness requires the full
    501-trait vocabulary per record and values are never imputed, so the strain is
    dropped whole (never partially imputed).
  * ``YGL013C`` (PDR1) and ``YDR011W`` (SNQ2): these appear as target ORFs but are
    ALREADY deleted in the 3Delta background, so they cannot be independent quadruple
    deletions (deleting an already-deleted gene is not a new perturbation). Dropped per
    the same background-collision convention used by the Vanacloig 2022 loader.
"""

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
from torchcell.datamodels.calmorph_labels import CALMORPH_STATISTICS
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MarkerDeletionPerturbation,
    Media,
    NatMxDeletionPerturbation,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.gene_name_reconcile import (
    default_genome,
    reconcile_systematic_names,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Raw SCMD2 quadruple-set CalMorph matrices, mirrored + sha256-pinned. Source (retrieval
# metadata; the loader reads the mirror and verifies these sha256):
#   http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=quad1982data.tsv (pj=quadruple)
#   http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=wt749data.tsv    (pj=quadruple)
MUTANT_FILE = "quad1982data.tsv"
WT_FILE = "wt749data.tsv"
MUTANT_SHA256 = "5a1d45005c1249a77b0608ee7e6678c045c14464cf67bfc149b24fcaeb4854c0"
WT_SHA256 = "4603aadf6ae5a3187e447c5d0df1f6bfb30a2292ddb687beb9a871b14769094d"

# Local library mirror holding the sha256-pinned raw tsvs (under $DATA_ROOT).
MIRROR_SUBPATH = osp.join(
    "torchcell-library", "ohnukiHighthroughputPlatformYeast2022", "data"
)

# The constant drug-hypersensitive 3Delta background, deleted in EVERY strain (paper
# Methods, strain Y13206: "snq2Delta::KlLEU2 pdr3Delta::KlURA3 pdr1Delta::NATMX").
# Systematic names verified against SGD R64: PDR1=YGL013C, PDR3=YBL005W, SNQ2=YDR011W.
BACKGROUND_GENES = frozenset({"YGL013C", "YBL005W", "YDR011W"})

# CV traits carry raw CalMorph coefficients-of-variation; the remaining traits are base.
_CV_LABELS = frozenset(CALMORPH_STATISTICS)


def _sha256(path: str) -> str:
    """Return the hex sha256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _background_perturbations() -> list[Any]:
    """The constant 3Delta background deletions (shared across every record)."""
    return [
        NatMxDeletionPerturbation(
            systematic_gene_name="YGL013C", perturbed_gene_name="PDR1"
        ),
        MarkerDeletionPerturbation(
            systematic_gene_name="YBL005W", perturbed_gene_name="PDR3", marker="KlURA3"
        ),
        MarkerDeletionPerturbation(
            systematic_gene_name="YDR011W", perturbed_gene_name="SNQ2", marker="KlLEU2"
        ),
    ]


@register_dataset
class ScmdOhnuki2022Dataset(ExperimentDataset):
    """CalMorph morphology for 1979 quadruple (target + 3Delta) deletion strains."""

    def __init__(
        self,
        root: str = "data/torchcell/scmd_ohnuki2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        genome: SCerevisiaeGenome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; an optional genome reconciles target ORF names to R64.

        If ``genome`` is not supplied one is constructed from ``DATA_ROOT`` during
        processing (used only to reconcile the target ORF name to the current annotation).
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
        """Return the raw mutant and 3Delta-reference TSV filenames."""
        return [MUTANT_FILE, WT_FILE]

    def download(self) -> None:
        """Copy both raw tsvs from the sha256-pinned library mirror and verify each."""
        from dotenv import load_dotenv

        load_dotenv()
        mirror_dir = osp.join(os.environ["DATA_ROOT"], MIRROR_SUBPATH)
        os.makedirs(self.raw_dir, exist_ok=True)
        for fname, expected in ((MUTANT_FILE, MUTANT_SHA256), (WT_FILE, WT_SHA256)):
            dest = osp.join(self.raw_dir, fname)
            if osp.exists(dest):
                got = _sha256(dest)
                if got != expected:
                    raise RuntimeError(
                        f"{fname} sha256 mismatch in raw dir: got {got}, "
                        f"expected {expected}"
                    )
                continue
            src = osp.join(mirror_dir, fname)
            if not osp.exists(src):
                raise RuntimeError(f"mirror file missing: {src}")
            src_digest = _sha256(src)
            if src_digest != expected:
                raise RuntimeError(
                    f"{fname} sha256 mismatch in mirror: got {src_digest}, "
                    f"expected {expected}"
                )
            shutil.copyfile(src, dest)
            dest_digest = _sha256(dest)
            if dest_digest != expected:
                raise RuntimeError(
                    f"{fname} sha256 mismatch after copy: got {dest_digest}, "
                    f"expected {expected}"
                )
            log.info(
                "Copied %s from mirror (%d bytes, sha256 verified)",
                fname,
                osp.getsize(dest),
            )

    @post_process
    def process(self) -> None:
        """Load raw tsvs, build quadruple-deletion CalMorph experiments, write LMDB."""
        df_mutant = pd.read_csv(osp.join(self.raw_dir, MUTANT_FILE), sep="\t")
        df_wt = pd.read_csv(osp.join(self.raw_dir, WT_FILE), sep="\t")

        df_mutant["ORF"] = df_mutant["ORF"].str.strip().str.upper()
        feature_cols = [c for c in df_mutant.columns if c != "ORF"]

        # Reconcile the target ORF name to the current R64 annotation (retain-all,
        # collision-safe; shared genome resolver). NOT dropped for naming here -- only the
        # NaN and 3Delta-background filters below drop rows.
        if self.genome is None:
            self.genome = default_genome()
        df_mutant["systematic_gene_name"] = reconcile_systematic_names(
            self.genome, df_mutant["ORF"], label="Ohnuki 2022"
        )

        # Drop the single strain with missing CalMorph values (never impute).
        has_all = df_mutant[feature_cols].notna().all(axis=1)
        n_nan = int((~has_all).sum())
        if n_nan:
            log.info(
                "Ohnuki: dropping %d mutant row(s) with missing CalMorph values: %s",
                n_nan,
                df_mutant.loc[~has_all, "systematic_gene_name"].tolist(),
            )
        # Drop target ORFs that coincide with the 3Delta background (cannot be an
        # independent quadruple deletion of an already-deleted gene). Checked on the
        # reconciled name so a legacy alias of a background gene is also caught.
        not_bg = ~df_mutant["systematic_gene_name"].isin(BACKGROUND_GENES)
        n_bg = int((~not_bg).sum())
        if n_bg:
            log.info(
                "Ohnuki: dropping %d target row(s) colliding with the 3Delta "
                "background: %s",
                n_bg,
                df_mutant.loc[~not_bg, "systematic_gene_name"].tolist(),
            )
        df_mutant = df_mutant.loc[has_all & not_bg].reset_index(drop=True)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        df_mutant.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        wt_reference = self._calculate_wt_reference(df_wt, feature_cols)
        background = _background_perturbations()
        publication = Publication(
            pubmed_id="35087094",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/35087094/",
            doi="10.1038/s41540-022-00212-1",
            doi_url="https://doi.org/10.1038/s41540-022-00212-1",
        )

        log.info(
            "Processing Ohnuki 2022 CalMorph morphology (%d strains)...", len(df_mutant)
        )
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))
        with env.begin(write=True) as txn:
            for index, row in tqdm(df_mutant.iterrows(), total=df_mutant.shape[0]):
                experiment, reference = self._build_experiment(
                    self.name, row, feature_cols, wt_reference, background
                )
                txn.put(
                    f"{index}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
        env.close()

    @staticmethod
    def _split_traits(
        values: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Split a trait dict into (base, CV) via the CALMORPH_STATISTICS vocabulary."""
        base: dict[str, float] = {}
        cv: dict[str, float] = {}
        for key, value in values.items():
            if key in _CV_LABELS:
                cv[key] = value
            else:
                base[key] = value
        return base, cv

    def _calculate_wt_reference(
        self, df_wt: pd.DataFrame, feature_cols: list[str]
    ) -> CalMorphPhenotype:
        """Aggregate the 749 replicate 3Delta profiles into one mean reference phenotype."""
        wt_means = {
            col: float(pd.to_numeric(df_wt[col], errors="raise").mean())
            for col in feature_cols
        }
        base, cv = self._split_traits(wt_means)
        return CalMorphPhenotype(
            calmorph=base, calmorph_coefficient_of_variation=cv if cv else None
        )

    def _build_experiment(
        self,
        dataset_name: str,
        row: pd.Series,
        feature_cols: list[str],
        wt_reference: CalMorphPhenotype,
        background: list[Any],
    ) -> tuple[CalMorphExperiment, CalMorphExperimentReference]:
        """Build one quadruple-deletion CalMorph experiment + 3Delta reference."""
        orf = str(row["systematic_gene_name"])
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                ),
                *background,
            ]
        )

        # Liquid YPD at 25 C (paper Methods: "Yeast cell culture and harvest").
        environment = Environment(
            media=Media(name="YPD", state="liquid", is_synthetic=False),
            temperature=Temperature(value=25),
        )

        measurements = {col: float(row[col]) for col in feature_cols}
        base, cv = self._split_traits(measurements)
        phenotype = CalMorphPhenotype(
            calmorph=base, calmorph_coefficient_of_variation=cv if cv else None
        )

        # Reference genome is a WT-only placeholder (ExperimentReference cannot carry the
        # 3Delta background genotype); the 3Delta parent's MEASURED averages populate the
        # reference phenotype. See module docstring (flagged design decision).
        reference = CalMorphExperimentReference(
            dataset_name=dataset_name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=wt_reference,
        )
        experiment = CalMorphExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        return experiment, reference

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
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
    root = osp.join(data_root, "data/torchcell/scmd_ohnuki2022")
    dataset = ScmdOhnuki2022Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
