# torchcell/datasets/scerevisiae/mulleder2016
# [[torchcell.datasets.scerevisiae.mulleder2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/mulleder2016
# Test file: tests/torchcell/datasets/scerevisiae/test_mulleder2016.py
"""Mulleder 2016 amino-acid metabolome dataset (genome-wide deletion screen).

Mulleder et al. 2016 (Cell, doi:10.1016/j.cell.2016.09.007) measured the intracellular
concentration of 19 amino acids (the 20 proteinogenic amino acids minus cysteine, which
oxidises) by LC-SRM/MS in ~4678 strains of the yeast deletion collection, grown
exponentially on synthetic MINIMAL (SM) agar. The collection is PROTOTROPHIC (the
auxotrophic markers were restored episomally, Mulleder et al. 2012) so the strains grow
on minimal medium without amino-acid supplementation -- essential for reading the
amino-acid metabolome.

Source (scriptable, hash-pinned): Mendeley Data DOI 10.17632/bnzdhd6ck8.1, file
``Table_S3_Complete_Dataset.xls``. We ingest the ``intracellular_concentration_mM``
sheet: one row per ORF x 19 amino acids in mM (batch-normalised, adjusted for dilution,
extraction volume, cell number and volume). The reference/baseline per amino acid is the
population robust mean from the ``robust_summary_statistics`` sheet (Minimum Covariance
Determinant), the same central tendency the paper's Z-scores are computed against.

Maps to ``MetabolitePhenotype`` (WS4): ``metabolite_level = {amino_acid -> mM}`` with
``measurement_type = "intracellular_concentration_mM"``. The genome-wide screen is n=1
per strain (single culture; the 237x QC sample is for analytical performance, not
per-strain averaging -- Methods "Yeast" / "Quantification of Amino Acids in High
Throughput"), so ``n_replicates = 1`` per amino acid and ``metabolite_level_se = None``.
Amino acids are NATIVE Yeast9 metabolites, so ``target_metabolite_ids`` (amino acid ->
``s_NNNN``) is populatable for constraint-based-model linkage; left ``None`` here and
deferred to a follow-up that sources the ids from YeastGEM (never guessed).
"""

import hashlib
import logging
import os
import os.path as osp
import pickle
import re
import urllib.request
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "intracellular_concentration_mM"
_CONC_SHEET = "intracellular_concentration_mM"
_SUMMARY_SHEET = "robust_summary_statistics"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")

# Mendeley Data 10.17632/bnzdhd6ck8.1 -> Table_S3_Complete_Dataset.xls (direct, pinned).
DATA_URL = (
    "https://data.mendeley.com/public-files/datasets/bnzdhd6ck8/files/"
    "621f3646-9e51-488a-b6b8-f6427b40fc87/file_downloaded"
)
DATA_FILENAME = "Table_S3_Complete_Dataset.xls"
DATA_SHA256 = "a7fcb4bc8aa5e394e7f6e2b99e327eaa88fa04111ab5602fc7cb3445f653802e"

# 19 amino acids measured (20 proteinogenic minus cysteine); the exact sheet columns.
AMINO_ACIDS = [
    "alanine",
    "aspartate",
    "glutamate",
    "phenylalanine",
    "glycine",
    "histidine",
    "isoleucine",
    "lysine",
    "leucine",
    "methionine",
    "asparagine",
    "proline",
    "glutamine",
    "arginine",
    "serine",
    "threonine",
    "valine",
    "tryptophan",
    "tyrosine",
]


@register_dataset
class AminoAcidMulleder2016Dataset(ExperimentDataset):
    """Genome-wide intracellular amino-acid metabolome of the yeast deletion collection."""

    def __init__(
        self,
        root: str = "data/torchcell/amino_acid_mulleder2016",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (ORFs are systematic ids; no genome injection needed)."""
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
        """The Mendeley Table S3 workbook required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Download Table S3 from Mendeley Data and verify its sha256."""
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Mulleder Table S3 from %s", DATA_URL)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != DATA_SHA256:
            raise RuntimeError(
                f"Mulleder Table S3 sha256 mismatch: got {got}, expected {DATA_SHA256}"
            )
        with open(dest, "wb") as handle:
            handle.write(data)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @post_process
    def process(self) -> None:
        """Parse Table S3 into per-ORF Metabolite experiments and write LMDB."""
        path = osp.join(self.raw_dir, DATA_FILENAME)
        conc = pd.read_excel(path, sheet_name=_CONC_SHEET)
        summary = pd.read_excel(path, sheet_name=_SUMMARY_SHEET)

        missing = [aa for aa in AMINO_ACIDS if aa not in conc.columns]
        if missing:
            raise RuntimeError(
                f"Table S3 concentration sheet missing columns: {missing}"
            )

        # Reference baseline: population robust mean per amino acid (MCD estimate).
        self._reference_levels = {
            str(r["amino acid"]): float(r["mean (mM)"]) for _, r in summary.iterrows()
        }
        ref_missing = [aa for aa in AMINO_ACIDS if aa not in self._reference_levels]
        if ref_missing:
            raise RuntimeError(f"summary sheet missing amino acids: {ref_missing}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        n_bad_orf = 0
        seen: set[str] = set()
        collisions: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _, row in conc.iterrows():
            orf = str(row["ORF"]).strip()
            if not _SYSTEMATIC_RE.match(orf):
                n_bad_orf += 1
                continue
            if orf in seen:
                collisions.add(orf)
                continue
            seen.add(orf)
            rows.append({"orf": orf, **{aa: float(row[aa]) for aa in AMINO_ACIDS}})
        log.info(
            "Mulleder: %d usable ORFs, %d non-systematic ORF names skipped, "
            "%d ORF collisions deduped",
            len(rows),
            n_bad_orf,
            len(collisions),
        )
        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for record_row in tqdm(rows):
                experiment, reference, publication = self.create_experiment(record_row)
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
        log.info("Wrote %d Mulleder amino-acid experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one ORF."""
        # PROTOTROPHIC deletion collection (auxotrophy restored episomally, Mulleder 2012);
        # the prototrophy-restoring marker(s) are not yet modeled as a GeneAddition here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["orf"]
                )
            ]
        )
        # Synthetic minimal (SM) agar, 30 C, exponential growth (Methods "Yeast").
        environment = Environment(
            media=Media(name="SM", state="solid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        phenotype = MetabolitePhenotype(
            metabolite_level={aa: row[aa] for aa in AMINO_ACIDS},
            metabolite_level_se=None,  # genome-wide screen is n=1 per strain; no per-strain SE
            n_replicates={aa: 1 for aa in AMINO_ACIDS},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,  # deferred: amino acid -> Yeast9 s_NNNN (YeastGEM)
        )
        # Reference = WT-equivalent baseline (population robust mean per amino acid).
        phenotype_reference = MetabolitePhenotype(
            metabolite_level=dict(self._reference_levels),
            metabolite_level_se=None,
            n_replicates={aa: 1 for aa in self._reference_levels},
            measurement_type=MEASUREMENT_TYPE,
        )
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="27693354",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27693354/",
            doi="10.1016/j.cell.2016.09.007",
            doi_url="https://doi.org/10.1016/j.cell.2016.09.007",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the dataset for interactive debugging.

    Loads the existing LMDB if already built. To step through
    ``process()``/``create_experiment`` under a debugger, delete ``<root>/processed``
    first so the build re-runs.
    """
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(os.environ["DATA_ROOT"], "data/torchcell/amino_acid_mulleder2016")
    dataset = AminoAcidMulleder2016Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
