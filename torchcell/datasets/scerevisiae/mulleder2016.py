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
``measurement_type = "intracellular_concentration_mM"``. ``metabolite_level_se = None``:
no sheet releases a per-strain error.

Per-strain ``n_replicates`` (issue #488) is COUNTED from the released ``data_raw`` sheet
("Raw concentration data of the cell extract in [uM]; includes QC sample and slow
growing strains", Overview sheet): for each released ORF and amino acid, the number of
``data_raw`` rows for that ORF with a value in that amino-acid column. On the pinned
workbook 4,487 of the 4,678 released ORFs have one row and 191 have two to four (167 x 2,
20 x 3, 4 x 4); 8 cells on 6 of those multi-row rows are blank, so 7 ORF x amino-acid
counts of 5 strains are below the strain's row count. What is PUBLISHED about repeated
measurement is only ``SOURCED_VALUES["repeat_screen"]``: the 479 lowest-concentration
strains were re-plated and the analysis repeated (batch 12). How a strain's two to four
rows become its one released mM value is NOT published. That the released value is the
mean of the normalized rows is a diagnostic inference from the 2026.09.29 re-verification
(scratch script ``mulleder_multi_average_test.py``, report
``notes/assets/verification/2026.09.29/mulleder2016.md``, claim M2: the mean of the
normalized rows fits better than the best single row for 180 of the 191 strains), not a
published statement. The 237 QC injections (``QC_*`` identifiers, no ORF) are a pooled
extract for analytical performance and are never counted for a strain.

Reference ``n_replicates`` (issue #489) is the number of strains the population robust
mean was estimated over, counted from the released concentration sheet (4,678, equal to
the published ``SOURCED_VALUES["n_profiled_strains"]``). The Minimum Covariance
Determinant (``SOURCED_VALUES["reference_estimator"]``) down-weights outlying strains,
so this is a population estimate over 4,678 different deletion strains, NOT 4,678
replicate measurements of one sample; read it as the size of the population the
reference summarizes. Decision recorded 2026.10.02 and open to revision: the schema's
``n_replicates`` ("number of independent replicates") has no population-estimate
field, and the previous value 1 described neither.

Amino acids are NATIVE Yeast9 metabolites, so ``target_metabolite_ids`` (amino acid ->
``s_NNNN``) is populatable for constraint-based-model linkage; left ``None`` here and
deferred to a follow-up that sources the ids from YeastGEM (never guessed).
"""

import logging
import math
import numbers
import os
import os.path as osp
import pickle
import re
import urllib.request
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.media import SM_AGAR
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "intracellular_concentration_mM"
_CONC_SHEET = "intracellular_concentration_mM"
_SUMMARY_SHEET = "robust_summary_statistics"
_RAW_SHEET = "data_raw"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")

# Mendeley Data 10.17632/bnzdhd6ck8.1 -> Table_S3_Complete_Dataset.xls (direct, pinned).
DATA_URL = (
    "https://data.mendeley.com/public-files/datasets/bnzdhd6ck8/files/"
    "621f3646-9e51-488a-b6b8-f6427b40fc87/file_downloaded"
)
DATA_FILENAME = "Table_S3_Complete_Dataset.xls"
DATA_SHA256 = "a7fcb4bc8aa5e394e7f6e2b99e327eaa88fa04111ab5602fc7cb3445f653802e"

# Mirrored paper (MinerU OCR); every SOURCED_VALUES quote is a verbatim substring of it.
CITATION_KEY = "mullederFunctionalMetabolomicsDescribes2016"
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "20412bec5b930d1fa43d326d8f9baca267130ddef5fdaa804f19e1209f925e6e"


def _sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` anchored to the mirrored ``paper.md`` by sha256."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (mirrored artifact)",
        ),
        quote=quote,
        note=note,
    )


SOURCED_VALUES: dict[str, SourcedValue] = {
    "reference_estimator": _sv(
        "Minimum Covariance Determinant robust mean over the profiled strains",
        "We used the Minimum Covariance Determinant (MCD) estimator to obtain robust "
        "values for mean, standard deviation, correlation and covariance for each amino "
        "acid (Figures 1 and 2A).",
        note="paper.md line 387; the reference metabolite_level is the "
        "robust_summary_statistics 'mean (mM)' column. MCD down-weights outlying "
        "strains, so it is a population estimate, not replicates of one sample",
    ),
    "n_profiled_strains": _sv(
        4678,
        "Eventually, we completed a metabolic profile for 4678 yeast deletion strains",
        note="paper.md line 395; the reference n_replicates is COUNTED from the "
        "released concentration sheet at build time (4,678 on the pinned workbook, "
        "equal to this value). Line 387 says the MCD average is over 'all analyzed "
        "strains'; that this set is exactly the 4,678 released strains is supported, "
        "not proven, by refitting an MCD on them (experiments/036-dataset-fixes-"
        "before-kg-build/scripts/mulleder2016_replicates.py)",
    ),
    "repeat_screen": _sv(
        479,
        "In a second screening round, the 479 deletion strains that showed lowest "
        "metabolite concentrations were arranged on 6 additional plates and the "
        "analysis was repeated (batch 12).",
        note="paper.md line 341; the only published statement about measuring a "
        "strain more than once. How repeated rows combine into the released value "
        "is not stated",
    ),
}

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


class MissingRawMeasurementError(ValueError):
    """A released ORF x amino acid with no value in the ``data_raw`` sheet.

    ``n_replicates`` is counted from ``data_raw``; a released value with zero raw
    measurements behind it has no count. The pinned workbook has none (every one of the
    4,678 x 19 released cells has at least one raw value).
    """


class InvalidConcentrationError(ValueError):
    """A Table S3 concentration cell that is blank or not a finite number.

    The pinned ``intracellular_concentration_mM`` sheet has none (0 of 4,678 x 19
    cells), so such a cell is refused by name rather than served as NaN or failing in
    Python's ``float()``.
    """


def _concentration(value: Any, orf: str, amino_acid: str) -> float:
    """Return one concentration cell as a float, refusing a blank or non-numeric cell."""
    if isinstance(value, numbers.Real) and math.isfinite(value):
        return float(value)
    raise InvalidConcentrationError(
        f"Mulleder Table S3: {orf} {amino_acid} concentration {value!r} is not a "
        "finite number"
    )


def _raw_counts(raw_counts: pd.DataFrame, orf: str) -> dict[str, int]:
    """Per amino acid, the ``data_raw`` rows of ``orf`` that carry a value.

    Raises ``MissingRawMeasurementError`` for an ORF absent from ``data_raw`` or an
    amino acid with no raw value for it (the pinned workbook has neither).
    """
    if orf not in raw_counts.index:
        raise MissingRawMeasurementError(
            f"Mulleder Table S3: released ORF {orf} has no data_raw row"
        )
    values = raw_counts[AMINO_ACIDS].loc[orf].to_numpy(dtype=int)
    counts = {aa: int(n) for aa, n in zip(AMINO_ACIDS, values, strict=True)}
    empty = [aa for aa, n in counts.items() if n == 0]
    if empty:
        raise MissingRawMeasurementError(
            f"Mulleder Table S3: released ORF {orf} has no data_raw value for {empty}"
        )
    return counts


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
        write_verified(data, dest, DATA_SHA256, DATA_URL)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @post_process
    def process(self) -> None:
        """Parse Table S3 into per-ORF Metabolite experiments and write LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        path = osp.join(self.raw_dir, DATA_FILENAME)
        conc = pd.read_excel(path, sheet_name=_CONC_SHEET)
        summary = pd.read_excel(path, sheet_name=_SUMMARY_SHEET)
        raw = pd.read_excel(path, sheet_name=_RAW_SHEET)
        # Per ORF x amino acid: how many data_raw rows carry a value (issue #488).
        # QC rows have no ORF and drop out of the groupby.
        raw_counts = raw.groupby("ORF")[AMINO_ACIDS].count()
        raw_row_counts = raw.groupby("ORF").size()

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
        n_collision_rows = 0
        seen: set[str] = set()
        collisions: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _, row in conc.iterrows():
            orf = str(row["ORF"]).strip()
            if not _SYSTEMATIC_RE.match(orf):
                n_bad_orf += 1
                continue
            if orf in seen:
                n_collision_rows += 1
                collisions.add(orf)
                continue
            seen.add(orf)
            rows.append(
                {
                    "orf": orf,
                    **{aa: _concentration(row[aa], orf, aa) for aa in AMINO_ACIDS},
                    "n_replicates": _raw_counts(raw_counts, orf),
                    "n_raw_rows": int(raw_row_counts[orf]),
                }
            )
        log.info(
            "Mulleder: %d usable ORFs, %d non-systematic ORF names skipped, "
            "%d repeated-ORF rows dropped (%d ORFs kept at their first row)",
            len(rows),
            n_bad_orf,
            n_collision_rows,
            len(collisions),
        )
        # Reference n: the strains the population robust mean summarizes (issue #489).
        self._n_reference_strains = len(rows)
        n_rows = pd.Series([r["n_raw_rows"] for r in rows])
        log.info(
            "Mulleder: data_raw rows per released ORF %s; reference n_replicates %d "
            "(strains the robust mean summarizes)",
            {int(k): int(v) for k, v in n_rows.value_counts().sort_index().items()},
            self._n_reference_strains,
        )
        pd.DataFrame(
            [{k: v for k, v in r.items() if k != "n_replicates"} for r in rows]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

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
        # Synthetic minimal (SM) agar, 30 C, exponential growth (Methods "Yeast"):
        # 6.7 g/L YNB without amino acids (Sigma Y0626) + 2% glucose + 2% agar, sourced
        # and quoted on ``SM_AGAR``. The spots on that agar inoculate a liquid SM
        # subculture, which is what the amino acids are extracted from, so whether this
        # record's state should be liquid (``SM``) is an open question flagged in
        # [[torchcell.datamodels.media-components]]; the recipe is the same either way.
        environment = Environment(media=SM_AGAR, temperature=Temperature(value=30))
        phenotype = MetabolitePhenotype(
            metabolite_level={aa: row[aa] for aa in AMINO_ACIDS},
            metabolite_level_se=None,  # no sheet releases a per-strain error
            # data_raw measurements behind each value (module docstring, issue #488)
            n_replicates=dict(row["n_replicates"]),
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,  # deferred: amino acid -> Yeast9 s_NNNN (YeastGEM)
        )
        # Reference = WT-equivalent baseline (population robust mean per amino acid).
        # n = the strains the MCD estimate summarizes (issue #489), not replicates of
        # one sample: MCD down-weights outliers, so this is a population estimate.
        phenotype_reference = MetabolitePhenotype(
            metabolite_level=dict(self._reference_levels),
            metabolite_level_se=None,
            n_replicates={
                aa: self._n_reference_strains for aa in self._reference_levels
            },
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
