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

ONE RECORD PER RELEASED CULTURE (2026-11-24, tracking issue #1178). Beside the 4,678
released mM rows, which stay exactly as they were, every strain row of the ``data_raw``
sheet is its own record: ``measurement_type = "cell_extract_concentration_uM"``
(``RAW_MEASUREMENT_TYPE``), ``metabolite_level`` = that row's raw cell-extract uM values
verbatim (a blank cell is left out of the dict), ``n_replicates = 1`` per amino acid,
``metabolite_level_se = None``, and ``replicate_id`` = the row's own ``identifier``
(plate_row_column of the well, e.g. ``41_G_10``; batch 12 prints ``01_D12``), which is
unique over the sheet. A row is one culture: each strain was grown in its own well of a
96-deep-well plate and extracted on its own (Methods "Yeast", "Metabolite Extraction";
``SOURCED_VALUES["batch_cultivation"]``), so no row is a repeat injection of another.
On the pinned workbook that is 5,053 records: 4,897 rows of the 4,678 released ORFs
(4,487 x 1, 167 x 2, 20 x 3, 4 x 4) and 156 rows of 153 ORFs the mM sheet does not carry
(slow growers, the two complemented controls, and strains the authors left out of the
analysis), every one stored because the sheet releases it. The raw uM precedes the
batch normalization and the dilution / volume / cell-number corrections, so it is a
measurement of its own and is never compared with the mM value; the mM-from-uM recipe
is unpublished (#1172). The 237 ``QC_*`` rows are repeat injections of one pooled
extract of 351 strains (``SOURCED_VALUES["qc_sample"]``); a pool has no statable
genotype, so they are not records.

The raw records' reference is the mean of the complemented control strains' ``data_raw``
rows (``his3D::kanMX4`` YOR202W and ``met17D::kanMX4`` YLR303W, one row each), which the
paper names as the isogenic reference wild type (``SOURCED_VALUES["reference_wt"]``):
``SD / sqrt(n)`` and ``n`` per amino acid, ``replicate_id = None``, restricted to the
amino acids the record measured. The two control rows are records themselves, under
their released deletion. The mM records keep the robust-mean reference.
"""

import logging
import math
import numbers
import os
import os.path as osp
import pickle
import re
import urllib.request
from typing import Any, cast

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
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
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "intracellular_concentration_mM"
#: The ``data_raw`` sheet's unit: "Raw concentration data of the cell extract in [uM]"
#: (Overview sheet), before batch normalization and the dilution / volume / cell-number
#: corrections that give ``MEASUREMENT_TYPE``.
RAW_MEASUREMENT_TYPE = "cell_extract_concentration_uM"
#: The complemented deletion strains the paper uses as the isogenic wild type:
#: his3D::kanMX4 and met17D::kanMX4 (``SOURCED_VALUES["reference_wt"]``).
COMPLEMENTED_CONTROL_ORFS = ("YOR202W", "YLR303W")
_QC_PREFIX = "QC_"
_CONC_SHEET = "intracellular_concentration_mM"
_SUMMARY_SHEET = "robust_summary_statistics"
_RAW_SHEET = "data_raw"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")

#: A concentration-sheet ORF cell that names no deletion strain: a blank cell (pandas
#: reads it as NaN, stringified ``nan``) or the WT background row.
_NOT_A_STRAIN = frozenset({"nan", "WT"})


def _released_deletion(
    orf: str,
) -> KanMxDeletionPerturbation | ReleasedIdentifierPerturbation:
    """The deletion under the released identifier (#1045): an ORF-shaped one on the
    placed KanMX leaf with both names the ORF, any other on the released-identifier leaf
    verbatim (kind deletion, marker KanMX).
    """
    if _SYSTEMATIC_RE.match(orf):
        return KanMxDeletionPerturbation(
            systematic_gene_name=orf, perturbed_gene_name=orf
        )
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=orf,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=orf,
        identifier_kind="strain column entry of Table S3, as released",
        released_kind=ReleasedKind.deletion,
        cassette="KanMX",
    )


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
    "batch_cultivation": _sv(
        "one data_raw row is one culture",
        "Strains from the same batch were cultivated and their metabolites extracted "
        "on the same day and analyzed together by LC-MS/MS.",
        note="paper.md line 341; line 333 grows each strain in its own well of "
        "'96-deep well plates', and every data_raw row sits at its own plate "
        "identifier (all 410 rows of the 191 multi-row released ORFs are at distinct "
        "wells, measured on the pinned workbook), so each row is a separate culture "
        "and extract",
    ),
    "qc_sample": _sv(
        237,
        "The QC was prepared by joining the extracts of the first batch including "
        "metabolites of 351 deletion strains.",
        note="paper.md line 341 (same paragraph: 'a quality control sample (QC) was "
        "assessed every 24 samples and in total 237 times'); the 237 QC_* rows of "
        "data_raw are repeat injections of this pool, which has no statable genotype",
    ),
    "reference_wt": _sv(
        COMPLEMENTED_CONTROL_ORFS,
        "contain a complemented his3 deletion strain representing the isogenic "
        "reference wt strain",
        note="paper.md line 487 (Figure S5B legend); line 459 (Figure S1F) shows the "
        "WT profile 'for the complemented deletion strains his3D::kanMX4 and "
        "met17D::kanMX4'. YOR202W is HIS3 and YLR303W is MET17 in data_raw; both are "
        "left out of the mM sheet (line 379, 'we excluded the complemented deletion "
        "strains')",
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


class InvalidRawRowError(ValueError):
    """A ``data_raw`` row the loader cannot store as one culture without guessing.

    Raised for a repeated ``identifier`` (the row's replicate label), a row with no ORF
    that is not a ``QC_*`` injection, a text cell, a row blank in every amino acid, or a
    complemented control strain with no row. The pinned workbook has none of these.
    """


def _raw_level(value: Any, identifier: str, amino_acid: str) -> float | None:
    """One ``data_raw`` cell: a finite float, or None for a blank cell (left out)."""
    if isinstance(value, numbers.Real) and math.isfinite(value):
        return float(value)
    if isinstance(value, numbers.Real) and math.isnan(value):
        return None
    if value is None:
        return None
    raise InvalidRawRowError(
        f"Mulleder Table S3 data_raw: {identifier} {amino_acid} value {value!r} is not "
        "a finite number"
    )


def _raw_cultures(raw: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    """Every strain row of ``data_raw`` as one culture, in sheet order, plus the QC count.

    Each culture is ``{"identifier", "orf", "batch", "levels"}`` with ``levels`` the
    row's non-blank uM values. A row with no ORF is a ``QC_*`` injection of the pooled
    extract (counted, not a culture); any other ORF-less row, a repeated identifier, a
    text cell or a row blank in all 19 amino acids raises ``InvalidRawRowError``.
    """
    cultures: list[dict[str, Any]] = []
    seen: set[str] = set()
    n_qc = 0
    for _, row in raw.iterrows():
        identifier = str(row["identifier"]).strip()
        if identifier in seen:
            raise InvalidRawRowError(
                f"Mulleder Table S3 data_raw: identifier {identifier} is on two rows; "
                "a culture is one row"
            )
        seen.add(identifier)
        if pd.isna(row["ORF"]):
            if not identifier.startswith(_QC_PREFIX):
                raise InvalidRawRowError(
                    f"Mulleder Table S3 data_raw: row {identifier} has no ORF and is "
                    "not a QC injection"
                )
            n_qc += 1
            continue
        levels = {
            aa: level
            for aa in AMINO_ACIDS
            if (level := _raw_level(row[aa], identifier, aa)) is not None
        }
        if not levels:
            raise InvalidRawRowError(
                f"Mulleder Table S3 data_raw: row {identifier} is blank in every "
                "amino acid"
            )
        cultures.append(
            {
                "identifier": identifier,
                "orf": str(row["ORF"]).strip(),
                "batch": str(row["batch"]).strip(),
                "levels": levels,
            }
        )
    return cultures, n_qc


def _control_summary(cultures: list[dict[str, Any]]) -> MetabolitePhenotype:
    """The raw-uM reference: mean, ``SD / sqrt(n)`` and n over the complemented controls.

    One row per control ORF on the pinned workbook, so n = 2 per amino acid. A control
    ORF with no row raises ``InvalidRawRowError``.
    """
    controls = [c for c in cultures if c["orf"] in COMPLEMENTED_CONTROL_ORFS]
    absent = sorted(set(COMPLEMENTED_CONTROL_ORFS) - {c["orf"] for c in controls})
    if absent:
        raise InvalidRawRowError(
            f"Mulleder Table S3 data_raw: complemented control {absent} has no row"
        )
    level: dict[str, float] = {}
    se: dict[str, float] = {}
    n_rep: dict[str, int] = {}
    for aa in AMINO_ACIDS:
        values = [c["levels"][aa] for c in controls if aa in c["levels"]]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        level[aa] = mean
        n_rep[aa] = n
        if n >= 2:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample SD
            se[aa] = math.sqrt(var) / math.sqrt(n)
        else:
            se[aa] = math.nan  # one control value: SD undefined
    return MetabolitePhenotype(
        metabolite_level=level,
        metabolite_level_se=se,
        n_replicates=n_rep,
        measurement_type=RAW_MEASUREMENT_TYPE,
        target_metabolite_ids=None,
    )


def _restricted(summary: MetabolitePhenotype, keys: set[str]) -> MetabolitePhenotype:
    """``summary`` on the amino acids in ``keys`` only (a record's measured set)."""
    se = summary.metabolite_level_se or {}
    return MetabolitePhenotype(
        metabolite_level={
            k: v for k, v in summary.metabolite_level.items() if k in keys
        },
        metabolite_level_se={k: v for k, v in se.items() if k in keys},
        n_replicates={k: v for k, v in summary.n_replicates.items() if k in keys},
        measurement_type=summary.measurement_type,
        target_metabolite_ids=None,
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

    organism = Organism.scerevisiae

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
        """Parse Table S3 into per-ORF mM records plus one raw uM record per culture."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        path = osp.join(self.raw_dir, DATA_FILENAME)
        conc = pd.read_excel(path, sheet_name=_CONC_SHEET)
        summary = pd.read_excel(path, sheet_name=_SUMMARY_SHEET)
        # batch is a label ("01" ... "12", "03-2"), read as text so "01" stays "01"
        raw = pd.read_excel(path, sheet_name=_RAW_SHEET, dtype={"batch": str})
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
        released_names: list[str] = []
        blank_or_control: list[str] = []
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _, row in conc.iterrows():
            orf = str(row["ORF"]).strip()
            if orf in _NOT_A_STRAIN:
                # a blank ORF cell (read as NaN) names no strain, and WT is the
                # unperturbed background the robust mean summarizes, not a deletion
                blank_or_control.append(orf)
                continue
            if not _SYSTEMATIC_RE.match(orf):
                # A released identifier with no ORF form goes on the released-gene-name
                # leaf verbatim (#1045; 0 of 4,678 released rows).
                released_names.append(orf)
            if orf in seen:
                raise RuntimeError(
                    f"Mulleder Table S3: released ORF {orf} is on two rows of the "
                    "concentration sheet; one strain is one row, refusing to keep the "
                    "first (0 of 4,678 released rows)"
                )
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
            "Mulleder: %d released ORFs stored, %d on the released-identifier leaf%s, "
            "%d blank or WT rows%s",
            len(rows),
            len(released_names),
            f" ({sorted(released_names)})" if released_names else "",
            len(blank_or_control),
            f" ({sorted(blank_or_control)})" if blank_or_control else "",
        )
        # Reference n: the strains the population robust mean summarizes (issue #489).
        self._n_reference_strains = len(rows)
        n_rows = pd.Series([r["n_raw_rows"] for r in rows])
        log.info(
            "Mulleder: data_raw rows per released ORF %s; reference n_replicates %d "
            "(strains the robust mean summarizes)",
            {
                int(cast(int, k)): int(v)
                for k, v in n_rows.value_counts().sort_index().items()
            },
            self._n_reference_strains,
        )
        pd.DataFrame(
            [{k: v for k, v in r.items() if k != "n_replicates"} for r in rows]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        # One record per data_raw culture (#1178), built before anything is written.
        cultures, n_qc = _raw_cultures(raw)
        self._control_summary = _control_summary(cultures)
        released = {r["orf"] for r in rows}
        log.info(
            "Mulleder: %d data_raw cultures stored as raw uM records (%d of released "
            "ORFs, %d of %d ORFs not in the mM sheet), %d QC injections not records",
            len(cultures),
            sum(c["orf"] in released for c in cultures),
            sum(c["orf"] not in released for c in cultures),
            len({c["orf"] for c in cultures} - released),
            n_qc,
        )
        pd.DataFrame(
            [
                {
                    "identifier": c["identifier"],
                    "orf": c["orf"],
                    "batch": c["batch"],
                    **{aa: c["levels"].get(aa) for aa in AMINO_ACIDS},
                }
                for c in cultures
            ]
        ).to_csv(osp.join(self.preprocess_dir, "data_raw.csv"), index=False)

        records = [self.create_experiment(record_row) for record_row in rows] + [
            self.create_raw_experiment(culture) for culture in cultures
        ]
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for experiment, reference, publication in tqdm(records):
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
            "Wrote %d Mulleder amino-acid experiments to LMDB (%d mM, %d raw uM)",
            idx,
            len(rows),
            len(cultures),
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    @staticmethod
    def _genome_reference() -> ReferenceGenome:
        """BY4741 on R64, the background of the prototrophic deletion collection."""
        # PROTOTROPHIC deletion collection (auxotrophy restored episomally, Mulleder 2012);
        # the prototrophy-restoring marker(s) are not yet modeled as a GeneAddition here.
        return ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

    @staticmethod
    def _environment() -> Environment:
        """SM agar at 30 C, the screen environment of every record."""
        # Synthetic minimal (SM) agar, 30 C, exponential growth (Methods "Yeast"):
        # 6.7 g/L YNB without amino acids (Sigma Y0626) + 2% glucose + 2% agar, sourced
        # and quoted on ``SM_AGAR``. The spots on that agar inoculate a liquid SM
        # subculture, which is what the amino acids are extracted from, so whether this
        # record's state should be liquid (``SM``) is an open question flagged in
        # [[torchcell.datamodels.media-components]]; the recipe is the same either way.
        return Environment(media=SM_AGAR, temperature=Temperature(value=30))

    @staticmethod
    def _publication() -> Publication:
        """Mulleder et al. 2016, Cell."""
        return Publication(
            pubmed_id="27693354",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27693354/",
            doi="10.1016/j.cell.2016.09.007",
            doi_url="https://doi.org/10.1016/j.cell.2016.09.007",
        )

    def create_raw_experiment(
        self, culture: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """One ``data_raw`` culture: its raw uM values verbatim, n = 1, no dispersion.

        The reference is the complemented-control summary on the amino acids this
        culture measured (``_control_summary``), ``replicate_id`` None.
        """
        environment = self._environment()
        levels: dict[str, float] = culture["levels"]
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=Genotype(perturbations=[_released_deletion(culture["orf"])]),
            environment=environment,
            phenotype=MetabolitePhenotype(
                metabolite_level=dict(levels),
                metabolite_level_se=None,  # one culture: no dispersion
                n_replicates=dict.fromkeys(levels, 1),
                measurement_type=RAW_MEASUREMENT_TYPE,
                target_metabolite_ids=None,
                replicate_id=culture["identifier"],
            ),
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=self._genome_reference(),
            environment_reference=environment.model_copy(),
            phenotype_reference=_restricted(self._control_summary, set(levels)),
        )
        return experiment, reference, self._publication()

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one ORF."""
        genome_reference = self._genome_reference()
        genotype = Genotype(perturbations=[_released_deletion(row["orf"])])
        environment = self._environment()
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
        return experiment, reference, self._publication()


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
