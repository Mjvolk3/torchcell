# torchcell/datasets/ecoli/schmidt2016_replicate_intensity
# [[torchcell.datasets.ecoli.schmidt2016_replicate_intensity]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/schmidt2016_replicate_intensity
# Test file: tests/torchcell/datasets/ecoli/test_schmidt2016_replicate_intensity.py
"""Schmidt 2016 Table S8: the per-culture normalized LC-MS intensities, one record each.

Schmidt et al. 2016 (Nat Biotechnol 34:104, doi:10.1038/nbt.3418) grew BW25113 in 22
conditions "in biological triplicates" and released, in Supplementary Table 8, the
SafeQuant-normalized label-free abundance of every protein in every one of those
cultures ("Normalized Abundance of Biological Replicates Determined by LFQ using
SafeQuant"). :class:`ProteomeReplicateIntensitySchmidt2016Dataset` serves that block as
**one ``ProteinAbundanceExperiment`` per culture: 66 records = 22 conditions x 3
cultures**, the three glucose cultures included as records (#1178, 2026-11-24).

WHY A DATASET OF ITS OWN. ``ProteomeSchmidt2016Dataset`` stores Table S6's protein
copies per cell, which the release gives only as one value per (protein, condition):
the authors' aggregate over the three cultures, a statistic the source reports and so
data that stays. The per-culture quantity the release gives is a different number, a
normalized MS1 intensity on SafeQuant's scale, not copies per cell. ``verify_protein_dataset``
requires one ``measurement_type`` per dataset (the reason ``schmidt2016_srm`` is two
datasets), so the per-culture intensities are their own dataset. The two join on the
protein key and on the culture's condition: within a condition, Table S6's copies/cell
is a per-protein multiple of the median of these three values (``schmidt2016.py``,
docstring fact 3), which is how a model recovers the copies/cell spread.

WHAT A RECORD IS. One culture: ``protein_abundance`` is that culture's released
``normInt_<file>.1`` for every Table S8 row, verbatim; ``n_replicates`` is 1 for every
key; ``protein_abundance_se`` is ``None`` with a ``not_applicable_single_measurement``
gap; ``replicate_id`` is the culture's LC-MS file name as Table S25 lists it
(``A14-07036``). ``Table S25`` titles its condition column "Growth Condition (in
biological triplicates)", and the Methods say "All samples of data set 2 were prepared
in biological triplicates." Table S8 releases one ``normInt`` column per Table S25 file,
one file per culture, so nothing technical is combined inside a record. (Supplementary
Figure 4 counts 72 data-set-2 LC-MS runs against Table S8's 69 ``normInt`` columns; the
release does not identify the other 3.)

THE REFERENCE. Every record's ``phenotype_reference`` is the authors' own
``medianNormInt_Glucose.1``: the median of the three glucose cultures, released by the
source (and asserted equal to the median of the three released columns for every row),
``n_replicates`` 3, ``replicate_id`` ``None``, no SE (the release states a CV).

THE PROTEIN KEY IS THE COPIES/CELL STORE'S KEY. Table S8 prints older or truncated
symbols on 5 rows (``yacF`` where Table S6 prints ``zapD``, ``yt`` for ``ytfL``,
``Mett`` for ``MettuDRAFT_4149``, an empty symbol for ``B8LFD5``), so the key is joined
through the row's released UniProt accession to the Table S6 row and takes that row's
stored key (``schmidt2016.select_proteins``), which is what lets the two stores join.
Every Table S8 accession is on a Table S6 dataset-2 row (measured: 2,058 of 2,058).
The one exception is ``P63284`` (``clpB``), which BOTH tables file on two rows, so no
Table S8 row can be attributed to one Table S6 row: its two rows are keyed by their own
Table S8 row (``clpB | Table S8 row 1040``), both stored. Released Table S8 symbols and
the stored keys are in ``preprocess/protein_identifiers.csv``.

NOT STORED, AND WHY: THE SECOND GLUCOSE TRIPLICATE (``Glucose.2``, A14-07088..90). The
Supplementary Note states that it is a re-analysis of the same three glucose samples, not
three more cultures: "Further, the 3 glucose growth condition samples were analyzed at
the beginning and end of the quantitative LC-MS experiment to check if the data was
impaired by any systematic performance changes of the LC-MS platform employed throughout
the whole analysis time". A technical repeat that the design does not need to form a
value is not a new biological replicate, and the release does not say which repeat file
belongs to which culture, so it can be neither stored as three cultures nor combined
into the culture it repeats. ``ProteinAbundancePhenotype`` has no field for a technical
repeat; the three columns are listed in ``preprocess/dropped_records.json`` and left to
the owner (#1176). Table S9 (the MG1655 / NCM3722 comparison, a separate SafeQuant and
absolute calibration of its own) is not read here either; see the dendron note.

DATA. The one consumed file is ``si2.xlsx``, the same pinned workbook ``schmidt2016``
consumes, from the same raw mirror under the same sha256 pin.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import pickle
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    Experiment,
    ExperimentReference,
    Genotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
)
from torchcell.datasets.bacteria_common import assembly_reference, bacterial_genome
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import schmidt2016 as sm
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

#: The paper's citation key: a class registered after the candidate gate landed carries
#: it, and its verdict is ``database/candidates/<key>.json``.
CITATION_KEY = "schmidtQuantitativeConditiondependentEscherichia2016"
DATASET_ROOT_REL = "data/torchcell/proteome_replicate_intensity_schmidt2016"

#: What one stored value IS: one culture's SafeQuant-normalized label-free MS1 intensity
#: from the data-set-2 LC-MS experiment. Not copies per cell.
MEASUREMENT_TYPE = "label_free_ms1_intensity_safequant_normalized_dataset2"
#: Table S8's median column of the glucose triplicate Table S6 is built on.
REFERENCE_MEDIAN_COLUMN = "medianNormInt_Glucose.1"
#: A Table S8 row whose accession Table S8 files on more than one row is keyed by its
#: own Table S8 row (``clpB | Table S8 row 1040``).
S8_ROW_SEPARATOR = " | Table S8 row "
#: 22 conditions x 3 cultures, measured on the pinned sheet.
N_CULTURES_PER_CONDITION = 3
EXPECTED_RECORDS = 66
#: Table S8 rows, every one a key of every record (no released normInt cell is empty).
EXPECTED_PROTEIN_KEYS = 2058
#: Table S8 groups the build does not store: the glucose re-analysis.
UNSTORED_GROUP_SUFFIX = "Glucose.2"
#: Relative tolerance of the median identity against the released bytes.
IDENTITY_RTOL = 1e-9

_Q_TABLE_S8_BLOCK = (
    "Normalized Abundance of Biological Replicates Determined by LFQ using SafeQuant "
    "(sample names see below)"
)
_Q_TABLE_S25_HEADER = "Growth Condition (in biological triplicates)"
_Q_DATASET2_TRIPLICATES = sm._Q_TRIPLICATES
_Q_GLUCOSE_REANALYZED = (
    "Further, the 3 glucose growth condition samples were analyzed at the beginning and "
    "end of the quantitative LC-MS experiment to check if the data was impaired by any "
    "systematic performance changes of the LC-MS platform employed throughout the whole "
    "analysis time of the high number of $\\mathtt { > 6 0 }$ samples."
)
_Q_SAFEQUANT_NORMALIZATION = (
    "This analysis included global data normalization by equalizing the total MS1 peak "
    "areas across all LC-MS runs, summation of MS1 peak areas per protein and LC-MS/MS "
    "run, followed by calculation of protein abundance ratios."
)

#: The Supplementary Information's OCR markdown in the literature mirror.
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "24660453432a8007290b31d7448c82044bd65766c971d3dc902d775a042c334a"

PAPER = Provenance(
    source_uri=sm.PAPER_MD, citation_key=sm.CITATION_KEY, sha256=sm.PAPER_MD_SHA256
)
SI2_SOURCE = Provenance(
    source_uri=sm.SI2_MIRROR_RELPATH, citation_key=sm.CITATION_KEY, sha256=sm.SI2_SHA256
)
SI1_SOURCE = Provenance(
    source_uri=SI1_MD, citation_key=sm.CITATION_KEY, sha256=SI1_MD_SHA256
)

#: Quotes by provenance, by the constant's own name; the tests assert each is a
#: substring of its pinned artifact.
PAPER_QUOTES: dict[str, str] = {"dataset2_triplicates": _Q_DATASET2_TRIPLICATES}
SI1_QUOTES: dict[str, str] = {
    "glucose_reanalyzed": _Q_GLUCOSE_REANALYZED,
    "safequant_normalization": _Q_SAFEQUANT_NORMALIZATION,
}
SI2_QUOTES: dict[str, str] = {
    "table_s8_replicate_block": _Q_TABLE_S8_BLOCK,
    "table_s25_header": _Q_TABLE_S25_HEADER,
}

SOURCED_VALUES: dict[str, SourcedValue] = {
    "replicate_kind": SourcedValue(
        value="biological",
        provenance=PAPER,
        quote=_Q_DATASET2_TRIPLICATES,
        note="Table S8's block is titled per biological replicate and Table S25's "
        "condition column reads 'in biological triplicates'",
    ),
    "replicate_block": SourcedValue(
        value="Table S8 normInt_<file>.1",
        provenance=SI2_SOURCE,
        quote=_Q_TABLE_S8_BLOCK,
    ),
    "replicate_label": SourcedValue(
        value="Table S25 File Name",
        provenance=SI2_SOURCE,
        quote=_Q_TABLE_S25_HEADER,
        note="the header of the column beside the file names; the file name is the "
        "source's own label of a culture's sample",
    ),
    "normalization": SourcedValue(
        value="SafeQuant global normalization (total MS1 peak area equalized per run)",
        provenance=SI1_SOURCE,
        quote=_Q_SAFEQUANT_NORMALIZATION,
    ),
    "glucose_2_is_a_reanalysis": SourcedValue(
        value="Glucose.2 (A14-07088..90) re-analyzes the three Glucose.1 samples",
        provenance=SI1_SOURCE,
        quote=_Q_GLUCOSE_REANALYZED,
        note="the 6 glucose runs are 3 samples analyzed twice; the release does not "
        "pair a repeat file with its culture, so the repeat is not stored (owner "
        "decision, #1176)",
    ),
}

#: The typed absence on a record's SE map: one culture has no dispersion.
RECORD_SE_GAP = ProvenanceGap(
    field="protein_abundance_se",
    reason=ProvenanceGapReason.not_applicable_single_measurement,
    note="one culture's one normalized intensity per protein: a single measurement has "
    "no standard error",
)
#: The typed absence on the reference's SE map: the release states a CV instead.
REFERENCE_SE_GAP = ProvenanceGap(
    field="protein_abundance_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the reference is the released medianNormInt_Glucose.1; the release states "
    "the coefficient of variation of the triplicate (cv_Glucose.1), not a standard "
    "error, and the three cultures are records",
)


# --------------------------------------------------------------------------- #
# Reading Table S8
# --------------------------------------------------------------------------- #
class IntensityRow(BaseModel):
    """One Table S8 row: its identifiers and every per-culture normalized intensity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_number: int = Field(description="1-based row in Table S8, for the ledger.")
    uniprot: str
    released_gene: str | None = Field(
        description="Table S8's Gene cell, verbatim; None where the cell is empty"
    )
    intensity: dict[str, float] = Field(
        description="Table S25 file name -> released normInt_<file>.1"
    )
    reference_median: float = Field(description="released medianNormInt_Glucose.1")


def condition_files() -> dict[str, tuple[str, ...]]:
    """``{Table S6 condition column: its three culture file names}``, in sheet order."""
    return {spec.s6_column: spec.s25_files for spec in sm.CONDITIONS}


def read_table_s8_intensities(path: str) -> list[IntensityRow]:
    """Every Table S8 row with every stored culture's normalized intensity.

    The block is located by the ``normInt_<file>.1`` headers of the declared files, never
    by position, and a non-numeric cell in a stored column stops the build.
    """
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book[sm.SHEET_S8].iter_rows(values_only=True))
    finally:
        book.close()
    index = sm._header_index(rows[sm.HEADER_ROW - 1])
    files = [name for names in condition_files().values() for name in names]
    columns = {name: index[sm._s8_column(name)] for name in files}
    median = index[REFERENCE_MEDIAN_COLUMN]
    out: list[IntensityRow] = []
    for offset, row in enumerate(rows[sm.HEADER_ROW :]):
        if row[0] is None:
            continue
        values: dict[str, float] = {}
        for name, column in columns.items():
            cell = row[column]
            if not isinstance(cell, (int, float)) or isinstance(cell, bool):
                raise RuntimeError(
                    f"Table S8 row {sm.HEADER_ROW + 1 + offset}: {sm._s8_column(name)} "
                    f"is {cell!r}, not a number"
                )
            values[name] = float(cell)
        gene = row[index[sm.COL_GENE]]
        out.append(
            IntensityRow(
                row_number=sm.HEADER_ROW + 1 + offset,
                uniprot=str(row[0]).strip(),
                released_gene=None if gene is None else str(gene).strip(),
                intensity=values,
                reference_median=float(row[median]),
            )
        )
    return out


def unstored_columns(path: str) -> list[str]:
    """The Table S8 normInt columns of the files no record stores (``Glucose.2``)."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        header = next(
            row
            for number, row in enumerate(book[sm.SHEET_S8].iter_rows(values_only=True))
            if number == sm.HEADER_ROW - 1
        )
    finally:
        book.close()
    stored = {
        sm._s8_column(name) for names in condition_files().values() for name in names
    }
    return [
        str(cell)
        for cell in header
        if cell is not None
        and str(cell).startswith("normInt_")
        and str(cell) not in stored
    ]


def check_reference_median(rows: Sequence[IntensityRow]) -> dict[str, Any]:
    """The released glucose median IS the median of the three glucose cultures."""
    glucose = condition_files()[sm.REFERENCE_CONDITION]
    worst = 0.0
    for row in rows:
        computed = statistics.median(row.intensity[name] for name in glucose)
        deviation = abs(computed - row.reference_median) / abs(row.reference_median)
        worst = max(worst, deviation)
        if deviation > IDENTITY_RTOL:
            raise RuntimeError(
                f"Table S8 row {row.row_number}: {REFERENCE_MEDIAN_COLUMN} "
                f"{row.reference_median} is not the median {computed} of {glucose}"
            )
    return {"n_rows": len(rows), "worst_rel_dev": worst, "files": list(glucose)}


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def protein_keys(
    rows: Sequence[IntensityRow], s6_rows: Sequence[sm.ProteinRow]
) -> dict[int, str]:
    """``{Table S8 row: stored key}``, the copies/cell store's key wherever it is defined.

    A Table S8 accession on exactly one Table S8 row and one Table S6 row takes that
    Table S6 row's stored key. An accession Table S8 files on more than one row cannot be
    attributed to one Table S6 row, so each of its rows is keyed by its own released
    symbol and Table S8 row. Any other shape stops the build.
    """
    selection = sm.select_proteins(s6_rows, label="Table S6 gene symbols")
    by_accession: dict[str, list[sm.ProteinRow]] = {}
    for s6 in s6_rows:
        by_accession.setdefault(s6.uniprot, []).append(s6)
    s8_counts: dict[str, int] = {}
    for row in rows:
        s8_counts[row.uniprot] = s8_counts.get(row.uniprot, 0) + 1
    keys: dict[int, str] = {}
    for row in rows:
        matches = by_accession.get(row.uniprot, [])
        if not matches:
            raise RuntimeError(f"Table S8 accession {row.uniprot} is not in Table S6")
        if s8_counts[row.uniprot] > 1:
            keys[row.row_number] = (
                f"{row.released_gene}{S8_ROW_SEPARATOR}{row.row_number}"
            )
        elif len(matches) == 1:
            keys[row.row_number] = selection.key[matches[0].row_number]
        else:
            raise RuntimeError(
                f"Table S8 accession {row.uniprot} is on one Table S8 row and "
                f"{len(matches)} Table S6 rows"
            )
    if len(set(keys.values())) != len(rows):
        raise RuntimeError("two Table S8 rows share one stored key")
    return keys


# --------------------------------------------------------------------------- #
# Phenotypes
# --------------------------------------------------------------------------- #
def culture_phenotype(
    rows: Sequence[IntensityRow], keys: Mapping[int, str], file_name: str
) -> ProteinAbundancePhenotype:
    """One culture's released normalized intensities, verbatim: n = 1, no SE."""
    abundance = {keys[row.row_number]: row.intensity[file_name] for row in rows}
    return ProteinAbundancePhenotype(
        protein_abundance=abundance,
        protein_abundance_se=None,
        n_replicates=dict.fromkeys(abundance, 1),
        measurement_type=MEASUREMENT_TYPE,
        replicate_id=file_name,
        provenance_gaps=[RECORD_SE_GAP],
    )


def reference_phenotype(
    rows: Sequence[IntensityRow], keys: Mapping[int, str]
) -> ProteinAbundancePhenotype:
    """The released glucose median over its three cultures, each of them a record."""
    abundance = {keys[row.row_number]: row.reference_median for row in rows}
    return ProteinAbundancePhenotype(
        protein_abundance=abundance,
        protein_abundance_se=None,
        n_replicates=dict.fromkeys(abundance, N_CULTURES_PER_CONDITION),
        measurement_type=MEASUREMENT_TYPE,
        replicate_id=None,
        provenance_gaps=[REFERENCE_SE_GAP],
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeReplicateIntensitySchmidt2016Dataset(ExperimentDataset):
    """Per-culture BW25113 normalized LC-MS intensities, one record per culture."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = sm.SCHMIDT_REFERENCE_STRAIN

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the loader reads no annotation."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The one consumed file, linked from the raw mirror."""
        return [sm.SI2]

    def download(self) -> None:
        """Link the mirrored workbook into ``raw/`` after checking manifest and sha256."""
        data_root = os.environ["DATA_ROOT"]
        manifest = sm.load_manifest(data_root)
        check_manifest_pin(
            sm.SI2_MIRROR_RELPATH,
            sm.manifest_sha256(manifest, sm.SI2_MIRROR_RELPATH),
            sm.SI2_SHA256,
        )
        src = sm.raw_mirror_dir(data_root) / sm.SI2_MIRROR_RELPATH
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, sm.SI2), sm.SI2_SHA256)

    def compute_gene_set(self) -> GeneSet:
        """The protein keys the stored profiles carry (every record is wild type)."""
        if self.env is None:
            self._init_db()
        genes = GeneSet()
        with self.env.begin() as txn:
            for _, value in txn.cursor():
                record = pickle.loads(value)
                genes.update(record["experiment"]["phenotype"]["protein_abundance"])
        self.close_lmdb()
        return genes

    @post_process
    def process(self) -> None:
        """Build one record per culture and write the LMDB."""
        verify_raw_files(self.raw_dir, sm.DATA_SHA256)
        path = osp.join(self.raw_dir, sm.SI2)
        sm.check_sample_map(sm.read_table_s25(path))
        s6_rows, _ = sm.read_table_s6(path)
        rows = read_table_s8_intensities(path)
        keys = protein_keys(rows, s6_rows)
        median_check = check_reference_median(rows)
        environments = {
            spec.s6_column: sm.build_environment(spec) for spec in sm.CONDITIONS
        }
        sm.check_environments_distinct(environments)
        reference = ProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(self.REFERENCE_STRAIN),
            environment_reference=environments[sm.REFERENCE_CONDITION],
            phenotype_reference=reference_phenotype(rows, keys),
        )
        pub = sm.publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        records: list[dict[str, Any]] = []
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for column, files in tqdm(condition_files().items(), desc="schmidt2016 S8"):
                for file_name in files:
                    phenotype = culture_phenotype(rows, keys, file_name)
                    experiment = ProteinAbundanceExperiment(
                        dataset_name=self.name,
                        genotype=Genotype(perturbations=[]),
                        environment=environments[column],
                        phenotype=phenotype,
                    )
                    txn.put(
                        f"{index}".encode(),
                        self._intern_record(experiment, reference, pub, itxn),
                    )
                    index += 1
                    records.append(
                        {
                            "s6_column": column,
                            "replicate_id": file_name,
                            "s8_column": sm._s8_column(file_name),
                            "is_reference_condition": column == sm.REFERENCE_CONDITION,
                            "n_protein_keys": len(phenotype.protein_abundance),
                        }
                    )
        env.close()
        interned_env.close()
        if index != EXPECTED_RECORDS:
            raise RuntimeError(f"{index} records, the module states {EXPECTED_RECORDS}")
        if len(rows) != EXPECTED_PROTEIN_KEYS:
            raise RuntimeError(
                f"{len(rows)} Table S8 rows, the module states {EXPECTED_PROTEIN_KEYS}"
            )
        self._write_ledgers(path, rows, keys, records, median_check)
        log.info(
            "Schmidt 2016 Table S8: %d culture records x %d protein keys",
            index,
            len(rows),
        )

    def _write_ledgers(
        self,
        path: str,
        rows: Sequence[IntensityRow],
        keys: Mapping[int, str],
        records: Sequence[Mapping[str, Any]],
        median_check: Mapping[str, Any],
    ) -> None:
        """The retention ledger, the sourcing table and the identifier table."""
        out = Path(self.preprocess_dir)
        unstored = unstored_columns(path)
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_normint_columns": len(records) + len(unstored),
                    "kept_records": len(records),
                    "records_by_condition": {
                        column: len(files)
                        for column, files in condition_files().items()
                    },
                    "source_protein_rows": len(rows),
                    "kept_protein_keys": len(rows),
                    "dropped_protein_rows": 0,
                    "not_stored": {
                        "columns": unstored,
                        "group": UNSTORED_GROUP_SUFFIX,
                        "rule": "technical_reanalysis_without_culture_pairing",
                        "description": (
                            "the second glucose triplicate re-analyzes the three "
                            "glucose samples (Supplementary Note quote in "
                            "sourced_values.json, glucose_2_is_a_reanalysis); the "
                            "release does not pair a repeat file with its culture, and "
                            "ProteinAbundancePhenotype has no technical-repeat field, so "
                            "the three columns wait for an owner decision (#1176)"
                        ),
                    },
                    "notes": [
                        f"{len(records)} records = 22 conditions x "
                        f"{N_CULTURES_PER_CONDITION} cultures, the three glucose "
                        "cultures included; every record carries every Table S8 row",
                        "the phenotype_reference is the released "
                        f"{REFERENCE_MEDIAN_COLUMN} (replicate_id None)",
                        "Table S9 (MG1655 / NCM3722 comparison) is not read by this "
                        "loader",
                    ],
                },
                indent=2,
            )
        )
        (out / "released_statistics_check.json").write_text(
            json.dumps({"reference_median": dict(median_check)}, indent=2)
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                },
                indent=2,
            )
        )
        pd.DataFrame(list(records)).to_csv(out / "records.csv", index=False)
        pd.DataFrame(
            [
                {
                    "uniprot": row.uniprot,
                    "released_gene_table_s8": row.released_gene,
                    "table_s8_row": row.row_number,
                    "stored_key": keys[row.row_number],
                }
                for row in rows
            ]
        ).to_csv(out / "protein_identifiers.csv", index=False)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
def culture_uniqueness_rule(records: Sequence[Mapping[str, Any]]) -> Any:
    """SUPPLEMENTARY L1: (environment, culture) is unique and each condition has three.

    Every genotype is the same empty wild type, so the environment and the culture's
    file name ARE the record identity.
    """
    from torchcell.verification.report import Level, LevelResult

    keys = [
        (
            json.dumps(record["experiment"]["environment"], sort_keys=True),
            record["experiment"]["phenotype"]["replicate_id"],
        )
        for record in records
    ]
    per_environment: dict[str, int] = {}
    for environment, _ in keys:
        per_environment[environment] = per_environment.get(environment, 0) + 1
    counts = sorted(set(per_environment.values()))
    passed = len(set(keys)) == len(records) and counts == [N_CULTURES_PER_CONDITION]
    return LevelResult(
        level=Level.L1,
        name="culture_uniqueness",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {len(set(keys))} distinct (environment, culture) keys over "
            f"{len(records)} records, {len(per_environment)} environments with "
            f"{counts} cultures each"
        ),
        details={
            "n_records": len(records),
            "n_distinct": len(set(keys)),
            "n_environments": len(per_environment),
            "cultures_per_environment": counts,
        },
    )


def verify_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the protein L0-L3 gate plus this module's rows and write the report."""
    from torchcell.verification.protein import verify_protein_dataset
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    if genome is None:
        genome = bacterial_genome("ecoli", sm.SCHMIDT_REFERENCE_STRAIN, data_root)
    report = verify_protein_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{sm.RAW_DIR_REL}/{sm.SI2_MIRROR_RELPATH}",
            citation_key=sm.CITATION_KEY,
            sha256=sm.SI2_SHA256,
            method=(
                "Table S8 'normInt_<file>.1' columns: one ProteinAbundanceExperiment "
                "per BW25113 culture (22 conditions x 3), the released normalized "
                "intensity verbatim; reference = released medianNormInt_Glucose.1"
            ),
            page="si2.xlsx sheets 'Table S8', 'Table S6', 'Table S25'",
            retrieved=sm.SI2_RETRIEVED_AT,
        ),
        expected_count=expected_count,
    )
    report.add(culture_uniqueness_rule(records))
    report.add(sm.assembly_pin_rule(records))
    report.add(sm.gene_containment_rule(records, set(genome.genbank.loci)))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev-tree LMDB, or ``verify`` an already built one."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.schmidt2016_replicate_intensity"
    )
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument(
        "--root", default=DATASET_ROOT_REL, help="relative to DATA_ROOT"
    )
    args = parser.parse_args(argv)
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, args.root)
    if args.command == "build":
        dataset = ProteomeReplicateIntensitySchmidt2016Dataset(root=root)
        print(f"len = {len(dataset)}")
        return 0
    report = verify_build(root, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
