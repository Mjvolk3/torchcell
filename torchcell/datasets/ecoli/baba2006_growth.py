# torchcell/datasets/ecoli/baba2006_growth
# [[torchcell.datasets.ecoli.baba2006_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/baba2006_growth
# Test file: tests/torchcell/datasets/ecoli/test_baba2006_growth.py
"""Baba 2006 Supplementary Table 3: the OD600 growth yield of every obtained Keio mutant.

Baba et al. 2006 (Mol Syst Biol 2:2006.0008, doi:10.1038/msb4100050, citation key
``babaConstructionEscherichiaColi2006``) grew each obtained BW25113 deletion mutant 22 h
in LB with kanamycin, then 24 h and 48 h in 0.4% glucose MOPS with 2 mM Pi and kanamycin,
and released the absorbance at 600 nm of each culture in Table 3's column ``8. cell
growth (OD600)`` (sub-columns ``LB_22hr``, ``MOPS_24hr``, ``MOPS_48hr``). The sibling
:mod:`torchcell.datasets.ecoli.baba2006` loads the same table's construction outcomes;
this module loads the growth arm (issue #927).

WHAT IS STORED. One ``EnvironmentResponseExperiment`` per released numeric OD600
cell, ``MeasurementType.optical_density`` (absolute, unnormalized), in one of three
environments that differ in medium and in ``duration_hours``. The record keeps the
released identity: the MG1655 ``b number`` of Table 3 as ``systematic_gene_name``
(``ecoli_k12_mg1655_bnumber``, stored as released, no crosswalk), the released ``gene``
as ``perturbed_gene_name``, and the Keio ``JW id`` as ``construction.strain_accession``.
The JW id is what tells apart the two Keio strains Table 3 lists for one b-number (dcuC
``JW0613`` / ``JW0616``, gatA ``JW2078`` / ``JW2081``, rcsC ``JW5917`` / ``JW5920``).
Counts, measured on the pinned table: 3,911 rows with a b-number carry three numeric
readings and 1 (rcsC ``JW5920``) carries two, so 11,735 records.

THE REFERENCE IS A TYPED GAP. Table 3 carries no BW25113 row, and the paper states no
wild-type OD600, so each environment's reference is the parent BW25113 on the same
medium without kanamycin (it carries no resistance marker; the paper grows it "with or
without antibiotics") with ``environment_response`` a ``ProvenanceGap``. The Results
compare mutants to "wild type"; that reading is not released.

NOT STORED, each counted in ``preprocess/not_stored.json``:

* 73 obtained rows whose ``ECK number``, ``gene`` and ``b number`` are all ``none`` and
  whose only identifiers are the JW id and W3110 coordinates (219 readings). Every
  bacterial gene-perturbation leaf keys on a locus tag, and none of the 73 JW ids is on
  the pinned BW25113 annotation (measured: ``resolve_gene_name`` returns ``retired`` for
  all 73), so the schema has no carrier for them. Owner question in the PR.
* rcsC ``JW5920``'s ``MOPS_48hr`` cell, released as the string ``NT`` and defined
  nowhere in the paper: not a reading.

Uncertainty: one reading per mutant per condition, and no replicate count or dispersion
per value is released (Figure 6's grey band is 2 x s.d. of the whole population), so the
uncertainty fields and ``n_samples`` are typed gaps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import xlrd
from pydantic import BaseModel, ConfigDict

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB_LENNOX
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    DeletionStrainPerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    StrainConstruction,
    Temperature,
)
from torchcell.datasets.bacteria_common import assembly_reference
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import baba2006 as bb
from torchcell.sequence.genome.ecoli.k12 import EcoliK12StrainName
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

log = logging.getLogger(__name__)

CITATION_KEY: Final = "babaConstructionEscherichiaColi2006"
DATASET_ROOT_REL: Final = "data/torchcell/growth_yield_baba2006"
TABLE_S3: Final = bb.TABLE_S3
TABLE_S3_SHA256: Final = bb.DATA_SHA256[bb.TABLE_S3]
MG1655_NAMESPACE: Final = "ecoli_k12_mg1655_bnumber"
REFERENCE_STRAIN: Final[EcoliK12StrainName] = "BW25113"

#: Table 3's growth sub-headers, by (row, column), verbatim.
GROWTH_HEADER: Final[dict[tuple[int, int], str]] = {
    (1, 17): "8. cell growth (OD600)",
    (3, 17): "LB_22hr",
    (3, 18): "MOPS_24hr",
    (3, 19): "MOPS_48hr",
}
ConditionName = Literal["LB_22hr", "MOPS_24hr", "MOPS_48hr"]
CONDITIONS: Final[tuple[tuple[ConditionName, int], ...]] = (
    ("LB_22hr", 17),
    ("MOPS_24hr", 18),
    ("MOPS_48hr", 19),
)
NOT_OBTAINED: Final = "N.A."
NO_IDENTIFIER: Final = "none"
B_NUMBER: Final = re.compile(r"^b\d{4}$")
NUMERIC_TEXT: Final = re.compile(r"^\d+(\.\d+)?$")

#: Build oracles, measured on the pinned table.
EXPECTED_RECORDS: Final = 11735
EXPECTED_ROWS_WITH_READING: Final = 3985
EXPECTED_NO_IDENTIFIER_ROWS: Final = 73
EXPECTED_NON_NUMERIC_CELLS: Final = {"JW5920 MOPS_48hr": "NT"}
#: Records the environment-response L3 rule counts as carrying no environmental edit:
#: no record has a perturbation, and the rule's baseline is the MODAL medium, which is
#: the MOPS medium (24 h: 3,912 records, 48 h: 3,911), so the 3,912 LB records are the
#: ones that differ from it. ``duration_hours`` is not part of the rule's baseline.
EXPECTED_UNPERTURBED: Final = 7823

UNITS: Final = "OD600 (absorbance at 600 nm of a 200-ul culture in a 96-well plate)"


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in PMC's sha256-pinned plain text."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=bb.PAPER_TEXT_REL,
            citation_key=bb.CITATION_KEY,
            sha256=bb.PAPER_TEXT_SHA256,
            method="PMC Article Datasets plain text of the article (raw mirror)",
        ),
    )


_LB_QUOTE: Final = (
    "Mutants were tested for growth in 200-μl LB medium with kanamycin as a rich medium "
    "in 96-well microplates inoculated directly with 96 inoculation pins (Genetix "
    "Limited, UK) and incubated for 22 h at 37°C without shaking."
)
_MOPS_QUOTE: Final = (
    "Mutants were transferred with 96-inoculation pins (Genetix Limited, UK) from LB "
    "into 200-μl 0.4% glucose MOPS medium with 2 mM Pi (Wanner, 1994) and kanamycin as "
    "minimal medium and incubated for 24 and 48 h at 37°C without shaking."
)
_ANTIBIOTIC_QUOTE: Final = (
    "with or without antibiotics at 50 μg/ml for ampicillin (Wako, Osaka, Japan) and 30 "
    "μg/ml for kanamycin (Wako, Osaka, Japan)"
)

SOURCED_VALUES: Final[dict[str, SourcedValue]] = {
    "lb_assay": _paper({"hours": 22.0, "celsius": 37.0}, _LB_QUOTE),
    "mops_assay": _paper(
        {"hours": (24.0, 48.0), "celsius": 37.0, "glucose_pct": 0.4, "pi_mm": 2.0},
        _MOPS_QUOTE,
        note="the rest of the MOPS composition is deferred by the paper to Wanner 1994, "
        "which is not in the mirror",
    ),
    "readout": _paper(
        "absorbance at 600 nm",
        "Absorbance at 600 nm was measured after mixing for 5 s in a 96-well plate "
        "reader (Molecular Dynamics).",
    ),
    "readout_mops": _paper(
        "absorbance at 600 nm", "Absorbances were similarly measured."
    ),
    "kanamycin_ug_per_ml": _paper(
        30.0,
        _ANTIBIOTIC_QUOTE,
        note="the paper's one statement of its kanamycin concentration; the growth "
        "assay says 'with kanamycin' and no other amount",
    ),
    "growth_yield": _paper(
        "growth yield",
        "All mutants were profiled for growth yield in both rich (LB) and minimal glucose "
        "MOPS media (Figure 6).",
    ),
    "table_3": _paper(
        "Supplementary Table 3", "Complete information is in Supplementary Table 3."
    ),
    "figure_6": _paper(
        "maximum cell density",
        "Maximum cell density values are plotted.",
        note="Figure 6's legend: the released value is a single end-point reading",
    ),
    "wild_type_comparison": _paper(
        "not released",
        "The vast majority showed no differences from wild type.",
        note="a wild-type comparison is described; no wild-type reading is released",
    ),
}

_TABLE_LOOKED_IN: Final = Provenance(
    source_uri=f"$DATA_ROOT/{bb.RAW_DIR_REL}/{bb.mirror_relpath(TABLE_S3)}",
    citation_key=bb.CITATION_KEY,
    sha256=TABLE_S3_SHA256,
    page="Supplementary Table 3, column 8",
)
_PAPER_LOOKED_IN: Final = Provenance(
    source_uri=bb.PAPER_TEXT_REL,
    citation_key=bb.CITATION_KEY,
    sha256=bb.PAPER_TEXT_SHA256,
    method="Results, Materials and methods and the Figure 6 legend read in full",
)


def _gap(
    field: str, note: str, looked_in: Provenance = _PAPER_LOOKED_IN
) -> ProvenanceGap:
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=looked_in,
        note=note,
    )


RECORD_GAPS: Final = (
    _gap(
        "environment_response_uncertainty",
        "one OD600 per mutant per condition; no per-value dispersion is released "
        "(Figure 6's grey band is 2 x s.d. of the whole population)",
    ),
    _gap("environment_response_se", "no dispersion is released to reduce to an SE"),
    _gap("n_samples", "no replicate count per value is stated"),
    _gap("sample_unit", "what one reading is beyond one culture is not stated"),
)
REFERENCE_VALUE_GAP: Final = _gap(
    "environment_response",
    "Table 3 has no BW25113 row and the paper states no wild-type OD600, although the "
    "Results compare the mutants to wild type",
    looked_in=_TABLE_LOOKED_IN,
)


# --------------------------------------------------------------------------- #
# Reading Table 3's growth columns
# --------------------------------------------------------------------------- #
class GrowthRow(BaseModel):
    """One obtained Keio mutant's released identity and its numeric readings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sheet_row: int
    eck: str
    gene: str
    jw_id: str
    b_number: str
    readings: dict[str, float]
    non_numeric: dict[str, str]


def _reading(value: Any) -> float | str | None:
    """A float, a numeric string as its float, ``None`` for ``N.A.``, else the text."""
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if text == NOT_OBTAINED:
        return None
    if NUMERIC_TEXT.match(text):
        return float(text)
    return text


def read_growth_rows(path: str | Path) -> list[GrowthRow]:
    """Every Table 3 row with at least one released OD600 cell that is not ``N.A.``."""
    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    if sheet.cell_value(0, 0) != bb.TABLE_S3_TITLE:
        raise bb.TableFormatError(f"Table 3 title {sheet.cell_value(0, 0)!r}")
    for (r, c), text in {**bb.TABLE_S3_HEADER, **GROWTH_HEADER}.items():
        if sheet.cell_value(r, c) != text:
            raise bb.TableFormatError(
                f"Table 3 cell ({r}, {c}) {sheet.cell_value(r, c)!r}"
            )
    rows: list[GrowthRow] = []
    for r in range(4, sheet.nrows):
        cells = {
            name: _reading(sheet.cell_value(r, column)) for name, column in CONDITIONS
        }
        if all(value is None for value in cells.values()):
            continue
        rows.append(
            GrowthRow(
                sheet_row=r + 1,
                eck=str(sheet.cell_value(r, 0)),
                gene=str(sheet.cell_value(r, 1)),
                jw_id=str(sheet.cell_value(r, 2)),
                b_number=str(sheet.cell_value(r, 8)),
                readings={k: v for k, v in cells.items() if isinstance(v, float)},
                non_numeric={k: v for k, v in cells.items() if isinstance(v, str)},
            )
        )
    return rows


def keyable(row: GrowthRow) -> bool:
    """A row whose released b-number is a locus tag the deletion leaf can hold."""
    return bool(B_NUMBER.match(row.b_number))


# --------------------------------------------------------------------------- #
# Environments and records (pure)
# --------------------------------------------------------------------------- #
def _kanamycin() -> MediaComponent:
    statement = SOURCED_VALUES["kanamycin_ug_per_ml"]
    return MediaComponent(
        compound=resolved_compound("kanamycin"),
        role=MediaComponentRole.selection_agent,
        concentration=Concentration(
            value=float(statement.value), unit=ConcentrationUnit.ug_per_ml
        ),
        provenance=[statement],
        note="selects for the deletion's kanamycin cassette",
    )


def lb_medium(*, kanamycin: bool) -> Media:
    """The paper's LB (the LB_LENNOX amounts), liquid, with or without kanamycin."""
    components = list(LB_LENNOX.components)
    if kanamycin:
        components.append(_kanamycin())
    return Media(
        name="LB, Lennox amounts, liquid"
        + (", with 30 ug/ml kanamycin" if kanamycin else "")
        + " (Baba 2006 growth assay)",
        state="liquid",
        is_synthetic=False,
        base_medium="LB",
        components=components,
        provenance=[bb.LB_FORMULATION_STATEMENT, SOURCED_VALUES["lb_assay"]],
    )


def mops_medium(*, kanamycin: bool) -> Media:
    """0.4% glucose MOPS with 2 mM Pi (Wanner 1994), with or without kanamycin."""
    statement = SOURCED_VALUES["mops_assay"]
    components = [
        MediaComponent(
            compound=resolved_compound("D-glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=Concentration(value=0.4, unit=ConcentrationUnit.percent_w_v),
            provenance=[statement],
        ),
        MediaComponent(
            compound=resolved_compound("phosphate"),
            role=MediaComponentRole.bulk_salt,
            concentration=Concentration(value=2.0, unit=ConcentrationUnit.millimolar),
            provenance=[statement],
            note="'2 mM Pi'; the salt it was added as is not stated",
        ),
    ]
    if kanamycin:
        components.append(_kanamycin())
    return Media(
        name="0.4% glucose MOPS medium with 2 mM Pi (Wanner 1994)"
        + (", with 30 ug/ml kanamycin" if kanamycin else "")
        + " (Baba 2006 growth assay)",
        state="liquid",
        is_synthetic=True,
        base_medium="MOPS_MINIMAL",
        components=components,
        provenance=[statement],
    )


def environment(condition: ConditionName, *, kanamycin: bool = True) -> Environment:
    """One of the three growth conditions at 37 C for its stated time.

    ``aerobicity`` is the field's default: the cultures were static 96-well plates
    ("without shaking") and the schema has no gap for an unstated oxygen regime.
    """
    if condition == "LB_22hr":
        media, hours, source = lb_medium(kanamycin=kanamycin), 22.0, "lb_assay"
    else:
        media = mops_medium(kanamycin=kanamycin)
        hours = 24.0 if condition == "MOPS_24hr" else 48.0
        source = "mops_assay"
    return Environment(
        media=media,
        temperature=Temperature(value=float(SOURCED_VALUES[source].value["celsius"])),
        duration_hours=hours,
    )


def phenotype(value: float) -> EnvironmentResponsePhenotype:
    """One released OD600 reading, verbatim, with its typed gaps."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.optical_density,
        assay_type=AssayType.liquid_od_growth,
        environment_response=value,
        units=UNITS,
        provenance_gaps=list(RECORD_GAPS),
    )


def reference_phenotype() -> EnvironmentResponsePhenotype:
    """The parent's OD600: not released, a typed gap on the value."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.optical_density,
        assay_type=AssayType.liquid_od_growth,
        environment_response=None,
        units=UNITS,
        provenance_gaps=[REFERENCE_VALUE_GAP, *RECORD_GAPS],
    )


def genotype(row: GrowthRow) -> Genotype:
    """The Keio deletion under its released b-number, gene and JW id."""
    return Genotype(
        perturbations=[
            DeletionStrainPerturbation(
                systematic_gene_name=row.b_number,
                perturbed_gene_name=row.gene,
                gene_namespace=MG1655_NAMESPACE,
                collection=bb.KEIO_COLLECTION,
                cassette=bb.KEIO_CASSETTE,
                construction=StrainConstruction(strain_accession=row.jw_id),
            )
        ]
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GrowthYieldBaba2006Dataset(ExperimentDataset):
    """Baba 2006 Table 3: 11,735 released OD600 readings of obtained Keio mutants."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize at the dev-tree root."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The one consumed file, Table 3."""
        return [TABLE_S3]

    def download(self) -> None:
        """Link Table 3 from the raw mirror after checking the manifest pin and sha256."""
        data_root = os.environ["DATA_ROOT"]
        relpath = bb.mirror_relpath(TABLE_S3)
        check_manifest_pin(
            relpath,
            bb.manifest_sha256(bb.load_manifest(data_root), relpath),
            TABLE_S3_SHA256,
        )
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(
            bb.raw_mirror_dir(data_root) / relpath,
            osp.join(self.raw_dir, TABLE_S3),
            TABLE_S3_SHA256,
        )

    @post_process
    def process(self) -> None:
        """One record per released numeric OD600 cell of a b-number row."""
        verify_raw_files(self.raw_dir, {TABLE_S3: TABLE_S3_SHA256})
        rows = read_growth_rows(osp.join(self.raw_dir, TABLE_S3))
        if len(rows) != EXPECTED_ROWS_WITH_READING:
            raise RuntimeError(
                f"{len(rows)} rows carry a reading, expected {EXPECTED_ROWS_WITH_READING}"
            )
        unkeyed = [row for row in rows if not keyable(row)]
        if len(unkeyed) != EXPECTED_NO_IDENTIFIER_ROWS or any(
            (row.eck, row.gene, row.b_number) != (NO_IDENTIFIER,) * 3 for row in unkeyed
        ):
            raise RuntimeError("the rows without a b-number are not the pinned 73")
        non_numeric = {
            f"{row.jw_id} {name}": text
            for row in rows
            for name, text in row.non_numeric.items()
        }
        if non_numeric != EXPECTED_NON_NUMERIC_CELLS:
            raise RuntimeError(f"non-numeric cells {non_numeric}")
        keyed = [row for row in rows if keyable(row)]

        genome_reference = assembly_reference(REFERENCE_STRAIN)
        references = {
            name: EnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=genome_reference,
                environment_reference=environment(name, kanamycin=False),
                phenotype_reference=reference_phenotype(),
            )
            for name, _ in CONDITIONS
        }
        environments = {name: environment(name) for name, _ in CONDITIONS}
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        index = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in keyed:
                for name, _ in CONDITIONS:
                    if name not in row.readings:
                        continue
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype(row),
                        environment=environments[name],
                        phenotype=phenotype(row.readings[name]),
                    )
                    txn.put(
                        f"{index}".encode(),
                        self._intern_record(
                            experiment, references[name], bb.PUBLICATION, itxn
                        ),
                    )
                    index += 1
        env.close()
        interned_env.close()
        if index != EXPECTED_RECORDS:
            raise RuntimeError(f"wrote {index} records, expected {EXPECTED_RECORDS}")
        self._write_ledgers(rows, keyed, unkeyed, index)
        log.info("Baba 2006 growth yield: %d records", index)

    def _write_ledgers(
        self,
        rows: Sequence[GrowthRow],
        keyed: Sequence[GrowthRow],
        unkeyed: Sequence[GrowthRow],
        n_records: int,
    ) -> None:
        """The not-stored ledger (with counts) and the sourcing table."""
        out = Path(self.preprocess_dir)
        out.mkdir(parents=True, exist_ok=True)
        per_condition = Counter(name for row in keyed for name in row.readings)
        (out / "not_stored.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "rows_with_a_reading": len(rows),
                    "stored_records": n_records,
                    "stored_per_condition": dict(per_condition),
                    "not_stored": [
                        {
                            "what": "obtained rows whose ECK number, gene and b number "
                            "are all 'none' (only a JW id and W3110 coordinates)",
                            "rows": len(unkeyed),
                            "readings": sum(len(r.readings) for r in unkeyed),
                            "why": "every bacterial gene-perturbation leaf keys on a "
                            "locus tag; none of these JW ids is on the pinned BW25113 "
                            "annotation, so no carrier exists (owner question)",
                            "jw_ids": [r.jw_id for r in unkeyed],
                        },
                        {
                            "what": "non-numeric released cells",
                            "cells": EXPECTED_NON_NUMERIC_CELLS,
                            "why": "'NT' is not a reading and the paper defines it "
                            "nowhere",
                        },
                    ],
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {k: v.model_dump(mode="json") for k, v in SOURCED_VALUES.items()},
                indent=2,
            )
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process()."""
        raise NotImplementedError(
            "GrowthYieldBaba2006Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
Record = Mapping[str, Any]

VERIFIER_PROVENANCE: Final = Provenance(
    source_uri=f"$DATA_ROOT/{bb.RAW_DIR_REL}/{bb.mirror_relpath(TABLE_S3)}",
    citation_key=bb.CITATION_KEY,
    sha256=TABLE_S3_SHA256,
    method="one EnvironmentResponseExperiment per released numeric OD600 cell "
    "of a Table 3 row with a b-number, MeasurementType.optical_density; the parent's "
    "reading is a typed gap on each of the three references",
    page="Supplementary Table 3, column 8",
)


def _l2_values_equal_table(
    records: Sequence[Record], rows: Sequence[GrowthRow]
) -> LevelResult:
    """L2: every stored reading equals its Table 3 cell, keyed by JW id and duration."""
    hours_to_condition = {22.0: "LB_22hr", 24.0: "MOPS_24hr", 48.0: "MOPS_48hr"}
    by_jw = {row.jw_id: row for row in rows}
    mismatched: list[str] = []
    for record in records:
        perturbation = record["experiment"]["genotype"]["perturbations"][0]
        jw = perturbation["construction"]["strain_accession"]
        condition = hours_to_condition[
            record["experiment"]["environment"]["duration_hours"]
        ]
        row = by_jw[jw]
        if (
            record["experiment"]["phenotype"]["environment_response"]
            != row.readings[condition]
            or perturbation["systematic_gene_name"] != row.b_number
            or perturbation["perturbed_gene_name"] != row.gene
        ):
            mismatched.append(f"{jw} {condition}")
    return LevelResult(
        level=Level.L2,
        name="value_and_identity_fidelity_to_table_3",
        passed=not mismatched,
        message=f"{len(records) - len(mismatched)} of {len(records)} records equal their "
        "released cell and identity",
        details={"mismatched": mismatched[:20]},
    )


def _l4_b_numbers_in_mg1655(
    records: Sequence[Record], universe: set[str]
) -> LevelResult:
    """L4 (a finding, never a gate): released b-numbers on the MG1655 annotation."""
    tags = {
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in records
    }
    missing = sorted(tags - universe)
    return LevelResult(
        level=Level.L4,
        name="released_b_numbers_on_mg1655_annotation",
        passed=True,
        message=f"{len(tags) - len(missing)} of {len(tags)} released b-numbers are MG1655 "
        f"gene rows; {len(missing)} are not (reported as a finding: the genome never "
        "gates storage)",
        details={"n_tags": len(tags), "missing": missing},
    )


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """The environment-response L0-L3 gate (absolute branch) plus L2 fidelity and L4."""
    from torchcell.datamodels.schema import BACTERIAL_ASSEMBLY_SETS
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import (
        BACTERIAL_GENE_ASSEMBLIES,
        _bacterial_gene_set,
        load_records,
    )

    base = data_root or os.environ["DATA_ROOT"]
    records = load_records(dataset_root)
    report = verify_environment_response_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=VERIFIER_PROVENANCE,
        expected_count=EXPECTED_RECORDS,
        reference_centered=False,
        expected_unperturbed=EXPECTED_UNPERTURBED,
        expected_gapped_reference=EXPECTED_RECORDS,
    )
    rows = read_growth_rows(bb.raw_mirror_dir(base) / bb.mirror_relpath(TABLE_S3))
    report.add(_l2_values_equal_table(records, rows))
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, Path(base) / "torchcell-raw"))
    universe = _bacterial_gene_set(
        BACTERIAL_GENE_ASSEMBLIES[BACTERIAL_ASSEMBLY_SETS["MG1655"]], base
    )
    report.add(_l4_b_numbers_in_mg1655(records, universe))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev store or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.baba2006_growth"
    )
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        print(f"len = {len(GrowthYieldBaba2006Dataset(root=root))}")
        return 0
    report = verify_build(root, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
