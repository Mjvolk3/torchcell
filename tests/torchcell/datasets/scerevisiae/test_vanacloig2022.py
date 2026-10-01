# tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
# [[tests.torchcell.datasets.scerevisiae.test_vanacloig2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
"""Vanacloig-Pedros 2022 loader: retention rules, sourced encoding, batch-matched control.

Everything but the mirror audit runs on a synthetic count matrix and a fake genome, so
no build tree and no network are needed. The audit test binds each module-level
``SourcedValue`` to its verbatim quote in the sha256-pinned mirror and skips when the
mirror is not mounted.

2026.09.30 (Phase 13): a gzipped TSV in the GEO GSE186866 layout (``gene`` =
``<ORF>_<barcode>``, ``std_name``, then ``ControlN_CG00b`` and ``<token>_CG00b_repN``
count columns) read by the real ``_load_matrix``. Two batches with two controls each;
six compound tokens (Furfural, MMS, SodiumGlyoxylate, DMSO, MBO, QUADRIS1) with three
replicate columns each (batches 001, 002, 001, except MMS 001, 002, 002). Eight rows:
YAL001C, YAL002W, YAL003W (legacy spelling of YAL002W), YPL999C (retired), YGL013C (a
3DeltaAlpha background gene), YBR001C (one NaN count), YBR002C (no barcode) and a
non-ORF ``Filler`` row whose counts bring EVERY column total to 1,000,000, so CPM equals
the raw count (``c / 1e6 * 1e6 == c`` exactly for these integers, checked in Python).

Expected values, with ``L(x) = log2(x + 1)``:

- YAL001C / Furfural (paired): controls CG001 (3, 3) -> L = 2, CG002 (7, 7) -> L = 3;
  replicates 15 (CG001), 63 (CG002), 63 (CG001) -> 4 - 2, 6 - 3, 6 - 2 = 2, 3, 4;
  response 3.0, sample SD 1.0, SE 1/sqrt(3).
- YAL001C / MMS (pooled over all four controls, mean 5 -> log2 6): replicates 11, 23,
  47 -> log2(12 / 6), log2(24 / 6), log2(48 / 6) = 1, 2, 3; response 2.0, SD 1.0. Paired
  controls would have given log2(12) - 2 for replicate 1, not an integer.
- YAL001C / SodiumGlyoxylate: replicates equal their batch controls; 0.0, SD 0.0.
- YAL002W: controls all 0; Furfural 0, 1, 3 -> 0, 1, 2; response 1.0, SD 1.0. MMS and
  SodiumGlyoxylate are all-zero cells and are dropped.
- YBR002C: every count 1, so every ratio is 0; kept with barcode ``""`` and its own
  systematic name as ``perturbed_gene_name`` (no standard name in the genome).

Records, compound-major over the kept compounds sorted (Furfural, MMS,
SodiumGlyoxylate) then kept rows: 0 YAL001C/Furfural, 1 YAL002W/Furfural, 2
YBR002C/Furfural, 3 YAL001C/MMS, 4 YBR002C/MMS, 5 YAL001C/SodiumGlyoxylate, 6
YBR002C/SodiumGlyoxylate. Ledger: source 8 rows x 6 tokens = 48; vehicle DMSO 1 x 8 = 8;
unidentified MBO, QUADRIS1 2 x 8 = 16; non-ORF 1 + one-NaN 1 + background 1 = 3 rows x 3
kept compounds = 9; retired YPL999C 3; legacy YAL003W 3; all-zero cells 2; so 41
dropped and 7 kept. Every reference is the inhibitor-free control, so Furfural and
SodiumGlyoxylate share one (paired units) and MMS has its own (pooled units).

Findings pinned (issue #501 is the ingestion audit; source lines in
``vanacloig2022.py``): the stored value is CPM against the library size summed over
every released row, not the paper's TMM (#501 finding 1; lines 865-868); an identified
compound the paper never reported (SodiumGlyoxylate) is served (#501 finding 2); DMSO is
dropped as the vehicle and a DMSO-delivered compound's environment names no DMSO (#501
finding 3; line 746); MBO is dropped as unidentified (#501 finding 5; lines 747-752); a
single NaN count drops the whole row under the rule described as "every count column is
missing" (line 760 versus line 811); a row with no barcode is served with ``barcode ""``
(line 758); a replicate set whose counts are equal but nonzero keeps SD exactly 0 (only
all-zero cells are dropped, line 904).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import os.path as osp
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import MEDIA_LIBRARY, SYNBASE
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    MarkerDeletionPerturbation,
    MeasurementType,
    NatMxDeletionPerturbation,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import vanacloig2022 as v
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

# The synthetic library: two current genes, one retired ORF, one legacy spelling of a
# gene the library ALSO carries under its current name.
_GENES = {"YAL001C", "YAL002W", "YBR001C"}
_STANDARD = {"TFC3": ["YAL001C"], "VPS8": ["YAL002W"], "SPO23": ["YBR001C"]}
_RENAMED = {"YAL003W": "YAL002W"}  # legacy spelling of a gene already in the pool


class _Resolution:
    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _FakeGenome:
    """The slice of ``SCerevisiaeGenome`` the loader's gene-name policy reads."""

    gene_set = _GENES
    feature_index = {"standard_to_ids": _STANDARD}

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.upper()
        if upper in _GENES:
            return _Resolution("current", upper)
        if upper in _RENAMED:
            return _Resolution("renamed", _RENAMED[upper])
        for standard, ids in _STANDARD.items():
            if upper == standard:
                return _Resolution("renamed", ids[0])
        return _Resolution("retired", upper)


def _matrix() -> pd.DataFrame:
    """Four library rows x (2 controls per batch + 2 compounds x 3 reps)."""
    rows = [
        ("YAL001C_AAAACCCCGGGGTTTTACGT", "TFC3"),
        ("YAL002W_CCCCGGGGTTTTAAAACGTA", "VPS8"),
        ("YAL003W_GGGGTTTTAAAACCCCGTAC", "YAL003W"),  # legacy duplicate -> dropped
        ("YPL999C_TTTTAAAACCCCGGGGTACG", "NOPE"),  # retired -> dropped
    ]
    data: dict[str, Any] = {
        "gene": [r[0] for r in rows],
        "std_name": [r[1] for r in rows],
        "Control1_CG003": [100, 200, 300, 400],
        "Control2_CG003": [120, 220, 320, 420],
        "Control1_CG004": [400, 300, 200, 100],
        "Control2_CG004": [420, 320, 220, 120],
        # Benomyl: one replicate per batch (paired), row 1 is an all-zero cell
        "Benomyl_CG003_rep1": [110, 0, 310, 410],
        "Benomyl_CG004_rep2": [410, 0, 210, 110],
        "Benomyl_CG003_rep3": [130, 0, 330, 430],
        # DMSO: the vehicle control column, dropped whole
        "DMSO_CG003_rep1": [90, 190, 290, 390],
        "DMSO_CG004_rep2": [390, 290, 190, 90],
        "DMSO_CG003_rep3": [95, 195, 295, 395],
        # MBO: no structure identifier resolves, dropped whole
        "MBO_CG003_rep1": [70, 170, 270, 370],
        "MBO_CG004_rep2": [370, 270, 170, 70],
        "MBO_CG003_rep3": [75, 175, 275, 375],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def built(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the synthetic dataset end to end into a temporary root."""
    monkeypatch.setattr(v, "default_genome", lambda: _FakeGenome())
    monkeypatch.setattr(v.EnvChemgenVanacloig2022Dataset, "download", lambda self: None)
    monkeypatch.setattr(
        v.EnvChemgenVanacloig2022Dataset, "_load_matrix", lambda self: _matrix()
    )
    root = str(tmp_path / "env_chemgen_vanacloig2022")
    os.makedirs(osp.join(root, "raw"), exist_ok=True)
    with gzip.open(osp.join(root, "raw", v.DATA_FILENAME), "wt") as handle:
        handle.write("placeholder\n")  # presence only; _load_matrix is patched
    return v.EnvChemgenVanacloig2022Dataset(root=root)


def test_drop_rules_account_for_every_source_record(built: Any) -> None:
    log = v.DropLog.model_validate_json(
        open(osp.join(built.root, "preprocess", "dropped_records.json")).read()
    )
    rules = {rule.rule: rule for rule in log.rules}
    assert rules["vehicle_control_served_as_a_treatment"].items == ["DMSO"]
    assert rules["compound_without_a_structure_identifier"].items == ["MBO"]
    assert rules["orf_is_not_a_current_genome_gene"].items == ["YPL999C"]
    assert rules["orf_is_a_legacy_spelling_of_another_library_orf"].items == ["YAL003W"]
    # one compound (Benomyl) x two kept rows, minus the all-zero cell
    assert rules["all_three_replicate_counts_are_zero"].n_records == 1
    assert log.source_records == 4 * 3
    assert log.kept_records == 1
    assert log.dropped_records == sum(rule.n_records for rule in log.rules)
    assert len(built) == log.kept_records


def test_record_carries_the_barcode_the_canonical_name_and_the_shared_medium(
    built: Any,
) -> None:
    record = built[0]
    experiment = record["experiment"]
    deletion = experiment["genotype"]["perturbations"][0]
    assert deletion["perturbation_type"] == "barcoded_kanmx_deletion"
    assert deletion["systematic_gene_name"] == "YAL001C"
    assert (
        deletion["perturbed_gene_name"] == "TFC3"
    )  # the genome's spelling, not std_name
    assert deletion["barcode"] == "AAAACCCCGGGGTTTTACGT"
    assert deletion["collection"] == v.LIBRARY_COLLECTION.value
    # the three constant background deletions ride on every genotype
    assert {
        p["systematic_gene_name"] for p in experiment["genotype"]["perturbations"]
    } == {"YAL001C", "YGL013C", "YBL005W", "YDR011W"}
    assert experiment["environment"]["media"] == MEDIA_LIBRARY["SYNBASE"].model_dump()
    assert experiment["environment"]["aerobicity"] == "anaerobic"
    assert experiment["environment"]["duration_hours"] == 48.0
    assert experiment["environment"]["duration_generations"] == 6.5


def test_environment_carries_the_compound_and_a_typed_ph(built: Any) -> None:
    perturbations = built[0]["experiment"]["environment"]["perturbations"]
    compound = next(
        p for p in perturbations if p["perturbation_type"] == "small_molecule"
    )
    assert compound["compound"]["name"] == "benomyl"
    assert compound["compound"]["inchikey"] == "RIOXQFHNBCKOKP-UHFFFAOYSA-N"
    assert compound["concentration"]["value"] == 10.0
    assert compound["concentration"]["unit"] == "ug/mL"
    assert compound["concentration"]["basis"] == "fixed"
    # the vehicle is the field that is actually None, so it is the one that is gapped;
    # the dose is not gapped, because an IC30 / fixed basis is always known
    assert compound["solvent"] is None
    assert [g["field"] for g in compound["provenance_gaps"]] == ["solvent"]
    assert compound["provenance_gaps"][0]["reason"] == "deferred_pending_source_review"
    assert compound["provenance_gaps"][0]["resolve_with"]["page"] == "Table S1"
    ph = next(
        p for p in perturbations if p["perturbation_type"] == "environment_physical"
    )
    assert ph["factor"] == "pH"
    assert ph["magnitude"]["value"] == 5.0 and ph["magnitude"]["unit"] == "pH"
    assert ph["agent"]["name"] == "hydrochloric acid"  # the acid the paper names
    assert ph["provenance_gaps"] == []


def test_reference_environment_holds_no_inhibitor(built: Any) -> None:
    reference = built[0]["reference"]
    types = {
        p["perturbation_type"]
        for p in reference["environment_reference"]["perturbations"]
    }
    assert types == {"environment_physical"}  # pH only: the control is inhibitor-free
    assert reference["phenotype_reference"]["environment_response"] == 0.0
    assert reference["genome_reference"]["strain"] == "S288C"


def test_response_is_the_batch_matched_log2_ratio_with_a_nonzero_sample_sd(
    built: Any,
) -> None:
    phenotype = built[0]["experiment"]["phenotype"]
    assert phenotype["measurement_type"] == "log2_ratio"
    assert phenotype["assay_type"] == "pooled_competitive_growth_barcode"
    assert phenotype["n_samples"] == 3
    assert phenotype["sample_unit"] == "biological_replicate"
    assert (
        phenotype["environment_response_uncertainty_type"] == UncertaintyType.sample_sd
    )
    assert phenotype["environment_response_uncertainty"] > 0.0
    assert phenotype["environment_response_se"] == pytest.approx(
        phenotype["environment_response_uncertainty"] / math.sqrt(3)
    )
    assert "SAME CG batch" in phenotype["units"]


def test_unpaired_compound_keeps_the_pooled_control_in_its_units() -> None:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    assert "pooled rather than batch-matched" in dataset._units("MMS")
    assert "SAME CG batch" in dataset._units("Furfural")
    # MMS's dose was published, not set to an IC30, but its unit is not stated
    mms = dataset._concentration("MMS")
    assert mms.value is None and mms.basis is DoseBasis.fixed
    assert dataset._concentration("Furfural").basis is DoseBasis.IC30


def test_resolve_library_rows_separates_retired_from_legacy_duplicates() -> None:
    rows = v.resolve_library_rows(
        pd.Series(["YAL001C", "YAL002W", "YAL003W", "YPL999C"]),
        pd.Series(["AAAA", "TTTT", "CCCC", "GGGG"]),
        _FakeGenome(),  # type: ignore[arg-type]  # the slice of the genome API used
    )
    assert rows.keep_mask == [True, True, False, False]
    assert rows.systematic == ["YAL001C", "YAL002W"]
    assert rows.common == ["TFC3", "VPS8"]
    assert rows.dropped_retired == ["YPL999C"]
    assert rows.dropped_legacy_duplicate == ["YAL003W"]


def test_every_sourced_value_is_backed_by_a_verbatim_quote_in_the_mirror() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT not set")
    library = osp.join(data_root, "torchcell-library")
    if not osp.isdir(osp.join(library, v.CITATION_KEY)):
        pytest.skip("paper mirror not mounted")
    values = [
        getattr(v, name)
        for name in dir(v)
        if isinstance(getattr(v, name), SourcedValue)
    ]
    assert len(values) >= 12
    for value in values:
        assert audit_sourced_value(value, library).passed


def test_perturbation_leaves_are_the_typed_ones() -> None:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    genotype = dataset._genotype("YAL001C", "TFC3", "ACGT")
    assert isinstance(genotype.perturbations[0], BarcodedKanMxDeletionPerturbation)
    environment = dataset._environment("Furfural")
    assert isinstance(environment.perturbations[0], SmallMoleculePerturbation)
    assert isinstance(environment.perturbations[1], EnvironmentPhysicalPerturbation)
    assert environment.perturbations[1].factor is PhysicalFactor.ph


# --------------------------------------------------------------------------- #
# Phase 13: the GEO-layout matrix, the exact records, the ledger, the refusals
# --------------------------------------------------------------------------- #
_CONTROLS = ["Control1_CG001", "Control2_CG001", "Control1_CG002", "Control2_CG002"]
_BATCHES = {"MMS": ("001", "002", "002")}
_TOKENS = ["Furfural", "MMS", "SodiumGlyoxylate", "DMSO", "MBO", "QUADRIS1"]


def _replicates(token: str) -> list[str]:
    batches = _BATCHES.get(token, ("001", "002", "001"))
    return [f"{token}_CG{b}_rep{i}" for i, b in enumerate(batches, start=1)]


_COLUMNS = _CONTROLS + [c for token in _TOKENS for c in _replicates(token)]
_FILLED = 5.0  # DMSO, MBO and QUADRIS1 counts for every non-filler row


def _row(
    gene: str, std: str, counts: dict[str, list[float]], default: float
) -> dict[str, Any]:
    """One matrix row: named groups of columns, every other column at ``default``."""
    row: dict[str, Any] = {"gene": gene, "std_name": std}
    row.update(dict.fromkeys(_COLUMNS, default))
    for group, values in counts.items():
        names = _CONTROLS if group == "controls" else _replicates(group)
        row.update(zip(names, values, strict=True))
    for token in ("DMSO", "MBO", "QUADRIS1"):
        row.update(dict.fromkeys(_replicates(token), _FILLED))
    return row


def _geo_matrix() -> pd.DataFrame:
    rows = [
        _row(
            "YAL001C_AAAACCCC",
            "TFC3",
            {
                "controls": [3, 3, 7, 7],
                "Furfural": [15, 63, 63],
                "MMS": [11, 23, 47],
                "SodiumGlyoxylate": [3, 7, 3],
            },
            0,
        ),
        _row(
            "YAL002W_CCCCGGGG",
            "VPS8",
            {
                "controls": [0, 0, 0, 0],
                "Furfural": [0, 1, 3],
                "MMS": [0, 0, 0],
                "SodiumGlyoxylate": [0, 0, 0],
            },
            0,
        ),
        _row("YAL003W_GGGGTTTT", "YAL003W", {}, 2),
        _row("YPL999C_TTTTAAAA", "NOPE", {}, 2),
        _row("YGL013C_ACGTACGT", "PDR1", {}, 2),
        _row("YBR001C_TTAATTAA", "SPO23", {}, 2),
        _row("YBR002C", "YBR002C", {}, 1),
    ]
    rows[5]["Control1_CG001"] = float("nan")
    frame = pd.DataFrame(rows)
    filler: dict[str, Any] = {"gene": "Filler_row", "std_name": "none"}
    for column in _COLUMNS:
        filler[column] = 1_000_000 - frame[column].sum(skipna=True)
    matrix = pd.concat([frame, pd.DataFrame([filler])], ignore_index=True)
    # The fixture's premise: library size 1e6 in every column, so CPM equals the count.
    assert {float(matrix[c].sum(skipna=True)) for c in _COLUMNS} == {1_000_000.0}
    return matrix


_GEO_GENES = {"YAL001C", "YAL002W", "YBR001C", "YBR002C"}


class _GeoGenome(_FakeGenome):
    """``_FakeGenome`` with YBR002C, a current gene with no standard name."""

    gene_set = _GEO_GENES

    def resolve_gene_name(self, name: str) -> _Resolution:
        if name == "YBR002C":
            return _Resolution("current", name)
        return super().resolve_gene_name(name)


def _write_matrix(path: str, frame: pd.DataFrame) -> None:
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, sep="\t", index=False)


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> Any:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(v, "default_genome", lambda: _GeoGenome())
    root = tmp_path / "env_chemgen_vanacloig2022"
    (root / "raw").mkdir(parents=True)
    _write_matrix(str(root / "raw" / v.DATA_FILENAME), frame)
    return v.EnvChemgenVanacloig2022Dataset(root=str(root))


@pytest.fixture()
def geo_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The eight-row matrix of the module docstring, built end to end."""
    return _build(tmp_path, monkeypatch, _geo_matrix())


def _solvent_gap() -> ProvenanceGap:
    return ProvenanceGap(
        field="solvent",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        resolve_with=v.TABLE_S1,
        note=v.VEHICLE_CONTROL.quote
        + " Which compounds that covers is in Table S1, which is not mirrored, so the "
        "vehicle of any one compound is unknown rather than absent.",
    )


def _ph() -> EnvironmentPhysicalPerturbation:
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.ph,
        magnitude=Concentration(value=5.0, unit=ConcentrationUnit.ph),
        agent=resolved_compound("hydrochloric acid"),
    )


def _synbase(perturbations: list[Any]) -> Environment:
    return Environment(
        media=SYNBASE,
        temperature=Temperature(value=30.0),
        perturbations=perturbations,
        aerobicity="anaerobic",
        duration_hours=48.0,
        duration_generations=6.5,
    )


def _genotype(systematic: str, common: str, barcode: str) -> Genotype:
    return Genotype(
        perturbations=[
            BarcodedKanMxDeletionPerturbation(
                systematic_gene_name=systematic,
                perturbed_gene_name=common,
                barcode=barcode,
                collection="3DeltaAlpha drug-sensitive yeast deletion collection of "
                "4309 mutants",
            ),
            NatMxDeletionPerturbation(
                systematic_gene_name="YGL013C", perturbed_gene_name="PDR1"
            ),
            MarkerDeletionPerturbation(
                systematic_gene_name="YBL005W",
                perturbed_gene_name="PDR3",
                marker="KlURA3",
            ),
            MarkerDeletionPerturbation(
                systematic_gene_name="YDR011W",
                perturbed_gene_name="SNQ2",
                marker="KlLEU2",
            ),
        ]
    )


def _phenotype(response: float, sd: float, units: str) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=response,
        environment_response_uncertainty=sd,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=3,
        sample_unit=SampleUnit.biological_replicate,
        units=units,
    )


def _reference(units: str) -> dict[str, Any]:
    return EnvironmentResponseExperimentReference(
        dataset_name="EnvChemgenVanacloig2022Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=_synbase([_ph()]),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            units=units,
        ),
    ).model_dump()


def test_records_are_compound_major_with_the_closed_form_responses(
    geo_built: Any,
) -> None:
    """(screened ORF, compound, response, SD) for all seven records, in LMDB order.

    ``Genotype`` stores perturbations sorted by systematic name, so the screened
    deletion is found by its type, not its position (YBL005W sorts before YBR002C).
    """
    got = []
    for i in range(len(geo_built)):
        experiment = geo_built[i]["experiment"]
        compound = experiment["environment"]["perturbations"][0]["compound"]["name"]
        (screened,) = [
            p
            for p in experiment["genotype"]["perturbations"]
            if p["perturbation_type"] == "barcoded_kanmx_deletion"
        ]
        got.append(
            (
                screened["systematic_gene_name"],
                compound,
                experiment["phenotype"]["environment_response"],
                experiment["phenotype"]["environment_response_uncertainty"],
            )
        )
    assert got == [
        ("YAL001C", "furfural", 3.0, 1.0),
        ("YAL002W", "furfural", 1.0, 1.0),
        ("YBR002C", "furfural", 0.0, 0.0),
        ("YAL001C", "methyl methanesulfonate", 2.0, 1.0),
        ("YBR002C", "methyl methanesulfonate", 0.0, 0.0),
        ("YAL001C", "sodium glyoxylate", 0.0, 0.0),
        ("YBR002C", "sodium glyoxylate", 0.0, 0.0),
    ]


def test_full_paired_record_equals_the_hand_built_experiment(geo_built: Any) -> None:
    """Record 0, YAL001C / Furfural: CPM log2 ratio against its OWN batch's controls.

    Finding (#501 finding 1): the value is CPM-normalized against the library size
    summed over every released row, filler and dropped rows included (lines 865-868),
    not the paper's edgeR TMM. Finding (#501 finding 3): the environment carries no
    DMSO; the vehicle is only a typed ``solvent`` gap. Pinned until #501 is resolved.
    """
    experiment = EnvironmentResponseExperiment(
        dataset_name="EnvChemgenVanacloig2022Dataset",
        genotype=_genotype("YAL001C", "TFC3", "AAAACCCC"),
        environment=_synbase(
            [
                SmallMoleculePerturbation(
                    compound=resolved_compound("Furfural"),
                    concentration=Concentration(basis=DoseBasis.IC30),
                    provenance_gaps=[_solvent_gap()],
                ),
                _ph(),
            ]
        ),
        phenotype=_phenotype(3.0, 1.0, v._PAIRED_UNITS),
    )
    record = geo_built[0]
    assert record["experiment"] == experiment.model_dump()
    assert record["experiment"]["phenotype"]["environment_response_se"] == (
        1.0 / math.sqrt(3)
    )
    assert record["reference"] == _reference(v._PAIRED_UNITS)
    assert (
        record["publication"]
        == Publication(
            doi="10.1093/femsyr/foac036",
            doi_url="https://doi.org/10.1093/femsyr/foac036",
        ).model_dump()
    )


def test_pooled_mms_edge_record_equals_the_hand_built_experiment(
    geo_built: Any,
) -> None:
    """Record 3, YAL001C / MMS: the pooled control and the fixed, unitless MMS dose."""
    experiment = EnvironmentResponseExperiment(
        dataset_name="EnvChemgenVanacloig2022Dataset",
        genotype=_genotype("YAL001C", "TFC3", "AAAACCCC"),
        environment=_synbase(
            [
                SmallMoleculePerturbation(
                    compound=resolved_compound("MMS"),
                    concentration=Concentration(basis=DoseBasis.fixed),
                    provenance_gaps=[_solvent_gap()],
                ),
                _ph(),
            ]
        ),
        phenotype=_phenotype(2.0, 1.0, v._POOLED_UNITS),
    )
    assert geo_built[3]["experiment"] == experiment.model_dump()
    assert geo_built[3]["reference"] == _reference(v._POOLED_UNITS)


def test_a_row_without_a_barcode_is_served_with_an_empty_barcode(
    geo_built: Any,
) -> None:
    """Finding: ``YBR002C`` has no ``_<barcode>`` suffix; ``fillna("")`` (line 758)
    serves it with ``barcode ""`` instead of refusing it, and with no standard name the
    genome's canonical-name map falls back to the systematic name. Its equal nonzero
    counts give SD exactly 0, which the all-zero rule (line 904) does not catch. Pinned
    until a barcodeless row is dropped with a reason.
    """
    experiment = EnvironmentResponseExperiment(
        dataset_name="EnvChemgenVanacloig2022Dataset",
        genotype=_genotype("YBR002C", "YBR002C", ""),
        environment=_synbase(
            [
                SmallMoleculePerturbation(
                    compound=resolved_compound("SodiumGlyoxylate"),
                    concentration=Concentration(basis=DoseBasis.IC30),
                    provenance_gaps=[_solvent_gap()],
                ),
                _ph(),
            ]
        ),
        phenotype=_phenotype(0.0, 0.0, v._PAIRED_UNITS),
    )
    assert geo_built[6]["experiment"] == experiment.model_dump()


def test_drop_ledger_is_written_exactly(geo_built: Any) -> None:
    """Finding (#501 findings 2, 3, 5): SodiumGlyoxylate, a compound the paper never
    reported, is served because it resolves; DMSO is dropped as the vehicle; MBO is
    dropped as unidentified. Finding: YBR001C has ONE NaN count yet is dropped under a
    rule described as "every count column is missing" (``any`` at line 760). Pinned
    until #501 and the NaN rule are resolved.
    """
    log = json.loads(
        (Path(geo_built.preprocess_dir) / "dropped_records.json").read_text()
    )
    rules = [(r["rule"], r["scope"], r["n_records"], r["items"]) for r in log["rules"]]
    assert (log["dataset"], log["source_records"], log["kept_records"]) == (
        "EnvChemgenVanacloig2022Dataset",
        48,
        7,
    )
    assert log["dropped_records"] == 41
    assert rules == [
        ("vehicle_control_served_as_a_treatment", "compound", 8, ["DMSO"]),
        (
            "compound_without_a_structure_identifier",
            "compound",
            16,
            ["MBO", "QUADRIS1"],
        ),
        ("row_is_not_a_barcoded_orf_or_carries_no_counts", "library_row", 9, []),
        ("orf_is_not_a_current_genome_gene", "library_row", 3, ["YPL999C"]),
        (
            "orf_is_a_legacy_spelling_of_another_library_orf",
            "library_row",
            3,
            ["YAL003W"],
        ),
        ("all_three_replicate_counts_are_zero", "cell", 2, []),
    ]
    assert log["rules"][2]["description"].startswith(
        "the gene column is not '<systematic ORF>_<barcode>', or every count column "
        "is missing (a QC-dropped barcode)"
    )


def test_reference_index_splits_paired_from_pooled_and_the_gene_set(
    geo_built: Any,
) -> None:
    """The inhibitor-free reference is shared across paired compounds; MMS differs
    only in its units string. The gene set includes the three background genes.
    """
    index = geo_built.experiment_reference_index
    assert index is not None
    by_units = {e.reference.phenotype_reference.units: e.member_indices for e in index}
    assert by_units == {v._PAIRED_UNITS: [0, 1, 2, 5, 6], v._POOLED_UNITS: [3, 4]}
    assert json.loads(
        (Path(geo_built.preprocess_dir) / "gene_set.json").read_text()
    ) == ["YAL001C", "YAL002W", "YBL005W", "YBR002C", "YDR011W", "YGL013C"]


def _with_columns(extra: dict[str, float], drop: list[str]) -> pd.DataFrame:
    frame = _geo_matrix().drop(columns=drop)
    for column, value in extra.items():
        frame[column] = value
    return frame


def test_an_unparseable_sample_column_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _with_columns({"Furfural_rep4": 1.0}, [])
    with pytest.raises(
        RuntimeError, match="^unparseable sample column: 'Furfural_rep4'$"
    ):
        _build(tmp_path, monkeypatch, frame)


def test_a_matrix_without_control_columns_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _with_columns({}, _CONTROLS)
    with pytest.raises(
        RuntimeError, match="^no ControlN_CG\\* columns found in GSE186866 matrix$"
    ):
        _build(tmp_path, monkeypatch, frame)


def test_a_kept_compound_with_two_replicates_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _with_columns({}, ["Furfural_CG001_rep3"])
    with pytest.raises(
        RuntimeError, match="^Furfural: 2 replicate columns, expected 3$"
    ):
        _build(tmp_path, monkeypatch, frame)


def test_two_barcodes_of_one_orf_refuse_as_a_merged_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The duplicate identifier: a second barcode of YAL001C would merge two strains."""
    frame = _geo_matrix()
    duplicate = frame.iloc[[0]].copy()
    duplicate["gene"] = "YAL001C_TTTTGGGG"
    frame = pd.concat([duplicate, frame], ignore_index=True)
    with pytest.raises(RuntimeError) as err:
        _build(tmp_path, monkeypatch, frame)
    assert str(err.value) == (
        "two retained library rows resolve to the same systematic gene; the "
        "legacy-duplicate rule did not separate them"
    )


def test_canonical_common_names_keep_the_first_standard_name_that_resolves() -> None:
    """Only names resolving back to a current gene count; the first one wins."""

    class _Aliases(_FakeGenome):
        feature_index = {
            "standard_to_ids": {"TFC3": ["YAL001C"], "ALT3": ["YAL001C"], "GONE": []}
        }

        def resolve_gene_name(self, name: str) -> _Resolution:
            if name == "ALT3":
                return _Resolution("renamed", "YAL001C")
            if name == "GONE":
                return _Resolution("retired", "GONE")
            return super().resolve_gene_name(name)

    assert v._canonical_common_names(_Aliases()) == {"YAL001C": "TFC3"}  # type: ignore[arg-type]  # the slice of the genome API used


def _mirror(data_root: Path, payload: bytes, digest: str | None) -> Path:
    """A raw mirror holding ``payload`` and a manifest recording ``digest``."""
    root = v.raw_mirror_dir(str(data_root))
    (root / "data").mkdir(parents=True)
    (root / v.DATA_REL).write_bytes(payload)
    files = []
    if digest is not None:
        files.append(
            {
                "path": v.DATA_REL,
                "role": "raw_data",
                "bytes": len(payload),
                "sha256": digest,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"citation_key": v.CITATION_KEY, "files": files})
    )
    return root / v.DATA_REL


def _bare_dataset(root: Path) -> Any:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.root = str(root)
    return dataset


def test_download_links_the_verified_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(v, "DATA_SHA256", hashlib.sha256(b"counts").hexdigest())
    src = _mirror(data_root, b"counts", v.DATA_SHA256)
    dataset = _bare_dataset(tmp_path / "build")
    dataset.download()
    dest = tmp_path / "build" / "raw" / v.DATA_FILENAME
    assert os.readlink(dest) == str(src)
    dataset.download()  # an existing link is left alone
    assert os.readlink(dest) == str(src)


def test_download_refuses_a_manifest_digest_off_the_module_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #561): ``DATA_SHA256`` is the one pin and the manifest is its
    retrieval record. A manifest recording any other digest is refused by name with
    both digests before the mirror bytes are read, and nothing is linked.
    """
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _mirror(data_root, b"counts", "ab" * 32)
    dataset = _bare_dataset(tmp_path / "build")
    with pytest.raises(ManifestPinMismatchError) as err:
        dataset.download()
    assert str(err.value) == (
        f"raw-mirror manifest records sha256 {'ab' * 32} for {v.DATA_REL}, but the "
        "loader pins e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a"
    )
    assert not (tmp_path / "build" / "raw").exists()


def test_download_refuses_mirror_bytes_off_the_module_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the manifest agreeing with ``DATA_SHA256``, mirror bytes hashing to anything
    else raise ``RawSha256MismatchError`` naming the mirror file, the pin and the
    observed digest, and nothing is linked.
    """
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _mirror(data_root, b"tampered", v.DATA_SHA256)
    dataset = _bare_dataset(tmp_path / "build")
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.download()
    assert str(err.value) == (
        f"sha256 mismatch for {v.raw_mirror_dir(str(data_root)) / v.DATA_REL}: "
        "expected e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a, "
        f"observed {hashlib.sha256(b'tampered').hexdigest()}"
    )
    assert list((tmp_path / "build" / "raw").iterdir()) == []


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #524's sweep): with the count matrix already in ``raw/`` PyG
    skips ``download()``, so ``process()`` verifies it against ``DATA_SHA256`` first and
    raises ``RawSha256MismatchError`` naming it and both digests before a row is read;
    no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(v, [v.DATA_FILENAME])
    raw = staged.root / "raw" / v.DATA_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        v.EnvChemgenVanacloig2022Dataset(root=str(staged.root))
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_download_refuses_a_mirror_missing_the_file_or_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    src = _mirror(data_root, b"counts", None)
    dataset = _bare_dataset(tmp_path / "build")
    with pytest.raises(KeyError) as missing_record:
        dataset.download()
    assert missing_record.value.args == (
        f"{v.DATA_REL} is not in the raw-mirror manifest",
    )
    manifest = src.parent.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "citation_key": v.CITATION_KEY,
                "files": [
                    {
                        "path": v.DATA_REL,
                        "role": "raw_data",
                        "bytes": 1,
                        "sha256": v.DATA_SHA256,
                    }
                ],
            }
        )
    )
    src.unlink()
    with pytest.raises(RuntimeError) as missing_file:
        dataset.download()
    assert (
        str(missing_file.value) == f"required raw artifact missing from mirror: {src}"
    )


def test_download_without_a_manifest_is_a_bare_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_manifest`` reads the file directly, so no mirror is a FileNotFoundError."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    with pytest.raises(FileNotFoundError) as err:
        _bare_dataset(tmp_path / "build").download()
    assert err.value.filename == str(v.raw_mirror_dir() / "manifest.json")


class _FrozenDatetime:
    @staticmethod
    def now(tz: Any = None) -> datetime:
        return datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def test_deposit_raw_mirror_writes_the_file_and_an_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = tmp_path / "counts.txt.gz"
    counts.write_bytes(b"synthetic GEO counts")
    pin = hashlib.sha256(b"synthetic GEO counts").hexdigest()
    monkeypatch.setattr(v, "DATA_SHA256", pin)
    monkeypatch.setattr(v, "datetime", _FrozenDatetime)
    data_root = tmp_path / "dr"

    root = v.deposit_raw_mirror(counts_path=counts, data_root=str(data_root))

    assert root == data_root / "torchcell-raw" / v.CITATION_KEY
    assert (root / v.DATA_REL).read_bytes() == b"synthetic GEO counts"
    assert json.loads((root / "manifest.json").read_text()) == {
        "version": 1,
        "citation_key": "vanacloig-pedrosComparativeChemicalGenomic2022",
        "doi": "10.1093/femsyr/foac036",
        "title": "Comparative chemical genomic profiling across plant-based hydrolysate "
        "toxins reveals widespread antagonism in fitness contributions",
        "library_id": None,
        "zotero_item_key": None,
        "collections": [],
        "files": [
            {
                "path": v.DATA_REL,
                "role": "raw_data",
                "bytes": 20,
                "sha256": pin,
                "source": v.DATA_URL,
                "zotero_md5": None,
                "retrieval": {
                    "method": "direct_url",
                    "source_url": v.DATA_URL,
                    "retriever": "torchcell.literature.retrieve.direct_url",
                    "params": {"url": v.DATA_URL},
                    "sha256": pin,
                    "retrieved_at": "2026-09-13",
                    "last_check": None,
                },
                "processing": None,
            }
        ],
        "si_data_sources": [
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186866",
            v.DATA_URL,
        ],
        "si_expected": [
            "Table S1 (per-compound IC30 molar values + DMSO control pairing) -- "
            "academic.oup.com is not scriptable, so it is NOT mirrored",
            "Dataset2_mclust_cdt (clustered edgeR logFC matrix) -- same publisher gate",
        ],
        "provenance_complete": True,
        "created_at": "2026-09-30T12:00:00+00:00",
    }
    assert v.manifest_sha256(v.load_manifest(str(data_root)), v.DATA_REL) == pin
    mtime = (root / v.DATA_REL).stat().st_mtime_ns
    v.deposit_raw_mirror(counts_path=counts, data_root=str(data_root))
    assert (root / v.DATA_REL).stat().st_mtime_ns == mtime


def test_deposit_refuses_a_source_off_the_pin_before_touching_the_mirror(
    tmp_path: Path,
) -> None:
    counts = tmp_path / "counts.txt.gz"
    counts.write_bytes(b"not the GEO matrix")
    data_root = tmp_path / "dr"
    with pytest.raises(RuntimeError) as err:
        v.deposit_raw_mirror(counts_path=counts, data_root=str(data_root))
    assert str(err.value) == (
        f"{counts} sha256 mismatch: got "
        f"{hashlib.sha256(b'not the GEO matrix').hexdigest()}, expected "
        "e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a"
    )
    assert not data_root.exists()


def test_deposit_refuses_to_overwrite_a_differing_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = tmp_path / "counts.txt.gz"
    counts.write_bytes(b"synthetic GEO counts")
    monkeypatch.setattr(
        v, "DATA_SHA256", hashlib.sha256(b"synthetic GEO counts").hexdigest()
    )
    dest = v.raw_mirror_dir(str(tmp_path / "dr")) / v.DATA_REL
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"other bytes")
    with pytest.raises(RuntimeError) as err:
        v.deposit_raw_mirror(counts_path=counts, data_root=str(tmp_path / "dr"))
    assert str(err.value) == f"{dest} exists with a different sha256; refusing"
    assert dest.read_bytes() == b"other bytes"


def test_schema_classes_raw_file_and_the_inline_stubs() -> None:
    """The class wiring; ``preprocess_raw`` is an identity and ``create_experiment``
    refuses because both steps live inside ``process``.
    """
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is EnvironmentResponseExperimentReference
    assert dataset.raw_file_names == ["GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"]
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
