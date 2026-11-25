# tests/torchcell/datasets/scerevisiae/test_smith2016.py
"""Smith 2016 loader: the vendor-code drop, the shared medium, the dose parsing.

The retention and encoding tests are pure (no build tree, no network). The mirror audit
binds each module-level ``SourcedValue`` to its verbatim quote and skips when the
``torchcell-library`` mirror is not mounted. The real-data tests at the end read the five
pinned workbooks out of the raw mirror and pin the per-replicate record counts (#1178);
they skip when the mirror is not mounted.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import cast

import pandas as pd
import pytest

from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import MEDIA_LIBRARY, SC_URA
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    EnvironmentResponsePhenotype,
    MeasurementType,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import smith2016 as s
from torchcell.datasets.scerevisiae.hoepfner2014 import ReleasedIdentity
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

#: Every non-DMSO public compound the screen dosed, and every vendor catalog code.
PUBLIC_DRUGS = [
    "Fluconazole",
    "Aureobasidin-A",
    "Cerulenin",
    "Rapamycin",
    "Doxorubicin",
    "Cantharidin",
    "NSC-180973",
    "Daunorubicin",
    "DMSO",
]
VENDOR_CODES = [
    "1181-0519",
    "4130-1276",
    "9121982",
    "9125678",
    "7312221",
    "ST016598",
    "CBF-666774",
    "6630449",
    "0KPI-0099",
    "9150499",
]


@pytest.mark.parametrize("drug", PUBLIC_DRUGS)
def test_public_drugs_carry_a_structure_identifier(drug: str) -> None:
    resolution = resolve_compound_identity(name=drug)
    assert resolution.identified, f"{drug} lost its identity"
    assert resolution.inchikey is not None


def test_the_released_tamoxifen_alias_is_what_keeps_nsc_180973() -> None:
    resolution = resolve_compound_identity(name="NSC-180973")
    assert resolution.name == "tamoxifen"
    assert resolution.inchikey == "NKANXQFJJICGDU-QPLCGJKRSA-N"


@pytest.mark.parametrize("code", VENDOR_CODES)
def test_vendor_catalog_codes_are_unidentified_and_held_out(code: str) -> None:
    """A vendor code is PROPRIETARY in the pinned table and the admission table is
    empty, so ``drug_is_identified`` is False and its rows are held out under
    ``UNIDENTIFIED_COMPOUND_RULE`` (2026.10.15), never stored with an inchikey gap.
    """
    resolution = resolve_compound_identity(name=code)
    assert not resolution.identified
    assert resolution.status.value == "PROPRIETARY"
    assert s.RELEASED_IDENTITIES == {}
    assert s.drug_is_identified(code) is False


@pytest.mark.parametrize("drug", PUBLIC_DRUGS)
def test_public_drugs_are_identified(drug: str) -> None:
    assert s.drug_is_identified(drug) is True


def test_an_admission_table_entry_admits_a_vendor_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``RELEASED_IDENTITIES`` entry is the only route in for a held-out code: with
    one, the code is identified and its compound is built from the entry's identity.
    """
    release = ReleasedIdentity(name="Fluconazole", source=s.COMPOUND_LIST)
    monkeypatch.setitem(s.RELEASED_IDENTITIES, "1181-0519", release)
    assert s.drug_is_identified("1181-0519") is True
    compound = s.released_compound("1181-0519")
    assert compound.name == resolved_compound("Fluconazole").name
    assert compound.inchikey == resolved_compound("Fluconazole").inchikey
    assert compound.inchikey is not None


def test_the_medium_is_the_shared_sc_ura_object() -> None:
    assert SC_URA in MEDIA_LIBRARY.values()
    assert MEDIA_LIBRARY["SC_URA"] is SC_URA
    assert [dropout.name for dropout in SC_URA.dropouts] == ["uracil"]


def test_dmso_control_dose_is_the_sourced_one_percent() -> None:
    concentration = s.parse_concentration(s.DMSO, 0.01)
    assert concentration.value == 1.0
    assert concentration.unit is ConcentrationUnit.percent_v_v
    assert s.DMSO_PERCENT.value == 1.0


@pytest.mark.parametrize(
    ("raw", "value", "unit"),
    [
        ("20 uM", 20.0, ConcentrationUnit.micromolar),
        ("2.2 nM", 2.2, ConcentrationUnit.nanomolar),
        ("1 mM", 1.0, ConcentrationUnit.millimolar),
    ],
)
def test_released_doses_parse_into_typed_units(
    raw: str, value: float, unit: ConcentrationUnit
) -> None:
    concentration = s.parse_concentration("Fluconazole", raw)
    assert concentration.value == value
    assert concentration.unit is unit


def test_an_unparseable_dose_raises_instead_of_being_guessed() -> None:
    with pytest.raises(ValueError, match="unparseable concentration"):
        s.parse_concentration("Fluconazole", "a lot")


def test_exposure_duration_is_recorded_in_generations_with_a_gap_for_hours() -> None:
    assert s.DURATION_GENERATIONS.value == 20.0
    assert "20 culture doublings" in s.DURATION_GENERATIONS.quote
    assert s.DURATION_HOURS_GAP.field == "duration_hours"
    assert s.DURATION_HOURS_GAP.reason.value == "not_reported_by_primary"


def test_the_assay_is_a_pooled_barcode_competition() -> None:
    assert s.ASSAY.value is AssayType.pooled_competitive_growth_barcode


def test_the_replicate_structure_names_the_workbook_of_each_condition() -> None:
    """Each of those replicates is its own record (#1178), so n_samples stays 1."""
    assert s.REPLICATE_STRUCTURE.value == {"1% DMSO": 8, "20 uM fluconazole": 3}
    assert "combined" in s.REPLICATE_STRUCTURE.quote
    assert s.REPLICATE_WORKBOOKS[s.DMSO] == (
        s.DMSO_REPLICATES_FILENAME,
        s.DMSO_REPLICATES_SHEET,
        8,
    )
    assert s.REPLICATE_WORKBOOKS["Fluconazole"] == (
        s.FLUCONAZOLE_REPLICATES_FILENAME,
        s.FLUCONAZOLE_REPLICATES_SHEET,
        3,
    )
    assert s.REPLICATE_DOSE.value == {s.DMSO: 0.01, "Fluconazole": "20uM"}
    assert s.FIRST_REPLICATE == "1"


def test_the_replicates_are_biological_per_the_paper() -> None:
    """The quote that makes a per-replicate record a culture, not a repeat read."""
    assert s.BIOLOGICAL_REPLICATES.value == "biological"
    assert "biological replicates" in s.BIOLOGICAL_REPLICATES.quote
    assert "Additional files 6 and 7" in s.BIOLOGICAL_REPLICATES.quote


def test_the_atc_dose_is_parsed_out_of_the_sample_sheet_header() -> None:
    """Additional file 5 releases the induction dose only inside its column header."""
    column = s.atc_column(["MiSeq Run #", "Pool Name", "ATC (250ng/ml)", "Replicate"])
    assert column == "ATC (250ng/ml)"
    dose = s.atc_concentration(column)
    assert dose.value == 0.25
    assert dose.unit is ConcentrationUnit.ug_per_ml


@pytest.mark.parametrize(
    "columns", [["Pool Name", "Replicate"], ["ATC (250ng/ml)", "ATC (125ng/ml)"]]
)
def test_a_sample_sheet_without_exactly_one_atc_column_is_refused(
    columns: list[str],
) -> None:
    with pytest.raises(ValueError, match="exactly one ATc-dose column header"):
        s.atc_column(columns)


def test_two_spellings_of_one_dose_key_as_one_condition() -> None:
    """Additional file 5 writes a trailing space Additional file 10 omits."""
    assert s.dose_key("Fluconazole", "20uM ".strip()) == s.dose_key(
        "Fluconazole", "20 uM"
    )
    assert s.dose_key(s.DMSO, 0.01) == "1.0percent_v/v"
    assert s.dose_key("Rapamycin", "10nM") == "10.0nM"


def test_every_screened_pool_maps_to_its_sequenced_sample_pool() -> None:
    """The 18 bp and 20 bp halves of one library were sequenced as one sample."""
    assert s.POOL_TO_SAMPLE_POOL == {
        "broad_tiling": "broad_tiling",
        "gene_tiling_18bp": "gene_tiling18 & gene_tiling20",
        "gene_tiling_20bp": "gene_tiling18 & gene_tiling20",
        "mutant18": "mutant18 & mutant20",
        "mutant20": "mutant18 & mutant20",
    }


def test_the_environment_is_sc_ura_with_one_typed_drug_edit() -> None:
    dataset = s.CrispriChemgenSmith2016Dataset.__new__(s.CrispriChemgenSmith2016Dataset)
    environment = s.CrispriChemgenSmith2016Dataset._environment(
        dataset, "Fluconazole", "20 uM"
    )
    assert environment.media is SC_URA
    assert environment.duration_generations == 20.0
    assert environment.duration_hours is None
    assert [gap.field for gap in environment.provenance_gaps] == ["duration_hours"]
    perturbation = cast(SmallMoleculePerturbation, environment.perturbations[0])
    assert len(environment.perturbations) == 1
    assert perturbation.compound.name == "fluconazole"
    assert perturbation.compound.inchikey is not None
    assert perturbation.solvent is None


def test_an_induced_culture_carries_the_atc_dose_beside_its_drug() -> None:
    """A culture record states its own induction; an A record is the contrast, so not."""
    dataset = s.CrispriChemgenSmith2016Dataset.__new__(s.CrispriChemgenSmith2016Dataset)
    induced = s.CrispriChemgenSmith2016Dataset._environment(
        dataset, "Fluconazole", "20uM", atc=s.atc_concentration("ATC (250ng/ml)")
    )
    uninduced = s.CrispriChemgenSmith2016Dataset._environment(
        dataset, "Fluconazole", "20uM"
    )
    assert [
        cast(SmallMoleculePerturbation, p).compound.name for p in induced.perturbations
    ] == ["fluconazole", "anhydrotetracycline"]
    assert (
        cast(SmallMoleculePerturbation, induced.perturbations[1]).concentration.value
        == 0.25
    )
    assert [
        cast(SmallMoleculePerturbation, p).compound.name
        for p in uninduced.perturbations
    ] == ["fluconazole"]


def test_a_culture_reference_is_the_pools_median_guide() -> None:
    """The fitness f is median centred in its pool, so the culture reference is f = 0."""
    dataset = s.CrispriChemgenSmith2016Dataset.__new__(s.CrispriChemgenSmith2016Dataset)
    dataset.name = "crispri_chemgen_smith2016"
    environment = s.CrispriChemgenSmith2016Dataset._environment(dataset, s.DMSO, 0.01)
    reference = s.CrispriChemgenSmith2016Dataset._reference(
        dataset, environment, s.CULTURE_UNITS
    )
    phenotype = reference.phenotype_reference
    assert phenotype.environment_response == 0.0
    assert phenotype.units == s.CULTURE_UNITS
    assert phenotype.replicate_id is None
    assert "median" in s.CULTURE_UNITS


def test_reference_is_the_uninduced_zero_baseline() -> None:
    dataset = s.CrispriChemgenSmith2016Dataset.__new__(s.CrispriChemgenSmith2016Dataset)
    environment = s.CrispriChemgenSmith2016Dataset._environment(dataset, s.DMSO, 0.01)
    dataset.name = "crispri_chemgen_smith2016"
    reference = s.CrispriChemgenSmith2016Dataset._reference(dataset, environment)
    phenotype = reference.phenotype_reference
    assert phenotype.environment_response == 0.0
    assert phenotype.assay_type is AssayType.pooled_competitive_growth_barcode


def test_uncertainty_is_the_variance_of_a_single_released_estimate() -> None:
    """SE = sqrt(var(A)/1): n_samples must stay 1 or the SE shrinks by sqrt(8)."""
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=-1.5,
        environment_response_uncertainty=0.25,
        environment_response_uncertainty_type=UncertaintyType.variance,
        n_samples=1,
        sample_unit=SampleUnit.pooled,
        units=s.UNITS,
    )
    assert phenotype.environment_response_se == pytest.approx(0.5)


def _library_root() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = osp.join(data_root, "torchcell-library")
    return root if osp.isdir(root) else None


def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    root = _library_root()
    if root is None:
        pytest.skip("torchcell-library mirror not mounted")
    values = [value for value in vars(s).values() if isinstance(value, SourcedValue)]
    assert values, "the loader must carry module-level SourcedValues"
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


# --------------------------------------------------------------------------- #
# Real data (#1178): the pinned workbooks in the raw mirror. Skips without it.
# --------------------------------------------------------------------------- #
def _mirror() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = osp.join(data_root, s.RAW_DIR_REL, "si", "si_data")
    files = [
        s.EFFECT_FILENAME,
        s.GUIDE_FILENAME,
        s.SAMPLES_FILENAME,
        s.DMSO_REPLICATES_FILENAME,
        s.FLUCONAZOLE_REPLICATES_FILENAME,
    ]
    return root if all(osp.isfile(osp.join(root, name)) for name in files) else None


def _sheet(filename: str, sheet: str) -> pd.DataFrame:
    root = _mirror()
    if root is None:
        pytest.skip("the Smith 2016 raw mirror is not mounted")
    return pd.read_excel(osp.join(root, filename), sheet_name=sheet)


def test_the_released_replicate_counts_are_what_the_loader_stores() -> None:
    """The record counts of a build, measured on the pinned workbooks.

    7,053 A records (14,463 Additional file 10 rows minus the 7,410 vendor-code rows),
    7,216 DMSO culture records (451 guides with data x 8 replicates x 2 arms) and 3,438
    fluconazole ones (637 x 2 arms over replicates 1 and 2, 445 over replicate 3), so
    17,707 in all.
    """
    effects = _sheet(s.EFFECT_FILENAME, s.EFFECT_SHEET)
    assert len(effects) == 14463
    held_out = sum(1 for drug in effects["Drug"] if not s.drug_is_identified(str(drug)))
    assert held_out == 7410
    assert len(effects) - held_out == 7053
    counts = {}
    for drug, (filename, sheet, n_replicates) in s.REPLICATE_WORKBOOKS.items():
        frame = _sheet(filename, sheet)
        total = 0
        for replicate in range(1, n_replicates + 1):
            for arm in ("noATC", "plusATC"):
                column = pd.to_numeric(
                    frame[f"Rep-{replicate}_{arm}_f_mean"], errors="coerce"
                )
                total += int(column.notna().sum())
        counts[drug] = total
    assert counts == {s.DMSO: 7216, "Fluconazole": 3438}
    assert 7053 + sum(counts.values()) == 17707


def test_every_effect_row_is_replicate_one_of_its_condition() -> None:
    """The check the build runs: Additional file 10's counts are the Rep-1 counts."""
    effects = _sheet(s.EFFECT_FILENAME, s.EFFECT_SHEET)
    samples = _sheet(s.SAMPLES_FILENAME, s.SAMPLES_SHEET)
    inventory: dict[tuple[str, str, str], set[int]] = {}
    for pool, drug, dose, replicate in zip(
        samples["Pool Name"],
        samples["Drug"],
        samples["Drug concentration"],
        samples["Replicate"],
        strict=True,
    ):
        named = str(drug).strip()
        key = (str(pool).strip(), named, s.dose_key(named, str(dose).strip()))
        inventory.setdefault(key, set()).add(int(replicate))
    groups = effects.groupby(["#Pool", "Drug", "Concentration"]).size()
    assert len(groups) == 67
    for pool, drug, dose in groups.index:
        key = (
            s.POOL_TO_SAMPLE_POOL[str(pool).strip()],
            str(drug).strip(),
            s.dose_key(str(drug).strip(), str(dose).strip()),
        )
        assert min(inventory[key]) == 1, key
    frames = {
        drug: _sheet(filename, sheet)
        for drug, (filename, sheet, _) in s.REPLICATE_WORKBOOKS.items()
    }
    compared = 0
    for drug, frame in frames.items():
        first = frame.set_index(["#Pool", "Guide"])
        rows = effects[effects["Drug"].map(lambda value: str(value).strip()) == drug]
        for _, row in rows.iterrows():
            index = (str(row["#Pool"]), str(row["Guide"]))
            if index not in first.index:
                continue
            for released, column in (
                (row["-ATC_count"], "Rep-1_noATC_count"),
                (row["+ATC_count"], "Rep-1_plusATC_count"),
            ):
                pinned = pd.to_numeric(first.loc[index, column], errors="coerce")
                if pd.isna(pinned):
                    continue
                assert float(released) == float(pinned), (drug, index, column)
                compared += 1
    assert compared == 2148


def test_the_eight_replicate_means_reproduce_the_released_combined_mean() -> None:
    """The authors' summary is a derivation over these records, so it is not stored.

    Measured on the pinned workbooks: the plain mean over a guide's replicates of
    ``f(+ATc) - f(-ATc)`` equals the released ``Combined_A_mean`` to within 0.001, which
    is the rounding of the released cells (three decimals): 0.000875 for Additional file
    6 and 0.001 for Additional file 7.
    """
    for drug, (filename, sheet, n_replicates) in s.REPLICATE_WORKBOOKS.items():
        frame = _sheet(filename, sheet)
        for column in frame.columns[2:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        worst = 0.0
        compared = 0
        for _, row in frame.iterrows():
            if pd.isna(row["Combined_A_mean"]):
                continue
            per_replicate = [
                row[f"Rep-{replicate}_plusATC_f_mean"]
                - row[f"Rep-{replicate}_noATC_f_mean"]
                for replicate in range(1, n_replicates + 1)
                if not pd.isna(row[f"Rep-{replicate}_noATC_f_mean"])
            ]
            mean = sum(per_replicate) / len(per_replicate)
            worst = max(worst, abs(mean - row["Combined_A_mean"]))
            compared += 1
        assert compared == {s.DMSO: 451, "Fluconazole": 637}[drug]
        assert worst == pytest.approx(0.001, abs=1e-9) or worst < 0.001, (drug, worst)
