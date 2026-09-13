# tests/torchcell/datasets/scerevisiae/test_lian2019.py
"""Lian 2019 loader: the SED-URA/G418 correction and the CRISPRd guide/donor split.

The split is the correctness claim of this build, so it is tested three ways: the pure
splitter, the positional join against the published design library, and (when the mirror
is mounted) the genome evidence that the last 21 nt really is a SaCas9 spacer.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import cast

import pandas as pd
import pytest

from torchcell.datamodels.media import MEDIA_LIBRARY, SED_URA_G418
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    CrisprActivationPerturbation,
    CrisprDeletionPerturbation,
    CrisprInterferencePerturbation,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import lian2019 as ln
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

#: A real CRISPRd design cassette (row 1 of Supplementary Data 3, gene ACS1).
ACS1_CASSETTE = (
    "TGGGATGAACACCTTATCGAATGGCTTAGACCAGTTTAAAAATTGGGTAGTTCAATAGACTCCTTGTGCAAGC"
    "GCTGATAGTCCTGCAACCCGTCCAAGTCTTTAGAACCGAAGAACTTAG"
)


def test_the_screen_medium_is_sed_ura_g418_not_sed_g418() -> None:
    assert MEDIA_LIBRARY["SED_URA_G418"] is SED_URA_G418
    assert [dropout.name for dropout in SED_URA_G418.dropouts] == ["uracil"]
    assert "SED-URA/G418" in ln.MEDIUM.quote
    selection = [
        component
        for component in SED_URA_G418.components
        if component.role.value == "selection_agent"
    ]
    assert len(selection) == 1
    g418 = selection[0]
    assert g418.compound.name == "G418 (geneticin)"
    assert g418.concentration is not None
    assert g418.concentration.value == 200.0
    assert g418.concentration.unit is ConcentrationUnit.ug_per_ml


def test_deletion_cassette_splits_into_a_21_nt_spacer_and_a_100_nt_donor() -> None:
    assert len(ACS1_CASSETTE) == 121
    spacer, donor = ln.split_deletion_cassette(ACS1_CASSETTE)
    assert len(spacer) == ln.D_SPACER_LEN == 21
    assert len(donor) == 100
    assert donor + spacer == ACS1_CASSETTE
    # the previous build stored THIS instead, which is donor sequence only
    assert ACS1_CASSETTE[: ln.D_BARCODE_LEN] != spacer
    assert ACS1_CASSETTE[: ln.D_BARCODE_LEN] in donor


def test_a_cassette_too_short_to_split_raises() -> None:
    with pytest.raises(ValueError, match="too short to split"):
        ln.split_deletion_cassette("ACGT")


def _cassettes(
    monkeypatch: pytest.MonkeyPatch, table: pd.DataFrame, design: pd.DataFrame
) -> list[tuple[str, str]]:
    """Run ``_deletion_cassettes`` against an in-memory design library."""
    dataset = ln.CrisprMagicLian2019Dataset.__new__(ln.CrisprMagicLian2019Dataset)
    dataset.root = "/unused"
    monkeypatch.setattr(pd, "read_excel", lambda *_a, **_k: design)
    return ln.CrisprMagicLian2019Dataset._deletion_cassettes(dataset, table)


def test_the_positional_join_returns_one_split_per_design_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = "A" * 100 + "CGTACGTACGTACGTACGTAC"
    table = pd.DataFrame(
        {
            "mod": ["a", "d", "d"],
            "spacer": [
                "CCACGGCATGTCAACAGGTGAGT",
                ACS1_CASSETTE[: ln.D_BARCODE_LEN],
                other[: ln.D_BARCODE_LEN],
            ],
        }
    )
    design = pd.DataFrame({"Sequence": [ACS1_CASSETTE, other]})
    assert _cassettes(monkeypatch, table, design) == [
        ln.split_deletion_cassette(ACS1_CASSETTE),
        ln.split_deletion_cassette(other),
    ]


def test_a_barcode_that_does_not_prefix_its_design_row_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pd.DataFrame(
        {"mod": ["d", "d"], "spacer": ["T" * ln.D_BARCODE_LEN, "T" * ln.D_BARCODE_LEN]}
    )
    design = pd.DataFrame({"Sequence": [ACS1_CASSETTE, "G" * 121]})
    with pytest.raises(RuntimeError, match="do not positionally join"):
        _cassettes(monkeypatch, table, design)


def test_a_row_count_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    table = pd.DataFrame({"mod": ["d"], "spacer": ["A" * ln.D_BARCODE_LEN]})
    design = pd.DataFrame({"Sequence": [ACS1_CASSETTE, "G" * 121]})
    with pytest.raises(RuntimeError, match="do not match"):
        _cassettes(monkeypatch, table, design)


def test_each_modality_builds_its_own_perturbation_leaf_and_effector() -> None:
    spacer, donor = ln.split_deletion_cassette(ACS1_CASSETTE)
    deletion = ln.crispr_perturbation("YAL054C", "ACS1", "d", spacer, donor)
    assert isinstance(deletion, CrisprDeletionPerturbation)
    assert deletion.crispr.effector == "SaCas9"
    assert deletion.crispr.guide_sequence == spacer
    assert deletion.donor_sequence == donor

    activation = ln.crispr_perturbation("YAL054C", "ACS1", "a", "A" * 23)
    assert isinstance(activation, CrisprActivationPerturbation)
    assert activation.crispr.effector == "dLbCas12a-VP"

    interference = ln.crispr_perturbation("YAL054C", "ACS1", "i", None)
    assert isinstance(interference, CrisprInterferencePerturbation)
    assert interference.crispr.effector == "dSpCas9-RD1152"
    assert interference.crispr.n_guides is None


def test_the_environment_is_sed_ura_g418_with_typed_duration_gaps() -> None:
    dataset = ln.CrisprMagicLian2019Dataset.__new__(ln.CrisprMagicLian2019Dataset)
    environment = ln.CrisprMagicLian2019Dataset._environment(dataset, 10.0)
    assert environment.media is SED_URA_G418
    assert environment.duration_hours is None
    assert environment.duration_generations is None
    assert {gap.field for gap in environment.provenance_gaps} == {
        "duration_hours",
        "duration_generations",
    }
    assert len(environment.perturbations) == 1
    perturbation = cast(SmallMoleculePerturbation, environment.perturbations[0])
    assert perturbation.compound.name == "furfural"
    assert perturbation.compound.inchikey == "HYBBIBNJHNGZAN-UHFFFAOYSA-N"
    assert perturbation.concentration.value == 10.0
    assert perturbation.concentration.unit is ConcentrationUnit.millimolar


def test_the_round_furfural_ladder_and_backgrounds_are_sourced() -> None:
    assert ln.FURFURAL_MM.value == {1: 5.0, 2: 10.0, 3: 15.0}
    assert ln.ROUND_BACKGROUND[1] == []
    assert ln.ROUND_BACKGROUND[2] == [("YDR409W", "SIZ1", "i")]
    assert ln.ROUND_BACKGROUND[3][1] == ("YDL040C", "NAT1", "a")


def test_the_assay_is_a_pooled_barcode_competition() -> None:
    assert ln.ASSAY.value is AssayType.pooled_competitive_growth_barcode


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
    values = [value for value in vars(ln).values() if isinstance(value, SourcedValue)]
    assert values, "the loader must carry module-level SourcedValues"
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"
