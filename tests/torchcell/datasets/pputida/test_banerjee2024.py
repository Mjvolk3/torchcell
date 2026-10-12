# tests/torchcell/datasets/pputida/test_banerjee2024.py
# [[tests.torchcell.datasets.pputida.test_banerjee2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_banerjee2024.py
"""The Banerjee 2024 P. putida isoprenol titer loader.

The synthetic tests run everywhere and carry the weight here, because this loader's
values are sourced CONSTANTS rather than a parsed data file: they pin the twenty titers
and their units, the five build oracles that recompute the paper's own fold-changes (and
each oracle's refusal), the genotype builder including the two cases the release forces
(``phaABC`` as three tags from one range, ``gacA`` as no typed perturbation at all), the
environment and phenotype builders with the typed gaps this paper forces, and the
reference baselines.

The oracle tests are the reason a transcription slip cannot land: each one perturbs a
stored constant and asserts the build refuses. A fixture with invented titers would
exercise the plumbing while retiring the check, so the oracles run against the real
constants.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: they re-read every sourced
quote out of the two pinned mirrors, pin the deposited manifest's digests against the
module constants, and run L0 to L4 over the built LMDB. They are skipped unless the
mirrors and the built store are present.

Derived expectations for the pinned bytes: 20 records (6 EZ-rich knockout rounds, 4
EZ-rich pathway panel, 3 M9 minimal, 4 fed-batch, 3 hydrolysate); 42 verbatim quotes; 4
media; 11 released locus tags plus one deletion (``gacA``) the release gives no tag for;
10 released values that no record holds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import torchcell.datasets.pputida.banerjee2024 as b24
from torchcell.datamodels.media import (
    EZ_RICH_BANERJEE2024,
    M9_BANERJEE2024,
    M9_MODIFIED_FEDBATCH_BANERJEE2024,
    M9_MODIFIED_TUBE_BANERJEE2024,
)
from torchcell.datamodels.schema import (
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
_LIBRARY = Path(DATA_ROOT, "torchcell-library", b24.CITATION_KEY) if DATA_ROOT else None
_MIRROR = Path(DATA_ROOT, "torchcell-raw", b24.CITATION_KEY) if DATA_ROOT else None
_STORE = (
    Path(DATA_ROOT, "data/torchcell/isoprenol_titer_banerjee2024")
    if DATA_ROOT
    else None
)

requires_library = pytest.mark.skipif(
    _LIBRARY is None or not (_LIBRARY / b24.PAPER_MD).exists(),
    reason="the Banerjee 2024 OCR mirror is not present under $DATA_ROOT",
)
requires_mirror = pytest.mark.skipif(
    _MIRROR is None or not (_MIRROR / b24.SI_PDF).exists(),
    reason="the Banerjee 2024 raw mirror is not deposited under $DATA_ROOT",
)
requires_store = pytest.mark.skipif(
    _STORE is None or not (_STORE / "processed" / "lmdb").exists(),
    reason="the Banerjee 2024 dev-tree LMDB is not built",
)


# --------------------------------------------------------------------------- #
# The twenty released titers
# --------------------------------------------------------------------------- #
def test_the_twenty_titers_are_the_statements_the_article_body_makes() -> None:
    """Every stored titer, its unit and its family, pinned value by value."""
    assert b24.EXPECTED_RECORDS == 20
    assert len(b24.TITER_ROWS) == 20
    stored = {
        (row.strain, row.condition, row.time_hours): (row.titer, row.unit)
        for row in b24.TITER_ROWS
    }
    assert len(stored) == 20
    assert stored[("XW12", "tube_ezrich_glc20", 48.0)] == (164.0, "ug/mL")
    assert stored[("XW15", "tube_ezrich_glc20", 48.0)] == (241.0, "ug/mL")
    assert stored[("XW16", "tube_ezrich_glc20", 48.0)] == (19.0, "ug/mL")
    assert stored[("XW17", "tube_ezrich_glc20", 48.0)] == (12.0, "ug/mL")
    assert stored[("XW18", "tube_ezrich_glc20", 48.0)] == (13.0, "ug/mL")
    assert stored[("XW19", "tube_ezrich_glc20", 48.0)] == (114.0, "ug/mL")
    assert stored[("IY782", "tube_ezrich_glc20", 48.0)] == (80.0, "ug/mL")
    assert stored[("IY721", "tube_ezrich_glc20", 48.0)] == (395.0, "ug/mL")
    assert stored[("IY846", "tube_ezrich_glc20", 48.0)] == (536.0, "ug/mL")
    assert stored[("IY939", "tube_ezrich_glc20", 48.0)] == (1111.0, "ug/mL")
    assert stored[("IY1452", "tube_m9_glc20", None)] == (762.0, "ug/mL")
    assert stored[("IY1252", "tube_m9_glc20", 48.0)] == (816.0, "ug/mL")
    assert stored[("IY1252", "tube_m9_glc20", 72.0)] == (259.0, "ug/mL")
    assert stored[("IY1245", "bioreactor_m9mod_fedbatch", 72.0)] == (0.5, "g/L")
    assert stored[("IY1262", "bioreactor_m9mod_fedbatch", 96.0)] == (2.3, "g/L")
    assert stored[("IY1452", "bioreactor_m9mod_fedbatch", 96.0)] == (3.5, "g/L")
    assert stored[("IY1485", "bioreactor_m9mod_fedbatch", 96.0)] == (2.4, "g/L")
    assert stored[("IY1452", "tube_m9mod_glc20", 72.0)] == (841.0, "ug/mL")
    assert stored[("IY1452", "tube_m9mod_hydrolysate_5", 72.0)] == (409.0, "ug/mL")
    assert stored[("IY1452", "tube_m9mod_hydrolysate_10", 72.0)] == (432.0, "ug/mL")


def test_the_families_are_six_four_three_four_three() -> None:
    """The five condition families and their record counts."""
    counts: dict[str, int] = {}
    for row in b24.TITER_ROWS:
        counts[row.family] = counts.get(row.family, 0) + 1
    assert counts == {
        "ezrich_knockouts": 6,
        "ezrich_pathway": 4,
        "m9_minimal": 3,
        "fedbatch": 4,
        "hydrolysate": 3,
    }


def test_only_the_fed_batch_family_releases_an_uncertainty() -> None:
    """The +- numbers the paper prints, and nowhere else carries one."""
    released = {
        (row.strain, row.time_hours): row.uncertainty
        for row in b24.TITER_ROWS
        if row.uncertainty is not None
    }
    assert released == {
        ("IY1245", 72.0): 0.1,
        ("IY1262", 96.0): 0.3,
        ("IY1452", 96.0): 0.3,
        ("IY1485", 96.0): 0.3,
    }
    assert all(row.family == "fedbatch" for row in b24.TITER_ROWS if row.uncertainty)


def test_every_titer_is_on_the_unit_its_own_statement_released() -> None:
    """A mg/L statement is stored as ug/mL; a g/L statement stays g/L."""
    by_unit: dict[str, int] = {}
    for row in b24.TITER_ROWS:
        by_unit[row.unit] = by_unit.get(row.unit, 0) + 1
    assert by_unit == {"ug/mL": 16, "g/L": 4}
    assert {row.unit for row in b24.TITER_ROWS if row.family == "fedbatch"} == {"g/L"}


# --------------------------------------------------------------------------- #
# The references, including the one substitution
# --------------------------------------------------------------------------- #
def test_the_four_reference_baselines_and_their_strains() -> None:
    """Each family's denominator, as the paper's own fold-changes use it."""
    assert set(b24.REFERENCE_BASELINES) == {"ezrich", "m9", "fedbatch", "hydrolysate"}
    assert [
        (key, baseline.strain, baseline.titer, baseline.unit)
        for key, baseline in b24.REFERENCE_BASELINES.items()
    ] == [
        ("ezrich", "XW11", 104.0, "ug/mL"),
        ("m9", "IY1452", 762.0, "ug/mL"),
        ("fedbatch", "IY1245", 0.5, "g/L"),
        ("hydrolysate", "IY1452", 841.0, "ug/mL"),
    ]
    assert {row.reference for row in b24.TITER_ROWS} == set(b24.REFERENCE_BASELINES)


def test_the_m9_reference_is_recorded_as_a_substitution() -> None:
    """The one judgment call says so on the record, in the word the PR uses."""
    note = b24.REFERENCE_BASELINES["m9"].note
    assert note.startswith("A SUBSTITUTION, flagged for review.")
    assert "over 4-fold higher than WT (IY1245)" in note
    assert "reconstructing one from the" in note


def test_xw11_is_a_reference_and_never_a_record() -> None:
    """Banerjee attributes the 104 mg/L to Wang 2022, so it is not this dataset's."""
    assert "XW11" not in {row.strain for row in b24.TITER_ROWS}
    assert b24.REFERENCE_BASELINES["ezrich"].strain == "XW11"
    assert "Wang et al. 2022" in b24.REFERENCE_BASELINES["ezrich"].note
    assert "in our previous work (Wang et al., 2022)" in b24._Q_XW11_BASELINE


# --------------------------------------------------------------------------- #
# The build oracles, and each one's refusal
# --------------------------------------------------------------------------- #
def test_the_stored_constants_reproduce_the_papers_own_fold_changes() -> None:
    """The ratios the article states in words, recomputed here from the stored values.

    Computed in the test rather than by calling the module's oracle, so this is an
    INDEPENDENT statement of the same relations: if the oracle and these ever diverge,
    one of them is wrong and the pair says so. The oracle's own firing is covered by
    the refusal tests below and by the end-to-end build.
    """
    by_key = {
        (row.strain, row.condition, row.time_hours): row.titer for row in b24.TITER_ROWS
    }
    ezrich = b24.REFERENCE_BASELINES["ezrich"].titer
    assert by_key[("XW12", "tube_ezrich_glc20", 48.0)] / ezrich == pytest.approx(
        1.6, abs=0.05
    )
    assert by_key[("XW15", "tube_ezrich_glc20", 48.0)] / ezrich == pytest.approx(
        2.3, abs=0.05
    )
    assert by_key[("IY939", "tube_ezrich_glc20", 48.0)] / ezrich == pytest.approx(
        10.0, abs=1.0
    )
    high = by_key[("IY1252", "tube_m9_glc20", 48.0)]
    low = by_key[("IY1252", "tube_m9_glc20", 72.0)]
    assert (high - low) / high == pytest.approx(0.70, abs=0.03)


@pytest.mark.parametrize(
    ("strain", "condition", "hours", "titer", "expected"),
    [
        ("XW12", "tube_ezrich_glc20", 48.0, 300.0, "1.6-fold"),
        ("XW15", "tube_ezrich_glc20", 48.0, 100.0, "2.3-fold"),
        ("IY939", "tube_ezrich_glc20", 48.0, 300.0, "10-fold"),
        ("IY1252", "tube_m9_glc20", 72.0, 800.0, "about 70%"),
        ("IY1452", "bioreactor_m9mod_fedbatch", 96.0, 9.9, "3.5 g/L"),
    ],
)
def test_a_slipped_titer_constant_stops_the_build(
    monkeypatch: pytest.MonkeyPatch,
    strain: str,
    condition: str,
    hours: float,
    titer: float,
    expected: str,
) -> None:
    """Perturb one stored constant and the oracle that covers it must refuse."""
    rows = tuple(
        row.model_copy(update={"titer": titer})
        if (row.strain, row.condition, row.time_hours) == (strain, condition, hours)
        else row
        for row in b24.TITER_ROWS
    )
    monkeypatch.setattr(b24, "TITER_ROWS", rows)
    with pytest.raises(RuntimeError, match=expected):
        b24.IsoprenolTiterBanerjee2024Dataset._assert_titer_oracles()


def test_an_uncertainty_on_the_wrong_family_stops_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the fed-batch statements print a +-; inventing one elsewhere refuses."""
    rows = tuple(
        row.model_copy(update={"uncertainty": 5.0})
        if row.family == "ezrich_knockouts" and row.strain == "XW12"
        else row
        for row in b24.TITER_ROWS
    )
    monkeypatch.setattr(b24, "TITER_ROWS", rows)
    with pytest.raises(RuntimeError, match="only the fed-batch family"):
        b24.IsoprenolTiterBanerjee2024Dataset._assert_titer_oracles()


def test_a_missing_fed_batch_uncertainty_stops_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping a released SD would store a gap where the paper printed a value."""
    rows = tuple(
        row.model_copy(update={"uncertainty": None})
        if row.family == "fedbatch" and row.strain == "IY1262"
        else row
        for row in b24.TITER_ROWS
    )
    monkeypatch.setattr(b24, "TITER_ROWS", rows)
    with pytest.raises(RuntimeError, match="releases a .- number"):
        b24.IsoprenolTiterBanerjee2024Dataset._assert_titer_oracles()


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
def test_pha_cluster_is_three_tags_from_the_stated_range() -> None:
    """The paper states a range, so three typed deletions come out of it."""
    perturbations = b24.deletion_perturbations("phaABC")
    assert [
        (pert.systematic_gene_name, pert.perturbed_gene_name) for pert in perturbations
    ] == [("PP_5003", "phaA"), ("PP_5004", "phaB"), ("PP_5005", "phaC")]


def test_gaca_has_no_typed_perturbation_and_is_recorded_instead() -> None:
    """The release names the gene and no tag, and no tag is invented for it."""
    assert b24.deletion_perturbations("gacA") == []
    assert b24.GENOTYPE_NOT_TYPED == (("IY1485", "gacA", b24.GENOTYPE_NOT_TYPED[0][2]),)
    reason = b24.GENOTYPE_NOT_TYPED[0][2]
    assert "PP_4099 in Schmidt 2022 and Thompson 2020" in reason
    assert "PP_1650 in Borchert 2023" in reason


def test_iy1485_carries_iy1452s_genotype_because_gaca_cannot_be_typed() -> None:
    """The consequence is stated rather than hidden: the two genotypes coincide."""
    assert b24.strain_genotype("IY1485") == b24.strain_genotype("IY1452")
    assert b24.STRAIN_SPECS["IY1485"].deletions == (
        "phaABC",
        "mvaB",
        "hbdH",
        "ldhA",
        "PP_2675",
        "gacA",
    )


def test_every_deletion_locus_tag_is_the_one_the_paper_states() -> None:
    """Table 1's gene-association column, plus the PP_2675 the Results name."""
    assert {
        symbol: str(value.value) for symbol, value in b24.DELETION_LOCUS_TAGS.items()
    } == {
        "mvaB": "PP_3540",
        "hbdH": "PP_3073",
        "gntZ": "PP_4043",
        "ldhA": "PP_1649",
        "ppsA": "PP_2082",
        "gltA": "PP_4194",
        "aceA": "PP_4116",
        "PP_2675": "PP_2675",
    }


def test_each_strains_genotype_carries_exactly_one_plasmid() -> None:
    """The plasmid is the construct; one per strain, as Table S1 states them."""
    for strain in b24.STRAIN_SPECS:
        genotype = b24.strain_genotype(strain)
        constructs = {
            pert.construct_name
            for pert in genotype.perturbations
            if pert.perturbation_type == "heterologous_pathway"
        }
        assert constructs == {b24.STRAIN_SPECS[strain].plasmid}


def test_only_piy670_carries_the_aphA_phosphatase() -> None:
    """pIY670 is pIY554 plus AphA, which is what the 1111 mg/L strain adds."""
    assert [gene.token for gene in b24.CASSETTES["pIY670"].genes] == [
        "MvaSef",
        "MvaEef",
        "MKmm",
        "PMDHKQ",
        "AphA",
    ]
    assert "AphA" not in [gene.token for gene in b24.CASSETTES["pIY554"].genes]
    assert len(b24.CASSETTES["pXW1"].genes) == 4


def test_a_cassette_part_not_in_its_quoted_construct_string_refuses() -> None:
    """The part tokens stay the source's own, checked against its quoted description."""
    cassette = b24.CASSETTES["pIY670"].model_copy(
        update={"construct_string": "pRK2-Kan-araC-PBAD-MvaSef"}
    )
    with pytest.raises(RuntimeError, match="is not in the quoted pIY670"):
        b24.cassette_perturbations(cassette)


def test_the_pmd_organism_is_sourced_by_deferral_and_the_rest_are_not() -> None:
    """Which organism statement comes from where, per part."""
    sources = {
        gene.token: gene.organism_source for gene in b24.CASSETTES["pIY670"].genes
    }
    assert sources == {
        "MvaSef": "si",
        "MvaEef": "si",
        "MKmm": "si",
        "PMDHKQ": "carruthers",
        "AphA": "paper",
    }


# --------------------------------------------------------------------------- #
# Environment and phenotype
# --------------------------------------------------------------------------- #
def test_the_six_conditions_name_the_four_media() -> None:
    """Each stated condition and the medium object it carries."""
    assert {key: spec.medium for key, spec in b24.CONDITIONS.items()} == {
        "tube_ezrich_glc20": "ez_rich",
        "tube_m9_glc20": "m9",
        "bioreactor_m9mod_fedbatch": "m9_modified_fedbatch",
        "tube_m9mod_glc20": "m9_modified_tube",
        "tube_m9mod_hydrolysate_5": "m9_modified_tube",
        "tube_m9mod_hydrolysate_10": "m9_modified_tube",
    }
    assert [
        b24.environment(b24.CONDITIONS[key]).media
        for key in ("tube_ezrich_glc20", "tube_m9_glc20", "bioreactor_m9mod_fedbatch")
    ] == [EZ_RICH_BANERJEE2024, M9_BANERJEE2024, M9_MODIFIED_FEDBATCH_BANERJEE2024]
    assert (
        b24.environment(b24.CONDITIONS["tube_m9mod_hydrolysate_5"]).media
        is M9_MODIFIED_TUBE_BANERJEE2024
    )


def test_the_fed_batch_environment_is_the_bioreactor_at_25c() -> None:
    """The one condition with a different temperature and no shaking speed."""
    env = b24.environment(
        b24.CONDITIONS["bioreactor_m9mod_fedbatch"], duration_hours=96.0
    )
    assert env.temperature is not None
    assert env.temperature.value == 25.0
    assert env.culture_format is not None
    assert env.culture_format.vessel == "2 L bioreactor (Biostat B, Sartorius)"
    assert env.culture_format.working_volume_ul == 1_000_000.0
    assert env.culture_format.shaking_rpm is None
    assert env.culture_format.inoculum_od600 == 0.3
    assert env.duration_hours == 96.0
    assert env.aerobicity == "aerobic"


def test_the_shaken_conditions_are_30c_at_200_rpm() -> None:
    """Every other condition, as Methods 2.5 and 2.8 state them."""
    for key in ("tube_ezrich_glc20", "tube_m9_glc20", "tube_m9mod_glc20"):
        env = b24.environment(b24.CONDITIONS[key], duration_hours=48.0)
        assert env.temperature is not None
        assert env.temperature.value == 30.0
        assert env.culture_format is not None
        assert env.culture_format.shaking_rpm == 200.0
        assert env.culture_format.working_volume_ul == 5000.0


def test_an_unstated_duration_is_a_typed_gap_not_a_guess() -> None:
    """IY1452's 762 mg/L carries no sampling time, so the endpoint is unset."""
    env = b24.environment(b24.CONDITIONS["tube_m9_glc20"], duration_hours=None)
    assert env.duration_hours is None
    assert env.culture_format is not None
    assert env.culture_format.endpoint is None
    assert [gap.field for gap in env.provenance_gaps] == ["duration_hours"]


def test_the_hydrolysate_is_a_carbon_source_perturbation_beside_the_glucose() -> None:
    """Both carbon additions travel as typed physical factors."""
    env = b24.environment(b24.CONDITIONS["tube_m9mod_hydrolysate_10"])
    carbon = [
        pert
        for pert in env.perturbations
        if isinstance(pert, EnvironmentPhysicalPerturbation)
    ]
    assert all(pert.magnitude is not None for pert in carbon)
    assert [
        (pert.factor, pert.magnitude.value, pert.magnitude.unit)  # type: ignore[union-attr]
        for pert in carbon
    ] == [
        (PhysicalFactor.carbon_source, 20.0, ConcentrationUnit.g_per_l),
        (PhysicalFactor.carbon_source, 10.0, ConcentrationUnit.percent_v_v),
    ]


def test_the_ez_rich_records_carry_no_inducer_because_the_source_disjoins() -> None:
    """Methods 2.5 says "IPTG or arabinose" and never assigns one per strain."""
    env = b24.environment(b24.CONDITIONS["tube_ezrich_glc20"])
    doses = [
        pert
        for pert in env.perturbations
        if isinstance(pert, SmallMoleculePerturbation)
    ]
    assert doses == []
    assert b24.CONDITIONS["tube_ezrich_glc20"].arabinose_percent is None
    assert "disjunction" in str(b24.EZRICH_INDUCER_NOT_ASSIGNED.note)
    arabinose = b24.environment(b24.CONDITIONS["tube_m9_glc20"]).perturbations
    assert any(
        isinstance(pert, SmallMoleculePerturbation)
        and pert.concentration.value == 2.0
        and pert.concentration.unit == ConcentrationUnit.percent_w_v
        for pert in arabinose
    )


def test_a_released_uncertainty_becomes_a_sample_sd_with_its_n() -> None:
    """The fed-batch phenotype: the number, its type, its n and the derived SE."""
    phenotype = b24.titer_phenotype(3.5, ConcentrationUnit.g_per_l, uncertainty=0.3)
    assert phenotype.titer == 3.5
    assert phenotype.titer_unit == ConcentrationUnit.g_per_l
    assert phenotype.titer_uncertainty == 0.3
    assert phenotype.titer_uncertainty_type == UncertaintyType.sample_sd
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit == SampleUnit.biological_replicate
    assert phenotype.titer_se is not None
    assert phenotype.titer_se == pytest.approx(0.3 / 3**0.5)
    assert {gap.field for gap in phenotype.provenance_gaps} == {
        "productivity",
        "productivity_unit",
        "product_yield",
        "product_yield_unit",
    }


def test_an_unreleased_uncertainty_is_a_typed_gap_on_both_fields() -> None:
    """A figure error bar is not a number, so neither half is half-filled."""
    phenotype = b24.titer_phenotype(
        164.0, ConcentrationUnit.ug_per_ml, uncertainty=None
    )
    assert phenotype.titer_uncertainty is None
    assert phenotype.titer_uncertainty_type is None
    assert {"titer_uncertainty", "titer_uncertainty_type"} <= {
        gap.field for gap in phenotype.provenance_gaps
    }
    assert phenotype.n_samples == 3
    assert phenotype.quantification_method == "GC-FID"


def test_the_product_is_isoprenol_through_the_shared_compound_layer() -> None:
    """One compound entity, so this joins Kang 2026's isoprenol records."""
    assert b24.isoprenol().name == "isoprenol"
    assert b24.PRODUCT_NAME == "isoprenol"


# --------------------------------------------------------------------------- #
# What is not stored, and why
# --------------------------------------------------------------------------- #
def test_the_not_stored_ledger_counts_every_released_value_no_record_holds() -> None:
    """Five entries, ten values, each with its reason."""
    entries = b24.not_stored()
    assert [(entry.quantity, entry.values) for entry in entries] == [
        ("fed-batch OD600 maxima, with their standard deviations", 4),
        ("XW17 residual glucose at 48 h in EZ rich", 1),
        ("the XW13 / XW14 titer range", 2),
        ("specific growth rates of the mvaB-deleted strains against WT", 2),
        ("the IY1485 gacA deletion, as a typed perturbation", 1),
        ("every per-strain titer, OD600 and glucose SERIES", 0),
    ]
    assert sum(entry.values for entry in entries) == 10
    assert entries[0].issue == 776


def test_the_four_od600_values_are_the_ones_the_paper_prints() -> None:
    """The fed-batch maxima, which no phenotype class can hold."""
    assert b24.OD600_NOT_STORED == (
        ("IY1245", 25.5, 0.7, 72.0),
        ("IY1262", 20.5, 3.0, 96.0),
        ("IY1452", 21.0, 3.4, 72.0),
        ("IY1485", 17.9, 0.2, 48.0),
    )
    assert "#776 item 3" in b24.OD600_REASON


def test_the_build_accounting_refuses_a_dropped_record() -> None:
    """A drop is a defect in this loader, so the accounting will not write one."""
    accounting = b24.BuildAccounting(
        dataset="x",
        source_statements=20,
        candidate_records=20,
        kept_records=19,
        dropped_records=1,
        quotes_verified=42,
        not_stored=(),
        notes=(),
    )
    with pytest.raises(RuntimeError, match="1 records were dropped"):
        accounting.check()


def test_the_build_accounting_refuses_an_arithmetic_that_does_not_add_up() -> None:
    """Kept plus dropped must equal the candidates."""
    accounting = b24.BuildAccounting(
        dataset="x",
        source_statements=20,
        candidate_records=20,
        kept_records=18,
        dropped_records=0,
        quotes_verified=42,
        not_stored=(),
        notes=(),
    )
    with pytest.raises(RuntimeError, match="!= 20 candidates"):
        accounting.check()


# --------------------------------------------------------------------------- #
# Provenance plumbing
# --------------------------------------------------------------------------- #
def test_the_dataset_declares_its_organism_and_its_experiment_classes() -> None:
    """The registry and KG mapping read these."""
    dataset_class = b24.IsoprenolTiterBanerjee2024Dataset
    assert dataset_class.organism.value == "pputida"
    assert dataset_class.experiment_class.fget is not None  # type: ignore[attr-defined]
    assert b24.PATHWAY_ISOPRENOL == "isoprenol via the mevalonate IPP-bypass"


def test_every_elsevier_url_is_built_from_the_pii() -> None:
    """The retrieval record's URL is reproducible from the article's own PII."""
    assert b24.elsevier_url("mmc1.pdf") == (
        "https://ars.els-cdn.com/content/image/1-s2.0-S109671762400020X-mmc1.pdf"
    )


def test_the_quote_inventory_is_every_quote_constant() -> None:
    """``every_quote`` is what the audit walks, so it must find them all."""
    quotes = b24.every_quote()
    assert len(quotes) == 42
    assert all(name.startswith("_Q_") for name in quotes)
    assert "_Q_FEDBATCH_IY1452" in quotes
    assert "_Q_DATA_AVAILABILITY" in quotes


# --------------------------------------------------------------------------- #
# The real mirrors and the built store
# --------------------------------------------------------------------------- #
@pytest.mark.data
@requires_library
def test_every_sourced_quote_is_verbatim_in_its_pinned_mirror() -> None:
    """Re-read the bytes: a hash pin does not make a transcription verbatim."""
    assert b24.verify_quotes() == 42


@pytest.mark.data
@requires_mirror
def test_the_deposited_manifest_pins_what_the_module_pins() -> None:
    """The raw mirror's recorded digests against the module constants."""
    manifest = b24.load_manifest()
    assert manifest.citation_key == b24.CITATION_KEY
    assert manifest.doi == b24.DOI
    assert b24.manifest_sha256(manifest, b24.SI_PDF) == b24.SI_PDF_SHA256
    assert b24.manifest_sha256(manifest, b24.SI_MD) == b24.SI_MD_SHA256
    assert manifest.provenance_complete is True
    assert any(
        "NO per-strain measurement table" in line for line in manifest.si_expected
    )


@pytest.mark.data
@requires_store
def test_the_built_store_holds_the_twenty_records_and_passes_l0_to_l4() -> None:
    """The store, read back: its count, its classes and the full report."""
    assert _STORE is not None
    report = b24.verify_build(str(_STORE))
    assert report.passed, report.summary()
    assert len(report.levels_covered) == 5
    ledger = json.loads((_STORE / "preprocess" / "build_accounting.json").read_text())
    assert ledger["kept_records"] == 20
    assert ledger["dropped_records"] == 0
    assert ledger["quotes_verified"] == 42


@pytest.mark.data
@requires_store
def test_every_stored_record_validates_as_its_schema_classes() -> None:
    """The experiment and its reference, revalidated from the stored bytes."""
    assert _STORE is not None
    from torchcell.verification.runners import load_records

    records: list[dict[str, Any]] = load_records(str(_STORE))
    assert len(records) == 20
    for record in records:
        ProductTiterExperiment.model_validate(record["experiment"])
        ProductTiterExperimentReference.model_validate(record["reference"])
    assert {
        record["experiment"]["phenotype"]["product"]["name"] for record in records
    } == {"isoprenol"}
