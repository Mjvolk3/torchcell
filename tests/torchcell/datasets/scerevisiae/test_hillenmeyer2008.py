# tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008.py
"""Unit tests for the Hillenmeyer 2008 header parser, drop rules and raw-mirror map.

Every test runs on synthetic headers shaped like the released ones, so nothing here
touches the 41 MB matrices or the built LMDB.
"""

from __future__ import annotations

from typing import Any

import pytest

from torchcell.datamodels.media import (
    HILLENMEYER_DROPOUT_MEDIA,
    HILLENMEYER_PARTIAL_DROPOUT_LEVELS,
    SC,
    SD,
    YP_GLYCEROL_LIQUID,
    YPD_LIQUID,
)
from torchcell.datamodels.schema import (
    BarcodedKanMxDeletionPerturbation,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    HeterozygousDeletionPerturbation,
    OrfHistoryRelation,
    PhysicalFactor,
    PreCultureSource,
    SmallMoleculePerturbation,
    heterozygous_deletion_functional_copies,
)
from torchcell.datamodels.strain_background import BRACHMANN_1998
from torchcell.datasets.scerevisiae.hillenmeyer2008 import (
    CITATION_KEY,
    DROP_HEAT_SHOCK_CYCLE,
    DROP_KEY_HEADER_COMPOUND_CONFLICT,
    DROP_UNIDENTIFIABLE,
    DROP_UNNAMED_MEDIA_AGENT,
    DROP_ZERO_GENERATIONS,
    MATRICES,
    SOURCED_VALUES,
    SUSPICIOUS_BATCHES,
    WAYBACK_TIMESTAMPS,
    KeyHeaderOutcome,
    KeyHeaderRule,
    StrainRow,
    build_environment,
    canonical_concentration,
    check_key_header,
    constructed_orf,
    hillenmeyer_background,
    marker_loci,
    parse_columns,
    parse_condition,
    raw_relpaths,
    read_control_set_map,
    read_key_conditions,
    strain_perturbation,
    wayback_url,
)

BLANK = ("", "", "")


def _parse(cond1: str, conc1: str = "", unit1: str = "", *cond2: str) -> Any:
    second = cond2 if cond2 else BLANK
    return parse_condition(cond1, conc1, unit1, second[0], second[1], second[2])


# --------------------------------------------------------------------------- #
# Dose canonicalization (the sorbitol 1.5 M / 1.5e+06 uM duplicate)
# --------------------------------------------------------------------------- #
def test_molar_spellings_of_one_dose_canonicalize_identically() -> None:
    molar = canonical_concentration("1.5", "m")
    micro = canonical_concentration("1.5e+06", "um")
    assert (molar.value, molar.unit) == (1.5, ConcentrationUnit.molar)
    assert (micro.value, micro.unit) == (molar.value, molar.unit)


@pytest.mark.parametrize(
    ("conc", "unit", "value", "expected"),
    [
        ("1000", "um", 1.0, ConcentrationUnit.millimolar),
        ("1", "mm", 1.0, ConcentrationUnit.millimolar),
        ("2000", "um", 2.0, ConcentrationUnit.millimolar),
        ("6.9", "um", 6.9, ConcentrationUnit.micromolar),
        ("0.5", "um", 500.0, ConcentrationUnit.nanomolar),
        ("15000", "um", 15.0, ConcentrationUnit.millimolar),
    ],
)
def test_molar_family_picks_the_largest_unit_at_or_above_one(
    conc: str, unit: str, value: float, expected: ConcentrationUnit
) -> None:
    dose = canonical_concentration(conc, unit)
    assert (dose.value, dose.unit) == (value, expected)


def test_non_molar_units_are_left_alone_and_a_missing_unit_is_a_fixed_dose() -> None:
    ugml = canonical_concentration("1", "ug/ml")
    assert (ugml.value, ugml.unit) == (1.0, ConcentrationUnit.ug_per_ml)
    assert canonical_concentration("", "").basis is not None
    assert canonical_concentration("", "").value is None


def test_an_unknown_unit_raises_rather_than_defaulting() -> None:
    with pytest.raises(ValueError, match="unrecognized concentration unit"):
        canonical_concentration("5", "furlongs")


# --------------------------------------------------------------------------- #
# Condition classification
# --------------------------------------------------------------------------- #
def test_a_second_condition_is_appended_in_every_branch_not_only_small_molecules() -> (
    None
):
    """``pH7.5 + FK506`` must not be stored as a plain pH stress."""
    parse = _parse("pH7.5", "", "", "FK506", "1", "ug/ml")
    assert parse.drop_reason is None
    kinds = [p.perturbation_type for p in parse.perturbations]
    assert kinds == ["environment_physical", "small_molecule"]
    physical = parse.perturbations[0]
    assert isinstance(physical, EnvironmentPhysicalPerturbation)
    assert physical.factor is PhysicalFactor.ph
    assert physical.magnitude is not None and physical.magnitude.value == 7.5
    drug = parse.perturbations[1]
    assert isinstance(drug, SmallMoleculePerturbation)
    assert drug.compound.inchikey is not None
    assert drug.concentration.value == 1.0


def test_plain_ph_stays_a_single_physical_edit() -> None:
    parse = _parse("pH8")
    assert [p.perturbation_type for p in parse.perturbations] == [
        "environment_physical"
    ]


def test_irradiated_splits_three_ways() -> None:
    """Bare UV, and the two photo-activated crosslinkers, are different conditions."""
    only = _parse("no drug irradiated")
    assert [p.factor for p in only.perturbations] == [PhysicalFactor.radiation]

    for label, conc, unit in (
        ("angelicin irradiated", "62.5", "um"),
        ("psoralen irradiated", "0.5", "um"),
    ):
        parse = _parse(label, conc, unit)
        assert parse.drop_reason is None
        assert [p.perturbation_type for p in parse.perturbations] == [
            "small_molecule",
            "environment_physical",
        ]
        compound = parse.perturbations[0]
        assert isinstance(compound, SmallMoleculePerturbation)
        assert compound.compound.inchikey is not None
        assert compound.concentration.value is not None


def test_media_swaps_map_onto_the_shared_library_objects() -> None:
    assert _parse("minimal media").media == SD
    assert _parse("synthetic complete").media == SC
    assert _parse("YP glycerol").media == YP_GLYCEROL_LIQUID
    assert _parse("benomyl", "6.9", "um").media == YPD_LIQUID


def test_nutrient_dropouts_are_derived_media_not_perturbations() -> None:
    parse = _parse("tryptophan dropout")
    assert parse.perturbations == []
    assert parse.media == HILLENMEYER_DROPOUT_MEDIA["tryptophan dropout"]
    assert parse.media.base_medium == "SC"
    assert any(c.name == "L-tryptophan" for c in parse.media.dropouts)


def test_the_vitamin_dropout_control_is_plain_sc_not_a_vitamin_dropout() -> None:
    assert _parse("vitamin drop-out control media").media == SC


def test_temperature_is_a_scalar_not_a_perturbation() -> None:
    parse = _parse("37 degrees C")
    assert parse.perturbations == []
    assert parse.temperature_c == 37.0
    assert parse.media == YPD_LIQUID


def test_compounds_resolve_to_the_tables_canonical_name() -> None:
    """The source label is the lookup key; the canonical name is what is stored."""
    nacl = _parse("NaCl", "1", "m").perturbations[0]
    assert isinstance(nacl, SmallMoleculePerturbation)
    assert nacl.compound.name == "sodium chloride"
    assert nacl.compound.inchikey is not None


# --------------------------------------------------------------------------- #
# Drop rules
# --------------------------------------------------------------------------- #
def test_heat_shock_cycle_is_dropped_rather_than_stored_as_a_steady_temperature() -> (
    None
):
    parse = _parse("37c, 45c")
    assert parse.drop_reason == DROP_HEAT_SHOCK_CYCLE
    assert parse.temperature_c is None


@pytest.mark.parametrize(
    "label",
    [
        "chemical diversity labs 14a",
        "tyrphostin",
        "amphotericin",
        "FeCl4",
        "DMSO 1%",
        "methotrexate, 30ul total up/dn",
        "ptp2",
    ],
)
def test_a_compound_with_no_structure_identifier_drops_its_records(label: str) -> None:
    parse = _parse(label, "100", "um")
    assert parse.drop_reason == DROP_UNIDENTIFIABLE
    assert parse.drop_detail is not None and label in parse.drop_detail


def test_an_unidentifiable_second_compound_also_drops_the_column() -> None:
    parse = _parse("fluconazole", "50", "um", "amphotericin", "10", "um")
    assert parse.drop_reason == DROP_UNIDENTIFIABLE


def test_a_dose_on_a_media_swap_with_no_named_agent_drops() -> None:
    parse = _parse("minimal media", "400", "um")
    assert parse.drop_reason == DROP_UNNAMED_MEDIA_AGENT
    assert parse.drop_detail == "minimal media: 400 um"


def test_an_undosed_media_swap_is_kept() -> None:
    assert _parse("minimal media").drop_reason is None


# --------------------------------------------------------------------------- #
# Environment assembly
# --------------------------------------------------------------------------- #
def test_generation_sign_is_a_protocol_flag_so_duration_stores_the_magnitude() -> None:
    """``-5gen`` is a frozen-stock start, ``5gen`` a YPD log-phase pre-culture to OD600
    2.0 of ~10 generations; both expose 5 generations, and the two environments differ.
    """
    parse = _parse("benomyl", "6.9", "um")
    frozen = build_environment(parse, "-5gen")
    grown = build_environment(parse, "5gen")
    assert frozen.duration_generations == grown.duration_generations == 5.0
    assert frozen.pre_culture is not None and grown.pre_culture is not None
    assert frozen.pre_culture.source is PreCultureSource.frozen_stock
    assert (frozen.pre_culture.medium, frozen.pre_culture.source_label) == (
        None,
        "-5gen",
    )
    assert grown.pre_culture.source is PreCultureSource.log_phase_culture
    assert grown.pre_culture.medium == YPD_LIQUID
    assert (
        grown.pre_culture.generations,
        grown.pre_culture.od600_at_transfer,
        grown.pre_culture.source_label,
    ) == (10.0, 2.0, "5gen")
    assert frozen.model_dump() != grown.model_dump()
    with pytest.raises(ValueError, match="states no pre-culture"):
        build_environment(parse, "0gen")
    with pytest.raises(ValueError, match="no generations field"):
        build_environment(parse, "")


def test_an_unstated_temperature_is_a_typed_gap_not_a_defaulted_thirty() -> None:
    environment = build_environment(_parse("benomyl", "6.9", "um"), "20gen")
    assert environment.temperature is None
    gaps = {gap.field: gap for gap in environment.provenance_gaps}
    assert "temperature" in gaps
    assert gaps["temperature"].reason.value == "deferred_pending_source_review"
    assert gaps["temperature"].resolve_with is not None
    assert "pierce" in str(gaps["temperature"].resolve_with.citation_key).lower()


def test_a_stated_temperature_carries_no_temperature_gap() -> None:
    """Only the culture format (vessel, aeration) stays gapped."""
    environment = build_environment(_parse("37 degrees C"), "5gen")
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0
    assert environment.gapped_fields() == {"culture_format"}


def test_minimal_media_is_sd_with_a_typed_supplement_gap() -> None:
    """BY4743 cannot grow on unsupplemented SD; the supplement is implied, never named."""
    environment = build_environment(_parse("minimal media"), "5gen")
    assert environment.media == SD
    assert environment.auxotroph_supplements is None
    assert environment.gapped_fields() == {
        "auxotroph_supplements",
        "culture_format",
        "temperature",
    }
    assert (
        "auxotroph_supplements"
        not in build_environment(_parse("synthetic complete"), "5gen").gapped_fields()
    )


def test_every_dosed_compound_gaps_its_unstated_vehicle() -> None:
    drug = _parse("benomyl", "6.9", "um").perturbations[0]
    assert isinstance(drug, SmallMoleculePerturbation)
    assert drug.solvent is None
    assert drug.gapped_fields() == {"solvent"}


def test_the_partial_dropout_level_is_read_and_checked_against_the_medium() -> None:
    """#505 E4: the release states 25 % (biotin) and 12.5 % (pyridoxine)."""
    parse = _parse("biotin partial drop-out", "25", "%")
    biotin = next(c for c in parse.media.components if c.compound.name == "biotin")
    assert "'25 %'" in (biotin.note or "")
    assert any(
        sv.quote == "biotin partial drop-out:25:%" for sv in parse.media.provenance
    )
    assert HILLENMEYER_PARTIAL_DROPOUT_LEVELS["pyridoxine HCl partial drop-out"] == (
        "12.5",
        "%",
    )
    with pytest.raises(ValueError, match="released level"):
        _parse("biotin partial drop-out", "50", "%")
    with pytest.raises(ValueError, match="unexplained dose"):
        _parse("tryptophan dropout", "5", "um")


# --------------------------------------------------------------------------- #
# Column grouping
# --------------------------------------------------------------------------- #
def _header(*columns: str) -> list[str]:
    return ["Orf", *columns]


def test_arrays_of_one_condition_scored_on_different_control_sets_stay_distinct() -> (
    None
):
    header = _header(
        "01_04_24_02:benomyl:6.9:um::::20gen:het_04_01_2:old scanner",
        "01_04_24_03:benomyl:6.9:um::::20gen:het_04_01_2:old scanner",
        "04_03_17_01:benomyl:6.9:um::::20gen:het_06_03:new scanner",
    )
    control_sets = {
        "01_04_24_02": "het_04_01_2::old scanner::20::tag3::YPD::dmso::0",
        "01_04_24_03": "het_04_01_2::old scanner::20::tag3::YPD::dmso::0",
        "04_03_17_01": "het_06_03::new scanner::20::tag3::YPD::dmso::0",
    }
    specs = parse_columns(header, control_sets, dict.fromkeys(control_sets, "benomyl"))
    assert len({s.group_key for s in specs}) == 2


def test_the_signed_generation_count_survives_on_the_control_set() -> None:
    """``-5gen`` and ``5gen`` share a duration magnitude and must not merge."""
    header = _header(
        "03_06_05_01:benomyl:6.9:um::::-5gen:het_09_02:old scanner",
        "03_06_05_04:benomyl:6.9:um::::5gen:het_09_02:old scanner",
    )
    control_sets = {
        "03_06_05_01": "het_09_02::old scanner::-5::tag3::YPD::dmso::0",
        "03_06_05_04": "het_09_02::old scanner::5::tag3::YPD::dmso::0",
    }
    specs = parse_columns(header, control_sets, dict.fromkeys(control_sets, "benomyl"))
    assert [
        s.environment.duration_generations for s in specs if s.environment is not None
    ] == [5.0, 5.0]
    assert len({s.group_key for s in specs}) == 2


def test_an_array_with_no_control_set_raises() -> None:
    header = _header("01_04_24_02:benomyl:6.9:um::::20gen:het_04_01_2:old scanner")
    with pytest.raises(ValueError, match="no control set"):
        parse_columns(header, {}, {})


def test_two_spellings_of_one_dose_group_together() -> None:
    header = _header(
        "01_12_11_02:sorbitol:1.5:m::::15gen:hom_05_01:old scanner",
        "01_11_28_02:sorbitol:1.5e+06:um::::15gen:hom_05_01:old scanner",
    )
    control_sets = dict.fromkeys(
        ["01_12_11_02", "01_11_28_02"], "hom_05_01::old scanner::15::tag3::YPD::dmso::0"
    )
    specs = parse_columns(header, control_sets, dict.fromkeys(control_sets, "sorbitol"))
    assert len({s.group_key for s in specs}) == 1


# --------------------------------------------------------------------------- #
# Raw mirror + constants
# --------------------------------------------------------------------------- #
def test_every_consumed_file_has_an_archived_url() -> None:
    consumed = {
        name
        for spec in MATRICES.values()
        for name in (spec.filename, spec.keyfile, spec.controls_file)
    }
    assert consumed <= set(WAYBACK_TIMESTAMPS)
    for name in WAYBACK_TIMESTAMPS:
        assert raw_relpaths()[name] == f"data/{name}"
        assert wayback_url(name).startswith("http://web.archive.org/web/")
        assert wayback_url(name).endswith(f"/{name}")


def test_the_som_names_ten_suspicious_construction_batches() -> None:
    assert len(SUSPICIOUS_BATCHES) == 10
    assert len(set(SUSPICIOUS_BATCHES)) == 10
    assert all(batch.startswith("chr") for batch in SUSPICIOUS_BATCHES)


# --------------------------------------------------------------------------- #
# Header vs key file (#505 E1)
# --------------------------------------------------------------------------- #
def test_key_header_agreement_spelling_variants_and_normalizations() -> None:
    assert check_key_header("a", "benomyl", "Benomyl").outcome is KeyHeaderOutcome.agree
    for header, key in (
        ("NaF", "sodium fluoride"),
        ("minimal media", "no drug minimal media"),
        ("synthetic complete", "synthetic complete for BY4743"),
    ):
        check = check_key_header("a", header, key)
        assert (check.outcome, check.rule) == (KeyHeaderOutcome.spelling_variant, None)
    # the key abbreviates ChemDiv, drops ' irradiated' and the hybridization volume
    for header, key in (
        ("chemical diversity labs 6842", "chemdiv 6842"),
        ("psoralen irradiated", "psoralen"),
        ("alverine citrate, 30ul total up/dn", "alverine citrate"),
    ):
        assert check_key_header("a", header, key).outcome is KeyHeaderOutcome.agree


def test_an_unexamined_key_header_disagreement_raises() -> None:
    with pytest.raises(ValueError, match="no written rule covers the pair"):
        check_key_header("x1", "benomyl", "nocodazole")


def _one_column(header_cond: str, key_cond: str, generations: str = "20gen") -> Any:
    header = _header(f"x1:{header_cond}::::::{generations}:hom_09_02:new scanner")
    control = {"x1": "hom_09_02::new scanner::20::tag3::YPD::dmso::0"}
    return parse_columns(header, control, {"x1": key_cond})[0]


def test_ph_conflict_serves_the_header_because_table_s1_lists_only_high_ph() -> None:
    """Hom 04_11_17_01: header pH8, key file ph4."""
    column = _one_column("pH8", "ph4")
    assert column.key_check.rule is KeyHeaderRule.header_wins_table_s1
    assert column.drop_reason is None
    physical = column.environment.perturbations[0]
    assert isinstance(physical, EnvironmentPhysicalPerturbation)
    assert physical.magnitude is not None and physical.magnitude.value == 8.0


def test_minimal_media_called_sc_for_by4743_in_the_key_is_served_on_sc() -> None:
    """The 18 hom arrays: header 'minimal media', key 'synthetic complete for BY4743'."""
    column = _one_column("minimal media", "synthetic complete for BY4743")
    assert column.key_check.rule is KeyHeaderRule.key_wins_strain_medium
    assert column.environment.media == SC
    assert "auxotroph_supplements" not in column.environment.gapped_fields()


@pytest.mark.parametrize(
    ("header_cond", "key_cond"),
    [
        ("bathophenanthroline disulfonate", "bisphenol s"),
        ("sodium arsenite", "sodium sulfate"),
        ("colchicine", "colchiceine"),
        ("chemical diversity labs 14a", "Tyrphostin"),
        ("ptp2", "phosphatase inhibitor"),
    ],
)
def test_a_compound_name_conflict_drops_the_array_with_both_strings(
    header_cond: str, key_cond: str
) -> None:
    column = _one_column(header_cond, key_cond)
    assert column.key_check.rule is KeyHeaderRule.drop_compound_conflict
    assert column.drop_reason == DROP_KEY_HEADER_COMPOUND_CONFLICT
    assert column.drop_detail == f"header {header_cond!r} vs key file {key_cond!r}"


def test_a_zero_generation_array_is_dropped_with_no_environment() -> None:
    column = _one_column("minimal media", "synthetic complete for BY4743", "0gen")
    assert column.drop_reason == DROP_ZERO_GENERATIONS
    assert column.environment is None


# --------------------------------------------------------------------------- #
# Strain background and the screened edit (#505 G1-G5)
# --------------------------------------------------------------------------- #
def test_background_is_by4743_named_by_the_key_file_with_every_allele_pending() -> None:
    background = hillenmeyer_background()
    assert (background.name, background.ploidy) == ("BY4743", "diploid")
    assert background.provenance is not None
    quotes = {sv.quote for sv in background.provenance}
    assert "synthetic complete for BY4743" in quotes
    assert "The deletion collections have been described (1)." in quotes
    assert "1. G. Giaever et al., Nature 418, 387 (Jul 25, 2002)." in quotes
    assert "2. S. E. Pierce et al., Nat Methods 3, 601 (Aug, 2006)." in quotes
    key = next(sv for sv in background.provenance if "BY4743" in sv.quote)
    assert key.provenance.sha256 == (
        "312f76309547ae2e2870029a2ca1f3580176ba0b97a24308eeb5afe17de9131b"
    )
    assert background.gapped_fields() == {"mating_type", "parents", "construction"}
    assert {a.allele_name for a in background.alleles} == {
        "his3Δ1",
        "leu2Δ0",
        "lys2Δ0",
        "met15Δ0",
        "ura3Δ0",
    }
    for allele in background.alleles:
        assert allele.provenance is None
        assert [g.resolve_with for g in allele.provenance_gaps] == [BRACHMANN_1998]
    assert [background.functional_copies(g) for g in ("YOR202W", "YBR115C")] == [0, 1]


def _row(orf: str, source: str | None = None, batch: str = "chr1_1") -> StrainRow:
    return StrainRow(
        row_id=f"{source or orf}:{batch}",
        source_orf=source or orf,
        orf=orf,
        batch=batch,
        values=[None, 1.0],
    )


def test_het_strain_is_a_kanmx4_allele_replacement_whose_dose_respects_the_background() -> (
    None
):
    """G2: HIS3 is null before and after (0); LYS2/MET15 depend on the replaced allele,
    a typed gap (None); an ordinary gene keeps 1 of 2.
    """
    background = hillenmeyer_background()
    markers = marker_loci(background)
    ordinary = strain_perturbation("het", _row("YAL001C"), "het_04_01_2", None, markers)
    assert isinstance(ordinary, HeterozygousDeletionPerturbation)
    assert (ordinary.cassette, ordinary.collection) == ("kanMX4", "het_04_01_2")
    assert ordinary.construction is not None and ordinary.construction.batch == "chr1_1"
    assert ordinary.gapped_fields() == {"barcode", "downtag_barcode"}
    assert heterozygous_deletion_functional_copies(background, ordinary) == 1
    doses = {}
    for orf in ("YOR202W", "YBR115C", "YLR303W"):
        pert = strain_perturbation("het", _row(orf), "het_04_01_2", None, markers)
        assert isinstance(pert, HeterozygousDeletionPerturbation)
        assert "replaced_allele" in pert.gapped_fields()
        doses[orf] = heterozygous_deletion_functional_copies(background, pert)
    assert doses == {"YOR202W": 0, "YBR115C": None, "YLR303W": None}


def test_hom_strain_is_a_barcoded_kanmx4_deletion_gapped_at_marker_loci() -> None:
    markers = marker_loci(hillenmeyer_background())
    hom = strain_perturbation("hom", _row("YAL001C"), "hom_09_02", None, markers)
    assert isinstance(hom, BarcodedKanMxDeletionPerturbation)
    assert (hom.cassette, hom.collection) == ("kanMX4", "hom_09_02")
    marker = strain_perturbation("hom", _row("YLR303W"), "hom_09_02", None, markers)
    assert marker.constructed_orf is None
    assert marker.gapped_fields() == {"barcode", "downtag_barcode", "constructed_orf"}


def test_two_source_orfs_on_one_gene_are_merged_a_lone_rename_is_gapped() -> None:
    """G3: YAR044W built against its own ORF is a different strain from YAR042W."""
    merged = constructed_orf("YAR044W", "YAR042W", 2)
    assert (merged.source_systematic_name, merged.relation) == (
        "YAR044W",
        OrfHistoryRelation.merged,
    )
    assert merged.gapped_fields() == {"deleted_span"}
    lone = constructed_orf("YOLD01W", "YBR002W", 1)
    assert lone.relation is None
    assert lone.gapped_fields() == {"deleted_span", "relation"}
    markers = marker_loci(hillenmeyer_background())
    a = strain_perturbation("hom", _row("YAR042W"), "hom_09_02", None, markers)
    b = strain_perturbation(
        "hom", _row("YAR042W", source="YAR044W"), "hom_09_02", merged, markers
    )
    assert a.systematic_gene_name == b.systematic_gene_name == "YAR042W"
    assert a != b


def test_every_sourced_value_quote_is_verbatim_in_its_pinned_file() -> None:
    """The audit anchor is only real if the quote is still in the sha256-pinned file."""
    import hashlib
    import os
    from pathlib import Path

    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    for key, sourced in SOURCED_VALUES.items():
        uri = sourced.provenance.source_uri
        base = "torchcell-raw" if uri.startswith("data/") else "torchcell-library"
        path = Path(data_root) / base / CITATION_KEY / uri
        if not path.exists():
            pytest.skip(f"{path} is not mounted on this machine")
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sourced.provenance.sha256, key
        assert sourced.quote in raw.decode(), key


def test_the_release_disagrees_with_itself_on_exactly_33_arrays() -> None:
    """#505 E1 pinned on the real release: 4 het + 29 hom conflicts, each with its rule."""
    import os
    from collections import Counter
    from pathlib import Path

    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    raw = Path(data_root) / "torchcell-raw" / CITATION_KEY / "data"
    if not raw.exists():
        pytest.skip("the raw mirror is not mounted on this machine")
    rules: dict[str, Counter[str]] = {}
    for arm, spec in MATRICES.items():
        with open(raw / spec.filename) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        columns = parse_columns(
            header,
            read_control_set_map(raw / spec.keyfile),
            read_key_conditions(raw / spec.keyfile),
        )
        rules[arm] = Counter(
            c.key_check.rule.value for c in columns if c.key_check.rule is not None
        )
        if arm == "hom":
            ph8 = next(c for c in columns if c.filename == "04_11_17_01")
            assert (ph8.key_check.header_condition, ph8.key_check.key_condition) == (
                "pH8",
                "ph4",
            )
            assert ph8.drop_reason is None
    assert rules == {
        "het": Counter({"drop_compound_conflict": 4}),
        "hom": Counter(
            {
                "key_wins_strain_medium": 18,
                "drop_compound_conflict": 7,
                "header_wins_table_s1": 4,
            }
        ),
    }
