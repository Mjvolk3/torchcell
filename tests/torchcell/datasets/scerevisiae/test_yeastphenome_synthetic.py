# tests/torchcell/datasets/scerevisiae/test_yeastphenome_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_yeastphenome_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_yeastphenome_synthetic.py
"""Hermetic build tests for the YeastPhenome loader on two synthetic screens.

``SCREENS`` (44 pinned real screens) is monkeypatched to two synthetic entries, and each
screen's ``<pmid>_<stem>_valuez.txt`` is hand-written under ``<root>/raw/`` so PyG skips
``download()``. The compound resolver is the pinned in-repo table (no network). The
data-gated mirror test lives in ``test_yeastphenome.py``; this file pins the column
grammar and the exact records without any ``$DATA_ROOT``.

Screen A (PMID 99999901, stem ``synthetic_a``), header cells and their fate (2026.10.12,
#1045: every growth column the schema can carry is stored):

    1  hom | growth (colony size) | furfural [10 mM] | YPD | synthetic      kept, BY4743 diploid
    2  hap a | growth (culture turbidity) | quinine [2 mM] | SD | synthetic kept, BY4741 haploid
    3  het | growth (colony size) | furfural [10 mM] | YPD | synthetic      kept, 2 -> 1 copy in BY4743
    4  hom | growth (colony size) | temperature [37 C], hydrogen peroxide [1-1.2 mM] | YPD | synthetic
                                                                            condition_not_representable (dose range)
    5  hom | growth (colony size) | furfural [10 mM] | synthetic            kept, Media gapped (4 fields)
    6  hom | expression (microarray) | furfural [10 mM] | YPD | synthetic   not_growth
    7  hom | growth                                                         malformed_header
    8  hom | growth (colony size) | furfural [10 mM] | LB | synthetic       kept, is_synthetic gapped (LB)
    9  (empty cell)                                                         skipped silently
    10 hom | growth (colony size) | temperature [37 C], time [48 h] | YPD | synthetic
                                                                            kept, 37 C and 48 h on the environment

    rows: YAL001C 1.5 / -2.25 / 1 ...; YBR002C 0.5 / (blank) / 1 ...; not_an_orf 3.0 ...
    (deferred to the released-gene-name leaf); Q0250 -0.75 / 0.125 / 1 ...

Screen B (PMID 99999902, stem ``synthetic_b``): one column
``hap alpha | growth (pooled barseq) | furfural [10 mM] | SC + EtOH | synthetic`` (SC
synthetic liquid, BY4742 haploid), one row YAL001C 2.0.

Records in write order: A col 1 -> YAL001C, YBR002C, Q0250 (idx 0-2); A col 2 -> YAL001C,
Q0250 (idx 3-4); A col 3 -> YAL001C, YBR002C, Q0250 (idx 5-7); A col 5 -> 8-10; A col 8 ->
11-13; A col 10 -> 14-16; B col 1 -> YAL001C (idx 17). Eighteen records.
"""

import json
import socket
from pathlib import Path
from typing import Literal

import pytest

from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.identity import identity_sha256, media_identity
from torchcell.datamodels.schema import (
    BiologicAgentClass,
    BiologicPerturbation,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    EngineeredCopyNumberPerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    ReferenceGenome,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import yeastphenome
from torchcell.datasets.scerevisiae.yeastphenome import (
    _COMPONENT_NOTES,
    CURATION,
    NPV_UNITS,
    ParsedCondition,
    YeastPhenomeDataset,
    _base_medium,
    _column_meta,
    _media_statement,
    _parse_agent,
    _parse_conditions,
    _parse_media,
    _parse_media_token,
    _readout_method,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification import ProvenanceGap, ProvenanceGapReason

SCREEN_A = {"pmid": "99999901", "stem": "synthetic_a", "valuez_sha256": "0" * 64}
SCREEN_B = {"pmid": "99999902", "stem": "synthetic_b", "valuez_sha256": "1" * 64}

HEADER_A = [
    "orf",
    "hom | growth (colony size) | furfural [10 mM] | YPD | synthetic",
    "hap a | growth (culture turbidity) | quinine [2 mM] | SD | synthetic",
    "het | growth (colony size) | furfural [10 mM] | YPD | synthetic",
    "hom | growth (colony size) | temperature [37 C], hydrogen peroxide [1-1.2 mM] | YPD "
    "| synthetic",
    "hom | growth (colony size) | furfural [10 mM] | synthetic",
    "hom | expression (microarray) | furfural [10 mM] | YPD | synthetic",
    "hom | growth",
    "hom | growth (colony size) | furfural [10 mM] | LB | synthetic",
    "",
    "hom | growth (colony size) | temperature [37 C], time [48 h] | YPD | synthetic",
]
ROWS_A = [
    ["YAL001C", "1.5", "-2.25", "1", "1", "1", "1", "1", "1", "1", "1"],
    ["YBR002C", "0.5", "", "1", "1", "1", "1", "1", "1", "1", "1"],
    ["not_an_orf", "3.0", "3.0", "1", "1", "1", "1", "1", "1", "1", "1"],
    ["Q0250", "-0.75", "0.125", "1", "1", "1", "1", "1", "1", "1", "1"],
]
HEADER_B = [
    "orf",
    "hap alpha | growth (pooled barseq) | furfural [10 mM] | SC + EtOH | synthetic",
]
ROWS_B = [["YAL001C", "2.0"]]

FURFURAL = Compound(
    name="furfural",
    inchikey="HYBBIBNJHNGZAN-UHFFFAOYSA-N",
    smiles="C1=COC(=C1)C=O",
    pubchem_cid=7362,
    chebi_id="CHEBI:30976",
)
QUININE_UNRESOLVED = Compound(
    name="quinine",
    provenance_gaps=[
        ProvenanceGap(
            field="inchikey", reason=ProvenanceGapReason.deferred_pending_source_review
        )
    ],
)
TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_carried_by_curation,
    looked_in=CURATION,
)


def _phenotype_gaps() -> list[ProvenanceGap]:
    return [
        ProvenanceGap(
            field=f,
            reason=ProvenanceGapReason.not_carried_by_curation,
            looked_in=CURATION,
        )
        for f in ("n_samples", "environment_response_uncertainty", "sample_unit")
    ]


def _environment(media: Media, compound: Compound, mm: float) -> Environment:
    return Environment(
        media=media,
        temperature=None,
        perturbations=[
            SmallMoleculePerturbation(
                compound=compound,
                concentration=Concentration(
                    value=mm, unit=ConcentrationUnit.millimolar
                ),
            )
        ],
        provenance_gaps=[TEMPERATURE_GAP],
    )


# --- begin lane X: the released medium token's composition, in typed fields --------- #
#: The ethanol vehicle of ``SC + EtOH``: ``other`` is the role a vehicle takes, since
#: ``MediaComponentRole`` has no solvent member, and the curation states no fraction.
_VEHICLE_ETHANOL = MediaComponent(
    compound=resolved_compound("ethanol"),
    role=MediaComponentRole.other,
    concentration=None,
    note=_COMPONENT_NOTES["vehicle"],
)


def _media(
    token: str,
    state: str,
    *,
    screen: dict[str, str] = SCREEN_A,
    is_synthetic: bool | None = None,
    components: list[MediaComponent] | None = None,
    dropouts: list[Compound] | None = None,
    dropout_roles: list[MediaComponentRole] | None = None,
    gaps: list[ProvenanceGap] | None = None,
) -> Media:
    """The medium the loader stores for a released token: the token verbatim as ``name``,
    its composition in typed fields, and one medium-level statement quoting the token
    against the pinned bytes of the screen file it was read from.
    """
    return Media(
        name=token,
        state=state,
        is_synthetic=is_synthetic,
        base_medium=_base_medium(token.split("+")[0].strip()),
        components=components or [],
        dropouts=dropouts or [],
        dropout_roles=dropout_roles or [],
        provenance=[_media_statement(token, screen)],
        provenance_gaps=gaps or [],
    )


# --- end lane X -------------------------------------------------------------------- #


def _experiment(
    orf: str, npv: float, env: Environment, readout: str
) -> EnvironmentResponseExperiment:
    return EnvironmentResponseExperiment(
        dataset_name="YeastPhenomeDataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        ),
        environment=env,
        phenotype=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=npv,
            units=f"{NPV_UNITS} [readout: {readout}]",
            provenance_gaps=_phenotype_gaps(),
        ),
    )


def _reference(
    env: Environment, readout: str, strain: str, ploidy: Literal["haploid", "diploid"]
) -> EnvironmentResponseExperimentReference:
    return EnvironmentResponseExperimentReference(
        dataset_name="YeastPhenomeDataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=strain,
            ploidy=ploidy,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=env,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=0.0,
            units=f"{NPV_UNITS} [readout: {readout}]",
        ),
    )


def _write_screen(
    raw: Path, screen: dict[str, str], header: list[str], rows: list[list[str]]
) -> None:
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    (raw / f"{screen['pmid']}_{screen['stem']}_valuez.txt").write_text(
        "\n".join(lines) + "\n"
    )


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[YeastPhenomeDataset, Path]:
    monkeypatch.setattr(yeastphenome, "SCREENS", [SCREEN_A, SCREEN_B])
    root = tmp_path / "yeastphenome"
    raw = root / "raw"
    raw.mkdir(parents=True)
    _write_screen(raw, SCREEN_A, HEADER_A, ROWS_A)
    _write_screen(raw, SCREEN_B, HEADER_B, ROWS_B)
    return YeastPhenomeDataset(root=str(root)), root


# --------------------------------------------------------------------------- #
# Column grammar helpers
# --------------------------------------------------------------------------- #


def test_column_meta_five_and_four_fields_and_malformed() -> None:
    """Media is present only with 5 fields; the author is always last; <4 fields -> None."""
    assert _column_meta(HEADER_A[1]) == {
        "zygosity": "hom",
        "phenotype": "growth (colony size)",
        "condition": "furfural [10 mM]",
        "media": "YPD",
    }
    assert _column_meta(
        "hom | growth (colony size) | furfural [10 mM] | synthetic"
    ) == {
        "zygosity": "hom",
        "phenotype": "growth (colony size)",
        "condition": "furfural [10 mM]",
        "media": None,
    }
    assert _column_meta("hom | growth | x") is None


@pytest.mark.parametrize(
    ("phenotype", "expected"),
    [
        ("growth (colony size)", "colony size"),
        ("growth (culture turbidity)", "culture turbidity"),
        ("growth", "growth"),
        ("  growth (pooled barseq)  ", "pooled barseq"),
    ],
)
def test_readout_method(phenotype: str, expected: str) -> None:
    """The parenthesized modality is the readout; without one the stripped string is."""
    assert _readout_method(phenotype) == expected


def test_parse_media_rich_synthetic_state_prefix_and_gaps() -> None:
    """Base = token before '+'; rich vs synthetic from the two tables, else by prefix.

    ``colony`` / ``spot`` / ``killing`` in the phenotype -> solid, otherwise liquid;
    ``SC - Ura`` is synthetic by prefix; ``LB`` has no table row and no classifying
    prefix, so it is stored with ``is_synthetic`` a typed gap; a None field (4-field
    header) stores a ``Media`` with name, state and synthetic flag all typed gaps.
    Every named medium joins the shared family its leading word or prefix names in
    ``MEDIA_LIBRARY`` through ``base_medium`` (#1045: the released name stays as
    written and no library object is substituted), and the classification reads the
    BASE WORD, so a dropout suffix or a '+' addition does not change it (lane X, 2026-10-15).
    """
    assert _parse_media("YPD", "growth (colony size)", SCREEN_A) == _media(
        "YPD", "solid", is_synthetic=False
    )
    assert _parse_media("SC + EtOH", "growth (pooled barseq)", SCREEN_A) == _media(
        "SC + EtOH", "liquid", is_synthetic=True, components=[_VEHICLE_ETHANOL]
    )
    assert _parse_media("YPD", "growth (spot assay)", SCREEN_A) == _media(
        "YPD", "solid", is_synthetic=False
    )
    assert _parse_media("SD", "growth (killing zone)", SCREEN_A) == _media(
        "SD", "solid", is_synthetic=True
    )
    assert _parse_media("SC - Ura", "growth (culture turbidity)", SCREEN_A) == _media(
        "SC - Ura", "liquid", is_synthetic=True, dropouts=[resolved_compound("uracil")]
    )
    assert _parse_media("CSM", "growth (colony size)", SCREEN_A) == _media(
        "CSM", "solid", is_synthetic=True
    )
    lb = _parse_media("LB", "growth (colony size)", SCREEN_A)
    assert (lb.name, lb.state, lb.is_synthetic, lb.base_medium) == (
        "LB",
        "solid",
        None,
        "LB",
    )
    assert [(g.field, g.reason) for g in lb.provenance_gaps] == [
        ("is_synthetic", ProvenanceGapReason.not_carried_by_curation)
    ]
    absent = _parse_media(None, "growth (colony size)", SCREEN_A)
    assert (absent.name, absent.state, absent.is_synthetic) == (None, None, None)
    assert [g.field for g in absent.provenance_gaps] == [
        "name",
        "state",
        "is_synthetic",
    ]


# --- begin lane X: the released medium token's composition, in typed fields --------- #
def test_the_whole_released_media_token_is_the_stored_name() -> None:
    """``name`` is the token VERBATIM, never ``token.split('+')[0]``.

    The pre-lane X of the ontology unification loader truncated at '+', which put six released tokens on the plain
    name ``SC`` over 212,402 records, six on ``YPD`` over 239,081 and five on ``SD``
    over 23,354: 474,837 of 524,682 records carried a name that is not the token, and
    the ``+`` half of the token reached no field at all.
    """
    tokens = (
        "SC + DMSO [0.5%]",
        "YPD + G418 [200 ug/ml] + DMSO [1%]",
        "SD + his + leu + lys + ura",
        "YPD + EtOH",
    )
    names = [
        _parse_media(token, "growth (colony size)", SCREEN_A).name for token in tokens
    ]
    assert names == list(tokens)


def test_a_named_dropout_in_the_token_is_a_resolved_compound_in_dropouts() -> None:
    """``SC - Ade`` drops adenine, and the dropout joins the same compound entity another
    dataset's medium lists as a component (the resolver's InChIKey, not the spelling).
    """
    media = _parse_media("SC - Ade", "growth (pooled culture, microarray)", SCREEN_A)
    assert [(c.name, c.inchikey) for c in media.dropouts] == [
        ("adenine", "GFFGJBXGBJISGV-UHFFFAOYSA-N")
    ]
    assert media.components == []
    assert media.dropout_roles == []


def test_a_carbon_source_and_a_buffer_in_the_token_are_typed_components() -> None:
    """``SC, Glycerol [2%]`` and ``SC, HEPES [25 mM]`` state a dose; the dose is stored
    against the role it fills, not left inside the name.
    """
    glycerol = _parse_media(
        "SC, Glycerol [2%]", "growth (colony size)", SCREEN_A
    ).components
    assert [(c.compound.name, c.role, c.concentration) for c in glycerol] == [
        (
            "glycerol",
            MediaComponentRole.carbon_source,
            Concentration(value=2.0, unit=ConcentrationUnit.percent),
        )
    ]
    hepes = _parse_media(
        "SC, HEPES [25 mM]", "growth (colony size)", SCREEN_A
    ).components
    assert [(c.compound.name, c.role, c.concentration) for c in hepes] == [
        (
            "HEPES",
            MediaComponentRole.buffer,
            Concentration(value=25.0, unit=ConcentrationUnit.millimolar),
        )
    ]


def test_a_role_level_dropout_is_typed_and_separates_the_arm_from_its_control() -> None:
    """Galardini 2019's nitrogen-free arm and its own control were ONE medium identity.

    ``SD + His + Leu + Lys + Ura, -N`` and ``SD + His + Leu + Lys + Ura (2)`` are two
    columns of one screen, both with condition ``standard``, and both stored the name
    ``SD`` with no composition, so they shared one served environment node and the L1
    rule counted 4,837 records as repeats of each other. The nitrogen class the
    curation declares absent names no compound, so it is a ``dropout_roles`` entry.
    """
    starved = _parse_media(
        "SD + His + Leu + Lys + Ura, -N", "growth (colony size)", SCREEN_A
    )
    control = _parse_media(
        "SD + His + Leu + Lys + Ura (2)", "growth (colony size)", SCREEN_A
    )
    assert starved.dropout_roles == [MediaComponentRole.nitrogen_source]
    assert control.dropout_roles == []
    assert [c.compound.name for c in starved.components] == [
        c.compound.name for c in control.components
    ]
    assert identity_sha256(media_identity(starved)) != identity_sha256(
        media_identity(control)
    )
    carbon_free = _parse_media("SC (-C)", "growth (colony size)", SCREEN_A)
    assert carbon_free.dropout_roles == [MediaComponentRole.carbon_source]
    assert identity_sha256(media_identity(carbon_free)) != identity_sha256(
        media_identity(_parse_media("SC", "growth (colony size)", SCREEN_A))
    )


def test_a_dose_on_the_base_word_is_an_unexpanded_component_not_a_guess() -> None:
    """``YPG [3%]`` states a dose for an ingredient the curation abbreviates to a letter.

    The dose is kept against a ``composition_deferred`` carbon-source line rather than
    being attached to a molecule the release never names, and the line is identity, so
    a dosed base is not the node its undosed family is.
    """
    ypg = _parse_media("YPG [3%]", "growth (colony size)", SCREEN_A)
    assert ypg.name == "YPG [3%]"
    assert ypg.base_medium == "YP"
    assert [
        (c.role, c.definition, c.concentration, c.compound.inchikey)
        for c in ypg.components
    ] == [
        (
            MediaComponentRole.carbon_source,
            ComponentDefinition.composition_deferred,
            Concentration(value=3.0, unit=ConcentrationUnit.percent),
            None,
        )
    ]
    assert ypg.open_gaps == [ypg.components[0].compound.name]


def test_an_unknown_media_token_form_raises_rather_than_storing_it_unread() -> None:
    """No fallback: a form the grammar does not cover is a human's job, not a silent
    medium whose statement sits inside its name.
    """
    with pytest.raises(ValueError, match="is in no _MEDIUM_COMPOUNDS row"):
        _parse_media_token("SC + sorbitol")
    with pytest.raises(ValueError, match="_BASE_DOSED_ROLE does not name"):
        _parse_media_token("SC [4%]")
    with pytest.raises(ValueError, match="ConcentrationUnit does not carry"):
        _parse_media_token("SC + DMSO [3 drops]")
    with pytest.raises(ValueError, match="_ROLE_DROPOUTS does not name"):
        _parse_media_token("SC (-P)")


def test_the_medium_statement_quotes_the_token_against_the_pinned_screen_bytes() -> (
    None
):
    """Every typed field on the medium traces to one quote of the released token, pinned
    to the sha256 of the screen file whose header states it.
    """
    media = _parse_media("SC + DMSO [0.5%]", "growth (pooled barseq)", SCREEN_B)
    assert len(media.provenance) == 1
    statement = media.provenance[0]
    assert statement.quote == "SC + DMSO [0.5%]"
    assert statement.value == "SC + DMSO [0.5%]"
    assert statement.provenance.sha256 == SCREEN_B["valuez_sha256"]
    assert statement.provenance.citation_key == CURATION.citation_key
    assert statement.provenance.source_uri == "Datasets/99999902/synthetic_b_valuez.txt"


def test_a_vehicle_is_a_medium_component_and_a_selection_agent_is_too() -> None:
    """Both are constant to the medium rather than the studied edit, and the curation
    states both in the medium field, so both are components (the schema's own ruling for
    a selection agent). A vehicle control has no perturbation to carry a ``Solvent``.
    """
    media = _parse_media(
        "YPD + G418 [200 ug/ml] + DMSO [1%]", "growth (colony size)", SCREEN_A
    )
    assert [(c.compound.name, c.role) for c in media.components] == [
        ("G418 (geneticin)", MediaComponentRole.selection_agent),
        ("dimethyl sulfoxide", MediaComponentRole.other),
    ]


def test_auxotroph_supplements_in_the_token_are_roled_components() -> None:
    """``SD + his + leu + lys + ura`` names four nutrients that complement the
    collection's auxotrophies; each is stored at the role it fills, with no dose, since
    the curation states none.
    """
    media = _parse_media("SD + his + leu + lys + ura", "growth (colony size)", SCREEN_A)
    assert [(c.compound.name, c.role, c.concentration) for c in media.components] == [
        ("L-histidine", MediaComponentRole.amino_acid, None),
        ("L-leucine", MediaComponentRole.amino_acid, None),
        ("L-lysine", MediaComponentRole.amino_acid, None),
        ("uracil", MediaComponentRole.nucleobase, None),
    ]
    assert media.base_medium == "SD"


# --- end lane X -------------------------------------------------------------------- #


def test_parse_agent_small_molecule_resolved_and_unresolved() -> None:
    """``furfural [10 mM]`` resolves through the pinned table; ``quinine [2 mM]`` does not.

    Finding: quinine is absent from the compound table, so its ``Compound`` carries an
    ``inchikey`` gap with reason ``deferred_pending_source_review`` and no identifiers.
    Whether quinine is the condition of a real pinned screen is not verified here.
    """
    assert _parse_agent("furfural [10 mM]") == SmallMoleculePerturbation(
        compound=FURFURAL,
        concentration=Concentration(value=10.0, unit=ConcentrationUnit.millimolar),
    )
    assert _parse_agent("quinine [2 mM]") == SmallMoleculePerturbation(
        compound=QUININE_UNRESOLVED,
        concentration=Concentration(value=2.0, unit=ConcentrationUnit.millimolar),
    )


def test_parse_agent_defensin_and_unit_map() -> None:
    """A defensin name routes to a peptide biologic; every mapped unit spelling resolves."""
    assert _parse_agent("NaD1 [0.5 uM]") == BiologicPerturbation(
        agent_class=BiologicAgentClass.peptide,
        name="NaD1",
        concentration=Concentration(value=0.5, unit=ConcentrationUnit.micromolar),
    )
    units = {
        "M": ConcentrationUnit.molar,
        "mM": ConcentrationUnit.millimolar,
        "uM": ConcentrationUnit.micromolar,
        "µM": ConcentrationUnit.micromolar,
        "nM": ConcentrationUnit.nanomolar,
        "ug/ml": ConcentrationUnit.ug_per_ml,
        "ug/mL": ConcentrationUnit.ug_per_ml,
        "g/L": ConcentrationUnit.g_per_l,
        "g/l": ConcentrationUnit.g_per_l,
        "%": ConcentrationUnit.percent,
        "% v/v": ConcentrationUnit.percent_v_v,
        "% w/v": ConcentrationUnit.percent_w_v,
    }
    for spelling, unit in units.items():
        pert = _parse_agent(f"furfural [3 {spelling}]")
        assert isinstance(pert, SmallMoleculePerturbation)
        assert pert.concentration == Concentration(value=3.0, unit=unit)


def _compound_name(perturbation: object) -> str:
    assert isinstance(perturbation, SmallMoleculePerturbation)
    return perturbation.compound.name


def test_parse_conditions_environment_fields_and_components() -> None:
    """``standard`` is the base condition; temperature, time and oxygen set the
    environment's own fields; comma-joined agents are each a perturbation; an
    identical repeated component counts once.
    """
    assert _parse_conditions("standard") == ParsedCondition()
    parsed = _parse_conditions(
        "sodium chloride [0.6 M], temperature [39 C], time [72 h], sodium chloride [0.6 M]"
    )
    assert [_compound_name(p) for p in parsed.perturbations] == ["sodium chloride"]
    assert parsed.temperature == Temperature(value=39.0)
    assert (parsed.duration_hours, parsed.duration_generations) == (72.0, None)
    assert parsed.unrepresentable == []
    two = _parse_conditions("furfural [10 mM], acetic acid [5 mM]")
    assert [_compound_name(p) for p in two.perturbations] == ["furfural", "acetic acid"]
    assert _parse_conditions("oxygen [0%]").aerobicity == "anaerobic"
    assert _parse_conditions("time [5 gen]").duration_generations == 5.0


def test_parse_conditions_reports_what_the_schema_cannot_carry() -> None:
    """A dose range, an undosed agent, a unit the enum lacks, a non-zero oxygen level
    and a genotype edit written as a condition are listed under ``unrepresentable``;
    the representable components beside them are still parsed.
    """
    parsed = _parse_conditions(
        "hydrogen peroxide [1-1.2 mM], temperature [40 C], Mospilan, "
        "rapamycin [4 ng/ml], oxygen [5%], expression of human CYP1A2, [10 mM]"
    )
    assert parsed.temperature == Temperature(value=40.0)
    assert parsed.perturbations == []
    assert parsed.unrepresentable == [
        "hydrogen peroxide [1-1.2 mM]",
        "Mospilan",
        "rapamycin [4 ng/ml]",
        "oxygen [5%]",
        "expression of human CYP1A2",
        "[10 mM]",
    ]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def test_len_gene_set_and_manifest(dataset: tuple[YeastPhenomeDataset, Path]) -> None:
    """Twenty-four records over two screens; the gene set is the four released row ids.

    ``not_an_orf`` rows are stored on the released-gene-name leaf (#1045); ``Q0250``
    passes the Q-plus-four-digits alternative. No ``preprocess/data.csv`` is written by this loader
    (``df`` is None).
    """
    ds, root = dataset
    assert len(ds) == 24
    pre = root / "preprocess"
    assert json.loads((pre / "gene_set.json").read_text()) == [
        "Q0250",
        "YAL001C",
        "YBR002C",
        "not_an_orf",
    ]
    assert not (pre / "data.csv").exists()
    assert ds.df is None
    manifest = json.loads((pre / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "yeastphenome"
    assert manifest["loader_class"] == "YeastPhenomeDataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.yeastphenome"
    assert manifest["hostname"] == socket.gethostname()
    assert set(manifest["closure"]) >= {
        "EnvironmentResponseExperiment",
        "EnvironmentResponsePhenotype",
        "SmallMoleculePerturbation",
        "BiologicPerturbation",
    }


def test_hom_colony_size_records(dataset: tuple[YeastPhenomeDataset, Path]) -> None:
    """Records 0, 1 and 3 are screen A column 1: YPD solid, furfural 10 mM, BY4743
    diploid; record 2 is the ``not_an_orf`` row on the released-gene-name leaf.

    NPVs 1.5, 0.5, -0.75 for YAL001C, YBR002C, Q0250. Temperature is None with a
    ``not_carried_by_curation`` gap anchored to ``CURATION``; the phenotype carries the
    three curation gaps; the reference NPV is 0.0 in the same environment.
    """
    ds, _ = dataset
    env = _environment(_media("YPD", "solid", is_synthetic=False), FURFURAL, 10.0)
    for idx, orf, npv in [
        (0, "YAL001C", 1.5),
        (1, "YBR002C", 0.5),
        (3, "Q0250", -0.75),
    ]:
        record = ds[idx]
        assert (
            record["experiment"]
            == _experiment(orf, npv, env, "colony size").model_dump()
        )
        assert record["reference"] == (
            _reference(env, "colony size", "BY4743", "diploid").model_dump()
        )
        assert (
            record["publication"]
            == Publication(
                pubmed_id="99999901",
                pubmed_url="https://pubmed.ncbi.nlm.nih.gov/99999901/",
            ).model_dump()
        )
    assert (
        EnvironmentResponseExperiment.model_validate(ds[0]["experiment"]).model_dump()
        == ds[0]["experiment"]
    )


def test_hap_a_turbidity_records_skip_blank_cell(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Records 4 and 6 are screen A column 2: SD liquid, unresolved quinine 2 mM, BY4741
    haploid (record 5 is the ``not_an_orf`` row).

    YBR002C's blank cell yields no record, so the column contributes YAL001C (-2.25) and
    Q0250 (0.125) only. ``culture turbidity`` -> liquid state and the readout in units.
    """
    ds, _ = dataset
    env = _environment(
        _media("SD", "liquid", is_synthetic=True), QUININE_UNRESOLVED, 2.0
    )
    for idx, orf, npv in [(4, "YAL001C", -2.25), (6, "Q0250", 0.125)]:
        record = ds[idx]
        assert record["experiment"] == (
            _experiment(orf, npv, env, "culture turbidity").model_dump()
        )
        assert record["reference"] == (
            _reference(env, "culture turbidity", "BY4741", "haploid").model_dump()
        )
        assert record["publication"]["pubmed_id"] == "99999901"


def test_het_column_is_a_copy_number_drop_in_by4743(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Records 7-10 are screen A column 3 (``het``): one copy dropped 2 -> 1 with a
    KanMX marker, the same furfural environment as column 1, referenced to diploid
    BY4743; the ``not_an_orf`` row (record 9) is a ``heterozygous_deletion`` on the
    released-gene-name leaf.
    """
    ds, _ = dataset
    env = _environment(_media("YPD", "solid", is_synthetic=False), FURFURAL, 10.0)
    perturbation = EngineeredCopyNumberPerturbation(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="YAL001C",
        copy_number=1,
        reference_copy_number=2,
        marker="KanMX",
    )
    assert ds[7]["experiment"]["genotype"]["perturbations"] == [
        perturbation.model_dump()
    ]
    assert ds[7]["experiment"]["environment"] == env.model_dump()
    assert ds[7]["experiment"]["phenotype"]["environment_response"] == 1.0
    (released,) = ds[9]["experiment"]["genotype"]["perturbations"]
    assert (
        released["perturbation_type"],
        released["systematic_gene_name"],
        released["released_kind"],
        released["cassette"],
    ) == ("released_identifier", "not_an_orf", "heterozygous_deletion", "KanMX")
    assert ds[2]["experiment"]["genotype"]["perturbations"][0]["released_kind"] == (
        "deletion"
    )
    assert ds[7]["reference"] == (
        _reference(env, "colony size", "BY4743", "diploid").model_dump()
    )


def test_gapped_media_columns_are_stored(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Records 8-10 (4-field header) carry a ``Media`` whose name, state and synthetic
    flag are typed gaps; records 11-13 (``LB``) carry the released name with
    ``is_synthetic`` a typed gap.
    """
    ds, _ = dataset
    absent = ds[11]["experiment"]["environment"]["media"]
    assert (absent["name"], absent["state"], absent["is_synthetic"]) == (
        None,
        None,
        None,
    )
    assert [g["field"] for g in absent["provenance_gaps"]] == [
        "name",
        "state",
        "is_synthetic",
    ]
    lb = ds[15]["experiment"]["environment"]["media"]
    assert (lb["name"], lb["state"], lb["is_synthetic"]) == ("LB", "solid", None)
    assert [g["field"] for g in lb["provenance_gaps"]] == ["is_synthetic"]


def test_temperature_and_time_components_set_the_environment(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Records 14-16 (``temperature [37 C], time [48 h]``) store 37 C with no
    temperature gap, 48 h and no perturbation.
    """
    ds, _ = dataset
    environment = ds[19]["experiment"]["environment"]
    assert environment["temperature"] == {"value": 37.0, "unit": "Celsius"}
    assert environment["duration_hours"] == 48.0
    assert environment["perturbations"] == []
    assert environment["provenance_gaps"] == []


def test_second_screen_hap_alpha_record_and_publication(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Record 17 is screen B: SC (from ``SC + EtOH``) liquid, BY4742 haploid, PMID 99999902."""
    ds, _ = dataset
    env = _environment(
        _media(
            "SC + EtOH",
            "liquid",
            screen=SCREEN_B,
            is_synthetic=True,
            components=[_VEHICLE_ETHANOL],
        ),
        FURFURAL,
        10.0,
    )
    record = ds[23]
    assert record["experiment"] == (
        _experiment("YAL001C", 2.0, env, "pooled barseq").model_dump()
    )
    assert record["reference"] == (
        _reference(env, "pooled barseq", "BY4742", "haploid").model_dump()
    )
    assert (
        record["publication"]
        == Publication(
            pubmed_id="99999902", pubmed_url="https://pubmed.ncbi.nlm.nih.gov/99999902/"
        ).model_dump()
    )


def test_reference_index_groups_by_environment(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Six references: the hom and het furfural columns share one (same environment,
    units and BY4743 diploid reference), then hap a quinine, the gapped-media column,
    the LB column, the 37 C column and the hap alpha screen.
    """
    ds, root = dataset
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    members = [
        [0, 1, 2, 3, 7, 8, 9, 10],
        [4, 5, 6],
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22],
        [23],
    ]
    assert [e["member_indices"] for e in index] == members
    assert [e["reference"]["genome_reference"]["strain"] for e in index] == [
        "BY4743",
        "BY4741",
        "BY4743",
        "BY4743",
        "BY4743",
        "BY4742",
    ]
    loaded = ds.experiment_reference_index
    assert loaded is not None
    assert [eri.member_indices for eri in loaded] == members


def test_columns_not_stored_are_logged_by_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The final log line counts every reason once for screen A: the dose-range column
    (not representable), the expression column (not growth), the 2-field header
    (malformed); the ``not_an_orf`` rows of the six stored columns are records on the
    released-gene-name leaf, listed in the ledger.
    ``dropped_records.json`` lists the not-representable column with its components and
    its 4 released values.
    """
    monkeypatch.setattr(yeastphenome, "SCREENS", [SCREEN_A])
    root = tmp_path / "yeastphenome"
    (root / "raw").mkdir(parents=True)
    _write_screen(root / "raw", SCREEN_A, HEADER_A, ROWS_A)
    with caplog.at_level("INFO", logger="torchcell.datasets.scerevisiae.yeastphenome"):
        ds = YeastPhenomeDataset(root=str(root))
    assert len(ds) == 23
    summary = [r for r in caplog.records if r.getMessage().startswith("Wrote ")]
    assert len(summary) == 1
    assert summary[0].getMessage() == (
        "Wrote 23 YeastPhenome records over 6 environments (1 screens); dropped "
        "columns/rows: {'condition_not_representable': 1, 'not_growth': 1, "
        "'malformed_header': 1}"
    )
    ledger = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert ledger["condition_not_representable"] == [
        {
            "screen": "synthetic_a",
            "pmid": "99999901",
            "column": 4,
            "condition": "temperature [37 C], hydrogen peroxide [1-1.2 mM]",
            "components": ["hydrogen peroxide [1-1.2 mM]"],
            "n_records": 4,
        }
    ]
    assert ledger["stored_on_released_gene_name_leaf"] == ["not_an_orf"]


def test_raw_file_names_follow_patched_screens(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raw_file_names`` is derived from the module-level ``SCREENS`` at call time."""
    monkeypatch.setattr(yeastphenome, "SCREENS", [SCREEN_A, SCREEN_B])
    prop = vars(YeastPhenomeDataset)["raw_file_names"]
    names = prop.fget(object.__new__(YeastPhenomeDataset))
    assert names == [
        "99999901_synthetic_a_valuez.txt",
        "99999902_synthetic_b_valuez.txt",
    ]
