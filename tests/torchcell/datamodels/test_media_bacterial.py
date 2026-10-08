# tests/torchcell/datamodels/test_media_bacterial.py
"""The bacterial media library additions ([[plan.bacteria-ontology-genome]] Step 5).

Four properties, one per claim the additions make:

1. **No pre-existing medium moved.** ``media.py`` is in the KG value surface, so adding a
   key is a value-drift finding at admission; the acknowledgment's reason is that no
   EXISTING key's composition changed. ``PRE_EXISTING_MEDIA_DIGESTS`` pins the
   ``media_identity`` digest of every key the library held before the bacterial entries
   (computed 2026-10-07 on main at bb31eabbe and on this branch, identical), so that
   reason is a check, not an assertion. Every library key is either pinned here or
   declared in ``BACTERIAL_MEDIA_USES``, so a key cannot slip in unpinned.
2. **Every entry imports and joins.** ``_check_library`` runs at import; every
   single-substance component and dropout carries a structure identifier from the shared
   table, and every undefined preparation is a bare, typed mixture.
3. **Every entry says where its carbon comes from**, or why it names none.
4. **Every quote is verbatim** in the mirrored file it names (``--data``: reads the
   ``$DATA_ROOT/torchcell-library`` mirror; CI without the mirror skips).
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path

import pytest

from torchcell.datamodels import ontology_checks as oc
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.identity import identity_sha256, media_identity
from torchcell.datamodels.media import (
    BACTERIAL_MEDIA_USES,
    CARBON_FREE_MEDIA,
    LB,
    LB_AGAR,
    LB_LENNOX,
    M9,
    M9_NOCARBON_PRICE2018,
    M9_NONITROGEN_WETMORE2015,
    M9_SCHMIDT2016,
    MEDIA_LIBRARY,
    MOPS_MINIMAL,
    _check_library,
)
from torchcell.datamodels.schema import (
    ComponentDefinition,
    ConcentrationUnit,
    Media,
    MediaComponentRole,
)
from torchcell.verification.report import sha256_file
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

#: ``identity_sha256(media_identity(m))`` for every key MEDIA_LIBRARY held before the
#: bacterial additions. A change here means an existing medium's served node id moved.
PRE_EXISTING_MEDIA_DIGESTS: dict[str, str] = {
    "SC": "d7ea21cf75fb333f87cbc98974171f865f6466ac226020596c4c5ce8b6536f7d",
    "SC_MINUS_4_AMINOBENZOIC_ACID": "9487cb0bd907a7ac5025cb3f28f26f280084c79093b1103a6263c4896ecc13bb",
    "SC_MINUS_ADENINE": "83ddd124e6c6dd9a767cdc5cd7221afac849a05d2b1f839edb2c63cdc4f24d9d",
    "SC_MINUS_FOLIC_ACID": "3e6232f5c94a1198a28199e4faece001edff6659bfaa5c01290ad326a1117de1",
    "SC_MINUS_L_ARGININE": "c14d01fb15b765047aeed5317ac796e1f4b5da9923a6016d9b5650861939faf0",
    "SC_MINUS_L_ISOLEUCINE": "53a32e3ce6bb4af127ad308a5467efaca4d648847a33904e2ab915215c6617dc",
    "SC_MINUS_L_LYSINE": "707ea7765d07287f4fd6664b3855af319bd4e6adc059a62ab153bf6afb84df16",
    "SC_MINUS_L_THREONINE": "31500c6d8ff6a8d4567dd67a0ccea5c72f2ebd83a59dc4e59d00d5a59aba977c",
    "SC_MINUS_L_TRYPTOPHAN": "3413d64e6c42a237532a3aeadbf6a75ddabe0c7ebe2f75c96bb84e7243d5fdc8",
    "SC_MINUS_L_TYROSINE": "04d09ebdda25d33a02c5a52251380640070e2db23a84848493bfa111d0085a87",
    "SC_MINUS_MYO_INOSITOL": "c7dbc81c7a73f0ee9a5c424f19197ebad70132d995666cbea83bf940180a5c3e",
    "SC_MINUS_NIACIN": "c532a3630f0933d0e05d8f17bfa157de1c44711014232b2a5ec6b1344c228237",
    "SC_MINUS_THIAMINE_HYDROCHLORIDE": "9472c478b07f2cfee773167fca799cc52c5e88e233a2d53a7461d46e95d76c0d",
    "SC_PARTIAL_BIOTIN": "47111a14706cb5d8c5a386b6d1f9f078847b3297c52e8f0d2c58e9211db511c1",
    "SC_PARTIAL_CALCIUM_PANTOTHENATE": "05255dc8ba6196d1a7db35a884ca132ea9f04798319dbc9b3ce8187d924bf7ca",
    "SC_PARTIAL_PYRIDOXINE_HYDROCHLORIDE": "f1ae99ef6a39c08bde30136b3f538b0fbc38f94682693823d551d7a94676b9c6",
    "SC_URA": "5a6804ee3c9194e6c9851083d387ef63e970a1e93b7fb9e1e8b4b87e225f34ca",
    "SD": "9f97e5f646c370915786053f2559c2b751fdc416400831105d433206e85e8a09",
    "SD_MSG": "a298d4ac601fd016fe4f0c73a2effd5b9c90b97db359503fdbec9831c1636850",
    "SED_URA": "d77aac10cf3334b4c6f1addc54bb80c14fccad9b37b92dd0156a3fe5066c7546",
    "SED_URA_G418": "024433748ed7299ae82574a41e7a7e831dfd409bab189063dcf28a9e1b917abe",
    "SGA_DM_SELECTION": "3345fde79b3b375aae9a50091addd49f5b2866c2671ec123f377edbf3addd60e",
    "SGA_DM_SELECTION_GALACTOSE": "79f34cb292d7b71a5558d8d21a3b18ef6599abe2a157a87815b92618243a8918",
    "SGA_TM_SELECTION": "12a731c33fa9c6a7d4b476fbb7b29865dedc6b6cc4a577d23d363318697aae40",
    "SM": "dbabb035f9b19f89372341d24b22849c41587e24ef60fd5465012dbdcbe3e2d6",
    "SM_AGAR": "28eee7363aae21f4b077cf85163132709db49ab24d9208d7d397fc187fadd212",
    "SM_DEFERRED": "70273303dd7a8f5d112a45f7852c75801617b2f4ad56b6afeca51c9a83657568",
    "SYNBASE": "0d98c0290f1a6c0072179abf52844eee8e5d2c561933414d760493925aeaeef2",
    "SYNH3_MINUS": "83e630b702a97c4b21b4babb59156a3e7ab865e1ae12dd928ca8b8ba2340254f",
    "YNB": "672d96a8c8909c79b88fd590ffe4d275a66c2943435c49c401947b03a4a69fc5",
    "YNB_GLUCOSE_SOLID": "0ccb74b1348c87cd3360d0b08ff7d9025f306405e216d1b1377689949b153da7",
    "YNB_YE_B": "9399e4f03e1fddd54c5be5fdaa33800d153049795d4a0e19f31e383a258d9087",
    "YP": "975a0c44988efcf29f1f3e7f175cbcb4d7c944b4bd79d432d4c44f9d0d193a21",
    "YPAD": "564931db8675456f0feb08ad2919737f2a25c7c70971323534b5da61ebe94aa2",
    "YPB": "38c797d0083f4b5ba719f503894a25e1714ea9db4ddfdf30344a85f7cd07e037",
    "YPBA": "dee785999cb455040cfdddaed6c0fc6b362a70f77535ab5e13cecfe973e6d1c1",
    "YPBM": "0e2dd32bbcb2f4c7bb4eeb862b070685fe6842288412da6f50852c554980bb12",
    "YPBO": "5592ddd6508bc5011503dfd548bab4a29b88551f4e54def6b12c5ca51681f85d",
    "YPD": "2b16078429d06bb069bbca07329a0e8ded6ad7eb180035b05cce7be85e04e846",
    "YPD_AGAR": "efd2cc1c393c0a0e3e345cbc31c80ef0f05cdbc5f6260693e1717ed137867ec0",
    "YPD_ETHANOL": "9effc24c4fe17e8cbcffbd3b8d40e7347e15bf77378254aece944c333d51b77c",
    "YPD_LIQUID": "aea23796700e8b3a95fce85defa3c999208717499a582a5b7862db7f3f605232",
    "YP_ETHANOL": "265ff28ad0631a85b0a31b4cba1fa4e70fa71effae2e0b9da2f31bf3c5b9f8a8",
    "YP_FRUCTOSE": "4f59fac4c6c22931a91bdb4c7ae742e1b3d2588f1641bdad7d13250257eaaef8",
    "YP_GALACTOSE": "3107f2dcec7daad6d165deb1645a955083619bf953cc7f6d744027fbd50f69dd",
    "YP_GLYCEROL": "926edc450ce9721ef08d8c967a8b491e9c99072bbfb0556ed59a7650ed10b3f9",
    "YP_GLYCEROL_LIQUID": "06005208ecd2678d881ca7ed3f849efb756ae882bdf5627649f3998d1e8946f3",
    "YP_LACTATE": "dcffa6e62fce5d7709938256b7e3d38662ea3d3dcc7f3734fabe9538b603e97f",
    "YP_MALTOSE": "83f7ce2a33e84dcf2acfcdfb512c9fa31fcdce5e4cd1b84ddb59fac24369d83e",
    "YP_MANNOSE": "098b629a0c064922f92aa9f33148cd9d039842527d0ff6f5af4a4a15fbfa3354",
    "YP_RAFFINOSE": "e2168d8e2d6eaa2c4d415c13c5b205e38c9b513b99b591cc7e8db78ff90466a3",
    "YP_SUCROSE": "614872ef40cd02dc25c3a83784099647e3eb36fe386f50fd6e2c56e627e0e5dd",
    "YP_TREHALOSE": "e0a0dc8497e5bb47cae901306473adec17f7d5da1501dc446603317af78849ce",
    "YP_XYLOSE": "72bd8f212ce2869073470a1ba91d35fb0fbd4433e60b907d19fa181cdfd06915",
}

BACTERIAL_KEYS = sorted(BACTERIAL_MEDIA_USES)


def _media_digest(media: Media) -> str:
    return identity_sha256(media_identity(media))


def _sourced(media: Media) -> list[SourcedValue]:
    return [*media.provenance, *(sv for c in media.components for sv in c.provenance)]


# --- 1. no pre-existing medium moved ---------------------------------------- #
@pytest.mark.parametrize("key", sorted(PRE_EXISTING_MEDIA_DIGESTS))
def test_pre_existing_media_digest_is_unchanged(key: str) -> None:
    """The value-drift acknowledgment's evidence: an existing composition is untouched."""
    assert _media_digest(MEDIA_LIBRARY[key]) == PRE_EXISTING_MEDIA_DIGESTS[key]


def test_every_library_key_is_pinned_or_declared_bacterial() -> None:
    """The library is exactly the pinned keys plus the declared bacterial ones."""
    pinned = set(PRE_EXISTING_MEDIA_DIGESTS)
    bacterial = set(BACTERIAL_MEDIA_USES)
    assert pinned.isdisjoint(bacterial)
    assert set(MEDIA_LIBRARY) == pinned | bacterial


def test_a_bacterial_entry_is_a_new_node_not_an_existing_one() -> None:
    """No bacterial composition collides with a pre-existing medium's node id."""
    existing = set(PRE_EXISTING_MEDIA_DIGESTS.values())
    collisions = {
        key for key in BACTERIAL_KEYS if _media_digest(MEDIA_LIBRARY[key]) in existing
    }
    assert collisions == set()


def test_bacterial_entries_are_distinct_nodes() -> None:
    """Two papers' recipes never collapse onto one node unless they are one recipe."""
    digests = [_media_digest(MEDIA_LIBRARY[key]) for key in BACTERIAL_KEYS]
    assert len(set(digests)) == len(digests)


# --- 2. every entry imports and joins ---------------------------------------- #
def test_every_bacterial_base_is_a_library_member() -> None:
    """The bases the bacterial entries derive from are themselves library objects."""
    bases = {MEDIA_LIBRARY[key].base_medium for key in BACTERIAL_KEYS}
    assert bases == {
        "LB",
        "YT_2X",
        "M9",
        "M9_DEFERRED_BANERJEE2025",
        "M9_DIFCO",
        "DAVIS_MINIMAL",
        "MOPS_MINIMAL",
    }
    assert bases <= set(BACTERIAL_MEDIA_USES)
    assert oc.media_base_issues() == []


def test_library_check_refuses_a_base_that_is_not_a_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The import-time check is what would stop an entry naming an absent base."""
    orphan = M9.model_copy(update={"base_medium": "M9_NOT_A_KEY"})
    monkeypatch.setitem(MEDIA_LIBRARY, "ORPHAN", orphan)
    with pytest.raises(RuntimeError, match="M9_NOT_A_KEY"):
        _check_library()


@pytest.mark.parametrize("key", BACTERIAL_KEYS)
def test_every_single_substance_component_resolves(key: str) -> None:
    """A defined component or dropout is the shared table's compound, with an identifier.

    ``resolved_compound`` returns the table's canonical name, so re-resolving the stored
    name must give back the same compound; a bare name would carry no identifier. Agar
    is the one mixture (ChEBI and a CID, no single-molecule InChIKey); every salt, buffer,
    sugar and vitamin carries an InChIKey.
    """
    media = MEDIA_LIBRARY[key]
    singles = [
        *(
            c.compound
            for c in media.components
            if c.definition is ComponentDefinition.defined
        ),
        *media.dropouts,
    ]
    for compound in singles:
        assert oc.compound_has_identity(compound), f"{key}: {compound.name}"
        assert compound.inchikey is not None or compound.name == "agar", compound.name
        assert resolved_compound(compound.name) == compound, f"{key}: {compound.name}"


@pytest.mark.parametrize("key", BACTERIAL_KEYS)
def test_every_undefined_preparation_is_a_typed_bare_mixture(key: str) -> None:
    for component in MEDIA_LIBRARY[key].components:
        if component.definition is ComponentDefinition.defined:
            continue
        assert not oc.compound_has_identity(component.compound), component.compound.name
        assert component.compound.provenance_gaps == [], component.compound.name


def test_tryptone_and_yeast_extract_are_intrinsically_undefined() -> None:
    for media in (LB, LB_AGAR, LB_LENNOX):
        undefined = {
            c.compound.name: c.definition
            for c in media.components
            if c.role is MediaComponentRole.complex_ingredient
        }
        assert undefined == {
            "tryptone": ComponentDefinition.intrinsically_undefined,
            "yeast extract": ComponentDefinition.intrinsically_undefined,
        }
        assert media.is_synthetic is False


def test_lb_is_miller_and_lb_lennox_differs_only_in_sodium_chloride() -> None:
    """The RB-TnSeq releases' 'LB' is Lennox by their own tables; ``LB`` is Miller."""

    def nacl(media: Media) -> float | None:
        component = next(
            c for c in media.components if c.compound.name == "sodium chloride"
        )
        assert component.concentration is not None
        assert component.concentration.unit is ConcentrationUnit.g_per_l
        return component.concentration.value

    assert (nacl(LB), nacl(LB_LENNOX)) == (10.0, 5.0)
    assert [c for c in LB.components if c.compound.name != "sodium chloride"] != []
    assert {c.compound.name for c in LB.components} == {
        c.compound.name for c in LB_LENNOX.components
    }
    assert LB.base_medium == LB_LENNOX.base_medium == LB_AGAR.base_medium == "LB"
    extra = [c for c in LB_AGAR.components if c not in LB.components]
    assert [c.compound.name for c in extra] == ["agar"]


def test_m9_is_the_four_salts_and_every_m9_formulation_derives_from_it() -> None:
    assert sorted(c.compound.name for c in M9.components) == [
        "ammonium chloride",
        "disodium hydrogen phosphate",
        "potassium dihydrogen phosphate",
        "sodium chloride",
    ]
    m9_family = [
        key for key in BACTERIAL_KEYS if MEDIA_LIBRARY[key].base_medium == "M9"
    ]
    assert len(m9_family) == 17
    assert oc.media_derivation_issues() == []


def test_a_replaced_base_salt_is_a_dropout() -> None:
    """Hydrate and nitrogen-salt replacements are typed edits of the M9 salts."""
    assert [d.name for d in M9_SCHMIDT2016.dropouts] == ["ammonium chloride"]
    assert [d.name for d in M9_NOCARBON_PRICE2018.dropouts] == [
        "disodium hydrogen phosphate"
    ]
    assert sorted(d.name for d in M9_NONITROGEN_WETMORE2015.dropouts) == [
        "ammonium chloride",
        "disodium hydrogen phosphate",
    ]
    for media in (M9_NOCARBON_PRICE2018, M9_NONITROGEN_WETMORE2015):
        names = {c.compound.name for c in media.components}
        assert "disodium hydrogen phosphate heptahydrate" in names
        assert "disodium hydrogen phosphate" not in names


def test_mops_minimal_adjudicates_the_sulfate_row_and_defers_to_neidhardt() -> None:
    """Price 2018's alum row is recorded as potassium sulfate, with both rows quoted."""
    sulfate = next(
        c for c in MOPS_MINIMAL.components if c.compound.name == "potassium sulfate"
    )
    assert {sv.provenance.citation_key for sv in sulfate.provenance} == {
        "priceMutantPhenotypesThousands2018",
        "wetmoreRapidQuantificationMutant2015",
    }
    assert "Aluminum potassium sulfate dodecahydrate | 0.276 | mM" in (
        sulfate.provenance[0].quote
    )
    assert "Potassium Sulfate | 0.276 | mM" in sulfate.provenance[1].quote
    assert all(
        c.defers_to == ["neidhardtCultureMediumEnterobacteria1974"]
        for c in MOPS_MINIMAL.components
    )


def test_an_unstated_amount_stays_an_open_gap() -> None:
    """Schmidt 2016's thiamine amount is lost in the pinned OCR, so no number is written."""
    assert M9_SCHMIDT2016.open_gaps == ["thiamine"]


# --- 3. carbon ---------------------------------------------------------------- #
@pytest.mark.parametrize("key", BACTERIAL_KEYS)
def test_every_entry_names_its_carbon_source_or_says_why_not(key: str) -> None:
    media = MEDIA_LIBRARY[key]
    carbon = [c for c in media.components if c.role is MediaComponentRole.carbon_source]
    assert bool(carbon) != (key in CARBON_FREE_MEDIA), key


# --- 4. every quote is verbatim ---------------------------------------------- #
@pytest.mark.parametrize("key", BACTERIAL_KEYS)
def test_every_entry_is_sourced_to_a_pinned_artifact(key: str) -> None:
    sourced = _sourced(MEDIA_LIBRARY[key])
    assert sourced, key
    for sv in sourced:
        assert sv.provenance.citation_key, key
        assert sv.provenance.sha256 is not None and len(sv.provenance.sha256) == 64
        assert sv.quote.strip(), key


def test_every_entry_lists_its_rows_among_the_fifty() -> None:
    for key, uses in BACTERIAL_MEDIA_USES.items():
        assert uses, key
        ranks = [int(use.split(" ", 1)[0]) for use in uses if use[0].isdigit()]
        assert ranks, key
        assert all(1 <= rank <= 50 for rank in ranks), key


@functools.cache
def _xlsx_text(path: Path) -> str:
    """Every sheet's non-empty rows, cells joined by ' | ', rows by ' / '.

    The row rendering the xlsx quotes in ``media.py`` are cut from.
    """
    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(path, read_only=True)
    sheets = []
    for sheet in workbook.worksheets:
        rows = [
            " | ".join(str(cell) for cell in row if cell is not None)
            for row in sheet.iter_rows(values_only=True)
            if any(cell is not None for cell in row)
        ]
        sheets.append(" / ".join(rows))
    return "\n".join(sheets)


@functools.cache
def _docx_text(path: Path) -> str:
    """Every paragraph of a docx, one per line, as ``docx_paragraphs`` renders it.

    The rendering a docx quote in ``media.py`` is cut from. The renderer is the Lim 2022
    loader's, reused rather than reimplemented: two renderings of one file that disagree
    would make this audit lie in either direction.
    """
    from torchcell.datasets.pputida.lim2022 import docx_paragraphs

    return "\n".join(docx_paragraphs(path))


@pytest.mark.data
@pytest.mark.parametrize("key", BACTERIAL_KEYS)
def test_every_quote_is_verbatim_in_the_mirrored_file(key: str) -> None:
    """sha256 matches the pin and the quote is a substring of the pinned file.

    Three source shapes: a Markdown OCR is read as text (``audit_sourced_value``), an
    xlsx as its row rendering, and a docx as its paragraph rendering. The two binary
    shapes re-hash the file first, so a quote is trusted only against the pinned bytes.
    """
    root = Path(os.environ["DATA_ROOT"]) / "torchcell-library"
    for sv in _sourced(MEDIA_LIBRARY[key]):
        path = sv.source_path(root)
        if path.suffix in {".xlsx", ".docx"}:
            assert sha256_file(path) == sv.provenance.sha256, f"{key}: {path}"
            rendered = _xlsx_text(path) if path.suffix == ".xlsx" else _docx_text(path)
            assert sv.quote in rendered, f"{key}: {sv.quote[:60]!r}"
            continue
        result = audit_sourced_value(sv, root)
        assert result.passed, f"{key}: {result.message}"
