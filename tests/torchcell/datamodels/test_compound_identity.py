# tests/torchcell/datamodels/test_compound_identity
# [[tests.torchcell.datamodels.test_compound_identity]]
"""Unit tests for the shared, pure, offline compound-identity resolver (UI-2).

Every test reads ONLY the committed ``compound_identity_table.json`` -- none hits the
network (the resolver never does). Covers: resolution by name, by synonym and by CID; the
six ``CompoundResolutionStatus`` outcomes; the RDKit SMILES route and the curated row's
precedence over it; the canonical-name policy that collapses ``NaCl`` and
``sodium chloride`` onto one compound; mixtures that carry an identifier but no InChIKey;
UNRESOLVED / PROPRIETARY rows carrying an audit reason; and the pinned-table sha256
self-check plus lookup-key uniqueness.

2026.09.30 (Phase 17): the table loader's two refusals and its indexing run on tiny
tables written under ``tmp_path`` with ``_TABLE_PATH`` and ``_TABLE_SHA256`` patched (the
loader reads both module globals at call time). The sha256 refusal quotes the digest of
the bytes actually read; the collision refusal fires on ``sodium chloride`` versus
``NaCl`` because ``_SYNONYMS`` folds ``nacl`` onto ``sodium chloride`` before indexing.
A row's own name and synonyms folding onto one key is not a collision, and a CID shared
by two rows keeps the FIRST row (``setdefault``). The resolver pins: a name match outranks
the CID, the CID backs an unmatched name, ``known_proprietary`` only speaks when no row
matches, an unparseable SMILES leaves the bare resolution, and caller-supplied fields
win over the row while ``inchi`` and ``roles`` pass straight through. The furfural row
values (InChIKey ``HYBBIBNJHNGZAN-UHFFFAOYSA-N``, ChEBI ``CHEBI:30976``, SMILES
``C1=COC(=C1)C=O``) are the committed table's.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import torchcell.datamodels.compound_identity as compound_identity
from torchcell.datamodels.compound_identity import (
    _BY_NAME,
    _TABLE_PATH,
    _TABLE_SHA256,
    CompoundResolutionStatus,
    _load_table,
    inchikey_from_smiles,
    normalize_compound_name,
    resolve_compound_identity,
    resolve_compound_identity_from_smiles,
    resolved_compound,
)
from torchcell.datamodels.schema import CHEBI_ID_PATTERN, INCHIKEY_PATTERN


# --------------------------------------------------------------------------- #
# Resolution routes
# --------------------------------------------------------------------------- #
def test_resolved_returns_valid_structure() -> None:
    res = resolve_compound_identity(name="furfural")
    assert res.status == CompoundResolutionStatus.RESOLVED
    assert res.name == "furfural"
    assert res.inchikey is not None
    assert re.match(INCHIKEY_PATTERN, res.inchikey)
    assert res.pubchem_cid is not None and res.pubchem_cid > 0
    # furfural carries a hand-verified ChEBI id in the curated core
    assert res.chebi_id == "CHEBI:30976"
    assert re.match(CHEBI_ID_PATTERN, res.chebi_id)
    assert res.curated and res.identified


def test_resolve_by_pubchem_cid() -> None:
    # a wildenhain CID present in the table (resolved via the CID route)
    res = resolve_compound_identity(pubchem_cid=2795643)
    assert res.status == CompoundResolutionStatus.RESOLVED
    assert res.inchikey is not None
    assert re.match(INCHIKEY_PATTERN, res.inchikey)


def test_resolve_by_synonym_uses_the_curated_row() -> None:
    # Smith 2016 releases the alias "NSC-180973 (tamoxifen)"; the label is a synonym
    # of the tamoxifen row, so the label resolves to tamoxifen's structure.
    alias = resolve_compound_identity(name="NSC-180973")
    canonical = resolve_compound_identity(name="tamoxifen")
    assert alias.status == CompoundResolutionStatus.RESOLVED
    assert alias.inchikey == canonical.inchikey
    assert alias.name == canonical.name == "tamoxifen"


def test_unresolved_public_has_status_and_no_structure() -> None:
    res = resolve_compound_identity(name="totally-not-a-real-compound-xyz")
    assert res.status == CompoundResolutionStatus.UNRESOLVED_PUBLIC
    assert res.name is None
    assert not res.curated and not res.identified
    assert res.inchikey is None
    assert res.chebi_id is None
    assert res.pubchem_cid is None
    assert res.smiles is None


def test_known_proprietary_returns_proprietary_status() -> None:
    res = resolve_compound_identity(name="CMB99999 [x]", known_proprietary=True)
    assert res.status == CompoundResolutionStatus.PROPRIETARY
    assert res.inchikey is None


def test_name_normalization_case_whitespace_and_synonym() -> None:
    base = resolve_compound_identity(name="furfural")
    assert base.status == CompoundResolutionStatus.RESOLVED
    # case + surrounding whitespace fold to the same record
    spaced = resolve_compound_identity(name="  FuRfUrAl  ")
    assert spaced.inchikey == base.inchikey
    # documented synonym: 'H2O2' -> 'hydrogen peroxide'
    assert normalize_compound_name("H2O2") == "hydrogen peroxide"
    h2o2 = resolve_compound_identity(name="H2O2")
    hp = resolve_compound_identity(name="hydrogen peroxide")
    assert h2o2.status == CompoundResolutionStatus.RESOLVED
    assert h2o2.inchikey == hp.inchikey


# --------------------------------------------------------------------------- #
# Canonical-name policy
# --------------------------------------------------------------------------- #
def test_canonical_name_collapses_two_spellings_onto_one_compound() -> None:
    # Hillenmeyer spells it NaCl, the served Nadal-Ribelles loader spells it
    # sodium chloride. Both must produce the SAME Compound, node id included.
    hillenmeyer = resolved_compound("NaCl")
    nadal = resolved_compound("sodium chloride")
    assert hillenmeyer.name == nadal.name == "sodium chloride"
    assert hillenmeyer.model_dump() == nadal.model_dump()


def test_canonical_name_keeps_a_curated_spelling_over_the_pubchem_title() -> None:
    # media.py owns "D-glucose"; PubChem titles CID 5793 "Glucose". The bench spelling
    # wins and the PubChem-ish spelling resolves to it.
    curated = resolved_compound("D-glucose")
    assert curated.name == "D-glucose"
    assert resolved_compound("glucose").name == "D-glucose"
    assert resolved_compound("glucose").inchikey == curated.inchikey


def test_unmatched_label_keeps_its_own_name() -> None:
    compound = resolved_compound("totally-not-a-real-compound-xyz")
    assert compound.name == "totally-not-a-real-compound-xyz"


# --------------------------------------------------------------------------- #
# The RDKit SMILES route
# --------------------------------------------------------------------------- #
def test_inchikey_from_smiles_parses_and_reports_failure() -> None:
    assert inchikey_from_smiles("CCO") == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    # an unparseable SMILES is a measurement, not an exception
    assert inchikey_from_smiles("B1OC2C(O1)") is None


def test_curated_row_wins_over_the_smiles_route() -> None:
    # Measured: the two routes disagree in the stereo block for concanamycin A, so the
    # curated key must survive even when a caller also supplies the SMILES.
    curated = resolve_compound_identity(name="concanamycin A")
    assert curated.smiles is not None
    derived = inchikey_from_smiles(curated.smiles)
    assert curated.inchikey is not None
    assert derived is not None and derived != curated.inchikey
    assert derived.split("-")[0] == curated.inchikey.split("-")[0]  # same skeleton

    resolution = resolve_compound_identity_from_smiles("concanamycin A", curated.smiles)
    assert resolution.status == CompoundResolutionStatus.RESOLVED
    assert resolution.inchikey == curated.inchikey

    compound = resolved_compound(
        "concanamycin A", smiles=curated.smiles, derive_from_smiles=True
    )
    assert compound.inchikey == curated.inchikey


def test_smiles_route_fills_an_uncurated_label() -> None:
    resolution = resolve_compound_identity_from_smiles("not-in-the-table-xyz", "CCO")
    assert resolution.status == CompoundResolutionStatus.RESOLVED_FROM_SMILES
    assert resolution.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"

    compound = resolved_compound(
        "not-in-the-table-xyz", smiles="CCO", derive_from_smiles=True
    )
    assert compound.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert compound.provenance_gaps == []
    # without the opt-in the same call stays an honest gap
    assert resolved_compound("not-in-the-table-xyz", smiles="CCO").inchikey is None


# --------------------------------------------------------------------------- #
# Mixtures and terminal rows
# --------------------------------------------------------------------------- #
def test_homolog_mixture_carries_an_identifier_but_no_inchikey() -> None:
    res = resolve_compound_identity(name="tunicamycin")
    assert res.status == CompoundResolutionStatus.RESOLVED_MIXTURE
    assert res.inchikey is None, (
        "a homolog mixture must not carry a single-molecule key"
    )
    assert res.chebi_id == "CHEBI:29699"
    assert res.identified, "ChEBI alone satisfies the identity rule"
    assert res.unresolved_reason is not None and "mixture" in res.unresolved_reason

    compound = resolved_compound("tunicamycin")
    assert compound.inchikey is None
    assert compound.chebi_id == "CHEBI:29699"
    assert [g.reason.value for g in compound.provenance_gaps] == [
        "not_reported_by_primary"
    ]


def test_undefined_preparation_has_no_identifier_at_all() -> None:
    res = resolve_compound_identity(name="yeast extract")
    assert res.status == CompoundResolutionStatus.UNDEFINED_MIXTURE
    assert not res.identified
    assert res.unresolved_reason is not None

    compound = resolved_compound("peptone")
    assert compound.inchikey is None and compound.pubchem_cid is None
    assert [g.reason.value for g in compound.provenance_gaps] == [
        "not_reported_by_primary"
    ]


def test_vendor_catalog_code_is_proprietary_with_a_recorded_reason() -> None:
    # Smith 2016's ChemDiv code: Additional file 8 releases no structure for it, so the
    # drop is auditable from the table rather than from a code comment.
    res = resolve_compound_identity(name="1181-0519")
    assert res.status == CompoundResolutionStatus.PROPRIETARY
    assert res.inchikey is None and not res.identified
    assert res.unresolved_reason is not None and "ChemDiv" in res.unresolved_reason
    assert [g.reason.value for g in resolved_compound("1181-0519").provenance_gaps] == [
        "not_reported_by_primary"
    ]


def test_unresolved_row_records_why_and_stays_recoverable() -> None:
    res = resolve_compound_identity(name="Boromycin")
    assert res.status == CompoundResolutionStatus.UNRESOLVED_PUBLIC
    assert res.inchikey is None
    assert res.unresolved_reason is not None and "RDKit" in res.unresolved_reason
    assert [g.reason.value for g in resolved_compound("Boromycin").provenance_gaps] == [
        "deferred_pending_source_review"
    ]


def test_resolved_compound_no_clobber_caller_smiles() -> None:
    # hoepfner path: caller already set a SMILES; resolver must not overwrite it, and
    # (unresolved-by-name proprietary) must gap inchikey as not_reported_by_primary.
    caller_smiles = "C1=CC=CC=C1"
    compound = resolved_compound(
        "CMBfoo [tag]", smiles=caller_smiles, known_proprietary=True
    )
    assert compound.smiles == caller_smiles
    assert compound.inchikey is None
    assert len(compound.provenance_gaps) == 1
    gap = compound.provenance_gaps[0]
    assert gap.field == "inchikey"
    assert gap.reason.value == "not_reported_by_primary"


def test_resolved_compound_resolved_has_no_gap() -> None:
    compound = resolved_compound("furfural")
    assert compound.inchikey is not None
    assert compound.chebi_id == "CHEBI:30976"
    assert compound.provenance_gaps == []


# --------------------------------------------------------------------------- #
# Table integrity
# --------------------------------------------------------------------------- #
def test_table_sha256_self_check_matches_pinned() -> None:
    digest = hashlib.sha256(_TABLE_PATH.read_bytes()).hexdigest()
    assert digest == _TABLE_SHA256


def test_table_load_is_deterministic() -> None:
    records_a, by_name_a, by_cid_a = _load_table()
    records_b, by_name_b, by_cid_b = _load_table()
    assert [r.model_dump() for r in records_a] == [r.model_dump() for r in records_b]
    assert set(by_name_a) == set(by_name_b)
    assert set(by_cid_a) == set(by_cid_b)


def test_every_lookup_key_names_exactly_one_compound() -> None:
    # _load_table raises on a contested key; assert the pinned table is actually clean
    # and that the index is materially larger than the record count (synonyms exist).
    records, by_name, _ = _load_table()
    assert len(by_name) > len(records)
    assert len({id(r) for r in by_name.values()}) == len(
        {id(r) for r in records if r.name}
    )


def test_resolved_rows_carry_a_well_formed_structure() -> None:
    records, _, _ = _load_table()
    resolved = [r for r in records if r.resolution_status == "RESOLVED"]
    assert len(resolved) > 5000
    for record in resolved:
        assert record.inchikey is not None
        assert re.match(INCHIKEY_PATTERN, record.inchikey)
        assert record.unresolved_reason is None
    for record in records:
        if record.chebi_id is not None:
            assert re.match(CHEBI_ID_PATTERN, record.chebi_id)
        if record.resolution_status != "RESOLVED":
            assert record.inchikey is None
            assert record.unresolved_reason, f"{record.name} has no audit reason"


def test_every_record_round_trips_into_a_compound() -> None:
    # the schema's own validators (InChIKey shape, ChEBI CURIE, positive CID) are the
    # real check that the curated table is servable
    records, _, _ = _load_table()
    for record in records:
        compound = resolved_compound(record.name)
        assert compound.name == record.name
        assert compound.inchikey == record.inchikey
    assert _BY_NAME  # the module-level index was built at import


def test_table_is_sorted_and_unique_by_name() -> None:
    records, _, _ = _load_table()
    names = [r.name for r in records]
    assert names == sorted(names, key=str.lower)
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------- #
# Phase 17: normalization, route precedence, the SMILES misses, table refusals
# --------------------------------------------------------------------------- #
def test_normalize_folds_only_documented_spellings() -> None:
    """Case and outer whitespace fold, the ``_SYNONYMS`` table applies, nothing else.

    Internal whitespace is kept and a near-miss hyphenation is not folded, per the
    "never a fuzzy near-miss" rule.
    """
    assert normalize_compound_name("  NaCl ") == "sodium chloride"
    assert normalize_compound_name("Sodium Chloride (NaCl)") == "sodium chloride"
    assert normalize_compound_name("EtOH") == "ethanol"
    assert normalize_compound_name("N-Propanol") == "1-propanol"
    assert normalize_compound_name("Foo  Bar") == "foo  bar"
    assert normalize_compound_name("methyl-methanesulfonate") == (
        "methyl-methanesulfonate"
    )


def test_name_match_outranks_cid_and_cid_backs_an_unmatched_name() -> None:
    """Name first, then CID; ``known_proprietary`` speaks only when no row matched."""
    tamoxifen_cid = resolve_compound_identity(name="tamoxifen").pubchem_cid
    assert tamoxifen_cid is not None
    by_name = resolve_compound_identity(name="furfural", pubchem_cid=tamoxifen_cid)
    assert by_name.name == "furfural"
    by_cid = resolve_compound_identity(name="zzz-no-row", pubchem_cid=tamoxifen_cid)
    assert (by_cid.status, by_cid.name) == (
        CompoundResolutionStatus.RESOLVED,
        "tamoxifen",
    )
    row_wins = resolve_compound_identity(name="furfural", known_proprietary=True)
    assert row_wins.status == CompoundResolutionStatus.RESOLVED
    missing_cid = resolve_compound_identity(pubchem_cid=-1, known_proprietary=True)
    assert missing_cid.model_dump() == {
        "status": CompoundResolutionStatus.PROPRIETARY,
        "name": None,
        "inchikey": None,
        "chebi_id": None,
        "pubchem_cid": None,
        "smiles": None,
        "unresolved_reason": None,
    }


def test_smiles_route_with_an_unparseable_smiles_returns_the_bare_miss() -> None:
    """No row and a SMILES RDKit rejects: the name miss comes back, SMILES not carried."""
    resolution = resolve_compound_identity_from_smiles("zzz-no-row", "B1OC2C(O1)")
    assert resolution.model_dump() == {
        "status": CompoundResolutionStatus.UNRESOLVED_PUBLIC,
        "name": None,
        "inchikey": None,
        "chebi_id": None,
        "pubchem_cid": None,
        "smiles": None,
        "unresolved_reason": None,
    }


def test_resolved_compound_unparseable_smiles_keeps_the_caller_smiles_and_defers() -> (
    None
):
    """The derivation miss leaves a recoverable gap and the caller's SMILES in place."""
    compound = resolved_compound(
        "zzz-no-row", smiles="B1OC2C(O1)", derive_from_smiles=True
    )
    assert compound.model_dump(mode="json") == {
        "provenance_gaps": [
            {
                "field": "inchikey",
                "reason": "deferred_pending_source_review",
                "looked_in": None,
                "resolve_with": None,
                "note": None,
            }
        ],
        "name": "zzz-no-row",
        "inchikey": None,
        "inchi": None,
        "smiles": "B1OC2C(O1)",
        "pubchem_cid": None,
        "chebi_id": None,
        "roles": [],
    }


def test_resolved_compound_caller_fields_win_and_pass_through() -> None:
    """A caller CID and SMILES outrank the row; ``inchi`` and ``roles`` pass through."""
    compound = resolved_compound(
        "Furfural", pubchem_cid=1, smiles="O=Cc1ccco1", inchi="InChI=1S/x", roles=["r"]
    )
    assert compound.model_dump(mode="json") == {
        "provenance_gaps": [],
        "name": "furfural",
        "inchikey": "HYBBIBNJHNGZAN-UHFFFAOYSA-N",
        "inchi": "InChI=1S/x",
        "smiles": "O=Cc1ccco1",
        "pubchem_cid": 1,
        "chebi_id": "CHEBI:30976",
        "roles": ["r"],
    }


def _pin_table(
    monkeypatch: pytest.MonkeyPatch, path: Path, records: list[dict[str, object]]
) -> bytes:
    """Write a table to ``path`` and point the loader at it with a matching digest."""
    raw = json.dumps({"records": records}).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(compound_identity, "_TABLE_PATH", path)
    monkeypatch.setattr(
        compound_identity, "_TABLE_SHA256", hashlib.sha256(raw).hexdigest()
    )
    return raw


def test_load_table_refuses_bytes_that_miss_the_pinned_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A table whose bytes do not hash to ``_TABLE_SHA256`` is refused, both digests named."""
    path = tmp_path / "table.json"
    raw = _pin_table(monkeypatch, path, [])
    monkeypatch.setattr(compound_identity, "_TABLE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError) as excinfo:
        _load_table()
    assert str(excinfo.value) == (
        f"compound_identity_table.json sha256 mismatch: got "
        f"{hashlib.sha256(raw).hexdigest()}, expected {'0' * 64} (table tampered or "
        "re-built -- re-pin _TABLE_SHA256)"
    )


def test_load_table_refuses_a_key_claimed_by_two_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``NaCl`` folds onto ``sodium chloride`` before indexing, so two rows collide."""
    _pin_table(
        monkeypatch,
        tmp_path / "table.json",
        [
            {"name": "sodium chloride", "resolution_status": "RESOLVED"},
            {"name": "NaCl", "resolution_status": "RESOLVED"},
        ],
    )
    with pytest.raises(RuntimeError) as excinfo:
        _load_table()
    assert str(excinfo.value) == (
        "compound_identity_table.json: lookup key 'sodium chloride' is claimed by "
        "both 'sodium chloride' and 'NaCl'"
    )


def test_load_table_indexes_self_synonyms_once_and_keeps_the_first_cid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A row's own spellings folding to one key are fine; a shared CID keeps row one."""
    _pin_table(
        monkeypatch,
        tmp_path / "table.json",
        [
            {
                "name": "hydrogen peroxide",
                "synonyms": ["H2O2", "Hydrogen Peroxide"],
                "pubchem_cid": 784,
                "resolution_status": "RESOLVED",
            },
            {
                "name": "peroxide dup",
                "pubchem_cid": 784,
                "resolution_status": "RESOLVED",
            },
        ],
    )
    records, by_name, by_cid = _load_table()
    assert [r.name for r in records] == ["hydrogen peroxide", "peroxide dup"]
    assert {key: row.name for key, row in by_name.items()} == {
        "hydrogen peroxide": "hydrogen peroxide",
        "peroxide dup": "peroxide dup",
    }
    assert {cid: row.name for cid, row in by_cid.items()} == {784: "hydrogen peroxide"}
