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
"""

from __future__ import annotations

import hashlib
import re

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
