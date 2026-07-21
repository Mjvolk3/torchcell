# tests/torchcell/datamodels/test_compound_identity
# [[tests.torchcell.datamodels.test_compound_identity]]
"""Unit tests for the shared, pure, offline compound-identity resolver (UI-2).

Every test reads ONLY the committed ``compound_identity_table.json`` -- none hits the
network (the resolver never does). Covers: RESOLVED / UNRESOLVED_PUBLIC / PROPRIETARY
statuses, resolve-by-CID, additive no-clobber of a caller ``smiles``, name
normalization (case/whitespace + synonym map), and the pinned-table sha256 self-check +
load determinism.
"""

from __future__ import annotations

import hashlib
import re

from torchcell.datamodels.compound_identity import (
    _TABLE_PATH,
    _TABLE_SHA256,
    CompoundResolutionStatus,
    _load_table,
    normalize_compound_name,
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.schema import CHEBI_ID_PATTERN, INCHIKEY_PATTERN


def test_resolved_returns_valid_structure() -> None:
    res = resolve_compound_identity(name="furfural")
    assert res.status == CompoundResolutionStatus.RESOLVED
    assert res.inchikey is not None
    assert re.match(INCHIKEY_PATTERN, res.inchikey)
    assert res.pubchem_cid is not None and res.pubchem_cid > 0
    # furfural carries a hand-verified ChEBI id in the common core
    assert res.chebi_id == "CHEBI:30976"
    assert re.match(CHEBI_ID_PATTERN, res.chebi_id)


def test_unresolved_public_has_status_and_no_structure() -> None:
    res = resolve_compound_identity(name="totally-not-a-real-compound-xyz")
    assert res.status == CompoundResolutionStatus.UNRESOLVED_PUBLIC
    assert res.inchikey is None
    assert res.chebi_id is None
    assert res.pubchem_cid is None
    assert res.smiles is None


def test_known_proprietary_returns_proprietary_status() -> None:
    res = resolve_compound_identity(name="CMB99999 [x]", known_proprietary=True)
    assert res.status == CompoundResolutionStatus.PROPRIETARY
    assert res.inchikey is None


def test_resolve_by_pubchem_cid() -> None:
    # a wildenhain seed CID present in the table (resolved via the CID route)
    res = resolve_compound_identity(pubchem_cid=2795643)
    assert res.status == CompoundResolutionStatus.RESOLVED
    assert res.inchikey is not None
    assert re.match(INCHIKEY_PATTERN, res.inchikey)


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


def test_resolved_compound_public_unresolved_is_deferred_gap() -> None:
    compound = resolved_compound("tunicamycin")
    assert compound.inchikey is None
    assert len(compound.provenance_gaps) == 1
    assert compound.provenance_gaps[0].field == "inchikey"
    assert compound.provenance_gaps[0].reason.value == "deferred_pending_source_review"


def test_resolved_compound_resolved_has_no_gap() -> None:
    compound = resolved_compound("furfural")
    assert compound.inchikey is not None
    assert compound.chebi_id == "CHEBI:30976"
    assert compound.provenance_gaps == []


def test_table_sha256_self_check_matches_pinned() -> None:
    digest = hashlib.sha256(_TABLE_PATH.read_bytes()).hexdigest()
    assert digest == _TABLE_SHA256


def test_table_load_is_deterministic() -> None:
    records_a, by_name_a, by_cid_a = _load_table()
    records_b, by_name_b, by_cid_b = _load_table()
    assert [r.model_dump() for r in records_a] == [r.model_dump() for r in records_b]
    assert set(by_name_a) == set(by_name_b)
    assert set(by_cid_a) == set(by_cid_b)
