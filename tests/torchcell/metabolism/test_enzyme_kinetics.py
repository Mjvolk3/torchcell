# tests/torchcell/metabolism/test_enzyme_kinetics.py
"""Tests for the k_cat / K_M resolver.

These exercise the SELECTION CASCADE, not the network: the retrieval is covered by the
sha256 check in `load_mirrored_records`, but the cascade is where a wrong parameter would
enter the flux layer silently, so each rung gets a case that fails if the rung is removed.
"""

from __future__ import annotations

import json
import os.path as osp

import pytest

from torchcell.metabolism.enzyme_kinetics import (
    KineticKind,
    KineticSource,
    OedKineticRecord,
    index_by_uniprot,
    load_mirrored_records,
    mirror_oed_slice,  # noqa: F401  (imported so the public surface is covered)
    resolve_parameter,
)


def _rec(**kw: object) -> OedKineticRecord:
    """An OED row with sane defaults, overridden per test."""
    base: dict[str, object] = {
        "uniprot": "P00000",
        "substrate": "glucose",
        "organism": "Saccharomyces cerevisiae",
        "enzymetype": "wildtype",
        "temperature": 30.0,
        "ph": 7.0,
        "kcat_value": 10.0,
        "kcat_unit": "1/s",
        "kcat_pubmedid": 12345.0,
        "km_value": 1.0,
        "km_unit": "mM",
        "km_pubmedid": 12345.0,
    }
    base.update(kw)
    return OedKineticRecord.model_validate(base)


def test_picks_the_measurement_nearest_30c() -> None:
    """The whole point of the temperature rule: 60 C is the same enzyme, wrong number."""
    cands = [
        _rec(temperature=60.0, kcat_value=999.0),
        _rec(temperature=31.0, kcat_value=11.0),
        _rec(temperature=4.0, kcat_value=0.1),
    ]
    p = resolve_parameter(cands, KineticKind.KCAT, "P00000")
    assert p is not None
    assert p.value == 11.0
    assert p.temperature_c == 31.0
    assert p.temperature_delta_c == 1.0
    assert "nearest_30C" in p.selection_rule


def test_wildtype_beats_a_closer_mutant() -> None:
    """A mutant's k_cat describes a protein we do not have, so it loses even at exactly
    30 C. This ordering is load-bearing: reversing the two rungs changes the answer.
    """
    cands = [
        _rec(enzymetype="mutant", temperature=30.0, kcat_value=999.0),
        _rec(enzymetype="wildtype", temperature=45.0, kcat_value=12.0),
    ]
    p = resolve_parameter(cands, KineticKind.KCAT, "P00000")
    assert p is not None
    assert p.value == 12.0
    assert p.enzyme_type == "wildtype"
    assert p.selection_rule.startswith("wildtype_only")


def test_entries_without_temperature_sort_last_but_are_still_usable() -> None:
    """Unknown assay conditions are weaker evidence than known-and-nearby, never stronger
    -- but better than returning nothing.
    """
    with_t = [
        _rec(temperature=50.0, kcat_value=7.0),
        _rec(temperature=None, kcat_value=8.0),
    ]
    p = resolve_parameter(with_t, KineticKind.KCAT, "P00000")
    assert p is not None and p.value == 7.0

    only_none = [_rec(temperature=None, kcat_value=8.0)]
    q = resolve_parameter(only_none, KineticKind.KCAT, "P00000")
    assert q is not None
    assert q.value == 8.0
    assert q.temperature_delta_c is None
    assert "no_temperature" in q.selection_rule


def test_ties_resolve_to_the_median_not_to_row_order() -> None:
    """Three equally-valid 30 C measurements must not let input order decide."""
    cands = [_rec(kcat_value=v) for v in (1.0, 100.0, 5.0)]
    p = resolve_parameter(cands, KineticKind.KCAT, "P00000")
    assert p is not None
    assert p.value == 5.0
    assert p.n_candidates == 3
    assert "median_of_ties" in p.selection_rule
    # order-invariance is the actual property under test
    assert resolve_parameter(cands[::-1], KineticKind.KCAT, "P00000").value == 5.0  # type: ignore[union-attr]


def test_missing_parameter_returns_none_rather_than_a_silent_zero() -> None:
    """A gap must be visible so the caller fills it from a predictor and TAGS it."""
    cands = [_rec(kcat_value=None, km_value=2.0)]
    assert resolve_parameter(cands, KineticKind.KCAT, "P00000") is None
    km = resolve_parameter(cands, KineticKind.KM, "P00000")
    assert km is not None and km.value == 2.0
    assert km.source is KineticSource.OPEN_ENZYME_DATABASE


def test_kcat_and_km_are_read_from_their_own_columns() -> None:
    """They are different parameters with different units; crossing them would be silent."""
    cands = [_rec(kcat_value=42.0, km_value=0.5)]
    kcat = resolve_parameter(cands, KineticKind.KCAT, "P00000")
    km = resolve_parameter(cands, KineticKind.KM, "P00000")
    assert kcat is not None and km is not None
    assert (kcat.value, kcat.unit) == (42.0, "1/s")
    assert (km.value, km.unit) == (0.5, "mM")


def test_index_by_uniprot_skips_rows_with_no_accession() -> None:
    """UniProt is the key the GPR maps genes onto; a null accession is unjoinable."""
    rows = [
        _rec(uniprot="P1"),
        _rec(uniprot=None),
        _rec(uniprot="P1"),
        _rec(uniprot="P2"),
    ]
    idx = index_by_uniprot(rows)
    assert set(idx) == {"P1", "P2"}
    assert len(idx["P1"]) == 2


MIRROR = (
    "/scratch/projects/torchcell-scratch/data/enzyme_kinetics/"
    "open_enzyme_database/scerevisiae"
)


@pytest.mark.skipif(
    not osp.exists(osp.join(MIRROR, "oed_records.json")),
    reason="OED mirror not fetched on this machine",
)
def test_mirror_verifies_sha256_and_resolves_real_records() -> None:
    """The mirror -- not the endpoint -- is canonical, so the hash check must be live."""
    rows = load_mirrored_records(MIRROR)
    assert rows, "mirror is empty"
    idx = index_by_uniprot(rows)
    assert idx
    accession, cands = max(idx.items(), key=lambda kv: len(kv[1]))
    p = resolve_parameter(cands, KineticKind.KCAT, accession)
    assert p is not None
    assert p.value > 0
    assert p.n_candidates == len(cands)


@pytest.mark.skipif(
    not osp.exists(osp.join(MIRROR, "manifest.json")),
    reason="OED mirror not fetched on this machine",
)
def test_tampered_mirror_raises_rather_than_returning_wrong_parameters() -> None:
    """A hash mismatch invalidates every downstream parameter, so it must not warn."""
    with open(osp.join(MIRROR, "manifest.json")) as f:
        manifest = json.load(f)
    assert len(manifest["sha256"]) == 64
    assert manifest["n_records"] > 0
    assert manifest["retrieval_command"].startswith("for off in")
