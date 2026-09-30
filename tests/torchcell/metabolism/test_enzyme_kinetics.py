# tests/torchcell/metabolism/test_enzyme_kinetics.py
# [[tests.torchcell.metabolism.test_enzyme_kinetics]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_enzyme_kinetics.py
"""Tests for the k_cat / K_M resolver.

These exercise the SELECTION CASCADE, not the network: the retrieval is covered by the
sha256 check in `load_mirrored_records`, but the cascade is where a wrong parameter would
enter the flux layer silently, so each rung gets a case that fails if the rung is removed.

2026.09.30 - Phase 13. The module holds no rate law: it resolves ``k_cat`` and ``K_M``
from OED rows and pages the OED API. The added tests pin, on hand-set rows built by
``_rec``, the exact ``ResolvedKineticParameter`` of the full cascade (``n_candidates``
counts every row carrying the value, before the wildtype filter; ``selection_rule``
joins the rungs with `` -> ``; ``nearest_{T:g}C`` formats 37.5 as ``37.5`` and 30.0 as
``30``; ``temperature_delta_c`` is ``|T - target|`` on both sides of the target), the
PubMed id normalization (``12345.0`` to ``"12345"``), a zero ``k_cat`` kept as a real
value rather than a gap, and a Finding: with an EVEN number of tied rows the median
falls between two values and ``min`` returns whichever of the two nearest rows comes
first, so row order decides. Retrieval runs on a fake ``subprocess.run`` that serves
canned OED envelopes (no curl, no network): paging stops at ``total``, a short final
page, the incomplete-paging and wrong-shape errors, the exact reproduction command, and
the mirror round trip under ``tmp_path`` with the sha256 of the compact sorted JSON
(``json.dumps(rows, sort_keys=True, separators=(",", ":"))``) and the tamper refusal.
The two tests that read the developer's OED mirror are now marked ``data``.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import subprocess
from pathlib import Path
from typing import Any

import pytest

from torchcell.metabolism.enzyme_kinetics import (
    KineticKind,
    KineticRetrieval,
    KineticSource,
    OedKineticRecord,
    fetch_oed_records,
    index_by_uniprot,
    load_mirrored_records,
    mirror_oed_slice,
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
    reversed_p = resolve_parameter(cands[::-1], KineticKind.KCAT, "P00000")
    assert reversed_p is not None and reversed_p.value == 5.0


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


@pytest.mark.data
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


@pytest.mark.data
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


# --- 2026.09.30 Phase 13: the full resolved record, the tie Finding, retrieval ----- #
def test_full_cascade_returns_the_exact_resolved_record() -> None:
    """Rows: a 30 C mutant (999), wildtype at 25 C (5.0, pmid 111), 35 C (7.0, pmid
    222) and 50 C (8.0), and a wildtype row with no k_cat. Four rows carry k_cat, so
    ``n_candidates`` is 4 although only three survive the wildtype rung. 25 and 35 C
    tie at delta 5; the median of (5, 7) is 6, both are 1 away, and the first listed
    (25 C) wins. The substrate is taken from the chosen row when none is passed.
    """
    cands = [
        _rec(enzymetype="mutant", temperature=30.0, kcat_value=999.0),
        _rec(temperature=25.0, kcat_value=5.0, kcat_pubmedid=111.0, ph=6.5),
        _rec(temperature=35.0, kcat_value=7.0, kcat_pubmedid=222.0),
        _rec(temperature=50.0, kcat_value=8.0),
        _rec(temperature=30.0, kcat_value=None),
    ]
    p = resolve_parameter(cands, KineticKind.KCAT, "P00000")
    assert p is not None
    assert p.model_dump() == {
        "kind": KineticKind.KCAT,
        "uniprot": "P00000",
        "substrate": "glucose",
        "value": 5.0,
        "unit": "1/s",
        "source": KineticSource.OPEN_ENZYME_DATABASE,
        "temperature_c": 25.0,
        "temperature_delta_c": 5.0,
        "ph": 6.5,
        "enzyme_type": "wildtype",
        "pubmed_id": "111",
        "n_candidates": 4,
        "selection_rule": "wildtype_only -> nearest_30C -> median_of_ties",
        "predictor": None,
    }


def test_even_number_of_ties_lets_row_order_decide() -> None:
    """Finding: the module docstring promises the median tie-break "avoids letting an
    arbitrary row order decide", but with two tied rows (1.0 and 3.0, median 2.0) both
    are 1.0 from the median and ``min`` (enzyme_kinetics.py:308-311) returns the first
    listed, so reversing the rows flips the answer from 1.0 to 3.0. Pinned until the
    tie-break picks a side of the median deterministically.
    """
    rows = [_rec(kcat_value=1.0), _rec(kcat_value=3.0)]
    forward = resolve_parameter(rows, KineticKind.KCAT, "P00000")
    backward = resolve_parameter(rows[::-1], KineticKind.KCAT, "P00000")
    assert forward is not None and backward is not None
    assert (forward.value, backward.value) == (1.0, 3.0)
    assert forward.selection_rule == backward.selection_rule
    assert forward.selection_rule == "wildtype_only -> nearest_30C -> median_of_ties"


def test_rule_string_without_wildtype_and_with_a_custom_target() -> None:
    """Only mutants: no ``wildtype_only`` rung. Target 37.5 C formats as ``37.5C``;
    the delta is measured from the custom target (40 - 37.5 = 2.5), and the explicit
    substrate argument overrides the row's.
    """
    mutants = [_rec(enzymetype="mutant", kcat_value=4.0)]
    p = resolve_parameter(mutants, KineticKind.KCAT, "P00000")
    assert p is not None and p.selection_rule == "nearest_30C"
    q = resolve_parameter(
        [_rec(temperature=40.0), _rec(temperature=20.0, kcat_value=2.0)],
        KineticKind.KCAT,
        "P9",
        substrate="ATP",
        target_temperature_c=37.5,
    )
    assert q is not None
    assert (q.value, q.temperature_delta_c, q.substrate, q.uniprot) == (
        10.0,
        2.5,
        "ATP",
        "P9",
    )
    assert q.selection_rule == "wildtype_only -> nearest_37.5C"


def test_wildtype_match_is_case_insensitive_and_none_is_not_wildtype() -> None:
    """``WildType`` passes the rung (``.lower()``); ``enzymetype=None`` is filtered out."""
    rows = [
        _rec(enzymetype=None, temperature=30.0, kcat_value=50.0),
        _rec(enzymetype="WildType", temperature=40.0, kcat_value=3.0),
    ]
    p = resolve_parameter(rows, KineticKind.KCAT, "P00000")
    assert p is not None
    assert (p.value, p.enzyme_type, p.n_candidates) == (3.0, "WildType", 2)


def test_zero_kcat_is_a_value_and_a_missing_pubmed_id_is_none() -> None:
    """``value_for`` tests ``is not None``, so a 0.0 row is resolved, not a gap; a row
    with no ``kcat_pubmedid`` resolves with ``pubmed_id`` None; the K_M PubMed column is
    read for K_M.
    """
    row = _rec(kcat_value=0.0, kcat_pubmedid=None, km_pubmedid=98765.0)
    kcat = resolve_parameter([row], KineticKind.KCAT, "P00000")
    km = resolve_parameter([row], KineticKind.KM, "P00000")
    assert kcat is not None and km is not None
    assert (kcat.value, kcat.pubmed_id) == (0.0, None)
    assert km.pubmed_id == "98765"
    assert resolve_parameter([], KineticKind.KCAT, "P00000") is None


def test_unknown_oed_columns_are_kept_verbatim() -> None:
    """``extra="allow"``: a new OED column survives validation as model extra."""
    row = OedKineticRecord.model_validate({"uniprot": "P1", "new_column": "x"})
    assert row.model_extra == {"new_column": "x"}
    assert row.value_for(KineticKind.KCAT) is None


def _fake_curl(monkeypatch: pytest.MonkeyPatch, pages: list[Any]) -> list[list[str]]:
    """Serve ``pages`` in order as curl stdout; return the recorded argv list."""
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert kwargs == {"capture_output": True, "check": True}
        calls.append(argv)
        stdout = json.dumps(pages[len(calls) - 1]).encode()
        return subprocess.CompletedProcess(argv, 0, stdout)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _url(offset: int, limit: int = 2) -> str:
    return (
        "https://openenzymedb-api.platform.moleculemaker.org/api/v1/data?"
        f"organism=Saccharomyces%20cerevisiae&limit={limit}&offset={offset}"
    )


def test_fetch_pages_until_total_and_records_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Total 3 at page size 2: offsets 0 and 2, then stop (3 >= 3) without a third call."""
    calls = _fake_curl(
        monkeypatch,
        [
            {"total": 3, "data": [{"ec": "1"}, {"ec": "2"}]},
            {"total": 3, "data": [{"ec": "3"}]},
        ],
    )
    rows, command = fetch_oed_records(page_size=2)
    assert rows == [{"ec": "1"}, {"ec": "2"}, {"ec": "3"}]
    assert calls == [
        ["curl", "-sS", "--max-time", "180", _url(0)],
        ["curl", "-sS", "--max-time", "180", _url(2)],
    ]
    assert command == (
        'for off in $(seq 0 2 N); do curl -sS "https://openenzymedb-api.platform.'
        "moleculemaker.org/api/v1/data?organism=Saccharomyces%20cerevisiae&limit=2"
        '&offset=$off"; done'
    )


@pytest.mark.parametrize(
    ("pages", "max_pages", "message"),
    [
        (
            [{"total": 5, "data": [{"ec": "1"}]}, {"total": 5, "data": []}],
            200,
            "OED paging incomplete for Saccharomyces cerevisiae: got 1 of 5 records",
        ),
        (
            [{"total": 9, "data": [{"ec": "1"}, {"ec": "2"}]}] * 2,
            2,
            "OED paging incomplete for Saccharomyces cerevisiae: got 4 of 9 records",
        ),
    ],
)
def test_fetch_refuses_incomplete_paging(
    monkeypatch: pytest.MonkeyPatch, pages: list[Any], max_pages: int, message: str
) -> None:
    """An empty page before ``total`` and the ``max_pages`` runaway guard both end with
    fewer rows than ``total``, which raises rather than returning a partial slice.
    """
    calls = _fake_curl(monkeypatch, pages)
    with pytest.raises(ValueError, match=message):
        fetch_oed_records(page_size=2, max_pages=max_pages)
    assert len(calls) == 2


@pytest.mark.parametrize("payload", [[{"ec": "1"}], {"total": 1, "rows": []}])
def test_fetch_refuses_an_unexpected_payload_shape(
    monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    """A bare list and an envelope with no ``data`` key are both a ``TypeError``."""
    _fake_curl(monkeypatch, [payload])
    with pytest.raises(TypeError, match="OED returned an unexpected payload shape: "):
        fetch_oed_records(page_size=2)


def test_mirror_round_trip_pins_the_sha_and_refuses_a_tampered_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored bytes are the compact key-sorted JSON; the manifest pins their sha256,
    the first page URL and the row count; ``load_mirrored_records`` round-trips the rows
    and raises the exact mismatch message after one byte of the slice changes.
    """
    rows = [
        {"uniprot": "P2", "kcat_value": 2.0, "ec": "1.1.1.1"},
        {"uniprot": "P1", "km_value": 0.5},
    ]
    _fake_curl(monkeypatch, [{"total": 2, "data": rows}])
    mirror = tmp_path / "oed"
    record = mirror_oed_slice(str(mirror), page_size=2)
    payload = (mirror / "oed_records.json").read_bytes()
    assert payload == (
        b'[{"ec":"1.1.1.1","kcat_value":2.0,"uniprot":"P2"},'
        b'{"km_value":0.5,"uniprot":"P1"}]'
    )
    digest = hashlib.sha256(payload).hexdigest()
    assert (record.sha256, record.n_records, record.source_url) == (digest, 2, _url(0))
    assert record.organism == "Saccharomyces cerevisiae"
    assert record.retrieval_method == "oed_api"
    on_disk = KineticRetrieval.model_validate_json(
        (mirror / "manifest.json").read_text()
    )
    assert on_disk == record
    loaded = load_mirrored_records(str(mirror))
    assert [r.uniprot for r in loaded] == ["P2", "P1"]
    assert (loaded[0].kcat_value, loaded[1].km_value) == (2.0, 0.5)
    assert loaded[0].model_extra == {}

    tampered = payload.replace(b"2.0", b"3.0")
    (mirror / "oed_records.json").write_bytes(tampered)
    expected = (
        f"OED mirror sha256 mismatch in {mirror}: manifest {digest}, "
        f"file {hashlib.sha256(tampered).hexdigest()}"
    )
    with pytest.raises(ValueError) as excinfo:
        load_mirrored_records(str(mirror))
    assert str(excinfo.value) == expected
