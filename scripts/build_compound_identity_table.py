# scripts/build_compound_identity_table.py
# [[scripts.build_compound_identity_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/build_compound_identity_table
"""One-time builder for the pinned compound-identity table (UI-2).

This is the ONLY thing in the pipeline that hits the network. It queries the
PubChem PUG REST API for name->(InChIKey, SMILES, CID) and cid->(InChIKey, SMILES),
rate-limited (<=5 req/s, PubChem policy), deterministic, and cached. On ANY lookup
failure it records the name as ``UNRESOLVED_PUBLIC`` -- it NEVER guesses a structure.
Each row records the exact retrieval endpoint + method + date so the table is
rebuildable from scratch.

Run ONCE by a human from the repo root:

    ~/miniconda3/envs/torchcell/bin/python scripts/build_compound_identity_table.py

It writes ``torchcell/datamodels/compound_identity_table.json`` (deterministic
bytes: sorted records, stable formatting) and prints the sha256 to pin into
``torchcell/datamodels/compound_identity.py::_TABLE_SHA256``. The committed JSON is
the canonical artifact thereafter; the resolver reads it offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
_PROPERTIES = "InChIKey,ConnectivitySMILES"
_RETRIEVAL_METHOD = "pubchem_api"
_MIN_INTERVAL_S = 0.25  # <=5 req/s (well under PubChem's 5 req/s + 400/min limits)

_TABLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "torchcell"
    / "datamodels"
    / "compound_identity_table.json"
)

# --- Seed set: STATIC-FIRST from the statically-enumerable loaders + a
# hand-verified common core. Canonical spellings (the resolver's synonym map folds
# loader aliases like 'H2O2'/'NaCl'/'MMS' onto these). Order-independent: the table
# is sorted before writing. ---
SEED_NAMES: list[str] = [
    # auesukaree2009
    "ethanol",
    "methanol",
    "1-propanol",
    "sodium chloride",
    "hydrogen peroxide",
    # costanzo2021
    "actinomycin D",
    "benomyl",
    "bortezomib",
    "caspofungin",
    "concanamycin A",
    "cycloheximide",
    "fluconazole",
    "geldanamycin",
    "methyl methanesulfonate",
    "monensin",
    "rapamycin",
    "sorbitol",
    "tunicamycin",
    "galactose",
    # mota2024
    "acetic acid",
    "butyric acid",
    "octanoic acid",
    # smith2006 (acetate folds to acetic acid via synonym map)
    "oleic acid",
    "myristic acid",
    # lian2019
    "furfural",
    # hand-verified common core
    "tamoxifen",
    "glucose",
    "glycerol",
]

# wildenhain path: resolve a handful of its released PubChem CIDs BY CID (the CID
# route). Real CIDs sampled from the wildenhain 1159580 datapoint export.
SEED_CIDS: list[int] = [2795643, 2795858, 2796808, 2797866, 2798589]

# Hand-verified ChEBI CURIEs for the common core (v1 is PubChem-first and does NOT
# query ChEBI, so chebi_id is otherwise None -- its own future gap). Only long-stable,
# canonical ChEBI ids we are confident about are added here; never guessed.
_CHEBI: dict[str, str] = {
    "furfural": "CHEBI:30976",
    "acetic acid": "CHEBI:15366",
    "ethanol": "CHEBI:16236",
    "methanol": "CHEBI:17790",
    "hydrogen peroxide": "CHEBI:16240",
    "sodium chloride": "CHEBI:26710",
}


class _RateLimiter:
    def __init__(self, min_interval_s: float) -> None:
        self._min = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self._min:
            time.sleep(self._min - delta)
        self._last = time.monotonic()


def _get_json(url: str, limiter: _RateLimiter) -> dict[str, Any] | None:
    """GET a PubChem PUG-REST JSON payload; None on any failure (never guess)."""
    limiter.wait()
    request = urllib.request.Request(url, headers={"User-Agent": "torchcell-compound-builder"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return data
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError):
        return None


def _first_property(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    props = payload.get("PropertyTable", {}).get("Properties")
    if not props:
        return None
    first: dict[str, Any] = props[0]
    return first


def _name_url(name: str) -> str:
    quoted = urllib.parse.quote(name, safe="")
    return f"{_PUG_BASE}/name/{quoted}/property/{_PROPERTIES}/JSON"


def _cid_url(cid: int) -> str:
    return f"{_PUG_BASE}/cid/{cid}/property/{_PROPERTIES}/JSON"


def _record_from_property(
    name: str, prop: dict[str, Any] | None, source_url: str, today: str
) -> dict[str, Any]:
    inchikey = prop.get("InChIKey") if prop else None
    smiles = (prop.get("ConnectivitySMILES") if prop else None) or (
        prop.get("CanonicalSMILES") if prop else None
    )
    cid = prop.get("CID") if prop else None
    resolved = inchikey is not None
    return {
        "name": name,
        "inchikey": inchikey,
        "pubchem_cid": int(cid) if cid is not None else None,
        "chebi_id": _CHEBI.get(name.strip().lower()),
        "smiles": smiles,
        "source_url": source_url,
        "retrieval_method": _RETRIEVAL_METHOD,
        "retrieved_at": today,
        "resolution_status": "RESOLVED" if resolved else "UNRESOLVED_PUBLIC",
    }


def build(cache_path: Path | None) -> dict[str, Any]:
    """Query PubChem for every seed name + CID, returning the payload + resolve report."""
    limiter = _RateLimiter(_MIN_INTERVAL_S)
    today = date.today().isoformat()
    cache: dict[str, dict[str, Any] | None] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    records: list[dict[str, Any]] = []
    resolved_names: list[str] = []
    failed_names: list[str] = []

    for name in SEED_NAMES:
        url = _name_url(name)
        if url not in cache:
            cache[url] = _first_property(_get_json(url, limiter))
        record = _record_from_property(name, cache[url], url, today)
        records.append(record)
        (resolved_names if record["inchikey"] else failed_names).append(name)

    for cid in SEED_CIDS:
        url = _cid_url(cid)
        if url not in cache:
            cache[url] = _first_property(_get_json(url, limiter))
        record = _record_from_property(f"CID {cid}", cache[url], url, today)
        # ensure the CID row carries the seeded CID even if PubChem echoes it back
        record["pubchem_cid"] = cid
        records.append(record)
        (resolved_names if record["inchikey"] else failed_names).append(f"CID {cid}")

    if cache_path:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    records.sort(key=lambda r: r["name"].lower())
    payload = {"schema_version": 1, "records": records}
    return {
        "payload": payload,
        "resolved": resolved_names,
        "failed": failed_names,
    }


def main() -> None:
    """Build the table, write deterministic JSON bytes, and print the sha256 to pin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="optional response-cache JSON path (re-runs skip re-querying)",
    )
    parser.add_argument("--out", type=Path, default=_TABLE_PATH)
    args = parser.parse_args()

    result = build(args.cache)
    # Deterministic bytes: sorted keys, stable indent, trailing newline.
    text = json.dumps(result["payload"], indent=2, sort_keys=True, ensure_ascii=False)
    args.out.write_text(text + "\n", encoding="utf-8")
    sha = hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()

    print(f"wrote {args.out} ({len(result['payload']['records'])} records)")
    print(f"sha256 = {sha}")
    print(f"RESOLVED ({len(result['resolved'])}): {result['resolved']}")
    print(f"UNRESOLVED ({len(result['failed'])}): {result['failed']}")


if __name__ == "__main__":
    main()
