# torchcell/datamodels/compound_identity_curate.py
# [[torchcell.datamodels.compound_identity_curate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/compound_identity_curate
r"""Reproducible curator for the pinned compound-identity table (UI-2, serve-50).

This is the ONLY thing in the compound-identity subsystem that touches the network.
It reads committed input lists (``compound_identity_inputs/<source>.txt``), queries the
PubChem PUG REST API at <= 5 requests per second, and writes
``compound_identity_table.json`` with deterministic bytes. The resolver
(:mod:`torchcell.datamodels.compound_identity`) then reads that table offline forever.

Run from the repo root::

    PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python \\
        -m torchcell.datamodels.compound_identity_curate \\
        --names torchcell/datamodels/compound_identity_inputs/curated_core.txt \\
        --names torchcell/datamodels/compound_identity_inputs/hillenmeyer2008.txt \\
        ... \\
        --cids torchcell/datamodels/compound_identity_inputs/wildenhain2015_cids.txt \\
        --out torchcell/datamodels/compound_identity_table.json

It prints the sha256 to re-pin into ``compound_identity.py::_TABLE_SHA256``.

Input-file grammar (one entry per line; ``#`` starts a comment, and the first comment
line must name the review the list came from)::

    LABEL                          look LABEL up by name; canonical name = PubChem Title, lowercased
    LABEL | query=<name>           look <name> up instead; LABEL becomes a synonym
    LABEL | cid=<int>              look the CID up directly (the `CID <n>` labels, and the CID lists)
    LABEL | canonical              LABEL itself is the canonical name (a curated row already exists)
    LABEL | chebi=CHEBI:<n>        hand-verified ChEBI CURIE, overriding the PubChem synonym scan
    LABEL | mixture=<why>          a defined mixture: keep CID/ChEBI, leave `inchikey` null
    LABEL | unresolved=<why>       no lookup; a RECOVERABLE gap, with its reason recorded
    LABEL | proprietary=<why>      no lookup; a TERMINAL vendor or catalog code
    LABEL | undefined=<why>        no lookup; a TERMINAL undefined preparation (yeast extract)

Directives combine with ``|``, and a prose reason may itself contain ``|`` (it consumes
the rest of the line). Two labels that resolve to the same PubChem CID collapse into ONE
row carrying both labels as synonyms, which is what makes ``NaCl`` (Hillenmeyer) and
``sodium chloride`` (Nadal-Ribelles) the same compound node. A lookup key claimed by two
compounds is awarded by precedence (curated > PubChem name index > CID only) and the
losing rows are qualified ``<name> (CID <n>)``; a tie leaves the key unassigned, because
a label that names two compounds equally well names neither.

Three rules keep this honest:

- **Never guess.** A PubChem name lookup that 404s yields an ``UNRESOLVED_PUBLIC`` row
  whose ``unresolved_reason`` records the fault, not a near-miss structure.
- **One ChEBI or none.** A ChEBI CURIE is adopted from PubChem's synonym list only when
  that list carries exactly ONE distinct ChEBI id; several (acetic acid carries
  ``CHEBI:15366`` and ``CHEBI:47622``) means the choice is a curation decision, so the
  field stays null unless an input line states it with ``chebi=``.
- **A mixture gets no InChIKey.** Tunicamycin is at least ten homologues (ChEBI:29699);
  its identity is the ChEBI class, and fabricating a single-molecule key for it would be
  a lie about what was in the flask.

HTTP errors are surfaced, never swallowed: the opener passes 4xx bodies through (PubChem
answers a missing name with a JSON ``Fault`` and HTTP 404), while a transport failure
raises and stops the run rather than silently degrading a row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
_PROPERTIES = "Title,InChIKey,ConnectivitySMILES"
_RETRIEVAL_METHOD = "pubchem_api"
_MIN_INTERVAL_S = 0.25  # <= 5 requests/s, PubChem's published ceiling
_MAX_BUSY_RETRIES = 6
_BUSY_BACKOFF_S = 5.0
_CID_BATCH = 100
_SCHEMA_VERSION = 2
_CHEBI_RE = re.compile(r"\bCHEBI:\d+\b")
_DIRECTIVE_SEP = "|"
#: Directives whose value is prose and therefore swallows the rest of the line.
_FREE_TEXT_DIRECTIVES = {
    "mixture": "mixture_reason",
    "unresolved": "unresolved_reason",
    "proprietary": "proprietary_reason",
    "undefined": "undefined_reason",
}


# --------------------------------------------------------------------------- #
# Input grammar
# --------------------------------------------------------------------------- #
class CurationDirective(BaseModel):
    """One parsed input line: a loader label plus how to resolve it."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="the string a loader passes to resolved_compound")
    source: str = Field(description="input-file stem, e.g. 'hillenmeyer2008'")
    query: str | None = Field(
        default=None, description="PubChem name to look up instead of the label"
    )
    cid: int | None = Field(default=None, description="resolve by this PubChem CID")
    canonical: bool = Field(
        default=False, description="the label IS the canonical row name"
    )
    chebi_id: str | None = Field(default=None, description="hand-verified ChEBI CURIE")
    mixture_reason: str | None = Field(
        default=None, description="why no single-molecule InChIKey exists"
    )
    unresolved_reason: str | None = Field(
        default=None, description="why PubChem cannot be asked, a recoverable gap"
    )
    proprietary_reason: str | None = Field(
        default=None, description="why this is a terminal vendor/catalog code"
    )
    undefined_reason: str | None = Field(
        default=None, description="why this preparation has no structure at all"
    )

    @property
    def name_query(self) -> str:
        """The string handed to PubChem's name index."""
        return self.query if self.query is not None else self.label

    @property
    def skip_lookup(self) -> str | None:
        """The terminal reason that makes a PubChem call pointless, if any."""
        return (
            self.unresolved_reason or self.proprietary_reason or self.undefined_reason
        )


def parse_line(line: str, source: str) -> CurationDirective:
    """Parse ``LABEL [| key=value]...`` into a typed directive."""
    parts = [p.strip() for p in line.split(_DIRECTIVE_SEP)]
    fields: dict[str, Any] = {"label": parts[0], "source": source}
    index = 1
    while index < len(parts):
        directive = parts[index]
        index += 1
        if directive == "canonical":
            fields["canonical"] = True
            continue
        key, _, value = directive.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "query":
            fields["query"] = value
        elif key == "cid":
            fields["cid"] = int(value)
        elif key == "chebi":
            fields["chebi_id"] = value
        elif key in _FREE_TEXT_DIRECTIVES:
            # A prose reason may itself contain the separator (a quoted SOM table
            # row), so it consumes the rest of the line verbatim.
            rest = f" {_DIRECTIVE_SEP} ".join([value, *parts[index:]])
            fields[_FREE_TEXT_DIRECTIVES[key]] = rest
            index = len(parts)
        else:
            raise ValueError(f"{source}: unknown directive {directive!r} in {line!r}")
    return CurationDirective.model_validate(fields)


def read_name_list(path: Path) -> list[CurationDirective]:
    """Parse an input list of labels (comments and blank lines skipped)."""
    source = path.stem
    out: list[CurationDirective] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip() if raw.lstrip().startswith("#") else raw
        line = line.strip()
        if not line:
            continue
        out.append(parse_line(line, source))
    return out


def read_cid_list(path: Path) -> list[CurationDirective]:
    """Parse an input list of bare PubChem CIDs (one per line)."""
    source = path.stem
    out: list[CurationDirective] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("#"):
            continue
        line = raw.strip()
        if not line:
            continue
        cid = int(line)
        out.append(CurationDirective(label=f"CID {cid}", source=source, cid=cid))
    return out


# --------------------------------------------------------------------------- #
# PubChem access
# --------------------------------------------------------------------------- #
class _PassThroughErrorProcessor(urllib.request.HTTPErrorProcessor):
    """Return 4xx/5xx responses instead of raising, so a Fault body can be read.

    PubChem answers an unknown name with HTTP 404 and a JSON ``Fault``; that body is
    the EVIDENCE that the name is unresolvable, so it must reach the caller. Transport
    failures still raise and abort the run.
    """

    def http_response(self, request: Any, response: Any) -> Any:
        return response

    https_response = http_response


class PubChemProperty(BaseModel):
    """One row of a PUG REST ``PropertyTable``."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    cid: int = Field(alias="CID")
    title: str | None = Field(default=None, alias="Title")
    inchikey: str | None = Field(default=None, alias="InChIKey")
    smiles: str | None = Field(default=None, alias="ConnectivitySMILES")


class PubChemClient:
    """Rate-limited PUG REST client with an on-disk response cache."""

    def __init__(self, cache_path: Path | None) -> None:
        """Build the client; an existing cache file is loaded so a rerun skips the network."""
        self._opener = urllib.request.build_opener(_PassThroughErrorProcessor)
        self._last = 0.0
        self._cache_path = cache_path
        self._cache: dict[str, Any] = (
            json.loads(cache_path.read_text())
            if cache_path and cache_path.exists()
            else {}
        )

    def _wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - delta)
        self._last = time.monotonic()

    def _request(self, url: str, body: bytes | None = None) -> dict[str, Any]:
        """One PUG REST call, backing off while PubChem's dynamic throttle is engaged.

        Only ``PUGREST.NotFound`` is data (the name has no CID). ``PUGREST.ServerBusy``
        is a throttle signal and is retried after a growing pause; any other fault, or a
        throttle that survives :data:`_MAX_BUSY_RETRIES`, aborts the run. Recording a
        throttled call as "unresolved" would silently curate a compound out of existence.
        """
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "User-Agent": "torchcell-compound-curator",
                **(
                    {"Content-Type": "application/x-www-form-urlencoded"}
                    if body is not None
                    else {}
                ),
            },
        )
        for attempt in range(_MAX_BUSY_RETRIES + 1):
            self._wait()
            with self._opener.open(request, timeout=60) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            code = payload.get("Fault", {}).get("Code")
            if code is None or code == "PUGREST.NotFound":
                return payload
            if code != "PUGREST.ServerBusy":
                raise RuntimeError(
                    f"PubChem fault {code} for {url}: {payload['Fault']}"
                )
            time.sleep(_BUSY_BACKOFF_S * (attempt + 1))
        raise RuntimeError(
            f"PubChem stayed busy after {_MAX_BUSY_RETRIES} retries for {url}"
        )

    def flush(self) -> None:
        """Persist the response cache so a rerun skips the network."""
        if self._cache_path is not None:
            self._cache_path.write_text(json.dumps(self._cache, sort_keys=True))

    def property_by_name(self, name: str) -> PubChemProperty | None:
        """First property row for a name, or None when PubChem reports no CID."""
        url = name_property_url(name)
        if url not in self._cache:
            self._cache[url] = self._request(url)
        return _first_property(self._cache[url])

    def properties_by_cids(self, cids: list[int]) -> dict[int, PubChemProperty]:
        """Batched CID -> property lookup (one POST per :data:`_CID_BATCH` CIDs)."""
        out: dict[int, PubChemProperty] = {}
        pending = [cid for cid in cids if cid_property_url(cid) not in self._cache]
        for start in range(0, len(pending), _CID_BATCH):
            chunk = pending[start : start + _CID_BATCH]
            payload = self._request(
                f"{_PUG_BASE}/cid/property/{_PROPERTIES}/JSON",
                body=urllib.parse.urlencode(
                    {"cid": ",".join(str(c) for c in chunk)}
                ).encode("ascii"),
            )
            rows = payload.get("PropertyTable", {}).get("Properties", [])
            found = {int(r["CID"]): r for r in rows}
            for cid in chunk:
                row = found.get(cid)
                self._cache[cid_property_url(cid)] = (
                    {"PropertyTable": {"Properties": [row]}}
                    if row is not None
                    else payload
                )
        for cid in cids:
            prop = _first_property(self._cache[cid_property_url(cid)])
            if prop is not None and prop.cid == cid:
                out[cid] = prop
        return out

    def chebi_by_cids(self, cids: list[int]) -> dict[int, str | None]:
        """Batched CID -> ChEBI CURIE, adopted only when exactly one id is listed."""
        out: dict[int, str | None] = {}
        pending = [cid for cid in cids if synonyms_url(cid) not in self._cache]
        for start in range(0, len(pending), _CID_BATCH):
            chunk = pending[start : start + _CID_BATCH]
            payload = self._request(
                f"{_PUG_BASE}/cid/synonyms/JSON",
                body=urllib.parse.urlencode(
                    {"cid": ",".join(str(c) for c in chunk)}
                ).encode("ascii"),
            )
            rows = payload.get("InformationList", {}).get("Information", [])
            found = {int(r["CID"]): r.get("Synonym", []) for r in rows}
            for cid in chunk:
                self._cache[synonyms_url(cid)] = found.get(cid, [])
        for cid in cids:
            out[cid] = _single_chebi(self._cache[synonyms_url(cid)])
        return out


def name_property_url(name: str) -> str:
    """The canonical single-name GET endpoint (also the row's recorded ``source_url``)."""
    return f"{_PUG_BASE}/name/{urllib.parse.quote(name, safe='')}/property/{_PROPERTIES}/JSON"


def cid_property_url(cid: int) -> str:
    """The canonical single-CID GET endpoint that reproduces one batched row."""
    return f"{_PUG_BASE}/cid/{cid}/property/{_PROPERTIES}/JSON"


def synonyms_url(cid: int) -> str:
    """The canonical single-CID synonyms endpoint (the ChEBI source)."""
    return f"{_PUG_BASE}/cid/{cid}/synonyms/JSON"


def _first_property(payload: Any) -> PubChemProperty | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("PropertyTable", {}).get("Properties")
    if not rows or rows[0] is None:
        return None
    return PubChemProperty.model_validate(rows[0])


def _single_chebi(synonyms: Any) -> str | None:
    """The ChEBI CURIE when a synonym list carries exactly one; else None."""
    if not isinstance(synonyms, list):
        return None
    hits = sorted({m.group(0) for s in synonyms for m in _CHEBI_RE.finditer(str(s))})
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
class CuratedRow(BaseModel):
    """One assembled table row, before it is serialized to JSON."""

    model_config = ConfigDict(extra="forbid")

    name: str
    inchikey: str | None = None
    pubchem_cid: int | None = None
    chebi_id: str | None = None
    smiles: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    source_url: str | None = None
    retrieval_method: str | None = None
    retrieved_at: str | None = None
    resolution_status: str
    unresolved_reason: str | None = None
    sources: list[str] = Field(default_factory=list)
    precedence: int = Field(
        default=0,
        description="claim strength on a lookup key: 2 curated, 1 PubChem name index, 0 CID only",
    )


def _merge_key(directive: CurationDirective, prop: PubChemProperty | None) -> str:
    """Identity under which two labels collapse into one row."""
    if prop is not None:
        return f"cid:{prop.cid}"
    return f"name:{directive.label.strip().lower()}"


def assemble(
    directives: list[CurationDirective],
    properties: dict[str, PubChemProperty | None],
    chebi: dict[int, str | None],
    today: str,
) -> list[CuratedRow]:
    """Collapse directives + PubChem answers into deduplicated, canonical rows."""
    groups: dict[str, list[tuple[CurationDirective, PubChemProperty | None]]] = {}
    for directive in directives:
        prop = properties.get(directive.label)
        groups.setdefault(_merge_key(directive, prop), []).append((directive, prop))

    rows: list[CuratedRow] = []
    for members in groups.values():
        first_prop = next((p for _, p in members if p is not None), None)
        curated = [d for d, _ in members if d.canonical]
        mixture = next((d.mixture_reason for d, _ in members if d.mixture_reason), None)
        terminal = next(
            (
                (status, reason)
                for d, _ in members
                for status, reason in (
                    ("UNRESOLVED_PUBLIC", d.unresolved_reason),
                    ("PROPRIETARY", d.proprietary_reason),
                    ("UNDEFINED_MIXTURE", d.undefined_reason),
                )
                if reason
            ),
            None,
        )
        hand_chebi = next((d.chebi_id for d, _ in members if d.chebi_id), None)

        claimed = sorted({d.label for d in curated})
        if len(claimed) > 1:
            raise ValueError(
                f"conflicting canonical names for one compound: {claimed} "
                f"(sources {sorted({d.source for d in curated})}); exactly one input "
                "line may claim `canonical` per compound"
            )
        precedence = (
            2 if claimed else (1 if any(d.cid is None for d, _ in members) else 0)
        )
        if claimed:
            name = claimed[0]
        elif first_prop is not None and first_prop.title:
            name = first_prop.title.lower()
        else:
            name = members[0][0].label

        labels = sorted({d.label for d, _ in members} - {name})
        sources = sorted({d.source for d, _ in members})
        cid = first_prop.cid if first_prop is not None else None
        source_url = (
            cid_property_url(cid)
            if cid is not None and any(d.cid is not None for d, _ in members)
            else (
                name_property_url(members[0][0].name_query)
                if first_prop is not None
                else None
            )
        )

        if terminal is not None:
            rows.append(
                CuratedRow(
                    name=name,
                    synonyms=labels,
                    resolution_status=terminal[0],
                    unresolved_reason=terminal[1],
                    retrieved_at=today,
                    sources=sources,
                    precedence=precedence,
                )
            )
            continue

        if first_prop is None:
            rows.append(
                CuratedRow(
                    name=name,
                    chebi_id=hand_chebi,
                    synonyms=labels,
                    source_url=name_property_url(members[0][0].name_query),
                    retrieval_method=_RETRIEVAL_METHOD,
                    retrieved_at=today,
                    resolution_status=(
                        "RESOLVED_MIXTURE"
                        if mixture is not None and hand_chebi is not None
                        else "UNRESOLVED_PUBLIC"
                    ),
                    unresolved_reason=(
                        mixture
                        if mixture is not None
                        else "PubChem name lookup returned no CID (PUGREST.NotFound)"
                    ),
                    sources=sources,
                    precedence=precedence,
                )
            )
            continue

        resolved_chebi = (
            hand_chebi if hand_chebi is not None else chebi.get(first_prop.cid)
        )
        if mixture is not None:
            rows.append(
                CuratedRow(
                    name=name,
                    pubchem_cid=first_prop.cid,
                    chebi_id=resolved_chebi,
                    synonyms=labels,
                    source_url=source_url,
                    retrieval_method=_RETRIEVAL_METHOD,
                    retrieved_at=today,
                    resolution_status="RESOLVED_MIXTURE",
                    unresolved_reason=mixture,
                    sources=sources,
                    precedence=precedence,
                )
            )
            continue

        rows.append(
            CuratedRow(
                name=name,
                inchikey=first_prop.inchikey,
                pubchem_cid=first_prop.cid,
                chebi_id=resolved_chebi,
                smiles=first_prop.smiles,
                synonyms=labels,
                source_url=source_url,
                retrieval_method=_RETRIEVAL_METHOD,
                retrieved_at=today,
                resolution_status=(
                    "RESOLVED"
                    if first_prop.inchikey is not None
                    else "UNRESOLVED_PUBLIC"
                ),
                unresolved_reason=(
                    None
                    if first_prop.inchikey is not None
                    else "PubChem returned a CID with no InChIKey"
                ),
                sources=sources,
                precedence=precedence,
            )
        )

    rows.sort(key=lambda r: (r.name.lower(), r.pubchem_cid or 0))
    return _disambiguate(rows)


def _disambiguate(rows: list[CuratedRow]) -> list[CuratedRow]:
    """Resolve lookup-key collisions by PRECEDENCE, so no key has a silent winner.

    Two PubChem records can share a Title (Cisplatin is titled on more than one CID), and
    one compound's synonym can be another's name. A key is awarded to the single highest-
    precedence row that claims it: a row an input line marked ``canonical`` outranks a row
    resolved through PubChem's NAME index, which outranks one reached only by CID. The
    losers keep the identity and lose the key: a losing name becomes ``<name> (CID <n>)``,
    a losing synonym is dropped. A tie at the top means nobody gets the key, because a
    label that names two compounds equally well names neither.
    """
    keys = [[k.strip().lower() for k in [row.name, *row.synonyms]] for row in rows]
    owners: dict[str, list[int]] = {}
    for index, row_keys in enumerate(keys):
        for key in set(row_keys):
            owners.setdefault(key, []).append(index)
    winner: dict[str, int | None] = {}
    for key, claimants in owners.items():
        best = max(rows[i].precedence for i in claimants)
        top = [i for i in claimants if rows[i].precedence == best]
        winner[key] = top[0] if len(top) == 1 else None

    out: list[CuratedRow] = []
    for index, row in enumerate(rows):
        name_key = row.name.strip().lower()
        keep = [
            s
            for s in row.synonyms
            if winner[s.strip().lower()] == index and s.strip().lower() != name_key
        ]
        if winner[name_key] == index:
            out.append(row.model_copy(update={"synonyms": keep}))
            continue
        if row.pubchem_cid is None:
            raise ValueError(
                f"name collision on {row.name!r} with no CID to qualify it"
            )
        out.append(
            row.model_copy(
                update={"name": f"{row.name} (CID {row.pubchem_cid})", "synonyms": keep}
            )
        )
    out.sort(key=lambda r: (r.name.lower(), r.pubchem_cid or 0))
    return out


def serialize(rows: list[CuratedRow]) -> str:
    """Deterministic table bytes (sorted keys, stable indent, trailing newline)."""
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "records": [r.model_dump(exclude={"sources", "precedence"}) for r in rows],
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def curate(
    name_files: list[Path], cid_files: list[Path], cache: Path | None
) -> tuple[list[CuratedRow], list[CurationDirective]]:
    """Read the input lists, query PubChem, and assemble the rows."""
    directives: list[CurationDirective] = []
    for path in name_files:
        directives.extend(read_name_list(path))
    for path in cid_files:
        directives.extend(read_cid_list(path))

    client = PubChemClient(cache)
    cid_directives = [d for d in directives if d.cid is not None and not d.skip_lookup]
    name_directives = [d for d in directives if d.cid is None and not d.skip_lookup]

    by_cid = client.properties_by_cids(sorted({d.cid for d in cid_directives if d.cid}))
    properties: dict[str, PubChemProperty | None] = {}
    for directive in cid_directives:
        assert directive.cid is not None
        properties[directive.label] = by_cid.get(directive.cid)

    queried: dict[str, PubChemProperty | None] = {}
    for i, directive in enumerate(name_directives, start=1):
        query = directive.name_query
        if query not in queried:
            queried[query] = client.property_by_name(query)
            if i % 50 == 0:
                client.flush()
                print(f"  ... {i}/{len(name_directives)} name lookups", flush=True)
        properties[directive.label] = queried[query]

    all_cids = sorted({p.cid for p in properties.values() if p is not None})
    chebi = client.chebi_by_cids(all_cids)
    client.flush()

    rows = assemble(directives, properties, chebi, date.today().isoformat())
    return rows, directives


def main() -> None:
    """Curate the table from the committed input lists and print the sha256 to pin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--names",
        type=Path,
        action="append",
        default=[],
        help="label input list (repeatable)",
    )
    parser.add_argument(
        "--cids",
        type=Path,
        action="append",
        default=[],
        help="bare-CID input list (repeatable)",
    )
    parser.add_argument("--out", type=Path, required=True, help="table JSON to write")
    parser.add_argument("--cache", type=Path, default=None, help="response-cache JSON")
    args = parser.parse_args()

    rows, directives = curate(args.names, args.cids, args.cache)
    text = serialize(rows)
    args.out.write_text(text, encoding="utf-8")

    print(f"wrote {args.out} ({len(rows)} records)")
    print(f"sha256 = {hashlib.sha256(text.encode('utf-8')).hexdigest()}")

    by_source: dict[str, dict[str, int]] = {}
    status = {
        key.strip().lower(): r.resolution_status
        for r in rows
        for key in [r.name, *r.synonyms]
    }
    for directive in directives:
        bucket = by_source.setdefault(directive.source, {})
        state = status.get(directive.label.strip().lower(), "AMBIGUOUS_LABEL")
        bucket[state] = bucket.get(state, 0) + 1
    for source in sorted(by_source):
        print(f"  {source}: {dict(sorted(by_source[source].items()))}")
    for row in rows:
        if row.resolution_status != "RESOLVED":
            print(f"  [{row.resolution_status}] {row.name}: {row.unresolved_reason}")
    keys = {r.name.strip().lower() for r in rows} | {
        s.strip().lower() for r in rows for s in r.synonyms
    }
    dropped = sorted(
        {d.label for d in directives if d.label.strip().lower() not in keys}
    )
    if dropped:
        print(f"  AMBIGUOUS labels dropped as lookup keys ({len(dropped)}): {dropped}")


if __name__ == "__main__":
    main()
