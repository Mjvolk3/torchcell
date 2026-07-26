# torchcell/metabolism/enzyme_kinetics
# [[torchcell.metabolism.enzyme_kinetics]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/enzyme_kinetics
# Test file: tests/torchcell/metabolism/test_enzyme_kinetics.py

"""Enzyme kinetic parameters (k_cat, K_M) with provenance, from the Open Enzyme Database.

WHY THIS EXISTS, AND WHY NOT ecYeastGEM
---------------------------------------
The flux layer needs two parameters and they are NOT interchangeable:

* ``k_cat`` bounds capacity, ``|v_j| <= k_cat_gj * E_g``. It is the ONLY source of
  magnitude in the model -- 4,129 of Yeast9's 4,131 reaction bounds carry none -- and it
  is what actually shrinks the feasible polytope.
* ``K_M`` enters only through the saturation factor
  ``eta_sat = prod_i c_i / (K_M_ij + c_i)``. On its own it adds a parameter *and* a free
  variable (the concentration), so it is extra SLACK, not extra constraint. What makes it
  necessary is promiscuity: Wu 2026 measured underground reactions as having ~2x higher
  ``K_M`` with INDISTINGUISHABLE ``k_cat``, so a k_cat-only model cannot tell a promiscuous
  edge from a native one and routes flux through it at full native capacity for free.

**GECKO / ecYeastGEM carries k_cat only -- it has no K_M at all.** So the two parameters
have genuinely different sources and this module keeps them separate rather than pretending
one mirror supplies both.

THE 30 C RULE
-------------
Kinetic constants are assay-condition dependent, and the OED ships 105,520 entries with a
linked temperature precisely so that dependence can be respected. Our phenotype data is
generated at 30 C (SGA is 26-30 C; the pigment screens and Mulleder are 30 C), so a
``k_cat`` measured at 60 C in a thermostability study is the wrong number even though it is
the same enzyme. :func:`resolve_parameter` therefore selects the candidate whose assay
temperature is CLOSEST TO 30 C, and records how far it had to reach.

Selection is a strict, documented cascade rather than a heuristic blend, because a silently
averaged parameter cannot be traced back to a measurement:

1. keep candidates that actually carry a value for the requested parameter;
2. prefer ``wildtype`` over mutant -- a mutant's k_cat describes a protein we do not have;
3. minimize ``|T - 30 C|``; entries with no recorded temperature sort last, never first;
4. break remaining ties on the MEDIAN value, which is deterministic and avoids letting an
   arbitrary row order decide.

Every resolved value keeps ``n_candidates`` and ``selection_rule`` so a reviewer can see
how much choice was involved, and the PubMed id of the row it came from.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import statistics
import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field

#: Live OED service. The stored slice + its sha256 is canonical, NOT this URL -- the URL
#: is retrieval metadata. On rebuild we replay ``retrieval_command`` and verify the hash.
OED_API_BASE = "https://openenzymedb-api.platform.moleculemaker.org/api/v1"

#: Assay temperature our phenotype data was generated at. See module docstring.
TARGET_TEMPERATURE_C = 30.0

#: OED spells the organism out; there is no taxid filter on the endpoint.
SCEREVISIAE = "Saccharomyces cerevisiae"


class KineticKind(StrEnum):
    """Which constant is being resolved. They are sourced and used differently."""

    KCAT = "kcat"
    KM = "km"


class KineticSource(StrEnum):
    """Where a value came from. Published before predicted, always, and tagged so that
    a ``published-only`` ablation is possible without re-running retrieval.
    """

    OPEN_ENZYME_DATABASE = "open_enzyme_database"
    PREDICTED = "predicted"


class KineticRetrieval(BaseModel):
    """Provenance for one retrieved OED slice.

    The stored artifact plus its ``sha256`` is canonical. ``source_url`` and
    ``retrieval_command`` are historical retrieval metadata: on rebuild we run the command
    and verify the hash, so upstream drift is DETECTED rather than silently followed.
    """

    source_url: str
    retrieval_method: Literal["oed_api"] = "oed_api"
    retrieval_command: str
    sha256: str
    retrieved_at: datetime
    n_records: int
    organism: str


class OedKineticRecord(BaseModel):
    """One OED row, kept verbatim.

    Stored unmodified so the selection cascade is auditable against the source rather than
    against an already-filtered view. ``model_config`` allows unknown columns because the
    OED is a live community resource and gaining a column must not break ingestion.
    """

    model_config = {"extra": "allow"}

    ec: str | None = None
    substrate: str | None = None
    organism: str | None = None
    uniprot: str | None = None
    enzymetype: str | None = None
    ph: float | None = None
    temperature: float | None = None
    smiles: str | None = None
    kcat_value: float | None = None
    kcat_unit: str | None = None
    kcat_pubmedid: float | None = None
    km_value: float | None = None
    km_unit: str | None = None
    km_pubmedid: float | None = None

    def value_for(self, kind: KineticKind) -> float | None:
        """The value of the requested constant, or None if this row does not carry it."""
        return self.kcat_value if kind is KineticKind.KCAT else self.km_value

    def unit_for(self, kind: KineticKind) -> str | None:
        """The unit string OED recorded alongside the value."""
        return self.kcat_unit if kind is KineticKind.KCAT else self.km_unit

    def pubmed_for(self, kind: KineticKind) -> str | None:
        """PubMed id of the measurement, normalized to a string without the float tail."""
        raw = self.kcat_pubmedid if kind is KineticKind.KCAT else self.km_pubmedid
        return None if raw is None else str(int(raw))


class ResolvedKineticParameter(BaseModel):
    """A single parameter chosen for a (protein, substrate) pair, with its audit trail.

    ``selection_rule`` and ``n_candidates`` are not decoration: a value picked from 40
    candidates spanning 20-70 C deserves different trust than one with a single 30 C
    measurement, and the flux layer's uncertainty handling (quantile-in-the-box) needs to
    know which it is.
    """

    kind: KineticKind
    uniprot: str
    substrate: str | None
    value: float
    unit: str | None
    source: KineticSource
    temperature_c: float | None
    temperature_delta_c: float | None = Field(
        default=None,
        description="|assay T - 30 C|; None when the entry has no recorded T",
    )
    ph: float | None
    enzyme_type: str | None
    pubmed_id: str | None
    n_candidates: int
    selection_rule: str
    predictor: str | None = None


def _sha256(payload: bytes) -> str:
    """Hash of the exact bytes we stored, which is what makes the mirror canonical."""
    return hashlib.sha256(payload).hexdigest()


def _data_url(organism: str, limit: int, offset: int) -> str:
    """Build one page URL. Kept separate so it can be recorded as retrieval provenance."""
    return (
        f"{OED_API_BASE}/data?organism={quote(organism)}&limit={limit}&offset={offset}"
    )


def fetch_oed_records(
    organism: str = SCEREVISIAE, page_size: int = 1000, max_pages: int = 200
) -> tuple[list[dict[str, Any]], str]:
    """Page the OED ``/data`` endpoint for one organism.

    Returns the rows and the exact shell command that reproduces the retrieval. Paging
    stops on a short page, so ``max_pages`` is only a runaway guard -- it is not expected
    to bind, and if it does the caller sees a suspiciously round record count.
    """
    rows: list[dict[str, Any]] = []
    total: int | None = None
    for page in range(max_pages):
        url = _data_url(organism, page_size, page * page_size)
        out = subprocess.run(
            ["curl", "-sS", "--max-time", "180", url], capture_output=True, check=True
        )
        # OED wraps pages as {"total": N, "offset": .., "limit": .., "data": [...]}.
        # `total` is authoritative for paging -- trusting a short page alone would stop
        # early if the service ever caps a page below the requested size.
        envelope = json.loads(out.stdout)
        if not isinstance(envelope, dict) or "data" not in envelope:
            raise TypeError(f"OED returned an unexpected payload shape: {url}")
        batch = envelope["data"]
        total = int(envelope["total"])
        rows.extend(batch)
        if len(rows) >= total or not batch:
            break
    if total is not None and len(rows) != total:
        raise ValueError(
            f"OED paging incomplete for {organism}: got {len(rows)} of {total} records"
        )
    command = (
        f"for off in $(seq 0 {page_size} N); do curl -sS "
        f'"{OED_API_BASE}/data?organism={quote(organism)}&limit={page_size}&offset=$off"; done'
    )
    return rows, command


def mirror_oed_slice(
    mirror_dir: str, organism: str = SCEREVISIAE, page_size: int = 1000
) -> KineticRetrieval:
    """Fetch an organism slice, write it with a sha256-pinned provenance record.

    The mirror -- not the endpoint -- is what the flux layer reads, so a run is
    reproducible after the service changes or goes away.
    """
    rows, command = fetch_oed_records(organism=organism, page_size=page_size)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    os.makedirs(mirror_dir, exist_ok=True)
    with open(osp.join(mirror_dir, "oed_records.json"), "wb") as f:
        f.write(payload)
    record = KineticRetrieval(
        source_url=_data_url(organism, page_size, 0),
        retrieval_command=command,
        sha256=_sha256(payload),
        retrieved_at=datetime.now(UTC),
        n_records=len(rows),
        organism=organism,
    )
    with open(osp.join(mirror_dir, "manifest.json"), "w") as f:
        f.write(record.model_dump_json(indent=2))
    return record


def load_mirrored_records(mirror_dir: str) -> list[OedKineticRecord]:
    """Read the mirror and VERIFY its sha256 before returning anything.

    A mismatch means the stored slice is not the one the manifest describes, which
    invalidates every parameter downstream -- so it raises rather than warning.
    """
    with open(osp.join(mirror_dir, "oed_records.json"), "rb") as f:
        payload = f.read()
    with open(osp.join(mirror_dir, "manifest.json")) as f:
        manifest = KineticRetrieval.model_validate_json(f.read())
    digest = _sha256(payload)
    if digest != manifest.sha256:
        raise ValueError(
            f"OED mirror sha256 mismatch in {mirror_dir}: "
            f"manifest {manifest.sha256}, file {digest}"
        )
    return [OedKineticRecord.model_validate(r) for r in json.loads(payload)]


def resolve_parameter(
    candidates: list[OedKineticRecord],
    kind: KineticKind,
    uniprot: str,
    substrate: str | None = None,
    target_temperature_c: float = TARGET_TEMPERATURE_C,
) -> ResolvedKineticParameter | None:
    """Choose one value from many measurements, by the documented cascade.

    Returns None when no candidate carries a value for ``kind`` -- an explicit gap that the
    caller fills from a predictor and TAGS as predicted, rather than a silent zero.
    """
    with_value = [c for c in candidates if c.value_for(kind) is not None]
    if not with_value:
        return None
    n_candidates = len(with_value)
    rules: list[str] = []

    wildtype = [c for c in with_value if (c.enzymetype or "").lower() == "wildtype"]
    if wildtype:
        with_value = wildtype
        rules.append("wildtype_only")

    # Entries without a temperature sort last: an unknown assay condition is weaker
    # evidence than a known-and-nearby one, but still better than nothing.
    def temp_key(c: OedKineticRecord) -> tuple[int, float]:
        if c.temperature is None:
            return (1, 0.0)
        return (0, abs(c.temperature - target_temperature_c))

    best = min(temp_key(c) for c in with_value)
    with_value = [c for c in with_value if temp_key(c) == best]
    rules.append(
        "no_temperature" if best[0] == 1 else f"nearest_{target_temperature_c:g}C"
    )

    if len(with_value) > 1:
        values = sorted(float(c.value_for(kind)) for c in with_value)  # type: ignore[arg-type]
        median = statistics.median(values)
        chosen = min(
            with_value,
            key=lambda c: abs(float(c.value_for(kind)) - median),  # type: ignore[arg-type]
        )
        rules.append("median_of_ties")
    else:
        chosen = with_value[0]

    temperature = chosen.temperature
    return ResolvedKineticParameter(
        kind=kind,
        uniprot=uniprot,
        substrate=substrate if substrate is not None else chosen.substrate,
        value=float(chosen.value_for(kind)),  # type: ignore[arg-type]
        unit=chosen.unit_for(kind),
        source=KineticSource.OPEN_ENZYME_DATABASE,
        temperature_c=temperature,
        temperature_delta_c=(
            None if temperature is None else abs(temperature - target_temperature_c)
        ),
        ph=chosen.ph,
        enzyme_type=chosen.enzymetype,
        pubmed_id=chosen.pubmed_for(kind),
        n_candidates=n_candidates,
        selection_rule=" -> ".join(rules),
    )


def index_by_uniprot(
    records: list[OedKineticRecord],
) -> dict[str, list[OedKineticRecord]]:
    """Group rows by UniProt accession, the key the GPR maps yeast genes onto."""
    out: dict[str, list[OedKineticRecord]] = {}
    for r in records:
        if r.uniprot:
            out.setdefault(r.uniprot, []).append(r)
    return out
