# experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.aggregation_decomposition_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit
r"""Decompose every served bacterial compendium into its source studies (plan decision 16).

Owner policy, 2026-10-10: a served compendium is a processing reference and a sanity
check, NOT the source of record. Each original study inside it is loaded from its OWN
per-record release when it has one, attributed, and cross-checked against the
compendium's copy of the same samples; the compendium stays served only for samples whose
only artifact is the compendium itself. This script measures what that implies for the
four served compendia (Price 2018 Keio, Borchert 2024 fModules, Lim 2022
putidaPRECISE321, Lamoureux 2023 PRECISE-1K and its Public K-12 arm):

* the source studies, from each compendium's own attribution column, and for the RB-TnSeq
  compendia also from the Fitness Browser's own per-experiment ``pubId``;
* per study: whether it has its own per-record release (and the URL, scriptable or
  walled), whether a mirror holds it, whether a dataset loads it, how many compendium
  samples it accounts for and how many the served loader keeps;
* what the compendium dropped or recomputed relative to the original;
* the L4 cross-check wherever both sides are in hand: the original's values against the
  compendium's copy, with the statistic and the tolerance the compendium's stated
  reprocessing implies.

Inputs are read-only: the raw and literature mirrors, the dev stores' ``preprocess/``
ledgers (never an LMDB), and a Fitness Browser snapshot (``feba.db``, the SQLite database
the browser serves at ``https://fit.genomics.lbl.gov/cgi_data/feba.db``; the HTML pages
answer scripts with a Cloudflare 403, the static ``cgi_data/`` files do not). The
snapshot is not mirrored (10.2 GB); its sha256, size, Last-Modified and ETag are pinned
below and checked before it is opened. Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py \
        --feba-db <path to feba.db> --network
    python experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py \
        --feba-db <path to feba.db>

``--network`` re-probes every release URL (GEO supplementary listings, ENA read runs,
PubMed DOIs, the Fitness Browser and the RB-TnSeq release pages) and rewrites
``aggregation_decomposition_probes.json``; without it the committed probes are carried
forward unchanged. The audit goes to ``aggregation_decomposition_audit.json``, whose
``decompositions`` key is what ``python -m torchcell.candidates gate --decomposition``
folds into each verdict's ``AggregationRecord.sources``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torchcell.candidates.gates import (
    LIBRARY_DIR,
    RAW_DIR,
    DecompositionRecord,
    module_index,
)
from torchcell.candidates.verdict import (
    OwnReleaseKind,
    ReleaseAccess,
    SourceCrossCheck,
    SourceDecomposition,
    SourceRole,
)
from torchcell.datasets.ecoli import lamoureux2023 as p1k
from torchcell.datasets.ecoli import price2018, wetmore2015
from torchcell.datasets.pputida import borchert2023 as b23
from torchcell.datasets.pputida import borchert2024 as b24
from torchcell.datasets.pputida import lim2022
from torchcell.literature.manifest import MANIFEST_FILENAME, Manifest, sha256_file

REPO: Final = Path(__file__).resolve().parents[3]
RESULTS: Final = REPO / "experiments/036-dataset-fixes-before-kg-build/results"
AUDIT_JSON: Final = RESULTS / "aggregation_decomposition_audit.json"
PROBES_JSON: Final = RESULTS / "aggregation_decomposition_probes.json"
TABLES_MD: Final = RESULTS / "aggregation_decomposition_tables.md"
YEAST_JSON: Final = RESULTS / "aggregation_decomposition_yeast_sources.json"
SCRIPT: Final = (
    "experiments/036-dataset-fixes-before-kg-build/scripts/"
    "aggregation_decomposition_audit.py"
)

# --------------------------------------------------------------------------- #
# The Fitness Browser snapshot (retrieved 2026-10-11T04:29:05Z, not mirrored)
# --------------------------------------------------------------------------- #
FEBA_URL: Final = "https://fit.genomics.lbl.gov/cgi_data/feba.db"
FEBA_SHA256: Final = "250d4a03ffc41ea918027ed7adcb2fa9ed3ddc127ad3de29b40602fc6264b4ef"
FEBA_BYTES: Final = 10_205_732_864
FEBA_LAST_MODIFIED: Final = "Fri, 09 Oct 2026 18:08:11 GMT"
FEBA_ETAG: Final = '"2604f2000-65d6c3b3c80ac"'
FEBA_RETRIEVED_AT: Final = "2026-10-11T04:29:05Z"
FEBA_RETRIEVAL_COMMAND: Final = (
    'curl -s -D headers.txt -A "torchcell-audit/1.0" -o feba.db ' + FEBA_URL
)
FEBA_AGAINST: Final = f"Fitness Browser feba.db GeneFitness.fit (sha256 {FEBA_SHA256})"
FB_ORG_PAGE: Final = "https://fit.genomics.lbl.gov/cgi-bin/org.cgi?orgId=Putida"
FB_STRAIN_FITNESS: Final = (
    "https://fit.genomics.lbl.gov/cgi_data/db.StrainFitness.{org}"
)
RBARSEQ_URL: Final = "https://genomics.lbl.gov/supplemental/rbarseq/"
BIGFIT_URL: Final = "https://genomics.lbl.gov/supplemental/bigfit/"

#: Borchert 2024's release rounds to three decimals (Borchert 2023's note measured the
#: same half unit against its full-precision SI), so a copy agrees to 0.0005.
ROUNDING_HALF_UNIT: Final = b23.COMPENDIUM_HALF_ULP
ROUNDING_BASIS: Final = (
    "the compendium release rounds gene fitness to three decimals, so a faithful copy "
    "agrees with the full-precision original to the half unit 0.0005 "
    "(borchert2023.COMPENDIUM_HALF_ULP, measured on Borchert 2023's own SI)"
)
#: Lim 2022 and PRECISE-1K both refuse replicates below 0.95; a source's own processed
#: profile of the SAME library must agree with the compendium's at least that well.
LIM_REPLICATE_R2: Final = 0.95
P1K_REPLICATE_R: Final = 0.95

HTTP_TIMEOUT: Final = 90
HTTP_PAUSE: Final = 0.35
USER_AGENT: Final = (
    "torchcell-aggregation-audit/1.0 (+https://github.com/Mjvolk3/torchcell)"
)

#: A registered class serves a source's values from its own release only when it holds
#: the same samples. Measured for one source: Borchert 2023's 42 arms equal 42 compendium
#: samples to 0.0005 (borchert2023.match_arms, re-run below).
OWN_DATASET_FOR_SAMPLES: Final[dict[str, str]] = {
    b23.CITATION_KEY: "RbTnseqBorchert2023Dataset"
}

#: Branch loading Wetmore 2015 from its own release while this audit ran (not on main).
WETMORE_OWN_RELEASE_BRANCH: Final = "feat/ecoli-wetmore2015-own-release"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Probe(_Frozen):
    """One HTTP probe of a release URL, as measured."""

    url: str
    method: Literal["GET", "HEAD"]
    status: int | None = Field(description="None when the request raised")
    content_length: int | None
    probed_at: str
    error: str | None = None
    names: tuple[str, ...] = Field(
        default=(), description="file names a directory listing or filelist.txt shows"
    )
    runs: int | None = Field(default=None, description="ENA read runs listed")
    doi: str | None = Field(default=None, description="PubMed's DOI for a PMID probe")
    attempts: int = Field(
        default=1,
        ge=1,
        description="requests made; above 1 only when --reprobe-failed re-asked after "
        "a server error (5xx) or a raised request",
    )


class Probes(_Frozen):
    """Every probe, keyed by URL, plus when the set was taken."""

    probed_at: str
    probes: dict[str, Probe]


class DataAvailability(_Frozen):
    """A sentence of a mirrored paper that names where its data are."""

    citation_key: str
    source_uri: str
    sha256: str
    sentences: tuple[str, ...]


class SourceDetail(_Frozen):
    """Measured detail behind one decomposition entry (not part of the verdict)."""

    source_label: str
    dataset_class: str
    compendium_attribution: dict[str, int] = Field(
        description="the compendium's own attribution value -> samples"
    )
    second_attribution: dict[str, int] = Field(
        default_factory=dict,
        description="an independent attribution (Fitness Browser pubId) -> samples",
    )
    accessions: tuple[str, ...] = ()
    probe_urls: tuple[str, ...] = ()
    data_availability: DataAvailability | None = None
    registered_classes_for_key: tuple[str, ...] = ()


class CompendiumAudit(_Frozen):
    """One served compendium: its sources, counts, and what it dropped or recomputed."""

    citation_key: str
    dataset_classes: tuple[str, ...]
    attribution_column: str
    n_compendium_samples: int
    n_served_samples: int
    counts: dict[str, int]
    dropped_or_recomputed: tuple[str, ...]
    details: tuple[SourceDetail, ...]
    not_decomposed: tuple[str, ...] = ()


class AuditResult(_Frozen):
    """The results file: the CLI contract plus the audit's own detail."""

    script: str
    measured_at: str
    feba: dict[str, Any]
    invalid_manifests: tuple[str, ...]
    decompositions: tuple[DecompositionRecord, ...]
    audit: tuple[CompendiumAudit, ...]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def data_root() -> Path:
    """``DATA_ROOT`` from the repo ``.env``."""
    load_dotenv(REPO / ".env")
    root = os.environ.get("DATA_ROOT")
    if not root:
        raise SystemExit("DATA_ROOT is not set")
    return Path(root)


def now() -> str:
    """An ISO UTC timestamp."""
    return datetime.now(UTC).isoformat()


def norm_doi(text: str) -> str:
    """A DOI without its resolver prefix, lower-cased (DOIs are case-insensitive)."""
    doi = text.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
    if doi.lower().startswith("doi.org/"):
        doi = doi[len("doi.org/") :]
    return doi.lower()


def norm_title(text: str) -> str:
    """A title compared case- and punctuation-insensitively."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class MirrorIndex(_Frozen):
    """DOI and title of every manifest in the literature and raw mirrors."""

    by_doi: dict[str, str]
    by_title: dict[str, str]
    trees: dict[str, tuple[str, ...]]
    invalid_manifests: tuple[str, ...] = Field(
        description="manifests that do not validate as the current Manifest model; "
        "their citation_key, doi and title are still read"
    )


def mirror_index(root: Path) -> MirrorIndex:
    """Read every key's ``manifest.json`` of both mirrors (a key may sit in both).

    Only ``citation_key``, ``doi`` and ``title`` are read, so a manifest written under
    an older model still indexes; it is listed in ``invalid_manifests``.
    """
    by_doi: dict[str, str] = {}
    by_title: dict[str, str] = {}
    trees: dict[str, list[str]] = {}
    invalid: list[str] = []
    for tree in (LIBRARY_DIR, RAW_DIR):
        for path in sorted((root / tree).glob(f"*/{MANIFEST_FILENAME}")):
            if path.parent.name.startswith("_"):
                continue  # service directories (_bib, _sync_reports), not keys
            text = path.read_text()
            data = json.loads(text)
            try:
                Manifest.model_validate_json(text)
            except ValidationError:
                invalid.append(f"{tree}/{path.parent.name}")
            key = str(data["citation_key"])
            trees.setdefault(key, []).append(tree)
            if data.get("doi"):
                by_doi.setdefault(norm_doi(str(data["doi"])), key)
            if data.get("title"):
                by_title.setdefault(norm_title(str(data["title"])), key)
    return MirrorIndex(
        by_doi=by_doi,
        by_title=by_title,
        trees={k: tuple(v) for k, v in trees.items()},
        invalid_manifests=tuple(invalid),
    )


def registered_classes() -> dict[str, tuple[str, ...]]:
    """Registered dataset classes by their module's ``CITATION_KEY`` (ast, no import)."""
    out: dict[str, list[str]] = {}
    pattern = re.compile(r"@register_dataset\s*\nclass (\w+)\(")
    for module in module_index():
        if module.citation_key is None:
            continue
        for name in pattern.findall(module.text):
            out.setdefault(module.citation_key, []).append(name)
    return {k: tuple(sorted(v)) for k, v in out.items()}


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_AVAILABILITY = re.compile(
    r"Fitness Browser|fit\.genomics|Gene Expression Omnibus|\bGEO\b|\bGSE\d+|"
    r"Sequence Read Archive|\bSRA\b|\bPRJ[A-Z]{2}\d+|ArrayExpress|figshare|Zenodo|"
    r"github\.com|deposited|accession",
    re.IGNORECASE,
)


def data_availability(root: Path, key: str) -> DataAvailability | None:
    """Up to three sentences of the mirrored paper text that say where the data are."""
    for tree in (LIBRARY_DIR, RAW_DIR):
        base = root / tree / key
        candidates = [base / "paper.md", *sorted((base / "paper").glob("*.txt"))]
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits: list[str] = []
            for line in text.splitlines():
                for sentence in _SENTENCE.split(line):
                    if _AVAILABILITY.search(sentence) and sentence not in hits:
                        hits.append(sentence.strip())
            return DataAvailability(
                citation_key=key,
                source_uri=str(path.relative_to(root / tree / key)),
                sha256=sha256_file(path),
                sentences=tuple(hits[:3]),
            )
    return None


# --------------------------------------------------------------------------- #
# Network probes (only under --network)
# --------------------------------------------------------------------------- #
def geo_suppl_url(gse: str) -> str:
    """GEO's supplementary-file directory of a series."""
    stem = gse[:-3] + "nnn" if len(gse) > 6 else "GSEnnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stem}/{gse}/suppl/"


def ena_runs_url(bioproject: str) -> str:
    """ENA's read-run report of a BioProject (one line per run)."""
    return (
        "https://www.ebi.ac.uk/ena/portal/api/filereport?accession="
        f"{bioproject}&result=read_run&fields=run_accession&format=tsv"
    )


def pubmed_url(pmid: str) -> str:
    """E-utilities esummary of one PMID."""
    return (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id="
        f"{pmid}&retmode=json"
    )


_HREF = re.compile(r'href="([^"?/][^"]*)"')


def probe(session: requests.Session, url: str, method: Literal["GET", "HEAD"]) -> Probe:
    """One request; a raised error is recorded, never retried silently."""
    time.sleep(HTTP_PAUSE)
    stamp = now()
    try:
        response = session.request(method, url, timeout=HTTP_TIMEOUT, stream=True)
    except requests.RequestException as error:
        return Probe(
            url=url,
            method=method,
            status=None,
            content_length=None,
            probed_at=stamp,
            error=type(error).__name__,
        )
    length = response.headers.get("Content-Length")
    names: tuple[str, ...] = ()
    runs: int | None = None
    doi: str | None = None
    listing = any(
        marker in url
        for marker in ("ftp.ncbi.nlm.nih.gov/geo", "ebi.ac.uk/ena", "eutils.ncbi.nlm")
    )
    if method == "GET" and response.status_code == 200 and listing:
        body = response.text
        if "ftp.ncbi.nlm.nih.gov/geo" in url and not url.endswith("filelist.txt"):
            names = tuple(n for n in _HREF.findall(body) if not n.startswith("http"))
        elif url.endswith("filelist.txt"):
            names = tuple(
                line.split("\t")[1]
                for line in body.splitlines()[1:]
                if len(line.split("\t")) > 1
            )
        elif "ebi.ac.uk/ena" in url:
            runs = max(len(body.strip().splitlines()) - 1, 0)
        elif "eutils.ncbi.nlm.nih.gov" in url:
            result = json.loads(body)["result"]
            uid = result["uids"][0]
            ids = result[uid].get("articleids", [])
            dois = [a["value"] for a in ids if a["idtype"] == "doi"]
            doi = dois[0].lower() if dois else None
    response.close()
    return Probe(
        url=url,
        method=method,
        status=response.status_code,
        content_length=int(length) if length is not None else None,
        probed_at=stamp,
        names=names,
        runs=runs,
        doi=doi,
    )


def transient(record: Probe) -> bool:
    """A server error or a raised request: worth asking again, never read as an answer."""
    return record.status is None or record.status >= 500


def reprobe_failed(probes: Probes, *, attempts: int = 3, pause: float = 5.0) -> Probes:
    """Re-ask every transient probe up to ``attempts`` times in all; record the count."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    out = dict(probes.probes)
    for url, record in probes.probes.items():
        tries = record.attempts
        current = record
        while transient(current) and tries < attempts:
            time.sleep(pause)
            current = probe(session, url, record.method)
            tries += 1
        out[url] = current.model_copy(update={"attempts": tries})
    return Probes(probed_at=probes.probed_at, probes=out)


def run_probes(plan: Sequence[tuple[str, Literal["GET", "HEAD"]]]) -> Probes:
    """Probe every planned URL once, then each GEO RAW.tar's filelist.txt."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    out: dict[str, Probe] = {}
    for url, method in dict.fromkeys(plan):
        out[url] = probe(session, url, method)
    for url, record in list(out.items()):
        if "ftp.ncbi.nlm.nih.gov/geo" in url and "filelist.txt" in record.names:
            listing = url + "filelist.txt"
            out[listing] = probe(session, listing, "GET")
    return Probes(probed_at=now(), probes=out)


# --------------------------------------------------------------------------- #
# Release classification from probes
# --------------------------------------------------------------------------- #
_BROWSER_ONLY = re.compile(
    r"\.(bw|bigwig|wig|bed|bedgraph|bam|sam|cel|gff3?|gtf|fa|fasta|fna|peaks?|"
    r"narrowpeak|broadpeak|tdf|png|pdf)(\.gz)?$",
    re.IGNORECASE,
)


def processed_names(names: Iterable[str]) -> tuple[str, ...]:
    """Supplementary names that read as per-sample value tables (by name only)."""
    return tuple(
        n
        for n in names
        if n not in ("filelist.txt",)
        and not n.endswith("_RAW.tar")
        and not _BROWSER_ONLY.search(n)
    )


class ReleaseCall(_Frozen):
    """What a source's own release is, read off its probes."""

    kind: OwnReleaseKind
    url: str | None
    access: ReleaseAccess | None
    urls: tuple[str, ...]
    reason: str


def sequencing_release(
    geo: Sequence[str], bioprojects: Sequence[str], probes: Mapping[str, Probe]
) -> ReleaseCall:
    """GEO processed tables first, then GEO or ENA raw reads, else none.

    A GEO series counts as releasing per-record values when its supplementary listing
    (or its RAW.tar's filelist.txt) names a file that is not a browser track, an image or
    a sequence file; the classification is by file name, the files are not opened.
    """
    urls: list[str] = []
    processed: list[str] = []
    listed_ok = False
    for gse in geo:
        url = geo_suppl_url(gse)
        urls.append(url)
        record = probes[url]
        if transient(record):
            raise SystemExit(f"{url}: transient probe; run --reprobe-failed")
        if record.status == 200:
            listed_ok = True
            names = list(record.names)
            inner = probes.get(url + "filelist.txt")
            if inner is not None:
                names.extend(inner.names)
            processed.extend(processed_names(names))
    if processed:
        return ReleaseCall(
            kind="per_record_values",
            url=urls[0],
            access="scriptable",
            urls=tuple(urls),
            reason=f"GEO supplementary files {sorted(set(processed))[:4]}",
        )
    runs = 0
    ena_ok = False
    for project in bioprojects:
        url = ena_runs_url(project)
        urls.append(url)
        record = probes[url]
        if transient(record):
            raise SystemExit(f"{url}: transient probe; run --reprobe-failed")
        if record.status == 200 and record.runs:
            ena_ok = True
            runs += record.runs
    if ena_ok:
        return ReleaseCall(
            kind="raw_reads_only",
            url=next(u for u in urls if "ebi.ac.uk" in u),
            access="scriptable",
            urls=tuple(urls),
            reason=f"{runs} ENA read runs; no processed GEO table",
        )
    if listed_ok:
        return ReleaseCall(
            kind="raw_reads_only",
            url=urls[0],
            access="scriptable",
            urls=tuple(urls),
            reason="GEO series with no processed supplementary table",
        )
    return ReleaseCall(
        kind="none",
        url=None,
        access=None,
        urls=tuple(urls),
        reason="no accession in the release resolves to reads or values",
    )


# --------------------------------------------------------------------------- #
# The Fitness Browser
# --------------------------------------------------------------------------- #
class Feba:
    """Read-only access to a pinned ``feba.db``."""

    def __init__(self, path: Path, *, verify: bool = True) -> None:
        """Open ``path`` immutable and read-only, after checking it is the pinned file."""
        if verify:
            if path.stat().st_size != FEBA_BYTES:
                raise SystemExit(
                    f"{path}: {path.stat().st_size} bytes, pinned {FEBA_BYTES}"
                )
            digest = sha256_file(path)
            if digest != FEBA_SHA256:
                raise SystemExit(f"{path}: sha256 {digest}, pinned {FEBA_SHA256}")
        self.connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)

    def experiments(self, org: str) -> dict[str, dict[str, Any]]:
        """Every experiment of an organism, by name, as a column dict."""
        cursor = self.connection.execute(
            "select * from Experiment where orgId = ?", (org,)
        )
        columns = [c[0] for c in cursor.description]
        return {row[1]: dict(zip(columns, row, strict=True)) for row in cursor}

    def publications(self) -> dict[str, tuple[str, str]]:
        """Browser publications: ``pubId`` -> (title, URL)."""
        return {
            row[0]: (row[1], row[2].strip())
            for row in self.connection.execute(
                "select pubId, title, URL from Publication"
            )
        }

    def fitness(self, org: str, exp: str) -> dict[str, float]:
        """Gene fitness of one experiment, by ``locusId``."""
        return dict(
            self.connection.execute(
                "select locusId, fit from GeneFitness where orgId = ? and expName = ?",
                (org, exp),
            ).fetchall()
        )

    def media(self, name: str) -> int:
        """Components the browser lists for a medium."""
        row = self.connection.execute(
            "select count(*) from MediaComponents where media = ?", (name,)
        ).fetchone()
        return int(row[0])


def fb_study(
    pub_id: str, publications: Mapping[str, tuple[str, str]], mirrors: MirrorIndex
) -> tuple[str | None, str | None]:
    """(citation key, DOI) of a browser publication: title match first, then its URL.

    A browser publication often links a preprint; the mirror holds the journal version,
    which shares the title. A title match takes the mirrored DOI.
    """
    title, url = publications[pub_id]
    key = mirrors.by_title.get(norm_title(title))
    doi_match = re.search(r"10\.\d{4,9}/[^\s]+", url.replace("%2F", "/"))
    url_doi = norm_doi(doi_match.group(0)).rstrip("/") if doi_match else None
    if url_doi and url_doi.endswith("v1"):
        url_doi = url_doi[: -len("v1")]
    if key is None and url_doi is not None:
        key = mirrors.by_doi.get(url_doi)
    if key is not None:
        doi = next((d for d, k in mirrors.by_doi.items() if k == key), url_doi)
        return key, doi
    return None, url_doi


def fitness_cross_check(
    feba: Feba,
    org: str,
    samples: Sequence[str],
    compendium: Mapping[str, Mapping[str, float]],
    *,
    tolerance: float | None,
    basis: str,
) -> SourceCrossCheck:
    """Largest gene difference between the browser and the compendium over ``samples``."""
    worst = 0.0
    values = 0
    for sample in samples:
        fb = feba.fitness(org, sample)
        mine = compendium[sample]
        shared = [g for g in mine if g in fb]
        if not shared:
            raise ValueError(f"{org}:{sample}: no gene shared with the browser")
        diff = max(abs(fb[g] - mine[g]) for g in shared)
        worst = max(worst, diff)
        values += len(shared)
    return SourceCrossCheck(
        statistic="max_abs_diff",
        value=float(worst),
        tolerance=tolerance,
        tolerance_basis=basis,
        n_samples=len(samples),
        n_values=values,
        against=FEBA_AGAINST,
    )


# --------------------------------------------------------------------------- #
# Price 2018 (E. coli Keio arm)
# --------------------------------------------------------------------------- #
PRICE_CLASS: Final = "RbTnseqPrice2018EcoliDataset"


def price_store(root: Path) -> Path:
    """The dev store's preprocess dir (read-only)."""
    return root / "data/torchcell/rbtnseq_price2018_ecoli/preprocess"


def audit_price(
    root: Path, feba: Feba, mirrors: MirrorIndex, probes: Mapping[str, Probe]
) -> tuple[DecompositionRecord, CompendiumAudit]:
    """Wetmore 2015 and Price 2018 itself, against the browser and Wetmore's release."""
    samples = json.loads((price_store(root) / "samples.json").read_text())
    fb = feba.experiments("Keio")
    by_source: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by_source.setdefault(s["source"], []).append(s)
    second: dict[str, Counter[str]] = {
        k: Counter(
            (fb[s["name"]]["pubId"] or "no-pubId")
            if s["name"] in fb
            else "not-in-browser"
            for s in v
        )
        for k, v in by_source.items()
    }
    comparison = wetmore2015.release_comparison()
    wetmore_check = SourceCrossCheck(
        statistic="min_pearson_r",
        value=comparison.min_pearson_r,
        tolerance=None,
        tolerance_basis="Price 2018 re-ran the RB-TnSeq statistics (versions 1.0.3, "
        "1.1.0 or 1.1.1) over Wetmore 2015's 1.0.0 and states no expected agreement, "
        "so the measured minimum is a pin; identity rests on the shared sample name and "
        "metadata (wetmore2015.build_subsumption refuses a mismatch)",
        n_samples=len(comparison.agreements),
        n_values=sum(a.n_genes for a in comparison.agreements),
        against="Wetmore 2015 release fit_logratios_good.tab, statistics 1.0.0 "
        f"(sha256 {wetmore2015.manifest_sha256(wetmore2015.load_manifest(), wetmore2015.RELEASE_FITNESS)})",
    )
    # The compendium release against the browser, value for value.
    release = pd.read_csv(wetmore2015.raw_path(wetmore2015.SUPERSET_FITNESS), sep="\t")
    release["locusId"] = release["locusId"].astype(str)
    columns = wetmore2015._fitness_columns(release, "superset")
    table = release.set_index("locusId")
    compendium = {
        name: {str(g): float(v) for g, v in table[col].items()}
        for name, col in columns.items()
    }
    in_fb = sorted(n for n in compendium if n in fb)
    price_vs_fb = fitness_cross_check(
        feba,
        "Keio",
        in_fb,
        compendium,
        tolerance=None,
        basis="the browser is the release Price 2018's Data availability names, so "
        "identity is expected; the value is a pin",
    )
    registered = registered_classes()
    entries: list[SourceDecomposition] = []
    details: list[SourceDetail] = []
    for source, members in sorted(by_source.items()):
        served = sum(1 for s in members if s["drop_rule"] is None)
        if source == wetmore2015.CITATION_KEY:
            entry = SourceDecomposition(
                source_label=f"doi:{norm_doi(wetmore2015.PAPER_DOI)}",
                role="original_study",
                citation_key=source,
                doi=norm_doi(wetmore2015.PAPER_DOI),
                dataset_class=PRICE_CLASS,
                n_samples_in_compendium=len(members),
                n_samples_served=served,
                own_release=True,
                own_release_kind="per_record_values",
                own_release_url=RBARSEQ_URL,
                own_release_access=access_from_manifest(
                    wetmore2015.load_manifest(), wetmore2015.RELEASE_FITNESS
                ),
                mirrored=True,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=wetmore_check,
                note="statistics 1.0.0 per-sample fitness, already in the raw mirror; "
                f"being loaded on {WETMORE_OWN_RELEASE_BRANCH} (not on main when "
                "measured). The browser attributes these samples "
                f"{dict(second[source])}",
            )
        else:
            entry = SourceDecomposition(
                source_label=f"doi:{norm_doi(price2018.PAPER_DOI)}",
                role="compendium_paper",
                citation_key=source,
                doi=norm_doi(price2018.PAPER_DOI),
                dataset_class=PRICE_CLASS,
                n_samples_in_compendium=len(members),
                n_samples_served=served,
                own_release=False,
                own_release_kind="none",
                own_release_url=None,
                own_release_access=None,
                mirrored=True,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=None,
                note="first described in Price 2018; the Fitness Browser serves the "
                f"compendium release's values ({price_vs_fb.statistic} "
                f"{price_vs_fb.value:.3g} over {price_vs_fb.n_samples} samples)",
            )
        entries.append(entry)
        details.append(
            SourceDetail(
                source_label=entry.source_label,
                dataset_class=PRICE_CLASS,
                compendium_attribution={source: len(members)},
                second_attribution=dict(second[source]),
                probe_urls=(RBARSEQ_URL,) if entry.own_release else (),
                data_availability=data_availability(root, source),
                registered_classes_for_key=registered.get(source, ()),
            )
        )
    sub = wetmore2015.subsumption_record()
    dropped = json.loads((price_store(root) / "dropped_records.json").read_text())
    names = {s["name"] for s in samples}
    extra = Counter((e["pubId"] or "no-pubId") for n, e in fb.items() if n not in names)
    findings = (
        "statistics recomputed: the compendium re-ran the RB-TnSeq code (1.0.3) over "
        "Wetmore 2015's samples scored at 1.0.0; per-sample Pearson r min "
        f"{comparison.min_pearson_r:.4f}, median {comparison.median_pearson_r:.4f}, "
        f"largest single-gene difference {max(a.max_abs_difference for a in comparison.agreements):.3f} "
        f"over {len(comparison.agreements)} shared samples",
        f"genes: Wetmore 2015 values {comparison.n_source_genes} genes, the compendium "
        f"{comparison.n_superset_genes}; every Wetmore gene is in the compendium",
        f"samples Wetmore scored successful and the compendium dropped: {list(sub.not_covered)}; "
        f"failed there, carried here: {[f.name for f in sub.verdict_flips if not f.source.successful]}",
        f"withdrawn by the authors' 2021 note and still in the compendium release: "
        f"{list(sub.disregarded)}; the browser no longer carries them "
        f"({sum(1 for n in sub.disregarded if n not in fb)} of {len(sub.disregarded)} absent)",
        f"the browser's values equal the compendium release: max abs diff "
        f"{price_vs_fb.value:.3g} over {price_vs_fb.n_samples} samples x 3,789 genes",
        f"the browser holds {sum(extra.values())} more Keio experiments than Table S5 "
        f"({dict(extra)}); they are not in the served compendium",
        f"loader drops by source: kept {dropped['kept_samples_by_source']} of "
        f"{dropped['samples_by_source']}",
    )
    record = DecompositionRecord(
        citation_key=price2018.CITATION_KEY, sources=tuple(entries)
    )
    return record, CompendiumAudit(
        citation_key=price2018.CITATION_KEY,
        dataset_classes=(PRICE_CLASS,),
        attribution_column="Publication per sample (price2018.SampleSource); second "
        "column: Fitness Browser Experiment.pubId",
        n_compendium_samples=len(samples),
        n_served_samples=sum(1 for s in samples if s["drop_rule"] is None),
        counts=study_counts(entries),
        dropped_or_recomputed=findings,
        details=tuple(details),
        not_decomposed=(
            "GeneEssentialityPrice2018EcoliDataset: Price 2018's own call over the "
            "insertion maps, not an aggregation of other studies' values",
        ),
    )


def access_of(record: Probe) -> ReleaseAccess:
    """Scriptable when the recorded probe returned 200, else walled."""
    return "scriptable" if record.status == 200 else "walled"


def access_from_manifest(manifest: Manifest, relpath: str) -> ReleaseAccess:
    """A deposited file's recorded retrieval: walled only for a manual-browser recipe.

    For a file already in the raw mirror the retrieval that put it there is the
    measurement; a bare probe with this script's User-Agent can be refused by a host
    the recorded retriever reaches.
    """
    record = next(f for f in manifest.files if f.path == relpath)
    if record.retrieval is None:
        raise ValueError(f"{manifest.citation_key}/{relpath}: no retrieval record")
    return (
        "walled" if record.retrieval.method.value == "manual_browser" else "scriptable"
    )


# --------------------------------------------------------------------------- #
# Borchert 2024 (P. putida fModules)
# --------------------------------------------------------------------------- #
BORCHERT_CLASS: Final = "RbTnseqBorchert2024Dataset"
BORCHERT_RELEASE_REL: Final = f"{b24.RAW_DIR_REL}/{b24.DATA_RELPATH}"
BORCHERT2023_RELEASE_REL: Final = f"{b23.RAW_DIR_REL}/{b23.DATA_RELPATH}"


def audit_borchert(
    root: Path, feba: Feba, mirrors: MirrorIndex, probes: Mapping[str, Probe]
) -> tuple[DecompositionRecord, CompendiumAudit]:
    """Sources by the browser's pubId, else the compendium's own attribution."""
    release_path = root / BORCHERT_RELEASE_REL
    if sha256_file(release_path) != b24.DATA_SHA256:
        raise SystemExit(
            f"{release_path}: sha256 differs from borchert2024.DATA_SHA256"
        )
    release = b24.read_release(release_path)
    fb = feba.experiments("Putida")
    publications = feba.publications()
    gene_index = {g: i for i, g in enumerate(release.genes)}
    compendium = {
        s.exp_name: {g: float(release.fitness[i, j]) for g, i in gene_index.items()}
        for j, s in enumerate(release.samples)
    }
    groups: dict[str, list[b24.SampleMetadata]] = {}
    own_attr: dict[str, Counter[str]] = {}
    fb_attr: dict[str, Counter[str]] = {}
    for s in release.samples:
        attribution = b24.attribute_sample(s).study
        if s.exp_name in fb:
            pub = fb[s.exp_name]["pubId"] or ""
            label = f"fitness-browser:{pub}" if pub else "fitness-browser:no-pubId"
        else:
            label = f"compendium:{attribution}"
        groups.setdefault(label, []).append(s)
        own_attr.setdefault(label, Counter())[attribution] += 1
        fb_attr.setdefault(label, Counter())[
            (fb[s.exp_name]["pubId"] or "no-pubId")
            if s.exp_name in fb
            else "not-in-browser"
        ] += 1
    # Borchert 2023's own SI against the compendium (re-measured, not recalled).
    comparisons = b23.read_comparisons(root / BORCHERT2023_RELEASE_REL)
    arms = b23.match_arms(comparisons, release)
    b23_check = SourceCrossCheck(
        statistic="max_abs_diff",
        value=max(a.max_abs_diff for a in arms),
        tolerance=ROUNDING_HALF_UNIT,
        tolerance_basis=ROUNDING_BASIS,
        n_samples=len(arms),
        n_values=sum(a.n_shared_loci for a in arms),
        against=f"Borchert 2023 SI {b23.DATA_RELPATH} (sha256 {b23.DATA_SHA256})",
    )
    fb_genes = {
        exp: n
        for exp, n in feba.connection.execute(
            "select expName, count(*) from GeneFitness where orgId = 'Putida' "
            "group by expName"
        )
    }
    registered = registered_classes()
    entries: list[SourceDecomposition] = []
    details: list[SourceDetail] = []
    for label, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        names = [s.exp_name for s in members]
        served = sum(1 for s in members if b24.drop_reason(s) is None)
        if label.startswith("fitness-browser:"):
            pub = label.split(":", 1)[1]
            check = fitness_cross_check(
                feba,
                "Putida",
                names,
                compendium,
                tolerance=ROUNDING_HALF_UNIT,
                basis=ROUNDING_BASIS,
            )
            if pub == "no-pubId":
                key, doi, role = None, None, "unattributed_release"
                source_label = "fitness-browser:no-pubId"
            else:
                named = {k for k in own_attr[label] if k != "borchert2024"}
                if len(named) == 1:
                    # The compendium already names this study: its pinned DOI and key.
                    study = b24.SOURCE_STUDIES[named.pop()]  # type: ignore[index]
                    key, doi = study.citation_key, norm_doi(study.doi)
                else:
                    key, doi = fb_study(pub, publications, mirrors)
                role = "original_study"
                source_label = f"doi:{doi}" if doi else f"fitness-browser:{pub}"
            entry = SourceDecomposition(
                source_label=source_label,
                role=role,  # type: ignore[arg-type]
                citation_key=key,
                doi=doi,
                dataset_class=BORCHERT_CLASS,
                n_samples_in_compendium=len(members),
                n_samples_served=served,
                own_release=True,
                own_release_kind="per_record_values",
                own_release_url=FEBA_URL,
                own_release_access=access_of(probes[FEBA_URL]),
                mirrored=key is not None,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=check,
                note=f"Fitness Browser pubId {pub}; the compendium attributes these "
                f"samples {dict(own_attr[label])}; browser genes per sample "
                f"{sorted({fb_genes[n] for n in names})} against the compendium's "
                f"{len(release.genes)}",
            )
        elif label == "compendium:borchert2023":
            entry = SourceDecomposition(
                source_label=f"doi:{norm_doi(b23.DOI)}",
                role="original_study",
                citation_key=b23.CITATION_KEY,
                doi=norm_doi(b23.DOI),
                dataset_class=BORCHERT_CLASS,
                n_samples_in_compendium=len(members),
                n_samples_served=served,
                own_release=True,
                own_release_kind="per_record_values",
                own_release_url=b23.DATA_URL,
                own_release_access=access_from_manifest(
                    b23.load_manifest(), b23.DATA_RELPATH
                ),
                mirrored=True,
                loaded_as_own_dataset=True,
                own_dataset_class=OWN_DATASET_FOR_SAMPLES[b23.CITATION_KEY],
                cross_check=b23_check,
                note="per-replicate fitness with the 271 loci the compendium "
                "eliminated and Table S4's sampling times; not in the browser",
            )
        else:
            entry = SourceDecomposition(
                source_label=f"doi:{norm_doi(b24.DOI)}",
                role="compendium_paper",
                citation_key=b24.CITATION_KEY,
                doi=norm_doi(b24.DOI),
                dataset_class=BORCHERT_CLASS,
                n_samples_in_compendium=len(members),
                n_samples_served=served,
                own_release=False,
                own_release_kind="none",
                own_release_url=None,
                own_release_access=None,
                mirrored=True,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=None,
                note="Beckham-group samples (sets "
                f"{sorted({s.set_name for s in members})}) absent from the browser and "
                "from Borchert 2023's SI; their reads sit in SRA (PRJNA856070, "
                "PRJNA1011287, per the loader note), so the compendium is the only "
                "per-gene artifact",
            )
        entries.append(entry)
        details.append(
            SourceDetail(
                source_label=entry.source_label,
                dataset_class=BORCHERT_CLASS,
                compendium_attribution=dict(own_attr[label]),
                second_attribution=dict(fb_attr[label]),
                accessions=tuple(names),
                probe_urls=(entry.own_release_url,) if entry.own_release_url else (),
                data_availability=(
                    data_availability(root, entry.citation_key)
                    if entry.citation_key
                    else None
                ),
                registered_classes_for_key=registered.get(entry.citation_key or "", ()),
            )
        )
    names_in_release = {s.exp_name for s in release.samples}
    extra = Counter(
        (e["pubId"] or "no-pubId") for n, e in fb.items() if n not in names_in_release
    )
    fields_absent = (
        "vessel",
        "shaking",
        "pH",
        "nGenerations",
        "dateStarted",
        "timeZeroSet",
        "growthPlate",
        "gMed",
        "cor12",
        "mad12",
        "adjcor",
        "gccor",
    )
    filled = {
        f: sum(1 for n in names_in_release if n in fb and fb[n][f] not in (None, ""))
        for f in fields_absent
    }
    drops = Counter(b24.drop_reason(s) or "kept" for s in release.samples)
    in_fb = sum(1 for n in names_in_release if n in fb)
    findings = (
        f"loci eliminated: the compendium keeps {len(release.genes)} genes; the browser "
        f"values {sorted(set(fb_genes[n] for n in names_in_release if n in fb))} genes "
        f"for each of the {in_fb} browser-sourced samples, so "
        f"{min(fb_genes[n] for n in names_in_release if n in fb) - len(release.genes)} "
        "loci per sample are recoverable from the browser alone (Borchert 2023's SI "
        "recovers 271 for its 42 samples)",
        "values rounded: every browser-sourced sample equals the browser to the "
        f"three-decimal half unit ({ROUNDING_HALF_UNIT})",
        "attribution: the compendium names 5 studies; the browser's pubId names "
        f"{len({k for k in groups if k.startswith('fitness-browser:') and 'no pubId' not in k})} "
        "for its samples, including Incha 2020, Thompson 2019 (lysine and valerolactam), "
        "Garber 2020, Price 2018b and Price 2021b, which the compendium leaves "
        "compendium-attributed",
        f"browser metadata the compendium release does not carry (samples with a "
        f"value): {filled}",
        f"loader drops (served store): {dict(drops)}",
        f"the browser holds {sum(extra.values())} Putida experiments not in the "
        f"compendium ({dict(extra)})",
        "the browser lists the RCH2_defined_noCarbon recipe the loader lacks "
        f"({feba.media('RCH2_defined_noCarbon')} MediaComponents rows; issue #860)",
        "per-strain fitness (barcode, insertion position) is served at "
        f"{FB_STRAIN_FITNESS.format(org='Putida')} "
        f"(HTTP {probes[FB_STRAIN_FITNESS.format(org='Putida')].status}, "
        f"{probes[FB_STRAIN_FITNESS.format(org='Putida')].content_length} bytes), while "
        f"the browser's HTML pages answer {probes[FB_ORG_PAGE].status}",
    )
    record = DecompositionRecord(citation_key=b24.CITATION_KEY, sources=tuple(entries))
    return record, CompendiumAudit(
        citation_key=b24.CITATION_KEY,
        dataset_classes=(BORCHERT_CLASS,),
        attribution_column="Fitness Browser Experiment.pubId where the browser holds "
        "the sample; else borchert2024.attribute_sample",
        n_compendium_samples=len(release.samples),
        n_served_samples=sum(1 for s in release.samples if b24.drop_reason(s) is None),
        counts=study_counts(entries),
        dropped_or_recomputed=findings,
        details=tuple(details),
    )


# --------------------------------------------------------------------------- #
# Lim 2022 (putidaPRECISE321)
# --------------------------------------------------------------------------- #
LIM_CLASS: Final = "PutidaPrecise321Lim2022Dataset"
LIM_SI_REL: Final = f"{lim2022.RAW_DIR_REL}/{lim2022.SI_XLSX_REL}"


def lim_compendium(root: Path) -> pd.DataFrame:
    """The 321 flagged rows of the sample sheet (sha256 checked against the manifest)."""
    path = root / LIM_SI_REL
    expected = lim2022.manifest_sha256(
        lim2022.load_manifest(str(root)), lim2022.SI_XLSX_REL
    )
    if sha256_file(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the raw manifest")
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        samples = workbook.parse(lim2022.SAMPLE_SHEET, index_col=0)
    return lim2022.compendium_samples(samples)


def _cells(series: pd.Series) -> list[str]:
    return sorted({str(v).strip() for v in series.dropna() if str(v).strip()})


def lim_plan(root: Path) -> list[tuple[str, Literal["GET", "HEAD"]]]:
    """Probes the Lim audit reads."""
    comp = lim_compendium(root)
    plan: list[tuple[str, Literal["GET", "HEAD"]]] = []
    for _, group in comp.groupby("project"):
        for gse in _cells(group["GEO Series"]):
            plan.append((geo_suppl_url(gse), "GET"))
        for bp in _cells(group["BioProject"]):
            plan.append((ena_runs_url(bp), "GET"))
    for pmid in _cells(comp["PMID"]):
        plan.append((pubmed_url(pmid.split(";")[0].split(".")[0]), "GET"))
    return plan


def audit_lim(
    root: Path, mirrors: MirrorIndex, probes: Mapping[str, Probe]
) -> tuple[DecompositionRecord, CompendiumAudit]:
    """Sources by the sheet's DOI, else PMID, else project; in-house = Lim 2022."""
    comp = lim_compendium(root)
    ledger = json.loads(
        (
            root
            / "data/torchcell/putida_precise321_lim2022/preprocess/sample_ledger.json"
        ).read_text()
    )
    served_names = {e["sample_name"] for e in ledger if e["index"] is not None}
    groups: dict[str, list[tuple[str, pd.Series, lim2022.SourceStudy]]] = {}
    for name, row in comp.iterrows():
        source = lim2022.source_study(row)
        key = "lim2022" if source.generated_in_this_study else source.study_key
        groups.setdefault(key, []).append((str(row["Sample_name"]), row, source))
    registered = registered_classes()
    entries: list[SourceDecomposition] = []
    details: list[SourceDetail] = []
    for key, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rows = pd.DataFrame([m[1] for m in members])
        names = [m[0] for m in members]
        served = sum(1 for n in names if n in served_names)
        geo = _cells(rows["GEO Series"])
        bps = _cells(rows["BioProject"])
        sources = [m[2] for m in members]
        if key == "lim2022":
            entry = SourceDecomposition(
                source_label=f"doi:{norm_doi(lim2022.PAPER_DOI)}",
                role="compendium_paper",
                citation_key=lim2022.CITATION_KEY,
                doi=norm_doi(lim2022.PAPER_DOI),
                dataset_class=LIM_CLASS,
                n_samples_in_compendium=len(members),
                n_samples_served=served,
                own_release=False,
                own_release_kind="none",
                own_release_url=None,
                own_release_access=None,
                mirrored=True,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=None,
                note=f"in-house samples of projects {sorted({s.project for s in sources})}; "
                f"their reads are Lim 2022's own GEO deposits {geo}",
            )
            accessions = tuple(geo + bps)
            urls: tuple[str, ...] = ()
        else:
            doi = sources[0].doi
            pmid = sources[0].pmid
            if doi is None and pmid is not None:
                doi = probes[pubmed_url(pmid)].doi
            doi = norm_doi(doi) if doi else None
            ckey = mirrors.by_doi.get(doi) if doi else None
            role: SourceRole = (
                "original_study"
                if (sources[0].doi or sources[0].pmid)
                else "unattributed_release"
            )
            call = sequencing_release(geo, bps, probes)
            entry = SourceDecomposition(
                source_label=key
                if role == "original_study"
                else f"project:{sources[0].project} ({','.join(bps[:2])}{'...' if len(bps) > 2 else ''})",
                role=role,
                citation_key=ckey,
                doi=doi,
                dataset_class=LIM_CLASS,
                n_samples_in_compendium=len(members),
                n_samples_served=served,
                own_release=call.kind != "none",
                own_release_kind=call.kind,
                own_release_url=call.url,
                own_release_access=call.access,
                mirrored=ckey is not None,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=None,
                note=f"{call.reason}; projects {sorted({s.project for s in sources})}",
            )
            accessions = tuple(geo + bps)
            urls = call.urls
        entries.append(entry)
        details.append(
            SourceDetail(
                source_label=entry.source_label,
                dataset_class=LIM_CLASS,
                compendium_attribution={key: len(members)},
                accessions=accessions,
                probe_urls=urls,
                data_availability=(
                    data_availability(root, entry.citation_key)
                    if entry.citation_key and entry.role != "compendium_paper"
                    else None
                ),
                registered_classes_for_key=registered.get(entry.citation_key or "", ()),
            )
        )
    drop_reasons = Counter(e["drop_reason"] or "kept" for e in ledger)
    findings = (
        "values recomputed: the compendium re-aligned every library to AE015451.2 and "
        "re-quantified log2(TPM + 1) (lim2022.REFERENCE_GENOME, EXPRESSION_UNIT); no "
        "source's own processed values are carried",
        "conditions dropped: the sample sheet carries no medium or temperature column "
        "for any sample, so every environment but the aromatic project's is a "
        "composition-deferred placeholder (lim2022 ENVIRONMENT); the sources' own GEO "
        "sample records are where those conditions are stated",
        "replicate structure: Lim 2022 discarded libraries with replicate R^2 < 0.95 "
        "(lim2022.REPLICATE_FILTER) and kept 321 of 541 collected",
        f"loader drops (served store): {dict(drop_reasons)}",
        "16 ionic-liquid ALE samples pair with in-house SBRG count columns, not their "
        "own SRX columns, which do not reproduce the published values (lim2022 "
        "pair_count_columns)",
    )
    record = DecompositionRecord(
        citation_key=lim2022.CITATION_KEY, sources=tuple(entries)
    )
    return record, CompendiumAudit(
        citation_key=lim2022.CITATION_KEY,
        dataset_classes=(LIM_CLASS,),
        attribution_column="sample sheet DOI, else PMID, else project (Note 'in-house "
        "data' = Lim 2022)",
        n_compendium_samples=len(comp),
        n_served_samples=len(served_names),
        counts=study_counts(entries),
        dropped_or_recomputed=findings,
        details=tuple(details),
    )


# --------------------------------------------------------------------------- #
# Lamoureux 2023: PRECISE-1K and the Public K-12 arm
# --------------------------------------------------------------------------- #
P1K_CLASS: Final = "RnaseqLamoureux2023Dataset"
PK12_CLASS: Final = "RnaseqPublicK12Lamoureux2023Dataset"
P1K_META_REL: Final = "torchcell-raw/lamoureuxMultiscaleExpressionRegulation2023/data/precise1k/metadata_qc.csv"
PK12_META_REL: Final = "torchcell-raw/lamoureuxMultiscaleExpressionRegulation2023/data/k12_modulome/metadata_qc.csv"


def lamoureux_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PRECISE-1K's 1,035 rows and the Public K-12 arm's 1,675 non-p1k rows."""
    manifest = Manifest.model_validate_json(
        (
            root
            / "torchcell-raw/lamoureuxMultiscaleExpressionRegulation2023"
            / MANIFEST_FILENAME
        ).read_text()
    )
    pins = {f.path: f.sha256 for f in manifest.files}
    for rel in (P1K_META_REL, PK12_META_REL):
        inner = rel.split("lamoureuxMultiscaleExpressionRegulation2023/", 1)[1]
        if sha256_file(root / rel) != pins[inner]:
            raise SystemExit(f"{rel}: sha256 differs from the raw manifest")
    precise = pd.read_csv(root / P1K_META_REL, index_col=0)
    public = pd.read_csv(root / PK12_META_REL, index_col=0, low_memory=False)
    public = public[~public.index.astype(str).str.startswith("p1k_")]
    return precise, public


def _first(values: pd.Series) -> str | None:
    cells = _cells(values)
    return cells[0] if cells else None


def p1k_label(cell: Any) -> str:
    """The DOI column as a label: ``doi:`` for a DOI, ``url:`` for anything else."""
    if pd.isna(cell) or not str(cell).strip():
        return "lamoureux2023"
    text = str(cell).strip()
    doi = norm_doi(text)
    return f"doi:{doi}" if doi.startswith("10.") else f"url:{text}"


def pk12_label(row: pd.Series) -> tuple[str, SourceRole]:
    """PMID, else DOI, else GEO, else BioProject: the most specific attribution cell."""
    for column, prefix, role in (
        ("PMID", "pmid", "original_study"),
        ("DOI", "doi", "original_study"),
        ("GEO", "geo", "unattributed_release"),
        ("BioProject", "bioproject", "unattributed_release"),
    ):
        value = row[column]
        if pd.notna(value) and str(value).strip():
            text = str(value).strip()
            if prefix == "doi":
                text = norm_doi(text)
            return f"{prefix}:{text}", role  # type: ignore[return-value]
    raise ValueError(f"{row.name}: no attribution cell")


def lamoureux_plan(root: Path) -> list[tuple[str, Literal["GET", "HEAD"]]]:
    """Probes the Lamoureux audit reads."""
    precise, public = lamoureux_tables(root)
    plan: list[tuple[str, Literal["GET", "HEAD"]]] = []
    for gse in _cells(precise["GEO"].astype(str).str.split(";").explode()):
        plan.append((geo_suppl_url(gse), "GET"))
    for gse in _cells(public["GEO"].astype(str).str.split(";").explode()):
        if gse != "nan":
            plan.append((geo_suppl_url(gse), "GET"))
    for _, row in public.iterrows():
        if pd.isna(row["GEO"]) and pd.notna(row["BioProject"]):
            plan.append((ena_runs_url(str(row["BioProject"]).strip()), "GET"))
    for cell in _cells(public["PMID"]):
        plan.append((pubmed_url(cell.split(";")[0]), "GET"))
    return [(u, m) for u, m in plan if "/nan" not in u]


def _geo_list(values: pd.Series) -> list[str]:
    out: set[str] = set()
    for cell in values.dropna():
        for part in str(cell).split(";"):
            if part.strip().startswith("GSE"):
                out.add(part.strip())
    return sorted(out)


def audit_lamoureux(
    root: Path, mirrors: MirrorIndex, probes: Mapping[str, Probe]
) -> tuple[DecompositionRecord, CompendiumAudit]:
    """PRECISE-1K by its DOI column; Public K-12 by PMID, DOI, GEO, then BioProject."""
    precise, public = lamoureux_tables(root)
    store = root / "data/torchcell"
    served_p1k = set(
        json.loads(
            (store / "rnaseq_lamoureux2023/preprocess/record_samples.json").read_text()
        )
    )
    served_pk12 = set(
        json.loads(
            (
                store / "rnaseq_public_k12_lamoureux2023/preprocess/record_samples.json"
            ).read_text()
        )
    )
    registered = registered_classes()
    entries: list[SourceDecomposition] = []
    details: list[SourceDetail] = []
    p1k_doi = norm_doi(p1k.PAPER_DOI)

    def add(
        label: str,
        role: SourceRole,
        doi: str | None,
        klass: str,
        rows: pd.DataFrame,
        served: set[str],
        compendium_cells: dict[str, int],
    ) -> None:
        geo = _geo_list(rows["GEO"])
        bps = _cells(rows["BioProject"]) if "BioProject" in rows else []
        n_served = sum(1 for i in rows.index if str(i) in served)
        ckey = mirrors.by_doi.get(doi) if doi else None
        if role == "compendium_paper":
            entry = SourceDecomposition(
                source_label=label,
                role=role,
                citation_key=p1k.CITATION_KEY,
                doi=p1k_doi,
                dataset_class=klass,
                n_samples_in_compendium=len(rows),
                n_samples_served=n_served,
                own_release=False,
                own_release_kind="none",
                own_release_url=None,
                own_release_access=None,
                mirrored=True,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=None,
                note=f"first released in PRECISE-1K; raw reads in GEO {geo} for "
                f"{int(rows['GEO'].notna().sum())} of {len(rows)} samples",
            )
            urls: tuple[str, ...] = ()
        else:
            call = sequencing_release(geo, [] if geo else bps, probes)
            entry = SourceDecomposition(
                source_label=label,
                role=role,
                citation_key=ckey,
                doi=doi,
                dataset_class=klass,
                n_samples_in_compendium=len(rows),
                n_samples_served=n_served,
                own_release=call.kind != "none",
                own_release_kind=call.kind,
                own_release_url=call.url,
                own_release_access=call.access,
                mirrored=ckey is not None,
                loaded_as_own_dataset=False,
                own_dataset_class=None,
                cross_check=None,
                note=f"{call.reason}; projects {sorted(set(rows['project'].astype(str)))[:4]}",
            )
            urls = call.urls
        entries.append(entry)
        details.append(
            SourceDetail(
                source_label=label,
                dataset_class=klass,
                compendium_attribution=compendium_cells,
                accessions=tuple(geo + bps),
                probe_urls=urls,
                data_availability=(data_availability(root, ckey) if ckey else None),
                registered_classes_for_key=registered.get(ckey or "", ()),
            )
        )

    labels = precise["DOI"].map(p1k_label)
    for label, rows in sorted(
        precise.groupby(labels), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        if label == "lamoureux2023":
            add(
                f"doi:{p1k_doi}",
                "compendium_paper",
                p1k_doi,
                P1K_CLASS,
                rows,
                served_p1k,
                {"empty-DOI": len(rows)},
            )
        else:
            doi = label[4:] if label.startswith("doi:") else None
            add(
                label,
                "original_study",
                doi,
                P1K_CLASS,
                rows,
                served_p1k,
                {str(rows["DOI"].iloc[0]).strip(): len(rows)},
            )
    pk_labels = public.apply(lambda r: pk12_label(r)[0], axis=1)
    pk_roles = public.apply(lambda r: pk12_label(r)[1], axis=1)
    for label, rows in sorted(
        public.groupby(pk_labels), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        role = pk_roles.loc[rows.index[0]]
        if label.startswith("pmid:"):
            doi = probes[pubmed_url(label[5:].split(";")[0])].doi
        elif label.startswith("doi:"):
            doi = label[4:]
        else:
            doi = None
        add(label, role, doi, PK12_CLASS, rows, served_pk12, {label: len(rows)})
    p1k_drops = json.loads(
        (store / "rnaseq_lamoureux2023/preprocess/dropped_records.json").read_text()
    )
    pk12_drops = json.loads(
        (
            store / "rnaseq_public_k12_lamoureux2023/preprocess/dropped_records.json"
        ).read_text()
    )
    findings = (
        "attribution not stored: both loaders write Lamoureux 2023 as every record's "
        "Publication (lamoureux2023.py and lamoureux2023_public_k12.py build "
        "Publication(doi=PAPER_DOI)); the per-sample DOI, PMID, GEO and BioProject "
        "cells exist only in the release metadata",
        f"PRECISE-1K: {int(precise['DOI'].notna().sum())} of {len(precise)} samples name "
        "an earlier paper in the DOI column; the rest are first released in PRECISE-1K",
        "accessions missing: of the "
        f"{int(precise['DOI'].isna().sum())} first-released samples, "
        f"{int((precise['DOI'].isna() & precise['GEO'].isna() & precise['SRX'].isna()).sum())} "
        "carry no GEO or SRX accession and are named only by an s3://precise-db path in "
        "R1, although the Data Availability says the accession numbers are listed in "
        "the metadata file (lamoureux2023.DATA_AVAILABILITY's paragraph)",
        "values recomputed: every library re-aligned to NC_000913.3 and re-quantified "
        "(lamoureux2023.ALIGNMENT_REFERENCE, UNIT); the Public K-12 log2[TPM] is not "
        "released at all, so its served TPM is computed from released counts "
        "(lamoureux2023_public_k12 docstring)",
        "replicate structure: PRECISE-1K and Public K-12 keep only replicates with "
        "Pearson r >= 0.95 (lamoureux2023.REPLICATE_QC); conditions with one library "
        "were dropped from Public K-12 (lamoureux2023_public_k12.REPLICATE_RULE)",
        f"served: PRECISE-1K keeps {len(served_p1k)} of {len(precise)} "
        f"(dropped_records {p1k_drops.get('kept_records', '?')} kept); Public K-12 keeps "
        f"{len(served_pk12)} of {len(public)} (dropped_records "
        f"{pk12_drops.get('kept_records', '?')} kept)",
    )
    record = DecompositionRecord(citation_key=p1k.CITATION_KEY, sources=tuple(entries))
    return record, CompendiumAudit(
        citation_key=p1k.CITATION_KEY,
        dataset_classes=(P1K_CLASS, PK12_CLASS),
        attribution_column="metadata_qc.csv DOI (PRECISE-1K); PMID, else DOI, else "
        "GEO, else BioProject (Public K-12)",
        n_compendium_samples=len(precise) + len(public),
        n_served_samples=len(served_p1k) + len(served_pk12),
        counts=study_counts(entries),
        dropped_or_recomputed=findings,
        details=tuple(details),
        not_decomposed=(
            "GrowthRateLamoureux2023Dataset: the 89 served rates are PRECISE-1K "
            "metadata cells; whether each source paper reports its own growth rate "
            "needs the papers read, so its own-release status is not measured here",
        ),
    )


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def study_counts(entries: Sequence[SourceDecomposition]) -> dict[str, int]:
    """Counts over distinct studies (a study in two classes counts once)."""
    studies: dict[str, SourceDecomposition] = {}
    for e in entries:
        studies.setdefault(e.source_label, e)
    values = list(studies.values())
    loadable = [s for s in values if s.loadable_from_own_release]
    return {
        "studies": len(values),
        "original_studies": sum(1 for s in values if s.role == "original_study"),
        "unattributed_releases": sum(
            1 for s in values if s.role == "unattributed_release"
        ),
        "compendium_paper": sum(1 for s in values if s.role == "compendium_paper"),
        "loadable_from_own_release": len(loadable),
        "loadable_scriptable": sum(
            1 for s in loadable if s.own_release_access == "scriptable"
        ),
        "loadable_walled": sum(1 for s in loadable if s.own_release_access == "walled"),
        "loadable_per_record_values": sum(
            1 for s in loadable if s.own_release_kind == "per_record_values"
        ),
        "loadable_raw_reads_only": sum(
            1 for s in loadable if s.own_release_kind == "raw_reads_only"
        ),
        "compendium_only": sum(1 for s in values if s.compendium_only),
        "loaded_as_own_dataset": sum(1 for s in values if s.loaded_as_own_dataset),
        "mirrored": sum(1 for s in values if s.mirrored),
        "samples": sum(e.n_samples_in_compendium for e in entries),
        "samples_served": sum(e.n_samples_served for e in entries),
        "samples_loadable_from_own_release": sum(
            e.n_samples_in_compendium for e in entries if e.loadable_from_own_release
        ),
        "samples_compendium_only": sum(
            e.n_samples_in_compendium for e in entries if e.compendium_only
        ),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def static_plan() -> list[tuple[str, Literal["GET", "HEAD"]]]:
    """The fixed probes: the browser, the RB-TnSeq release pages, Borchert 2023's SI."""
    return [
        (FB_ORG_PAGE, "GET"),
        (FEBA_URL, "HEAD"),
        (FB_STRAIN_FITNESS.format(org="Putida"), "HEAD"),
        (FB_STRAIN_FITNESS.format(org="Keio"), "HEAD"),
        (RBARSEQ_URL, "GET"),
        (BIGFIT_URL, "GET"),
        (b23.DATA_URL, "GET"),
    ]


# --------------------------------------------------------------------------- #
# Markdown tables for the dendron note (regenerated from the results JSON)
# --------------------------------------------------------------------------- #
TABLE_HEADER: Final = (
    "| source | role | samples (served) | own release | access | mirrored | loaded | "
    "L4 cross-check | compendium's attribution |\n"
    "|---|---|---|---|---|---|---|---|---|"
)
COUNT_KEYS: Final = (
    "studies",
    "loadable_scriptable",
    "loadable_walled",
    "loadable_per_record_values",
    "loadable_raw_reads_only",
    "compendium_only",
    "loaded_as_own_dataset",
    "mirrored",
    "samples",
    "samples_served",
    "samples_loadable_from_own_release",
    "samples_compendium_only",
)


def _check_cell(check: SourceCrossCheck | None) -> str:
    if check is None:
        return "not measured"
    verdict = {True: "pass", False: "FAIL", None: "pin"}[check.passes]
    bound = "" if check.tolerance is None else f" vs {check.tolerance:g}"
    return (
        f"{check.statistic} {check.value:.4g}{bound} ({verdict}; "
        f"{check.n_samples} samples)"
    )


def _row(source: SourceDecomposition, detail: SourceDetail) -> str:
    attribution = "; ".join(
        f"{k} {v}" for k, v in detail.compendium_attribution.items()
    )
    if detail.second_attribution:
        browser = ", ".join(f"{k} {v}" for k, v in detail.second_attribution.items())
        attribution += f" (browser: {browser})"
    return (
        f"| {source.source_label} | {source.role} | {source.n_samples_in_compendium} "
        f"({source.n_samples_served}) | {source.own_release_kind} | "
        f"{source.own_release_access or '-'} | "
        f"{source.citation_key or 'no'} | {source.own_dataset_class or 'no'} | "
        f"{_check_cell(source.cross_check)} | {attribution.replace('|', '/')} |"
    )


def markdown_tables(result: AuditResult) -> str:
    """Per compendium: counts, the per-source table, and what it dropped or recomputed.

    A served class with more than 30 sources lists only those with a served sample and
    totals the rest by role and release kind.
    """
    blocks: list[str] = [
        f"Generated by `{SCRIPT}` from `results/{AUDIT_JSON.name}` "
        f"(measured {result.measured_at[:10]}). Do not edit by hand."
    ]
    for record, audit in zip(result.decompositions, result.audit, strict=True):
        details = {(d.source_label, d.dataset_class): d for d in audit.details}
        counts = ", ".join(f"{k} {audit.counts[k]}" for k in COUNT_KEYS)
        blocks.append(f"#### {audit.citation_key}\n\nCounts: {counts}.")
        for klass in audit.dataset_classes:
            sources = sorted(
                (s for s in record.sources if s.dataset_class == klass),
                key=lambda s: (-s.n_samples_in_compendium, s.source_label),
            )
            shown = (
                sources
                if len(sources) <= 30
                else [s for s in sources if s.n_samples_served > 0]
            )
            rows = [_row(s, details[(s.source_label, klass)]) for s in shown]
            rest = [s for s in sources if s not in shown]
            if rest:
                totals = Counter((s.role, s.own_release_kind) for s in rest)
                samples = sum(s.n_samples_in_compendium for s in rest)
                rows.append(
                    f"| {len(rest)} more, none served | "
                    + ", ".join(f"{r} {k}: {n}" for (r, k), n in sorted(totals.items()))
                    + f" | {samples} (0) | | | | | | |"
                )
            blocks.append(
                f"`{klass}` ({len(sources)} sources)\n\n"
                + "\n".join([TABLE_HEADER, *rows])
            )
        dropped = "\n".join(f"- {line}" for line in audit.dropped_or_recomputed)
        blocks.append(f"Dropped or recomputed relative to the originals:\n\n{dropped}")
        if audit.not_decomposed:
            blocks.append(
                "Not decomposed:\n\n"
                + "\n".join(f"- {line}" for line in audit.not_decomposed)
            )
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# Preparing for the yeast compendia: attribution only, nothing is loaded
# --------------------------------------------------------------------------- #
YEASTRACT_KEY: Final = "teixeiraYEASTRACTPortalExploitation2023"
YEASTRACT_ZIP: Final = "data/YeastractPlus_2022_all_regs.zip"
YEASTRACT_MEMBER: Final = "YeastractPlus_2022_Scerevisiae_regs_flatfile.tsv.gz"
#: The flat file has no header; the columns as the maintainers' release lays them out
#: (regulator ORF and name, target ORF and name, PubMed id, sign, evidence, method,
#: environment group, condition, medium), read off the rows.
YEASTRACT_COLUMNS: Final = (
    "regulator_orf",
    "regulator",
    "target_orf",
    "target",
    "pmid",
    "sign",
    "evidence",
    "method",
    "environment_group",
    "environment",
    "medium",
)
SPELL_DIR_REL: Final = "data/sgd/spell"
SPELL_ARCHIVE: Final = "all_spell_datasets.tar.gz"


class YeastSources(_Frozen):
    """How the two yeast compendia attribute their records, read off what is on disk."""

    measured_at: str
    yeastract_file: str
    yeastract_sha256: str
    yeastract_rows: int
    yeastract_pmids: int
    yeastract_top_pmids: dict[str, int]
    yeastract_evidence: dict[str, int]
    yeastract_methods: int
    yeastract_top_methods: dict[str, int]
    yeastract_regulators: int
    yeastract_targets: int
    yeastract_media_cells: int
    spell_dir: str
    spell_files_on_disk: int
    spell_archive_present: bool


def yeast_sources(root: Path) -> YeastSources:
    """YEASTRACT+'s per-row PubMed attribution, and whether the SPELL archive is on disk."""
    import gzip
    import zipfile

    path = root / RAW_DIR / YEASTRACT_KEY / YEASTRACT_ZIP
    manifest = Manifest.model_validate_json(
        (root / RAW_DIR / YEASTRACT_KEY / MANIFEST_FILENAME).read_text()
    )
    pinned = next(f.sha256 for f in manifest.files if f.path == YEASTRACT_ZIP)
    digest = sha256_file(path)
    if digest != pinned:
        raise SystemExit(f"{path}: sha256 {digest}, manifest {pinned}")
    with zipfile.ZipFile(path) as archive:
        text = gzip.decompress(archive.read(YEASTRACT_MEMBER)).decode("utf-8")
    rows = [
        dict(zip(YEASTRACT_COLUMNS, line.split("\t"), strict=True))
        for line in text.splitlines()
    ]
    pmids = Counter(r["pmid"] for r in rows)
    methods = Counter(r["method"] for r in rows)
    spell = root / SPELL_DIR_REL
    return YeastSources(
        measured_at=now(),
        yeastract_file=f"{RAW_DIR}/{YEASTRACT_KEY}/{YEASTRACT_ZIP}:{YEASTRACT_MEMBER}",
        yeastract_sha256=digest,
        yeastract_rows=len(rows),
        yeastract_pmids=len(pmids),
        yeastract_top_pmids=dict(pmids.most_common(5)),
        yeastract_evidence=dict(Counter(r["evidence"] for r in rows)),
        yeastract_methods=len(methods),
        yeastract_top_methods=dict(methods.most_common(6)),
        yeastract_regulators=len({r["regulator_orf"] for r in rows}),
        yeastract_targets=len({r["target_orf"] for r in rows}),
        yeastract_media_cells=sum(1 for r in rows if r["medium"] not in ("", "N/A")),
        spell_dir=f"$DATA_ROOT/{SPELL_DIR_REL}",
        spell_files_on_disk=sum(1 for _ in spell.rglob("*") if _.is_file()),
        spell_archive_present=(spell / SPELL_ARCHIVE).is_file(),
    )


def main(argv: list[str] | None = None) -> None:
    """Probe (optionally), audit, and write both results files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feba-db", type=Path)
    parser.add_argument(
        "--yeast-only",
        action="store_true",
        help="measure how YEASTRACT+ attributes its rows and whether SPELL is on disk",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="only re-render the markdown tables from the committed results JSON",
    )
    parser.add_argument("--network", action="store_true")
    parser.add_argument(
        "--reprobe-failed",
        action="store_true",
        help="re-ask the committed probes that answered 5xx or raised",
    )
    parser.add_argument(
        "--no-verify-feba",
        action="store_true",
        help="skip the 10 GB sha256 re-hash (size is still checked)",
    )
    args = parser.parse_args(argv)
    if args.yeast_only:
        record = yeast_sources(data_root())
        YEAST_JSON.write_text(record.model_dump_json(indent=2) + "\n")
        print(f"wrote {YEAST_JSON}")
        return
    if args.tables_only:
        result = AuditResult.model_validate_json(AUDIT_JSON.read_text())
        TABLES_MD.write_text(markdown_tables(result))
        print(f"wrote {TABLES_MD}")
        return
    if args.feba_db is None:
        parser.error("--feba-db is required unless --tables-only or --yeast-only")
    root = data_root()
    if args.network:
        plan = static_plan() + lim_plan(root) + lamoureux_plan(root)
        probes = run_probes(plan)
        PROBES_JSON.write_text(probes.model_dump_json(indent=2) + "\n")
        print(f"wrote {PROBES_JSON} ({len(probes.probes)} probes)")
    probes = Probes.model_validate_json(PROBES_JSON.read_text())
    if args.reprobe_failed:
        probes = reprobe_failed(probes)
        PROBES_JSON.write_text(probes.model_dump_json(indent=2) + "\n")
        left = sum(1 for p in probes.probes.values() if transient(p))
        print(f"re-probed; {left} probes still transient")
    if args.no_verify_feba and args.feba_db.stat().st_size != FEBA_BYTES:
        raise SystemExit(f"{args.feba_db}: size differs from the pin")
    feba = Feba(args.feba_db, verify=not args.no_verify_feba)
    mirrors = mirror_index(root)
    results = [
        audit_price(root, feba, mirrors, probes.probes),
        audit_borchert(root, feba, mirrors, probes.probes),
        audit_lim(root, mirrors, probes.probes),
        audit_lamoureux(root, mirrors, probes.probes),
    ]
    result = AuditResult(
        script=SCRIPT,
        measured_at=now(),
        feba={
            "url": FEBA_URL,
            "sha256": FEBA_SHA256,
            "bytes": FEBA_BYTES,
            "last_modified": FEBA_LAST_MODIFIED,
            "etag": FEBA_ETAG,
            "retrieved_at": FEBA_RETRIEVED_AT,
            "retrieval_command": FEBA_RETRIEVAL_COMMAND,
            "mirrored": False,
            "sha256_verified_this_run": not args.no_verify_feba,
        },
        invalid_manifests=mirrors.invalid_manifests,
        decompositions=tuple(r[0] for r in results),
        audit=tuple(r[1] for r in results),
    )
    AUDIT_JSON.write_text(result.model_dump_json(indent=2) + "\n")
    TABLES_MD.write_text(markdown_tables(result))
    for audit in result.audit:
        print(audit.citation_key, json.dumps(audit.counts))
    print(f"wrote {AUDIT_JSON}")


if __name__ == "__main__":
    main()
