# experiments/database/scripts/check_candidate_overlap.py
# [[experiments.database.expansion-100]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/check_candidate_overlap
r"""Does a candidate dataset already exist in the database, under another name?

Written because three separate rows of the candidate list turned out to be already
held and none was caught by inspection. The failure mode is always the same: the
loader is named for a DIFFERENT paper than the deposit it reads, so a name check
and even a DOI check pass while the data is already ingested.

  Wildenhain 2016  the loader is wildenhain2015.py, and neither the 2016 DOI nor
                   the AID appeared in the candidate's accession string, so there
                   was nothing to match on. Both describe PubChem AID 1159580.
  Turco 2023       yeastphenome.py already exists; the DOI check did catch this one.
  Lee 2014         not a duplicate but not available either, recorded as blocked in
                   the WS15 roadmap rather than anywhere a name check would look.

So the reliable key is the ACCESSION a loader actually reads, and the second key is
the PMID, since an aggregator such as SynLethDB records the primary PMID of every
pair it re-serves.

Three checks, in increasing strength:
  1. DOI       -- candidate DOI cited anywhere in a built loader.
  2. accession -- GEO / PRIDE / SRA / PubChem-AID / ArrayExpress tokens shared with
                  a built loader. This is the one that would have caught Wildenhain,
                  and only if the candidate records its AID.
  3. PMID      -- candidate PMID present in an aggregator's source-PMID column.
                  Currently SynLethDB (SL + SR); extend AGGREGATORS as others land.

DOI to PMID resolution uses the NCBI ID converter and is cached, so a re-run is
offline. Rows whose DOI does not resolve are REPORTED, never silently passed: an
unresolved row is an unchecked row.

Run from the repo root, after build_candidate_datasets_table.py:
  python experiments/database/scripts/check_candidate_overlap.py
  python experiments/database/scripts/check_candidate_overlap.py --offline
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
LOADERS = REPO / "torchcell" / "datasets" / "scerevisiae"
CANDIDATES = SCRIPT.parent.parent / "results" / "candidates" / "candidate_datasets.json"
CACHE = SCRIPT.parent.parent / "results" / "candidates" / "doi2pmid.json"

IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?format=json&ids="

# Accession shapes worth comparing. A bare number is never enough: "1159580" would
# match a line number, so the PubChem form keeps its AID prefix.
ACCESSION = re.compile(
    r"(GSE\d+|PXD\d+|PRJ[EN][AB]\d+|E-MTAB-\d+|SRP\d+|AID\s*\d+|MTBLS\d+|"
    r"10\.5061/dryad\.[A-Za-z0-9]+)"
)

# Aggregators already built, and the column holding each re-served pair's primary
# PMID. Relative to DATA_ROOT.
AGGREGATORS = [
    ("SynLethDB SL", "data/torchcell/syn_leth_db_yeast/raw/Yeast_SL.csv", 5),
    ("SynLethDB SR", "data/torchcell/syn_rescue_db_yeast/raw/Yeast_SR.csv", 5),
]


class Finding(BaseModel):
    candidate: str
    kind: str
    evidence: str


def loader_text() -> str:
    return "\n".join(p.read_text(errors="ignore") for p in LOADERS.glob("*.py"))


def resolve_pmids(dois: list[str], offline: bool) -> dict[str, str]:
    cache: dict[str, str] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [d for d in dois if d.lower() not in cache]
    if missing and not offline:
        for k in range(0, len(missing), 40):
            batch = missing[k : k + 40]
            url = IDCONV + urllib.parse.quote(",".join(batch))
            with urllib.request.urlopen(url, timeout=45) as r:
                data = json.load(r)
            for rec in data.get("records", []):
                if rec.get("pmid") and rec.get("doi"):
                    cache[rec["doi"].lower()] = rec["pmid"]
            time.sleep(1)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return cache


def aggregator_pmids(data_root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, rel, col in AGGREGATORS:
        path = data_root / rel
        if not path.exists():
            print(f"  [skip] {name}: not built at {path}")
            continue
        pm = set()
        for line in path.read_text(errors="ignore").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) > col and parts[col].strip().isdigit():
                pm.add(parts[col].strip())
        out[name] = pm
        print(f"  {name}: {len(pm)} source PMIDs")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="use only the cached DOI map")
    args = ap.parse_args()
    load_dotenv(REPO / ".env")
    import os

    data_root = Path(os.environ["DATA_ROOT"])

    payload = json.loads(CANDIDATES.read_text())
    cands = payload["candidates"]
    src = loader_text()

    print(f"{len(cands)} candidates against {len(list(LOADERS.glob('*.py')))} built loaders")
    aggs = aggregator_pmids(data_root)

    dois = {}
    for c in cands:
        m = re.match(r"https://doi\.org/(.+)$", c["url"])
        if m:
            dois[m.group(1).rstrip("/")] = c["name"]
    d2p = resolve_pmids(list(dois), args.offline)

    findings: list[Finding] = []
    unchecked: list[str] = []
    for c in cands:
        name = c["name"]
        doi = None
        m = re.match(r"https://doi\.org/(.+)$", c["url"])
        if m:
            doi = m.group(1).rstrip("/")
            if doi in src:
                findings.append(
                    Finding(candidate=name, kind="DOI in a built loader", evidence=doi)
                )
        for tok in ACCESSION.findall(c["accession"] + " " + c["url"]):
            if tok in src:
                findings.append(
                    Finding(candidate=name, kind="accession in a built loader", evidence=tok)
                )
        pmid = d2p.get(doi.lower()) if doi else None
        if not pmid:
            f = re.findall(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", c["url"])
            pmid = f[0] if f else None
        if pmid:
            for agg, pm in aggs.items():
                if pmid in pm:
                    findings.append(
                        Finding(
                            candidate=name,
                            kind=f"PMID re-served by {agg}",
                            evidence=pmid,
                        )
                    )
        else:
            unchecked.append(name)

    print()
    if findings:
        print(f"{len(findings)} OVERLAP(S):")
        for f in findings:
            print(f"  {f.candidate}\n      {f.kind}: {f.evidence}")
    else:
        print("no overlaps found")

    n_acc = sum(1 for c in cands if ACCESSION.search(c["accession"]))
    print()
    print(f"coverage: {len(cands) - len(unchecked)}/{len(cands)} rows carry a PMID; "
          f"{n_acc}/{len(cands)} carry a parseable accession")
    if unchecked:
        print("no PMID resolved, so NOT checked against aggregators:")
        for n in unchecked:
            print(f"  - {n}")
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
