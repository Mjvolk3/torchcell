# experiments/026-metabolism-flux/scripts/fetch_kinetics_assets.py
# [[experiments.026-metabolism-flux.scripts.fetch_kinetics_assets]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/fetch_kinetics_assets.py

r"""Mirror the two external assets that complete the k_cat / K_M predictor inputs.

WHAT IS STILL MISSING, AND WHAT EACH ASSET FIXES
-------------------------------------------------
:mod:`kinetics_input_audit` measured the inputs already on disk. Protein sequence is not
a gap: all 1,161 yeast-GEM genes resolve, from `swissprot.tsv` and independently from the
genome's own protein FASTA. Two gaps remain, and they are different gaps that different
assets close. They are easy to conflate because both get called "structure".

**Substrate structure, i.e. SMILES for the small molecule.** This is the binding gap.
yeast-GEM's shipped `smilesDB.tsv` matches 1,800 of 2,806 metabolites by name; 1,006 have
no entry. Of those, 670 carry another identifier (ChEBI 620, MetaNetX 514, KEGG 233).
MetaNetX's `chem_prop.tsv` carries SMILES keyed by MNXM id with cross-references, so one
file closes most of that. AlphaFold does NOT help here.

**Protein structure, i.e. a 3D model.** Only DeepEnzyme needs it. The AlphaFold Database
covers the yeast proteome, verified on four GEM accessions which all return both CIF and
PDB, so this gap is closable to essentially 100 %. It unlocks one predictor rather than
raising the coverage of the other ten.

PROVENANCE
----------
Every file records its source URL, the exact retrieval command, its sha256 and the
retrieval time into a manifest beside it, per the repo's rule that the stored artifact
plus its hash is canonical and the URL is historical retrieval metadata.

Network-bound and GPU-free on purpose, so it runs alongside a training sweep rather than
competing with it.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/fetch_kinetics_assets.py --what all
"""

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv(osp.join(os.getcwd(), ".env"))

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
KINETICS_ROOT = osp.join(DATA_ROOT, "data", "enzyme_kinetics")
METANETX_DIR = osp.join(KINETICS_ROOT, "metanetx")
AFDB_DIR = osp.join(KINETICS_ROOT, "alphafold", "scerevisiae")

METANETX_URL = "https://www.metanetx.org/ftp/latest/chem_prop.tsv"

#: AlphaFold's file URL is resolved PER ACCESSION from this endpoint rather than built
#: from a template. Two reasons, both measured: the model version is part of the filename
#: and it moves (v4 is now 404, the live answer is v6, created 2025-08-01), so a templated
#: version silently rots into a wall of 404s; and the API is the only place that states
#: which version is current for a given accession.
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

#: MetaNetX accepts a descriptive agent. AlphaFold answers 403 to one and 200 to the
#: default, so it is sent no override at all rather than a disguise.
USER_AGENT = "torchcell/026-metabolism-flux (research mirror)"


class RetrievedFile(BaseModel):
    """One mirrored file and everything needed to rebuild and verify it."""

    model_config = ConfigDict(extra="forbid")

    path: str
    source_url: str
    retrieval_method: str
    retrieval_command: str
    sha256: str
    bytes: int
    retrieved_at: str


def sha256_of(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _open(url: str, timeout: int, agent: str | None):
    req = urllib.request.Request(url)
    if agent is not None:
        req.add_header("User-Agent", agent)
    return urllib.request.urlopen(req, timeout=timeout)


def with_retry(fn, attempts: int = 4, base_delay: float = 2.0):
    """Retry a network call on TRANSPORT errors only, with backoff.

    A single DNS blip killed the first 1,139-file run outright, because only HTTPError
    was handled and a URLError propagated. Transport failures are retried; an HTTPError
    is NOT, because a 404 or a 403 is the server's actual answer and retrying it would
    turn a real result into a hang. The distinction is what keeps a long mirror honest.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(base_delay * (2**i))
    raise RuntimeError(f"transport failure after {attempts} attempts: {last}")


def download(
    url: str, dest: str, timeout: int = 300, agent: str | None = USER_AGENT
) -> None:
    os.makedirs(osp.dirname(dest), exist_ok=True)
    tmp = dest + ".part"

    def once() -> None:
        with _open(url, timeout, agent) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)

    with_retry(once)
    os.replace(tmp, dest)


def fetch_metanetx() -> RetrievedFile:
    """Mirror MetaNetX chem_prop.tsv, the SMILES table keyed by MNXM id."""
    dest = osp.join(METANETX_DIR, "chem_prop.tsv")
    print(f"metanetx -> {dest}", flush=True)
    download(METANETX_URL, dest, timeout=1800)
    digest, n = sha256_of(dest)
    print(f"  {n / 1e6:.1f} MB  sha256 {digest[:16]}...", flush=True)
    return RetrievedFile(
        path=dest,
        source_url=METANETX_URL,
        retrieval_method="direct_url",
        retrieval_command=f"curl -L -o {dest} {METANETX_URL}",
        sha256=digest,
        bytes=n,
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def gem_accessions() -> list[str]:
    """UniProt accessions for the GEM's genes, from the model's own swissprot table."""
    src = YeastGEM()
    gem_genes = {g.id.upper() for g in src.model.genes}
    db = osp.join(src.model_dir, "data", "databases", "swissprot.tsv")
    accs: dict[str, None] = {}
    with open(db) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            acc = (row.get("uniprot") or "").strip()
            if not acc:
                continue
            for token in (row.get("gene_id") or "").split():
                if token.strip().upper() in gem_genes:
                    accs.setdefault(acc, None)
                    break
    return sorted(accs)


def fetch_alphafold(accessions: list[str], sleep_s: float) -> list[RetrievedFile]:
    """Mirror one AlphaFold PDB per accession. Only DeepEnzyme consumes these."""
    os.makedirs(AFDB_DIR, exist_ok=True)
    out: list[RetrievedFile] = []
    missing: list[str] = []
    for i, acc in enumerate(accessions, 1):
        # Resolve the current file URL from the API. AlphaFold answers 403 to a custom
        # User-Agent, so this endpoint is called with none.
        try:
            meta = with_retry(
                lambda: json.load(_open(AFDB_API.format(acc=acc), 60, None))
            )
        except urllib.error.HTTPError as e:
            missing.append(f"{acc} api HTTP{e.code}")
            continue
        if not meta:
            missing.append(f"{acc} no model")
            continue
        url = meta[0]["pdbUrl"]
        dest = osp.join(AFDB_DIR, osp.basename(url))
        if osp.exists(dest):
            digest, n = sha256_of(dest)
        else:
            try:
                download(url, dest, timeout=120, agent=None)
            except urllib.error.HTTPError as e:
                # A 404 is a real answer: that accession has no AlphaFold model. It is
                # recorded as absent rather than retried, so coverage stays honest.
                missing.append(f"{acc} file HTTP{e.code}")
                continue
            digest, n = sha256_of(dest)
            time.sleep(sleep_s)
        out.append(
            RetrievedFile(
                path=dest,
                source_url=url,
                retrieval_method="direct_url",
                retrieval_command=f"curl -L -o {dest} {url}",
                sha256=digest,
                bytes=n,
                retrieved_at=datetime.now(UTC).isoformat(),
            )
        )
        if i % 100 == 0:
            print(
                f"  alphafold {i}/{len(accessions)}, {len(missing)} absent", flush=True
            )
    print(f"alphafold: {len(out)} mirrored, {len(missing)} absent", flush=True)
    if missing:
        print(f"  absent: {missing[:10]}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--what", default="all", choices=["all", "metanetx", "alphafold"]
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Pause between AlphaFold requests, to stay a polite client.",
    )
    args = parser.parse_args()

    os.makedirs(KINETICS_ROOT, exist_ok=True)
    records: dict[str, Any] = {}

    if args.what in ("all", "metanetx"):
        rec = fetch_metanetx()
        records["metanetx"] = [rec.model_dump()]
        with open(osp.join(METANETX_DIR, "manifest.json"), "w") as f:
            json.dump({"files": records["metanetx"]}, f, indent=2)

    if args.what in ("all", "alphafold"):
        accs = gem_accessions()
        print(f"alphafold: {len(accs)} GEM accessions to mirror", flush=True)
        recs = fetch_alphafold(accs, args.sleep)
        records["alphafold"] = [r.model_dump() for r in recs]
        with open(osp.join(AFDB_DIR, "manifest.json"), "w") as f:
            json.dump(
                {"n_accessions": len(accs), "files": records["alphafold"]}, f, indent=2
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {k: len(v) for k, v in records.items()}
    with open(osp.join(RESULTS_DIR, "kinetics_assets_fetched.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ndone: {summary}")


if __name__ == "__main__":
    main()
