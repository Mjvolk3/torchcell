#!/usr/bin/env python
# scripts/lit_bib_pull.py
# [[scripts.lit_bib_pull]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_bib_pull.py

r"""Pull one served bibliography from tc-lit and verify it against the manifest.

The client half of the bibliography store: ``GET /bib`` for the manifest, then
``GET /bib/<name>`` for the file, and the downloaded bytes are hashed and compared
with BOTH the manifest's sha256 and the ``X-Artifact-SHA256`` header. The manifest
is the trust anchor; a mismatch leaves the target file untouched and exits
non-zero. Nothing here talks to Zotero, so it runs on any machine that holds a
tc-lit key: the Mac, GilaHyper, a collaborator's laptop.

Usage::

    python scripts/lit_bib_pull.py --list                     # what the server holds
    python scripts/lit_bib_pull.py --name paper --out paper/nature-biotech/references.bib
    python scripts/lit_bib_pull.py --name eqtl-data-model --out notes-tex/eqtl-data-model/references.bib

``make bib-pull`` in a notes-tex document (and in ``paper/nature-biotech``) is a
thin wrapper that passes the document's own name. Reads ``TC_LIT_URL`` and
``TC_LIT_API_KEY`` from ``.env``; both are required.
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.bib_store import BibStoreManifest  # noqa: E402

log = logging.getLogger("lit_bib_pull")


def fetch_manifest(client: httpx.Client) -> BibStoreManifest:
    """``GET /bib`` as the typed store manifest."""
    resp = client.get("/bib")
    resp.raise_for_status()
    return BibStoreManifest.model_validate(resp.json())


def pull_bib(client: httpx.Client, name: str, out: Path) -> str:
    """Download ``name`` to ``out`` after verifying its sha256; return the hash.

    Raises ``RuntimeError`` on a hash mismatch, before anything is written.
    """
    record = fetch_manifest(client).get(name)
    if record is None:
        raise RuntimeError(f"server has no bibliography named {name!r}")
    resp = client.get(f"/bib/{name}")
    resp.raise_for_status()
    got = hashlib.sha256(resp.content).hexdigest()
    header = resp.headers.get("X-Artifact-SHA256")
    if got != record.sha256 or got != header:
        raise RuntimeError(
            f"sha256 mismatch for {name}: downloaded={got} manifest={record.sha256} "
            f"header={header}; nothing written"
        )
    if out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == got:
        log.info(
            "%s: already at %s (%d entries); unchanged", out, got[:12], record.n_entries
        )
        return got
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    log.info(
        "%s: wrote %d entries, sha256 %s (exported %s)",
        out,
        record.n_entries,
        got[:12],
        record.generated_at,
    )
    return got


def main() -> None:
    """List the served bibliographies, or pull one to a path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List served bibliographies."
    )
    parser.add_argument("--name", metavar="NAME", help="Bibliography to pull.")
    parser.add_argument("--out", metavar="PATH", help="Where to write it.")
    args = parser.parse_args()

    url = os.environ["TC_LIT_URL"]
    api_key = os.environ["TC_LIT_API_KEY"]
    with httpx.Client(
        base_url=url, headers={"X-API-Key": api_key}, timeout=60
    ) as client:
        if args.list:
            manifest = fetch_manifest(client)
            print(f"store exported {manifest.generated_at}")
            for record in manifest.bibs:
                print(
                    f"{record.name:<28} {record.n_entries:5d} entries  "
                    f"sha256={record.sha256[:12]}  <- {record.origin}"
                )
            return
        if not args.name or not args.out:
            parser.error("--name and --out are required unless --list")
        try:
            pull_bib(client, args.name, Path(args.out))
        except RuntimeError as exc:
            sys.exit(str(exc))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
