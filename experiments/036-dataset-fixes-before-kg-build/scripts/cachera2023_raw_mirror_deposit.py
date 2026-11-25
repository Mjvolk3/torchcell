# experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_raw_mirror_deposit.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_raw_mirror_deposit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_raw_mirror_deposit.py
"""Deposit the Cachera 2023 raw files into the raw mirror with their retrieval records.

The paper's Data Availability (``paper.md`` line 175) names the CRI-SPA GitHub
repository for the per-screen and the corrected, filtered datasets. The loader reads two
of them (``GA1_2_4_6.csv`` per gene, ``QD1_2_4_6.csv`` per colony, #1177), and the
repository ``README.md`` defines both files, so all three are fetched from the repository
at the pinned commit ``REPO_COMMIT`` through the registered ``direct_url`` retriever,
held to their sha256, written under
``$DATA_ROOT/torchcell-raw/cacheraCRISPAHighthroughputMethod2023/data/`` and recorded in
that directory's ``manifest.json``. A file already in the mirror with the pinned hash is
left alone; one with another hash raises.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_raw_mirror_deposit.py \
        --retrieved-at 2026-11-25T05:46:18Z
"""

import argparse
import hashlib
import os
import os.path as osp
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import cachera2023 as c
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.retrieve import direct_url

#: (file in the repository, its pinned sha256). README.md is the repository's own
#: definition of the QD and GA files; it was read 2026-11-25 and its hash is pinned here.
FILES: tuple[tuple[str, str], ...] = (
    (c.QD_FILENAME, c.QD_SHA256),
    (c.DATA_FILENAME, c.DATA_SHA256),
    ("README.md", "e116c0ad3c25bd153270cb9704314d76356d637187f313e99f4fae3d645522fa"),
)


def _url(name: str) -> str:
    return (
        f"https://raw.githubusercontent.com/pc2912/CRI-SPA_repo/{c.REPO_COMMIT}/{name}"
    )


def deposit(data_root: str, retrieved_at: str) -> Path:
    """Fetch, verify and write every file of ``FILES``, then the manifest."""
    root = Path(data_root) / c.RAW_MIRROR_REL
    records: list[ArtifactRecord] = []
    for name, sha256 in FILES:
        url = _url(name)
        data = direct_url(url)
        observed = hashlib.sha256(data).hexdigest()
        if observed != sha256:
            raise RuntimeError(f"{url}: sha256 {observed}, pinned {sha256}")
        dest = root / "data" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
                raise RuntimeError(f"{dest} exists with another sha256; refusing")
        else:
            dest.write_bytes(data)
        records.append(
            ArtifactRecord(
                path=f"data/{name}",
                role=ROLE_RAW_DATA,
                bytes=len(data),
                sha256=sha256,
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": url},
                    sha256=sha256,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    manifest = Manifest(
        citation_key=c.CITATION_KEY,
        doi="10.1093/nar/gkad656",
        title=(
            "CRI-SPA: a high-throughput method for systematic genetic editing of yeast "
            "libraries"
        ),
        library_id="6582362",
        zotero_item_key="PXYXS3TD",
        files=records,
        si_data_sources=[
            f"https://github.com/pc2912/CRI-SPA_repo/tree/{c.REPO_COMMIT}"
        ],
        si_expected=[
            "QD1_2_4_6.csv: per-colony raw and corrected values, screens 1/2/4/6 pooled",
            "GA1_2_4_6.csv: per-gene mean, std and count over those colonies",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return root


def main() -> None:
    """Deposit under ``$DATA_ROOT`` and print the mirror directory."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieved-at", required=True)
    args = parser.parse_args()
    out = deposit(os.environ["DATA_ROOT"], args.retrieved_at)
    print(osp.join(out, "manifest.json"))


if __name__ == "__main__":
    main()
