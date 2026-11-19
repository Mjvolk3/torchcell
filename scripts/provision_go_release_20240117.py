# scripts/provision_go_release_20240117.py
# [[scripts.provision_go_release_20240117]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/provision_go_release_20240117.py
"""Deposit ``go.obo`` of the GO release 2024-01-17 into the genomes tier.

``SCerevisiaeGenome`` keeps ``<go_root>/go.obo`` and, when it is absent, downloads the
CURRENT GO file. Every machine built so far holds the copy downloaded in March 2024
(``data-version: releases/2024-01-17``, sha256 :data:`GO_OBO_SHA256`), so a new machine
would read a different ontology than the stores were built with. This script pins that
file as the assembly set ``go_release_2024-01-17``: it downloads the dated release file,
refuses any bytes other than the pinned sha256 and size, and writes the manifest through
``deposit_assembly_set``. ``torchcell.sequence.genome.standard.load_genome`` then places
the pinned copy at ``<go_root>/go.obo`` before it constructs the yeast genome.

Run once per canonical tier, from the repository root::

    python scripts/provision_go_release_20240117.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx

from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.registry import (
    GO_RELEASE_20240117,
    ROLE_ONTOLOGY,
    GenomeManifest,
    assembly_set_dir,
    deposit_assembly_set,
)

GO_OBO = "go.obo"
GO_OBO_URL = "https://release.geneontology.org/2024-01-17/ontology/go.obo"
#: Measured on 2026-11-19: the dated release file and the copy cached on GilaHyper
#: since 2024-03-15 (``$DATA_ROOT/data/go/go.obo``) are the same 34,420,328 bytes.
GO_OBO_SHA256 = "ad2f050219a579bc7cb448dc2fac5378b3b6b38fa857ab0b4e8ded7d270aa9ff"
GO_OBO_BYTES = 34_420_328


def main() -> None:
    """Download, verify against the pin, and deposit the set."""
    directory = Path(assembly_set_dir(GO_RELEASE_20240117))
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / GO_OBO
    with httpx.stream("GET", GO_OBO_URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in r.iter_bytes(1 << 20):
                handle.write(chunk)
    got = sha256_file(dest)
    size = dest.stat().st_size
    if (got, size) != (GO_OBO_SHA256, GO_OBO_BYTES):
        dest.unlink()
        raise RuntimeError(
            f"{GO_OBO_URL}: {size} bytes sha256 {got}, pinned {GO_OBO_BYTES} bytes "
            f"sha256 {GO_OBO_SHA256}; nothing deposited"
        )
    now = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = GenomeManifest(
        assembly_set=GO_RELEASE_20240117,
        organism="not applicable (organism-agnostic ontology)",
        strain_or_population="not applicable",
        source="Gene Ontology Consortium",
        release="2024-01-17",
        source_url="https://release.geneontology.org/2024-01-17/ontology/",
        files=[
            ArtifactRecord(
                path=GO_OBO,
                role=ROLE_ONTOLOGY,
                bytes=size,
                sha256=got,
                source=GO_OBO_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=GO_OBO_URL,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": GO_OBO_URL},
                    sha256=got,
                    retrieved_at=now,
                ),
            )
        ],
        provenance_complete=True,
        created_at=now,
        notes=(
            "go.obo of the dated GO Consortium release 2024-01-17: the file "
            "SCerevisiaeGenome has read from <go_root>/go.obo on every machine since "
            "2024-03-15, when the class downloaded the then-current file. Pinned so a "
            "new machine reads the same ontology; go_release_2026-08-05 (go-basic.obo) "
            "is the bacterial genomes' file and is unrelated."
        ),
    )
    print(deposit_assembly_set(manifest))


if __name__ == "__main__":
    main()
