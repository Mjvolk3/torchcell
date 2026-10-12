# scripts/provision_yeast_genomes.py
# [[scripts.provision_yeast_genomes]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/provision_yeast_genomes.py

"""Deposit an NCBI yeast assembly set into the genomes tier.

The yeast counterpart of ``scripts/provision_bacterial_genomes.py``, for a strain the
SGD release does not cover. First set, for the Zhang 2020 row (#990):

- ``scerevisiae_CENPK113-7D_ASM257140v2``: NCBI ``GCA_002571405.2`` (ASM257140v2), the
  CEN.PK113-7D reference. Five members: the GenBank flat file (the primary ingest), the
  GenBank GFF3, the replicon FASTA, the proteins, and the assembly report, which is the
  replicon table a multi-chromosome genome class needs. No GO file: no GO release is
  keyed to this assembly's ``SCEN_*`` locus tags, so the set carries none and a genome
  class reading it reports GO as absent rather than borrowing S288C's.

Integrity is NCBI's own: every member is checked against the md5 its assembly
directory publishes in ``md5checksums.txt``, which is fetched with the members and is
not itself deposited. The sha256 the manifest pins is computed from the bytes that
passed that check. Nothing here is hashed from a value typed into this file, so there
is no drift table to keep: a changed upstream file fails its md5 check and stops the
run.

Steps, each halting the run before anything reaches the tier:

  (a) fetch: every member and the directory's ``md5checksums.txt`` through the
      ``direct_url`` retriever into ``--refetch-dir/<assembly_set>/``, skipped when the
      file is already there and still matches its ``<file>.retrieval.json`` sidecar;
  (b) md5: every member against the published listing; a mismatch stops the run;
  (c) manifest: one ``GenomeManifest``, printed; ``--dry-run`` stops here;
  (d) deposit: ``shutil.copy2`` into the set directory (an existing file stops the
      run), a re-hash against the fetch, ``deposit_assembly_set`` (which re-hashes
      again and refuses an existing manifest), then ``verify_assembly_set``.

Usage::

    PYTHONPATH=$WT python scripts/provision_yeast_genomes.py --refetch-dir $R --dry-run
    PYTHONPATH=$WT python scripts/provision_yeast_genomes.py --refetch-dir $R
    PYTHONPATH=$WT python scripts/provision_yeast_genomes.py --refetch-dir $R \
        --set scerevisiae_CENPK113-7D_ASM257140v2
"""

from __future__ import annotations

import argparse
import hashlib
import os.path as osp
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.retrieve import RETRIEVERS
from torchcell.sequence.genome.registry import (
    ROLE_ANNOTATION,
    ROLE_INDEX,
    ROLE_SEQUENCE,
    SCEREVISIAE_CENPK113_7D,
    GenomeManifest,
    assembly_set_dir,
    deposit_assembly_set,
    verify_assembly_set,
)

DIRECT_URL = "torchcell.literature.retrieve.direct_url"
NCBI_ALL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
MD5_LISTING = "md5checksums.txt"


class YeastAssemblySpec(BaseModel):
    """One NCBI yeast assembly set: where its files come from and what they are."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    organism: str
    strain: str
    source: str
    release: str
    accession: str = Field(description='e.g. "GCA_002571405.2".')
    assembly_name: str = Field(description='e.g. "ASM257140v2".')
    notes: str

    @property
    def stem(self) -> str:
        """The prefix every member file of an NCBI assembly directory carries."""
        return f"{self.accession}_{self.assembly_name}"

    @property
    def directory_url(self) -> str:
        """The assembly's own NCBI directory, where its md5 listing lives."""
        digits = self.accession.split("_")[1].split(".")[0]
        parts = "/".join(digits[i : i + 3] for i in (0, 3, 6))
        return f"{NCBI_ALL}/{self.accession[:3]}/{parts}/{self.stem}"

    @property
    def members(self) -> dict[str, str]:
        """Member file name to its role in the set."""
        return {
            f"{self.stem}_genomic.gbff.gz": ROLE_ANNOTATION,
            f"{self.stem}_genomic.gff.gz": ROLE_ANNOTATION,
            f"{self.stem}_genomic.fna.gz": ROLE_SEQUENCE,
            f"{self.stem}_protein.faa.gz": ROLE_SEQUENCE,
            f"{self.stem}_assembly_report.txt": ROLE_INDEX,
        }


CENPK113_7D = YeastAssemblySpec(
    assembly_set=SCEREVISIAE_CENPK113_7D,
    organism="Saccharomyces cerevisiae",
    strain="CEN.PK113-7D",
    source="NCBI GenBank (ASM257140v2)",
    release="2017-10 (GCA_002571405.2)",
    accession="GCA_002571405.2",
    assembly_name="ASM257140v2",
    notes=(
        "The CEN.PK113-7D reference the Zhang 2020 platform strain (CEN.PK2-1C) is "
        "written against (#990). Chromosome-level, haploid, 17 replicons by its own "
        'assembly report; the GenBank flat file states /strain="CEN.PK113-7D; CBS '
        '8340" and its locus tags are the SCEN_* namespace, which is the identifier '
        "space a genotype written against this assembly uses. No GO file is deposited: "
        "no GO release is keyed to SCEN_* tags, so GO is absent for this set rather "
        "than borrowed from S288C."
    ),
)

SPECS: dict[str, YeastAssemblySpec] = {CENPK113_7D.assembly_set: CENPK113_7D}


def md5_file(path: Path) -> str:
    """Streaming md5 of a file, the digest form NCBI's listing publishes."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> RetrievalRecord:
    """Fetch ``url`` to ``dest`` through the recorded retriever, or reuse the bytes.

    A file already present with its ``<file>.retrieval.json`` sidecar is reused when
    its sha256 still matches that record; a file whose bytes have changed stops the run
    rather than being silently re-fetched.
    """
    sidecar = dest.with_suffix(dest.suffix + ".retrieval.json")
    if dest.exists() and sidecar.exists():
        record = RetrievalRecord.model_validate_json(sidecar.read_text())
        if sha256_file(dest) != record.sha256:
            raise RuntimeError(f"{dest} no longer matches its recorded sha256")
        return record
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(RETRIEVERS[DIRECT_URL](url))
    record = RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=url,
        retriever=DIRECT_URL,
        params={"url": url},
        sha256=sha256_file(dest),
        retrieved_at=datetime.now(UTC).date().isoformat(),
    )
    sidecar.write_text(record.model_dump_json(indent=2))
    return record


def published_md5s(listing: Path) -> dict[str, str]:
    """The assembly directory's own ``md5checksums.txt``, keyed by member name."""
    out: dict[str, str] = {}
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split()
        out[path.lstrip("./")] = digest
    return out


def build_manifest(spec: YeastAssemblySpec, staged: Path) -> GenomeManifest:
    """Fetch and md5-check every member, then the manifest that pins their sha256s."""
    listing = staged / MD5_LISTING
    fetch(f"{spec.directory_url}/{MD5_LISTING}", listing)
    published = published_md5s(listing)
    files: list[ArtifactRecord] = []
    for name, role in spec.members.items():
        path = staged / name
        record = fetch(f"{spec.directory_url}/{name}", path)
        expected = published.get(name)
        if expected is None:
            raise RuntimeError(f"{name} is not in the directory's {MD5_LISTING}")
        measured = md5_file(path)
        if measured != expected:
            raise RuntimeError(
                f"{name}: md5 {measured} does not match the published {expected}"
            )
        files.append(
            ArtifactRecord(
                path=name,
                role=role,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source=f"{spec.directory_url}/{name}",
                retrieval=record,
            )
        )
    return GenomeManifest(
        assembly_set=spec.assembly_set,
        organism=spec.organism,
        strain_or_population=spec.strain,
        source=spec.source,
        release=spec.release,
        source_url=spec.directory_url,
        files=files,
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
        notes=spec.notes,
    )


def deposit(manifest: GenomeManifest, staged: Path, data_root: str | None) -> str:
    """Copy every member into the set directory, then deposit and verify the manifest."""
    target = Path(assembly_set_dir(manifest.assembly_set, data_root))
    target.mkdir(parents=True, exist_ok=True)
    for record in manifest.files:
        dest = target / record.path
        if dest.exists():
            raise RuntimeError(f"{dest} already exists; refusing to deposit over it")
        shutil.copy2(staged / record.path, dest)
        if sha256_file(dest) != record.sha256:
            raise RuntimeError(f"{dest} does not match the sha256 of the fetch")
    path = deposit_assembly_set(manifest, data_root)
    verify_assembly_set(manifest.assembly_set, data_root)
    return path


def main(argv: list[str] | None = None) -> int:
    """Provision the named sets (every set by default); ``--dry-run`` stops at (c)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refetch-dir", required=True)
    parser.add_argument("--set", action="append", dest="sets", default=[])
    parser.add_argument("--data-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    names = args.sets or sorted(SPECS)
    unknown = [name for name in names if name not in SPECS]
    if unknown:
        print(f"unknown set(s): {unknown}; known: {sorted(SPECS)}", file=sys.stderr)
        return 2
    for name in names:
        spec = SPECS[name]
        staged = Path(args.refetch_dir) / name
        manifest = build_manifest(spec, staged)
        print(f"{name}: {len(manifest.files)} member(s), every md5 matched NCBI")
        for record in manifest.files:
            print(f"  {record.path:58} {record.role:10} {record.sha256[:16]}")
        if args.dry_run:
            print("  --dry-run: nothing deposited")
            continue
        path = deposit(manifest, staged, args.data_root)
        print(f"  deposited and verified: {osp.dirname(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
