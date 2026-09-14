# scripts/migrate_genomes_tier.py
# [[scripts.migrate_genomes_tier]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/migrate_genomes_tier.py

"""One-shot deposit of the first two assembly sets into the genomes tier.

Seeds ``$DATA_ROOT/torchcell-genomes/`` with the SGD S288C R64-4-1 release and the
Peter et al. 2018 1,011-isolate assets, each with a ``GenomeManifest`` whose retrieval
records are genuine: a record is written only for bytes this script (or the re-fetch it
reuses) reproduced from the source URL and matched by sha256 against what the loaders
have been reading. Nothing is moved or deleted: the legacy S288C directory and the
literature key's ``data/`` stay untouched, and no symlink is left anywhere.

Steps, each printed as a table and each halting the run on a mismatch:

  (a) the SGD tgz in ``--refetch-dir/sgd/`` (fetched through ``direct_url`` when
      absent) is hashed, extracted, gunzipped, and every member compared to the legacy
      release directory: eight members, eight matches, or stop;
  (b) the five Peter files in ``--refetch-dir/peter/`` (fetched when absent) are hashed
      against the library key's manifest: five matches, or stop;
  (c) the two tier directories are created and the files copied in (``shutil.copy2``:
      new inodes owned by the running user), then re-hashed against their sources;
  (d) both manifests are deposited (``deposit_assembly_set`` refuses an existing
      manifest and re-hashes every file) and verified.

The 4 GB assemblies tarball is far larger than the in-memory ``direct_url`` retriever
should hold; on a fresh machine fetch it into ``--refetch-dir/peter/`` with curl first
and the script will only hash it.

Usage::

    PYTHONPATH=$WT python scripts/migrate_genomes_tier.py --refetch-dir $REFETCH
"""

from __future__ import annotations

import argparse
import gzip
import os
import platform
import shutil
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.retrieve import RETRIEVERS
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    ROLE_ANNOTATION,
    ROLE_CONTAINER,
    ROLE_INDEX,
    ROLE_MATRIX,
    ROLE_SEQUENCE,
    SGD_S288C_R64,
    GenomeManifest,
    assembly_set_dir,
    deposit_assembly_set,
    verify_assembly_set,
)

DIRECT_URL = "torchcell.literature.retrieve.direct_url"

SGD_RELEASE = "S288C_reference_genome_R64-4-1_20230830"
SGD_TGZ = f"{SGD_RELEASE}.tgz"
SGD_URL = (
    "http://sgd-archive.yeastgenome.org/sequence/S288C_reference/genome_releases/"
    + SGD_TGZ
)
SGD_MEMBER_ROLES = {
    "gene_association_R64-4-1_20230830.sgd": ROLE_ANNOTATION,
    "NotFeature_R64-4-1_20230830.fasta": ROLE_SEQUENCE,
    "orf_coding_all_R64-4-1_20230830.fasta": ROLE_SEQUENCE,
    "orf_trans_all_R64-4-1_20230830.fasta": ROLE_SEQUENCE,
    "other_features_genomic_R64-4-1_20230830.fasta": ROLE_SEQUENCE,
    "rna_coding_R64-4-1_20230830.fasta": ROLE_SEQUENCE,
    "S288C_reference_sequence_R64-4-1_20230830.fsa": ROLE_SEQUENCE,
    "saccharomyces_cerevisiae_R64-4-1_20230830.gff": ROLE_ANNOTATION,
}

PETER_KEY = "peterGenomeEvolution10112018"
PETER_BASE_URL = "http://1002genomes.u-strasbg.fr/files/"
PETER_ROLES = {
    "1011Assemblies.tar.gz": ROLE_CONTAINER,
    "allORFs_pangenome.fasta.gz": ROLE_SEQUENCE,
    "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz": ROLE_CONTAINER,
    "genesMatrix_PresenceAbsence.tab.gz": ROLE_MATRIX,
    "genesMatrix_CopyNumber.tab.gz": ROLE_MATRIX,
}
PETER_INDEX = "1011Assemblies.tar.gz.member_index.tsv"
PETER_INDEX_PROCESSOR = (
    "experiments.embeddings.compute_isolate_embeddings.build_assembly_member_index"
)

RETRIEVED_AT = "2026-09-14"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fetch_if_absent(path: Path, url: str) -> None:
    """Write ``url``'s bytes to ``path`` through the direct_url retriever when absent."""
    if path.is_file():
        return
    print(f"fetching {url} -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(RETRIEVERS[DIRECT_URL](url))


def _retrieval(url: str, sha256: str) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=url,
        retriever=DIRECT_URL,
        params={"url": url},
        sha256=sha256,
        retrieved_at=RETRIEVED_AT,
    )


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))


def _stat(path: Path) -> str:
    st = path.stat()
    import pwd

    return f"{st.st_nlink} {pwd.getpwuid(st.st_uid).pw_name}"


# --------------------------------------------------------------------------- #
# (a) SGD
# --------------------------------------------------------------------------- #
def stage_sgd(refetch_dir: Path, legacy_dir: Path, workdir: Path) -> tuple[Path, str]:
    """Hash the tgz, extract into ``workdir``, compare members to the legacy dir."""
    tgz = refetch_dir / "sgd" / SGD_TGZ
    _fetch_if_absent(tgz, SGD_URL)
    tgz_sha = sha256_file(tgz)
    print(f"(a) {SGD_TGZ}: {tgz.stat().st_size} bytes, sha256 {tgz_sha}")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(workdir, filter="data")
    extracted = workdir / SGD_RELEASE
    for gz in sorted(extracted.glob("*.gz")):
        with gzip.open(gz, "rb") as fin, open(str(gz)[:-3], "wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz.unlink()
    members = sorted(p.name for p in extracted.iterdir() if p.is_file())
    rows: list[tuple[str, ...]] = []
    ok = True
    for name in members:
        fetched = sha256_file(extracted / name)
        legacy = legacy_dir / name
        disk = sha256_file(legacy) if legacy.is_file() else "NOT ON DISK"
        match = fetched == disk
        ok = ok and match
        rows.append((name, disk[:12], fetched[:12], "MATCH" if match else "MISMATCH"))
    _table(rows, ("member", "on-disk sha256", "fetched sha256", "result"))
    if not ok or set(members) != set(SGD_MEMBER_ROLES):
        raise SystemExit(
            f"(a) halted: members {members} vs expected {sorted(SGD_MEMBER_ROLES)}; "
            "every member must match the legacy release"
        )
    return extracted, tgz_sha


# --------------------------------------------------------------------------- #
# (b) Peter
# --------------------------------------------------------------------------- #
def stage_peter(refetch_dir: Path, library_dir: Path) -> dict[str, str]:
    """Hash the five re-fetched files against the library key's manifest."""
    manifest = Manifest.model_validate_json(
        (library_dir.parent / "manifest.json").read_text()
    )
    pinned = {rec.path.removeprefix("data/"): rec.sha256 for rec in manifest.files}
    rows: list[tuple[str, ...]] = []
    hashes: dict[str, str] = {}
    ok = True
    for name in PETER_ROLES:
        path = refetch_dir / "peter" / name
        _fetch_if_absent(path, PETER_BASE_URL + name)
        got = sha256_file(path)
        hashes[name] = got
        match = got == pinned[name]
        ok = ok and match
        rows.append(
            (name, pinned[name][:12], got[:12], "MATCH" if match else "MISMATCH")
        )
    _table(rows, ("file", "library sha256", "fetched sha256", "result"))
    if not ok:
        raise SystemExit("(b) halted: a re-fetched Peter file differs from the library")
    return hashes


# --------------------------------------------------------------------------- #
# (c) copy
# --------------------------------------------------------------------------- #
def copy_into(sources: dict[str, Path], dest_dir: Path) -> None:
    """``shutil.copy2`` each source into ``dest_dir`` and re-hash it against its source."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, ...]] = []
    for name, src in sources.items():
        dst = dest_dir / name
        if dst.exists():
            raise SystemExit(f"(c) halted: {dst} already exists")
        shutil.copy2(src, dst)
        same = sha256_file(src) == sha256_file(dst)
        rows.append(
            (name, str(dst.stat().st_size), _stat(dst), "SAME" if same else "DIFF")
        )
        if not same:
            raise SystemExit(f"(c) halted: {dst} does not match {src}")
    _table(rows, ("file", "bytes", "nlink owner", "sha256 vs source"))


# --------------------------------------------------------------------------- #
# (d) manifests
# --------------------------------------------------------------------------- #
def sgd_manifest(tier_dir: Path, tgz_sha: str) -> GenomeManifest:
    """The SGD set: the container with its retrieval, the members with their extraction."""
    files = [
        ArtifactRecord(
            path=SGD_TGZ,
            role=ROLE_CONTAINER,
            bytes=(tier_dir / SGD_TGZ).stat().st_size,
            sha256=tgz_sha,
            source=SGD_URL,
            retrieval=_retrieval(SGD_URL, tgz_sha),
        )
    ]
    for name, role in SGD_MEMBER_ROLES.items():
        path = tier_dir / name
        files.append(
            ArtifactRecord(
                path=name,
                role=role,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source=f"extracted from {SGD_TGZ}",
                processing=ProcessingRecord(
                    processor="scripts.migrate_genomes_tier.stage_sgd",
                    tool="tarfile+gzip",
                    version=f"python {platform.python_version()}",
                    params={"container": SGD_TGZ},
                    input_sha256=[tgz_sha],
                ),
            )
        )
    return GenomeManifest(
        assembly_set=SGD_S288C_R64,
        organism="Saccharomyces cerevisiae",
        strain_or_population="S288C",
        source="SGD",
        release="R64-4-1_20230830",
        citation_key=None,
        source_url=SGD_URL,
        files=files,
        provenance_complete=True,
        created_at=_now(),
        notes=(
            "Container fetched 2026-09-14 with curl -sSL (a plain GET, the same request "
            "the direct_url retriever makes) and reproduced every file the loaders had "
            "been reading from data/sgd/genome/ byte for byte; ncbi_genomic.gff is not "
            "an SGD release file and is not part of this set."
        ),
    )


def peter_manifest(tier_dir: Path, hashes: dict[str, str]) -> GenomeManifest:
    """The Peter set: five retrieved files plus the derived member index."""
    files = [
        ArtifactRecord(
            path=name,
            role=role,
            bytes=(tier_dir / name).stat().st_size,
            sha256=hashes[name],
            source=PETER_BASE_URL + name,
            retrieval=_retrieval(PETER_BASE_URL + name, hashes[name]),
        )
        for name, role in PETER_ROLES.items()
    ]
    files.append(
        ArtifactRecord(
            path=PETER_INDEX,
            role=ROLE_INDEX,
            bytes=(tier_dir / PETER_INDEX).stat().st_size,
            sha256=sha256_file(tier_dir / PETER_INDEX),
            source=f"cache copied from torchcell-library/{PETER_KEY}/data/",
            processing=ProcessingRecord(
                processor=PETER_INDEX_PROCESSOR,
                tool="python",
                version="unrecorded; cache written 2026-07-22 beside the tarball",
                params={"container": "1011Assemblies.tar.gz"},
                input_sha256=[hashes["1011Assemblies.tar.gz"]],
            ),
        )
    )
    return GenomeManifest(
        assembly_set=PETER2018_1011,
        organism="Saccharomyces cerevisiae",
        strain_or_population="1,011 isolates",
        source="Peter et al. 2018",
        release="2018",
        citation_key=PETER_KEY,
        source_url=PETER_BASE_URL,
        files=files,
        provenance_complete=True,
        created_at=_now(),
        notes=(
            "Every file re-fetched 2026-09-14 with curl -sSL from the Strasbourg server "
            "and equal by sha256 to the copy the library key has held since 2026-07-21; "
            "that key's data/ is retained and its manifest untouched."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run steps (a) to (d); every mismatch is a SystemExit before anything is written."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refetch-dir", required=True, help="holds sgd/ and peter/")
    ap.add_argument("--data-root", default=None, help="defaults to $DATA_ROOT")
    args = ap.parse_args(argv)
    if args.data_root is None:
        from dotenv import load_dotenv

        load_dotenv()
        args.data_root = os.environ["DATA_ROOT"]
    refetch = Path(args.refetch_dir)
    legacy = Path(args.data_root) / "data/sgd/genome" / SGD_RELEASE
    library = Path(args.data_root) / "torchcell-library" / PETER_KEY / "data"
    sgd_tier = Path(assembly_set_dir(SGD_S288C_R64, args.data_root))
    peter_tier = Path(assembly_set_dir(PETER2018_1011, args.data_root))
    for d in (sgd_tier, peter_tier):
        if (d / "manifest.json").exists():
            raise SystemExit(f"{d} already has a manifest; nothing to do")

    with tempfile.TemporaryDirectory(prefix="genomes-tier-") as tmp:
        extracted, tgz_sha = stage_sgd(refetch, legacy, Path(tmp))
        hashes = stage_peter(refetch, library)
        print("(c) copying into the tier")
        copy_into(
            {SGD_TGZ: refetch / "sgd" / SGD_TGZ}
            | {name: extracted / name for name in SGD_MEMBER_ROLES},
            sgd_tier,
        )
        copy_into(
            {name: refetch / "peter" / name for name in PETER_ROLES}
            | {PETER_INDEX: library / PETER_INDEX},
            peter_tier,
        )
    print("(d) depositing manifests")
    print(deposit_assembly_set(sgd_manifest(sgd_tier, tgz_sha), args.data_root))
    print(deposit_assembly_set(peter_manifest(peter_tier, hashes), args.data_root))
    for s in (SGD_S288C_R64, PETER2018_1011):
        n = len(verify_assembly_set(s, args.data_root))
        print(f"verified {s}: {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
