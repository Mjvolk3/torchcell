# scripts/provision_pkudriavzevii_genome.py
# [[scripts.provision_pkudriavzevii_genome]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/provision_pkudriavzevii_genome.py

"""Deposit the Pichia kudriavzevii SD108 v1 genome into the genomes tier.

The set ``pkudriavzevii_SD108_v1`` is the group's curated SD108 assembly (Issatchenkia
orientalis), released as the private GitHub repository
``Aurosish/I.orientalis_SD108_Genome_Reconstruction``. Every member is taken from the
repository at the pinned commit :data:`COMMIT`, read out of that commit's tree (``git
show <commit>:<path>``), and deposited flat under its basename. The repository is
private, so a rebuild without the owner's ``gh`` authentication reads the deposited
bytes; the retrieval record still names the command that produced them.

Integrity has two anchors, each checked before anything reaches the tier:

- the repository's own ``Assembly_Build/final/MD5SUMS`` (the release's integrity
  listing, the counterpart of NCBI's ``md5checksums.txt``): ``md5sum -c`` over the
  checkout, and the md5 of every member that listing names, as read from the commit;
- the git object id: every member's bytes must hash (as a git blob) to the id the
  pinned commit's tree records for its path, which ties each file to the commit.

Steps, each halting the run before anything reaches the tier:

  (a) checkout: ``gh repo clone`` + fetch + checkout of :data:`COMMIT` into
      ``--refetch-dir/<assembly_set>/repo``, skipped when a clone already sits at that
      commit (a clone at another commit stops the run);
  (b) integrity: ``md5sum -c MD5SUMS`` in ``Assembly_Build/final``, then the md5 and
      git-blob checks above for every member, which are staged under
      ``--refetch-dir/<assembly_set>/staged/``;
  (c) manifest: one ``GenomeManifest``, printed; ``--dry-run`` stops here;
  (d) deposit: ``shutil.copy2`` into the set directory (an existing file stops the
      run), a re-hash, ``deposit_assembly_set`` (which re-hashes again and refuses an
      existing manifest), then ``verify_assembly_set``.

Usage::

    PYTHONPATH=$WT python scripts/provision_pkudriavzevii_genome.py --refetch-dir $R --dry-run
    PYTHONPATH=$WT python scripts/provision_pkudriavzevii_genome.py --refetch-dir $R
"""

from __future__ import annotations

import argparse
import hashlib
import os.path as osp
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.retrieve import (
    RETRIEVERS,
    git_checkout,
    git_clone_commands,
    git_show,
)
from torchcell.sequence.genome.registry import (
    PKUDRIAVZEVII_SD108,
    ROLE_ANNOTATION,
    ROLE_INDEX,
    ROLE_SEQUENCE,
    GenomeManifest,
    assembly_set_dir,
    deposit_assembly_set,
    verify_assembly_set,
)

GIT_REPOSITORY: Final = "torchcell.literature.retrieve.git_repository"
REPO: Final = "Aurosish/I.orientalis_SD108_Genome_Reconstruction"
COMMIT: Final = "8ca6a106809d6b708d4ba621e3834649dd122720"
REPO_URL: Final = f"https://github.com/{REPO}"
FINAL: Final = "Assembly_Build/final"
MD5_LISTING: Final = f"{FINAL}/MD5SUMS"


class Member(BaseModel):
    """One deposited file: where it is in the repository and the role it plays."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_in_repo: str
    role: str

    @property
    def name(self) -> str:
        """The deposited name: the basename (no member of this set collides)."""
        return osp.basename(self.path_in_repo)


#: Every member of the set, in deposit order. The haplotype-2 basenames all carry a
#: ``_hap2`` part already, so none collides with a primary member and none is renamed.
MEMBERS: Final[tuple[Member, ...]] = (
    Member(path_in_repo=f"{FINAL}/SD108_genome_v1.fasta", role=ROLE_SEQUENCE),
    Member(path_in_repo=f"{FINAL}/SD108_genome_v1.fasta.fai", role=ROLE_INDEX),
    Member(path_in_repo=f"{FINAL}/SD108_genome_v1.sources.tsv", role=ROLE_INDEX),
    Member(
        path_in_repo=f"{FINAL}/annotation/SD108_genome_v1.gff3", role=ROLE_ANNOTATION
    ),
    Member(path_in_repo=f"{FINAL}/annotation/proteins.faa", role=ROLE_SEQUENCE),
    Member(path_in_repo=f"{FINAL}/annotation/cds.fna", role=ROLE_SEQUENCE),
    Member(path_in_repo=f"{FINAL}/annotation/transcripts.fna", role=ROLE_SEQUENCE),
    Member(path_in_repo=f"{FINAL}/annotation/gene_xref.tsv", role=ROLE_ANNOTATION),
    Member(path_in_repo=f"{FINAL}/annotation/NOMENCLATURE.md", role=ROLE_ANNOTATION),
    Member(path_in_repo=MD5_LISTING, role=ROLE_INDEX),
    Member(path_in_repo="ID_mappings/SD108_to_NCBI_JL09_map.csv", role=ROLE_ANNOTATION),
    Member(
        path_in_repo="Haplotype2_Build/final/SD108_genome_v1_hap2.fasta",
        role=ROLE_SEQUENCE,
    ),
    Member(
        path_in_repo=(
            "Haplotype2_Build/final/annotation_hap2/SD108_genome_v1_hap2.gff3"
        ),
        role=ROLE_ANNOTATION,
    ),
    Member(
        path_in_repo="Haplotype2_Build/final/annotation_hap2/proteins_hap2.faa",
        role=ROLE_SEQUENCE,
    ),
    Member(
        path_in_repo="Haplotype2_Build/final/annotation_hap2/cds_hap2.fna",
        role=ROLE_SEQUENCE,
    ),
)

NOTES: Final = (
    "Pichia kudriavzevii (Issatchenkia orientalis) SD108, genome v1: the group's "
    "curated diploid-strain reference (one haploid copy: SD108_chr1..5, the circular "
    "SD108_mito and the unplaced SD108_rDNA_unit; 10.87 Mb), its annotation (5,159 "
    "protein-coding genes = 5,144 nuclear + 15 mitochondrial, 199 tRNA, 18 rRNA), the "
    "haplotype-2 copy of chr1..5 and the SD108-to-NCBI JL09 gene map. Gene ids are "
    "the assembly's own SD108_X<seq><type><number> namespace (NOMENCLATURE.md); S. "
    "cerevisiae names are attributes, never identifiers. No GO file: no GO release is "
    "keyed to SD108 ids. Retrieved from the private repository "
    f"{REPO} at commit {COMMIT} with: "
    + " && ".join(" ".join(c) for c in git_clone_commands(REPO, COMMIT, "<dir>"))
    + ". Integrity: md5sum -c of the repository's Assembly_Build/final/MD5SUMS, and "
    "every member's git blob id against the pinned commit's tree."
)


def md5_bytes(data: bytes) -> str:
    """md5 of a byte string, the digest form ``MD5SUMS`` lists."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def git_blob_id(data: bytes) -> str:
    """The git object id of ``data`` as a blob (``git hash-object``)."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def tree_blob_id(directory: str, path_in_repo: str) -> str:
    """The blob id the pinned commit's tree records for ``path_in_repo``."""
    return subprocess.run(
        ["git", "-C", directory, "rev-parse", f"{COMMIT}:{path_in_repo}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkout(clone: Path) -> None:
    """Step (a): a clone of :data:`REPO` at :data:`COMMIT`, made or reused."""
    if clone.exists():
        head = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != COMMIT:
            raise RuntimeError(f"{clone} is at {head}, not the pinned {COMMIT}")
        return
    clone.parent.mkdir(parents=True, exist_ok=True)
    git_checkout(REPO, COMMIT, str(clone))


def published_md5s(listing: bytes) -> dict[str, str]:
    """``MD5SUMS`` keyed by repository path (its paths are relative to ``final/``)."""
    out: dict[str, str] = {}
    for line in listing.decode().splitlines():
        if line.strip():
            digest, path = line.split()
            out[f"{FINAL}/{path}"] = digest
    return out


def build_manifest(clone: Path, staged: Path) -> GenomeManifest:
    """Step (b) and (c): check and stage every member, then the manifest."""
    subprocess.run(
        ["md5sum", "-c", "--quiet", "MD5SUMS"], cwd=clone / FINAL, check=True
    )
    published = published_md5s(git_show(str(clone), COMMIT, MD5_LISTING))
    staged.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(UTC).date().isoformat()
    files: list[ArtifactRecord] = []
    names = [m.name for m in MEMBERS]
    if len(set(names)) != len(names):
        raise RuntimeError(f"member basenames collide: {sorted(names)}")
    for member in MEMBERS:
        data = git_show(str(clone), COMMIT, member.path_in_repo)
        expected_blob = tree_blob_id(str(clone), member.path_in_repo)
        if git_blob_id(data) != expected_blob:
            raise RuntimeError(f"{member.path_in_repo}: not blob {expected_blob}")
        listed = published.get(member.path_in_repo)
        if listed is not None and md5_bytes(data) != listed:
            raise RuntimeError(
                f"{member.path_in_repo}: md5 {md5_bytes(data)} != MD5SUMS {listed}"
            )
        dest = staged / member.name
        dest.write_bytes(data)
        sha = sha256_file(dest)
        files.append(
            ArtifactRecord(
                path=member.name,
                role=member.role,
                bytes=dest.stat().st_size,
                sha256=sha,
                source=f"{REPO_URL}/blob/{COMMIT}/{member.path_in_repo}",
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.git_repository,
                    source_url=REPO_URL,
                    retriever=GIT_REPOSITORY,
                    params={
                        "repo": REPO,
                        "commit": COMMIT,
                        "path_in_repo": member.path_in_repo,
                    },
                    sha256=sha,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    return GenomeManifest(
        assembly_set=PKUDRIAVZEVII_SD108,
        organism="Pichia kudriavzevii",
        strain_or_population="SD108",
        source=f"{REPO} (GitHub, private) at commit {COMMIT}",
        release="v1",
        citation_key=None,
        source_url=f"{REPO_URL}/tree/{COMMIT}",
        files=files,
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
        notes=NOTES,
    )


def deposit(manifest: GenomeManifest, staged: Path, data_root: str | None) -> str:
    """Step (d): copy every member into the set directory, deposit, verify."""
    target = Path(assembly_set_dir(manifest.assembly_set, data_root))
    target.mkdir(parents=True, exist_ok=True)
    for record in manifest.files:
        dest = target / record.path
        if dest.exists():
            raise RuntimeError(f"{dest} already exists; refusing to deposit over it")
        shutil.copy2(staged / record.path, dest)
        if sha256_file(dest) != record.sha256:
            raise RuntimeError(f"{dest} does not match the sha256 of the checkout")
    path = deposit_assembly_set(manifest, data_root)
    verify_assembly_set(manifest.assembly_set, data_root)
    return path


def main(argv: list[str] | None = None) -> int:
    """Provision the set; ``--dry-run`` stops after printing the manifest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refetch-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if GIT_REPOSITORY not in RETRIEVERS:
        raise RuntimeError(f"{GIT_REPOSITORY} is not a registered retriever")
    root = Path(args.refetch_dir) / PKUDRIAVZEVII_SD108
    clone = root / "repo"
    checkout(clone)
    manifest = build_manifest(clone, root / "staged")
    print(
        f"{manifest.assembly_set}: {len(manifest.files)} member(s) from {REPO} at "
        f"{COMMIT}; md5sum -c and every git blob id matched; provenance_complete="
        f"{manifest.provenance_complete}"
    )
    for record in manifest.files:
        print(f"  {record.path:36} {record.role:10} {record.bytes:>10} {record.sha256}")
    if args.dry_run:
        print("  --dry-run: nothing deposited")
        return 0
    path = deposit(manifest, root / "staged", args.data_root)
    print(f"  deposited and verified: {osp.dirname(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
