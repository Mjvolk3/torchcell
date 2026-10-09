# torchcell/sequence/genome/registry.py
# [[torchcell.sequence.genome.registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/registry.py
# Test file: tests/torchcell/sequence/genome/test_registry.py

"""The genomes tier: sequence-level reference data that is NOT stored in the graph.

``$DATA_ROOT/torchcell-genomes/<assembly_set>/`` holds one flat directory per assembly
set (a reference release, a panel of isolate assemblies), each with a pydantic
``manifest.json`` that pins every file by sha256 and records how it was retrieved. The
graph stores pointers (an assembly-set id, a member path, a sha256); this module is the
one place those pointers are dereferenced to a file on disk, and it verifies the sha256
on every resolve.

There is deliberately no fallback to the legacy locations (``data/sgd/genome``, the
literature mirror): a machine without the tier fails at ``resolve`` with the rsync
command that seeds it, rather than running on bytes nothing has verified.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import ArtifactRecord, sha256_file

GENOMES_DIR = "torchcell-genomes"
MANIFEST_FILENAME = "manifest.json"
GENOME_MANIFEST_VERSION = 1

#: The SGD R64-4-1 (2023-08-30) S288C reference release, as fetched from the SGD archive.
SGD_S288C_R64 = "sgd_S288C_R64-4-1_20230830"
#: Peter et al. 2018: the 1,011 isolate assemblies and the pangenome matrices.
PETER2018_1011 = "peter2018_1011_assemblies"
#: E. coli K-12 MG1655, NCBI assembly ASM584v2 (GCA_000005845.2 with its RefSeq
#: annotation GCF_000005845.2): b-number locus tags, plus the GO Consortium
#: ``ECOLI-uniprot.gaf.gz`` keyed to b-numbers.
ECOLI_K12_MG1655 = "ecoli_K12_MG1655_ASM584v2"
#: E. coli K-12 BW25113 (the Keio background), NCBI assembly ASM75055v1
#: (GCA_000750555.1 / GCF_000750555.1). One set per strain, never one per host: a
#: ``BW25113_`` number is not an MG1655 b-number. Its own GO is the RefSeq GFF's
#: inline terms and NCBI's GAF; MG1655's GAF is referenced by the ``ECOLI_K12_MG1655``
#: id, never copied, and which source a loader reads is still open.
ECOLI_K12_BW25113 = "ecoli_K12_BW25113_ASM75055v1"
#: E. coli B REL606 (the Lenski long-term evolution ancestor), NCBI assembly ASM1798v1
#: (GCA_000017985.1 / GCF_000017985.1, replicon CP000819.1 / NC_012967.1): ``ECB_``
#: locus tags. A B strain, not K-12, so neither K-12 set stands in for it. No GO
#: Consortium or EBI GOA file covers it; its GO is the RefSeq GFF's inline terms.
ECOLI_B_REL606 = "ecoli_B_REL606_ASM1798v1"
#: E. coli K-12 W3110, NCBI assembly ASM1024v1 (GCA_000010245.1 / GCF_000010245.2,
#: replicon AP009048.1 / NC_007779.1): a third K-12 strain, so neither MG1655's
#: b-numbers nor BW25113's ``BW25113_`` numbers are its locus tags. Deposited for the
#: Teteneva 2024 RB-TnSeq row, and NOT yet readable by ``BacterialGenome``: the GenBank
#: member carries 4,444 gene features and NOT ONE ``locus_tag`` (the 2006 DDBJ/NIG
#: annotation keys genes by symbol and carries ``ECK:JW:b`` crosswalk notes on the CDS),
#: so the tier's GenBank-first ingest refuses it and the primary-annotation choice is
#: open. ``ecoli_K12_W3110_ASM1024v1`` is therefore in no schema vocabulary yet; see
#: ``torchcell/datasets/ecoli/teteneva2024.py``.
ECOLI_K12_W3110 = "ecoli_K12_W3110_ASM1024v1"
#: P. putida KT2440, NCBI assembly ASM756v2 (GCA_000007565.2 / GCF_000007565.2):
#: ``PP_`` locus tags, plus the EBI GOA proteome file keyed to them.
PPUTIDA_KT2440 = "pputida_KT2440_ASM756v2"
#: The GO Consortium release of 2026-08-05: ``go-basic.obo``, one copy that every
#: bacterial set pins by this id instead of carrying its own.
GO_RELEASE_20260805 = "go_release_2026-08-05"

#: Roles a file in an assembly set can play (explicit per file; never inferred).
ROLE_CONTAINER = "container"  # the archive the release was fetched as
ROLE_SEQUENCE = "sequence"  # FASTA of chromosomes, ORFs, proteins, assemblies
ROLE_ANNOTATION = "annotation"  # GFF, gene associations
ROLE_MATRIX = "matrix"  # gene-by-isolate presence or copy-number tables
ROLE_INDEX = "index"  # member paths of a container, or a locus or replicon table
ROLE_ONTOLOGY = "ontology"  # an ontology release (OBO), not a gene annotation

#: Stored-record sentinels that name an assembly set without a filesystem path. The
#: Bloom 2019 BY parent stores the first string (``bloom2019.S288C_ASSEMBLY``); a
#: dereferencer maps it here so the stored records never change.
SENTINEL_ASSEMBLY_SETS: dict[str, str] = {
    "S288C_reference_genome_R64-4-1_20230830 (SGD; torchcell reference)": SGD_S288C_R64
}

#: Where the canonical tier lives, for the seeding hint in error messages.
CANONICAL_HOST_ROOT = "gilahyper:/scratch/projects/torchcell-scratch/torchcell-genomes"


class GenomeIntegrityError(RuntimeError):
    """A tier file's bytes do not match the sha256 its manifest pins."""


class GenomeManifest(BaseModel):
    """Provenance and integrity record for one assembly set.

    A sibling of the literature ``Manifest`` (which is keyed by a paper) reusing its
    per-file ``ArtifactRecord``: a genome belongs to an organism and a source release,
    not to a citation, so the top level names those instead. ``provenance_complete``
    keeps the literature meaning: False when a file's bytes are sha256-pinned but the
    retrieval chain could not be reproduced, never when anything was fabricated.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=GENOME_MANIFEST_VERSION)
    assembly_set: str = Field(description="Directory name under torchcell-genomes/.")
    organism: str
    strain_or_population: str = Field(description='"S288C" or "1,011 isolates".')
    source: str = Field(description='"SGD" or "Peter et al. 2018".')
    release: str = Field(description='"R64-4-1_20230830" or "2018".')
    citation_key: str | None = None
    source_url: str | None = Field(
        default=None, description="Set only when every retrieval record is genuine."
    )
    files: list[ArtifactRecord] = Field(default_factory=list)
    provenance_complete: bool
    created_at: str
    notes: str | None = None

    def record(self, filename: str) -> ArtifactRecord:
        """The record for one file; a name the manifest does not list raises."""
        for rec in self.files:
            if rec.path == filename:
                return rec
        raise KeyError(f"{self.assembly_set}: {filename!r} is not in the manifest")


def genomes_root(data_root: str | None = None) -> str:
    """``$DATA_ROOT/torchcell-genomes``; ``DATA_ROOT`` comes from the environment."""
    if data_root is None:
        from dotenv import load_dotenv

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
    return osp.join(data_root, GENOMES_DIR)


def assembly_set_dir(assembly_set: str, data_root: str | None = None) -> str:
    """Absolute directory of one assembly set (not checked for existence)."""
    return osp.join(genomes_root(data_root), assembly_set)


def _seed_hint(assembly_set: str, root: str) -> str:
    return (
        f"assembly set {assembly_set!r} is not present under {root}; seed it with: "
        f"rsync -a {CANONICAL_HOST_ROOT}/{assembly_set}/ {root}/{assembly_set}/"
    )


def load_genome_manifest(
    assembly_set: str, data_root: str | None = None
) -> GenomeManifest:
    """Validate and return ``<tier>/<assembly_set>/manifest.json``."""
    root = genomes_root(data_root)
    path = osp.join(root, assembly_set, MANIFEST_FILENAME)
    if not osp.isfile(path):
        raise FileNotFoundError(_seed_hint(assembly_set, root))
    manifest = GenomeManifest.model_validate_json(Path(path).read_text())
    if manifest.assembly_set != assembly_set:
        raise GenomeIntegrityError(
            f"{path} names assembly set {manifest.assembly_set!r}, not {assembly_set!r}"
        )
    return manifest


def resolve(
    assembly_set: str,
    filename: str,
    *,
    verify: bool = True,
    data_root: str | None = None,
) -> str:
    """Absolute path of one file in an assembly set, sha256-verified by default.

    Raises ``FileNotFoundError`` (with the seeding command) when the tier, the set or
    the file is absent, ``KeyError`` when the manifest does not list the file, and
    ``GenomeIntegrityError`` when the bytes on disk do not match the pinned sha256.
    """
    manifest = load_genome_manifest(assembly_set, data_root)
    rec = manifest.record(filename)
    path = osp.join(assembly_set_dir(assembly_set, data_root), rec.path)
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"{assembly_set}/{rec.path} is in the manifest but not on disk; "
            + _seed_hint(assembly_set, genomes_root(data_root))
        )
    if verify:
        got = sha256_file(Path(path))
        if got != rec.sha256:
            raise GenomeIntegrityError(
                f"{assembly_set}/{rec.path}: sha256 {got} on disk, manifest pins "
                f"{rec.sha256}"
            )
    return path


def verify_assembly_set(
    assembly_set: str, data_root: str | None = None
) -> dict[str, str]:
    """Hash every manifest file; return ``{filename: sha256}``; raise on any defect."""
    manifest = load_genome_manifest(assembly_set, data_root)
    return {
        rec.path: sha256_file(
            Path(resolve(assembly_set, rec.path, verify=True, data_root=data_root))
        )
        for rec in manifest.files
    }


def deposit_assembly_set(manifest: GenomeManifest, data_root: str | None = None) -> str:
    """Write ``manifest.json`` for a set whose files are already in place.

    Refuses to overwrite an existing manifest, and refuses a manifest whose records
    disagree with the bytes on disk, so a manifest can never claim a hash it did not
    measure. Returns the manifest path.
    """
    directory = assembly_set_dir(manifest.assembly_set, data_root)
    path = osp.join(directory, MANIFEST_FILENAME)
    if osp.exists(path):
        raise FileExistsError(f"{path} exists; a deposited manifest is never rewritten")
    if not manifest.files:
        raise ValueError(f"{manifest.assembly_set}: a manifest must list its files")
    for rec in manifest.files:
        file_path = Path(directory) / rec.path
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{manifest.assembly_set}/{rec.path} is not on disk"
            )
        got = sha256_file(file_path)
        if got != rec.sha256:
            raise GenomeIntegrityError(
                f"{manifest.assembly_set}/{rec.path}: record pins {rec.sha256}, "
                f"disk has {got}"
            )
        size = file_path.stat().st_size
        if size != rec.bytes:
            raise GenomeIntegrityError(
                f"{manifest.assembly_set}/{rec.path}: record says {rec.bytes} bytes, "
                f"disk has {size}"
            )
    Path(path).write_text(manifest.model_dump_json(indent=2))
    return path
