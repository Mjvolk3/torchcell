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
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import ArtifactRecord, sha256_file

GENOMES_DIR = "torchcell-genomes"
MANIFEST_FILENAME = "manifest.json"
GENOME_MANIFEST_VERSION = 1

#: The SGD R64-4-1 (2023-08-30) S288C reference release, as fetched from the SGD archive.
SGD_S288C_R64: Final = "sgd_S288C_R64-4-1_20230830"
#: The NCBI accession of the sequence that release is on: GenBank assembly "R64"
#: (GCA_000146045.2, submitted by SGD, released 2011-04-18; its RefSeq pair is
#: GCF_000146045.2). MEASURED on 2026-10-15, not assumed: all 16 nuclear chromosomes of
#: ``S288C_reference_sequence_R64-4-1_20230830.fsa`` (the deposited set) are byte-identical
#: to the 16 sequences of ``GCA_000146045.2_R64_genomic.fna.gz`` (sha256
#: b764924296b034817bbde44a87b36611c0918e7d00c650be4497118e70c3c43c); only the
#: mitochondrion (NC_001224, 85,779 bp) is outside the NCBI set. R64-2-1 through R64-4-1
#: revised the annotation and not the sequence, so a coordinate read on any R64 release
#: is a coordinate on this accession.
SGD_S288C_R64_GENBANK_ACCESSION: Final = "GCA_000146045.2"
#: Peter et al. 2018: the 1,011 isolate assemblies and the pangenome matrices.
PETER2018_1011: Final = "peter2018_1011_assemblies"
#: E. coli K-12 MG1655, NCBI assembly ASM584v2 (GCA_000005845.2 with its RefSeq
#: annotation GCF_000005845.2): b-number locus tags, plus the GO Consortium
#: ``ECOLI-uniprot.gaf.gz`` keyed to b-numbers.
ECOLI_K12_MG1655: Final = "ecoli_K12_MG1655_ASM584v2"
#: E. coli K-12 BW25113 (the Keio background), NCBI assembly ASM75055v1
#: (GCA_000750555.1 / GCF_000750555.1). One set per strain, never one per host: a
#: ``BW25113_`` number is not an MG1655 b-number. Its own GO is the RefSeq GFF's
#: inline terms and NCBI's GAF; MG1655's GAF is referenced by the ``ECOLI_K12_MG1655``
#: id, never copied, and which source a loader reads is still open.
ECOLI_K12_BW25113: Final = "ecoli_K12_BW25113_ASM75055v1"
#: BW25113's background genotype, VERBATIM from the ``/note`` qualifier of the ``source``
#: feature of ``GCA_000750555.1_ASM75055v1_genomic.gbff.gz`` in the
#: ``ecoli_K12_BW25113_ASM75055v1`` set above. The full qualifier reads "from B. L.
#: Wanner laboratory; genotype: rrnB3 lacZ4787 hsdR514 (araBAD)567 (rhaBAD)568 rph-1";
#: this constant is the genotype clause of it. It lives in the genomes tier and not in
#: ``torchcell/datamodels/schema.py`` because it is DATA about one deposited assembly set
#: and not a record contract: a strain constant in the schema surface means every future
#: host adds a data binding to the most sensitive file in the database, and the
#: fingerprinted closure of every loader that reaches it moves when the string does.
#: Its provenance is the tier file, whose sha256 ``resolve`` verifies on every call, so
#: the string has provenance without a paper quote and
#: ``tests/torchcell/sequence/genome/test_registry.py`` re-reads it from those bytes.
BW25113_BACKGROUND_GENOTYPE: Final = (
    "rrnB3 lacZ4787 hsdR514 (araBAD)567 (rhaBAD)568 rph-1"
)
#: The same genotype split into its six lesion designations, each verbatim. Deliberately
#: NOT expanded into ``BacterialBackgroundAllele`` records here: the ``/note`` gives each
#: lesion's DESIGNATION and nothing else, while an allele record also asserts an edit
#: kind, a functional status and a locus. Three of the six name a gene BW25113's own
#: annotation still carries (measured: lacZ at BW25113_0344, hsdR at BW25113_4350, rph at
#: BW25113_3643) and the two operon deletions name genes it does NOT annotate at all
#: (araB, araA, rhaB, rhaA are absent, which is what being deleted looks like), so
#: mapping those to a locus would mean borrowing MG1655 b-numbers -- the cross-strain
#: inference this plan refuses. The loader that needs the typed alleles sources the edit
#: kinds from the BW25113 construction paper and records which lesion it mapped how.
BW25113_BACKGROUND_LESIONS: tuple[str, ...] = (
    "rrnB3",
    "lacZ4787",
    "hsdR514",
    "(araBAD)567",
    "(rhaBAD)568",
    "rph-1",
)
#: E. coli B REL606 (the Lenski long-term evolution ancestor), NCBI assembly ASM1798v1
#: (GCA_000017985.1 / GCF_000017985.1, replicon CP000819.1 / NC_012967.1): ``ECB_``
#: locus tags. A B strain, not K-12, so neither K-12 set stands in for it. No GO
#: Consortium or EBI GOA file covers it; its GO is the RefSeq GFF's inline terms.
ECOLI_B_REL606: Final = "ecoli_B_REL606_ASM1798v1"
#: E. coli K-12 W3110, NCBI assembly ASM1024v1 (GCA_000010245.1 / GCF_000010245.2,
#: replicon AP009048.1 / NC_007779.1): a third K-12 strain, so neither MG1655's
#: b-numbers nor BW25113's ``BW25113_`` numbers are its locus tags. Deposited for the
#: Teteneva 2024 RB-TnSeq row. The GenBank member carries 4,444 gene features and NOT ONE
#: ``locus_tag`` (the 2006 DDBJ/NIG annotation keys genes by symbol and carries
#: ``ECK:JW:b`` crosswalk notes on the CDS), so the set is read RefSeq first (owner route,
#: #910, 2026-10-12): ``Y75_RS`` locus tags, ``EcoliK12W3110Genome`` in
#: ``torchcell/sequence/genome/ecoli/w3110.py``.
ECOLI_K12_W3110: Final = "ecoli_K12_W3110_ASM1024v1"
#: P. putida KT2440, NCBI assembly ASM756v2 (GCA_000007565.2 / GCF_000007565.2):
#: ``PP_`` locus tags, plus the EBI GOA proteome file keyed to them.
PPUTIDA_KT2440: Final = "pputida_KT2440_ASM756v2"
#: The GO Consortium release of 2026-08-05: ``go-basic.obo``, one copy that every
#: bacterial set pins by this id instead of carrying its own.
GO_RELEASE_20260805: Final = "go_release_2026-08-05"
#: Hale et al. 2024 Supplemental Data 13: the 3S (322134S) parent's variants against
#: S288C (SNVs and single-base indels). A VARIANT TABLE, not an assembly: no public 3S
#: assembly exists, so this is the sequence reference the BYx3S segregant mosaics pin
#: their second parent to (``torchcell/datasets/scerevisiae/hale2024.py``).
HALE2024_3S_VARIANTS: Final = "hale2024_322134S_variants"

#: CEN.PK113-7D, NCBI GenBank assembly ASM257140v2 (GCA_002571405.2): the reference the
#: CEN.PK-derived engineering strains are written against (#990, Zhang 2020). Its locus
#: tags are the ``SCEN_*`` namespace, which is a different identifier space from S288C's
#: systematic names, and no GO release is keyed to it, so the set carries no GO file.
#: Deposited by ``scripts/provision_yeast_genomes.py``.
SCEREVISIAE_CENPK113_7D: Final = "scerevisiae_CENPK113-7D_ASM257140v2"
#: Its one accession: the set's deposited assembly report names a GenBank accession and
#: no RefSeq one, and NCBI (dataset report, 2026-10-15) lists no paired RefSeq assembly.
SCEREVISIAE_CENPK113_7D_GENBANK_ACCESSION: Final = "GCA_002571405.2"

#: Pichia kudriavzevii (Issatchenkia orientalis) SD108, genome v1: the group's curated
#: assembly (5 chromosomes, the circular mitochondrion and an unplaced rDNA unit) with
#: its own annotation, taken from the GitHub repository
#: ``Aurosish/I.orientalis_SD108_Genome_Reconstruction`` at commit
#: ``8ca6a106809d6b708d4ba621e3834649dd122720``. Its gene ids are the assembly's own
#: ``SD108_X<seq><type><number>`` namespace; it has no NCBI accession and no GO file.
#: Deposited by ``scripts/provision_pkudriavzevii_genome.py``, read by
#: ``PKudriavzeviiSD108Genome`` in ``torchcell/sequence/genome/pkudriavzevii/sd108.py``.
PKUDRIAVZEVII_SD108: Final = "pkudriavzevii_SD108_v1"

#: Roles a file in an assembly set can play (explicit per file; never inferred).
ROLE_CONTAINER = "container"  # the archive the release was fetched as
ROLE_SEQUENCE = "sequence"  # FASTA of chromosomes, ORFs, proteins, assemblies
ROLE_ANNOTATION = "annotation"  # GFF, gene associations
ROLE_MATRIX = "matrix"  # gene-by-isolate presence or copy-number tables
ROLE_INDEX = "index"  # member paths of a container, or a locus or replicon table
ROLE_ONTOLOGY = "ontology"  # an ontology release (OBO), not a gene annotation
ROLE_VARIANTS = "variants"  # a strain's variants against a reference (not an assembly)

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


# --------------------------------------------------------------------------- #
# ADDITIVE (Peter 2018 phenome, yeast wave 1): grow a deposited set by new files.
# ``deposit_assembly_set`` never rewrites a manifest, because rewriting could
# change a hash a stored record already points at. Adding a file changes no
# existing claim, so this is the one sanctioned rewrite: every existing record
# must come back byte-identical, and the new ones are checked against the bytes
# on disk exactly as a first deposit checks them.
# --------------------------------------------------------------------------- #
def extend_assembly_set(
    assembly_set: str, new_files: list[ArtifactRecord], data_root: str | None = None
) -> str:
    """Append ``new_files`` to a deposited set's manifest; return the manifest path.

    Refuses an empty list, a path the manifest already lists, a file that is not on
    disk, and a record whose sha256 or byte count disagrees with the disk. Existing
    records, ``provenance_complete`` and every top-level field are carried over
    unchanged; only ``files`` grows. A set without a manifest is refused through
    ``load_genome_manifest`` (with its seeding hint), since a first deposit goes
    through ``deposit_assembly_set``.
    """
    if not new_files:
        raise ValueError(f"{assembly_set}: no files to add")
    manifest = load_genome_manifest(assembly_set, data_root)
    listed = {rec.path for rec in manifest.files}
    clashes = sorted(rec.path for rec in new_files if rec.path in listed)
    if clashes:
        raise FileExistsError(
            f"{assembly_set}: {clashes} already in the manifest; a listed file is "
            "never re-recorded"
        )
    directory = assembly_set_dir(assembly_set, data_root)
    for rec in new_files:
        file_path = Path(directory) / rec.path
        if not file_path.is_file():
            raise FileNotFoundError(f"{assembly_set}/{rec.path} is not on disk")
        got = sha256_file(file_path)
        if got != rec.sha256:
            raise GenomeIntegrityError(
                f"{assembly_set}/{rec.path}: record pins {rec.sha256}, disk has {got}"
            )
        size = file_path.stat().st_size
        if size != rec.bytes:
            raise GenomeIntegrityError(
                f"{assembly_set}/{rec.path}: record says {rec.bytes} bytes, disk has "
                f"{size}"
            )
    extended = manifest.model_copy(update={"files": [*manifest.files, *new_files]})
    path = osp.join(directory, MANIFEST_FILENAME)
    Path(path).write_text(extended.model_dump_json(indent=2))
    return path


# --------------------------------------------------------------------------- #
# end ADDITIVE (Peter 2018 phenome)
# --------------------------------------------------------------------------- #
