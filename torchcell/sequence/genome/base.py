# torchcell/sequence/genome/base
# [[torchcell.sequence.genome.base]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/base
# Test file: tests/torchcell/sequence/genome/test_base.py

"""Organism-agnostic annotated genome over one GFF3 + FASTA release in the genomes tier.

:class:`AnnotatedGenome` holds what every host genome shares: the release files
resolved through :func:`~torchcell.sequence.genome.registry.resolve` (never a download),
the ``genome_root`` cache with its recorded gffutils ``data.db`` and the build, install,
migration and private-copy machinery that keeps that file trustworthy under concurrent
readers, the gene set, the GO properties, the locus index and the layered
:meth:`AnnotatedGenome.resolve_gene_name`. :class:`AnnotatedGene` holds one gene's
coordinates, sequences and windows. What varies by organism is a class-level hook that
the subclass sets:

* ``AnnotatedGenome.LOCUS_FEATURE_TYPES``: the gene-like GFF feature types a source
  gene name may resolve to.
* :meth:`AnnotatedGene.seqid_to_chromosome` and :meth:`AnnotatedGenome.fasta_chromosome`:
  a GFF seqid, and a record of the DNA FASTA, to the integer ``chromosome`` key.
* ``AnnotatedGene.GO_ATTRIBUTE`` and :meth:`AnnotatedGenome._prepare_go_obo`: the GFF
  attribute that carries a gene's GO terms, and where the GO ontology file comes from.
* :meth:`AnnotatedGenome.release_files`: the GFF and FASTA members of the assembly set.
* :meth:`AnnotatedGenome.gene_class`: the gene class :meth:`AnnotatedGenome.__getitem__`
  builds, whose :meth:`AnnotatedGene.coding_feature` picks the feature that gives a gene
  its sequence and whose :meth:`AnnotatedGene.annotate` copies its GFF attributes.
* ``ANNOTATION_NAME`` / ``ANNOTATION_RELEASE``: how a resolution's ``note`` names the
  annotation.
* ``AnnotatedGene.FEATURE_ID_PREFIX``: what the GFF puts before a gene id in the
  feature's ``ID`` (NCBI writes ``ID=gene-b0002`` for the locus tag ``b0002``; SGD writes
  the bare systematic name, the default ``""``).
* :meth:`AnnotatedGenome._read_sequences`: how the FASTA members become
  ``fasta_dna`` / ``fasta_protein`` / ``fasta_cds``. The default parses three plain
  FASTAs; a release with no CDS FASTA (``GenomeReleaseFiles.cds_fasta`` is ``None``)
  derives the CDS sequences in its override.

Nothing here is organism-specific. The S. cerevisiae S288C genome is the SGD subclass
``SCerevisiaeGenome`` in ``torchcell.sequence.genome.scerevisiae.s288c``, which
re-exports every name defined here that it defined before the extraction (2026.10.07).
"""

import fcntl
import filecmp
import hashlib
import json
import logging
import os
import os.path as osp
import re
import secrets
import shutil
import socket
import sqlite3
import stat
import tempfile
import weakref
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, SupportsIndex, cast

import attrs
import gffutils
import pandas as pd
from attrs import define, field
from Bio import SeqIO
from gffutils.feature import Feature
from goatools.obo_parser import GODag
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sortedcontainers import SortedDict, SortedSet

from torchcell.literature.manifest import sha256_file
from torchcell.sequence import (
    DnaSelectionResult,
    DnaWindowResult,
    Gene,
    GeneSet,
    Genome,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    compute_codon_frequency,
)
from torchcell.sequence.db_connection import GffutilsConnectionManager
from torchcell.sequence.genome.registry import resolve

log = logging.getLogger(__name__)

nucleotides = ["A", "T", "G", "C"]
all_codons = ["".join(codon) for codon in product(nucleotides, repeat=3)]


class GeneNameStatus(StrEnum):
    """Outcome of resolving a source gene name against a genome's current annotation.

    The layered resolver (:meth:`AnnotatedGenome.resolve_gene_name`) reconciles a
    dataset's source gene name -- systematic (e.g. Ohya 2005) or common (e.g. Cachera) --
    to the genome's current annotation (for S288C, R64-4-1). Datasets carry historical
    names: an annotation authority renames/merges features and retires dubious ORFs (SGD
    does), so a 2005-era systematic name may no longer be a live "gene". This status tells
    the loader what the name resolved to so it can RETAIN the record (a real strain /
    perturbation) with the correct identifier and provenance, rather than silently
    dropping it.
    """

    CURRENT = "current"  # a live "gene" feature (systematic_name == input)
    RENAMED = (
        "renamed"  # alias of exactly one current gene (systematic_name = that gene)
    )
    NON_GENE_FEATURE = "non_gene_feature"  # a valid locus feature that is not a "gene"
    RETIRED = "retired"  # not in the annotation at all; retained as a legacy name
    AMBIGUOUS = "ambiguous"  # alias mapping to >1 current feature; needs human review


class GeneNameResolution(BaseModel):
    """Typed result of :meth:`AnnotatedGenome.resolve_gene_name` (pydantic-first)."""

    input_name: str
    status: GeneNameStatus
    # The identifier the loader should store. For CURRENT/RENAMED this is a live systematic
    # gene id; for NON_GENE_FEATURE the (valid) feature id; for RETIRED the original name
    # kept as a legacy systematic identifier; None only for AMBIGUOUS.
    systematic_name: str | None
    feature_type: str | None = None  # GFF featuretype for NON_GENE_FEATURE resolutions
    candidates: list[str] = Field(default_factory=list)  # populated only for AMBIGUOUS
    note: str | None = None

    @property
    def is_current_gene(self) -> bool:
        """True when the name resolved to a live gene (CURRENT or RENAMED)."""
        return self.status in (GeneNameStatus.CURRENT, GeneNameStatus.RENAMED)


#: The gffutils database every genome builds under its ``genome_root``.
GENOME_DB_FILENAME = "data.db"
#: Where a migrated database whose rows differ from a fresh build is kept (one file).
UNTRUSTED_DB_FILENAME = GENOME_DB_FILENAME + ".untrusted"
#: The table inside ``data.db`` that records what the database was built from.
SOURCE_TABLE = "torchcell_genome_db_source"
#: The :class:`GenomeDatabaseRecord` schema version this code writes and reads. ANY
#: change to the fields of :class:`GenomeDatabaseRecord` or :class:`GenomeDatabaseSource`
#: (added, removed, renamed or retyped) must bump it; the test suite pins both field
#: sets against this number.
RECORD_VERSION = 1
#: The ``gffutils.create_db`` arguments every build uses (also recorded in the source).
CREATE_DB_KWARGS: dict[str, Any] = {
    "keep_order": True,
    "merge_strategy": "merge",
    "sort_attribute_values": True,
}
#: Temporary files this module writes in a genome directory: ``data.db.<host>.<pid>.
#: <random>.building`` (a build) and ``data.db.untrusted.<host>.<pid>.<random>.
#: building`` (the copy of an untrusted database on its way to ``data.db.untrusted``).
_BUILD_TEMP = re.compile(
    r"^data\.db\.(?:untrusted\.)?(?P<host>.+)\.(?P<pid>\d+)\.[a-z0-9_]+"
    r"\.building(?:-journal)?$"
)
#: Largest pid a sweep considers (a 32-bit pid_t); a name beyond it is not ours.
_PID_LIMIT = 2**31 - 1
#: Private database copies in the temp dir, ``torchcell-genome-<host>-<pid>-<random>.db``:
#: an instance's write copy, or a short-lived peek copy (``...-peek<random>.db``, with
#: its ``-journal`` while the copy is rolled back).
_PRIVATE_COPY = re.compile(
    r"^torchcell-genome-(?P<host>.+)-(?P<pid>\d+)-[a-z0-9_]+\.db(?:-journal)?$"
)


class GenomeDatabaseSource(BaseModel):
    """What a genome ``data.db`` is built from: the pinned GFF and the build arguments.

    Any change to these fields bumps :data:`RECORD_VERSION`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    gff_filename: str
    gff_sha256: str
    keep_order: bool
    merge_strategy: str
    sort_attribute_values: bool


class GenomeDatabaseRecord(BaseModel):
    """The one row of :data:`SOURCE_TABLE`: the source plus what the build left.

    The record lives inside the sqlite file, so the record and the rows it describes
    are replaced together by one atomic rename. At every open the row counts (one
    ``GROUP BY`` over ``features``, one ``COUNT`` over ``relations``) and sqlite's file
    change counter (header bytes 24-27, advanced by every committed write in the
    rollback-journal modes gffutils uses) are compared with the record, so any write
    made in place after the build, by a process running pre-2026.10.01 code, is
    detected: a deletion changes the counts, any write changes the counter.

    ``version`` is :data:`RECORD_VERSION` of the code that wrote it. A reader treats a
    lower (or absent) version as untrusted and migrates it once; a higher version was
    written by newer code and is refused by name
    (:class:`GenomeDatabaseVersionError`), so two code versions never rebuild the
    file back and forth. Any change to these fields bumps :data:`RECORD_VERSION`; a
    current-version record that does not validate raises
    :class:`GenomeDatabaseRecordError`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    source: GenomeDatabaseSource
    featuretype_counts: dict[str, int]
    relations_count: int
    change_counter: int


class GenomeDatabaseSourceError(RuntimeError):
    """``data.db`` records a different source than this genome's pinned GFF."""


class GenomeDatabaseVersionError(RuntimeError):
    """``data.db`` carries a record written by newer code than this checkout."""


class GenomeDatabaseRecordError(RuntimeError):
    """``data.db`` carries a record this checkout cannot read at its own version."""


#: The remedy every record-version or record-schema refusal names.
_UPDATE_CHECKOUT = "Update this checkout, or resubmit the job from an updated checkout."


class GenomeDatabaseUnavailableError(RuntimeError):
    """``data.db`` cannot be read for a reason that says nothing about its rows (it is
    locked by another process, cannot be opened, or an I/O error occurred); every
    file is left alone.
    """


class GenomeDatabaseInstallError(OSError):
    """A finished build could not be renamed onto ``data.db``; the build was removed."""


#: sqlite primary result codes that say the file's CONTENT is damaged or is not this
#: database (a missing table is SQLITE_ERROR).
_DAMAGE_CODES = frozenset(
    {sqlite3.SQLITE_ERROR, sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}
)
#: sqlite primary result codes that say another process holds the database.
_LOCK_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
#: sqlite primary result codes that say this process may not open the file.
_ACCESS_CODES = frozenset({sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_PERM})
#: Files sqlite keeps beside a database. One left beside an OLD file must never sit
#: beside a new one: sqlite would apply the old file's pages to it at the next open.
_COMPANION_SUFFIXES = ("-journal", "-wal", "-shm")


def require_damage(db_path: str, exc: sqlite3.DatabaseError) -> None:
    """Return when ``exc`` says ``db_path`` itself is damaged: corrupt, not a
    database, a missing table, or a hot rollback journal a killed in-place writer left
    (``SQLITE_READONLY_ROLLBACK``). Raise :class:`GenomeDatabaseUnavailableError`
    for every other sqlite error (locked, cannot open, permission, I/O), which says
    nothing about the rows, so the file must not be migrated or replaced.

    An error with no sqlite code was raised by Python's sqlite3 layer, not by sqlite:
    at these reads that is a TEXT cell that does not decode as UTF-8, which is damaged
    content.
    """
    code = getattr(exc, "sqlite_errorcode", None)
    if code is None:
        return
    primary = code & 0xFF
    if primary in _DAMAGE_CODES or code == sqlite3.SQLITE_READONLY_ROLLBACK:
        return
    if primary in _LOCK_CODES:
        raise GenomeDatabaseUnavailableError(
            f"{db_path} is locked by another process ({exc.sqlite_errorname}: {exc}); "
            "every file is left alone. Retry when that process has finished."
        ) from exc
    if primary in _ACCESS_CODES:
        mode = (
            oct(os.stat(db_path).st_mode & 0o777) if osp.exists(db_path) else "absent"
        )
        raise GenomeDatabaseUnavailableError(
            f"{db_path} cannot be opened by this process ({exc.sqlite_errorname}: "
            f"{exc}; mode {mode}); every file is left alone. Fix its permissions or "
            "ownership, then retry."
        ) from exc
    raise GenomeDatabaseUnavailableError(
        f"{db_path} cannot be read right now ({exc.sqlite_errorname}: {exc}); every "
        "file is left alone. Retry, and check the filesystem if it persists."
    ) from exc


def _ro_uri(path: str) -> str:
    """A read-only sqlite URI for ``path``, percent-encoded, so a ``?``, ``#`` or ``%``
    in a directory name stays part of the path instead of starting a query.
    """
    return Path(path).absolute().as_uri() + "?mode=ro"


@contextmanager
def _root_lock(db_dir: str) -> Iterator[None]:
    """An exclusive ``flock`` on the genome-root directory for the duration of a
    migration's or rebuild's critical section, so two migrators never interleave
    their keep and install steps. The lock is released when the descriptor closes. A
    filesystem that cannot take the lock is refused by name; nothing proceeds
    unlocked.
    """
    fd = os.open(db_dir, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise GenomeDatabaseUnavailableError(
                f"{db_dir} cannot be locked ({exc}); a migration or rebuild needs an "
                "exclusive lock on the genome root, so every file is left alone. Put "
                "genome_root on a filesystem that supports flock."
            ) from exc
        yield
    finally:
        os.close(fd)


def _remove_if_present(path: str) -> None:
    """Remove ``path``; a path that is already gone is fine."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def install_genome_database(
    tmp_path: str, db_path: str, kept: str | None = None, kept_as_is: bool = False
) -> list[str]:
    """Rename the finished build ``tmp_path`` onto ``db_path``.

    A companion file left beside the OLD file (``-journal``, ``-wal``, ``-shm``) is
    first moved beside the kept copy ``kept`` (``data.db.untrusted-journal``, ...),
    or removed when nothing is kept, so it is never paired with the new build. When
    ``kept_as_is`` (:func:`_keep_copy` left a byte-equal kept copy in place) and the
    kept copy already has that companion, ``db_path``'s own companion is removed
    instead of overwriting the kept one, which may be the only file that rolls the
    pair back to the committed file; on bytes equal to the kept file, ``db_path``'s
    companion undoes nothing the kept one does not. Returns the companions removed
    that way, so the caller's one WARNING names them. A companion another migrator
    moved first is skipped. If the rename fails, the build is removed and
    :class:`GenomeDatabaseInstallError` names both paths and the OS error.
    """
    removed_beside_kept: list[str] = []
    for suffix in _COMPANION_SUFFIXES:
        companion = db_path + suffix
        try:
            if kept is None:
                os.remove(companion)
            elif kept_as_is and osp.lexists(kept + suffix):
                # The kept copy (byte-equal to data.db) already has its own
                # companion: data.db's undoes nothing the kept one does not, and
                # must not overwrite the kept pair's journal.
                os.remove(companion)
                removed_beside_kept.append(companion)
            else:
                os.replace(companion, kept + suffix)
        except FileNotFoundError:  # absent, or another migrator moved it first
            continue
    try:
        os.replace(tmp_path, db_path)
    except OSError as exc:
        os.remove(tmp_path)
        raise GenomeDatabaseInstallError(
            f"the build {tmp_path} could not be renamed onto {db_path} ({exc}); the "
            "build was removed."
        ) from exc
    return removed_beside_kept


class GenomeRootNotFoundError(FileNotFoundError):
    """``genome_root`` does not exist and the constructor was not asked to build."""


class GenomeRootNotWritableError(PermissionError):
    """``data.db`` must be (re)built but this process cannot write ``genome_root``."""


def genome_database_source(
    assembly_set: str, gff_filename: str, gff_path: str
) -> GenomeDatabaseSource:
    """The source a database built from ``gff_path`` now would record."""
    return GenomeDatabaseSource(
        assembly_set=assembly_set,
        gff_filename=gff_filename,
        gff_sha256=sha256_file(Path(gff_path)),
        **CREATE_DB_KWARGS,
    )


def _change_counter(db_path: str) -> int:
    """Sqlite's file change counter: header bytes 24-27, big-endian."""
    try:
        with open(db_path, "rb") as fh:
            header = fh.read(28)
    except FileNotFoundError as exc:
        raise _vanished(db_path, exc) from exc
    return int.from_bytes(header[24:28], "big")


def _database_counts(conn: sqlite3.Connection) -> tuple[dict[str, int], int]:
    """Per-featuretype row counts of ``features`` and the row count of ``relations``."""
    featuretype_counts = {
        str(featuretype): int(n)
        for featuretype, n in conn.execute(
            "SELECT featuretype, COUNT(*) FROM features GROUP BY featuretype"
        )
    }
    relations_count = int(conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0])
    return featuretype_counts, relations_count


def database_content_digest(db_path: str) -> str:
    """sha256 over every ``features`` and ``relations`` row, in a fixed order."""
    h = hashlib.sha256()
    conn = sqlite3.connect(_ro_uri(db_path), uri=True)
    for row in conn.execute("SELECT * FROM features ORDER BY id"):
        h.update(repr(row).encode())
    h.update(b"relations")
    for row in conn.execute("SELECT * FROM relations ORDER BY parent, child, level"):
        h.update(repr(row).encode())
    conn.close()
    return h.hexdigest()


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # alive, owned by another user
        return True
    return True


def _sweep_dead(directory: str, pattern: re.Pattern[str]) -> list[str]:
    """Remove this host's and this user's regular files in ``directory`` matching
    ``pattern`` whose writer pid is dead; another host's, another user's, a live pid's,
    a directory and a name whose pid is out of range are left. Safe to run in many
    processes at once: a file another sweeper removed first is skipped. Returns the
    names this call removed.
    """
    host = socket.gethostname()
    removed = []
    for name in sorted(os.listdir(directory)):
        m = pattern.match(name)
        if m is None or m["host"] != host or not 0 < int(m["pid"]) <= _PID_LIMIT:
            continue
        path = osp.join(directory, name)
        try:
            st = os.lstat(path)
        except FileNotFoundError:  # another sweeper removed it first
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            continue
        if _pid_alive(int(m["pid"])):
            continue
        try:
            os.remove(path)
        except FileNotFoundError:  # another sweeper removed it first
            continue
        removed.append(name)
    return removed


def _temp_in(directory: str, prefix: str) -> str:
    """A new empty file ``<prefix>.<host>.<pid>.<random>.building`` in ``directory``."""
    fd, path = tempfile.mkstemp(
        prefix=f"{prefix}.{socket.gethostname()}.{os.getpid()}.",
        suffix=".building",
        dir=directory,
    )
    os.close(fd)
    return path


def write_genome_database(
    gff_path: str, db_dir: str, source: GenomeDatabaseSource
) -> str:
    """Build a complete, recorded database in a new temporary file in ``db_dir``.

    The file is in ``db_dir`` itself, so the rename that installs it is atomic, and its
    name is unique per call. Returns its path; the caller renames it into place. On any
    failure the temporary file is removed before the exception propagates.
    """
    tmp_path = _temp_in(db_dir, GENOME_DB_FILENAME)
    try:
        gffutils.create_db(gff_path, dbfn=tmp_path, force=True, **CREATE_DB_KWARGS)
        conn = sqlite3.connect(tmp_path)
        featuretype_counts, relations_count = _database_counts(conn)
        expected_counter = _change_counter(tmp_path) + 1
        record = GenomeDatabaseRecord(
            version=RECORD_VERSION,
            source=source,
            featuretype_counts=featuretype_counts,
            relations_count=relations_count,
            change_counter=expected_counter,
        )
        # One explicit transaction, so the change counter advances exactly once.
        conn.isolation_level = None
        conn.execute("BEGIN")
        conn.execute(f"CREATE TABLE {SOURCE_TABLE} (record TEXT NOT NULL)")
        conn.execute(
            f"INSERT INTO {SOURCE_TABLE} (record) VALUES (?)",
            (record.model_dump_json(),),
        )
        conn.execute("COMMIT")
        conn.close()
        if _change_counter(tmp_path) != expected_counter:
            raise RuntimeError(
                f"{tmp_path}: the record transaction left the sqlite change counter at "
                f"{_change_counter(tmp_path)}, not {expected_counter}"
            )
        os.chmod(tmp_path, 0o644)
    except BaseException:
        os.remove(tmp_path)
        raise
    return tmp_path


def _read_record_json(db_path: str, name: str | None = None) -> str | None:
    """The raw record JSON stored inside ``db_path``, or None when it carries none.

    A file sqlite cannot read raises ``sqlite3.DatabaseError``; the connection is
    closed on every path.
    """
    with closing(sqlite3.connect(_ro_uri(db_path), uri=True)) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (SOURCE_TABLE,),
        ).fetchone()
        if has_table is None:
            return None
        rows = conn.execute(f"SELECT record FROM {SOURCE_TABLE}").fetchall()
    if len(rows) != 1:
        raise GenomeDatabaseSourceError(
            f"{name or db_path}: {SOURCE_TABLE} holds {len(rows)} rows, expected "
            "exactly 1"
        )
    return str(rows[0][0])


def read_genome_database_record(db_path: str) -> GenomeDatabaseRecord | None:
    """The record stored inside ``db_path``, or None when it carries none."""
    raw = _read_record_json(db_path)
    if raw is None:
        return None
    return GenomeDatabaseRecord.model_validate_json(raw)


def validation_summary(errors: Sequence[Mapping[str, Any]]) -> str:
    """``loc: msg`` for each pydantic error, sorted by location then message, so the
    summary is the same whatever order a pydantic version reports the errors in.
    """
    keyed = sorted(
        (tuple(str(part) for part in err["loc"]), str(err["msg"])) for err in errors
    )
    return "; ".join(f"{'.'.join(loc)}: {msg}" for loc, msg in keyed)


def record_version(db_path: str, raw: str) -> int:
    """The ``version`` of the record JSON ``raw`` read from ``db_path``; 0 when absent
    (records written before versioning). A version newer than this checkout's raises
    :class:`GenomeDatabaseVersionError`; a record that is not valid JSON, not a JSON
    object, or whose version is not an integer, raises
    :class:`GenomeDatabaseRecordError`.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenomeDatabaseRecordError(
            f"{db_path} carries a record that is not valid JSON ({exc}). "
            f"{_UPDATE_CHECKOUT}"
        ) from exc
    if not isinstance(data, dict):
        raise GenomeDatabaseRecordError(
            f"{db_path} carries a record that is not a JSON object ({type(data).__name__})"
            f". {_UPDATE_CHECKOUT}"
        )
    version = data.get("version", 0)
    if not isinstance(version, int) or isinstance(version, bool):
        raise GenomeDatabaseRecordError(
            f"{db_path} carries a record whose version is {version!r}, not an "
            f"integer. {_UPDATE_CHECKOUT}"
        )
    if version > RECORD_VERSION:
        raise GenomeDatabaseVersionError(
            f"{db_path} carries a record of version {version}, written by newer code "
            f"than this checkout (which reads version {RECORD_VERSION}); refusing to "
            f"replace it. {_UPDATE_CHECKOUT}"
        )
    return version


def refuse_newer_record(db_path: str) -> None:
    """Refuse an explicit rebuild over a database written by newer code.

    Raises :class:`GenomeDatabaseVersionError` when ``db_path`` carries a record newer
    than this checkout (``overwrite=True`` must not downgrade it), and
    :class:`GenomeDatabaseRecordError` when its record is not valid JSON, not a JSON
    object, or has a non-integer version (it may be newer code's). Returns when
    ``db_path`` is absent, carries no record, or is a file sqlite cannot read (a
    killed in-place rebuild): such a file carries no newer record, so the rebuild
    proceeds and repairs it.
    """
    if not osp.exists(db_path):
        return
    try:
        raw = _read_record_json_checked(db_path)
    except sqlite3.DatabaseError as exc:  # damaged: it carries no newer record
        require_damage(db_path, exc)
        return
    if raw is not None:
        record_version(db_path, raw)


def _committed_record_json(db_path: str) -> str | None:
    """The record as last committed: ``db_path`` and its hot journal are copied into
    the temp dir (``torchcell-genome-<host>-<pid>-peek<random>.db``, so a killed
    process's copy is swept), the copy is opened read-write so sqlite rolls the
    journal back, and the record is read from it. None when the rolled-back copy has
    no readable record. A defective record table names ``db_path``, not the copy.
    """
    fd, copy_path = tempfile.mkstemp(
        prefix=f"torchcell-genome-{socket.gethostname()}-{os.getpid()}-peek",
        suffix=".db",
    )
    os.close(fd)
    try:
        try:
            shutil.copyfile(db_path, copy_path)
            shutil.copyfile(db_path + "-journal", copy_path + "-journal")
        except FileNotFoundError as exc:
            raise _vanished(exc.filename or db_path, exc) from exc
        try:
            with closing(sqlite3.connect(copy_path)) as conn:
                conn.execute("SELECT 1 FROM sqlite_master").fetchone()
            return _read_record_json(copy_path, name=db_path)
        except sqlite3.DatabaseError:
            return None
    finally:
        _remove_if_present(copy_path)
        _remove_if_present(copy_path + "-journal")


def _read_record_json_checked(db_path: str, name: str | None = None) -> str | None:
    """:func:`_read_record_json`, except that a hot rollback journal (which makes the
    read fail with ``SQLITE_READONLY_ROLLBACK``) does not hide a record this checkout
    must refuse: the committed record is checked first (a newer version or an
    unreadable record raises by name, naming ``db_path``), then the error is
    re-raised for the caller to treat as damage. The record checked is the committed
    one: the file plus its journal, rolled back in a private copy.
    """
    try:
        return _read_record_json(db_path, name=name)
    except sqlite3.DatabaseError as exc:
        if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_READONLY_ROLLBACK:
            peeked = _committed_record_json(db_path)
            if peeked is not None:
                check_record(name or db_path, peeked)
        raise


def check_record(db_path: str, raw: str) -> GenomeDatabaseRecord | str:
    """The validated record in ``raw`` (read from ``db_path``), or the untrusted
    reason when it is an older version. A newer version raises
    :class:`GenomeDatabaseVersionError`; a record this checkout cannot read at its own
    version raises :class:`GenomeDatabaseRecordError`.
    """
    version = record_version(db_path, raw)
    if version < RECORD_VERSION:
        return f"its record is version {version}, older than {RECORD_VERSION}"
    try:
        return GenomeDatabaseRecord.model_validate_json(raw)
    except ValidationError as exc:
        summary = validation_summary(exc.errors())
        raise GenomeDatabaseRecordError(
            f"{db_path} carries a version {version} record that this checkout cannot "
            f"read ({exc.error_count()} errors: {summary}); it was written by code "
            f"with a different record schema at the same version. {_UPDATE_CHECKOUT}"
        ) from exc


def untrusted_reason(
    db_path: str,
    expected: GenomeDatabaseSource,
    rebuild_call: str,
    name: str | None = None,
) -> str | None:
    """Why the existing ``db_path`` cannot be trusted, or None when it can.

    Untrusted (returned as a reason): no record, which is every database built before
    2026.10.01 and every one rebuilt in place by a process still running older code; a
    record of a lower (or absent) :data:`RECORD_VERSION`; row counts that differ from
    the record (rows deleted in place by such a process); or a change counter that
    differs from the record (any other write in place). A record of a higher version
    raises :class:`GenomeDatabaseVersionError`, and a record for a different source
    raises :class:`GenomeDatabaseSourceError`: the pinned GFF changed, which is a real
    source change, not a migration. ``name`` is the path named in those refusals
    (``data.db`` when ``db_path`` is a private copy of it).
    """
    shown = name or db_path
    try:
        raw = _read_record_json_checked(db_path, name=shown)
    except sqlite3.DatabaseError as exc:
        require_damage(db_path, exc)
        return f"sqlite cannot read it ({exc})"
    if raw is None:
        return f"it carries no {SOURCE_TABLE} record"
    record = check_record(shown, raw)
    if isinstance(record, str):
        return record
    if record.source != expected:
        raise GenomeDatabaseSourceError(
            f"{shown} was built from {record.source.model_dump()} but this genome's "
            f"source is {expected.model_dump()}. Rebuild it deliberately, once, while "
            f"no job reads it: {rebuild_call}"
        )
    try:
        with closing(sqlite3.connect(_ro_uri(db_path), uri=True)) as conn:
            featuretype_counts, relations_count = _database_counts(conn)
    except sqlite3.DatabaseError as exc:
        require_damage(db_path, exc)
        return f"sqlite cannot read it ({exc})"
    if (featuretype_counts, relations_count) != (
        record.featuretype_counts,
        record.relations_count,
    ):
        return (
            "its row counts differ from its record (features "
            f"{sum(featuretype_counts.values())} vs "
            f"{sum(record.featuretype_counts.values())} recorded, relations "
            f"{relations_count} vs {record.relations_count} recorded)"
        )
    counter = _change_counter(db_path)
    if counter != record.change_counter:
        return (
            "it was written in place after its build (sqlite change counter "
            f"{counter}, {record.change_counter} recorded)"
        )
    if _journal_moved_to_kept(db_path):
        return (
            "its hot journal was moved beside data.db.untrusted by a migration that "
            "did not finish"
        )
    return None


def _journal_moved_to_kept(db_path: str) -> bool:
    """``data.db`` is byte-identical to the kept copy, a kept journal sits beside the
    copy, and rolling the kept copy back with that journal changes it: a migrator
    moved the hot journal and was killed before renaming the fresh build in. The torn
    file's cheap checks pass, so this is what marks it untrusted. A journal of
    ``data.db``'s own does not clear it: a pre-2026.10.01 writer holding a
    transaction on the torn file has one, and the file is still torn (a ``data.db``
    byte-equal to a kept file whose rollback changes it is torn whatever companion it
    has). The comparison is a full byte read of both files, and when they are equal
    the rollback runs on a private copy in the temp dir.
    """
    if osp.basename(db_path) != GENOME_DB_FILENAME:
        return False
    kept = osp.join(osp.dirname(db_path), UNTRUSTED_DB_FILENAME)
    if not osp.isfile(kept + "-journal"):
        return False
    if not osp.isfile(kept):
        return False
    try:
        if not filecmp.cmp(db_path, kept, shallow=False):
            return False
    except PermissionError:  # another user's kept copy: not this file's journal pair
        return False
    except FileNotFoundError as exc:
        raise _vanished(exc.filename or db_path, exc) from exc
    return not _rollback_equals(kept, db_path)


def _rollback_equals(kept: str, db_path: str) -> bool:
    """Whether ``kept`` rolled back with its journal (in a private copy in the temp
    dir) is byte-identical to ``db_path``: then the journal undoes nothing and
    ``db_path`` is not a torn file (a fresh build can equal a kept copy whose hot
    journal never reached its pages).
    """
    fd, copy_path = tempfile.mkstemp(
        prefix=f"torchcell-genome-{socket.gethostname()}-{os.getpid()}-peek",
        suffix=".db",
    )
    os.close(fd)
    try:
        try:
            shutil.copyfile(kept, copy_path)
            shutil.copyfile(kept + "-journal", copy_path + "-journal")
        except FileNotFoundError as exc:
            raise _vanished(exc.filename or kept, exc) from exc
        try:
            with closing(sqlite3.connect(copy_path)) as conn:
                conn.execute("SELECT 1 FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError:
            return False
        try:
            return filecmp.cmp(copy_path, db_path, shallow=False)
        except FileNotFoundError as exc:
            raise _vanished(db_path, exc) from exc
    finally:
        _remove_if_present(copy_path)
        _remove_if_present(copy_path + "-journal")


def _has_hot_journal(db_path: str) -> bool:
    """Whether a journal beside ``db_path`` is hot: the read-only read fails with
    ``SQLITE_READONLY_ROLLBACK`` (it must be rolled back before the file is read).
    """
    if not osp.isfile(db_path + "-journal"):
        return False
    try:
        _read_record_json(db_path)
    except sqlite3.DatabaseError as exc:
        return (
            getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_READONLY_ROLLBACK
        )
    return False


def _keep_copy(copy_path: str, kept: str, db_path: str) -> bool:
    """Move ``copy_path`` (a copy of the untrusted ``db_path``) onto ``kept``, dropping
    the earlier kept copy's companions first so a kept file and its journal always
    belong together. Must run under :func:`_root_lock`.

    When ``kept`` already holds the same bytes, it is left as is with its companions
    and True is returned: a migration killed after keeping the file (and possibly
    moving its journal) left exactly this pair, and replacing it would drop that
    journal. Otherwise ``copy_path`` replaces it and False is returned. A kept path
    that is not a regular file is refused by name; one this process cannot read, or
    one removed (by hand) after the check that it is a file, counts as a different
    file.
    """
    if osp.lexists(kept) and (osp.islink(kept) or not osp.isfile(kept)):
        raise GenomeDatabaseUnavailableError(
            f"{kept} is not a regular file, so the untrusted {db_path} cannot be kept "
            "there; every file is left alone. Remove or rename it, then retry."
        )
    try:
        same = osp.isfile(kept) and filecmp.cmp(copy_path, kept, shallow=False)
    except PermissionError:  # another user's kept copy: a different file
        same = False
    except FileNotFoundError:  # removed since the isfile check: nothing to compare
        same = False
    if same:
        return True
    for suffix in _COMPANION_SUFFIXES:
        _remove_if_present(kept + suffix)
    os.replace(copy_path, kept)
    return False


def _copy_preserving_mode(db_path: str, copy_path: str) -> None:
    """Copy ``db_path`` onto ``copy_path`` (a mkstemp file, mode 0600) and give the copy
    the mode of the file it preserves, so other users of a group-writable root can
    read the kept copy as they could the original.
    """
    try:
        shutil.copyfile(db_path, copy_path)
        os.chmod(copy_path, stat.S_IMODE(os.stat(db_path).st_mode))
    except FileNotFoundError as exc:
        raise _vanished(db_path, exc) from exc


def _vanished(path: str, exc: FileNotFoundError) -> GenomeDatabaseUnavailableError:
    return GenomeDatabaseUnavailableError(
        f"{path} vanished while this process was reading, migrating or rebuilding "
        f"it ({exc}); another process (pre-2026.10.01 code) is rebuilding it. Every "
        "file is left alone. Retry when it has finished."
    )


def _kept_wording(kept: str, as_is: bool, removed: Sequence[str]) -> str:
    """How the one WARNING of a migration or rebuild names the kept pair: replaced,
    or left as is (:func:`_keep_copy`) together with every companion of data.db that
    :func:`install_genome_database` removed instead of moving it over the kept one.
    """
    if not as_is:
        return f"{kept} (replacing any earlier one)"
    wording = f"{kept} (which already held these bytes and was left in place"
    if removed:
        wording += (
            f"; {', '.join(removed)} beside data.db was removed instead of "
            "overwriting the kept companion"
        )
    return wording + ")"


def _identity(path: str) -> tuple[int, int, int]:
    try:
        st = os.stat(path)
    except FileNotFoundError as exc:
        raise _vanished(path, exc) from exc
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def _content_digest_or_none(db_path: str) -> str | None:
    """:func:`database_content_digest` of an untrusted file, or None when sqlite
    cannot read its rows (then it differs from any fresh build and is kept).
    """
    try:
        return database_content_digest(db_path)
    except sqlite3.DatabaseError as exc:
        require_damage(db_path, exc)
        return None


def migrate_genome_database(
    gff_path: str,
    db_path: str,
    expected: GenomeDatabaseSource,
    rebuild_call: str,
    reason: str,
) -> None:
    """Replace the untrusted ``db_path`` with a fresh build, keeping at most one file.

    A fresh build is written beside ``db_path`` and compared with it by
    :func:`database_content_digest`; when the rows differ, ``db_path`` is copied (with
    its mode). The keep and install steps then run under :func:`_root_lock`: if
    ``db_path`` is trusted again (another migrator finished first) nothing is done;
    if it was replaced or written since it was inspected, the migration is refused by
    name; identical rows (an old-code rebuild without a record) replace ``db_path``
    and nothing is kept; different rows (deleted or rewritten in place) are kept as
    ``data.db.untrusted`` (:func:`_keep_copy`) before the fresh build is renamed onto
    ``db_path`` (:func:`install_genome_database`, which moves any companion journal
    beside the kept copy). The path never goes missing and readers holding the old
    inode keep it. One WARNING names the case. Every temporary file is removed on
    every path.
    """
    db_dir = osp.dirname(db_path)
    tmp_path = write_genome_database(gff_path, db_dir, expected)
    copy_path: str | None = None
    try:
        inspected = _identity(db_path)
        rows_equal = _content_digest_or_none(db_path) == database_content_digest(
            tmp_path
        )
        if not rows_equal:
            copy_path = _temp_in(db_dir, UNTRUSTED_DB_FILENAME)
            _copy_preserving_mode(db_path, copy_path)
        with _root_lock(db_dir):
            if untrusted_reason(db_path, expected, rebuild_call) is None:
                return  # another migrator finished first; it kept the original
            if _identity(db_path) != inspected:
                raise GenomeDatabaseUnavailableError(
                    f"{db_path} was replaced or written by another process while this "
                    "one was migrating it; every file is left alone. Retry."
                )
            if copy_path is None:
                install_genome_database(tmp_path, db_path)
                log.warning(
                    "genome database %s was not trusted (%s); its rows equal a fresh "
                    "build, so it was replaced by the recorded build and nothing was "
                    "kept",
                    db_path,
                    reason,
                )
                return
            copy_trusted = (
                untrusted_reason(copy_path, expected, rebuild_call, name=db_path)
                is None
            )
            kept = osp.join(db_dir, UNTRUSTED_DB_FILENAME)
            as_is = _keep_copy(copy_path, kept, db_path)
            removed = install_genome_database(tmp_path, db_path, kept, as_is)
        if copy_trusted:
            # The file's own pages match its record; only a companion journal beside
            # it (moved to ``kept``-journal with it) made it untrusted.
            log.warning(
                "genome database %s was not trusted (%s); its own pages match its "
                "record and only a companion journal made it unreadable, so the pair "
                "was kept as %s and replaced by the recorded build",
                db_path,
                reason,
                _kept_wording(kept, as_is, removed),
            )
            return
        log.warning(
            "genome database %s was not trusted (%s); its rows differ from a fresh "
            "build, so it was kept as %s and replaced by the recorded build",
            db_path,
            reason,
            _kept_wording(kept, as_is, removed),
        )
    finally:
        for path in (tmp_path, copy_path):
            if path is not None and osp.exists(path):
                os.remove(path)


def rebuild_genome_database(
    gff_path: str, db_path: str, expected: GenomeDatabaseSource
) -> None:
    """The explicit rebuild (``overwrite=True``): build, then under
    :func:`_root_lock` rename the build onto ``db_path``. When a HOT journal sits
    beside the old file, the old file and its journal are kept as a pair (as a
    migration would) instead of removing the journal, so no crash point leaves a
    torn file without its journal; one WARNING names the kept pair.
    """
    db_dir = osp.dirname(db_path)
    tmp_path = write_genome_database(gff_path, db_dir, expected)
    copy_path: str | None = None
    try:
        with _root_lock(db_dir):
            if not _has_hot_journal(db_path):
                install_genome_database(tmp_path, db_path)
                return
            copy_path = _temp_in(db_dir, UNTRUSTED_DB_FILENAME)
            _copy_preserving_mode(db_path, copy_path)
            kept = osp.join(db_dir, UNTRUSTED_DB_FILENAME)
            as_is = _keep_copy(copy_path, kept, db_path)
            removed = install_genome_database(tmp_path, db_path, kept, as_is)
        log.warning(
            "genome database %s had a hot journal; the explicit rebuild kept the old "
            "file with its journal as %s and installed the recorded build",
            db_path,
            _kept_wording(kept, as_is, removed),
        )
    finally:
        for path in (tmp_path, copy_path):
            if path is not None and osp.exists(path):
                os.remove(path)


def _meta_rows(db_path: str) -> int:
    """Rows in gffutils' ``meta`` table of ``db_path`` (read-only)."""
    with closing(sqlite3.connect(_ro_uri(db_path), uri=True)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0])


def _meta_rows_or_zero(db_path: str) -> int:
    """:func:`_meta_rows`, or 0 when ``db_path`` has no readable ``meta`` table."""
    try:
        return _meta_rows(db_path)
    except sqlite3.DatabaseError:
        return 0


def _remove_private_copy(path: str, owner_pid: int) -> None:
    """Delete an instance's private database copy, only in the process that made it."""
    if os.getpid() == owner_pid and osp.exists(path):
        os.remove(path)


def _restore_annotated_genome[GenomeT: "AnnotatedGenome[Any]"](
    cls: type[GenomeT], init_kwargs: dict[str, Any], private_db_path: str | None
) -> GenomeT:
    """Unpickling target: reopen with ``overwrite=False``, on the same database file."""
    genome = cls(**init_kwargs)
    if private_db_path is not None:
        genome._db_connection_manager = GffutilsConnectionManager(private_db_path)
    return genome


class GenomeReleaseFiles(BaseModel):
    """The members of an assembly set an :class:`AnnotatedGenome` reads, by filename.

    Each is resolved through the genomes tier (sha256-verified) at construction. The
    GFF is the source of ``data.db``; the DNA FASTA gives the chromosome sequences; the
    protein and CDS FASTAs are keyed by gene id. ``cds_fasta`` is required and may be
    ``None``, which states that the release ships no CDS FASTA: the subclass then
    derives the CDS sequences in :meth:`AnnotatedGenome._read_sequences` (an NCBI
    GenBank assembly carries them as the CDS features of its flat file).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dna_fasta: str
    gff: str
    protein_fasta: str
    cds_fasta: str | None


@define
class AnnotatedGene(Gene):
    """One gene of an :class:`AnnotatedGenome`, resolved from its GFF database and FASTAs.

    The organism's conventions are hooks: :attr:`GO_ATTRIBUTE` (the GFF attribute whose
    ``GO:`` values are the gene's GO terms), :meth:`seqid_to_chromosome` (GFF seqid to
    the integer chromosome key), :meth:`coding_feature` (the feature whose coordinates
    give the gene its sequence; the gene row by default) and :meth:`annotate` (the GFF
    attributes the gene exposes).
    """

    #: The GFF attribute whose ``GO:``-prefixed values are the gene's GO terms.
    GO_ATTRIBUTE: ClassVar[str]
    #: What the GFF writes before a gene id in the gene feature's ``ID``: ``""`` when
    #: the ``ID`` is the gene id itself (SGD), ``"gene-"`` for NCBI's ``ID=gene-b0002``.
    FEATURE_ID_PREFIX: ClassVar[str] = ""

    id: str = field(repr=False)
    db: Any = field(repr=False)
    fasta_dna: dict[str, Any] = field(repr=False)
    fasta_protein: dict[str, Any] = field(repr=False)
    fasta_cds: dict[str, Any] = field(repr=False)
    chr_to_nc: dict[int, str] = field(repr=False)
    chromosome_lengths: dict[int, int] = field(repr=False)
    # below are set in __attrs_post_init__
    chromosome: int = field(default=None)
    start: int = field(default=None)
    end: int = field(default=None)
    seq: str = field(default=None, repr=True)
    feature: Feature = field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        """Resolve the gene feature, coordinates, sequence, and GO terms from the DB."""
        feature = self.coding_feature()
        gene_feature = self.db[self.feature_id]

        #
        self.id = self.id
        # chromosome
        self.chromosome = self.seqid_to_chromosome(gene_feature.seqid)
        # others
        self.start = feature.start
        self.end = feature.end
        self.strand = feature.strand
        if self.strand not in ("+", "-"):
            raise ValueError(
                f"Gene {self.id} has strand {self.strand!r}; a gene needs '+' or '-' "
                "to orient its sequence and windows"
            )

        # dna sequence
        chr = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            self.seq = str(self.fasta_dna[chr].seq[self.start - 1 : self.end])
        elif self.strand == "-":
            self.seq = str(
                self.fasta_dna[chr].seq[self.start - 1 : self.end].reverse_complement()
            )

        # protein sequence
        self.protein = self.fasta_protein.get(self.id)
        # cds sequence
        self.cds: Any = self.fasta_cds.get(self.id)

        # Must use the gene since it has all of the annotations, not the CDS feature
        self.annotate(gene_feature)

    @classmethod
    @abstractmethod
    def seqid_to_chromosome(cls, seqid: str) -> int:
        """The integer chromosome key of the GFF ``seqid`` a gene sits on."""

    @property
    def feature_id(self) -> str:
        """The ``ID`` of this gene's feature in ``data.db``: :attr:`FEATURE_ID_PREFIX`
        followed by the gene id.
        """
        return self.FEATURE_ID_PREFIX + self.id

    def coding_feature(self) -> Feature:
        """The feature whose coordinates give the gene its sequence: the gene row."""
        return self.db[self.feature_id]

    def annotate(self, gene_feature: Feature) -> None:
        """Copy the gene's GFF3 reserved attributes and its GO terms.

        ``Alias``, ``Name`` and ``Note`` are GFF3 reserved attributes; ``go`` is the
        ``GO:`` values of :attr:`GO_ATTRIBUTE` (None when the gene carries no such
        attribute, empty when it carries no ``GO:`` value). A subclass adds the
        attributes its source exposes by extending this method.
        """
        self.alias = gene_feature.attributes.get("Alias", None)
        self.name = gene_feature.attributes.get("Name", None)
        self.note = gene_feature.attributes.get("Note", None)

        # Handle GO terms
        go_terms = gene_feature.attributes.get(self.GO_ATTRIBUTE, None)
        if go_terms is not None:
            self.go = SortedSet([term for term in go_terms if term.startswith("GO:")])
        else:
            self.go = None

    @property
    def alias_to_systematic(self) -> dict[str, str]:
        """Return a mapping from each gene alias to its systematic gene ID."""
        alias_map = {}
        for gene_id in self.gene_set:  # type: ignore[attr-defined]  # references Genome API on Gene (pre-existing)
            gene = self[gene_id]  # type: ignore[index]  # references Genome API on Gene (pre-existing)
            if gene and gene.alias:
                for alias in gene.alias:
                    if alias in alias_map:
                        log.warning(
                            f"Duplicate alias {alias} mapped to multiple genes."
                        )
                    alias_map[alias] = gene_id
        return alias_map

    @property
    def codon_frequency(self) -> SortedDict[str, float]:
        """Return the codon-usage frequency of the gene's CDS sequence."""
        codon_frequency = compute_codon_frequency(self.cds.seq)
        return codon_frequency

    def window(self, window_size: int, is_max_size: bool = True) -> DnaWindowResult:
        """Return a DNA sequence window centered on the gene's coding region."""
        if is_max_size:
            start_window, end_window = calculate_window_bounds(
                start=self.start - 1,
                end=self.end,
                strand=self.strand,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )

        else:
            start_window, end_window = calculate_window_bounds_symmetric(
                start=self.start - 1,
                end=self.end,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
            )
        return DnaWindowResult(
            id=self.id,
            chromosome=self.chromosome,
            strand=self.strand,
            start=self.start,
            end=self.end,
            seq=seq,
            start_window=start_window,
            end_window=end_window,
        )

    def window_five_prime(  # type: ignore[override]  # adds include_start_codon kwarg vs base ABC
        self,
        window_size: int,
        include_start_codon: bool = False,
        allow_undersize: bool = False,
    ) -> DnaWindowResult:
        """Return the sequence window upstream of the gene's 5' start.

        The window never includes a base of the gene unless ``include_start_codon``
        is set, in which case it ends with the 3 bases of the start codon. On ``+``
        it is ``[start0 - w, start0)`` (``[start0 + 3 - w, start0 + 3)`` with the
        codon), where ``start0 = self.start - 1`` is the 0-based first base; on ``-``
        it is the reverse complement of ``[end, end + w)`` (``[end - 3, end - 3 + w)``
        with the codon).
        """
        # offset for gff file 1
        start = self.start - 1
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            if include_start_codon:
                start = start + 3
            start_window = start - window_size
            end_window = start
            if start_window < 0 and allow_undersize:
                start_window = 0
                end_window = start
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"five prime size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            if include_start_codon:
                end = self.end - 3
            else:
                end = self.end
            start_window = end
            end_window = end + window_size
            if (
                end_window > self.chromosome_lengths[self.chromosome]
                and allow_undersize
            ):
                end_window = self.chromosome_lengths[self.chromosome]
            elif (
                end_window > self.chromosome_lengths[self.chromosome]
                and not allow_undersize
            ):
                outside = abs(end_window - self.chromosome_lengths[self.chromosome])
                raise ValueError(
                    f"five prime size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )

            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
            )
        return DnaWindowResult(
            id=self.id,
            seq=seq,
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand,
            start_window=start_window,
            end_window=end_window,
        )

    def window_three_prime(  # type: ignore[override]  # adds include_stop_codon kwarg vs base ABC
        self,
        window_size: int,
        include_stop_codon: bool = False,
        allow_undersize: bool = False,
    ) -> DnaWindowResult:
        """Return the sequence window downstream of the gene's 3' end."""
        # offset for gff file 1
        start = self.start - 1
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            if include_stop_codon:
                end = self.end - 3
            else:
                end = self.end
            start_window = end
            end_window = end + window_size
            if (
                end_window > self.chromosome_lengths[self.chromosome]
                and allow_undersize
            ):
                end_window = self.chromosome_lengths[self.chromosome]
            elif (
                end_window > self.chromosome_lengths[self.chromosome]
                and not allow_undersize
            ):
                outside = abs(end_window - self.chromosome_lengths[self.chromosome])
                raise ValueError(
                    f"3utr size ({window_size}) too large"
                    f"('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            if include_stop_codon:
                start = start + 3
            else:
                start = start
            start_window = start - window_size
            end_window = start
            if start_window < 0 and allow_undersize:
                start_window = 0
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"3utr size ({window_size}) too large"
                    f"('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
            )

        return DnaWindowResult(
            id=self.id,
            seq=seq,
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand,
            start_window=start_window,
            end_window=end_window,
        )

    def __repr__(self) -> str:
        """Return a string with the gene's ID, location, strand, and sequence."""
        return f"DnaSelectionResult(id={self.id}, chromosome={self.chromosome}, strand={self.strand}, start={self.start}, end={self.end},  seq={self.seq})"


@define(eq=False)
class AnnotatedGenome[GeneT: AnnotatedGene](Genome):
    """A genome built from one GFF3 + FASTA release in the genomes tier, with GO.

    ``<genome_root>/data.db`` is the gffutils database built from the pinned GFF. It
    is shared by every process on the same ``genome_root``, and this code never writes
    it after it is built (in single-version use; the mixed-version exception is under
    Known limits): every build goes to a unique temporary file in ``genome_root`` that
    is renamed into place (:func:`write_genome_database`). The ``drop_*`` methods
    (:meth:`drop_empty_go`, and a subclass's own, such as
    ``SCerevisiaeGenome.drop_chrmt``) and :meth:`remove_deprecated_go_terms` write to
    a private copy owned by this instance, ``torchcell-genome-<host>-<pid>-<random>.db`` in
    ``tempfile.gettempdir()`` (so ``TMPDIR`` controls where it goes), made on its
    first write and deleted when the instance is collected in the process that made
    it. A process killed before that leaves its copy behind; every creation of a
    private copy removes this host's and this user's copies whose pid is dead. The
    writes are logged on the instance, so a pickled copy whose parent's private file
    is gone rebuilds its own from the shared file at its first read.

    ``genome_root`` must exist unless ``overwrite=True``. ``overwrite`` decides how
    the constructor treats the shared file:

    * ``False`` (the default): build it when absent. When present, open it if its
      record (:class:`GenomeDatabaseRecord`) names this genome's source and its row
      counts and change counter still match. A record for a different source (the
      pinned GFF changed) raises :class:`GenomeDatabaseSourceError`; a record of a
      newer version raises :class:`GenomeDatabaseVersionError`; a record this
      checkout cannot read at its own version raises
      :class:`GenomeDatabaseRecordError`. A database with no record, an older
      record, rows or a change counter that no longer match the record, or a file
      sqlite cannot read at all (an in-place rebuild by pre-2026.10.01 code that was
      killed mid-write) is untrusted and migrated (:func:`migrate_genome_database`).
      This is a stated one-time migration of a database that cannot be trusted, not a
      fallback: the replacement is built from the same sha256-pinned GFF, and an
      unreadable file is kept as ``data.db.untrusted``. When the root is not
      writable, a build or migration raises :class:`GenomeRootNotWritableError` and
      the untrusted file is never opened by gffutils.
    * ``True``: rebuild it (atomically), also over an unreadable file, unless it
      carries a record of a newer version (:class:`GenomeDatabaseVersionError`) or
      one this checkout cannot read (:class:`GenomeDatabaseRecordError`). Pass it
      only deliberately.

    A sqlite error that says nothing about the rows (the file is locked by another
    process, cannot be opened, or an I/O error occurred) raises
    :class:`GenomeDatabaseUnavailableError` on either path and leaves every file
    alone. A companion journal (``-journal``, ``-wal``, ``-shm``) beside the old file
    is never left beside a new build: it follows the kept copy or is removed.

    The open-time check reads the record, the per-featuretype counts (served from
    covering indexes) and the change counter, not every page. Damage made out of band
    in pages it does not touch (for example a zeroed ``features`` table page) opens
    as trusted, and the first read that reaches the damaged page raises
    ``sqlite3.DatabaseError: database disk image is malformed``. Construct once with
    ``overwrite=True`` to repair it. No old-code kill produced this state in testing,
    and a full ``integrity_check`` at every open is not run because of its cost.

    Construction also removes this host's ``data.db.*.building`` files whose writer
    pid is dead (a build killed mid-way), when the root is writable.

    The root lock (an exclusive ``flock`` on the genome-root directory) is taken only
    by a migration and by an explicit rebuild, never by a reader of a trusted file. It
    has no timeout: a migrator or rebuilder waits behind a stopped holder (for
    example a SIGSTOPped process) for as long as the holder is stopped. A holder
    killed with SIGKILL releases it, and a child forked while the lock is held keeps
    it until the child exits.

    A ``data.db`` that is a symlink to a trusted database is followed and read. When
    the target is untrusted, or ``overwrite=True``, the build is renamed onto the
    LINK, which becomes a regular file; the target is never written.

    When a migration kept a pair (``data.db.untrusted`` and
    ``data.db.untrusted-journal``) and ``data.db`` is byte-equal to the kept copy
    while the kept journal undoes nothing, ``data.db`` is trusted, but every
    construction pays a full byte comparison of the two files plus a rollback of the
    kept pair on a private copy in the temp dir (about +30 ms per construction at
    real size, measured by the eighth review) for as long as the kept pair exists.
    The way out is to move ``data.db.untrusted`` and ``data.db.untrusted-journal`` out
    of the root by hand once they are no longer wanted as evidence.

    Known limits:

    * A kept journal that is foreign (another database's) or has a corrupted size
      field makes the rollback check report ``data.db`` as torn at every
      construction, so every construction migrates again with a WARNING and a reader
      of a read-only root gets :class:`GenomeRootNotWritableError`. Only tampering
      with the kept files produced this; no natural path to it was found.
    * A process killed with SIGKILL inside that rollback check leaves its private
      copy (about 15 MB) in the temp dir. Later constructions do not remove it; only
      the dead-pid sweep run by the next private-copy creation (a ``drop_*`` write)
      does.
    * A kept journal this user cannot read beside a kept copy this user can read
      raises ``PermissionError``. sqlite gives a journal the database's mode, so this
      arises only when someone changes the journal's mode by hand.
    * A ``data.db`` this process cannot open that is unlinked between
      :func:`require_damage`'s existence check and its ``stat`` raises a bare
      ``FileNotFoundError`` instead of the named error. It needs two unlinks around
      one expression; no realistic path to it was found.
    * Mixed-version use (found by the tenth review's runs at e65ca38b0). The one
      read-write open of the shared path is gffutils' ``FeatureDB`` connection
      (``GffutilsConnectionManager(db_path)``), which origin/main has too; this code
      adds no other. When pre-2026.10.01 code unlinks ``data.db`` and writes a new
      file with a hot or live journal at the same path, a plain read on an instance
      holding the old inode makes sqlite recovery roll back and delete that journal,
      on this code and on main alike. So "the shared ``data.db`` is never written
      after it is built" and "never rolls a journal back into the shared file" hold
      for this code's own operations and for single-version use, and do not hold in
      that mixed-version state. The next construction by this code finds the result
      untrusted, migrates it and keeps it; it never trusts it.
    * A fresh instance whose first operation is :meth:`remove_deprecated_go_terms`
      on a file with an empty or partial ``meta`` table raises the same unnamed
      ``TypeError`` as main. No production caller does this.
    * A ``drop_*`` write waits for as long as another process holds an EXCLUSIVE
      sqlite lock taken between the drop's read and its private copy (the sqlite
      backup API retries on BUSY; the tenth review measured 30.0 s).
    * The named error for a file that vanished on the private-copy path advises
      fixing its permissions, which is the wrong advice for a file that is gone.
    * The rebuild WARNING in the kept-as-is case says the old file was kept with its
      journal and then that the journal was removed.
    """

    #: The assembly set in the genomes tier this class reads its release files from.
    ASSEMBLY_SET: ClassVar[str]
    #: The release whose files the constructor resolves.
    GENOME_VERSION: ClassVar[str]
    #: The gene-like LOCUS feature types of the annotation (a deletion or perturbation
    #: can target these): the only features :attr:`feature_index` indexes, so a
    #: non-locus feature id (a region, CDS, mRNA, ...) never shadows a real gene name
    #: during resolution. ``"gene"`` is the protein-coding universe (== gene_set).
    LOCUS_FEATURE_TYPES: ClassVar[frozenset[str]]
    #: The annotation as a resolution ``note`` names a valid non-gene locus
    #: ("valid <name> <featuretype>, not a gene feature").
    ANNOTATION_NAME: ClassVar[str]
    #: The release as a RETIRED resolution's ``note`` names it ("not found in <release>").
    ANNOTATION_RELEASE: ClassVar[str]

    genome_root: str = field(init=True, repr=False)
    overwrite: bool = field(init=True, repr=True, default=False)
    fasta_dna: dict[str, Any] = field(init=False, default=None, repr=False)
    chr_to_nc: dict[int, str] = field(init=False, default=None, repr=False)
    nc_to_chr: dict[str, int] = field(init=False, default=None, repr=False)
    chr_to_len: dict[int, int] = field(init=False, default=None, repr=False)
    _gene_set: GeneSet = field(init=False, default=None, repr=False)
    _dna_fasta_path: str = field(init=False, default=None, repr=False)
    _protein_fasta_path: str = field(init=False, default=None, repr=False)
    _cds_fasta_path: str | None = field(init=False, default=None, repr=False)
    _gff_path: str = field(init=False, default=None, repr=False)
    _go: SortedSet[str] = field(init=False, default=None, repr=False)
    _go_genes: SortedDict[str, SortedSet[str]] = field(
        init=False, default=None, repr=False
    )
    _alias_to_systematic: dict[str, list[str]] = field(
        init=False, default=None, repr=False
    )
    # Cached all-feature index (upper-cased) backing resolve_gene_name.
    _feature_index: dict[str, Any] | None = field(init=False, default=None, repr=False)
    # Use factory to ensure GO DAG is not pickled
    _go_dag: GODag | None = field(init=False, factory=lambda: None, repr=False)
    _obo_path: str | None = field(init=False, default=None, repr=False)
    # This instance's private copy of data.db (made on its first write) and the
    # (pid, instance token) that made it; a pickled or forked copy writes to a copy of its own.
    _private_db_path: str | None = field(init=False, default=None, repr=False)
    _private_db_owner: tuple[int, str] | None = field(
        init=False, default=None, repr=False
    )
    # A random token per instance: ownership never depends on id(), which CPython
    # reuses after collection.
    _instance_token: str = field(
        init=False, factory=lambda: secrets.token_hex(8), repr=False
    )
    # Every write this instance made, in order, so a copy can be rebuilt from the
    # shared file: ("delete", ids) or ("remove_deprecated_go_terms", ()). Entries are
    # immutable tuples, so a copy's new list never shares anything mutable.
    _db_writes: list[tuple[str, tuple[str, ...]]] = field(
        init=False, factory=list, repr=False
    )

    @classmethod
    @abstractmethod
    def gene_class(cls) -> type[GeneT]:
        """The :class:`AnnotatedGene` subclass :meth:`__getitem__` builds."""

    @classmethod
    @abstractmethod
    def release_files(cls) -> GenomeReleaseFiles:
        """The GFF and FASTA members of :attr:`ASSEMBLY_SET` this genome reads."""

    @classmethod
    @abstractmethod
    def fasta_chromosome(cls, record: Any) -> int:
        """The integer chromosome key of one record of the DNA FASTA."""

    @abstractmethod
    def _prepare_go_obo(self) -> str:
        """The path of the GO OBO file :attr:`go_dag` loads.

        Called once, at the end of construction. It may provision the file (the SGD
        subclass downloads it when absent); the DAG itself is loaded lazily.
        """

    @classmethod
    def _gene_id_of(cls, feature_id: str) -> str:
        """The gene id of a ``data.db`` feature ``ID``: the id without the gene class's
        ``FEATURE_ID_PREFIX``. An ``ID`` that lacks the prefix is refused by name.
        """
        prefix = cls.gene_class().FEATURE_ID_PREFIX
        if not feature_id.startswith(prefix):
            raise ValueError(
                f"{cls.__name__}: data.db feature id {feature_id!r} does not start with "
                f"the gene id prefix {prefix!r} of {cls.gene_class().__name__}"
            )
        return feature_id[len(prefix) :]

    def _read_sequences(self) -> None:
        """Parse the FASTA members into ``fasta_dna`` (keyed by FASTA record id),
        ``fasta_protein`` and ``fasta_cds`` (keyed by gene id).

        Called once during construction, after ``data.db`` is open and before the
        chromosome maps are built from ``fasta_dna``. This default reads three plain
        FASTA files and refuses a release without a CDS FASTA; a subclass whose release
        has none, or whose FASTAs are keyed by something other than the gene id,
        overrides it.
        """
        if self._cds_fasta_path is None:
            raise ValueError(
                f"{type(self).__name__}.release_files() names no cds_fasta; a release "
                "without a CDS FASTA derives its CDS sequences in an override of "
                "_read_sequences"
            )
        self.fasta_dna = SeqIO.to_dict(SeqIO.parse(self._dna_fasta_path, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        self.fasta_protein = SeqIO.to_dict(  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            SeqIO.parse(self._protein_fasta_path, "fasta")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        )
        self.fasta_cds = SeqIO.to_dict(SeqIO.parse(self._cds_fasta_path, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped

    def __attrs_post_init__(self) -> None:
        """Resolve the release files from the genomes tier and build the GFF database."""
        # Call parent class init to ensure all base attributes are set
        super().__init__(data_root=self.genome_root)
        self.genome_version = self.GENOME_VERSION

        # The release files come from the genomes tier, sha256-verified on every
        # resolve; genome_root stays the CACHE root (data.db, and for S288C, through
        # SCerevisiaeGraph.sgd_root, the genes/ and graph/ caches). There is no
        # download path: a machine without the tier fails here with the rsync that
        # seeds it, never with unpinned bytes.
        files = self.release_files()
        self._dna_fasta_path: str = resolve(self.ASSEMBLY_SET, files.dna_fasta)
        gff_filename = files.gff
        self._gff_path: str = resolve(self.ASSEMBLY_SET, gff_filename)
        self._protein_fasta_path = resolve(self.ASSEMBLY_SET, files.protein_fasta)
        if files.cds_fasta is not None:
            self._cds_fasta_path = resolve(self.ASSEMBLY_SET, files.cds_fasta)

        db_path = osp.join(self.genome_root, GENOME_DB_FILENAME)
        source = genome_database_source(self.ASSEMBLY_SET, gff_filename, self._gff_path)
        rebuild_call = self._constructor_call("overwrite=True")
        if osp.isdir(db_path):
            raise GenomeDatabaseUnavailableError(
                f"{db_path} is a directory, not a database file; every file is left "
                "alone. Remove or rename it, then retry."
            )
        if osp.lexists(db_path) and not osp.isfile(db_path):
            raise GenomeDatabaseUnavailableError(
                f"{db_path} is not a regular file (a FIFO, socket or device, or a "
                "symlink that does not lead to a regular file); every file is left "
                "alone. Remove or rename it, then retry."
            )
        if not osp.isdir(self.genome_root):
            if not self.overwrite:
                raise GenomeRootNotFoundError(
                    f"genome_root {self.genome_root!r} does not exist. Pass the "
                    "existing genome cache directory, or build a new one "
                    f"deliberately: {rebuild_call}"
                )
            os.makedirs(self.genome_root)
        writable = os.access(self.genome_root, os.W_OK)
        if writable:
            _sweep_dead(self.genome_root, _BUILD_TEMP)
        if self.overwrite or not osp.exists(db_path):
            why = "overwrite=True" if self.overwrite else "it does not exist"
            refuse_newer_record(db_path)
            self._require_writable(writable, db_path, why)
            rebuild_genome_database(self._gff_path, db_path, source)
        else:
            reason = untrusted_reason(db_path, source, rebuild_call)
            if reason is not None:
                self._require_writable(writable, db_path, reason)
                migrate_genome_database(
                    self._gff_path, db_path, source, rebuild_call, reason
                )

        # Set up connection manager for thread/process-safe database access
        self._db_connection_manager = GffutilsConnectionManager(db_path)

        self._read_sequences()
        # Create mapping from chromosome number to sequence identifier
        self.chr_to_nc = {
            self.fasta_chromosome(self.fasta_dna[key]): key
            for key in self.fasta_dna.keys()
        }
        self.nc_to_chr = {v: k for k, v in self.chr_to_nc.items()}
        self.chr_to_len = {
            self.nc_to_chr[chr]: len(self.fasta_dna[chr].seq)
            for chr in self.fasta_dna.keys()
        }

        # The GO ontology file is the subclass's; the DAG is loaded lazily (go_dag).
        self._obo_path = self._prepare_go_obo()
        # Call the method to remove deprecated GO terms
        # BUG this line doesn't work with ddp, I think the issue is merge=replace
        # self.remove_deprecated_go_terms()

    def _constructor_call(self, *extra: str) -> str:
        """``Class(field=value, ...)`` over this genome's init fields except
        ``overwrite``, followed by ``extra``: the construction a refusal names.
        """
        args = [
            f"{a.alias}={getattr(self, a.name)!r}"
            for a in attrs.fields(type(self))
            if a.init and a.name != "overwrite"
        ]
        return f"{type(self).__name__}({', '.join([*args, *extra])})"

    def _require_writable(self, writable: bool, db_path: str, why: str) -> None:
        """Refuse by name a build or migration this process cannot write."""
        if not writable:
            raise GenomeRootNotWritableError(
                f"{db_path} must be built ({why}), but {self.genome_root} is not "
                "writable by this process, so the database is not opened or built "
                f"here. Construct {self._constructor_call()} once from a process "
                "that can write that directory."
            )

    @classmethod
    def database_untrusted_reason(cls, genome_root: str) -> str | None:
        """Why ``<genome_root>/data.db`` would be built or migrated by a construction, or
        None when a construction would open it as is. Reads only; data-gated tests call
        it so that a test never performs the first migration of a real root.
        """
        gff_filename = cls.release_files().gff
        gff_path = resolve(cls.ASSEMBLY_SET, gff_filename)
        db_path = osp.join(genome_root, GENOME_DB_FILENAME)
        if not osp.exists(db_path):
            return "it does not exist"
        source = genome_database_source(cls.ASSEMBLY_SET, gff_filename, gff_path)
        return untrusted_reason(db_path, source, f"{cls.__name__}(..., overwrite=True)")

    @property
    def db(self) -> Any:
        """Return the gffutils connection (non-None in this genome).

        An instance unpickled from a genome that had written reads the writer's
        private copy; when that file is gone (the writer was collected), the first
        read rebuilds a private copy of its own from the shared file and replays the
        logged writes.
        """
        if (
            self._private_db_path is not None
            and self._private_db_owner != (os.getpid(), self._instance_token)
            and not osp.exists(self._private_db_path)
        ):
            self._writable_db()
        try:
            return super().db
        except TypeError as exc:
            # gffutils' FeatureDB raises a bare TypeError when the meta table is
            # empty: a process running pre-2026.10.01 code is rebuilding the file in
            # place under this one.
            assert self._db_connection_manager is not None
            path = self._db_connection_manager.db_path
            if _meta_rows(path) != 0:
                raise
            raise GenomeDatabaseUnavailableError(
                f"{path} has no gffutils metadata: another process (pre-2026.10.01 "
                "code) is rebuilding it in place; nothing was changed. Retry when it "
                "has finished; the next construction migrates its result."
            ) from exc

    @property
    def go_dag(self) -> GODag:
        """Lazy-load GO DAG - one per process"""
        if self._go_dag is None:
            if self._obo_path and osp.exists(self._obo_path):
                self._go_dag = GODag(self._obo_path)
            else:
                raise FileNotFoundError(f"GO OBO file not found at {self._obo_path}")
        return self._go_dag

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        """Custom pickling that handles non-pickleable objects."""
        # Get the attrs-generated __getstate__ if it exists
        state: dict[str, Any] = cast(
            "dict[str, Any]",
            (
                self.__getstate__()
                if hasattr(self, "__getstate__")
                else self.__dict__.copy()
            ),
        )

        # Clear non-pickleable objects
        if "_go_dag" in state:
            state["_go_dag"] = None
        # An unpickled instance never owns the pickled private path: it reads it,
        # and copies on its own first write (or first read, once the path is gone).
        state["_private_db_owner"] = None
        state["_instance_token"] = secrets.token_hex(8)
        # Mutable state a copy must not share (copy.copy passes ``state`` by
        # reference): the write log, and the gene-set cache the drops edit in place.
        state["_db_writes"] = list(self._db_writes)
        if self._gene_set is not None:
            state["_gene_set"] = GeneSet(self._gene_set)

        # Reconstruct with overwrite=False on the same database file (the private
        # copy when this instance has written): unpickling in a worker must never
        # rebuild the shared file. The original ``overwrite`` comes back in ``state``.
        init_kwargs = {
            a.alias: getattr(self, a.name) for a in attrs.fields(type(self)) if a.init
        }
        init_kwargs["overwrite"] = False
        return (
            _restore_annotated_genome,
            (self.__class__, init_kwargs, self._private_db_path),
            state,
            None,
            iter([]),
        )

    def _writable_db(self) -> Any:
        """This instance's private copy of the database, made on its first write.

        The copy is a sqlite backup of the connection this instance reads (the file
        it opened, even when pre-2026.10.01 code has since unlinked or is rewriting
        ``data.db``), so every later read and write of this instance sees exactly the
        database it had, and this method never writes the shared ``data.db``. A copy
        that cannot be made raises the named error (or the error propagates) and is
        removed; no failure leaves it in the temp dir. A pickled or forked copy of the
        instance (another pid, or an unpickled instance) makes a copy of its own; when
        the file it reads is gone, it copies the shared file and replays
        :attr:`_db_writes`.
        """
        owner = (os.getpid(), self._instance_token)
        if self._private_db_owner != owner:
            assert self._db_connection_manager is not None
            source_path = self._db_connection_manager.db_path
            replay = self._private_db_path is not None and not osp.exists(source_path)
            if replay:
                source_path = osp.join(self.genome_root, GENOME_DB_FILENAME)
            temp_dir = tempfile.gettempdir()
            _sweep_dead(temp_dir, _PRIVATE_COPY)
            fd, path = tempfile.mkstemp(
                prefix=f"torchcell-genome-{socket.gethostname()}-{os.getpid()}-",
                suffix=".db",
                dir=temp_dir,
            )
            os.close(fd)
            try:
                with closing(sqlite3.connect(path)) as dst:
                    try:
                        if replay:
                            with closing(
                                sqlite3.connect(_ro_uri(source_path), uri=True)
                            ) as src:
                                src.backup(dst)
                        else:
                            # Back up the connection this instance reads, not the
                            # path: a pre-2026.10.01 rebuild that unlinked or is
                            # rewriting the path does not change the file this
                            # connection holds.
                            source = super().db
                            assert source is not None
                            source.conn.backup(dst)
                    except sqlite3.DatabaseError as exc:
                        require_damage(source_path, exc)
                        raise GenomeDatabaseUnavailableError(
                            f"{source_path} cannot be copied for this instance's "
                            f"writes ({exc}); every file is left alone. Retry when "
                            "the process rebuilding it has finished."
                        ) from exc
                if _meta_rows_or_zero(path) == 0:
                    raise GenomeDatabaseUnavailableError(
                        f"{source_path} has no gffutils metadata: another process "
                        "(pre-2026.10.01 code) is rebuilding it in place; nothing "
                        "was changed. Retry when it has finished."
                    )
            except BaseException:
                # The error is named above or propagates as is; the copy never
                # outlives a failed first write.
                os.remove(path)
                _remove_if_present(path + "-journal")
                raise
            self._db_connection_manager = GffutilsConnectionManager(path)
            self._private_db_path = path
            self._private_db_owner = owner
            weakref.finalize(self, _remove_private_copy, path, os.getpid())
            if replay:
                for op, ids in self._db_writes:
                    self._apply_write(op, ids)
        return super().db

    def _apply_write(self, op: str, ids: tuple[str, ...]) -> None:
        """Apply one logged write to this instance's private copy (no backup file)."""
        db = super().db
        assert db is not None
        if op == "delete":
            for feature_id in ids:
                db.delete(feature_id, make_backup=False)
        else:
            db.update(
                self._deprecated_go_updates(db),
                merge_strategy="replace",
                make_backup=False,
            )
        db.conn.commit()

    def _write(self, op: str, ids: list[str]) -> None:
        """Make or reuse the private copy, apply ``op`` to it, and log it."""
        self._writable_db()
        entry = (op, tuple(ids))
        self._apply_write(*entry)
        self._db_writes.append(entry)

    def remove_deprecated_go_terms(self) -> None:
        """Drop GO terms absent from or obsolete in the GO DAG in this instance's copy."""
        self._write("remove_deprecated_go_terms", [])

    def _deprecated_go_updates(self, db: Any) -> list[Feature]:
        """The gene features of ``db`` with absent or obsolete GO terms removed."""
        # Create a list to hold updated features
        updated_features = []

        # Iterate over each feature in the database
        invalid_go_terms: dict[str, list[str]] = {"not_in_go_dag": [], "obsolete": []}
        go_attribute = self.gene_class().GO_ATTRIBUTE
        for feature in db.features_of_type("gene"):
            # Check if the feature has the GO attribute
            if go_attribute in feature.attributes:
                # Filter out deprecated GO terms
                valid_onto_terms = []
                valid_go_terms = []
                for term in feature.attributes[go_attribute]:
                    if term.startswith("GO:"):
                        if term not in self.go_dag:
                            invalid_go_terms["not_in_go_dag"].append(term)
                        elif self.go_dag[term].is_obsolete:
                            invalid_go_terms["obsolete"].append(term)
                        else:
                            valid_go_terms.append(term)
                    else:
                        valid_onto_terms.append(term)
                # Update the GO attribute for the feature
                if valid_go_terms:
                    feature.attributes[go_attribute] = valid_go_terms + valid_onto_terms
                else:
                    del feature.attributes[go_attribute]

                # Add the updated feature to the list
                updated_features.append(feature)

        return updated_features

    @property
    def alias_to_systematic(self) -> dict[str, list[str]]:
        """Return a cached mapping from each alias to its list of systematic IDs."""
        if self._alias_to_systematic is None:
            alias_map: dict[str, list[str]] = {}
            for gene_id in self.gene_set:
                gene = self[gene_id]
                if gene and gene.alias:
                    for alias in gene.alias:
                        if alias not in alias_map:
                            alias_map[alias] = []
                        alias_map[alias].append(gene_id)
            self._alias_to_systematic = alias_map
        return self._alias_to_systematic

    @property
    def feature_index(self) -> dict[str, Any]:
        """Cached locus-feature index (upper-cased) backing :meth:`resolve_gene_name`.

        Indexes only GENE-LIKE LOCUS features (:attr:`LOCUS_FEATURE_TYPES`; for SGD,
        ``gene`` plus the RNA-gene / pseudogene / transposon-gene /
        ``blocked_reading_frame`` types) so that
        non-locus features (``region``, ``CDS``, ``mRNA``, ``ARS`` ...) never shadow a real
        gene name -- e.g. a ``region`` feature literally id'd ``ADE1`` must NOT intercept the
        gene ADE1. Unlike :attr:`alias_to_systematic` (``"gene"`` features only), this also
        covers non-``"gene"`` loci (e.g. ``blocked_reading_frame`` pseudogenes like
        ``YER109C``/FLO8). Keys: ``genes`` (upper ids of live ``"gene"`` features),
        ``locus_type`` (upper id -> featuretype for non-``"gene"`` loci), ``standard_to_ids``
        (upper standard/common name from the GFF ``gene`` attribute -> locus ids) and
        ``alias_to_ids`` (upper ``Alias`` value -> locus ids).
        """
        if self._feature_index is None:
            genes = {g.upper() for g in self.gene_set}
            locus_type: dict[str, str] = {}
            standard_to_ids: dict[str, list[str]] = {}
            alias_to_ids: dict[str, list[str]] = {}
            for feat in self.db.all_features():
                if feat.featuretype not in self.LOCUS_FEATURE_TYPES:
                    continue
                fid = self._gene_id_of(feat.id)
                if feat.featuretype != "gene":
                    locus_type[fid.upper()] = feat.featuretype
                for std in feat.attributes.get("gene", []) or []:
                    standard_to_ids.setdefault(std.strip().upper(), []).append(fid)
                for alias in feat.attributes.get("Alias", []) or []:
                    alias_to_ids.setdefault(alias.strip().upper(), []).append(fid)
            self._feature_index = {
                "genes": genes,
                "locus_type": locus_type,
                "standard_to_ids": standard_to_ids,
                "alias_to_ids": alias_to_ids,
            }
        return self._feature_index

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """Reconcile a source gene name to the current annotation (layered).

        Layers, in order: (1) exact live gene id; (2) a valid non-``"gene"`` LOCUS id
        (e.g. a ``blocked_reading_frame`` pseudogene); (3) the standard/common name of a
        locus (from the GFF ``gene`` attribute); (4) a secondary ``Alias`` of a locus; (5)
        not found -> retired, retained with the original name. The standard-name layer runs
        BEFORE the alias layer so a common name resolves to the gene that OWNS it (in S288C,
        ``AAP1`` -> ``YHR047C``, whose standard name is AAP1) rather than to a gene that
        merely lists it as a secondary alias (``Q0080``). Within layers 3-4 a unique gene
        wins; multiple genes are ``AMBIGUOUS``; a unique non-gene locus is
        ``NON_GENE_FEATURE``. The resolver is pure/per-name and applies NO drop policy --
        callers decide retention and any batch-level collision handling.
        """
        raw = name
        n = name.strip().upper()
        idx = self.feature_index
        genes: set[str] = idx["genes"]
        locus_type: dict[str, str] = idx["locus_type"]

        # 1. Exact live gene.
        if n in genes:
            return GeneNameResolution(
                input_name=raw, status=GeneNameStatus.CURRENT, systematic_name=n
            )
        # 2. A valid non-"gene" locus by its own id (e.g. blocked_reading_frame pseudogene).
        if n in locus_type:
            return GeneNameResolution(
                input_name=raw,
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name=n,
                feature_type=locus_type[n],
                note=f"valid {self.ANNOTATION_NAME} {locus_type[n]}, not a gene feature",
            )
        # 3. Standard/common name; then 4. secondary alias -- same resolution semantics.
        for mapping_key, layer in (
            ("standard_to_ids", "standard name"),
            ("alias_to_ids", "alias"),
        ):
            ids = sorted(set(idx[mapping_key].get(n, [])))
            if not ids:
                continue
            gene_ids = [i for i in ids if i.upper() in genes]
            if len(gene_ids) == 1:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.RENAMED,
                    systematic_name=gene_ids[0],
                    note=f"{layer} of current gene {gene_ids[0]}",
                )
            if len(gene_ids) > 1:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.AMBIGUOUS,
                    systematic_name=None,
                    candidates=gene_ids,
                    note=f"{layer} of multiple current genes",
                )
            if len(ids) == 1:
                t = ids[0]
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.NON_GENE_FEATURE,
                    systematic_name=t,
                    feature_type=locus_type.get(t.upper()),
                    note=f"{layer} of {locus_type.get(t.upper())} {t} (not a gene feature)",
                )
            return GeneNameResolution(
                input_name=raw,
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=ids,
                note=f"{layer} of multiple non-gene loci",
            )
        # 5. Not found anywhere in the annotation: a retired name (in S288C, a dubious
        # 2005-era ORF).
        return GeneNameResolution(
            input_name=raw,
            status=GeneNameStatus.RETIRED,
            systematic_name=n,
            note=f"not found in {self.ANNOTATION_RELEASE}; retained as a legacy "
            "systematic name",
        )

    @property
    def go(self) -> SortedSet[str]:
        """Return the cached set of all GO terms across genes in the gene set."""
        if self._go is None:
            all_go = SortedSet()

            # Iterate through all genes in self.gene_set
            for gene_id in self.gene_set:
                gene = self[gene_id]  # Retrieve the gene object

                # Use the go attribute of the gene object if it exists and is not None
                if gene and hasattr(gene, "go") and gene.go is not None:
                    all_go.update(gene.go)
            self._go = all_go
            return self._go
        else:
            return self._go

    def go_subset(self, gene_set: SortedSet[str]) -> SortedSet[str]:
        """Return the set of GO terms covered by the given subset of genes."""
        go_subset = SortedSet()

        # Iterate through the provided subset of genes
        for gene_id in gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                go_subset.update(gene.go)

        return go_subset

    @property
    def go_genes(self) -> SortedDict[str, SortedSet[str]]:
        """Return a cached mapping from each GO term to the genes annotated with it."""
        # CHECK could this contain obselete terms? We don't check if the terms are in self.go...
        if self._go_genes is None:
            go_genes_dict = SortedDict()

            # Iterate through all genes in self.gene_set
            for gene_id in self.gene_set:
                gene = self[gene_id]  # Retrieve the gene object

                # Use the go attribute of the gene object if it exists and is not None
                if gene and hasattr(gene, "go") and gene.go is not None:
                    for go_term in gene.go:
                        if go_term not in go_genes_dict:
                            go_genes_dict[go_term] = SortedSet()
                        go_genes_dict[go_term].add(gene_id)
            self._go_genes = go_genes_dict
            return go_genes_dict
        else:
            return self._go_genes

    def go_subset_genes(
        self, gene_set: SortedSet[str]
    ) -> SortedDict[str, SortedSet[str]]:
        """Return a GO-term-to-genes mapping restricted to the given gene subset."""
        go_subset_genes_dict = SortedDict()

        # Iterate through the provided subset of genes
        for gene_id in gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                for go_term in gene.go:
                    if go_term not in go_subset_genes_dict:
                        go_subset_genes_dict[go_term] = SortedSet()
                    go_subset_genes_dict[go_term].add(gene_id)

        return go_subset_genes_dict

    def get_seq(
        self, chr: int | str, start: int, end: int, strand: str
    ) -> DnaSelectionResult:
        """Return the DNA sequence for the given chromosome region and strand.

        ``chr`` is the integer chromosome key of :attr:`chr_to_nc` (for S288C, 0 is the
        mitochondrion), the number that ``DnaSelectionResult.chromosome`` stores; a
        FASTA key or any other string is refused by name, as is a strand other than
        ``+``/``-``.
        """
        if not isinstance(chr, int) or chr not in self.chr_to_nc:
            raise ValueError(
                f"Chromosome must be one of the chromosome numbers "
                f"{sorted(self.chr_to_nc)}, got {chr!r}"
            )
        if strand not in ("+", "-"):
            raise ValueError(f"Strand must be '+' or '-', got {strand!r}")
        fasta_key = self.chr_to_nc[chr]
        if strand == "+":
            seq = self.fasta_dna[fasta_key].seq[start:end]
        else:
            seq = self.fasta_dna[fasta_key].seq[start:end].reverse_complement()
        return DnaSelectionResult(
            id=self.id,  # type: ignore[attr-defined]  # no 'id' on genome (pre-existing)
            chromosome=chr,
            strand=strand,
            start=start,
            end=end,
            seq=str(seq),
        )

    @property
    def gene_attribute_table(self) -> pd.DataFrame:
        """Return a DataFrame of single-valued GFF attributes for every gene."""
        data = []
        for gene_feature in self.db.features_of_type("gene"):
            gene_data = {}
            for attr_name in gene_feature.attributes.keys():
                # We only add attributes with length 1 or less
                if len(gene_feature.attributes[attr_name]) <= 1:
                    # If the attribute is a list with one value, we unpack it
                    gene_data[attr_name] = (
                        gene_feature.attributes[attr_name][0]
                        if len(gene_feature.attributes[attr_name]) == 1
                        else None
                    )
            data.append(gene_data)
        return pd.DataFrame(data)

    @property
    def feature_types(self) -> list[str]:
        """Return the list of feature types present in the GFF database."""
        return list(self.db.featuretypes())

    def compute_gene_set(self) -> GeneSet:
        """Return the GeneSet of all gene IDs in the database."""
        genes = [
            self._gene_id_of(feat.id) for feat in list(self.db.features_of_type("gene"))
        ]
        assert len(genes) == len(set(genes)), (
            "Duplicate genes found... chekc handled by gff."
        )
        return GeneSet(genes)

    def drop_empty_go(self) -> None:
        """Remove genes without GO terms from this instance's database copy and gene set."""
        # Initialize a list to hold genes to be removed
        genes_to_remove = []

        # Iterate through all genes in the current gene_set
        for gene_id in self.gene_set:
            gene = self[gene_id]
            if gene is not None:
                # Check if the GO terms are empty
                # None case for never annotated, 0 for no GO terms
                if gene.go is None or len(gene.go) == 0:
                    genes_to_remove.append(gene_id)

        # Remove these genes from the gene set cache
        for gene_id in genes_to_remove:
            self._gene_set.discard(gene_id)

        # Remove these genes from this instance's private copy of the database (by
        # their data.db feature ids)
        prefix = self.gene_class().FEATURE_ID_PREFIX
        self._write("delete", [prefix + gene_id for gene_id in genes_to_remove])

        # As in SCerevisiaeGenome.drop_chrmt: the locus index and the GO-to-genes map
        # were built from the pre-drop gene set; reset them so the next access
        # rebuilds without the dropped genes.
        self._feature_index = None
        self._go_genes = None

    def __getitem__(self, item: str) -> GeneT | None:
        """Return the gene for a systematic ID, or None if absent."""
        # For now we only support the systematic names
        # ising region instead, since it give more options on dealing with gene processing in gene class
        try:
            gene = self.gene_class()(
                id=item,
                db=self.db,
                fasta_dna=self.fasta_dna,
                fasta_protein=self.fasta_protein,
                fasta_cds=self.fasta_cds,
                chr_to_nc=self.chr_to_nc,
                chromosome_lengths=self.chr_to_len,
            )
            return gene
        except KeyError:
            print(
                f"Gene {item} not found in genome, only systematic names (ID) are supported."
            )
            return None
