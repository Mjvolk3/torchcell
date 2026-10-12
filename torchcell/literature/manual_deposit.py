# torchcell/literature/manual_deposit.py
# [[torchcell.literature.manual_deposit]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/manual_deposit.py
# Test file: tests/torchcell/literature/test_manual_deposit.py

r"""Record a by-hand (``manual_browser``) deposit into a raw-mirror key, one way for all.

Some publishers answer every scripted request with a browser challenge, so the only
retrieval left is the owner downloading the files in a browser and copying them into
``$DATA_ROOT/torchcell-raw/<citation_key>/``. The bytes are then canonical, and the
record of how they arrived is the whole provenance chain, so it is written the same way
for every such deposit:

- :class:`ManualDeposit` is the typed record: the page the owner opened, the recipe a
  rebuild re-runs by hand, who did it, and one :class:`DepositedFile` per file with the
  sha256 and byte count of the bytes as deposited and the time the download arrived.
  A file unpacked from an archive the owner downloaded names that archive as its
  ``container``; :func:`verify_deposit` re-reads the member out of the archive and
  requires the two byte strings to be equal.
- :func:`write_deposit_records` writes ``DEPOSIT.md`` and ``SHA256SUMS.txt`` beside the
  files, one pair per directory the deposit touches. ``SHA256SUMS.txt`` names files by
  basename, so ``sha256sum -c SHA256SUMS.txt`` run in that directory re-checks them. An
  existing record with other content is refused, never overwritten.
- :func:`artifact_records` turns the deposit into manifest records with a
  ``RetrievalMethod.manual_browser`` retrieval whose ``params`` carry the recipe and
  point at ``DEPOSIT.md``; :func:`extend_manifest` adds records to a key's
  ``manifest.json`` without dropping or rewriting a record already there, and refuses a
  record whose sha256 disagrees with the one held.

Nothing here fetches: the owner already did, and a missing or altered file raises with
the recipe, which is the only way to produce it again.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)

DEPOSIT_RECORD_NAME = "DEPOSIT.md"
CHECKSUMS_NAME = "SHA256SUMS.txt"
#: The ``retriever`` of a by-hand retrieval: there is no function to re-run.
MANUAL_RETRIEVER = "manual"


class DepositMismatchError(RuntimeError):
    """A deposited file is missing, or its bytes are not the recorded ones."""


class ScriptedEquivalent(BaseModel):
    """A scriptable route measured to return the same bytes as a by-hand deposit.

    The deposit stays ``manual_browser`` (that is how the bytes arrived); this records
    that a rebuild does not need a person, because ``retriever(**params)`` returned
    bytes with the deposited sha256 when it was checked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    retriever: str = Field(description="dotted path in torchcell.literature.retrieve")
    params: dict[str, str]
    url: str
    checked_at: str = Field(description="ISO date the bytes were fetched and compared")


class DepositedFile(BaseModel):
    """One file of a manual deposit, as it sits in the key directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(description="path relative to the key directory")
    role: str = Field(description="manifest role (si_pdf, si_data, raw_data, ...)")
    sha256: str
    bytes: int
    original_filename: str = Field(
        description="the name the file arrived under in the owner's download"
    )
    arrived_at: str = Field(
        description="ISO timestamp the owner's download arrived (the downloaded file's "
        "mtime); a member of an archive arrived with its archive"
    )
    container: str | None = Field(
        default=None,
        description="key-relative path of the deposited archive this file was unpacked "
        "from, or None for a file that arrived on its own",
    )
    description: str = Field(description="what the file holds, measured or stated")
    source_url: str | None = Field(
        default=None,
        description="the page this file came from when it is not the deposit's own "
        "source_url (one owner download session can span two pages)",
    )
    scripted_equivalent: ScriptedEquivalent | None = None

    @model_validator(mode="after")
    def _check(self) -> DepositedFile:
        if len(self.sha256) != 64 or set(self.sha256) - set("0123456789abcdef"):
            raise ValueError(f"{self.path}: sha256 must be 64 lowercase hex digits")
        if self.bytes <= 0:
            raise ValueError(f"{self.path}: a deposited file has at least one byte")
        if (
            PurePosixPath(self.path).is_absolute()
            or ".." in PurePosixPath(self.path).parts
        ):
            raise ValueError(f"{self.path}: must be a path inside the key directory")
        return self

    @property
    def directory(self) -> str:
        """The key-relative directory holding the file (``.`` for the key root)."""
        return str(PurePosixPath(self.path).parent)

    @property
    def name(self) -> str:
        """The file's name inside its directory."""
        return PurePosixPath(self.path).name


class ManualDeposit(BaseModel):
    """One owner download, copied into one key of the raw mirror."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str
    doi: str
    title: str
    source_url: str = Field(description="the page the owner opened")
    retrieval_command: str = Field(
        description="the manual recipe a rebuild re-runs by hand before re-hashing"
    )
    retrieved_by: str
    deposited_at: str = Field(description="ISO date the files reached the mirror")
    issues: tuple[int, ...] = Field(description="GitHub issues the deposit answers")
    note: str = Field(description="what the deposit is, and anything odd about it")
    files: tuple[DepositedFile, ...]

    @model_validator(mode="after")
    def _check(self) -> ManualDeposit:
        if not self.files:
            raise ValueError(f"{self.citation_key}: a deposit holds at least one file")
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError(f"{self.citation_key}: a deposited path is repeated")
        names = {f.name for f in self.files if f.name in self.record_names}
        if names:
            raise ValueError(f"{self.citation_key}: {sorted(names)} are record names")
        for f in self.files:
            if f.container is not None and f.container not in paths:
                raise ValueError(
                    f"{f.path}: its container {f.container} is not a deposited file"
                )
        return self

    @property
    def record_names(self) -> tuple[str, str]:
        """The two record files written beside the deposited ones."""
        return (DEPOSIT_RECORD_NAME, CHECKSUMS_NAME)

    def directories(self) -> tuple[str, ...]:
        """Every key-relative directory holding a deposited file, sorted."""
        return tuple(sorted({f.directory for f in self.files}))

    def source_of(self, f: DepositedFile) -> str:
        """The page ``f`` came from: its own ``source_url``, else the deposit's."""
        return f.source_url if f.source_url is not None else self.source_url

    def files_in(self, directory: str) -> tuple[DepositedFile, ...]:
        """The deposited files of one directory, sorted by name."""
        return tuple(
            sorted(
                (f for f in self.files if f.directory == directory),
                key=lambda f: f.name,
            )
        )


def verify_deposit(deposit: ManualDeposit, key_root: str | Path) -> None:
    """Every deposited file is present with its recorded bytes and sha256.

    A file unpacked from a deposited archive must also equal the member of that name
    read out of the archive. Raises :class:`DepositMismatchError` naming the file and
    carrying the recipe, since re-running it by hand is the only repair.
    """
    root = Path(key_root)
    for f in deposit.files:
        path = root / f.path
        if not path.is_file():
            raise DepositMismatchError(
                f"{path} is missing; it is a manual deposit. MANUAL RECIPE: "
                f"{deposit.retrieval_command}"
            )
        size = path.stat().st_size
        got = sha256_file(path)
        if size != f.bytes or got != f.sha256:
            raise DepositMismatchError(
                f"{path}: {size} B sha256 {got}, recorded {f.bytes} B {f.sha256}"
            )
        if f.container is not None:
            with zipfile.ZipFile(root / f.container) as archive:
                member = archive.read(f.original_filename)
            if hashlib.sha256(member).hexdigest() != f.sha256:
                raise DepositMismatchError(
                    f"{path} is not the member {f.original_filename} of {f.container}"
                )


def checksums_text(deposit: ManualDeposit, directory: str) -> str:
    """``SHA256SUMS.txt`` for one directory: ``<sha256>  <name>`` lines, sorted."""
    return "".join(f"{f.sha256}  {f.name}\n" for f in deposit.files_in(directory))


def deposit_markdown(deposit: ManualDeposit, directory: str) -> str:
    """``DEPOSIT.md`` for one directory: the record a person reads beside the bytes."""
    issues = ", ".join(f"#{n}" for n in deposit.issues)
    lines = [
        "# Manual deposit record (RetrievalMethod.manual_browser)",
        "",
        f"- citation_key: {deposit.citation_key}",
        f"- doi: {deposit.doi}",
        f"- title: {deposit.title}",
        f"- source_url: {deposit.source_url}",
        "- retrieval_method: manual_browser",
        f"- retrieval_command (manual recipe): {deposit.retrieval_command}",
        f"- retrieved_by: {deposit.retrieved_by}",
        f"- deposited_at: {deposit.deposited_at}",
        f"- issues: {issues}",
        f"- note: {deposit.note}",
        f"- checksums: {CHECKSUMS_NAME} (this directory's files only; run "
        f"`sha256sum -c {CHECKSUMS_NAME}` here)",
        "- record written by: torchcell.literature.manual_deposit.write_deposit_records",
        "",
        "| file | arrived as | arrived at | bytes | sha256 | source | holds |",
        "|---|---|---|---|---|---|---|",
    ]
    equivalents: list[str] = []
    for f in deposit.files_in(directory):
        arrived_as = (
            f.original_filename
            if f.container is None
            else f"{f.original_filename} (member of {f.container})"
        )
        lines.append(
            f"| {f.name} | {arrived_as} | {f.arrived_at} | {f.bytes} | {f.sha256} | "
            f"{deposit.source_of(f)} | {f.description} |"
        )
        if f.scripted_equivalent is not None:
            eq = f.scripted_equivalent
            equivalents.append(
                f"- {f.name}: `{eq.retriever}` {eq.params} ({eq.url}) returned these "
                f"exact bytes on {eq.checked_at}"
            )
    if equivalents:
        lines += [
            "",
            "## Scripted equivalents (measured, not how the bytes arrived)",
            "",
        ]
        lines += equivalents
    return "\n".join(lines) + "\n"


def _write_once(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` unless an equal file is there; refuse a different one."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise DepositMismatchError(
                f"{path} exists with other content; move it aside before re-recording"
            )
        return path
    path.write_text(text, encoding="utf-8")
    return path


def write_deposit_records(deposit: ManualDeposit, key_root: str | Path) -> list[Path]:
    """Verify the deposit, then write ``DEPOSIT.md`` and ``SHA256SUMS.txt`` per directory.

    Idempotent: an identical record already on disk is left alone. A record with other
    content raises :class:`DepositMismatchError` rather than being replaced.
    """
    root = Path(key_root)
    verify_deposit(deposit, root)
    written: list[Path] = []
    for directory in deposit.directories():
        folder = root / directory
        written.append(
            _write_once(
                folder / DEPOSIT_RECORD_NAME, deposit_markdown(deposit, directory)
            )
        )
        written.append(
            _write_once(folder / CHECKSUMS_NAME, checksums_text(deposit, directory))
        )
    return written


def beside(f: DepositedFile, name: str) -> str:
    """Key-relative path of the record file ``name`` in ``f``'s directory."""
    return str(PurePosixPath(f.directory) / name)


def artifact_records(deposit: ManualDeposit) -> list[ArtifactRecord]:
    """One manifest record per deposited file, retrieval ``manual_browser``.

    ``params`` carries the recipe (the ``retrieval_command`` CLAUDE.md requires), who
    retrieved the bytes, and where ``DEPOSIT.md`` and ``SHA256SUMS.txt`` sit; a member of
    an archive also names the archive and its sha256.
    """
    by_path = {f.path: f for f in deposit.files}
    records: list[ArtifactRecord] = []
    for f in deposit.files:
        params: dict[str, object] = {
            "retrieval_command": deposit.retrieval_command,
            "retrieved_by": deposit.retrieved_by,
            "deposited_at": deposit.deposited_at,
            "issues": list(deposit.issues),
            "deposit_record": beside(f, DEPOSIT_RECORD_NAME),
            "checksums": beside(f, CHECKSUMS_NAME),
        }
        if f.container is not None:
            params["container"] = f.container
            params["container_sha256"] = by_path[f.container].sha256
            params["member"] = f.original_filename
        if f.scripted_equivalent is not None:
            params["scripted_equivalent"] = f.scripted_equivalent.model_dump()
        source = deposit.source_of(f)
        records.append(
            ArtifactRecord(
                path=f.path,
                role=f.role,
                bytes=f.bytes,
                sha256=f.sha256,
                source=source,
                original_filename=f.original_filename,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.manual_browser,
                    source_url=source,
                    retriever=MANUAL_RETRIEVER,
                    params=params,
                    sha256=f.sha256,
                    retrieved_at=f.arrived_at,
                ),
            )
        )
    return records


def extend_manifest(
    key_root: str | Path,
    records: Sequence[ArtifactRecord],
    *,
    citation_key: str,
    doi: str,
    title: str,
    si_expected: Sequence[str] | None = None,
    si_data_sources: Sequence[str] = (),
) -> Path:
    """Add ``records`` to the key's ``manifest.json``, keeping every record it holds.

    A record the manifest already holds at the same path and sha256 is kept as it is; a
    record whose sha256 disagrees raises :class:`DepositMismatchError`. A manifest that
    does not exist yet is created (the Mohiuddin 2022 precedent: a raw-mirror key may
    exist before its paper is in the literature mirror). ``si_expected``, when given,
    replaces the manifest's list, because it is the statement of what is still missing
    and a deposit changes it; ``si_data_sources`` is extended, never shortened.
    """
    root = Path(key_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_FILENAME
    manifest = (
        Manifest.model_validate_json(path.read_text())
        if path.exists()
        else Manifest(
            citation_key=citation_key,
            doi=doi,
            title=title,
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    if manifest.citation_key != citation_key:
        raise DepositMismatchError(
            f"{path} is the manifest of {manifest.citation_key}, not {citation_key}"
        )
    held = {record.path: record for record in manifest.files}
    for record in records:
        existing = held.get(record.path)
        if existing is None:
            manifest.files.append(record)
            held[record.path] = record
            continue
        if existing.sha256 != record.sha256:
            raise DepositMismatchError(
                f"{path} holds {record.path} at {existing.sha256}, the deposit records "
                f"{record.sha256}; refusing"
            )
    manifest.files.sort(key=lambda record: record.path)
    for url in [*(r.source for r in records), *si_data_sources]:
        if url is not None and url not in manifest.si_data_sources:
            manifest.si_data_sources.append(url)
    if si_expected is not None:
        manifest.si_expected = list(si_expected)
    path.write_text(manifest.model_dump_json(indent=2))
    return path
