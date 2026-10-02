# torchcell/literature/reocr_si.py
# [[torchcell.literature.reocr_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/reocr_si.py
# Test file: tests/torchcell/literature/test_reocr_si.py

r"""Re-OCR every SI PDF of a mirror key into per-PDF figure directories (issue #579).

Before PR #585 every ``si/si*.pdf`` of a key wrote its figures into one shared
``si/images/``, and each PDF's OCR deleted the previous one's, so a key with several
SI PDFs has SI markdown pointing at figures that are gone. :func:`reocr_keys` repairs
such keys in phases, each over every key before the next phase starts:

0. Precheck, before anything is OCR'd or retired: no key is shared by two or more
   top-level Zotero items (:class:`DuplicateCitationKeyError`; a duplicate of a key
   this run does not process is logged once at WARNING and does not block); every
   key is in the Zotero citation index (:class:`KeyNotInZoteroError`); a dry-run
   ``backfill_key`` for every key is ``enriched`` (:class:`NotEnrichedError`); no
   ``si/*.md`` that this run will not re-OCR references a flat ``si/images/<file>``
   (:class:`UnrewrittenFlatReferenceError`). Zotero is only read.
1. OCR every ``si/si*.pdf`` in natural order with
   :func:`torchcell.literature.ocr.ocr_pdf`. ``paper.pdf`` is NOT re-OCR'd.
2. Check that every SI figure reference resolves (:class:`UnresolvedFigureError`;
   nothing retired, no manifest written).
3. Retire the stale flat ``si/images/<file>`` figures: they are moved into
   :func:`staging_dir` and ``scripts/deprecate.sh`` (with the caller's full
   environment, so its refusal of a graveyard inside ``$DATA_ROOT`` applies) moves
   that directory to the graveyard. On failure the figures are moved back when the
   staging directory still exists, and :class:`RetireError` reports the exit code
   and where the files are. A staging directory left by a killed run is retired.
4. Check the references again (backstop; the precheck should have caught any
   markdown pointing at a flat figure). On refusal the retired figures are in the
   graveyard entry named in the log; the operator moves the named file back into
   ``si/images/`` and investigates that markdown.
5. Rewrite each ``manifest.json`` with ``backfill_key(force=True)``. If the result
   is not ``enriched`` the previous manifest bytes are restored and
   :class:`NotEnrichedError` is raised. An enriched manifest of an item with no DOI
   or title passes, carrying ``provenance_complete=False`` as ``backfill_key``
   writes it.

``scripts/lit_reocr_si.py`` is the command-line wrapper.
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from torchcell.literature._run_mineru import IMAGE_REF
from torchcell.literature.backfill import (
    DuplicateCitationKeyError,
    backfill_key,
    citation_index_with_duplicates,
    describe_duplicates,
    library_root,
)
from torchcell.literature.manifest import MANIFEST_FILENAME
from torchcell.literature.ocr import natural_key, ocr_pdf
from torchcell.literature.zotero import ZoteroLibrary

DEPRECATE_SH = Path(__file__).resolve().parents[2] / "scripts" / "deprecate.sh"
DEFAULT_GRAVEYARD = "/scratch/projects/torchcell-deprecated"
RETIRE_REASON = "issue #579: flat shared SI figures replaced by per-PDF re-OCR"

log = logging.getLogger(__name__)


class NoSiPdfError(FileNotFoundError):
    """A key handed to the re-OCR has no ``si/si*.pdf``."""


class KeyNotInZoteroError(LookupError):
    """A requested key has no item in the Zotero citation index."""


class UnrewrittenFlatReferenceError(RuntimeError):
    """An ``si/*.md`` with no PDF to re-OCR references a flat ``si/images/`` file."""


class UnresolvedFigureError(RuntimeError):
    """SI markdown references figures that are not on disk."""


class RetireError(RuntimeError):
    """``deprecate.sh`` failed while retiring the flat figures."""


class NotEnrichedError(RuntimeError):
    """A backfill of a key is not Zotero-enriched."""


def si_pdfs(key_dir: Path) -> list[Path]:
    """The key's ``si/si*.pdf`` in natural order (``si2`` before ``si10``)."""
    pdfs = sorted((key_dir / "si").glob("si*.pdf"), key=natural_key)
    if not pdfs:
        raise NoSiPdfError(f"no si/si*.pdf under {key_dir}")
    return pdfs


def flat_si_figures(key_dir: Path) -> list[Path]:
    """Files directly in ``si/images/``: the shared layout written before #579."""
    images = key_dir / "si" / "images"
    if not images.is_dir():
        return []
    return sorted(p for p in images.iterdir() if p.is_file())


def staging_dir(key_dir: Path) -> Path:
    """Where the flat figures wait for ``deprecate.sh``, named per key."""
    return key_dir / "si" / f"{key_dir.name}__si-images-flat-pre579"


def _si_refs(key_dir: Path) -> list[tuple[Path, str]]:
    """Every ``(markdown, images/<name>)`` figure reference in ``si/*.md``."""
    refs: list[tuple[Path, str]] = []
    for md in sorted((key_dir / "si").glob("*.md"), key=natural_key):
        for match in IMAGE_REF.finditer(md.read_text(encoding="utf-8")):
            refs.append((md, f"images/{match.group('name')}"))
    return refs


def unresolved_si_figure_refs(key_dir: Path) -> list[str]:
    """Every ``<md name>: images/<name>`` reference in ``si/*.md`` with no file."""
    si = key_dir / "si"
    return [
        f"{md.name}: {ref}" for md, ref in _si_refs(key_dir) if not (si / ref).is_file()
    ]


def unrewritten_flat_refs(key_dir: Path) -> list[str]:
    """References to a flat ``images/<file>`` from an ``si/*.md`` with no SI PDF.

    The re-OCR rewrites the markdown of every ``si/si*.pdf``; any other markdown
    keeps its references, and one to a flat figure would dangle once the flat
    figures are retired.
    """
    reocrd = {pdf.stem for pdf in (key_dir / "si").glob("si*.pdf")}
    return [
        f"{md.name}: {ref}"
        for md, ref in _si_refs(key_dir)
        if md.stem not in reocrd and "/" not in ref.removeprefix("images/")
    ]


def check_references(key_dirs: list[Path], stage: str) -> None:
    """Refuse with :class:`UnresolvedFigureError` if any SI reference dangles."""
    missing = [
        f"{key_dir.name}/si/{ref}"
        for key_dir in key_dirs
        for ref in unresolved_si_figure_refs(key_dir)
    ]
    if missing:
        raise UnresolvedFigureError(
            f"{len(missing)} SI figure reference(s) unresolved {stage}; no manifest "
            "written: " + ", ".join(missing)
        )


def retire_flat_figures(key_dir: Path, graveyard: str) -> Path | None:
    """Move the flat ``si/images/<file>`` figures to the graveyard.

    Returns the staging directory's former path, or None when there are no flat
    figures and no staging directory (left by a killed run) to retire. The files
    are moved into :func:`staging_dir` so the graveyard entry is named per key, then
    ``deprecate.sh`` moves that directory with the caller's full environment. If it
    exits non-zero while the staging directory still exists, every staged file is
    moved back into ``si/images/``; if the directory is gone, ``deprecate.sh`` had
    already moved it, and the error says to look in the graveyard. Either way
    :class:`RetireError` carries the real exit code and stderr.
    """
    files = flat_si_figures(key_dir)
    staging = staging_dir(key_dir)
    if not files and not staging.exists():
        return None
    staging.mkdir(exist_ok=True)
    for path in files:
        path.rename(staging / path.name)
    proc = subprocess.run(
        ["bash", str(DEPRECATE_SH), str(staging), RETIRE_REASON],
        env={**os.environ, "DEPRECATED_DIR": graveyard},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if staging.exists():
            images = key_dir / "si" / "images"
            images.mkdir(exist_ok=True)
            for path in sorted(staging.iterdir()):
                path.rename(images / path.name)
            staging.rmdir()
            where = "flat figures moved back into si/images/"
        else:
            where = f"{staging.name} is no longer in si/; look for it in {graveyard}"
        raise RetireError(
            f"deprecate.sh exited {proc.returncode} retiring {staging}; {where}: "
            f"{proc.stderr.strip()}"
        )
    log.info("retired %s: %s", staging, proc.stdout.strip())
    return staging


def _backfill_enriched(
    key_dir: Path, index: dict[str, dict[str, Any]], lib: ZoteroLibrary
) -> None:
    """Force-backfill ``key_dir``; restore the previous manifest unless enriched."""
    manifest = key_dir / MANIFEST_FILENAME
    previous = manifest.read_bytes() if manifest.is_file() else None
    result = backfill_key(key_dir, citation_index=index, lib=lib, force=True)
    if result.mode != "enriched":
        if previous is None:
            manifest.unlink()
        else:
            manifest.write_bytes(previous)
        raise NotEnrichedError(
            f"backfill of {key_dir.name} returned mode {result.mode!r}, not "
            "'enriched'; the previous manifest was restored"
        )
    log.info("backfill %s: enriched, %d files", key_dir.name, result.n_files)


def reocr_keys(
    root: Path, keys: list[str], *, graveyard: str, device_mode: str
) -> None:
    """Phases 0 to 5 of the module docstring for ``keys`` under the mirror ``root``."""
    key_dirs = [root / key for key in keys]
    lib = ZoteroLibrary.from_env()
    index, duplicates = citation_index_with_duplicates(lib)
    requested = {key: duplicates[key] for key in keys if key in duplicates}
    if requested:
        raise DuplicateCitationKeyError(
            "Zotero items share a requested citation key, nothing OCR'd "
            f"({describe_duplicates(requested)})"
        )
    if duplicates:
        log.warning(
            "Zotero items share a citation key this run does not process; "
            "proceeding (%s)",
            describe_duplicates(duplicates),
        )
    absent = [key for key in keys if key not in index]
    if absent:
        raise KeyNotInZoteroError(
            f"not in the Zotero citation index, nothing OCR'd: {', '.join(absent)}"
        )
    for key_dir in key_dirs:
        dry = backfill_key(
            key_dir, citation_index=index, lib=lib, force=True, dry_run=True
        )
        if dry.mode != "enriched":
            raise NotEnrichedError(
                f"dry-run backfill of {key_dir.name} returned mode {dry.mode!r}, not "
                "'enriched'; nothing OCR'd"
            )
    flat_refs = [
        f"{key_dir.name}/si/{ref}"
        for key_dir in key_dirs
        for ref in unrewritten_flat_refs(key_dir)
    ]
    if flat_refs:
        raise UnrewrittenFlatReferenceError(
            "markdown with no SI PDF to re-OCR references flat si/images/ figures, "
            "which retirement would break; nothing OCR'd: " + ", ".join(flat_refs)
        )
    plan = {key_dir: si_pdfs(key_dir) for key_dir in key_dirs}
    for pdfs in plan.values():
        for pdf in pdfs:
            log.info("re-OCR %s", pdf.relative_to(root))
            ocr_pdf(pdf, device_mode=device_mode)
    check_references(key_dirs, "after re-OCR, nothing retired")
    for key_dir in key_dirs:
        retire_flat_figures(key_dir, graveyard)
    check_references(key_dirs, f"after retiring flat figures to {graveyard}")
    for key_dir in key_dirs:
        _backfill_enriched(key_dir, index, lib)


def main(argv: list[str] | None = None) -> int:
    """Command line: ``--key`` (repeatable), ``--root``, ``--graveyard``,
    ``--device-mode``. The caller loads ``.env`` first.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--key", action="append", required=True, help="Citation key; repeatable."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="torchcell-library dir (default: $DATA_ROOT/torchcell-library).",
    )
    parser.add_argument("--graveyard", default=DEFAULT_GRAVEYARD)
    parser.add_argument("--device-mode", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args(argv)
    root = args.root or library_root(os.environ["DATA_ROOT"])
    reocr_keys(root, args.key, graveyard=args.graveyard, device_mode=args.device_mode)
    return 0
