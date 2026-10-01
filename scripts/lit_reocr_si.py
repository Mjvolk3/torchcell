#!/usr/bin/env python
# scripts/lit_reocr_si.py
# [[scripts.lit_reocr_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_reocr_si.py

r"""Re-OCR every SI PDF of a mirror key into per-PDF figure directories (issue #579).

Before PR #585 every ``si/si*.pdf`` of a key wrote its figures into one shared
``si/images/``, and each PDF's OCR deleted the previous one's, so a key with several
SI PDFs has SI markdown pointing at figures that are gone. This script repairs such
keys in phases, each over every ``--key`` before the next phase starts, so nothing is
retired before every OCR has succeeded and no manifest is written before every
retirement has:

0. Resolve every key in the Zotero citation index (Zotero is only read). A key
   that is not there raises :class:`KeyNotInZoteroError` before any OCR.
1. OCR every ``si/si*.pdf`` in natural order with
   :func:`torchcell.literature.ocr.ocr_pdf`. Each PDF's figures go to
   ``si/images/<stem>/``, its references are rewritten there, and its processing
   record is written beside its markdown. ``paper.pdf`` is NOT re-OCR'd.
2. Check that every SI markdown reference resolves; refuse with
   :class:`UnresolvedFigureError` otherwise (nothing retired, no manifest written).
3. Retire the stale flat ``si/images/<file>`` figures: they are moved into
   ``si/<key>__si-images-flat-pre579/`` and ``scripts/deprecate.sh`` moves that
   directory to the graveyard. If ``deprecate.sh`` refuses (for example a graveyard
   inside ``$DATA_ROOT``) the files are moved back and :class:`RetireError` is
   raised. A staging directory left by a killed run is retired on the next run.
4. Check the references again: a markdown that still pointed at a flat figure
   now dangles, and :class:`UnresolvedFigureError` names it before any manifest is
   written. The retired figures are in the graveyard entry named in the log; move
   the named file back into ``si/images/`` by hand and investigate that markdown.
5. Rewrite each ``manifest.json`` with ``backfill_key(force=True)`` and refuse
   with :class:`NotEnrichedError` unless the result is ``enriched``.

MinerU runs in its isolated env on the GPU by default (``--device-mode cuda``). Run
it under slurm with one GPU from the repo root; the card is the owner's choice, so
replace ``<CARD>`` (see the PR #585 body). ``--mem=64g`` is not measured::

    sbatch -p main -N 1 --ntasks=1 --gres=gpu:rtx6000:1 <CARD> --cpus-per-task=8 \
      --mem=64g --time=0-06:00:00 -J lit-reocr-579 \
      --output=$HOME/lit-reocr-579_%j.out \
      --wrap "cd $HOME/Documents/projects/torchcell && PYTHONPATH=\$PWD \
    $HOME/miniconda3/envs/torchcell/bin/python scripts/lit_reocr_si.py \
    --key leeMappingCellularResponse2014 \
    --key ohyaHighdimensionalLargescalePhenotyping2005"

Reads ``DATA_ROOT`` (the mirror is ``$DATA_ROOT/torchcell-library``, MinerU's model
cache ``$DATA_ROOT/models/mineru/hf_cache``) and the ``ZOTERO_*`` credentials from
the repo's ``.env``.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from torchcell.literature._run_mineru import IMAGE_REF
from torchcell.literature.backfill import (
    backfill_key,
    build_citation_index,
    library_root,
)
from torchcell.literature.ocr import natural_key, ocr_pdf
from torchcell.literature.zotero import ZoteroLibrary

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPRECATE_SH = _PROJECT_ROOT / "scripts" / "deprecate.sh"
DEFAULT_GRAVEYARD = "/scratch/projects/torchcell-deprecated"
RETIRE_REASON = "issue #579: flat shared SI figures replaced by per-PDF re-OCR"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class NoSiPdfError(FileNotFoundError):
    """A key handed to the re-OCR has no ``si/si*.pdf``."""


class KeyNotInZoteroError(LookupError):
    """A requested key has no item in the Zotero citation index."""


class UnresolvedFigureError(RuntimeError):
    """SI markdown references figures that are not on disk."""


class RetireError(RuntimeError):
    """``deprecate.sh`` failed; the flat figures were moved back."""


class NotEnrichedError(RuntimeError):
    """A backfill wrote a manifest that is not Zotero-enriched."""


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


def unresolved_si_figure_refs(key_dir: Path) -> list[str]:
    """Every ``<md name>: images/<name>`` reference in ``si/*.md`` with no file."""
    si = key_dir / "si"
    missing: list[str] = []
    for md in sorted(si.glob("*.md"), key=natural_key):
        for match in IMAGE_REF.finditer(md.read_text(encoding="utf-8")):
            ref = f"images/{match.group('name')}"
            if not (si / ref).is_file():
                missing.append(f"{md.name}: {ref}")
    return missing


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
    ``deprecate.sh`` moves that directory with the caller's full environment, so its
    refusal of a graveyard inside ``$DATA_ROOT`` applies. If it fails, every staged
    file is moved back into ``si/images/``, the staging directory is removed and
    :class:`RetireError` carries its exit code and stderr.
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
        images = key_dir / "si" / "images"
        images.mkdir(exist_ok=True)
        for path in sorted(staging.iterdir()):
            path.rename(images / path.name)
        staging.rmdir()
        raise RetireError(
            f"deprecate.sh exited {proc.returncode} retiring {staging}; flat figures "
            f"moved back: {proc.stderr.strip()}"
        )
    log.info("retired %s: %s", staging, proc.stdout.strip())
    return staging


def reocr_keys(
    root: Path, keys: list[str], *, graveyard: str, device_mode: str
) -> None:
    """Phases 0 to 5 of the module docstring for ``keys`` under the mirror ``root``."""
    key_dirs = [root / key for key in keys]
    lib = ZoteroLibrary.from_env()
    index: dict[str, dict[str, Any]] = build_citation_index(lib)
    absent = [key for key in keys if key not in index]
    if absent:
        raise KeyNotInZoteroError(
            f"not in the Zotero citation index, nothing OCR'd: {', '.join(absent)}"
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
        result = backfill_key(key_dir, citation_index=index, lib=lib, force=True)
        if result.mode != "enriched":
            raise NotEnrichedError(
                f"backfill of {key_dir.name} wrote a {result.mode} manifest, "
                "not enriched"
            )
        log.info("backfill %s: enriched, %d files", key_dir.name, result.n_files)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    load_dotenv(_PROJECT_ROOT / ".env")
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


if __name__ == "__main__":
    sys.exit(main())
