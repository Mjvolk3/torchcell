#!/usr/bin/env python
# scripts/lit_reocr_si.py
# [[scripts.lit_reocr_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_reocr_si.py

r"""Re-OCR every SI PDF of a mirror key into per-PDF figure directories (issue #579).

Before PR #585 every ``si/si*.pdf`` of a key wrote its figures into one shared
``si/images/``, and each PDF's OCR deleted the previous one's, so a key with several
SI PDFs has SI markdown pointing at figures that are gone. This script repairs such a
key, in an order where nothing is retired before the OCR has succeeded:

1. OCR every ``si/si*.pdf`` of every ``--key`` in natural order with
   :func:`torchcell.literature.ocr.ocr_pdf`. Each PDF's figures go to
   ``si/images/<stem>/``, its references are rewritten there, and its processing
   record is written beside its markdown. ``paper.pdf`` is NOT re-OCR'd, so
   ``paper.md`` keeps its bytes.
2. Refuse (exit 1, nothing retired, no manifest written) if any SI markdown of any
   key still references a figure that is not on disk.
3. Retire the stale flat ``si/images/<file>`` figures of the old layout: they are
   moved into ``si/<key>__si-images-flat-pre579/`` (a name unique per key) and that
   directory goes to the deprecation graveyard through ``scripts/deprecate.sh``.
   Nothing is deleted.
4. Rewrite each key's ``manifest.json`` with ``backfill_key(force=True)``, enriched
   from Zotero exactly as the existing manifests are (Zotero is only read).

MinerU runs in its isolated env on the GPU by default (``--device-mode cuda``), so
run it under slurm, one GPU, from the repo root::

    sbatch -p main -N 1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=64g \
      --time=0-06:00:00 -J lit-reocr-579 --output=lit-reocr-579_%j.out \
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


class UnresolvedFigureError(RuntimeError):
    """SI markdown references figures that are not on disk after the re-OCR."""


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


def retire_flat_figures(key_dir: Path, graveyard: str) -> Path | None:
    """Move the flat ``si/images/<file>`` figures to the graveyard; return the
    staging directory's former path, or None when there is nothing to retire.

    The files are first moved into ``si/<key>__si-images-flat-pre579/`` so the
    graveyard entry is named per key (``deprecate.sh`` names it after the basename),
    then ``deprecate.sh`` moves that directory and records where it came from.
    """
    files = flat_si_figures(key_dir)
    if not files:
        return None
    staging = key_dir / "si" / f"{key_dir.name}__si-images-flat-pre579"
    staging.mkdir()
    for path in files:
        path.rename(staging / path.name)
    env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
    env["DEPRECATED_DIR"] = graveyard
    subprocess.run(
        ["bash", str(DEPRECATE_SH), str(staging), RETIRE_REASON], env=env, check=True
    )
    return staging


def reocr_keys(
    root: Path, keys: list[str], *, graveyard: str, device_mode: str
) -> None:
    """Steps 1 to 4 of the module docstring for ``keys`` under the mirror ``root``."""
    key_dirs = [root / key for key in keys]
    plan = {key_dir: si_pdfs(key_dir) for key_dir in key_dirs}
    for key_dir, pdfs in plan.items():
        for pdf in pdfs:
            log.info("re-OCR %s", pdf.relative_to(root))
            ocr_pdf(pdf, device_mode=device_mode)
    missing = [
        f"{key_dir.name}/si/{ref}"
        for key_dir in key_dirs
        for ref in unresolved_si_figure_refs(key_dir)
    ]
    if missing:
        raise UnresolvedFigureError(
            f"{len(missing)} SI figure reference(s) unresolved after re-OCR; nothing "
            "retired, no manifest written: " + ", ".join(missing)
        )
    for key_dir in key_dirs:
        retire_flat_figures(key_dir, graveyard)
    lib = ZoteroLibrary.from_env()
    index = build_citation_index(lib)
    for key_dir in key_dirs:
        result = backfill_key(key_dir, citation_index=index, lib=lib, force=True)
        log.info(
            "backfill %s: %s, %d files",
            result.citation_key,
            result.mode,
            result.n_files,
        )


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
