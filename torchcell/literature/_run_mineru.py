# torchcell/literature/_run_mineru.py
# [[torchcell.literature._run_mineru]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/_run_mineru.py
#
# Standalone MinerU runner, executed by the ISOLATED mineru conda env (never
# imported into the torchcell env). It takes one PDF, runs MinerU, and flattens
# the nested output tree so <out-dir>/<stem>.md sits next to its figures in
# <out-dir>/<images-dir>/ (``images`` for paper.pdf, ``images/<stem>`` for every
# other PDF, chosen by ocr.py), with the markdown and content-list references
# rewritten to that directory. A run replaces only the files directly in its own
# figures directory, never a sibling PDF's directory or a nested one.
#
# It prints two facts on stdout for ocr.py's provenance record:
# ``MINERU_VERSION=<mineru.version.__version__>`` and ``MINERU_DPI=<effective dpi>``.
#
# Adapted from Swanki's scripts/run_mineru_swanki.py. Imports only `mineru` +
# stdlib so it stays loadable in the minimal mineru env.
#
# Exit codes: 0 ok | 2 PDF missing | 3 no markdown produced | 4 HF_HOME underivable
#             | 5 the markdown references a figure MinerU did not write

import argparse
import inspect
import os
import re
import shutil
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MinerU on a single PDF.")
    p.add_argument("--pdf-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--backend", default="pipeline")
    p.add_argument("--lang", default="en")
    p.add_argument("--method", default="auto")
    # Page rasterization DPI. MinerU hardcodes 200, which starves dense tables
    # (~22 px/row on A4) and deterministically drops rows. <=0 keeps the default.
    p.add_argument("--dpi", type=int, default=0)
    # Figures directory, relative to --out-dir. Several PDFs share one out-dir
    # (every si/si*.pdf writes into si/), so each needs its own directory.
    p.add_argument("--images-dir", default="images")
    return p.parse_args()


def _patch_dpi(dpi: int) -> None:
    """Override the DPI MinerU rasterizes pages at, repo-wide.

    ``do_parse`` does not expose DPI; it calls ``load_images_from_pdf`` with the
    hardcoded default. We replace that function -- in its defining module and in
    every module whose namespace bound it by name -- with one that forces ``dpi``. Must
    run after the MinerU import chain so the importer modules already exist.
    """
    import sys

    import mineru.utils.pdf_image_tools as pit

    original = pit.load_images_from_pdf

    def patched(pdf_bytes, dpi=dpi, **kwargs):  # type: ignore[no-untyped-def]
        return original(pdf_bytes, dpi=dpi, **kwargs)

    # Read each module's __dict__ (a None entry has none), never the attribute: a
    # lazy module (transformers) answers an attribute lookup by importing optional
    # heavy dependencies such as torchvision.
    for module in list(sys.modules.values()):
        if getattr(module, "__dict__", {}).get("load_images_from_pdf") is original:
            module.load_images_from_pdf = patched  # type: ignore[attr-defined]  # monkey-patch mineru DPI on its module object
    pit.load_images_from_pdf = patched
    print(f"[mineru] page rasterization DPI -> {dpi}")


def _ensure_hf_home() -> int:
    """Set HF_HOME before MinerU import if not already set. Return 0 ok / 4 fail.

    MinerU reads the HuggingFace cache path at import time, so this must run
    before `from mineru...`. Falls back to $DATA_ROOT/models/mineru/hf_cache.
    """
    if os.environ.get("HF_HOME"):
        return 0
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        print("ERROR: set HF_HOME or DATA_ROOT for MinerU", file=sys.stderr)
        return 4
    hf_home = Path(data_root) / "models" / "mineru" / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    return 0


def _find_first(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        return path
    return None


# A figure reference in exactly the forms MinerU writes: the markdown image
# ``![](images/<f>)``, an HTML table cell ``<img src="images/<f>"``, and the
# content list's ``"img_path": "images/<f>"``. Anchored to those prefixes so prose
# such as ``https://example.org/images/logo.png`` is neither rewritten nor checked.
IMAGE_REF = re.compile(
    r'(?P<pre>\]\(|<img src="|"img_path": ")images/(?P<name>[^)"\n]+)'
)


class UnresolvedImageRefError(ValueError):
    """MinerU's output references a figure it did not write."""


def _rewrite_image_refs(text: str, produced: set[str], images_rel: str) -> str:
    """Point every ``images/<file>`` reference at ``<images_rel>/<file>``.

    ``produced`` is the set of figure file names MinerU wrote. A reference to any
    other file raises :class:`UnresolvedImageRefError`, so a markdown never ships
    pointing at a figure that is not beside it.
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in produced:
            raise UnresolvedImageRefError(f"references images/{name}, not produced")
        return f"{match.group('pre')}{images_rel}/{name}"

    return IMAGE_REF.sub(_sub, text)


def _replace_images_dir(images_src: Path | None, dest: Path, stage_root: Path) -> None:
    """Make the files directly in ``dest`` exactly this run's figures.

    The figures are copied into ``<stage_root>/.images.new`` first; only then are
    the files directly in ``dest`` (this PDF's own figures from an earlier run) moved
    into ``<stage_root>/.images.old`` and the staged files moved into ``dest``. The
    caller removes ``stage_root``. Subdirectories of ``dest`` are never touched: the
    paper's flat ``images/`` can hold another root PDF's ``images/<stem>/`` (Costanzo
    2016's ``SOM.pdf``). A ``dest`` left empty (MinerU wrote no figures) is removed.
    """
    staged = stage_root / ".images.new"
    retired = stage_root / ".images.old"
    if images_src is not None:
        shutil.copytree(images_src, staged)
    retired.mkdir()
    if dest.is_dir():
        for old in sorted(p for p in dest.iterdir() if p.is_file()):
            old.rename(retired / old.name)
    if images_src is not None:
        dest.mkdir(parents=True, exist_ok=True)
        for new in sorted(staged.iterdir()):
            new.rename(dest / new.name)
    if dest.is_dir() and not any(dest.iterdir()):
        dest.rmdir()


def main() -> int:
    args = _parse_args()
    pdf_path: Path = args.pdf_path.resolve()
    out_dir: Path = args.out_dir.resolve()
    stem = pdf_path.stem

    if not pdf_path.is_file():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    rc = _ensure_hf_home()
    if rc != 0:
        return rc

    os.environ.setdefault("MINERU_MODEL_SOURCE", "huggingface")

    # MinerU import MUST come after HF_HOME is set.
    import mineru.utils.pdf_image_tools as pit  # noqa: E402
    from mineru.cli.common import do_parse  # noqa: E402
    from mineru.version import __version__ as mineru_version  # noqa: E402

    default_dpi = inspect.signature(pit.load_images_from_pdf).parameters["dpi"].default
    if args.dpi > 0:
        _patch_dpi(args.dpi)
    print(f"MINERU_VERSION={mineru_version}")
    print(f"MINERU_DPI={args.dpi if args.dpi > 0 else default_dpi}")

    # This PDF's private staging path. A run killed earlier can leave it behind
    # (half-staged figures), so every run starts from a fresh one.
    scratch = out_dir / f".mineru_scratch_{stem}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    do_parse(
        output_dir=str(scratch),
        pdf_file_names=[stem],
        pdf_bytes_list=[pdf_path.read_bytes()],
        p_lang_list=[args.lang],
        backend=args.backend,
        parse_method=args.method,
    )

    md_src = _find_first(scratch, f"{stem}.md")
    if md_src is None:
        print(f"ERROR: MinerU produced no {stem}.md under {scratch}", file=sys.stderr)
        return 3

    auto_dir = md_src.parent
    images_src = auto_dir / "images"
    produced = (
        {p.name for p in images_src.iterdir() if p.is_file()}
        if images_src.is_dir()
        else set()
    )
    images_rel: str = args.images_dir
    rewritten: dict[str, str] = {}
    for name in (f"{stem}.md", f"{stem}_content_list.json"):
        src = auto_dir / name
        if src.exists():
            try:
                rewritten[name] = _rewrite_image_refs(
                    src.read_text(encoding="utf-8"), produced, images_rel
                )
            except UnresolvedImageRefError as err:
                shutil.rmtree(scratch)
                print(f"ERROR: {name} {err}", file=sys.stderr)
                return 5

    # Figures first, markdown after: a failure in the swap leaves the previous
    # markdown with the figures it references.
    _replace_images_dir(
        images_src if images_src.is_dir() else None, out_dir / images_rel, scratch
    )
    for name, text in rewritten.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    middle = auto_dir / f"{stem}_middle.json"
    if middle.exists():
        shutil.copy2(middle, out_dir / middle.name)

    shutil.rmtree(scratch)
    print(f"OK: {pdf_path.name} -> {out_dir}/{stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
