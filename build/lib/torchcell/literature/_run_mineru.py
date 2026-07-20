# torchcell/literature/_run_mineru.py
# [[torchcell.literature._run_mineru]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/_run_mineru.py
#
# Standalone MinerU runner, executed by the ISOLATED mineru conda env (never
# imported into the torchcell env). It takes one PDF, runs MinerU, and flattens
# the nested output tree so <out-dir>/<stem>.md sits next to its images/.
#
# Adapted from Swanki's scripts/run_mineru_swanki.py. Imports only `mineru` +
# stdlib so it stays loadable in the minimal mineru env.
#
# Exit codes: 0 ok | 2 PDF missing | 3 no markdown produced | 4 HF_HOME underivable

import argparse
import os
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
    return p.parse_args()


def _patch_dpi(dpi: int) -> None:
    """Override the DPI MinerU rasterizes pages at, repo-wide.

    ``do_parse`` does not expose DPI; it calls ``load_images_from_pdf`` with the
    hardcoded default. We replace that function -- in its defining module and in
    every module that imported it by name -- with one that forces ``dpi``. Must
    run after the MinerU import chain so the importer modules already exist.
    """
    import sys

    import mineru.utils.pdf_image_tools as pit

    original = pit.load_images_from_pdf

    def patched(pdf_bytes, dpi=dpi, **kwargs):  # type: ignore[no-untyped-def]
        return original(pdf_bytes, dpi=dpi, **kwargs)

    for module in list(sys.modules.values()):
        if getattr(module, "load_images_from_pdf", None) is original:
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
    from mineru.cli.common import do_parse  # noqa: E402

    if args.dpi > 0:
        _patch_dpi(args.dpi)

    scratch = out_dir / f".mineru_scratch_{stem}"
    scratch.mkdir(parents=True, exist_ok=True)

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
    shutil.copy2(md_src, out_dir / f"{stem}.md")
    for extra in (f"{stem}_content_list.json", f"{stem}_middle.json"):
        src = auto_dir / extra
        if src.exists():
            shutil.copy2(src, out_dir / extra)
    images_src = auto_dir / "images"
    if images_src.is_dir():
        dest_images = out_dir / "images"
        if dest_images.exists():
            shutil.rmtree(dest_images)
        shutil.copytree(images_src, dest_images)

    shutil.rmtree(scratch, ignore_errors=True)
    print(f"OK: {pdf_path.name} -> {out_dir}/{stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
