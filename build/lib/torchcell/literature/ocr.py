# torchcell/literature/ocr.py
# [[torchcell.literature.ocr]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/ocr.py
# Test file: tests/torchcell/literature/test_ocr.py

"""OCR of paper and SI PDFs into markdown artifacts."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The isolated MinerU env. MinerU pins torch<2.11 + PaddleOCR, so it lives in
# its own conda env invoked as a subprocess -- never imported into torchcell.
# Override with $TORCHCELL_MINERU_PYTHON (e.g. once a torchcell-mineru env exists).
DEFAULT_MINERU_PYTHON = os.path.expanduser("~/miniconda3/envs/swanki-mineru/bin/python")
_RUNNER = Path(__file__).resolve().parent / "_run_mineru.py"


def _mineru_python() -> str:
    return os.environ.get("TORCHCELL_MINERU_PYTHON", DEFAULT_MINERU_PYTHON)


def _hf_home() -> str | None:
    """Resolve the HuggingFace cache for MinerU models.

    Honors an explicit $HF_HOME, else derives $DATA_ROOT/models/mineru/hf_cache.
    """
    explicit = os.environ.get("HF_HOME")
    if explicit:
        return explicit
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        return str(Path(data_root) / "models" / "mineru" / "hf_cache")
    return None


def _resolve_dpi(dpi: int | None) -> int:
    """DPI to rasterize at: explicit arg, else $TORCHCELL_MINERU_DPI, else 0.

    0 means "leave MinerU's default (200)". For the VLM backend on dense tables,
    ~350 is the sweet spot -- full quality just under Qwen's pixel budget.
    """
    if dpi is not None:
        return dpi
    return int(os.environ.get("TORCHCELL_MINERU_DPI", "0"))


def ocr_pdf(
    pdf_path: str | Path,
    *,
    backend: str = "pipeline",
    lang: str = "en",
    method: str = "auto",
    device_mode: str | None = None,
    dpi: int | None = None,
    timeout: int = 3600,
) -> Path:
    """OCR one PDF to markdown next to it (``<stem>.pdf`` -> ``<stem>.md``).

    Runs the standalone ``_run_mineru.py`` under the isolated MinerU env. The
    runner writes ``<stem>.md`` (plus ``images/`` and layout JSON) into the PDF's
    directory. Raises ``RuntimeError`` on non-zero exit -- no silent skip.

    Args:
        pdf_path: PDF to OCR.
        backend: MinerU backend (pipeline | vlm-auto-engine | hybrid-auto-engine).
        lang: OCR language hint passed to MinerU.
        method: MinerU parse method (auto | txt | ocr).
        device_mode: cuda | cpu; defaults to $MINERU_DEVICE_MODE or "cuda".
        dpi: Page rasterization DPI; defaults to $TORCHCELL_MINERU_DPI or MinerU's
            200. Raising this (e.g. 350) recovers rows that low resolution drops.
        timeout: Subprocess timeout (s). First run downloads models -- be generous.

    Returns:
        Path to the produced markdown file.
    """
    pdf_path = Path(pdf_path)
    out_dir = pdf_path.parent
    env = os.environ.copy()
    hf_home = _hf_home()
    if hf_home:
        env["HF_HOME"] = hf_home
    env["MINERU_MODEL_SOURCE"] = "huggingface"
    env["MINERU_DEVICE_MODE"] = device_mode or os.environ.get(
        "MINERU_DEVICE_MODE", "cuda"
    )

    cmd = [
        _mineru_python(),
        str(_RUNNER),
        "--pdf-path",
        str(pdf_path),
        "--out-dir",
        str(out_dir),
        "--backend",
        backend,
        "--lang",
        lang,
        "--method",
        method,
        "--dpi",
        str(_resolve_dpi(dpi)),
    ]
    log.info("MinerU: OCR %s (device=%s)", pdf_path.name, env["MINERU_DEVICE_MODE"])
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MinerU failed (exit {proc.returncode}) on {pdf_path.name}:\n"
            f"{proc.stderr[-2000:]}"
        )
    md_path = pdf_path.with_suffix(".md")
    if not md_path.exists():
        raise RuntimeError(f"MinerU reported success but {md_path} is missing")
    log.info("MinerU: wrote %s (%d bytes)", md_path, md_path.stat().st_size)
    return md_path


def ocr_artifact(artifact_dir: str | Path, **kwargs: Any) -> list[Path]:
    """OCR every PDF in an artifact directory: ``paper.pdf`` and each ``si/si*.pdf``.

    Returns the list of markdown paths produced (``paper.md``, ``si/si1.md``...).
    """
    artifact_dir = Path(artifact_dir)
    produced: list[Path] = []
    paper = artifact_dir / "paper.pdf"
    if paper.exists():
        produced.append(ocr_pdf(paper, **kwargs))
    si_dir = artifact_dir / "si"
    if si_dir.is_dir():
        for si_pdf in sorted(si_dir.glob("si*.pdf")):
            produced.append(ocr_pdf(si_pdf, **kwargs))
    return produced
