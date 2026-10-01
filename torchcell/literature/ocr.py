# torchcell/literature/ocr.py
# [[torchcell.literature.ocr]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/ocr.py
# Test file: tests/torchcell/literature/test_ocr.py

"""OCR of paper and SI PDFs into markdown artifacts."""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from torchcell.literature.manifest import (
    OCR_PROVENANCE_SUFFIX,
    ProcessingRecord,
    sha256_file,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The isolated MinerU env. MinerU pins torch<2.11 + PaddleOCR, so it lives in
# its own conda env invoked as a subprocess -- never imported into torchcell.
# Override with $TORCHCELL_MINERU_PYTHON (e.g. once a torchcell-mineru env exists).
DEFAULT_MINERU_PYTHON = os.path.expanduser("~/miniconda3/envs/swanki-mineru/bin/python")
_RUNNER = Path(__file__).resolve().parent / "_run_mineru.py"


class PdfNotFoundError(FileNotFoundError):
    """The PDF handed to :func:`ocr_pdf` is not a file."""


class MissingPaperPdfError(FileNotFoundError):
    """An artifact directory handed to :func:`ocr_artifact` has no ``paper.pdf``."""


class RunnerReportError(RuntimeError):
    """The runner exited 0 without reporting its MinerU version and DPI."""


#: Facts ``_run_mineru.py`` prints on stdout, one ``KEY=value`` line each.
_RUNNER_FACTS = ("MINERU_VERSION", "MINERU_DPI")


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


def images_dir_for(pdf_path: Path) -> str:
    """Figures directory for one PDF, relative to the PDF's directory.

    ``paper.pdf`` keeps the flat ``images/`` every mirrored key already has. Any other
    PDF gets ``images/<stem>/``: all of a key's ``si/si*.pdf`` write into ``si/``, and a
    key root can hold other PDFs beside ``paper.pdf`` (Costanzo 2016's ``SOM.pdf``,
    Lopez's ``thesis.pdf``), so a shared directory would hold only the last PDF's
    figures (issue #579). The runner replaces only the files directly in a PDF's own
    directory, so the paper's re-run never touches ``images/<stem>/``.
    """
    return "images" if pdf_path.name == "paper.pdf" else f"images/{pdf_path.stem}"


def natural_key(path: Path) -> list[int | str]:
    """Sort key that orders digit runs numerically: ``si2`` before ``si10``."""
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


def _runner_facts(stdout: str, pdf_name: str) -> dict[str, str]:
    """The ``KEY=value`` facts the runner printed, each required exactly once."""
    facts: dict[str, list[str]] = {key: [] for key in _RUNNER_FACTS}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in facts:
            facts[key].append(value)
    bad = [key for key, values in facts.items() if len(values) != 1]
    if bad:
        raise RunnerReportError(
            f"MinerU runner on {pdf_name} must print each of {', '.join(_RUNNER_FACTS)} "
            f"exactly once; wrong count for {', '.join(bad)}"
        )
    return {key: values[0] for key, values in facts.items()}


def ocr_provenance_path(md_path: Path) -> Path:
    """Where :func:`ocr_pdf` writes the processing record of ``<stem>.md``."""
    return md_path.with_name(f"{md_path.stem}{OCR_PROVENANCE_SUFFIX}")


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
    runner writes ``<stem>.md`` (plus its figures in :func:`images_dir_for` and the
    layout JSON) into the PDF's directory. Then the processing record
    (:class:`~torchcell.literature.manifest.ProcessingRecord`: the MinerU version
    and effective DPI the runner reported, the arguments, the exact command and
    the PDF's sha256) is written to ``<stem>_ocr_provenance.json`` beside the
    markdown, where ``build_manifest`` attaches it to the markdown's record.
    Raises :class:`PdfNotFoundError` before starting MinerU when the PDF is not a
    file, ``RuntimeError`` on a non-zero exit -- no silent skip.

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
    if not pdf_path.is_file():
        raise PdfNotFoundError(f"PDF not found: {pdf_path}")
    out_dir = pdf_path.parent
    images_rel = images_dir_for(pdf_path)
    env = os.environ.copy()
    hf_home = _hf_home()
    if hf_home:
        env["HF_HOME"] = hf_home
    env["MINERU_MODEL_SOURCE"] = "huggingface"
    env["MINERU_DEVICE_MODE"] = device_mode or os.environ.get(
        "MINERU_DEVICE_MODE", "cuda"
    )

    dpi_requested = _resolve_dpi(dpi)
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
        str(dpi_requested),
        "--images-dir",
        images_rel,
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
    facts = _runner_facts(proc.stdout, pdf_path.name)
    record = ProcessingRecord(
        processor="torchcell.literature.ocr.ocr_pdf",
        tool="mineru",
        version=facts["MINERU_VERSION"],
        params={
            "backend": backend,
            "lang": lang,
            "method": method,
            "device_mode": env["MINERU_DEVICE_MODE"],
            "dpi_requested": dpi_requested,
            "dpi": int(facts["MINERU_DPI"]),
            "images_dir": images_rel,
            "command": cmd,
        },
        input_sha256=[sha256_file(pdf_path)],
    )
    ocr_provenance_path(md_path).write_text(record.model_dump_json(indent=2))
    log.info("MinerU: wrote %s (%d bytes)", md_path, md_path.stat().st_size)
    return md_path


def ocr_artifact(artifact_dir: str | Path, **kwargs: Any) -> list[Path]:
    """OCR every PDF in an artifact directory: ``paper.pdf`` and each ``si/si*.pdf``.

    The SI PDFs run in natural order (``si2`` before ``si10``). A directory with no
    ``paper.pdf`` raises :class:`MissingPaperPdfError` before any OCR: every artifact
    is captured with its article as ``paper.pdf`` (``ZoteroLibrary.download_artifact``),
    so its absence means a broken capture, not an SI-only paper.

    Returns the list of markdown paths produced (``paper.md``, ``si/si1.md``,
    ``si/si2.md``...).
    """
    artifact_dir = Path(artifact_dir)
    paper = artifact_dir / "paper.pdf"
    if not paper.is_file():
        raise MissingPaperPdfError(f"no paper.pdf in artifact directory {artifact_dir}")
    produced = [ocr_pdf(paper, **kwargs)]
    si_dir = artifact_dir / "si"
    if si_dir.is_dir():
        for si_pdf in sorted(si_dir.glob("si*.pdf"), key=natural_key):
            produced.append(ocr_pdf(si_pdf, **kwargs))
    return produced
