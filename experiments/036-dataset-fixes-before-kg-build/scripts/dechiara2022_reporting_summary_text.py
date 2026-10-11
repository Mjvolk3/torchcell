# experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_reporting_summary_text.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.dechiara2022_reporting_summary_text]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_reporting_summary_text

"""Render the De Chiara 2022 Reporting Summary's text layer into the mirror.

The Reporting Summary (``si/si2.pdf``, MOESM2) is the only place the paper states its
replicate design ("Automatic growth curve were conducted in 4 replicates, median values
were taken for each isolates.") and its sporulation exclusion rule. MinerU's OCR of it
(``si/si2.md``) cut the study-design table to fragments ("least four replicas) to
verify reproducibility ..."), but the PDF carries a text layer, so ``pdftotext
-layout`` reads those sentences verbatim. This script writes that rendering to
``si/si2.pdftotext.txt`` beside the PDF and records it in the key's ``manifest.json``
with a ``ProcessingRecord`` (tool, version, flags, input sha256), the pattern of the
Caudal 2024 ``methods.md`` derivation. A loader quote then audits against these bytes.

Idempotent: a path already recorded with the same sha256 is left alone; a different
sha256 raises, since a re-render that changed would be a new version, not an overwrite.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_reporting_summary_text.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    ROLE_SI_TEXT,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    write_manifest,
)

CITATION_KEY = "dechiaraDomesticationReprogrammedBudding2022"
PDF_REL = "si/si2.pdf"
PDF_SHA256 = "d51bc60064c2514b3a1d1487eefdf274256423b9cd1df051863fd4d67cd11af7"
TEXT_REL = "si/si2.pdftotext.txt"
FLAGS = ["-layout"]
PROCESSOR = (
    "experiments/036-dataset-fixes-before-kg-build/scripts/"
    "dechiara2022_reporting_summary_text.py:render"
)


def _pdftotext_version() -> str:
    """The first line pdftotext prints for ``-v`` (it writes it to stderr)."""
    result = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    )
    return result.stderr.splitlines()[0].strip()


def render(pdf: Path) -> bytes:
    """``pdftotext -layout <pdf> -`` as bytes."""
    result = subprocess.run(
        ["pdftotext", *FLAGS, str(pdf), "-"], capture_output=True, check=True
    )
    return result.stdout


def main() -> None:
    """Render, write and record the Reporting Summary text for the De Chiara key."""
    load_dotenv()
    artifact_dir = Path(os.environ["DATA_ROOT"]) / "torchcell-library" / CITATION_KEY
    pdf = artifact_dir / PDF_REL
    observed = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if observed != PDF_SHA256:
        raise RuntimeError(f"{pdf}: sha256 {observed} != pinned {PDF_SHA256}")
    data = render(pdf)
    digest = hashlib.sha256(data).hexdigest()
    manifest = Manifest.model_validate_json(
        (artifact_dir / MANIFEST_FILENAME).read_text()
    )
    existing = [f for f in manifest.files if f.path == TEXT_REL]
    if existing:
        if existing[0].sha256 != digest:
            raise ValueError(
                f"{TEXT_REL} already recorded with sha256 {existing[0].sha256}, "
                f"rendered {digest}; version it by hand"
            )
        print(f"unchanged: {TEXT_REL} {digest}")
        return
    (artifact_dir / TEXT_REL).write_bytes(data)
    manifest.files.append(
        ArtifactRecord(
            path=TEXT_REL,
            role=ROLE_SI_TEXT,
            bytes=len(data),
            sha256=digest,
            source=f"derived from {PDF_REL}",
            processing=ProcessingRecord(
                processor=PROCESSOR,
                tool="pdftotext",
                version=_pdftotext_version(),
                params={"flags": FLAGS},
                input_sha256=[PDF_SHA256],
            ),
        )
    )
    write_manifest(artifact_dir, manifest)
    print(f"recorded: {TEXT_REL} {digest} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
