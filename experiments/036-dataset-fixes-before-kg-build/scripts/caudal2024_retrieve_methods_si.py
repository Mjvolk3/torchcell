# experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_retrieve_methods_si.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.caudal2024_retrieve_methods_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_retrieve_methods_si.py
"""Retrieve the Caudal 2024 online Methods + Supplementary Information into the mirror.

Issue #598: the mirror key ``caudalPantranscriptomeRevealsLarge2024`` held only the
9-page main text, and the Methods are online-only (``paper.md:151-153``), so nothing
mirrored defined the ``pan_absence`` classes or the 6,445-transcript set. This script
fetches, with the versioned retrievers in ``torchcell.literature.retrieve``:

- ``si/PMC11176082_fulltext.xml``: the Europe PMC full-text JATS XML (PMC11176082,
  open access), which carries the complete online Methods and Data availability.
  Two fetches on 2026-10-02 returned the same sha256, so the bytes are stable.
- ``si/41588_2024_1769_MOESM1_ESM.pdf``: Supplementary Information (Supplementary
  Figs. 1-12 and the Tables 1-9 descriptions), Springer ESM.
- ``si/41588_2024_1769_MOESM3_ESM.xlsx``: Supplementary Tables 1-9, Springer ESM.

and derives ``methods.md`` (the Methods, Data availability and supplementary-material
captions of the XML, one paragraph per line, so a loader quote can cite a line) with a
``ProcessingRecord`` pointing at ``jats_methods_to_markdown`` below.

Every file is recorded in the key's ``manifest.json`` (``ArtifactRecord`` with a
``RetrievalRecord`` or ``ProcessingRecord``). Re-running is idempotent: a path already
recorded with the same sha256 is left alone; a different sha256 raises (upstream drift
creates a new record by hand, it never overwrites the version a loader cites).

The NCBI OA web service (``torchcell.literature.retrieve.pmc_oa_api``) returned HTTP 404
for ``https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC11176082`` on 2026-10-02,
so the Europe PMC REST endpoint is used through ``direct_url`` instead.

Run from the repo root:
    python experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_retrieve_methods_si.py
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
    write_manifest,
)
from torchcell.literature.provenance import run_retriever

CITATION_KEY = "caudalPantranscriptomeRevealsLarge2024"
PMCID = "PMC11176082"
EPMC_XML_URL = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML"
ESM_BASE = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-024-01769-9/"
    "MediaObjects/"
)
XML_REL = f"si/{PMCID}_fulltext.xml"
METHODS_REL = "methods.md"
PROCESSOR = (
    "experiments.036-dataset-fixes-before-kg-build.scripts."
    "caudal2024_retrieve_methods_si.jats_methods_to_markdown"
)

# (relative path, role, retrieval method, retriever key, url)
RETRIEVALS: list[tuple[str, str, RetrievalMethod, str, str]] = [
    (
        XML_REL,
        "si_data",
        RetrievalMethod.direct_url,
        "torchcell.literature.retrieve.direct_url",
        EPMC_XML_URL,
    ),
    (
        "si/41588_2024_1769_MOESM1_ESM.pdf",
        "si_pdf",
        RetrievalMethod.springer_esm,
        "torchcell.literature.retrieve.springer_esm",
        ESM_BASE + "41588_2024_1769_MOESM1_ESM.pdf",
    ),
    (
        "si/41588_2024_1769_MOESM3_ESM.xlsx",
        "si_data",
        RetrievalMethod.springer_esm,
        "torchcell.literature.retrieve.springer_esm",
        ESM_BASE + "41588_2024_1769_MOESM3_ESM.xlsx",
    ),
]


def _tag(elem: ET.Element) -> str:
    """Local tag name without an XML namespace."""
    return elem.tag.rsplit("}", 1)[-1]


def _text(elem: ET.Element) -> str:
    """Flatten a JATS element to one whitespace-normalized line.

    Math in ``<alternatives>`` is rendered once, from its ``tex-math`` body (the part
    between ``$$`` delimiters); the parallel MathML and graphic are skipped.
    """
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        tag = _tag(node)
        if tag == "alternatives":
            for child in node:
                if _tag(child) == "tex-math" and child.text:
                    body = child.text.split("$$")
                    parts.append(f"${body[1]}$" if len(body) >= 3 else child.text)
            parts.append(node.tail or "")
            return
        parts.append(node.text or "")
        for child in node:
            walk(child)
        parts.append(node.tail or "")

    parts.append(elem.text or "")
    for child in elem:
        walk(child)
    return " ".join("".join(parts).split())


def _required(elem: ET.Element, path: str) -> ET.Element:
    """``elem.find(path)``, raising when the element is missing."""
    found = elem.find(path)
    if found is None:
        raise ValueError(f"JATS element {path!r} missing under <{_tag(elem)}>")
    return found


def jats_methods_to_markdown(xml_bytes: bytes) -> str:
    """Render the Methods, availability notes and SI captions of a JATS article.

    One paragraph per line, so a quote can be cited by line number. Raises when the
    article has no section titled ``Methods`` (nothing is silently emitted empty).
    """
    root = ET.fromstring(xml_bytes)
    title = _text(_required(root, ".//article-title"))
    methods = [
        sec
        for sec in root.iter("sec")
        if sec.find("title") is not None and _text(_required(sec, "title")) == "Methods"
    ]
    if len(methods) != 1:
        raise ValueError(f"expected one 'Methods' sec, found {len(methods)}")
    lines = [
        f"# {title} -- Methods (Europe PMC {PMCID} full-text XML)",
        "",
        f"Derived from `{XML_REL}` by `{PROCESSOR}`.",
        "",
    ]

    def emit_sec(sec: ET.Element, level: int) -> None:
        head = sec.find("title")
        if head is not None:
            lines.extend([f"{'#' * level} {_text(head)}", ""])
        for child in sec:
            if _tag(child) == "p":
                lines.extend([_text(child), ""])
            elif _tag(child) == "sec":
                emit_sec(child, level + 1)

    emit_sec(methods[0], 2)
    for sec in root.iter():
        head = sec.find("title")
        if (
            _tag(sec) in ("sec", "notes")
            and head is not None
            and "availability" in _text(head).lower()
        ):
            emit_sec(sec, 2)
    lines.extend(["## Supplementary material captions", ""])
    for supp in root.iter("supplementary-material"):
        media = supp.find("media")
        href = (
            media.get("{http://www.w3.org/1999/xlink}href", "")
            if media is not None
            else ""
        )
        label = media.find("label") if media is not None else None
        caption = media.find("caption") if media is not None else None
        text = "; ".join(_text(part) for part in (label, caption) if part is not None)
        lines.extend([f"- `{href}`: {text}", ""])
    return "\n".join(lines).rstrip() + "\n"


def _record(
    manifest: Manifest, artifact_dir: Path, record: ArtifactRecord, data: bytes
) -> None:
    """Write ``data`` at ``record.path`` and add the record, refusing silent drift."""
    existing = [f for f in manifest.files if f.path == record.path]
    if existing:
        if existing[0].sha256 != record.sha256:
            raise ValueError(
                f"{record.path} already recorded with sha256 {existing[0].sha256}, "
                f"retrieved {record.sha256}: upstream changed; version it by hand"
            )
        print(f"unchanged: {record.path} {record.sha256}")
        return
    dest = artifact_dir / record.path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    manifest.files.append(record)
    print(f"recorded: {record.path} {record.sha256} ({len(data)} bytes)")


def main() -> None:
    """Retrieve, write and record every Methods/SI artifact for the Caudal key."""
    load_dotenv()
    artifact_dir = Path(os.environ["DATA_ROOT"]) / "torchcell-library" / CITATION_KEY
    manifest = Manifest.model_validate_json(
        (artifact_dir / MANIFEST_FILENAME).read_text()
    )
    now = datetime.now(UTC).isoformat()
    xml_bytes = b""
    for rel, role, method, retriever, url in RETRIEVALS:
        retrieval = RetrievalRecord(
            method=method,
            source_url=url,
            retriever=retriever,
            params={"url": url},
            sha256="0" * 64,
            retrieved_at=now,
        )
        data = run_retriever(retrieval)
        digest = hashlib.sha256(data).hexdigest()
        retrieval = retrieval.model_copy(update={"sha256": digest})
        _record(
            manifest,
            artifact_dir,
            ArtifactRecord(
                path=rel,
                role=role,
                bytes=len(data),
                sha256=digest,
                source=url,
                retrieval=retrieval,
            ),
            data,
        )
        if rel == XML_REL:
            xml_bytes = data
    methods = jats_methods_to_markdown(xml_bytes).encode("utf-8")
    _record(
        manifest,
        artifact_dir,
        ArtifactRecord(
            path=METHODS_REL,
            role="si_ocr",
            bytes=len(methods),
            sha256=hashlib.sha256(methods).hexdigest(),
            source=f"derived:{XML_REL}",
            processing=ProcessingRecord(
                processor=PROCESSOR,
                tool="xml.etree.ElementTree",
                version="python-stdlib",
                params={"sections": ["Methods", "*availability*"]},
                input_sha256=[hashlib.sha256(xml_bytes).hexdigest()],
            ),
        ),
        methods,
    )
    write_manifest(artifact_dir, manifest)


if __name__ == "__main__":
    main()
