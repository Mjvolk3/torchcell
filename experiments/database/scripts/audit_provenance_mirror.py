# experiments/database/scripts/audit_provenance_mirror.py
# [[experiments.database.audit_provenance_mirror]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/audit_provenance_mirror
r"""Audit what the two provenance mirrors ACTUALLY store, against what they claim.

The mirrors are the rebuild guarantee: `$DATA_ROOT/torchcell-library/` holds each
paper's PDF + OCR artifacts and `$DATA_ROOT/torchcell-raw/` holds the raw dataset
files a loader consumed, each under a `manifest.json` that is supposed to pin every
file by sha256 and say how it was retrieved and processed. This script measures the
gap between that claim and the trees on disk:

- bytes, citation-key directories and manifests per tree
- files sitting in a key directory that appear in NO manifest, and their bytes
  (the integrity hole: bytes nothing can verify and nothing can re-fetch)
- manifest records whose file is gone from disk
- per-file fill rate of every `ArtifactRecord` / `RetrievalRecord` /
  `ProcessingRecord` field, and per-manifest fill rate of every `Manifest` field
- keys where EVERY file record carries a retrieval record (a key that can be
  re-fetched end to end)
- bytes by role, the retrieval-method histogram, and the `retriever` strings that
  do not resolve through `RETRIEVERS` (a rebuild cannot dispatch those)
- `paper.md` files whose `ProcessingRecord` pins the MinerU version, command and
  DPI, which is the OCR a rerun would reproduce
- manifests carrying fields outside the pydantic model, file records still in the
  pre-nesting retrieval shape (`retrieval_method` beside `retrieval_command` on
  the record itself rather than a nested `RetrievalRecord`), and `role` strings
  outside the module's documented ROLE_* vocabulary

Every manifest is validated through `Manifest`, so a format drift fails the run
instead of being counted as a silent miss. Nothing is written into either mirror.

Run from the repo root:
  python experiments/database/scripts/audit_provenance_mirror.py
  python experiments/database/scripts/audit_provenance_mirror.py --json
  python experiments/database/scripts/audit_provenance_mirror.py --data-root /some/root

Writes:
  experiments/database/results/provenance_audit.json
  notes-tex/database/database-expansion-hosts/tables/mirror.tex
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, computed_field

from torchcell.literature import manifest as manifest_module
from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    MIRROR_TREES,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalRecord,
    scan_mirror,
)
from torchcell.literature.provenance import RETRIEVERS

load_dotenv()

# `FileRecord.role` is a free string, so a manifest can carry a role no reader
# dispatches on. The documented vocabulary is the module's own ROLE_* constants
# plus the "other" fallback `_role_for` assigns, read by reflection so this audit
# never re-declares the schema.
DOCUMENTED_ROLES = frozenset(
    {v for k, v in vars(manifest_module).items() if k.startswith("ROLE_")} | {"other"}
)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results" / "provenance_audit.json"
TABLE = (
    REPO
    / "notes-tex"
    / "database"
    / "database-expansion-hosts"
    / "tables"
    / "mirror.tex"
)

# Keys a FileRecord carried before retrieval provenance was nested into a
# RetrievalRecord. A record holding any of these is in the pre-nesting shape, and
# pydantic drops them on load, so the information is invisible to every reader.
PRE_NESTING_FILE_KEYS = ("retrieval_method", "retrieval_command", "retrieved_at")

# A ProcessingRecord reproduces an OCR run only when it names the tool version and
# the exact invocation: `ocr.ocr_pdf` records MinerU's argv under `command` and the
# rasterization DPI under `dpi`.
MINERU_TOOL = "mineru"
MINERU_REQUIRED_PARAMS = ("command", "dpi")

# The OCR'd article markdown, one per captured paper.
PAPER_OCR_PATH = "paper.md"


def _is_filled(value: Any) -> bool:
    """Whether a field carries information.

    None is empty, an empty string / list / dict is empty, and ``False`` on a bool
    is empty: ``provenance_complete=False`` is the manifest saying its own
    provenance chain is unknown, so counting it as filled would read as the
    opposite of what it means. ``0`` on an int is a real value (a zero-byte file).
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str | list | tuple | dict | set):
        return len(value) > 0
    return True


class FillRate(BaseModel):
    """How many records of a population carry a given field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Dotted field name, e.g. retrieval.last_check.")
    n_filled: int
    n_total: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pct(self) -> float:
        """Percentage filled, 0.0 over an empty population."""
        return 100.0 * self.n_filled / self.n_total if self.n_total else 0.0


class MirrorAudit(BaseModel):
    """What one mirror tree holds, measured from the tree and its manifests."""

    model_config = ConfigDict(extra="forbid")

    tree: str
    root: str
    bytes_total: int = Field(
        description="Sum of file sizes over the whole tree, service dirs included."
    )
    bytes_in_key_dirs: int = Field(
        description="Sum of file sizes inside citation-key directories only."
    )
    n_key_dirs: int = Field(
        description="Citation-key directories, excluding the _-prefixed service dirs."
    )
    n_with_manifest: int
    n_file_records: int
    n_files_on_disk: int = Field(
        description="Files inside key dirs, manifest.json itself excluded."
    )
    n_unmanifested: int
    bytes_unmanifested: int
    pct_bytes_unmanifested: float
    unmanifested_by_name: dict[str, int] = Field(
        description="Basename (or top dir / basename) -> count, for the files in "
        "no manifest, so the bookkeeping sidecars are separable from real data."
    )
    n_records_missing_on_disk: int
    records_missing_on_disk: tuple[str, ...]
    fills: dict[str, FillRate]
    bytes_by_role: dict[str, int]
    undocumented_roles: dict[str, int] = Field(
        description="Role string -> file records, for roles outside the module's "
        "documented ROLE_* vocabulary."
    )
    n_paper_ocr: int
    n_reproducible_ocr: int
    retrieval_methods: dict[str, int]
    unresolvable_retrievers: dict[str, int]
    n_unresolvable_retriever_records: int
    n_unresolvable_retriever_strings: int
    manifests_with_extra_fields: dict[str, tuple[str, ...]]
    n_prenesting_retrieval_records: int
    keys_with_prenesting_retrieval: tuple[str, ...]


class ProvenanceAuditReport(BaseModel):
    """Both mirrors, measured in one pass."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    data_root: str
    script: str
    n_retrievers_registered: int
    mirrors: tuple[MirrorAudit, ...]


def _walk_key_dir(key_dir: Path) -> dict[str, int]:
    """Relative path -> byte size for every file in a key dir, bar manifest.json."""
    found: dict[str, int] = {}
    for dirpath, _, filenames in os.walk(key_dir):
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = str(path.relative_to(key_dir))
            if rel == MANIFEST_FILENAME:
                continue
            found[rel] = path.stat().st_size
    return found


def _tree_bytes(root: Path) -> int:
    """Sum of file sizes over a whole mirror tree, service dirs included."""
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            total += (Path(dirpath) / filename).stat().st_size
    return total


def _unmanifested_label(rel: str) -> str:
    """Group an unmanifested file by name, keeping one level of directory."""
    parts = rel.split("/")
    return parts[-1] if len(parts) == 1 else f"{parts[0]}/{parts[-1]}"


def _reproducible_ocr(processing: ProcessingRecord | None) -> bool:
    """Whether a ProcessingRecord pins the MinerU version, command and DPI."""
    if processing is None or processing.tool != MINERU_TOOL:
        return False
    if not _is_filled(processing.version):
        return False
    return all(_is_filled(processing.params.get(key)) for key in MINERU_REQUIRED_PARAMS)


def _fill_rates(
    manifests: dict[str, Manifest], records: list[ArtifactRecord]
) -> dict[str, FillRate]:
    """Fill rate of every model field, over the population that field lives in."""
    fills: dict[str, FillRate] = {}
    retrievals = [r.retrieval for r in records if r.retrieval is not None]
    processings = [r.processing for r in records if r.processing is not None]

    for name in ArtifactRecord.model_fields:
        n = sum(1 for r in records if _is_filled(getattr(r, name)))
        fills[f"file.{name}"] = FillRate(
            name=f"file.{name}", n_filled=n, n_total=len(records)
        )
    for name in RetrievalRecord.model_fields:
        n = sum(1 for r in retrievals if _is_filled(getattr(r, name)))
        fills[f"retrieval.{name}"] = FillRate(
            name=f"retrieval.{name}", n_filled=n, n_total=len(retrievals)
        )
    for name in ProcessingRecord.model_fields:
        n = sum(1 for p in processings if _is_filled(getattr(p, name)))
        fills[f"processing.{name}"] = FillRate(
            name=f"processing.{name}", n_filled=n, n_total=len(processings)
        )
    for name in Manifest.model_fields:
        n = sum(1 for m in manifests.values() if _is_filled(getattr(m, name)))
        fills[f"manifest.{name}"] = FillRate(
            name=f"manifest.{name}", n_filled=n, n_total=len(manifests)
        )
    return fills


def audit_mirror(root: Path, tree: str) -> MirrorAudit:
    """Measure one mirror tree. A manifest that fails validation fails the run."""
    scan = scan_mirror(root)
    if scan.failures:
        named = ", ".join(f.citation_key for f in scan.failures)
        raise RuntimeError(
            f"{tree}: {len(scan.failures)} manifest(s) do not validate as Manifest: "
            f"{named}. Fix the manifest or the model; the audit will not skip it."
        )

    key_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
    )
    records: list[ArtifactRecord] = []
    n_files_on_disk = 0
    bytes_in_key_dirs = 0
    n_unmanifested = 0
    bytes_unmanifested = 0
    unmanifested_by_name: Counter[str] = Counter()
    missing: list[str] = []
    n_keys_every_file_retrieved = 0
    bytes_by_role: Counter[str] = Counter()
    undocumented_roles: Counter[str] = Counter()
    n_paper_ocr = 0
    n_reproducible_ocr = 0
    methods: Counter[str] = Counter()
    unresolvable: Counter[str] = Counter()
    extra_fields: dict[str, tuple[str, ...]] = {}
    prenesting_keys: list[str] = []
    n_prenesting = 0

    for key_dir in key_dirs:
        key = key_dir.name
        on_disk = _walk_key_dir(key_dir)
        n_files_on_disk += len(on_disk)
        bytes_in_key_dirs += sum(on_disk.values())
        manifest = scan.manifests.get(key)
        listed = {r.path for r in manifest.files} if manifest is not None else set()
        for rel, size in on_disk.items():
            if rel in listed:
                continue
            n_unmanifested += 1
            bytes_unmanifested += size
            unmanifested_by_name[_unmanifested_label(rel)] += 1
        if manifest is None:
            continue

        records.extend(manifest.files)
        if manifest.files and all(r.retrieval is not None for r in manifest.files):
            n_keys_every_file_retrieved += 1
        for record in manifest.files:
            if record.path not in on_disk:
                missing.append(f"{key}/{record.path}")
            bytes_by_role[record.role] += record.bytes
            if record.role not in DOCUMENTED_ROLES:
                undocumented_roles[record.role] += 1
            if record.path == PAPER_OCR_PATH:
                n_paper_ocr += 1
                if _reproducible_ocr(record.processing):
                    n_reproducible_ocr += 1
            if record.retrieval is not None:
                methods[record.retrieval.method.value] += 1
                if record.retrieval.retriever not in RETRIEVERS:
                    unresolvable[record.retrieval.retriever] += 1

        # The raw JSON, for what pydantic drops on load: fields the model does not
        # declare, and the retrieval shape that predates RetrievalRecord.
        raw = json.loads((key_dir / MANIFEST_FILENAME).read_text())
        outside = tuple(sorted(set(raw) - set(Manifest.model_fields)))
        if outside:
            extra_fields[key] = outside
        n_pre = sum(
            1
            for f in raw.get("files", [])
            if any(k in f for k in PRE_NESTING_FILE_KEYS)
        )
        if n_pre:
            n_prenesting += n_pre
            prenesting_keys.append(key)

    fills = _fill_rates(scan.manifests, records)
    fills["keys.with_manifest"] = FillRate(
        name="keys.with_manifest", n_filled=len(scan.manifests), n_total=len(key_dirs)
    )
    fills["keys.every_file_retrieved"] = FillRate(
        name="keys.every_file_retrieved",
        n_filled=n_keys_every_file_retrieved,
        n_total=len(scan.manifests),
    )
    fills["files.in_a_manifest"] = FillRate(
        name="files.in_a_manifest",
        n_filled=n_files_on_disk - n_unmanifested,
        n_total=n_files_on_disk,
    )
    fills["paper_ocr.reproducible"] = FillRate(
        name="paper_ocr.reproducible", n_filled=n_reproducible_ocr, n_total=n_paper_ocr
    )
    fills["retriever.resolves"] = FillRate(
        name="retriever.resolves",
        n_filled=sum(methods.values()) - sum(unresolvable.values()),
        n_total=sum(methods.values()),
    )

    return MirrorAudit(
        tree=tree,
        root=str(root),
        bytes_total=_tree_bytes(root),
        bytes_in_key_dirs=bytes_in_key_dirs,
        n_key_dirs=len(key_dirs),
        n_with_manifest=len(scan.manifests),
        n_file_records=len(records),
        n_files_on_disk=n_files_on_disk,
        n_unmanifested=n_unmanifested,
        bytes_unmanifested=bytes_unmanifested,
        pct_bytes_unmanifested=(
            100.0 * bytes_unmanifested / bytes_in_key_dirs if bytes_in_key_dirs else 0.0
        ),
        unmanifested_by_name=dict(unmanifested_by_name.most_common()),
        n_records_missing_on_disk=len(missing),
        records_missing_on_disk=tuple(sorted(missing)),
        fills=fills,
        bytes_by_role=dict(bytes_by_role.most_common()),
        undocumented_roles=dict(undocumented_roles.most_common()),
        n_paper_ocr=n_paper_ocr,
        n_reproducible_ocr=n_reproducible_ocr,
        retrieval_methods=dict(methods.most_common()),
        unresolvable_retrievers=dict(unresolvable.most_common()),
        n_unresolvable_retriever_records=sum(unresolvable.values()),
        n_unresolvable_retriever_strings=len(unresolvable),
        manifests_with_extra_fields=extra_fields,
        n_prenesting_retrieval_records=n_prenesting,
        keys_with_prenesting_retrieval=tuple(sorted(prenesting_keys)),
    )


def build_report(data_root: Path) -> ProvenanceAuditReport:
    """Audit both mirror trees under ``data_root``."""
    mirrors = []
    for tree in MIRROR_TREES:
        root = data_root / tree
        if not root.is_dir():
            raise FileNotFoundError(f"{tree}: no such directory {root}")
        mirrors.append(audit_mirror(root, tree))
    return ProvenanceAuditReport(
        generated_at=datetime.now(UTC).isoformat(),
        data_root=str(data_root),
        script="experiments/database/scripts/audit_provenance_mirror.py",
        n_retrievers_registered=len(RETRIEVERS),
        mirrors=tuple(mirrors),
    )


def _thousands(n: int) -> str:
    """12345 -> 12,345."""
    return f"{n:,}"


def _gb(n: int) -> str:
    """Bytes as GB (10^9), two decimals."""
    return f"{n / 1e9:.2f} GB"


def _keys(n: int) -> str:
    """``1 key`` / ``4 keys``."""
    return f"{n} key" if n == 1 else f"{n} keys"


def _records(n: int) -> str:
    """``1 record`` / ``59 records``."""
    return f"{n} record" if n == 1 else f"{_thousands(n)} records"


def _plain(text: str) -> str:
    """Strip the LaTeX markup from a cell or label, for the terminal summary."""
    return re.sub(r"\\[a-z]+\{([^}]*)\}", r"\1", text).replace("\\", "")


def _fill_cell(mirror: MirrorAudit, name: str) -> str:
    r"""``12.3\% (45/366)``, the fill rate as a percentage and a count."""
    fill = mirror.fills[name]
    return (
        f"{fill.pct:.1f}\\% ({_thousands(fill.n_filled)}/{_thousands(fill.n_total)})"
        if fill.n_total
        else "no records"
    )


# One row per measured property, worst first: what a rebuild loses, then what a
# re-check loses, then the manifest metadata, then the shape drift, then the facts
# of the tree. ``fill`` rows take a key of ``MirrorAudit.fills``; ``value`` rows
# take a callable.
_FILL_ROWS: tuple[tuple[str, str], ...] = (
    ("File records with a retrieval record", "file.retrieval"),
    ("Keys where every file record has one", "keys.every_file_retrieved"),
    (
        "Retrieval source ever re-checked (\\texttt{last\\_check})",
        "retrieval.last_check",
    ),
    ("File records with a processing record", "file.processing"),
    ("\\texttt{paper.md} with reproducible MinerU OCR", "paper_ocr.reproducible"),
    ("Files on disk named in a manifest", "files.in_a_manifest"),
    (
        "Manifest \\texttt{si\\_expected} (SI completeness checklist)",
        "manifest.si_expected",
    ),
    ("Manifest \\texttt{si\\_data\\_sources}", "manifest.si_data_sources"),
    ("Manifest \\texttt{zotero\\_item\\_key}", "manifest.zotero_item_key"),
    ("Manifest \\texttt{collections}", "manifest.collections"),
    ("Manifest \\texttt{doi}", "manifest.doi"),
    ("Manifest \\texttt{provenance\\_complete} true", "manifest.provenance_complete"),
    ("File record \\texttt{original\\_filename}", "file.original_filename"),
    ("File record \\texttt{zotero\\_md5}", "file.zotero_md5"),
    ("File record \\texttt{source}", "file.source"),
    ("File record \\texttt{sha256}", "file.sha256"),
    (
        "Retrieval \\texttt{retriever} resolves in \\texttt{RETRIEVERS}",
        "retriever.resolves",
    ),
    ("Retrieval \\texttt{source\\_url}", "retrieval.source_url"),
)

_VALUE_ROWS: tuple[tuple[str, Any], ...] = (
    (
        "Files in no manifest",
        lambda m: (
            f"{_thousands(m.n_unmanifested)} files, {_gb(m.bytes_unmanifested)} "
            f"({m.pct_bytes_unmanifested:.1f}\\% of tree)"
        ),
    ),
    (
        "Manifest records whose file is gone",
        lambda m: _thousands(m.n_records_missing_on_disk),
    ),
    (
        "Unresolvable \\texttt{retriever} strings",
        lambda m: (
            f"{m.n_unresolvable_retriever_strings} distinct, "
            f"{_records(m.n_unresolvable_retriever_records)}"
        ),
    ),
    (
        "File records in the pre-nesting retrieval shape",
        lambda m: (
            f"{_records(m.n_prenesting_retrieval_records)} in "
            f"{_keys(len(m.keys_with_prenesting_retrieval))}"
        ),
    ),
    (
        "Manifests with fields outside the model",
        lambda m: _thousands(len(m.manifests_with_extra_fields)),
    ),
    (
        "File roles outside the documented vocabulary",
        lambda m: (
            f"{len(m.undocumented_roles)} distinct, "
            f"{_records(sum(m.undocumented_roles.values()))}"
        ),
    ),
    ("Retrieval methods in use", lambda m: str(len(m.retrieval_methods))),
)

_FACT_ROWS: tuple[tuple[str, Any], ...] = (
    ("Bytes on disk", lambda m: _gb(m.bytes_total)),
    ("Citation-key directories", lambda m: _thousands(m.n_key_dirs)),
    ("Keys carrying a manifest", lambda m: _fill_cell(m, "keys.with_manifest")),
    ("File records across all manifests", lambda m: _thousands(m.n_file_records)),
    ("Files on disk in key directories", lambda m: _thousands(m.n_files_on_disk)),
)

CAPTION = (
    "What the two provenance mirrors actually store, measured from the trees on "
    "disk and from every \\texttt{manifest.json} in them, validated through "
    "\\texttt{torchcell.literature.manifest.Manifest}. "
    "\\emph{Library} is \\texttt{\\$DATA\\_ROOT/torchcell-library/} (paper PDFs, "
    "OCR markdown, SI) and \\emph{Raw} is \\texttt{\\$DATA\\_ROOT/torchcell-raw/} "
    "(the raw dataset files a loader consumed). Each cell is the share of the "
    "population carrying the property, with the count and the population size. "
    "Rows are ordered by consequence, worst first: the first block is what a "
    "rebuild cannot reproduce, the second what a drift check cannot see, the "
    "third the manifest metadata, the fourth shape drift, the last the size of "
    "each tree. \\emph{Files in no manifest} is the integrity hole: bytes no "
    "sha256 pins and no recorded command re-fetches. Byte totals sum file sizes "
    "and exclude directory entries, so they read a little under \\texttt{du}. "
    "\\texttt{provenance\\_complete} counts manifests asserting a complete chain, "
    "so a false value counts as unfilled."
)


def render_table(report: ProvenanceAuditReport) -> str:
    """Render the two-column property table, library against raw."""
    library, raw = report.mirrors
    lines = [
        "%% GENERATED FILE -- do not hand-edit.",
        "%% SOURCE: experiments/database/scripts/audit_provenance_mirror.py",
        "\\begin{table}[H]\\centering",
        "\\footnotesize",
        f"\\caption[]{{{CAPTION}}}",
        "\\label{tab:mirror}",
        "\\begin{tabular}{@{}L{74mm} L{48mm} L{48mm}@{}}",
        "\\toprule",
        "Property & Library & Raw \\\\",
        "\\midrule",
    ]
    for label, key in _FILL_ROWS:
        lines.append(
            f"{label} & {_fill_cell(library, key)} & {_fill_cell(raw, key)} \\\\"
        )
    for block in (_VALUE_ROWS, _FACT_ROWS):
        lines.append("\\midrule")
        for label, render in block:
            lines.append(f"{label} & {render(library)} & {render(raw)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _print_summary(report: ProvenanceAuditReport) -> None:
    """Print the table's rows as plain text, so a run is readable in a terminal."""
    library, raw = report.mirrors
    width = max(
        len(_plain(label)) for label, _ in _FILL_ROWS + _VALUE_ROWS + _FACT_ROWS
    )
    print(f"{'property'.ljust(width)}  {'library':<46}  raw")
    for label, key in _FILL_ROWS:
        left, right = _fill_cell(library, key), _fill_cell(raw, key)
        print(f"{_plain(label).ljust(width)}  {_plain(left):<46}  {_plain(right)}")
    for block in (_VALUE_ROWS, _FACT_ROWS):
        print("-" * (width + 50))
        for label, render in block:
            left, right = str(render(library)), str(render(raw))
            print(f"{_plain(label).ljust(width)}  {_plain(left):<46}  {_plain(right)}")


def main(argv: list[str] | None = None) -> int:
    """Audit both mirrors, write the JSON result and the LaTeX table."""
    parser = argparse.ArgumentParser(
        prog="audit_provenance_mirror",
        description="Measure what the torchcell provenance mirrors store.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override DATA_ROOT (the parent of torchcell-library / torchcell-raw).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full report as JSON."
    )
    args = parser.parse_args(argv)

    data_root = args.data_root or Path(os.environ["DATA_ROOT"])
    report = build_report(data_root)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(report.model_dump_json(indent=2))
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(render_table(report))

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_summary(report)
    print(f"\nwrote {RESULTS}")
    print(f"wrote {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
