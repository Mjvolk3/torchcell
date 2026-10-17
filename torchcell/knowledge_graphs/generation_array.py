# torchcell/knowledge_graphs/generation_array.py
# [[torchcell.knowledge_graphs.generation_array]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/generation_array.py
# Test file: tests/torchcell/knowledge_graphs/test_generation_array.py
"""The per-adapter KG generation array: order, tiers, the DONE protocol, the merge.

The serial generation (``create_scerevisiae_kg_small``, one container, one loop over
every adapter, no checkpoint) restarts from adapter 1 when adapter 124 fails. This
module is the other shape: one slurm array task per adapter, each in its own container,
each writing into ``<out>/adapters/<iii>_<Adapter>/`` and ending with a ``DONE`` marker
(``torchcell.knowledge_graphs.generate_adapter``). Recovery is then "re-run the tasks
without a marker": ``missing`` prints them and the exact ``sbatch --array=`` per tier.

* **Order.** Index ``i`` is position ``i`` of ``build_adapter_map(include_private,
  organisms)`` (the public map, then the private map, filtered to the organism set),
  the same order the serial build walks. ``adapter_order`` is the one place that maps
  an array index to a dataset and an adapter.
* **Tiers.** ``conf/generation_tiers.yaml`` keys adapter names to a tier (cpus, memory,
  hours, concurrency). A slurm array has one resource spec, so the conductor submits
  one array per tier with that tier's ids.
* **Merge.** ``merge_outputs`` requires every directory in the order to carry its
  marker, checks each label's header is byte-identical across the directories that
  carry it, runs the duplicate check, and writes ONE ``neo4j-admin-import-call.sh`` at
  ``<out>/``: per label the first directory's header and the comma-joined part files
  of every directory carrying it. The command prefix (delimiter, quote,
  ``--skip-duplicate-nodes=true``, ``--skip-bad-relationships``) is transplanted from
  a per-directory call BioCypher wrote, so the flags are the BioCypher config's, and
  ``--skip-duplicate-nodes=true`` is verified to be among them. The hash pass measures
  the longest row of every label, and the call carries
  ``--read-buffer-size=<bytes>`` sized from it (``read_buffer_bytes``): neo4j's CSV
  reader needs a newline inside each read buffer, so a row longer than its 4 MiB
  default fails the whole import before the first node is written ("Weird input
  data, no newline character in the whole buffer 4194304", job 5581 on 2026-10-17,
  an interned Peter 2018 pan-genome genotype of 5.4 MB).
* **Duplicates.** BioCypher (and ``FastCsvSink``) deduplicate within ONE process;
  across directories content-id nodes recur (genome, environment, media, perturbation,
  genotype, experiment reference). A key that recurs across directories, node id or
  edge ``(start, end)``, keeps the copy of the lowest index and the later directories'
  rows are dropped into a rewritten part file under ``<out>/merged/<dir>/``, which the
  import call references; the check proves the dropped rows are byte-identical to the
  kept ones (a 128-bit key hash and a 64-bit row hash per row, sort-merged per label
  with numpy). Node rows are dropped too, not left to ``--skip-duplicate-nodes=true``:
  neo4j counts every skipped duplicate as a bad entry against ``--bad-tolerance``
  (default 1000) and aborts the import past it ("1063 bad entries were found, which
  exceeds the set fault tolerance", job 5590 on 2026-10-17, against 22 M duplicate
  Genotype rows), so the flag stays only as a guard for a key an adapter repeated
  inside ONE directory (``repeated_in_directory_rows``, kept as the adapter wrote
  them, expected zero). A duplicate that is not byte-identical is read back
  and reconciled against the identity design of ``torchcell.datamodels.identity``: a
  content id covers composition, value and typed unit, never who stated it, so two
  rows of one id may differ in the NON-IDENTITY fields (``NON_IDENTITY_COLUMNS``, the
  plain ``name``, ``compound_name`` and ``preferred_id`` columns, and
  ``NON_IDENTITY_JSON_KEYS`` inside a
  JSON column: ``provenance``, ``provenance_gaps``, ``note``, ``defers_to``, ``name``).
  Such a key is a *variant*: the lowest-index directory's row is the one imported (the
  first row neo4j sees, since the call lists directories in index order), the merge
  report names every variant key with its directories and the differing paths, and
  the build manifest carries the per-label counts. Measured on the KG 4.0 generation
  (2026-10-16): 116 ``Environment`` ids shared by two to six datasets differed only in
  per-source ``provenance_gaps`` notes, ``media.provenance`` quotes or a free-text
  ``media.name``, 5 ``EnvironmentPerturbation`` ids only in ``compound_name``, and 1
  ``FitnessPhenotype`` id only in ``preferred_id``.
  A difference anywhere else (a value, a unit, a composition) refuses the merge and
  names the label, the key, the directories and the paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator

from torchcell.data.experiment_dataset import (
    ALL_ORGANISMS,
    Organism,
    format_organisms,
    parse_organisms,
)

log = logging.getLogger(__name__)

DEFAULT_TIER_TABLE = Path(__file__).parent / "conf" / "generation_tiers.yaml"
DONE_FILENAME = "DONE"
ADAPTERS_DIRNAME = "adapters"
MERGED_DIRNAME = "merged"
FENCE_FILENAME = "dev_fence.txt"
MERGE_REPORT_FILENAME = "merge_report.json"
IMPORT_CALL_FILENAME = "neo4j-admin-import-call.sh"
DEFAULT_ARRAY_SCRIPT = (
    "database/slurm/scripts/gilahyper_generate_adapter_array-slurm_docker.slurm"
)
_HEADER_RE = re.compile(r"^(?P<label>[A-Za-z0-9]+)-header\.csv$")
_PART_RE = re.compile(r"^(?P<label>[A-Za-z0-9]+)-part\d+\.csv$")
_NODES_ARG_RE = re.compile(r'--nodes="([^"]+)" ')
_RELS_ARG_RE = re.compile(r'--relationships="([^"]+)" ')
# One argument of the import call must stay under the kernel's single-argument limit
# (MAX_ARG_STRLEN, 128 KiB); a label carried by every directory is one argument.
MAX_ARG_BYTES = 120 * 1024
_ROW_DTYPE = np.dtype([("k0", "<u8"), ("k1", "<u8"), ("row", "<u8")])
#: neo4j-admin import's default ``--read-buffer-size``; a row must fit in one buffer
#: with a newline, so the merged call raises it when any row comes near.
NEO4J_DEFAULT_READ_BUFFER_BYTES = 4 * 2**20
_READ_BUFFER_RE = re.compile(r"--read-buffer-size=\S+ ")


def read_buffer_bytes(longest_row_bytes: int) -> int:
    """The ``--read-buffer-size`` for a build whose longest row is this many bytes.

    neo4j's default when every row is under a quarter of it; otherwise twice the longest
    row, rounded up to a whole MiB, so a row plus its newline never spans two buffers.
    """
    needed = 2 * longest_row_bytes
    if needed <= NEO4J_DEFAULT_READ_BUFFER_BYTES:
        return NEO4J_DEFAULT_READ_BUFFER_BYTES
    return -(-needed // 2**20) * 2**20


# ----------------------------------------------------------------------------- tiers


class GenerationTier(BaseModel):
    """One resource tier: the slurm reservation and the container limits of a task."""

    name: str
    cpus: int = Field(gt=0)
    mem_gb: int = Field(gt=0)
    hours: int = Field(gt=0)
    concurrency: int = Field(gt=0, description="the %N cap of the tier's array")

    @property
    def time_limit(self) -> str:
        """The slurm ``--time`` of the tier, ``HH:00:00``."""
        return f"{self.hours:02d}:00:00"


class TierTable(BaseModel):
    """``generation_tiers.yaml``: tiers by name, a default, and adapter overrides."""

    default_tier: str
    tiers: dict[str, GenerationTier]
    adapters: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _names_resolve(self) -> TierTable:
        if self.default_tier not in self.tiers:
            raise ValueError(f"default_tier {self.default_tier!r} is not a tier")
        unknown = {a: t for a, t in self.adapters.items() if t not in self.tiers}
        if unknown:
            raise ValueError(f"adapters name tiers that do not exist: {unknown}")
        for name, tier in self.tiers.items():
            if tier.name != name:
                raise ValueError(f"tier {name!r} carries name {tier.name!r}")
        return self

    def tier_for(self, adapter_name: str) -> GenerationTier:
        """The tier of an adapter class name (the default when it is not listed)."""
        return self.tiers[self.adapters.get(adapter_name, self.default_tier)]


def load_tier_table(path: Path = DEFAULT_TIER_TABLE) -> TierTable:
    """Read the tier yaml; each tier's ``name`` is its key."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tiers = {
        str(name): GenerationTier(name=str(name), **spec)
        for name, spec in raw["tiers"].items()
    }
    return TierTable(
        default_tier=raw["default_tier"],
        tiers=tiers,
        adapters=raw.get("adapters") or {},
    )


# ----------------------------------------------------------------------------- order


class AdapterSlot(BaseModel):
    """One array task: its index in the build order, its classes, its directory."""

    index: int = Field(ge=0)
    dataset_class: str
    adapter_class: str

    @property
    def dirname(self) -> str:
        """``<iii>_<Adapter>``, the directory under ``<out>/adapters/``."""
        return f"{self.index:03d}_{self.adapter_class}"


def adapter_order(
    include_private: bool = False, organisms: frozenset[Organism] = ALL_ORGANISMS
) -> list[AdapterSlot]:
    """The build order: ``build_adapter_map`` positions, index 0 first.

    Imported lazily because the map imports every loader and adapter (seconds), which
    the tier and merge code paths do not need.
    """
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

    return [
        AdapterSlot(
            index=i, dataset_class=dataset.__name__, adapter_class=adapter.__name__
        )
        for i, (dataset, adapter) in enumerate(
            build_adapter_map(
                include_private=include_private, organisms=organisms
            ).items()
        )
    ]


def slot_by_index(order: Sequence[AdapterSlot], index: int) -> AdapterSlot:
    """The slot at ``index``; an index outside the order is an error, never a wrap."""
    if not 0 <= index < len(order):
        raise IndexError(
            f"adapter index {index} is outside the build order of {len(order)} adapters"
        )
    slot = order[index]
    if slot.index != index:
        raise AssertionError(f"order position {index} holds index {slot.index}")
    return slot


def write_order(order: Sequence[AdapterSlot], path: Path) -> None:
    """Record the order beside the outputs so a merge reads the list the array ran."""
    path.write_text(
        json.dumps([s.model_dump() for s in order], indent=1) + "\n", encoding="utf-8"
    )


def read_order(path: Path) -> list[AdapterSlot]:
    """Read an order written by :func:`write_order`."""
    return [AdapterSlot(**s) for s in json.loads(path.read_text(encoding="utf-8"))]


# ----------------------------------------------------------------------------- fence


def dev_fence_lines(
    dev_data_root: str, include_private: bool, organisms: frozenset[Organism]
) -> list[str]:
    """``<Class> <sha256 of its build manifest>``, sorted by class, one per mapped store.

    The same lines the live rebuild's ``<job>_dev_fence.py`` prints: the conductor
    writes them once before the array, every task re-digests and diffs before it
    generates, and the merge re-digests before the import.
    """
    from torchcell.database.build_dataset_lmdb import dataset_default_root
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

    lines = []
    for cls in sorted(
        build_adapter_map(include_private=include_private, organisms=organisms),
        key=lambda c: c.__name__,
    ):
        manifest = (
            Path(dev_data_root)
            / dataset_default_root(cls)
            / "preprocess"
            / "build_manifest.json"
        )
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        lines.append(f"{cls.__name__} {digest}")
    return lines


# ------------------------------------------------------------------------------ DONE


class DoneMarker(BaseModel):
    """What a task writes, last, into its directory: the proof its output is complete."""

    index: int
    adapter_class: str
    dataset_class: str
    dataset_len: int
    n_nodes: int
    n_edges: int
    generation_wall_s: float
    peak_memory_gb: float
    torchcell_version: str
    build_commit: str = Field(description="the commit the container installed")
    written_at: str


def done_path(out_dir: Path, slot: AdapterSlot) -> Path:
    """``<out>/adapters/<dir>/DONE``."""
    return out_dir / ADAPTERS_DIRNAME / slot.dirname / DONE_FILENAME


def read_done(out_dir: Path, slot: AdapterSlot) -> DoneMarker | None:
    """The slot's marker, or None when it is absent or names another slot."""
    path = done_path(out_dir, slot)
    if not path.is_file():
        return None
    marker = DoneMarker.model_validate_json(path.read_text(encoding="utf-8"))
    if (marker.index, marker.adapter_class) != (slot.index, slot.adapter_class):
        raise ValueError(
            f"{path} names index {marker.index} {marker.adapter_class}, "
            f"expected {slot.index} {slot.adapter_class}"
        )
    return marker


def missing_slots(out_dir: Path, order: Sequence[AdapterSlot]) -> list[AdapterSlot]:
    """The slots of the order whose directory carries no DONE marker."""
    return [slot for slot in order if read_done(out_dir, slot) is None]


# --------------------------------------------------------------------------- sbatch


def compress_ids(ids: Iterable[int]) -> str:
    """``[0, 1, 2, 5, 7, 8]`` -> ``"0-2,5,7-8"``, the slurm ``--array`` spelling."""
    sorted_ids = sorted(set(ids))
    if not sorted_ids:
        raise ValueError("no ids to compress")
    runs: list[tuple[int, int]] = []
    for i in sorted_ids:
        if runs and i == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


class TierSubmission(BaseModel):
    """One ``sbatch`` of one tier: the ids it covers and the command line."""

    tier: str
    indices: list[int]
    environment: dict[str, str]
    argv: list[str]

    def shell(self) -> str:
        """The command as one shell line (environment assignments first)."""
        env = " ".join(
            f"{k}={_shell_quote(v)}" for k, v in sorted(self.environment.items())
        )
        return f"{env} {' '.join(_shell_quote(a) for a in self.argv)}".strip()


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=,%+-]+", value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def tier_submissions(
    order: Sequence[AdapterSlot],
    tiers: TierTable,
    *,
    script: str,
    out_name: str,
    environment: dict[str, str],
    indices: Iterable[int] | None = None,
    job_name_prefix: str = "gen_adapter",
    dependency: str | None = None,
) -> list[TierSubmission]:
    """One submission per tier that has at least one of ``indices`` (all when None).

    Multi-member values (``KG_ORGANISMS=a,b``, hydra overrides) travel in the
    environment with ``--export=ALL``, never inside ``--export``'s own list, which
    sbatch splits on commas (job 4899 built yeast-only that way). ``dependency`` is
    passed as ``--dependency=`` (``afterany:4919``): the ``SBATCH_DEPENDENCY``
    environment variable was not honored by slurm 24.05 on 2026-10-15 (jobs 4920 and
    4921 were submitted with it and carried no dependency).
    """
    known = {slot.index for slot in order}
    wanted = known if indices is None else set(indices)
    outside = sorted(wanted - known)
    if outside:
        raise IndexError(f"indices outside the build order of {len(order)}: {outside}")
    by_tier: dict[str, list[int]] = {}
    for slot in order:
        if slot.index in wanted:
            by_tier.setdefault(tiers.tier_for(slot.adapter_class).name, []).append(
                slot.index
            )
    submissions = []
    for name in tiers.tiers:  # yaml order, so the big tier is first in the listing
        if name not in by_tier:
            continue
        tier = tiers.tiers[name]
        ids = by_tier[name]
        env = {
            **environment,
            "OUT_NAME": out_name,
            "GEN_TIER": name,
            "GEN_CPUS": str(tier.cpus),
            "GEN_MEM_GB": str(tier.mem_gb),
        }
        argv = [
            "sbatch",
            f"--job-name={job_name_prefix}_{name}",
            f"--array={compress_ids(ids)}%{tier.concurrency}",
            f"--cpus-per-task={tier.cpus}",
            f"--mem={tier.mem_gb}G",
            f"--time={tier.time_limit}",
            *([f"--dependency={dependency}"] if dependency else []),
            "--export=ALL",
            script,
        ]
        submissions.append(
            TierSubmission(tier=name, indices=ids, environment=env, argv=argv)
        )
    return submissions


def recovery_commands(
    missing: Sequence[AdapterSlot],
    tiers: TierTable,
    *,
    script: str,
    out_name: str,
    environment: dict[str, str],
) -> list[str]:
    """The exact sbatch lines that re-run the missing slots, one per tier."""
    order = list(missing)
    return [
        s.shell()
        for s in tier_submissions(
            order,
            tiers,
            script=script,
            out_name=out_name,
            environment=environment,
            indices=[s.index for s in order],
        )
    ]


def submit(submissions: Sequence[TierSubmission], *, dry_run: bool) -> list[int]:
    """Run each submission's sbatch (or print it); return the array job ids."""
    job_ids: list[int] = []
    for submission in submissions:
        line = submission.shell()
        if dry_run:
            print(f"DRY-RUN {line}")
            continue
        print(line)
        result = subprocess.run(
            submission.argv,
            env={**os.environ, **submission.environment},
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match is None:
            raise RuntimeError(f"sbatch printed no job id: {result.stdout!r}")
        job_ids.append(int(match.group(1)))
        print(
            f"submitted {submission.tier} ({len(submission.indices)} tasks): job {job_ids[-1]}"
        )
    return job_ids


# ---------------------------------------------------------------------------- merge


class LabelFiles(BaseModel):
    """One label's header and part files across the directories that carry it."""

    label: str
    kind: Literal["node", "edge"]
    columns: list[str]
    header: str = Field(description="host path of the first directory's header")
    parts: list[tuple[int, str]] = Field(
        description="(adapter index, host path of a part file), in build order"
    )


def discover_label_files(
    out_dir: Path, order: Sequence[AdapterSlot]
) -> dict[str, LabelFiles]:
    """Every label of every adapter directory, headers checked byte-identical.

    A directory's label is its ``<Label>-header.csv`` plus its ``<Label>-part*.csv``
    files; a header without parts or parts without a header is an error, as
    ``incremental_import.discover_csv_groups`` treats it.
    """
    labels: dict[str, LabelFiles] = {}
    header_bytes: dict[str, bytes] = {}
    for slot in order:
        directory = out_dir / ADAPTERS_DIRNAME / slot.dirname
        headers: dict[str, Path] = {}
        parts: dict[str, list[Path]] = {}
        for path in sorted(directory.iterdir()):
            if (m := _HEADER_RE.match(path.name)) is not None:
                headers[m.group("label")] = path
            elif (m := _PART_RE.match(path.name)) is not None:
                parts.setdefault(m.group("label"), []).append(path)
        orphans = sorted(set(parts) - set(headers))
        if orphans:
            raise ValueError(f"{directory}: part files without a header: {orphans}")
        for label, header in sorted(headers.items()):
            if label not in parts:
                raise ValueError(f"{directory}: header without part files: {header}")
            raw = header.read_bytes()
            columns = raw.decode("utf-8").rstrip("\n").split("\t")
            if label in labels:
                if header_bytes[label] != raw:
                    raise ValueError(
                        f"{label}: header in {directory} differs from {labels[label].header}"
                    )
                labels[label].parts.extend((slot.index, str(p)) for p in parts[label])
                continue
            if columns[0] == ":ID":
                kind: Literal["node", "edge"] = "node"
            elif columns[0] == ":START_ID":
                kind = "edge"
            else:
                raise ValueError(f"{header}: neither a node nor an edge header")
            header_bytes[label] = raw
            labels[label] = LabelFiles(
                label=label,
                kind=kind,
                columns=columns,
                header=str(header),
                parts=[(slot.index, str(p)) for p in parts[label]],
            )
    return labels


def _key_columns(files: LabelFiles) -> tuple[int, int | None]:
    """Column positions of the id (nodes) or of start and end (edges)."""
    if files.kind == "node":
        return files.columns.index(":ID"), None
    return files.columns.index(":START_ID"), files.columns.index(":END_ID")


def scan_part(
    path: str, n_columns: int, key_col: int, end_col: int | None
) -> tuple[NDArray[Any], int]:
    r"""Hash every row of a part file: (128-bit key, 64-bit row), one record per row,
    and the longest row in bytes, newline included.

    The key is the node id, or ``start\x00end`` for an edge. A row whose tab count is
    the header's has no tab inside a quoted value and is split directly; any other row
    is parsed with the csv module (quote ``'``), which is exact and slower.
    """
    import csv
    import io

    expected_tabs = n_columns - 1
    buf = bytearray()
    longest = 0
    with open(path, "rb", buffering=16 * 2**20) as handle:
        for line in handle:
            if len(line) > longest:
                longest = len(line)
            if line.count(b"\t") == expected_tabs:
                fields = line.rstrip(b"\n").split(b"\t")
            else:
                text = line.decode("utf-8")
                (parsed,) = csv.reader(
                    io.StringIO(text), delimiter="\t", quotechar="'", strict=True
                )
                fields = [f.encode("utf-8") for f in parsed]
            if end_col is None:
                key = fields[key_col]
            else:
                key = fields[key_col] + b"\x00" + fields[end_col]
            buf += hashlib.blake2b(key, digest_size=16).digest()
            buf += hashlib.blake2b(line, digest_size=8).digest()
    return np.frombuffer(bytes(buf), dtype=_ROW_DTYPE), longest


#: Plain columns a content-addressed node may carry differently per source: a label,
#: not identity (``media_identity`` drops ``name``; an environment perturbation is
#: addressed by its compound's InChIKey, not by the name the paper printed).
#: ``preferred_id`` is BioCypher's id-namespace hint, which the adapter writes as
#: ``phenotype_<id>`` on an experiment's phenotype and as the label name on a reference
#: phenotype, so one measured value that is also a reference (a fitness of exactly 1.0,
#: Kuzmin 2018 beside Kuzmin 2020 and Choe 2019) carries both.
NON_IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {"name", "compound_name", "preferred_id"}
)
#: Keys inside a JSON column under which two rows of one id may differ: who stated a
#: value and how (``provenance``, ``note``, ``defers_to``), which fields a source left
#: unreported (``provenance_gaps``), and a free-text name. Every path of a difference
#: must pass through one of them.
NON_IDENTITY_JSON_KEYS: frozenset[str] = frozenset(
    {"provenance", "provenance_gaps", "note", "defers_to", "name"}
)
#: More non-identical duplicate keys than this in one label is not a provenance
#: variant pattern but a broken identity, and the merge refuses without reading them.
MAX_VARIANT_KEYS = 100_000


class VariantSample(BaseModel):
    """One duplicate key whose rows differ only in non-identity fields."""

    key: str
    directories: list[str] = Field(
        description="adapter directory names carrying the key; the first is imported"
    )
    differences: list[str] = Field(
        description="column, or column followed by the JSON path, of each difference"
    )


class LabelDuplicates(BaseModel):
    """The cross-directory duplicate outcome of one label."""

    label: str
    kind: Literal["node", "edge"]
    rows: int
    duplicate_rows: int = Field(
        description=(
            "rows whose key already appeared in a lower-index directory; every one "
            "is dropped from a rewritten part"
        )
    )
    repeated_in_directory_rows: int = Field(
        default=0,
        description=(
            "rows whose key already appeared in the SAME directory; kept as the "
            "adapter wrote them, and for a node label each is one neo4j bad entry"
        ),
    )
    duplicate_keys: int
    conflicting_keys: int = Field(
        description="keys whose rows are not byte-identical across directories"
    )
    variant_keys: int = Field(
        default=0,
        description=(
            "of the conflicting keys, those whose rows differ only in non-identity "
            "fields; a merge that succeeded has variant_keys == conflicting_keys"
        ),
    )
    variants: list[VariantSample] = Field(
        default_factory=list, description="every variant key, located"
    )
    seconds: float = Field(default=0.0, description="wall time of the label's check")
    longest_row_bytes: int = Field(
        default=0, description="the longest row of the label, newline included"
    )
    drop_rows: dict[str, list[int]] = Field(
        default_factory=dict,
        description="part path -> row numbers (0-based) to drop from the rewrite",
    )


class ConflictSample(BaseModel):
    """One non-identical duplicate, located for the refusal message."""

    label: str
    key: str
    paths: list[str]
    differences: list[str] = Field(
        default_factory=list,
        description="the differing columns and JSON paths outside the non-identity set",
    )


def _scan_label(
    files: LabelFiles, workers: int
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], int]:
    """Rows of every part of a label: hashes, directory index, row number within the
    part, and the longest row in bytes.
    """
    key_col, end_col = _key_columns(files)
    n_columns = len(files.columns)
    parts = files.parts
    if workers > 1 and len(parts) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(parts))) as pool:
            scanned = list(
                pool.map(
                    scan_part,
                    [p for _, p in parts],
                    [n_columns] * len(parts),
                    [key_col] * len(parts),
                    [end_col] * len(parts),
                )
            )
    else:
        scanned = [scan_part(p, n_columns, key_col, end_col) for _, p in parts]
    arrays = [a for a, _ in scanned]
    longest = max((n for _, n in scanned), default=0)
    hashes = np.concatenate(arrays) if arrays else np.empty(0, dtype=_ROW_DTYPE)
    part_index = (
        np.concatenate(
            [np.full(len(a), i, dtype=np.int32) for i, a in enumerate(arrays)]
        )
        if arrays
        else np.empty(0, dtype=np.int32)
    )
    row_number = (
        np.concatenate([np.arange(len(a), dtype=np.int64) for a in arrays])
        if arrays
        else np.empty(0, dtype=np.int64)
    )
    return hashes, part_index, row_number, longest


def check_label_duplicates(
    files: LabelFiles, workers: int = 1
) -> tuple[LabelDuplicates, list[tuple[np.uint64, np.uint64]]]:
    """Sort-merge one label's rows on their key across directories.

    Returns the outcome and the key hashes of the conflicting keys (for the located
    sample). A key counts as a duplicate only when it appears in more than one
    directory; rows of a key inside one directory are what that adapter wrote and are
    kept as they are. Everything is vectorized per group with ``reduceat``: the cost
    is the sort, not the number of keys that recur (two copies of a 100 M-row label
    are 100 M duplicate keys).
    """
    hashes, part_index, row_number, longest = _scan_label(files, workers)
    n = len(hashes)
    outcome = LabelDuplicates(
        label=files.label,
        kind=files.kind,
        rows=n,
        duplicate_rows=0,
        duplicate_keys=0,
        conflicting_keys=0,
        longest_row_bytes=longest,
    )
    if n == 0:
        return outcome, []
    adapter_of_part = np.array([i for i, _ in files.parts], dtype=np.int64)
    order = np.lexsort((hashes["k1"], hashes["k0"]))
    k0, k1 = hashes["k0"][order], hashes["k1"][order]
    rows = hashes["row"][order]
    parts = part_index[order]
    numbers = row_number[order]
    del hashes, order
    adapters = adapter_of_part[parts]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = (k0[1:] != k0[:-1]) | (k1[1:] != k1[:-1])
    starts = np.flatnonzero(new_group)
    group_of_row = np.cumsum(new_group) - 1
    adapter_min = np.minimum.reduceat(adapters, starts)
    adapter_max = np.maximum.reduceat(adapters, starts)
    multi = adapter_min != adapter_max  # the key is in two or more directories
    later = multi[group_of_row] & (adapters != adapter_min[group_of_row])
    row_min = np.minimum.reduceat(rows, starts)
    row_max = np.maximum.reduceat(rows, starts)
    conflicting = multi & (row_min != row_max)
    outcome.duplicate_keys = int(multi.sum())
    outcome.duplicate_rows = int(later.sum())
    outcome.repeated_in_directory_rows = int((~new_group & ~later).sum())
    outcome.conflicting_keys = int(conflicting.sum())
    conflict_starts = starts[conflicting]
    # Every conflicting key is returned (bounded by MAX_VARIANT_KEYS plus one, so the
    # merge can tell "too many" from "exactly the cap"), because each is read back and
    # reconciled; the drop list is computed whether or not keys conflict, and the
    # merge uses it only once every conflict reconciled.
    conflicts = [(k0[i], k1[i]) for i in conflict_starts[: MAX_VARIANT_KEYS + 1]]
    drop_parts = parts[later]
    drop_numbers = numbers[later]
    for part in np.unique(drop_parts):
        chosen = drop_numbers[drop_parts == part]
        chosen.sort()
        outcome.drop_rows[files.parts[int(part)][1]] = chosen.tolist()
    return outcome, conflicts


def locate_conflicts(
    files: LabelFiles, conflicts: Sequence[tuple[np.uint64, np.uint64]], limit: int = 5
) -> list[ConflictSample]:
    """Re-read the label's parts to name the conflicting keys and their files."""
    wanted = {(int(a), int(b)) for a, b in conflicts[:limit]}
    key_col, end_col = _key_columns(files)
    found: dict[tuple[int, int], ConflictSample] = {}
    for _, path in files.parts:
        with open(path, "rb") as handle:
            for line in handle:
                fields = line.rstrip(b"\n").split(b"\t")
                if len(fields) != len(files.columns):
                    continue
                key = (
                    fields[key_col]
                    if end_col is None
                    else fields[key_col] + b"\x00" + fields[end_col]
                )
                digest = hashlib.blake2b(key, digest_size=16).digest()
                pair = (
                    int.from_bytes(digest[:8], "little"),
                    int.from_bytes(digest[8:], "little"),
                )
                if pair in wanted:
                    sample = found.setdefault(
                        pair,
                        ConflictSample(
                            label=files.label,
                            key=key.decode("utf-8", "replace").replace("\x00", " -> "),
                            paths=[],
                        ),
                    )
                    if path not in sample.paths:
                        sample.paths.append(path)
    return list(found.values())


def _key_hash_pair(key: bytes) -> tuple[int, int]:
    digest = hashlib.blake2b(key, digest_size=16).digest()
    return (int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little"))


def _parse_row(line: bytes, n_columns: int) -> list[str] | None:
    """One CSV row as unquoted field strings, or None when its width is not the header's."""
    import csv
    import io

    (parsed,) = csv.reader(
        io.StringIO(line.decode("utf-8")), delimiter="\t", quotechar="'", strict=True
    )
    return parsed if len(parsed) == n_columns else None


def _json_leaves(value: Any, prefix: str, out: dict[str, str]) -> None:
    """Flatten a parsed JSON value to ``path -> serialized leaf``."""
    if isinstance(value, dict):
        for key, inner in value.items():
            _json_leaves(inner, f"{prefix}.{key}", out)
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            _json_leaves(inner, f"{prefix}[{i}]", out)
    else:
        out[prefix] = json.dumps(value, sort_keys=True)


_PATH_KEY_RE = re.compile(r"\.([^.\[]+)")


def _passes_non_identity_key(path: str) -> bool:
    """Whether a JSON path (``.media.provenance[1].quote``) crosses a non-identity key."""
    return any(key in NON_IDENTITY_JSON_KEYS for key in _PATH_KEY_RE.findall(path))


def row_differences(
    columns: Sequence[str], kept: Sequence[str], other: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Compare two rows of one key: (non-identity differences, identity differences).

    A difference is named by its column (the header name before any ``:type``), and
    for a JSON column by the column and the path of the differing leaf. A plain column
    in ``NON_IDENTITY_COLUMNS`` is a non-identity difference; a JSON column's leaf
    difference is non-identity when its path passes through a ``NON_IDENTITY_JSON_KEYS``
    member; everything else, including a column that is JSON in one row and not in
    the other, is an identity difference.
    """
    non_identity: list[str] = []
    identity: list[str] = []
    for i, column in enumerate(columns):
        a, b = kept[i], other[i]
        if a == b:
            continue
        name = column.split(":")[0]
        if name in NON_IDENTITY_COLUMNS:
            non_identity.append(name)
            continue
        try:
            parsed_a, parsed_b = json.loads(a), json.loads(b)
        except ValueError:
            identity.append(name)
            continue
        if not (
            isinstance(parsed_a, (dict, list)) and isinstance(parsed_b, (dict, list))
        ):
            identity.append(name)
            continue
        leaves_a: dict[str, str] = {}
        leaves_b: dict[str, str] = {}
        _json_leaves(parsed_a, "", leaves_a)
        _json_leaves(parsed_b, "", leaves_b)
        for path in sorted(set(leaves_a) | set(leaves_b)):
            if leaves_a.get(path) == leaves_b.get(path):
                continue
            (non_identity if _passes_non_identity_key(path) else identity).append(
                f"{name}{path}"
            )
    return non_identity, identity


def reconcile_conflicts(
    files: LabelFiles, conflicts: Sequence[tuple[np.uint64, np.uint64]]
) -> tuple[list[VariantSample], list[ConflictSample]]:
    """Read back every conflicting key's rows and sort them into variants and conflicts.

    The kept row is the lowest-index directory's (what ``--skip-duplicate-nodes`` keeps
    and what the edge drop list keeps); every other row of the key is compared with it
    by :func:`row_differences`. A key whose rows differ only in non-identity fields is
    a :class:`VariantSample`; one with any identity difference is a
    :class:`ConflictSample` carrying the differing columns and paths. Rows are kept per
    (key, directory) once, so a key repeated inside one directory is compared on its
    first row, as the adapter wrote it.
    """
    wanted = {(int(a), int(b)) for a, b in conflicts}
    key_col, end_col = _key_columns(files)
    n_columns = len(files.columns)
    rows: dict[tuple[int, int], dict[int, tuple[str, list[str]]]] = {}
    for index, path in files.parts:
        directory = os.path.basename(os.path.dirname(path))
        with open(path, "rb") as handle:
            for line in handle:
                fields = line.rstrip(b"\n").split(b"\t")
                if len(fields) != n_columns:
                    parsed = _parse_row(line, n_columns)
                    if parsed is None:
                        continue
                    fields = [f.encode("utf-8") for f in parsed]
                key = (
                    fields[key_col]
                    if end_col is None
                    else fields[key_col] + b"\x00" + fields[end_col]
                )
                pair = _key_hash_pair(key)
                if pair not in wanted:
                    continue
                by_dir = rows.setdefault(pair, {})
                if index in by_dir:
                    continue
                unquoted = _parse_row(line, n_columns)
                if unquoted is None:
                    raise ValueError(f"{path}: a row of a conflicting key is malformed")
                by_dir[index] = (directory, unquoted)
    variants: list[VariantSample] = []
    refused: list[ConflictSample] = []
    for pair in sorted(rows, key=lambda p: min(rows[p])):
        by_dir = rows[pair]
        kept_index = min(by_dir)
        kept_dir, kept_row = by_dir[kept_index]
        key_text = (
            kept_row[key_col]
            if end_col is None
            else f"{kept_row[key_col]} -> {kept_row[end_col]}"
        )
        non_identity: list[str] = []
        identity: list[str] = []
        for index in sorted(by_dir):
            if index == kept_index:
                continue
            soft, hard = row_differences(files.columns, kept_row, by_dir[index][1])
            non_identity.extend(d for d in soft if d not in non_identity)
            identity.extend(d for d in hard if d not in identity)
        directories = [by_dir[i][0] for i in sorted(by_dir)]
        if identity:
            refused.append(
                ConflictSample(
                    label=files.label,
                    key=key_text,
                    paths=[
                        os.path.join(ADAPTERS_DIRNAME, d, f"{files.label}-part*.csv")
                        for d in directories
                    ],
                    differences=identity,
                )
            )
        else:
            variants.append(
                VariantSample(
                    key=key_text, directories=directories, differences=non_identity
                )
            )
    return variants, refused


class DuplicateConflict(RuntimeError):
    """A duplicate key whose rows differ in an identity field across directories."""


def rewrite_without_rows(src: Path, dst: Path, drop: Sequence[int]) -> int:
    """Copy ``src`` to ``dst`` minus the 0-based row numbers in ``drop``; return rows kept."""
    to_drop = set(drop)
    kept = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(src, "rb", buffering=16 * 2**20) as handle,
        open(dst, "wb", buffering=16 * 2**20) as out,
    ):
        for number, line in enumerate(handle):
            if number in to_drop:
                continue
            out.write(line)
            kept += 1
    return kept


class MergeReport(BaseModel):
    """What the merge checked and wrote; the duplicate counts go into the build manifest."""

    out_dir: str
    n_adapters: int
    labels: dict[str, LabelDuplicates]
    rewritten: dict[str, str] = Field(
        default_factory=dict, description="original part path -> rewritten part path"
    )
    import_call: str
    merged_at: str
    longest_row_bytes: int = Field(
        default=0, description="the longest row of any label, newline included"
    )
    read_buffer_bytes: int = Field(
        default=NEO4J_DEFAULT_READ_BUFFER_BYTES,
        description="the --read-buffer-size the import call carries",
    )

    @property
    def node_duplicates(self) -> dict[str, int]:
        """Duplicate node rows per label, each dropped from a rewritten part."""
        return {
            name: d.duplicate_rows
            for name, d in self.labels.items()
            if d.kind == "node"
        }

    @property
    def edge_duplicates(self) -> dict[str, int]:
        """Duplicate edge rows per label (dropped into the rewritten part files)."""
        return {
            name: d.duplicate_rows
            for name, d in self.labels.items()
            if d.kind == "edge"
        }

    @property
    def node_variants(self) -> dict[str, int]:
        """Per node label, the duplicate ids whose rows differed only in non-identity fields."""
        return {
            name: d.variant_keys
            for name, d in self.labels.items()
            if d.kind == "node" and d.variant_keys
        }

    @property
    def edge_variants(self) -> dict[str, int]:
        """Per edge label, the duplicate keys whose rows differed only in non-identity fields."""
        return {
            name: d.variant_keys
            for name, d in self.labels.items()
            if d.kind == "edge" and d.variant_keys
        }


def _per_directory_call(out_dir: Path, slot: AdapterSlot) -> str:
    path = out_dir / ADAPTERS_DIRNAME / slot.dirname / IMPORT_CALL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{slot.dirname} carries no {IMPORT_CALL_FILENAME}")
    return path.read_text(encoding="utf-8")


def call_prefix_and_suffix(per_directory_call: str) -> tuple[str, str, str]:
    """Split BioCypher's v5 import line: (prefix with flags, suffix, container dir).

    The line is ``bin/neo4j-admin database import full <flags> --nodes="..." ...
    --relationships="..." <database> ``. The prefix holds the delimiter, quote and
    skip flags of the BioCypher config; the container directory is read off the first
    ``--nodes`` argument's header path.
    """
    lines = per_directory_call.splitlines()
    v5 = next(line for line in lines if "database import full" in line)
    first = _NODES_ARG_RE.search(v5)
    if first is None:
        raise ValueError("the per-directory import call has no --nodes argument")
    prefix = v5[: first.start()]
    tail_match = list(_RELS_ARG_RE.finditer(v5)) or list(_NODES_ARG_RE.finditer(v5))
    suffix = v5[tail_match[-1].end() :]
    header = first.group(1).split(",")[0]
    container_dir = os.path.dirname(header)
    if "--skip-duplicate-nodes=true" not in prefix:
        raise ValueError(
            "the BioCypher config did not set skip_duplicate_nodes: true; the merged "
            "import needs it (content-id nodes recur across adapter directories)"
        )
    return prefix.lstrip("\t"), suffix, container_dir


def merge_outputs(
    out_dir: Path,
    order: Sequence[AdapterSlot],
    *,
    workers: int = 1,
    rewrite: bool = True,
) -> MergeReport:
    """Verify every DONE, check headers and duplicates, write the one import call."""
    missing = missing_slots(out_dir, order)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(order)} adapter directories carry no "
            f"{DONE_FILENAME}: indices {compress_ids(s.index for s in missing)} "
            f"({', '.join(s.adapter_class for s in missing[:8])}"
            f"{', ...' if len(missing) > 8 else ''})"
        )
    labels = discover_label_files(out_dir, order)
    prefix, suffix, first_container_dir = call_prefix_and_suffix(
        _per_directory_call(out_dir, order[0])
    )
    # <...>/biocypher-out/<ts>/adapters/<dir> -> <...>/biocypher-out/<ts>
    container_root = os.path.dirname(os.path.dirname(first_container_dir))
    host_root = str(out_dir)
    outcomes: dict[str, LabelDuplicates] = {}
    rewritten: dict[str, str] = {}
    for name in sorted(labels):
        files = labels[name]
        t0 = time.time()
        outcome, conflicts = check_label_duplicates(files, workers)
        outcome.seconds = time.time() - t0
        log.info(
            "%s: %d rows in %d parts checked in %.1f s (%d duplicate rows)",
            name,
            outcome.rows,
            len(files.parts),
            outcome.seconds,
            outcome.duplicate_rows,
        )
        if conflicts:
            what = "ids" if files.kind == "node" else "edge keys"
            if len(conflicts) > MAX_VARIANT_KEYS:
                raise DuplicateConflict(
                    f"{name}: {outcome.conflicting_keys} duplicate {what} map to rows "
                    f"that are not byte-identical across directories, over the "
                    f"{MAX_VARIANT_KEYS} the merge reconciles; first: "
                    + "; ".join(
                        f"{s.key} in {s.paths}"
                        for s in locate_conflicts(files, conflicts)
                    )
                )
            variants, refused = reconcile_conflicts(files, conflicts)
            if refused:
                raise DuplicateConflict(
                    f"{name}: {len(refused)} of {outcome.conflicting_keys} duplicate "
                    f"{what} map to rows that differ outside the non-identity fields "
                    f"(columns {sorted(NON_IDENTITY_COLUMNS)}, JSON keys "
                    f"{sorted(NON_IDENTITY_JSON_KEYS)}); first: "
                    + "; ".join(
                        f"{s.key} in {s.paths} at {s.differences}" for s in refused[:5]
                    )
                )
            outcome.variant_keys = len(variants)
            outcome.variants = variants
            log.info(
                "%s: %d duplicate %s differ only in non-identity fields "
                "(the lowest-index directory's row is imported): %s",
                name,
                len(variants),
                what,
                ", ".join(sorted({d for v in variants for d in v.differences})[:12]),
            )
        outcomes[name] = outcome
        if rewrite:
            for src, drop in outcome.drop_rows.items():
                rel = os.path.relpath(src, out_dir / ADAPTERS_DIRNAME)
                dst = out_dir / MERGED_DIRNAME / rel
                rewrite_without_rows(Path(src), dst, drop)
                rewritten[src] = str(dst)

    def container_path(host_path: str) -> str:
        rel = os.path.relpath(host_path, host_root)
        return os.path.join(container_root, rel)

    args: list[str] = []
    for kind in ("node", "edge"):
        flag = "--nodes" if kind == "node" else "--relationships"
        for name in sorted(labels):
            files = labels[name]
            if files.kind != kind:
                continue
            paths = [container_path(files.header)]
            for _, part in files.parts:
                paths.append(container_path(rewritten.get(part, part)))
            arg = f'{flag}="{",".join(paths)}" '
            if len(arg.encode("utf-8")) > MAX_ARG_BYTES:
                raise ValueError(
                    f"{name}: the {flag} argument is {len(arg)} bytes, over the "
                    f"{MAX_ARG_BYTES} byte single-argument limit"
                )
            args.append(arg)
    longest = max((o.longest_row_bytes for o in outcomes.values()), default=0)
    buffer = read_buffer_bytes(longest)
    prefix = _READ_BUFFER_RE.sub("", prefix)
    v5 = prefix + f"--read-buffer-size={buffer} " + "".join(args) + suffix
    log.info(
        "longest row %d bytes; the import call carries --read-buffer-size=%d",
        longest,
        buffer,
    )
    script = (
        "#!/bin/bash\n"
        "version=$(bin/neo4j-admin --version | cut -d '.' -f 1)\n"
        "if [[ $version -ge 5 ]]; then\n"
        f"\t{v5}\n"
        "else\n"
        "\techo 'neo4j 4 import is not supported by the merged call'; exit 1\n"
        "fi\n"
    )
    call_path = out_dir / IMPORT_CALL_FILENAME
    call_path.write_text(script, encoding="utf-8")
    report = MergeReport(
        out_dir=str(out_dir),
        n_adapters=len(order),
        labels=outcomes,
        rewritten=rewritten,
        import_call=str(call_path),
        merged_at=datetime.now(UTC).isoformat(),
        longest_row_bytes=longest,
        read_buffer_bytes=buffer,
    )
    (out_dir / MERGE_REPORT_FILENAME).write_text(
        report.model_dump_json(indent=1) + "\n", encoding="utf-8"
    )
    return report


def load_merge_report(path: Path) -> MergeReport:
    """Read a report written by :func:`merge_outputs`."""
    return MergeReport.model_validate_json(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- CLI


def _order_from_args(args: argparse.Namespace) -> list[AdapterSlot]:
    recorded = Path(args.out_dir) / "adapter_order.json" if args.out_dir else None
    if recorded is not None and recorded.is_file():
        return read_order(recorded)
    return adapter_order(
        include_private=args.include_private, organisms=parse_organisms(args.organisms)
    )


def _environment_from_args(args: argparse.Namespace) -> dict[str, str]:
    env = {
        "KG_ORGANISMS": format_organisms(parse_organisms(args.organisms)),
        "INCLUDE_PRIVATE": "1" if args.include_private else "0",
    }
    for key in (
        "TORCHCELL_SRC",
        "BUILD_COMMIT",
        "DEV_DATA_ROOT",
        "SERVE_ROOT",
        "BUILD_ROOT",
        "KG_CONFIG",
        "GEN_OVERRIDES",
        "GEN_IMAGE",
        "TCDB_TAGS",
    ):
        value = getattr(args, key.lower(), None)
        if value:
            env[key] = value
    return env


def main(argv: list[str] | None = None) -> int:
    """``plan`` (dry-run sbatch lines), ``submit``, ``missing``, ``fence``, ``merge``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_membership(p: argparse.ArgumentParser) -> None:
        p.add_argument("--include-private", action="store_true")
        p.add_argument("--organisms", default="scerevisiae,ecoli,pputida")
        p.add_argument("--tiers", default=str(DEFAULT_TIER_TABLE))

    def add_submission(p: argparse.ArgumentParser) -> None:
        add_membership(p)
        p.add_argument("--script", default=DEFAULT_ARRAY_SCRIPT)
        p.add_argument("--out-name", required=True, help="<ts> under biocypher-out")
        p.add_argument(
            "--out-dir", default=None, help="host path of <ts> (for the order)"
        )
        p.add_argument(
            "--indices", default=None, help="slurm array spelling, e.g. 0-3,7"
        )
        p.add_argument("--torchcell-src", default=None)
        p.add_argument("--build-commit", default=None)
        p.add_argument("--dev-data-root", default=None)
        p.add_argument("--serve-root", default=None)
        p.add_argument("--build-root", default=None)
        p.add_argument("--kg-config", default=None)
        p.add_argument(
            "--gen-overrides", default=None, help="hydra overrides, space-separated"
        )
        p.add_argument(
            "--gen-image",
            default=None,
            help=(
                "the prebuilt generation image tag every task runs "
                "(torchcell.knowledge_graphs.generation_image); omitted, each task "
                "installs torchcell itself"
            ),
        )
        p.add_argument("--tcdb-tags", default=None)
        p.add_argument("--job-name-prefix", default="gen_adapter")
        p.add_argument(
            "--dependency", default=None, help="sbatch --dependency, e.g. afterany:4919"
        )

    p_plan = sub.add_parser("plan", help="print the sbatch lines without submitting")
    add_submission(p_plan)
    p_submit = sub.add_parser("submit", help="submit one array per tier")
    add_submission(p_submit)
    p_submit.add_argument("--dry-run", action="store_true")
    p_missing = sub.add_parser(
        "missing", help="slots without DONE and their re-run lines"
    )
    add_submission(p_missing)
    p_fence = sub.add_parser("fence", help="print the dev-tree fence lines")
    add_membership(p_fence)
    p_fence.add_argument("--dev-data-root", required=True)
    p_merge = sub.add_parser(
        "merge", help="verify DONE, check duplicates, write the call"
    )
    add_membership(p_merge)
    p_merge.add_argument("--out-dir", required=True)
    p_merge.add_argument("--workers", type=int, default=1)
    p_merge.add_argument(
        "--expect", type=int, default=None, help="adapters the order must have"
    )
    p_merge.add_argument(
        "--no-rewrite", action="store_true", help="check only (measurement)"
    )
    p_order = sub.add_parser("order", help="print the build order")
    add_membership(p_order)
    p_order.add_argument(
        "--write", default=None, help="also write adapter_order.json here"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.command == "fence":
        print(
            "\n".join(
                dev_fence_lines(
                    args.dev_data_root,
                    args.include_private,
                    parse_organisms(args.organisms),
                )
            )
        )
        return 0
    if args.command == "order":
        order = adapter_order(args.include_private, parse_organisms(args.organisms))
        for slot in order:
            print(f"{slot.index:03d} {slot.dataset_class} {slot.adapter_class}")
        if args.write:
            write_order(order, Path(args.write))
        return 0
    tiers = load_tier_table(Path(args.tiers))
    if args.command == "merge":
        args.out_dir = str(args.out_dir)
        order = _order_from_args(args)
        if args.expect is not None and len(order) != args.expect:
            raise SystemExit(
                f"the order has {len(order)} adapters, expected {args.expect}"
            )
        report = merge_outputs(
            Path(args.out_dir), order, workers=args.workers, rewrite=not args.no_rewrite
        )
        for name, outcome in report.labels.items():
            if outcome.duplicate_rows or outcome.repeated_in_directory_rows:
                print(
                    f"{name}: {outcome.rows} rows, {outcome.duplicate_rows} duplicate rows "
                    f"over {outcome.duplicate_keys} keys ({outcome.kind}) dropped"
                    + (
                        f", {outcome.variant_keys} keys differ only in non-identity "
                        "fields (lowest-index row imported)"
                        if outcome.variant_keys
                        else ""
                    )
                    + (
                        f", {outcome.repeated_in_directory_rows} rows repeat a key "
                        "inside one directory (kept)"
                        if outcome.repeated_in_directory_rows
                        else ""
                    )
                )
        print(f"rewritten part files: {len(report.rewritten)}")
        print(
            f"longest row {report.longest_row_bytes} bytes; "
            f"--read-buffer-size={report.read_buffer_bytes}"
        )
        print(f"MERGED_CALL={report.import_call}")
        return 0
    order = _order_from_args(args)
    environment = _environment_from_args(args)
    if args.command == "missing":
        out_dir = Path(args.out_dir) if args.out_dir else None
        if out_dir is None:
            raise SystemExit("missing needs --out-dir")
        missing = missing_slots(out_dir, order)
        print(
            f"{len(missing)} of {len(order)} adapter directories carry no {DONE_FILENAME}"
        )
        for slot in missing:
            print(f"  {slot.index:03d} {slot.adapter_class}")
        for line in recovery_commands(
            missing,
            tiers,
            script=args.script,
            out_name=args.out_name,
            environment=environment,
        ):
            print(line)
        return 1 if missing else 0
    indices = None
    if args.indices:
        indices = _expand_ids(args.indices)
    submissions = tier_submissions(
        order,
        tiers,
        script=args.script,
        out_name=args.out_name,
        environment=environment,
        indices=indices,
        job_name_prefix=args.job_name_prefix,
        dependency=args.dependency,
    )
    job_ids = submit(submissions, dry_run=args.command == "plan" or args.dry_run)
    if job_ids:
        print(f"JOB_IDS={','.join(str(j) for j in job_ids)}")
    return 0


def _expand_ids(spec: str) -> list[int]:
    """``"0-2,5"`` -> ``[0, 1, 2, 5]``; a ``%N`` suffix is not part of the spelling."""
    ids: list[int] = []
    for token in spec.split(","):
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    return ids


if __name__ == "__main__":
    sys.exit(main())
