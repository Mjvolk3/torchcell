# torchcell/benchmark/bundle.py
# [[torchcell.benchmark.bundle]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/bundle.py
# Test file: tests/torchcell/benchmark/test_bundle.py

"""One benchmark dataset on disk: the public split and template, the private labels.

A bundle is a directory ``<datasets_root>/<slug>/`` with four files:

- ``benchmark.json``: a :class:`BenchmarkDataset` (title, loader class, targets, split
  sizes, and the sha256 of the three files below).
- ``splits.csv`` (public): ``record_id,split`` for every record of all three splits.
- ``template.csv`` (public): ``record_id,split,target,prediction`` with an empty
  prediction for every validation and test (record, target) pair. It is the exact set
  of rows a submission must fill, and all a submitter needs to validate locally.
- ``labels.csv`` (read by the grader only): ``record_id,split,target,value`` for the
  same pairs.

:func:`write_bundle` builds the directory from plain inputs (split id lists and a
``(record_id, target) -> value`` map), so it does not import a dataset loader; the
script that extracts those inputs from a built dataset lives with that dataset's
experiment. :meth:`BenchmarkBundle.load` verifies each file against the recorded
sha256 and refuses a bundle whose bytes changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from torchcell.benchmark.grading import TASK_METRICS, MetricName, PairKey, Task
from torchcell.benchmark.submission import PREDICTION_COLUMNS, SCORED_SPLITS, Split

BENCHMARK_FILENAME = "benchmark.json"
SPLITS_FILENAME = "splits.csv"
TEMPLATE_FILENAME = "template.csv"
LABELS_FILENAME = "labels.csv"
SPLITS_COLUMNS: tuple[str, ...] = ("record_id", "split")
LABELS_COLUMNS: tuple[str, ...] = ("record_id", "split", "target", "value")
MIN_RECORDS_PER_TARGET = 2

Slug = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$"),
]


def sha256_bytes(data: bytes) -> str:
    """sha256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


class SourceRecord(BaseModel):
    """One source the bundle's records or labels were built from, pinned by hash.

    A set of files retrieved the same way (one SGD phenotype file per gene, say) is one
    record with ``n_files`` set and ``sha256`` over the sorted per-file hashes,
    newline-joined with a trailing newline, the rule ``releases.content_sha256`` uses.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="What the source is, for a reader.")
    role: str = Field(
        description="What it contributed: universe, label 1, label 0, ..."
    )
    source_url: str | None = Field(
        default=None, description="Where it was retrieved from; historical, not live."
    )
    retrieval_method: str = Field(description="direct_url, tc_data_archive, ...")
    retrieved_at: datetime | None = None
    sha256: str
    bytes: int | None = None
    n_files: int | None = None
    note: str | None = None


class BundleProvenance(BaseModel):
    """Where a bundle's records and labels came from and how they were derived.

    Public: served with the dataset and shown at the top of its page, so a reader can
    answer "where did this come from and how was it done" without asking.
    """

    model_config = ConfigDict(frozen=True)

    script: str = Field(description="Repo path of the script that built the bundle.")
    label_rule: str
    split_rule: str
    sources: list[SourceRecord]
    notes: list[str] = Field(default_factory=list)


class BenchmarkDatasetPublic(BaseModel):
    """The public description of a benchmark dataset (what ``/datasets`` returns)."""

    model_config = ConfigDict(frozen=True)

    slug: Slug
    title: str
    description: str
    loader_class: str = Field(description="The torchcell loader class of the dataset.")
    citation_key: str = Field(description="Citation key of the source publication.")
    version: str = Field(
        description="Version of this bundle; a new split is a new one."
    )
    task: Task = Field(
        default="regression",
        description=(
            "regression: labels are real values. binary: labels are 0 or 1 and a "
            "prediction is a score, larger meaning more likely 1."
        ),
    )
    targets: list[str]
    n_train: int
    n_val: int
    n_test: int
    primary_metric: MetricName
    docs_url: str | None = None
    tc_data_slug: str | None = Field(
        default=None, description="Slug of the built dataset on the tc-data endpoint."
    )
    splits_sha256: str
    template_sha256: str
    provenance: BundleProvenance | None = Field(
        default=None, description="Sources, label rule and split rule of this bundle."
    )

    @model_validator(mode="after")
    def _primary_metric_belongs_to_the_task(self) -> Self:
        if self.primary_metric not in TASK_METRICS[self.task]:
            raise ValueError(
                f"primary metric {self.primary_metric!r} is not a {self.task} metric"
            )
        return self


class BenchmarkDataset(BenchmarkDatasetPublic):
    """``benchmark.json``: the public description plus the private labels' hash."""

    labels_sha256: str
    built_at: datetime

    def public(self) -> BenchmarkDatasetPublic:
        """This record without the grader-only fields."""
        return BenchmarkDatasetPublic.model_validate(
            self.model_dump(include=set(BenchmarkDatasetPublic.model_fields))
        )


class SubmissionSpec(BaseModel):
    """What a submission must cover: every scored (record, target) pair and its split."""

    model_config = ConfigDict(frozen=True)

    expected: dict[PairKey, Split]

    @classmethod
    def from_template_csv(cls, text: str) -> Self:
        """Read the spec from the text of a dataset's public ``template.csv``."""
        reader = csv.reader(io.StringIO(text, newline=""))
        header = tuple(next(reader))
        if header != PREDICTION_COLUMNS:
            raise ValueError(f"template header is {header}, not {PREDICTION_COLUMNS}")
        return cls(expected={(row[0], row[2]): Split(row[1]) for row in reader})


class BenchmarkBundle(BaseModel):
    """A loaded bundle: the dataset record, the submission spec, and the labels."""

    model_config = ConfigDict(frozen=True)

    dataset: BenchmarkDataset
    root: Path
    spec: SubmissionSpec
    labels: dict[PairKey, float]

    @classmethod
    def load(cls, root: Path) -> Self:
        """Load ``root`` and verify the three data files against their recorded sha256."""
        dataset = BenchmarkDataset.model_validate_json(
            (root / BENCHMARK_FILENAME).read_text(encoding="utf-8")
        )
        if dataset.slug != root.name:
            raise ValueError(
                f"bundle directory {root.name!r} holds slug {dataset.slug!r}"
            )
        recorded = {
            SPLITS_FILENAME: dataset.splits_sha256,
            TEMPLATE_FILENAME: dataset.template_sha256,
            LABELS_FILENAME: dataset.labels_sha256,
        }
        contents: dict[str, bytes] = {}
        for filename, expected_sha in recorded.items():
            data = (root / filename).read_bytes()
            if sha256_bytes(data) != expected_sha:
                raise ValueError(
                    f"{root / filename} does not match the sha256 in {BENCHMARK_FILENAME}"
                )
            contents[filename] = data
        spec = SubmissionSpec.from_template_csv(contents[TEMPLATE_FILENAME].decode())
        reader = csv.reader(io.StringIO(contents[LABELS_FILENAME].decode(), newline=""))
        if tuple(next(reader)) != LABELS_COLUMNS:
            raise ValueError(f"{LABELS_FILENAME} header is not {LABELS_COLUMNS}")
        labels = {(row[0], row[2]): float(row[3]) for row in reader}
        if labels.keys() != spec.expected.keys():
            raise ValueError("labels.csv and template.csv list different pairs")
        return cls(dataset=dataset, root=root, spec=spec, labels=labels)


def load_bundles(datasets_root: Path) -> dict[str, BenchmarkBundle]:
    """Every bundle under ``datasets_root`` (a directory holding ``benchmark.json``)."""
    return {
        path.name: BenchmarkBundle.load(path)
        for path in sorted(datasets_root.iterdir())
        if (path / BENCHMARK_FILENAME).is_file()
    }


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_bundle(
    datasets_root: Path,
    *,
    slug: str,
    title: str,
    description: str,
    loader_class: str,
    citation_key: str,
    version: str,
    primary_metric: MetricName,
    splits: Mapping[Split, Sequence[str]],
    values: Mapping[PairKey, float],
    task: Task = "regression",
    docs_url: str | None = None,
    tc_data_slug: str | None = None,
    built_at: datetime | None = None,
    provenance: BundleProvenance | None = None,
) -> BenchmarkDataset:
    """Write ``<datasets_root>/<slug>/`` from split id lists and a label map.

    ``splits`` lists the record ids of each of the three splits; ``values`` maps
    ``(record_id, target)`` to the label for validation and test records (a pair that
    was not measured is simply absent). Raises ``ValueError`` when the splits overlap, a
    scored record has no label, a label names a record outside the scored splits, a label
    is not finite, a label of a ``binary`` task is not 0 or 1, or a target has fewer
    than two records or constant labels in a scored split (its correlation, or its
    ranking metrics, would be undefined).
    """
    membership: dict[str, Split] = {}
    for split in Split:
        for record_id in splits[split]:
            if record_id in membership:
                raise ValueError(f"record {record_id!r} is in more than one split")
            membership[record_id] = split
    for (record_id, target), value in values.items():
        if membership.get(record_id) not in SCORED_SPLITS:
            raise ValueError(
                f"label for {record_id!r} is outside the val and test splits"
            )
        if not math.isfinite(value):
            raise ValueError(f"label for ({record_id!r}, {target!r}) is not finite")
        if task == "binary" and value not in (0.0, 1.0):
            raise ValueError(f"label for ({record_id!r}, {target!r}) is not 0 or 1")
    labeled_records = {record_id for record_id, _ in values}
    for split in SCORED_SPLITS:
        unlabeled = [r for r in splits[split] if r not in labeled_records]
        if unlabeled:
            raise ValueError(f"{len(unlabeled)} {split} records have no label")
    targets = sorted({target for _, target in values})
    for split in SCORED_SPLITS:
        for target in targets:
            column = [
                v
                for (record_id, t), v in values.items()
                if t == target and membership[record_id] == split
            ]
            if len(column) < MIN_RECORDS_PER_TARGET or min(column) == max(column):
                raise ValueError(
                    f"target {target!r} has fewer than {MIN_RECORDS_PER_TARGET} records "
                    f"or constant labels in the {split} split"
                )

    pairs = sorted(values, key=lambda key: (membership[key[0]], key[0], key[1]))
    splits_bytes = _csv_bytes(
        SPLITS_COLUMNS,
        [
            (r, membership[r])
            for r in sorted(membership, key=lambda r: (membership[r], r))
        ],
    )
    template_bytes = _csv_bytes(
        PREDICTION_COLUMNS, [(r, membership[r], t, "") for r, t in pairs]
    )
    labels_bytes = _csv_bytes(
        LABELS_COLUMNS, [(r, membership[r], t, repr(values[(r, t)])) for r, t in pairs]
    )
    dataset = BenchmarkDataset(
        slug=slug,
        title=title,
        description=description,
        loader_class=loader_class,
        citation_key=citation_key,
        version=version,
        task=task,
        targets=targets,
        n_train=len(splits[Split.TRAIN]),
        n_val=len(splits[Split.VAL]),
        n_test=len(splits[Split.TEST]),
        primary_metric=primary_metric,
        docs_url=docs_url,
        tc_data_slug=tc_data_slug,
        splits_sha256=sha256_bytes(splits_bytes),
        template_sha256=sha256_bytes(template_bytes),
        labels_sha256=sha256_bytes(labels_bytes),
        built_at=built_at or datetime.now(UTC),
        provenance=provenance,
    )
    root = datasets_root / dataset.slug
    root.mkdir(parents=True, exist_ok=False)
    (root / SPLITS_FILENAME).write_bytes(splits_bytes)
    (root / TEMPLATE_FILENAME).write_bytes(template_bytes)
    (root / LABELS_FILENAME).write_bytes(labels_bytes)
    (root / BENCHMARK_FILENAME).write_text(
        dataset.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return dataset


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: build a bundle from a meta JSON, a splits JSON and a values CSV.

    ``--meta`` holds the keyword fields of :func:`write_bundle` (slug, title,
    description, loader_class, citation_key, version, primary_metric, and optionally
    task, docs_url and tc_data_slug). ``--splits`` is ``{"train": [...], "val": [...],
    "test": [...]}``. ``--values`` is a CSV with the header ``record_id,target,value``.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    args = parser.parse_args(argv)

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    raw_splits = json.loads(args.splits.read_text(encoding="utf-8"))
    with args.values.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != ("record_id", "target", "value"):
            raise ValueError("values CSV header must be record_id,target,value")
        values = {(row[0], row[1]): float(row[2]) for row in reader}
    dataset = write_bundle(
        args.datasets_root,
        splits={split: raw_splits[split.value] for split in Split},
        values=values,
        **meta,
    )
    print(dataset.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
