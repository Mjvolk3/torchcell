# experiments/035-benchmark-bundles/scripts/set_bundle_grouping.py
# [[experiments.035-benchmark-bundles.scripts.set_bundle_grouping]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-benchmark-bundles/scripts/set_bundle_grouping
r"""Set the grouping fields of an existing benchmark bundle's ``benchmark.json`` in place.

``organism``, ``task_group`` and ``split_name`` place a board in the search index and
the browse tree of ``tc-bench`` (:mod:`torchcell.benchmark.search`); they change
nothing about grading. A bundle written before the fields existed gets them here
without a rebuild: ``benchmark.json`` holds the sha256 of the three CSVs and is not
itself hashed, so rewriting it leaves every pin valid. ``--organism`` is always set;
``--task-group`` and ``--split-name`` are set when given and keep their current value
otherwise.

The bundle is loaded with the service's own loader
(:meth:`torchcell.benchmark.bundle.BenchmarkBundle.load`) before the write, so a bundle
that is already broken is refused untouched, and again after it, which must return the
updated record. Run from the repo root::

    python experiments/035-benchmark-bundles/scripts/set_bundle_grouping.py \
        --bundle <TC_BENCH_DATASETS_ROOT>/gene-essentiality-sgd \
        --organism "Saccharomyces cerevisiae"
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from torchcell.benchmark.bundle import (
    BENCHMARK_FILENAME,
    BenchmarkBundle,
    BenchmarkDataset,
)


def set_grouping(
    bundle_root: Path,
    *,
    organism: str,
    task_group: str | None = None,
    split_name: str | None = None,
) -> BenchmarkDataset:
    """Rewrite ``bundle_root/benchmark.json`` with the grouping fields set.

    Args:
        bundle_root: The bundle directory, ``<datasets_root>/<slug>``.
        organism: Binomial name of the organism, e.g. ``"Saccharomyces cerevisiae"``.
        task_group: The task the board belongs to; unchanged when ``None``.
        split_name: Name of the board's split; unchanged when ``None``.

    Returns:
        The record as the service's loader reads it back after the write.

    Raises:
        ValueError: When the bundle does not load before the write, or the record
            loaded after it is not the one written.
    """
    before = BenchmarkBundle.load(bundle_root).dataset
    update: dict[str, str] = {"organism": organism}
    if task_group is not None:
        update["task_group"] = task_group
    if split_name is not None:
        update["split_name"] = split_name
    updated = BenchmarkDataset.model_validate(
        {**before.model_dump(mode="json"), **update}
    )
    (bundle_root / BENCHMARK_FILENAME).write_text(
        updated.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    after = BenchmarkBundle.load(bundle_root).dataset
    if after != updated:
        raise ValueError(
            f"{bundle_root / BENCHMARK_FILENAME} did not read back as written"
        )
    return after


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: set the grouping fields of one bundle and print the three values."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--organism", required=True)
    parser.add_argument("--task-group", default=None)
    parser.add_argument("--split-name", default=None)
    args = parser.parse_args(argv)
    dataset = set_grouping(
        args.bundle,
        organism=args.organism,
        task_group=args.task_group,
        split_name=args.split_name,
    )
    print(
        f"{dataset.slug}: organism={dataset.organism!r} "
        f"task_group={dataset.task_group!r} split_name={dataset.split_name!r}"
    )


if __name__ == "__main__":
    main()
