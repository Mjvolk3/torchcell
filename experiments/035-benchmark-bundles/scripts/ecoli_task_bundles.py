# experiments/035-benchmark-bundles/scripts/ecoli_task_bundles.py
# [[experiments.035-benchmark-bundles.scripts.ecoli_task_bundles]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-benchmark-bundles/scripts/ecoli_task_bundles
"""Build one benchmark bundle per exported E. coli board.

``experiments/050-ecoli-benchmark/scripts/export_boards.py`` (on main, where
``torchcell.tasks`` lives) writes one directory per board: ``board.json``, ``splits.csv``
and ``labels.csv``. This script turns each into a bundle with
:func:`torchcell.benchmark.bundle.write_bundle`, which checks the splits and labels and
pins the three CSVs by sha256. Each bundle is grouped for search and browsing with
organism ``Escherichia coli``, ``task_group`` = the board's ``task`` and ``split_name``
= its ``split``.

    python experiments/035-benchmark-bundles/scripts/ecoli_task_bundles.py \
        --boards <export>/boards --datasets-root <datasets root>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torchcell.benchmark.bundle import BundleProvenance, SourceRecord, write_bundle
from torchcell.benchmark.submission import Split

EXPORT_SCRIPT = "experiments/050-ecoli-benchmark/scripts/export_boards.py"
ORGANISM = "Escherichia coli"


def main() -> None:
    """Write a bundle for every board directory under ``--boards``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--boards", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    args = parser.parse_args()
    for board in sorted(
        p for p in args.boards.iterdir() if (p / "board.json").is_file()
    ):
        meta = json.loads((board / "board.json").read_text(encoding="utf-8"))
        splits: dict[Split, list[str]] = {split: [] for split in Split}
        with (board / "splits.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                splits[Split(row["split"])].append(row["record_id"])
        values: dict[tuple[str, str], float] = {}
        with (board / "labels.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                values[(row["record_id"], row["target"])] = float(row["value"])
        dataset = write_bundle(
            args.datasets_root,
            slug=meta["slug"],
            title=meta["title"],
            description=meta["description"],
            loader_class=meta["loader_class"],
            citation_key=meta["citation_key"],
            version="1",
            primary_metric="pearson",
            splits=splits,
            values=values,
            task="regression",
            docs_url=meta["docs_url"],
            tc_data_slug=meta["tc_data_slug"],
            organism=ORGANISM,
            task_group=meta["task"],
            split_name=meta["split"],
            provenance=BundleProvenance(
                script=EXPORT_SCRIPT,
                label_rule=(
                    "The label of a record as its dataset releases it, read by "
                    f"torchcell.tasks.load_task({meta['task']!r}); a record is a sample "
                    "when every deleted gene resolves to one protein-coding gene of the "
                    "genome its records name."
                ),
                split_rule=(
                    f"The pinned split {meta['split']} of the task, shipped in the "
                    "torchcell package (torchcell/tasks/data) and checked by sha256."
                ),
                sources=[
                    SourceRecord(
                        name=f"tc-data archive {meta['source']['archive']}",
                        role="records and labels",
                        source_url=meta["source"]["url"],
                        retrieval_method="tc_data_archive",
                        sha256=meta["source"]["archive_sha256"],
                        bytes=meta["source"]["archive_bytes"],
                        note=f"content sha256 {meta['source']['content_sha256']}",
                    )
                ],
                notes=[
                    "A record id is the row of the record in the built dataset store."
                ],
            ),
        )
        print(f"{meta['slug']}: bundle written ({len(values):,} labels)")
        del dataset


if __name__ == "__main__":
    main()
