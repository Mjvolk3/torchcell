# tests/torchcell/benchmark/test_set_bundle_grouping.py
# [[tests.torchcell.benchmark.test_set_bundle_grouping]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_set_bundle_grouping.py
"""``experiments/035-benchmark-bundles/scripts/set_bundle_grouping.py`` on ``toy-fitness``.

The script rewrites ``benchmark.json`` only: the three pinned CSVs keep their bytes, so
the bundle still loads, and every field other than the three set ones is unchanged.
A bundle that does not load is refused before anything is written.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from torchcell.benchmark.bundle import BenchmarkBundle

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "035-benchmark-bundles"
    / "scripts"
    / "set_bundle_grouping.py"
)
SLUG = "toy-fitness"
GROUPING = ("organism", "task_group", "split_name")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("set_bundle_grouping", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv_bytes(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.glob("*.csv"))}


def test_round_trip_sets_the_fields_and_keeps_everything_else(
    datasets_root: Path,
) -> None:
    root = datasets_root / SLUG
    before_record = json.loads((root / "benchmark.json").read_text())
    before_csv = _csv_bytes(root)

    dataset = _load().set_grouping(
        root,
        organism="Saccharomyces cerevisiae",
        task_group="toy-fitness-task",
        split_name="random_s0",
    )

    assert (dataset.organism, dataset.task_group, dataset.split_name) == (
        "Saccharomyces cerevisiae",
        "toy-fitness-task",
        "random_s0",
    )
    assert _csv_bytes(root) == before_csv
    after_record = json.loads((root / "benchmark.json").read_text())
    assert {k: after_record[k] for k in GROUPING} == {
        "organism": "Saccharomyces cerevisiae",
        "task_group": "toy-fitness-task",
        "split_name": "random_s0",
    }
    assert {k: v for k, v in after_record.items() if k not in GROUPING} == {
        k: v for k, v in before_record.items() if k not in GROUPING
    }
    assert BenchmarkBundle.load(root).dataset == dataset


def test_unset_task_group_and_split_name_keep_their_values(datasets_root: Path) -> None:
    root = datasets_root / SLUG
    module = _load()
    module.set_grouping(
        root, organism="Escherichia coli", task_group="g", split_name="s"
    )
    dataset = module.set_grouping(root, organism="Saccharomyces cerevisiae")
    assert (dataset.organism, dataset.task_group, dataset.split_name) == (
        "Saccharomyces cerevisiae",
        "g",
        "s",
    )


def test_cli_sets_organism_only_and_prints_the_result(
    datasets_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = datasets_root / SLUG
    _load().main(["--bundle", str(root), "--organism", "Saccharomyces cerevisiae"])
    assert capsys.readouterr().out == (
        "toy-fitness: organism='Saccharomyces cerevisiae' task_group=None "
        "split_name=None\n"
    )
    assert BenchmarkBundle.load(root).dataset.organism == "Saccharomyces cerevisiae"


def test_a_bundle_that_does_not_load_is_refused_untouched(datasets_root: Path) -> None:
    root = datasets_root / SLUG
    labels = root / "labels.csv"
    labels.write_bytes(labels.read_bytes().replace(b"0.5", b"0.6"))
    record = (root / "benchmark.json").read_bytes()
    with pytest.raises(ValueError, match="does not match the sha256"):
        _load().set_grouping(root, organism="Escherichia coli")
    assert (root / "benchmark.json").read_bytes() == record
