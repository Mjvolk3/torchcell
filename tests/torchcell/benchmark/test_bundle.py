# tests/torchcell/benchmark/test_bundle.py
# [[tests.torchcell.benchmark.test_bundle]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_bundle.py
"""``torchcell.benchmark.bundle`` on the ``toy-fitness`` bundle of ``conftest.py``.

The written files are checked byte for byte: ``splits.csv`` lists all ten records
ordered by split then id, ``template.csv`` lists the eight scored pairs with an empty
prediction, and ``labels.csv`` the same pairs with ``repr`` of the label. Loading
verifies each file against the sha256 in ``benchmark.json``, so one changed byte in
``labels.csv`` is refused. The public record drops ``labels_sha256`` and ``built_at``.
"""

import hashlib
import json
from pathlib import Path

import pytest

from torchcell.benchmark.bundle import (
    BenchmarkBundle,
    BenchmarkDataset,
    SubmissionSpec,
    load_bundles,
    main,
    write_bundle,
)
from torchcell.benchmark.submission import Split

SLUG = "toy-fitness"
SPLITS_CSV = (
    "record_id,split\n"
    "s1,test\ns2,test\ns3,test\ns4,test\n"
    "t1,train\nt2,train\n"
    "v1,val\nv2,val\nv3,val\nv4,val\n"
)
TEMPLATE_CSV = (
    "record_id,split,target,prediction\n"
    "s1,test,fitness,\ns2,test,fitness,\ns3,test,fitness,\ns4,test,fitness,\n"
    "v1,val,fitness,\nv2,val,fitness,\nv3,val,fitness,\nv4,val,fitness,\n"
)
LABELS_CSV = (
    "record_id,split,target,value\n"
    "s1,test,fitness,0.5\ns2,test,fitness,1.0\ns3,test,fitness,1.5\ns4,test,fitness,2.5\n"
    "v1,val,fitness,1.0\nv2,val,fitness,2.0\nv3,val,fitness,3.0\nv4,val,fitness,4.0\n"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_written_files_are_exact(datasets_root: Path) -> None:
    root = datasets_root / SLUG
    assert sorted(p.name for p in root.iterdir()) == [
        "benchmark.json",
        "labels.csv",
        "splits.csv",
        "template.csv",
    ]
    assert (root / "splits.csv").read_text() == SPLITS_CSV
    assert (root / "template.csv").read_text() == TEMPLATE_CSV
    assert (root / "labels.csv").read_text() == LABELS_CSV


def test_benchmark_json_records_counts_and_hashes(datasets_root: Path) -> None:
    dataset = BenchmarkDataset.model_validate_json(
        (datasets_root / SLUG / "benchmark.json").read_text()
    )
    assert (dataset.n_train, dataset.n_val, dataset.n_test) == (2, 4, 4)
    assert dataset.targets == ["fitness"]
    assert dataset.task == "regression"
    assert dataset.primary_metric == "pearson"
    assert dataset.splits_sha256 == _sha(SPLITS_CSV)
    assert dataset.template_sha256 == _sha(TEMPLATE_CSV)
    assert dataset.labels_sha256 == _sha(LABELS_CSV)
    assert dataset.built_at.isoformat() == "2026-10-01T12:00:00+00:00"


def test_public_record_hides_grader_fields(bundle: BenchmarkBundle) -> None:
    public = bundle.dataset.public().model_dump()
    assert "labels_sha256" not in public
    assert "built_at" not in public
    assert public["slug"] == SLUG
    assert public["template_sha256"] == _sha(TEMPLATE_CSV)


def test_load_builds_spec_and_labels(bundle: BenchmarkBundle) -> None:
    assert len(bundle.spec.expected) == 8
    assert bundle.spec.expected[("v1", "fitness")] is Split.VAL
    assert bundle.spec.expected[("s4", "fitness")] is Split.TEST
    assert bundle.labels[("s4", "fitness")] == 2.5
    assert bundle.labels.keys() == bundle.spec.expected.keys()


def test_spec_from_template_text() -> None:
    spec = SubmissionSpec.from_template_csv(TEMPLATE_CSV)
    assert len(spec.expected) == 8
    with pytest.raises(ValueError, match="template header"):
        SubmissionSpec.from_template_csv("id,split,target,prediction\n")


def test_load_refuses_changed_bytes(datasets_root: Path) -> None:
    labels = datasets_root / SLUG / "labels.csv"
    labels.write_text(LABELS_CSV.replace("2.5", "2.6"))
    with pytest.raises(ValueError, match="labels.csv does not match the sha256"):
        BenchmarkBundle.load(datasets_root / SLUG)


def test_load_refuses_renamed_directory(datasets_root: Path) -> None:
    renamed = datasets_root / "other-name"
    (datasets_root / SLUG).rename(renamed)
    with pytest.raises(ValueError, match="holds slug 'toy-fitness'"):
        BenchmarkBundle.load(renamed)


def test_load_bundles_skips_directories_without_a_record(datasets_root: Path) -> None:
    (datasets_root / "scratch").mkdir()
    assert list(load_bundles(datasets_root)) == [SLUG]


def _write(
    root: Path, splits: dict[Split, list[str]], values: dict[tuple[str, str], float]
) -> None:
    write_bundle(
        root,
        slug="bad",
        title="t",
        description="d",
        loader_class="L",
        citation_key="k",
        version="1",
        primary_metric="pearson",
        splits=splits,
        values=values,
    )


GOOD_SPLITS = {Split.TRAIN: ["t1"], Split.VAL: ["v1", "v2"], Split.TEST: ["s1", "s2"]}
GOOD_VALUES = {("v1", "y"): 1.0, ("v2", "y"): 2.0, ("s1", "y"): 1.0, ("s2", "y"): 3.0}


@pytest.mark.parametrize(
    ("splits", "values", "message"),
    [
        (
            {**GOOD_SPLITS, Split.TEST: ["s1", "s2", "v1"]},
            GOOD_VALUES,
            "record 'v1' is in more than one split",
        ),
        (
            GOOD_SPLITS,
            {**GOOD_VALUES, ("t1", "y"): 1.0},
            "label for 't1' is outside the val and test splits",
        ),
        (
            GOOD_SPLITS,
            {**GOOD_VALUES, ("zz", "y"): 1.0},
            "label for 'zz' is outside the val and test splits",
        ),
        (
            GOOD_SPLITS,
            {**GOOD_VALUES, ("s2", "y"): float("nan")},
            r"label for \('s2', 'y'\) is not finite",
        ),
        (
            GOOD_SPLITS,
            {k: v for k, v in GOOD_VALUES.items() if k != ("s2", "y")},
            "1 test records have no label",
        ),
        (
            GOOD_SPLITS,
            {**GOOD_VALUES, ("s2", "y"): 1.0},
            "target 'y' has fewer than 2 records or constant labels in the test split",
        ),
        (
            GOOD_SPLITS,
            {**GOOD_VALUES, ("v1", "z"): 1.0},
            "target 'z' has fewer than 2 records or constant labels in the val split",
        ),
    ],
)
def test_write_bundle_refuses_bad_inputs(
    tmp_path: Path,
    splits: dict[Split, list[str]],
    values: dict[tuple[str, str], float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _write(tmp_path, splits, values)
    assert not (tmp_path / "bad").exists()


def test_write_bundle_does_not_overwrite(datasets_root: Path) -> None:
    with pytest.raises(FileExistsError):
        write_bundle(
            datasets_root,
            slug=SLUG,
            title="t",
            description="d",
            loader_class="L",
            citation_key="k",
            version="2",
            primary_metric="pearson",
            splits=GOOD_SPLITS,
            values=GOOD_VALUES,
        )


def test_cli_builds_a_loadable_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "slug": "cli-set",
                "title": "CLI set",
                "description": "Built by the CLI.",
                "loader_class": "CliDataset",
                "citation_key": "cli2026",
                "version": "1",
                "primary_metric": "spearman",
            }
        )
    )
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"train": ["t1"], "val": ["v1", "v2"], "test": ["s1", "s2"]})
    )
    values = tmp_path / "values.csv"
    values.write_text(
        "record_id,target,value\nv1,y,1.0\nv2,y,2.0\ns1,y,1.0\ns2,y,3.0\n"
    )
    out = tmp_path / "datasets"
    out.mkdir()
    main(
        [
            "--datasets-root",
            str(out),
            "--meta",
            str(meta),
            "--splits",
            str(splits),
            "--values",
            str(values),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["slug"] == "cli-set"
    assert printed["primary_metric"] == "spearman"
    loaded = BenchmarkBundle.load(out / "cli-set")
    assert loaded.labels == {
        ("v1", "y"): 1.0,
        ("v2", "y"): 2.0,
        ("s1", "y"): 1.0,
        ("s2", "y"): 3.0,
    }


# ------------------------------------------------------------------- binary task

BINARY_SPLITS = {Split.TRAIN: ["t1"], Split.VAL: ["v1", "v2"], Split.TEST: ["s1", "s2"]}
BINARY_VALUES = {("v1", "e"): 1.0, ("v2", "e"): 0.0, ("s1", "e"): 0.0, ("s2", "e"): 1.0}


def _write_binary(
    root: Path, values: dict[tuple[str, str], float], primary_metric: str = "auroc"
) -> BenchmarkBundle:
    write_bundle(
        root,
        slug="toy-essential",
        title="t",
        description="d",
        loader_class="L",
        citation_key="k",
        version="1",
        task="binary",
        primary_metric=primary_metric,  # type: ignore[arg-type]
        splits=BINARY_SPLITS,
        values=values,
    )
    return BenchmarkBundle.load(root / "toy-essential")


def test_binary_bundle_records_its_task(tmp_path: Path) -> None:
    bundle = _write_binary(tmp_path, BINARY_VALUES)
    assert (bundle.dataset.task, bundle.dataset.primary_metric) == ("binary", "auroc")
    assert bundle.dataset.public().task == "binary"
    assert bundle.labels == BINARY_VALUES
    assert (tmp_path / "toy-essential" / "labels.csv").read_text() == (
        "record_id,split,target,value\n"
        "s1,test,e,0.0\ns2,test,e,1.0\nv1,val,e,1.0\nv2,val,e,0.0\n"
    )


def test_binary_bundle_refuses_other_labels_and_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"label for \('v1', 'e'\) is not 0 or 1"):
        _write_binary(tmp_path, {**BINARY_VALUES, ("v1", "e"): 0.5})
    with pytest.raises(ValueError, match="'pearson' is not a binary metric"):
        _write_binary(tmp_path, BINARY_VALUES, primary_metric="pearson")
    assert not (tmp_path / "toy-essential").exists()


def test_regression_bundle_refuses_a_binary_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'auroc' is not a regression metric"):
        write_bundle(
            tmp_path,
            slug="bad",
            title="t",
            description="d",
            loader_class="L",
            citation_key="k",
            version="1",
            primary_metric="auroc",
            splits=GOOD_SPLITS,
            values=GOOD_VALUES,
        )
    assert not (tmp_path / "bad").exists()
