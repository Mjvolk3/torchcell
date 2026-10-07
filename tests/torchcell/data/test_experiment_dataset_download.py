# tests/torchcell/data/test_experiment_dataset_download.py
# [[tests.torchcell.data.test_experiment_dataset_download]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_experiment_dataset_download.py
"""``ExperimentDataset._download`` through the tc-data endpoint when ``TC_DATA_URL`` is set.

A minimal ``ExperimentDataset`` subclass whose ``download`` and ``process`` both raise
proves that, with the endpoint set, construction unpacks the packaged artifact and never
runs either; with it unset, the publisher path runs as before.
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from torchcell.data.experiment_dataset import ExperimentDataset
from torchcell.datamodels.schema import FitnessExperiment, FitnessExperimentReference
from torchcell.datasets.artifact import ArtifactIndex
from torchcell.datasets.client import DatasetClient
from torchcell.datasets.server import DataKeys, DataServerConfig, create_app
from torchcell.provenance.build_manifest import MANIFEST_FILENAME, BuildManifest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import package_dataset_lmdb as pkg  # type: ignore[import-not-found]  # noqa: E402

KEY = "dataset-key"
SLUG = "smf_fake"


class FakeArtifactDataset(ExperimentDataset):
    """The smallest concrete ExperimentDataset; both build paths raise if reached."""

    @property
    def experiment_class(self) -> type[FitnessExperiment]:
        """Fitness records."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[FitnessExperimentReference]:
        """Fitness references."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """One publisher file, never present in a packaged artifact."""
        return ["raw.txt"]

    def download(self) -> None:
        """The publisher path; reaching it with the endpoint set is the failure."""
        raise AssertionError("publisher download ran")

    def process(self) -> None:
        """The build path; reaching it after an unpack is the failure."""
        raise AssertionError("process ran")

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Unused: no raw data is ever read here."""
        raise NotImplementedError

    def create_experiment(self) -> None:
        """Unused: no records are ever built here."""
        raise NotImplementedError


def _records() -> list[dict[str, Any]]:
    return [
        {
            "experiment": {
                "genotype": {"perturbations": []},
                "phenotype": {"fitness": i},
            },
            "reference": {},
            "publication": {},
        }
        for i in range(3)
    ]


def _build_source(root: Path, loader_class: str = "FakeArtifactDataset") -> Path:
    dataset_dir = root / "src" / "data" / "torchcell" / SLUG
    (dataset_dir / "preprocess").mkdir(parents=True)
    manifest = BuildManifest(
        dataset_name=SLUG,
        loader_class=loader_class,
        loader_module="tests.fake",
        surface_modules=["schema.py"],
        closure={},
        built_at="2026-09-14T07:25:14+00:00",
        hostname="gilahyper",
    )
    (dataset_dir / "preprocess" / MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2)
    )
    (dataset_dir / "processed" / "lmdb").mkdir(parents=True)
    env = lmdb.open(str(dataset_dir / "processed" / "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for i, record in enumerate(_records()):
            txn.put(f"{i}".encode(), pickle.dumps(record))
    env.close()
    return dataset_dir


def _serve(root: Path, dataset_dir: Path) -> TestClient:
    store = root / "store"
    pkg.package_dataset(dataset_dir, store, packaged_at="2026-09-29T10:00:00+00:00")
    raw = root / "raw"
    raw.mkdir()
    config = DataServerConfig(
        store_root=store,
        raw_root=raw,
        genomes_root=root / "genomes",
        objects_root=root / "objects",
        keys=DataKeys.from_pairs(f"t:{KEY}"),
    )
    return TestClient(create_app(config))


def _point_client_at(
    monkeypatch: pytest.MonkeyPatch, http: TestClient, url: str = "http://testserver"
) -> None:
    monkeypatch.setenv("TC_DATA_URL", url)
    monkeypatch.setenv("TC_DATA_API_KEY", KEY)
    monkeypatch.setattr(
        DatasetClient,
        "from_env",
        classmethod(lambda cls, http_=None: cls(url, KEY, http=http)),
    )


def test_endpoint_set_unpacks_the_artifact_and_skips_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _build_source(tmp_path)
    _point_client_at(monkeypatch, _serve(tmp_path, dataset_dir))
    root = tmp_path / "client" / SLUG

    dataset = FakeArtifactDataset(root=str(root))

    row = ArtifactIndex.load(tmp_path / "store" / "index.json").artifacts[0]
    archive = root / "artifacts" / row.archive
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == row.archive_sha256
    assert (root / "processed" / "lmdb" / "data.mdb").read_bytes() == (
        dataset_dir / "processed" / "lmdb" / "data.mdb"
    ).read_bytes()
    assert json.loads((root / "preprocess" / MANIFEST_FILENAME).read_text()) == (
        json.loads((dataset_dir / "preprocess" / MANIFEST_FILENAME).read_text())
    )
    assert len(dataset) == 3
    assert dataset[1] == _records()[1]
    assert not (root / "raw" / "raw.txt").exists()


def test_endpoint_set_with_nothing_compatible_is_an_error_not_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _build_source(tmp_path)
    http = _serve(tmp_path, dataset_dir)
    index_path = tmp_path / "store" / "index.json"
    index = ArtifactIndex.load(index_path)
    deprecated = index.model_copy(
        update={
            "artifacts": [
                index.artifacts[0].model_copy(update={"status": "deprecated"})
            ]
        }
    )
    assert ArtifactIndex.load(deprecated.save(index_path)).artifacts[0].status == (
        "deprecated"
    )
    _point_client_at(monkeypatch, http)
    with pytest.raises(
        RuntimeError,
        match=r"http://testserver has no supported artifact for 'smf_fake' compatible",
    ):
        FakeArtifactDataset(root=str(tmp_path / "client" / SLUG))
    assert not (tmp_path / "client" / SLUG / "processed").exists()


def test_artifact_built_by_another_loader_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _build_source(tmp_path, loader_class="SmfCostanzo2016Dataset")
    _point_client_at(monkeypatch, _serve(tmp_path, dataset_dir))
    with pytest.raises(
        RuntimeError,
        match="was built by SmfCostanzo2016Dataset, not FakeArtifactDataset",
    ):
        FakeArtifactDataset(root=str(tmp_path / "client" / SLUG))


def test_endpoint_unset_takes_the_publisher_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    with pytest.raises(AssertionError, match="publisher download ran"):
        FakeArtifactDataset(root=str(tmp_path / "client" / SLUG))


def test_existing_lmdb_never_contacts_the_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "client" / SLUG
    (root / "processed" / "lmdb").mkdir(parents=True)
    monkeypatch.setenv("TC_DATA_URL", "http://unreachable")

    def refuse(cls: type[DatasetClient], http: Any = None) -> DatasetClient:
        raise AssertionError("endpoint contacted")

    monkeypatch.setattr(DatasetClient, "from_env", classmethod(refuse))
    dataset = FakeArtifactDataset(root=str(root))
    assert dataset.name == "FakeArtifactDataset"
