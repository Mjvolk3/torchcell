# tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg.py
# [[tests.torchcell.knowledge_graphs.test_create_scerevisiae_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg.py
"""``create_scerevisiae_kg.main``: the hardcoded Costanzo 2016 double-mutant build.

The script's dataset list is fixed in code: one ``DmfCostanzo2016Dataset`` at
``data/torchcell/dmf_costanzo2016`` with ``io_workers`` = the total worker count and
``batch_size`` int(1e3) = 1000, adapted by ``DmfCostanzo2016Adapter``. Both names are
module attributes looked up at call time, so the tests bind them to a fake loader (4
records) and a fake adapter (one node and one edge per record). ``BioCypher``, ``wandb``,
``load_dotenv``, ``datetime`` and ``time`` are replaced the same way; the four
environment variables are set with ``monkeypatch.setenv`` because ``main`` reads them.

Derived values with ``SLURM_CPUS_PER_TASK`` 4 and ratio 0.5: ``io_workers`` 2,
``process_workers`` 2; chunk int(3.0) = 3, batch int(1.0) = 1. ``wandb.run.log_code``
receives the directory of the module file. Durations are 1.0 (ticking clock); the output
directory is ``<DATA_ROOT>/<BIOCYPHER_OUT_PATH>/2026-09-27_08-30-05``.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from tests.torchcell.knowledge_graphs._kg_build_fakes import (
    TIME_STR,
    FakeAdapter,
    FakeBioCypher,
    FakeDataset,
    FakeWandb,
    FixedDatetime,
    TickingTime,
    import_build_module,
    module_dir,
    reset_instances,
)

kg = import_build_module("create_scerevisiae_kg")


class FakeDmf(FakeDataset):
    """Stands in for ``DmfCostanzo2016Dataset``: four records."""

    n_records = 4
    instances: list[FakeDataset] = []


class FakeDmfAdapter(FakeAdapter):
    """Stands in for ``DmfCostanzo2016Adapter``."""

    instances: list[FakeAdapter] = []


CFG: dict[str, Any] = {
    "wandb": {"mode": "disabled", "project": "tcdb-test"},
    "adapters": {
        "io_to_total_worker_ratio": 0.5,
        "chunk_size": 3.0,
        "loader_batch_size": 1.0,
    },
}


def _cfg(mapping: dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(mapping)
    assert isinstance(cfg, DictConfig)
    return cfg


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Point the script at ``tmp_path`` and install every stand-in."""
    reset_instances(FakeDmf, FakeDmfAdapter, FakeBioCypher)
    monkeypatch.chdir(tmp_path)
    root = str(tmp_path / "root")
    wandb = FakeWandb()
    dotenv_calls: list[tuple[Any, ...]] = []
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    monkeypatch.setenv("DATA_ROOT", root)
    monkeypatch.setenv("BIOCYPHER_CONFIG_PATH", "bc.yaml")
    monkeypatch.setenv("SCHEMA_CONFIG_PATH", "schema.yaml")
    monkeypatch.setenv("BIOCYPHER_OUT_PATH", "biocypher-out")
    monkeypatch.setattr(kg, "load_dotenv", lambda *args: dotenv_calls.append(args))
    monkeypatch.setattr(kg, "DmfCostanzo2016Dataset", FakeDmf)
    monkeypatch.setattr(kg, "DmfCostanzo2016Adapter", FakeDmfAdapter)
    monkeypatch.setattr(kg, "BioCypher", FakeBioCypher)
    monkeypatch.setattr(kg, "wandb", wandb)
    monkeypatch.setattr(kg, "datetime", FixedDatetime)
    monkeypatch.setattr(kg, "time", TickingTime())
    return SimpleNamespace(
        root=root, wandb=wandb, tmp_path=tmp_path, dotenv_calls=dotenv_calls
    )


def test_get_num_workers_prefers_slurm_then_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SLURM_CPUS_PER_TASK`` "4" -> 4; unset -> ``mp.cpu_count()`` (patched to 9)."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert kg.get_num_workers() == 4
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setattr(kg.mp, "cpu_count", lambda: 9)
    assert kg.get_num_workers() == 9


def test_main_builds_the_costanzo_dmf_dataset_and_writes_the_graph(
    build: SimpleNamespace,
) -> None:
    """One loader at ``<DATA_ROOT>/data/torchcell/dmf_costanzo2016`` with
    ``{"io_workers": 4, "batch_size": 1000}``; one adapter with the 2/2 split, chunk 3,
    batch 1; BioCypher gets its nodes, its edges, the import call and the schema node;
    wandb logs the job id, the output stem, the worker split, the load time and the two
    write times; the code directory is logged; the script path is recorded.
    """
    root = build.root
    kg.main(_cfg(CFG))

    assert build.dotenv_calls == [()]
    wandb = build.wandb
    digest = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode("utf-8")).hexdigest()
    assert wandb.init_kwargs == [
        {
            "mode": "disabled",
            "project": "tcdb-test",
            "config": CFG,
            "group": f"777_{digest}",
            "save_code": True,
        }
    ]
    assert wandb.run.log_code_calls == [module_dir(kg)]
    (dataset,) = FakeDmf.instances
    assert dataset.root == osp.join(root, "data/torchcell/dmf_costanzo2016")
    assert dataset.kwargs == {"io_workers": 4, "batch_size": 1000}
    (adapter,) = FakeDmfAdapter.instances
    assert adapter.kwargs == {
        "dataset": dataset,
        "process_workers": 2,
        "io_workers": 2,
        "chunk_size": 3,
        "loader_batch_size": 1,
    }
    (bc,) = FakeBioCypher.instances
    assert bc.kwargs == {
        "output_directory": osp.join(root, "biocypher-out", TIME_STR),
        "biocypher_config_path": "bc.yaml",
        "schema_config_path": "schema.yaml",
    }
    assert bc.calls == [
        ("write_nodes", [("node", "FakeDmfAdapter", i) for i in range(4)]),
        ("write_edges", [("edge", "FakeDmfAdapter", i) for i in range(4)]),
        ("write_import_call",),
        ("write_schema_info", True),
    ]
    assert wandb.logged == [
        {"slurm_job_id": "777"},
        {"biocypher-out": TIME_STR},
        {"num_workers": 4, "io_workers": 2, "process_workers": 2},
        {"FakeDmf_time(s)": 1.0},
        {"FakeDmfAdapter_write_nodes_time(s)": 1.0},
        {"FakeDmfAdapter_write_edges_time": 1.0},
    ]
    assert wandb.finish_calls == 1
    assert (build.tmp_path / "biocypher_file_name.txt").read_text() == (
        f"biocypher-out/{TIME_STR}/neo4j-admin-import-call.sh"
    )
