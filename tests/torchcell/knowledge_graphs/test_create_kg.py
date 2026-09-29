# tests/torchcell/knowledge_graphs/test_create_kg.py
# [[tests.torchcell.knowledge_graphs.test_create_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_create_kg.py
"""``create_kg.main``: the registry-driven BioCypher build, run against stand-ins.

The module reads ``DATA_ROOT`` and the three BioCypher paths at import, so the tests set
the module attributes directly. ``dataset_registry`` maps two names to fake loaders:
``alpha`` (3 records, plain ``root`` + kwargs signature, yaml kwargs ``{"subset_n": 7}``)
and ``beta`` (5 records, declares ``genome`` and ``scerevisiae_graph``, yaml kwargs
null). ``dataset_adapter_map`` maps each loader class to a fake adapter that yields one
node and one edge per record. ``BioCypher``, ``wandb``, ``SCerevisiaeGenome``,
``SCerevisiaeGraph``, ``datetime`` and ``time`` are replaced at the module boundary.

Derived values with ``SLURM_CPUS_PER_TASK`` 6 and ratio 0.25: ``num_workers`` 6,
``io_workers`` ceil(1.5) = 2, ``process_workers`` 4; ``chunk_size`` int(4.0) = 4,
``loader_batch_size`` int(2.0) = 2. Every loader is handed ``io_workers`` = 6 (the total,
not the io share). The wandb group is ``<SLURM_JOB_ID>_<sha256 of the sorted-key JSON of
the config>``. ``time.time`` ticks 0, 1, 2, ... so each measured duration is 1.0. The
output directory is ``<DATA_ROOT>/<BIOCYPHER_OUT_PATH>/2026-09-27_08-30-05``.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import sys
import uuid
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
    reset_instances,
)

create_kg = import_build_module("create_kg")


class FakeAlpha(FakeDataset):
    """Three records; a plain ``root`` + kwargs loader signature."""

    n_records = 3
    instances: list[FakeDataset] = []


class FakeBeta(FakeDataset):
    """Five records; declares ``genome`` and ``scerevisiae_graph``."""

    n_records = 5
    instances: list[FakeDataset] = []

    def __init__(
        self, root: str, io_workers: int, genome: Any, scerevisiae_graph: Any
    ) -> None:
        """Record every kwarg the build script injects."""
        super().__init__(
            root,
            io_workers=io_workers,
            genome=genome,
            scerevisiae_graph=scerevisiae_graph,
        )


class FakeAdapterA(FakeAdapter):
    """Adapter for ``FakeAlpha``."""

    instances: list[FakeAdapter] = []


class FakeAdapterB(FakeAdapter):
    """Adapter for ``FakeBeta``."""

    instances: list[FakeAdapter] = []


class _Recording:
    instances: list[_Recording] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)


class FakeGenome(_Recording):
    """``SCerevisiaeGenome`` stand-in recording its kwargs."""

    instances: list[_Recording] = []


class FakeGraph(_Recording):
    """``SCerevisiaeGraph`` stand-in recording its kwargs."""

    instances: list[_Recording] = []


CFG: dict[str, Any] = {
    "wandb": {"mode": "disabled", "project": "tcdb-test"},
    "adapters": {
        "io_to_total_worker_ratio": 0.25,
        "chunk_size": 4.0,
        "loader_batch_size": 2.0,
    },
    "datasets": {
        "alpha": {"path": "data/torchcell/alpha", "kwargs": {"subset_n": 7}},
        "beta": {"path": "data/torchcell/beta", "kwargs": None},
    },
}


def _group(cfg: dict[str, Any], job_id: str) -> str:
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{job_id}_{digest}"


def _cfg(mapping: dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(mapping)
    assert isinstance(cfg, DictConfig)
    return cfg


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Point the module at ``tmp_path`` and install every stand-in."""
    reset_instances(
        FakeAlpha,
        FakeBeta,
        FakeAdapterA,
        FakeAdapterB,
        FakeGenome,
        FakeGraph,
        FakeBioCypher,
    )
    monkeypatch.chdir(tmp_path)
    root = str(tmp_path / "root")
    wandb = FakeWandb()
    ticking = TickingTime()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setattr(create_kg, "DATA_ROOT", root)
    monkeypatch.setattr(create_kg, "BIOCYPHER_CONFIG_PATH", "bc.yaml")
    monkeypatch.setattr(create_kg, "SCHEMA_CONFIG_PATH", "schema.yaml")
    monkeypatch.setattr(create_kg, "BIOCYPHER_OUT_PATH", "database/biocypher-out")
    monkeypatch.setattr(
        create_kg, "dataset_registry", {"alpha": FakeAlpha, "beta": FakeBeta}
    )
    monkeypatch.setattr(
        create_kg,
        "dataset_adapter_map",
        {FakeAlpha: FakeAdapterA, FakeBeta: FakeAdapterB},
    )
    monkeypatch.setattr(create_kg, "SCerevisiaeGenome", FakeGenome)
    monkeypatch.setattr(create_kg, "SCerevisiaeGraph", FakeGraph)
    monkeypatch.setattr(create_kg, "BioCypher", FakeBioCypher)
    monkeypatch.setattr(create_kg, "wandb", wandb)
    monkeypatch.setattr(create_kg, "datetime", FixedDatetime)
    monkeypatch.setattr(create_kg, "time", ticking)
    return SimpleNamespace(root=root, wandb=wandb, tmp_path=tmp_path)


def test_import_build_module_leaves_the_environment_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh import of ``create_kg`` runs its import-time ``load_dotenv()`` and
    ``SSL_CERT_FILE`` assignment; through the helper the process environment is
    identical before and after, and a new module object was executed.
    """
    before = dict(os.environ)
    monkeypatch.delitem(sys.modules, "torchcell.knowledge_graphs.create_kg")
    module = import_build_module("create_kg")
    assert dict(os.environ) == before
    assert module.__name__ == "torchcell.knowledge_graphs.create_kg"
    assert module is not create_kg


def test_get_num_workers_prefers_slurm_then_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SLURM_CPUS_PER_TASK`` "6" -> 6; unset -> ``mp.cpu_count()`` (patched to 3)."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert create_kg.get_num_workers() == 6
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setattr(create_kg.mp, "cpu_count", lambda: 3)
    assert create_kg.get_num_workers() == 3


def test_main_wires_registry_datasets_through_adapters_into_biocypher(
    build: SimpleNamespace,
) -> None:
    """Both registry datasets are built with ``io_workers`` 6 plus their yaml kwargs;
    ``beta`` triggers the genome and graph special cases (the genome is constructed
    twice, once for the graph and once for the ``genome`` kwarg, both ``overwrite=True``);
    the adapters get the 4/2 worker split, chunk 4 and batch 2; BioCypher receives
    nodes then edges per adapter in registry order, then the import call and the schema
    node; wandb sees the exact payload sequence with every duration 1.0.
    """
    root = build.root
    create_kg.main(_cfg(CFG))

    wandb = build.wandb
    assert wandb.init_kwargs == [
        {
            "mode": "disabled",
            "project": "tcdb-test",
            "config": CFG,
            "group": _group(CFG, "4242"),
            "save_code": True,
        }
    ]
    genome_kwargs = {
        "genome_root": osp.join(root, "data/sgd/genome"),
        "overwrite": True,
    }
    assert [g.kwargs for g in FakeGenome.instances] == [genome_kwargs, genome_kwargs]
    assert [g.kwargs for g in FakeGraph.instances] == [
        {
            "sgd_root": osp.join(root, "data/sgd/genome"),
            "string_root": osp.join(root, "data/string"),
            "tflink_root": osp.join(root, "data/tflink"),
            "genome": FakeGenome.instances[0],
        }
    ]
    (alpha,) = FakeAlpha.instances
    (beta,) = FakeBeta.instances
    assert alpha.root == osp.join(root, "data/torchcell/alpha")
    assert alpha.kwargs == {"subset_n": 7, "io_workers": 6}
    assert beta.root == osp.join(root, "data/torchcell/beta")
    assert beta.kwargs == {
        "io_workers": 6,
        "genome": FakeGenome.instances[1],
        "scerevisiae_graph": FakeGraph.instances[0],
    }
    (adapter_a,) = FakeAdapterA.instances
    (adapter_b,) = FakeAdapterB.instances
    worker_kwargs = {
        "process_workers": 4,
        "io_workers": 2,
        "chunk_size": 4,
        "loader_batch_size": 2,
    }
    assert adapter_a.kwargs == {"dataset": alpha, **worker_kwargs}
    assert adapter_b.kwargs == {"dataset": beta, **worker_kwargs}

    (bc,) = FakeBioCypher.instances
    assert bc.kwargs == {
        "output_directory": osp.join(root, "database/biocypher-out", TIME_STR),
        "biocypher_config_path": "bc.yaml",
        "schema_config_path": "schema.yaml",
    }
    assert bc.calls == [
        ("write_nodes", [("node", "FakeAdapterA", i) for i in range(3)]),
        ("write_edges", [("edge", "FakeAdapterA", i) for i in range(3)]),
        ("write_nodes", [("node", "FakeAdapterB", i) for i in range(5)]),
        ("write_edges", [("edge", "FakeAdapterB", i) for i in range(5)]),
        ("write_import_call",),
        ("write_schema_info", True),
    ]
    assert wandb.logged == [
        {"slurm_job_id": "4242"},
        {"biocypher-out": TIME_STR},
        {"num_workers": 6, "io_workers": 2, "process_workers": 4},
        {"FakeAlpha_time(s)": 1.0},
        {"FakeAlpha_len": 3},
        {"FakeBeta_time(s)": 1.0},
        {"FakeBeta_len": 5},
        {"FakeAdapterA_write_nodes_time(s)": 1.0},
        {"FakeAdapterA_write_edges_time": 1.0},
        {"FakeAdapterB_write_nodes_time(s)": 1.0},
        {"FakeAdapterB_write_edges_time": 1.0},
    ]
    assert wandb.finish_calls == 1


def test_recorded_script_path_hardcodes_biocypher_out(build: SimpleNamespace) -> None:
    """Finding: create_kg.py line 195 joins the literal "biocypher-out" into the path it
    writes to ``biocypher_file_name.txt``, while the BioCypher output directory (line 76)
    is built from ``BIOCYPHER_OUT_PATH``. With ``BIOCYPHER_OUT_PATH`` set to
    "database/biocypher-out" the recorded script path does not point into the directory
    the CSVs and import call were written to.
    """
    create_kg.main(_cfg(CFG))
    (bc,) = FakeBioCypher.instances
    assert bc.kwargs["output_directory"] == osp.join(
        build.root, "database/biocypher-out", TIME_STR
    )
    recorded = (build.tmp_path / "biocypher_file_name.txt").read_text()
    assert recorded == f"biocypher-out/{TIME_STR}/neo4j-admin-import-call.sh"
    assert not recorded.startswith("database/")


def test_group_falls_back_to_a_uuid_without_a_slurm_job(
    build: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``SLURM_JOB_ID`` the group prefix and the logged id are ``uuid4()``."""
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setattr(create_kg.uuid, "uuid4", lambda: uuid.UUID(int=7))
    cfg = {**CFG, "datasets": {"alpha": CFG["datasets"]["alpha"]}}
    create_kg.main(_cfg(cfg))
    fixed = "00000000-0000-0000-0000-000000000007"
    assert build.wandb.init_kwargs[0]["group"] == _group(cfg, fixed)
    assert build.wandb.logged[0] == {"slurm_job_id": fixed}
    assert FakeBeta.instances == []
    assert FakeGenome.instances == []
