# tests/torchcell/knowledge_graphs/test_gene_interactions_scerevisae_kg.py
# [[tests.torchcell.knowledge_graphs.test_gene_interactions_scerevisae_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_gene_interactions_scerevisae_kg.py
"""``gene_interactions_scerevisae_kg.main``: the hardcoded Kuzmin 2018 interaction build.

The script's dataset list is fixed in code: ``DmiKuzmin2018Dataset`` at
``data/torchcell/dmi_kuzmin2018`` then ``TmiKuzmin2018Dataset`` at
``data/torchcell/tmi_kuzmin2018``, each with ``io_workers`` = the total worker count and
no other kwargs (the Costanzo DMI entry is commented out and must not be built). Both
loader names and both adapter names are module attributes looked up at call time, so the
tests bind them to fakes: a 2-record DMI loader, a 3-record TMI loader, and adapters
yielding one node and one edge per record. ``BioCypher``, ``wandb``, ``load_dotenv``,
``datetime`` and ``time`` are replaced the same way.

Derived values with ``SLURM_CPUS_PER_TASK`` 5 and ratio 0.2: ``io_workers`` ceil(1.0) =
1, ``process_workers`` 4; chunk int(2.0) = 2, batch int(2.0) = 2. ``get_num_workers``
prints the raw variable before returning. Durations are 1.0 (ticking clock).
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

kg = import_build_module("gene_interactions_scerevisae_kg")


class FakeDmi(FakeDataset):
    """Stands in for ``DmiKuzmin2018Dataset``: two records."""

    n_records = 2
    instances: list[FakeDataset] = []


class FakeTmi(FakeDataset):
    """Stands in for ``TmiKuzmin2018Dataset``: three records."""

    n_records = 3
    instances: list[FakeDataset] = []


class FakeDmiCostanzo(FakeDataset):
    """Stands in for ``DmiCostanzo2016Dataset``, which the script must not build."""

    n_records = 1
    instances: list[FakeDataset] = []


class FakeDmiAdapter(FakeAdapter):
    """Stands in for ``DmiKuzmin2018Adapter``."""

    instances: list[FakeAdapter] = []


class FakeTmiAdapter(FakeAdapter):
    """Stands in for ``TmiKuzmin2018Adapter``."""

    instances: list[FakeAdapter] = []


CFG: dict[str, Any] = {
    "wandb": {"mode": "disabled", "project": "tcdb-test"},
    "adapters": {
        "io_to_total_worker_ratio": 0.2,
        "chunk_size": 2.0,
        "loader_batch_size": 2.0,
    },
}


def _cfg(mapping: dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(mapping)
    assert isinstance(cfg, DictConfig)
    return cfg


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Point the script at ``tmp_path`` and install every stand-in."""
    reset_instances(FakeDmi, FakeTmi, FakeDmiCostanzo, FakeDmiAdapter, FakeTmiAdapter)
    reset_instances(FakeBioCypher)
    monkeypatch.chdir(tmp_path)
    root = str(tmp_path / "root")
    wandb = FakeWandb()
    dotenv_calls: list[tuple[Any, ...]] = []
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    monkeypatch.setenv("SLURM_JOB_ID", "31")
    monkeypatch.setenv("DATA_ROOT", root)
    monkeypatch.setenv("BIOCYPHER_CONFIG_PATH", "bc.yaml")
    monkeypatch.setenv("SCHEMA_CONFIG_PATH", "schema.yaml")
    monkeypatch.setenv("BIOCYPHER_OUT_PATH", "biocypher-out")
    monkeypatch.setattr(kg, "load_dotenv", lambda *args: dotenv_calls.append(args))
    monkeypatch.setattr(kg, "DmiKuzmin2018Dataset", FakeDmi)
    monkeypatch.setattr(kg, "TmiKuzmin2018Dataset", FakeTmi)
    monkeypatch.setattr(kg, "DmiCostanzo2016Dataset", FakeDmiCostanzo)
    monkeypatch.setattr(kg, "DmiKuzmin2018Adapter", FakeDmiAdapter)
    monkeypatch.setattr(kg, "TmiKuzmin2018Adapter", FakeTmiAdapter)
    monkeypatch.setattr(kg, "BioCypher", FakeBioCypher)
    monkeypatch.setattr(kg, "wandb", wandb)
    monkeypatch.setattr(kg, "datetime", FixedDatetime)
    monkeypatch.setattr(kg, "time", TickingTime())
    return SimpleNamespace(
        root=root, wandb=wandb, tmp_path=tmp_path, dotenv_calls=dotenv_calls
    )


def test_get_num_workers_prints_the_variable_and_falls_back_to_cpu_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A value of "5" gives 5 after printing ``SLURM_CPUS_PER_TASK: 5``; unset prints
    ``None`` and returns ``mp.cpu_count()`` (patched to 12).
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    assert kg.get_num_workers() == 5
    assert capsys.readouterr().out == "SLURM_CPUS_PER_TASK: 5\n"
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setattr(kg.mp, "cpu_count", lambda: 12)
    assert kg.get_num_workers() == 12
    assert capsys.readouterr().out == "SLURM_CPUS_PER_TASK: None\n"


def test_main_builds_dmi_then_tmi_and_never_the_commented_out_costanzo(
    build: SimpleNamespace,
) -> None:
    """Two loaders, DMI then TMI, each with ``{"io_workers": 5}``; two adapters with the
    4/1 split, chunk 2, batch 2; BioCypher gets nodes then edges per adapter in that
    order, then the import call and the schema node; the Costanzo DMI loader is never
    constructed; wandb logs the exact payload sequence.
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
            "group": f"31_{digest}",
            "save_code": True,
        }
    ]
    assert wandb.run.log_code_calls == [module_dir(kg)]
    (dmi,) = FakeDmi.instances
    (tmi,) = FakeTmi.instances
    assert FakeDmiCostanzo.instances == []
    assert dmi.root == osp.join(root, "data/torchcell/dmi_kuzmin2018")
    assert tmi.root == osp.join(root, "data/torchcell/tmi_kuzmin2018")
    assert dmi.kwargs == {"io_workers": 5}
    assert tmi.kwargs == {"io_workers": 5}
    worker_kwargs = {
        "process_workers": 4,
        "io_workers": 1,
        "chunk_size": 2,
        "loader_batch_size": 2,
    }
    (dmi_adapter,) = FakeDmiAdapter.instances
    (tmi_adapter,) = FakeTmiAdapter.instances
    assert dmi_adapter.kwargs == {"dataset": dmi, **worker_kwargs}
    assert tmi_adapter.kwargs == {"dataset": tmi, **worker_kwargs}
    (bc,) = FakeBioCypher.instances
    assert bc.kwargs == {
        "output_directory": osp.join(root, "biocypher-out", TIME_STR),
        "biocypher_config_path": "bc.yaml",
        "schema_config_path": "schema.yaml",
    }
    assert bc.calls == [
        ("write_nodes", [("node", "FakeDmiAdapter", 0), ("node", "FakeDmiAdapter", 1)]),
        ("write_edges", [("edge", "FakeDmiAdapter", 0), ("edge", "FakeDmiAdapter", 1)]),
        ("write_nodes", [("node", "FakeTmiAdapter", i) for i in range(3)]),
        ("write_edges", [("edge", "FakeTmiAdapter", i) for i in range(3)]),
        ("write_import_call",),
        ("write_schema_info", True),
    ]
    assert wandb.logged == [
        {"slurm_job_id": "31"},
        {"biocypher-out": TIME_STR},
        {"num_workers": 5, "io_workers": 1, "process_workers": 4},
        {"FakeDmi_time(s)": 1.0},
        {"FakeTmi_time(s)": 1.0},
        {"FakeDmiAdapter_write_nodes_time(s)": 1.0},
        {"FakeDmiAdapter_write_edges_time": 1.0},
        {"FakeTmiAdapter_write_nodes_time(s)": 1.0},
        {"FakeTmiAdapter_write_edges_time": 1.0},
    ]
    assert wandb.finish_calls == 1
    assert (build.tmp_path / "biocypher_file_name.txt").read_text() == (
        f"biocypher-out/{TIME_STR}/neo4j-admin-import-call.sh"
    )
