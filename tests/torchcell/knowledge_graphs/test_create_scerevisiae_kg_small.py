# tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg_small.py
# [[tests.torchcell.knowledge_graphs.test_create_scerevisiae_kg_small]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg_small.py
"""``create_scerevisiae_kg_small.main``: membership, subsetting, skipping, and the two
import modes, run against stand-ins.

``dataset_adapter_map`` is replaced by three fake loaders keyed to fake adapters:
``FakeAlpha`` (5 records, default root ``data/torchcell/alpha``), ``FakeBeta`` (3
records, ``data/torchcell/beta``, declares ``genome`` and ``scerevisiae_graph``) and
``FakeGamma`` (``data/torchcell/gamma``, whose ``processed/lmdb`` is NOT created, so a
full build skips it and an incremental one refuses). ``select_indices`` is a recorder
returning ``[0, 2, 4]``; ``subset_dataset`` is the real one. ``BioCypher``, ``wandb``,
``SCerevisiaeGenome``, ``SCerevisiaeGraph``, ``load_dotenv``,
``prepare_incremental_import``, ``datetime`` and ``time`` are replaced at the module
boundary; the BioCypher config YAML is a real file naming database ``torchcell-test``.

Derived values with ``SLURM_CPUS_PER_TASK`` 8 and ratio 0.25: ``io_workers`` 2,
``process_workers`` 6; chunk int(4.0) = 4, batch int(2.0) = 2. In the full build the
config caps every dataset at 2 records except ``FakeBeta`` (per-dataset null = all) and
prefilters ``FakeAlpha`` to singles: ``subset_dataset(alpha, 2, 42, [0, 2, 4])`` draws
``sorted(random.Random(42).sample([0, 2, 4], 2))`` = [0, 4], so the alpha adapter
emits 2 nodes and 2 edges, beta 3 and 3, totals 5 and 5. Durations are 1.0 (ticking
clock); the output directory is ``<DATA_ROOT>/biocypher-out/2026-09-27_08-30-05``.

Genome injection is by parameter NAME (2026.10.07): with every genome class replaced by a
recording subclass of the real one (``tests/torchcell/datasets/_genome_injection_fakes.py``),
a loader naming ``genome`` receives the shared ``SCerevisiaeGenome``, one naming
``ecoli_genome`` an ``EcoliK12Genome`` of its ``REFERENCE_STRAIN``, and a yeast-only build
constructs no bacterial genome.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from tests.torchcell.datasets._genome_injection_fakes import (
    FakeYeastGenome,
    install_bacterial_fakes,
)
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
from torchcell.knowledge_graphs.head_ontology import (
    BIOLINK_SOURCE_URL,
    REPO_ONTOLOGY_PATH,
    HeadOntologyError,
)
from torchcell.knowledge_graphs.incremental_import import INCREMENTAL_CALL_FILENAME
from torchcell.knowledge_graphs.subset import RecordFilter
from torchcell.sequence.genome.ecoli.k12 import EcoliK12BW25113Genome, EcoliK12Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

ks = import_build_module("create_scerevisiae_kg_small")

# The repo's sha256-pinned Biolink mirror (issue #619): the build verifies the config's
# head ontology against its provenance record before constructing BioCypher.
MIRRORED_ONTOLOGY = Path(__file__).resolve().parents[3] / REPO_ONTOLOGY_PATH


class FakeAlpha(FakeDataset):
    """Five records at ``data/torchcell/alpha``; plain loader signature."""

    n_records = 5
    instances: list[FakeDataset] = []

    def __init__(self, root: str = "data/torchcell/alpha", io_workers: int = 1) -> None:
        """Record the root and ``io_workers`` the build script passes."""
        super().__init__(root, io_workers=io_workers)


class FakeBeta(FakeDataset):
    """Three records at ``data/torchcell/beta``; declares genome and graph."""

    n_records = 3
    instances: list[FakeDataset] = []

    def __init__(
        self,
        root: str = "data/torchcell/beta",
        io_workers: int = 1,
        genome: Any = None,
        scerevisiae_graph: Any = None,
    ) -> None:
        """Record every kwarg the build script injects."""
        super().__init__(
            root,
            io_workers=io_workers,
            genome=genome,
            scerevisiae_graph=scerevisiae_graph,
        )


class FakeGamma(FakeDataset):
    """Two records at ``data/torchcell/gamma``, whose LMDB is never staged."""

    n_records = 2
    instances: list[FakeDataset] = []

    def __init__(self, root: str = "data/torchcell/gamma", io_workers: int = 1) -> None:
        """Record the root and ``io_workers`` the build script passes."""
        super().__init__(root, io_workers=io_workers)


class FakeAdapterA(FakeAdapter):
    """Adapter for ``FakeAlpha``."""

    instances: list[FakeAdapter] = []


class FakeAdapterB(FakeAdapter):
    """Adapter for ``FakeBeta``."""

    instances: list[FakeAdapter] = []


class FakeAdapterC(FakeAdapter):
    """Adapter for ``FakeGamma``."""

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


BASE: dict[str, Any] = {
    "wandb": {"mode": "disabled", "project": "tcdb-test"},
    "adapters": {
        "io_to_total_worker_ratio": 0.25,
        "chunk_size": 4.0,
        "loader_batch_size": 2.0,
    },
}
FULL_CFG: dict[str, Any] = {
    **BASE,
    "subset": {
        "size": 2,
        "seed": 42,
        "per_dataset": {"FakeBeta": None},
        "prefilter": {"FakeAlpha": {"n_perturbations": [1]}},
    },
}
INCREMENT_CFG: dict[str, Any] = {
    **BASE,
    "subset": {"size": None, "seed": 42, "per_dataset": {}, "prefilter": {}},
    "datasets": ["FakeAlpha"],
    "import_mode": "incremental",
}
WORKER_KWARGS = {
    "process_workers": 6,
    "io_workers": 2,
    "chunk_size": 4,
    "loader_batch_size": 2,
}
PRELUDE = [
    {"slurm_job_id": "99"},
    {"biocypher-out": TIME_STR},
    {"num_workers": 8, "io_workers": 2, "process_workers": 6},
]


def _cfg(mapping: dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(mapping)
    assert isinstance(cfg, DictConfig)
    return cfg


def _plan(risk: bool) -> SimpleNamespace:
    return SimpleNamespace(
        node_labels=["Dataset", "Experiment"],
        edge_types=["ExperimentMemberOf"],
        out_dir=Path("/x/out"),
        analysis=SimpleNamespace(
            n_node_ids=7,
            n_external_ids=1,
            n_edges_between_external=1 if risk else 0,
            has_duplicate_edge_risk=risk,
            edges_between_external_sample=[["GenomeMemberOf", "G", "X"]]
            if risk
            else [],
        ),
    )


class FakeSampler:
    """Stand-in for ``ResourceSampler``: the real one reads the cgroup v2 memory files,
    which exist only inside a slurm job's cgroup, and samples in a thread.
    """

    instances: list[FakeSampler] = []

    def __init__(self, out_dir: Path, interval: float) -> None:
        """Record where the script would write telemetry and how often."""
        self.out_dir = out_dir
        self.interval = interval
        self.calls: list[str] = []
        FakeSampler.instances.append(self)

    def start(self) -> None:
        """Record the start."""
        self.calls.append("start")

    def stop(self) -> list[Any]:
        """Record the stop; no phase timings."""
        self.calls.append("stop")
        return []


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Point the script at ``tmp_path``, stage alpha and beta LMDBs, install stand-ins."""
    reset_instances(FakeAlpha, FakeBeta, FakeGamma, FakeAdapterA, FakeAdapterB)
    reset_instances(FakeAdapterC, FakeGenome, FakeGraph, FakeBioCypher, FakeSampler)
    monkeypatch.chdir(tmp_path)
    root = str(tmp_path / "root")
    for name in ("alpha", "beta"):
        (tmp_path / "root" / "data" / "torchcell" / name / "processed" / "lmdb").mkdir(
            parents=True
        )
    bc_config = tmp_path / "biocypher_config.yaml"
    bc_config.write_text(
        "biocypher:\n  head_ontology:\n"
        f"    url: {MIRRORED_ONTOLOGY}\n    root_node: entity\n"
        "neo4j:\n  database_name: torchcell-test\n"
    )
    wandb = FakeWandb()
    state = SimpleNamespace(
        root=root,
        wandb=wandb,
        tmp_path=tmp_path,
        dotenv_calls=[],
        select_calls=[],
        prepare_calls=[],
        plan=_plan(risk=False),
    )

    def select_indices(lmdb_path: str, record_filter: RecordFilter) -> list[int]:
        state.select_calls.append((lmdb_path, record_filter))
        return [0, 2, 4]

    def prepare_incremental_import(out_dir: Path, database: str) -> SimpleNamespace:
        state.prepare_calls.append((out_dir, database))
        plan: SimpleNamespace = state.plan
        return plan

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.delenv("TCDB_TAGS", raising=False)
    monkeypatch.delenv("TCDB_JOB_TYPE", raising=False)
    monkeypatch.setenv("DATA_ROOT", root)
    monkeypatch.setenv("BIOCYPHER_CONFIG_PATH", str(bc_config))
    monkeypatch.setenv("SCHEMA_CONFIG_PATH", "schema.yaml")
    monkeypatch.setenv("BIOCYPHER_OUT_PATH", "biocypher-out")
    monkeypatch.setattr(
        ks, "load_dotenv", lambda *args: state.dotenv_calls.append(args)
    )
    monkeypatch.setattr(
        ks,
        "dataset_adapter_map",
        {FakeAlpha: FakeAdapterA, FakeBeta: FakeAdapterB, FakeGamma: FakeAdapterC},
    )
    monkeypatch.setattr(ks, "SCerevisiaeGenome", FakeGenome)
    monkeypatch.setattr(ks, "SCerevisiaeGraph", FakeGraph)
    monkeypatch.setattr(ks, "BioCypher", FakeBioCypher)
    monkeypatch.setattr(ks, "ResourceSampler", FakeSampler)
    monkeypatch.setattr(ks, "wandb", wandb)
    monkeypatch.setattr(ks, "datetime", FixedDatetime)
    monkeypatch.setattr(ks, "time", TickingTime())
    monkeypatch.setattr(ks, "select_indices", select_indices)
    monkeypatch.setattr(ks, "prepare_incremental_import", prepare_incremental_import)
    return state


def test_get_num_workers_falls_back_to_ten_not_the_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: the docstring (create_scerevisiae_kg_small.py line 50) says "the number
    of CPUs allocated by SLURM", and with ``SLURM_CPUS_PER_TASK`` set it is ("8" -> 8),
    but with it unset the function returns the constant 10 (line 57; the
    ``mp.cpu_count()`` fallback of the sibling scripts is commented out), whatever the
    machine has.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert ks.get_num_workers() == 8
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert ks.get_num_workers() == 10


def test_count_while_writing_counts_what_the_writer_consumes() -> None:
    """Items stream through in order and are counted as consumed: a writer that drains
    the generator counts 3; one that takes a single item counts 1.
    """
    sink: list[str] = []
    assert ks._count_while_writing(sink.extend, iter(["a", "b", "c"])) == 3
    assert sink == ["a", "b", "c"]
    assert ks._count_while_writing(next, iter(["a", "b", "c"])) == 1


def test_full_build_prefilters_caps_skips_and_writes_the_import_call(
    build: SimpleNamespace,
) -> None:
    """Alpha is prefiltered (``select_indices`` on its LMDB with the singles filter) and
    capped to 2 of the pool [0, 2, 4] -> records [0, 4]; beta's per-dataset null takes
    all 3 records unchanged; gamma has no LMDB and is skipped, not built. Genome and
    graph are built once (``overwrite=False``) and injected only into beta. BioCypher
    gets nodes then edges per adapter, then the import call and schema node; wandb sees
    the per-adapter counts and running totals, then the final totals.
    """
    root = build.root
    ks.main(_cfg(FULL_CFG))

    assert build.dotenv_calls == [("/.env",)]
    wandb = build.wandb
    digest = hashlib.sha256(
        json.dumps(FULL_CFG, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert wandb.init_kwargs == [
        {
            "mode": "disabled",
            "project": "tcdb-test",
            "config": FULL_CFG,
            "group": f"99_{digest}",
            # No config tags and no TCDB_TAGS / TCDB_JOB_TYPE in the environment: the
            # untagged production build.
            "tags": [],
            "job_type": "build",
        }
    ]
    assert wandb.run.log_code_calls == [module_dir(ks)]
    assert [g.kwargs for g in FakeGenome.instances] == [
        {"genome_root": osp.join(root, "data/sgd/genome"), "overwrite": False}
    ]
    assert [g.kwargs for g in FakeGraph.instances] == [
        {
            "sgd_root": osp.join(root, "data/sgd/genome"),
            "string_root": osp.join(root, "data/string"),
            "tflink_root": osp.join(root, "data/tflink"),
            "genome": FakeGenome.instances[0],
        }
    ]
    alpha_root = osp.join(root, "data/torchcell/alpha")
    assert build.select_calls == [
        (osp.join(alpha_root, "processed", "lmdb"), RecordFilter(n_perturbations=[1]))
    ]
    (alpha,) = FakeAlpha.instances
    (beta,) = FakeBeta.instances
    assert FakeGamma.instances == []
    assert (alpha.root, alpha.kwargs) == (alpha_root, {"io_workers": 8})
    assert (beta.root, beta.kwargs) == (
        osp.join(root, "data/torchcell/beta"),
        {
            "io_workers": 8,
            "genome": FakeGenome.instances[0],
            "scerevisiae_graph": FakeGraph.instances[0],
        },
    )
    (adapter_a,) = FakeAdapterA.instances
    (adapter_b,) = FakeAdapterB.instances
    assert FakeAdapterC.instances == []
    alpha_view = adapter_a.kwargs["dataset"]
    assert alpha_view is not alpha
    assert alpha_view.selected == [0, 4]
    assert adapter_a.kwargs == {"dataset": alpha_view, **WORKER_KWARGS}
    assert adapter_b.kwargs == {"dataset": beta, **WORKER_KWARGS}

    (bc,) = FakeBioCypher.instances
    assert bc.kwargs == {
        "output_directory": osp.join(root, "biocypher-out", TIME_STR),
        "biocypher_config_path": str(build.tmp_path / "biocypher_config.yaml"),
        "schema_config_path": "schema.yaml",
    }
    assert bc.calls == [
        ("write_nodes", [("node", "FakeAdapterA", 0), ("node", "FakeAdapterA", 1)]),
        ("write_edges", [("edge", "FakeAdapterA", 0), ("edge", "FakeAdapterA", 1)]),
        ("write_nodes", [("node", "FakeAdapterB", i) for i in range(3)]),
        ("write_edges", [("edge", "FakeAdapterB", i) for i in range(3)]),
        ("write_import_call",),
        ("write_schema_info", True),
    ]
    assert wandb.logged == [
        *PRELUDE,
        {"FakeAlpha_time(s)": 1.0},
        {"FakeAlpha_len": 2},
        {"FakeBeta_time(s)": 1.0},
        {"FakeBeta_len": 3},
        {"n_adapters": 2, "skipped_datasets": ["FakeGamma"]},
        {"adapter_index": 0, "current_adapter": "FakeAdapterA"},
        {
            "FakeAdapterA_write_nodes_time(s)": 1.0,
            "FakeAdapterA_n_nodes": 2,
            "total_nodes": 2,
        },
        {
            "FakeAdapterA_write_edges_time": 1.0,
            "FakeAdapterA_n_edges": 2,
            "total_edges": 2,
        },
        {"adapter_index": 1, "current_adapter": "FakeAdapterB"},
        {
            "FakeAdapterB_write_nodes_time(s)": 1.0,
            "FakeAdapterB_n_nodes": 3,
            "total_nodes": 5,
        },
        {
            "FakeAdapterB_write_edges_time": 1.0,
            "FakeAdapterB_n_edges": 3,
            "total_edges": 5,
        },
        {
            "total_nodes": 5,
            "total_edges": 5,
            # The ticking clock advances 1 s per time() call after build_t0: two per
            # dataset build, two per adapter node pass, two per edge pass, one at the
            # end: 2 * 2 + 2 * 2 + 2 * 2 + 1.
            "generation_wall_s": 13.0,
            # The fake sampler yields no phase timings.
            "generation_cpu_core_s": 0,
            "generation_mem_peak_gb": 0.0,
        },
    ]
    assert wandb.finish_calls == 1
    (sampler,) = FakeSampler.instances
    assert sampler.out_dir == Path(
        osp.join(root, "biocypher-out", TIME_STR, "telemetry")
    )
    assert sampler.calls == ["start", "stop"]
    assert build.prepare_calls == []
    assert (build.tmp_path / "biocypher_file_name.txt").read_text() == (
        f"biocypher-out/{TIME_STR}/neo4j-admin-import-call.sh"
    )


def test_incremental_build_emits_one_dataset_and_prepares_the_increment(
    build: SimpleNamespace,
) -> None:
    """``datasets: [FakeAlpha]`` builds alpha alone, in full (5 records, no
    ``select_indices`` call); BioCypher writes its nodes and edges but NOT the import
    call or the schema node; ``prepare_incremental_import`` gets the output directory
    and the database name from the BioCypher config YAML; wandb logs the three
    increment counts; the recorded script is the incremental call file.
    """
    root = build.root
    ks.main(_cfg(INCREMENT_CFG))

    assert FakeBeta.instances == []
    assert FakeGamma.instances == []
    assert build.select_calls == []
    (adapter_a,) = FakeAdapterA.instances
    (alpha,) = FakeAlpha.instances
    assert adapter_a.kwargs["dataset"] is alpha
    (bc,) = FakeBioCypher.instances
    assert bc.calls == [
        ("write_nodes", [("node", "FakeAdapterA", i) for i in range(5)]),
        ("write_edges", [("edge", "FakeAdapterA", i) for i in range(5)]),
    ]
    assert build.prepare_calls == [
        (Path(osp.join(root, "biocypher-out", TIME_STR)), "torchcell-test")
    ]
    assert build.wandb.logged == [
        *PRELUDE,
        {"FakeAlpha_time(s)": 1.0},
        {"FakeAlpha_len": 5},
        {"n_adapters": 1, "skipped_datasets": []},
        {"adapter_index": 0, "current_adapter": "FakeAdapterA"},
        {
            "FakeAdapterA_write_nodes_time(s)": 1.0,
            "FakeAdapterA_n_nodes": 5,
            "total_nodes": 5,
        },
        {
            "FakeAdapterA_write_edges_time": 1.0,
            "FakeAdapterA_n_edges": 5,
            "total_edges": 5,
        },
        {
            "total_nodes": 5,
            "total_edges": 5,
            # One dataset build, one adapter: 2 + 2 + 2 + 1 ticks after build_t0.
            "generation_wall_s": 7.0,
            "generation_cpu_core_s": 0,
            "generation_mem_peak_gb": 0.0,
        },
        {
            "increment_n_node_ids": 7,
            "increment_n_external_ids": 1,
            "increment_n_edges_between_external": 0,
        },
    ]
    assert build.wandb.finish_calls == 1
    assert (build.tmp_path / "biocypher_file_name.txt").read_text() == (
        f"biocypher-out/{TIME_STR}/{INCREMENTAL_CALL_FILENAME}"
    )


def test_incremental_build_refuses_a_duplicate_edge_risk(
    build: SimpleNamespace,
) -> None:
    """A plan whose analysis flags edges between two existing nodes aborts after the
    CSVs are written, naming the sample and the directory left for inspection.
    """
    build.plan = _plan(risk=True)
    message = (
        "increment contains relationships whose BOTH endpoints already exist in the "
        "served graph; incremental import would duplicate them. Sample: "
        "[['GenomeMemberOf', 'G', 'X']]. CSVs left in /x/out for inspection."
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        ks.main(_cfg(INCREMENT_CFG))
    (bc,) = FakeBioCypher.instances
    assert [call[0] for call in bc.calls] == ["write_nodes", "write_edges"]
    assert build.wandb.logged[-1] == {
        "increment_n_node_ids": 7,
        "increment_n_external_ids": 1,
        "increment_n_edges_between_external": 1,
    }
    assert build.wandb.finish_calls == 0
    assert not (build.tmp_path / "biocypher_file_name.txt").exists()


def test_incremental_build_requires_the_staged_lmdb(build: SimpleNamespace) -> None:
    """In incremental mode a selected dataset with no LMDB is an error, not a skip."""
    cfg = {**INCREMENT_CFG, "datasets": ["FakeGamma"]}
    message = (
        f"FakeGamma: no LMDB at {osp.join(build.root, 'data/torchcell/gamma')}; stage "
        "the dataset into the build tree before an incremental admission"
    )
    with pytest.raises(FileNotFoundError, match=f"^{re.escape(message)}$"):
        ks.main(_cfg(cfg))
    assert FakeGamma.instances == []
    assert FakeBioCypher.instances[0].calls == []


@pytest.mark.parametrize(
    ("overrides", "exc", "message"),
    [
        (
            {"import_mode": "bogus"},
            ValueError,
            "import_mode must be 'full' or 'incremental', got 'bogus'",
        ),
        (
            {"datasets": [], "import_mode": "incremental"},
            ValueError,
            "import_mode: incremental requires a non-empty `datasets` list",
        ),
        (
            {"datasets": ["Nope", "FakeAlpha"], "import_mode": "full"},
            KeyError,
            "datasets not in dataset_adapter_map: ['Nope']",
        ),
    ],
    ids=["bad-mode", "incremental-without-datasets", "unknown-dataset"],
)
def test_membership_and_mode_are_validated_before_any_dataset_is_built(
    build: SimpleNamespace,
    overrides: dict[str, Any],
    exc: type[Exception],
    message: str,
) -> None:
    """Each bad config raises its exact message after wandb, BioCypher, the genome and
    the graph are built (the shared genome and graph precede membership validation at
    create_scerevisiae_kg_small.py lines 160-168), but before any loader is constructed
    or anything is written.
    """
    cfg = {**INCREMENT_CFG, **overrides}
    with pytest.raises(exc, match=re.escape(message)):
        ks.main(_cfg(cfg))
    assert len(FakeGenome.instances) == len(FakeGraph.instances) == 1
    assert FakeAlpha.instances == []
    assert FakeBioCypher.instances[0].calls == []
    assert build.wandb.finish_calls == 0


def test_a_config_without_a_local_head_ontology_stops_before_biocypher(
    build: SimpleNamespace,
) -> None:
    """Issue #619: a BioCypher config with no ``head_ontology`` would make BioCypher
    fetch Biolink from GitHub; the build raises naming the URL it refused, and no
    BioCypher instance is constructed.
    """
    Path(build.tmp_path / "biocypher_config.yaml").write_text(
        "neo4j:\n  database_name: torchcell-test\n"
    )
    with pytest.raises(HeadOntologyError, match=re.escape(BIOLINK_SOURCE_URL)):
        ks.main(_cfg(FULL_CFG))
    assert FakeBioCypher.instances == []


# Phase 24: the fast-writer sink path


class FakeSink:
    """``FastCsvSink`` stand-in: records its construction and every write, in order, and
    returns how many items each write consumed (the real sink's contract).
    """

    instances: list[FakeSink] = []

    def __init__(self, bc: Any, specs: Any) -> None:
        """Record the BioCypher instance and the row specs it is bound to."""
        self.bc = bc
        self.specs = specs
        self.calls: list[tuple[Any, ...]] = []
        FakeSink.instances.append(self)

    def write_nodes(self, nodes: Any) -> int:
        """Drain and record the node stream; return its length plus 100, so the logged
        totals can only come from the sink's return value.
        """
        items = list(nodes)
        self.calls.append(("write_nodes", items))
        return len(items) + 100

    def write_edges(self, edges: Any) -> int:
        """Drain and record the edge stream; return its length plus 100."""
        items = list(edges)
        self.calls.append(("write_edges", items))
        return len(items) + 100

    def finish(self) -> None:
        """Record the close."""
        self.calls.append(("finish",))


def test_fast_writer_routes_every_write_through_the_sink_and_finishes_it_once(
    build: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``adapters.fast_writer: true`` freezes row specs from the BioCypher instance
    (``build_row_specs(bc)``) and wraps it in one ``FastCsvSink(bc, specs)``. The same
    full build as ``test_full_build_prefilters_caps_skips_and_writes_the_import_call``
    then sends alpha's 2 nodes and 2 edges and beta's 3 and 3 to the sink, not to
    BioCypher, and calls ``finish`` exactly once, after the last edge write; BioCypher
    itself only writes the import call and the schema node. The counts the sink returns
    are the logged totals: the fake returns length + 100, so alpha logs 102 and beta
    103 of each, 205 in total, and every adapter carries the specs.
    """
    reset_instances(FakeSink)
    specs = SimpleNamespace(name="row-specs")
    spec_calls: list[Any] = []

    def build_row_specs(bc: Any) -> SimpleNamespace:
        spec_calls.append(bc)
        return specs

    monkeypatch.setattr(ks, "build_row_specs", build_row_specs)
    monkeypatch.setattr(ks, "FastCsvSink", FakeSink)
    cfg = {**FULL_CFG, "adapters": {**BASE["adapters"], "fast_writer": True}}
    ks.main(_cfg(cfg))

    (bc,) = FakeBioCypher.instances
    (sink,) = FakeSink.instances
    assert spec_calls == [bc]
    assert (sink.bc, sink.specs) == (bc, specs)
    assert sink.calls == [
        ("write_nodes", [("node", "FakeAdapterA", 0), ("node", "FakeAdapterA", 1)]),
        ("write_edges", [("edge", "FakeAdapterA", 0), ("edge", "FakeAdapterA", 1)]),
        ("write_nodes", [("node", "FakeAdapterB", i) for i in range(3)]),
        ("write_edges", [("edge", "FakeAdapterB", i) for i in range(3)]),
        ("finish",),
    ]
    assert bc.calls == [("write_import_call",), ("write_schema_info", True)]
    (adapter_a,) = FakeAdapterA.instances
    (adapter_b,) = FakeAdapterB.instances
    # set by main() after construction, so not a declared attribute of the fake
    assert vars(adapter_a)["row_specs"] is specs
    assert vars(adapter_b)["row_specs"] is specs
    totals = [d for d in build.wandb.logged if "generation_wall_s" in d]
    assert [(d["total_nodes"], d["total_edges"]) for d in totals] == [(205, 205)]
    assert build.wandb.logged[12:14] == [
        {
            "FakeAdapterB_write_nodes_time(s)": 1.0,
            "FakeAdapterB_n_nodes": 103,
            "total_nodes": 205,
        },
        {
            "FakeAdapterB_write_edges_time": 1.0,
            "FakeAdapterB_n_edges": 103,
            "total_edges": 205,
        },
    ]


class FakeKeio(FakeDataset):
    """Three records at ``data/torchcell/keio``; a BW25113 loader naming ``ecoli_genome``."""

    REFERENCE_STRAIN = "BW25113"
    n_records = 3
    instances: list[FakeDataset] = []

    def __init__(
        self,
        root: str = "data/torchcell/keio",
        io_workers: int = 1,
        ecoli_genome: Any = None,
    ) -> None:
        """Record every kwarg the build script injects."""
        super().__init__(root, io_workers=io_workers, ecoli_genome=ecoli_genome)


class FakeAdapterK(FakeAdapter):
    """Adapter for ``FakeKeio``."""

    instances: list[FakeAdapter] = []


def test_genomes_are_injected_by_parameter_name(
    build: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``FakeBeta`` names ``genome`` and still receives the shared ``SCerevisiaeGenome``;
    ``FakeKeio`` names ``ecoli_genome`` and receives an ``EcoliK12Genome`` of its
    REFERENCE_STRAIN (BW25113) from its default cache root with ``overwrite=False``,
    and no other loader is handed a bacterial genome.
    """
    reset_instances(FakeKeio, FakeAdapterK)
    log = install_bacterial_fakes(monkeypatch)
    monkeypatch.setattr(ks, "SCerevisiaeGenome", FakeYeastGenome)
    monkeypatch.setattr(
        ks,
        "dataset_adapter_map",
        {FakeAlpha: FakeAdapterA, FakeBeta: FakeAdapterB, FakeKeio: FakeAdapterK},
    )
    (
        build.tmp_path / "root" / "data" / "torchcell" / "keio" / "processed" / "lmdb"
    ).mkdir(parents=True)
    ks.main(_cfg(FULL_CFG))

    (alpha,) = FakeAlpha.instances
    (beta,) = FakeBeta.instances
    (keio,) = FakeKeio.instances
    assert alpha.kwargs == {"io_workers": 8}
    assert isinstance(beta.kwargs["genome"], SCerevisiaeGenome)
    injected = keio.kwargs["ecoli_genome"]
    assert isinstance(injected, EcoliK12Genome)
    assert isinstance(injected, EcoliK12BW25113Genome)
    assert keio.kwargs == {"io_workers": 8, "ecoli_genome": injected}
    assert [entry for entry in log if entry[0] != "FakeYeastGenome"] == [
        (
            "FakeBW25113Genome",
            {
                "genome_root": osp.join(build.root, "data/ecoli/bw25113/genome"),
                "overwrite": False,
            },
        )
    ]


def test_a_yeast_only_build_never_builds_a_bacterial_genome(
    build: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = install_bacterial_fakes(monkeypatch)
    ks.main(_cfg(FULL_CFG))
    assert len(FakeBeta.instances) == 1
    assert log == []
