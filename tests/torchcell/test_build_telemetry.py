# tests/torchcell/test_build_telemetry.py
# [[tests.torchcell.test_build_telemetry]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_build_telemetry.py
"""Tests for torchcell.build_telemetry, against hand-written cgroup v2 files.

The module has no single-sample method: ``ResourceSampler._run`` loops on
``self._stop.wait(interval)``. The tests replace ``_stop`` with a scripted Event whose
``wait`` runs one step (advance a fake clock, rewrite cgroup files, set a phase) and
returns False to take a sample or True to end, so ``_run`` runs synchronously and no
assertion depends on wall time. ``time``, ``wandb`` and the /proc parent-CPU reader are
replaced on the module; ``CGROUP_ROOT`` points at ``tmp_path``.

Derived values (``GIB = 1024**3``):

* ``memory.max`` 17179869184 B = 16 GiB -> ``mem_max_gb`` 16.0; ``cpu.max`` "400000
  100000" -> ``cpu_max`` 4.0.
* First sample: clock 1000 -> 1010 (10 s), ``usage_usec`` 1,000,000 -> 26,000,000,
  so cores = 25e6 / 1e6 / 10 = 2.5; parent CPU 4.0 s -> 9.0 s gives 0.5 cores;
  ``memory.current`` 3221225472 B = 3.0 GiB; ``memory.peak`` 5368709120 B = 5.0 GiB;
  ``anon`` 2147483648 B = 2.0 GiB; ``cgroup.procs`` lists 3 pids.
* ``memory.max`` of "max" reads as -1 bytes, so ``mem_max_gb`` = -1 / 1024**3.
* Two-phase run (interval 10): get_nodes from t=1000 to 1025 holds samples at 1010
  (2.0 cores, 3 GiB) and 1020 (4.0 cores, 6 GiB): 25 s, (2 + 4) * 10 = 60.0
  core-seconds, peak 6.0. get_edges from 1025 to 1035 holds the 1030 sample (1.0 core,
  4 GiB): 10 s, 10.0 core-seconds, peak 4.0.
"""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from torchcell import build_telemetry as bt
from torchcell.build_telemetry import BuildPhase, ResourceSampler

GIB = 1024**3


class FakeClock:
    """Stands in for the ``time`` module; ``time()`` returns ``now``."""

    def __init__(self, now: float) -> None:
        """Start the clock at ``now``."""
        self.now = now

    def time(self) -> float:
        """Return the current fake time."""
        return self.now


class ScriptedStop(threading.Event):
    """Each ``wait`` runs the next step and returns its result (False = sample)."""

    def __init__(self, steps: list[Callable[[], bool]]) -> None:
        """Hold the steps to run, one per ``wait``."""
        super().__init__()
        self.steps = steps

    def wait(self, timeout: float | None = None) -> bool:
        """Run the next step; never blocks."""
        return self.steps.pop(0)()


class FakeTable:
    """Records what ``wandb.Table`` was built with."""

    def __init__(self, columns: list[str | int], data: list[list[Any]]) -> None:
        """Keep the columns and rows."""
        self.columns = columns
        self.data = data


def write_cgroup(root: Path, usage_usec: int, current: int) -> None:
    (root / "cpu.stat").write_text(
        f"usage_usec {usage_usec}\nuser_usec 7\nsystem_usec 3\n"
    )
    (root / "memory.current").write_text(f"{current}\n")


@pytest.fixture
def cgroup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text("17179869184\n")
    (root / "cpu.max").write_text("400000 100000\n")
    (root / "memory.peak").write_text("5368709120\n")
    (root / "memory.stat").write_text("file 4096\nanon 2147483648\nshmem 0\n")
    (root / "cgroup.procs").write_text("101\n102\n103\n")
    write_cgroup(root, 1_000_000, 3 * GIB)
    monkeypatch.setattr(bt, "CGROUP_ROOT", root)
    monkeypatch.setattr(BuildPhase, "adapter", "")
    monkeypatch.setattr(BuildPhase, "method", "")
    monkeypatch.setattr(BuildPhase, "kind", "setup")
    monkeypatch.setattr(BuildPhase, "_transitions", [])
    return root


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock(1000.0)
    monkeypatch.setattr(bt, "time", fake)
    return fake


@pytest.fixture
def wandb_logs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    monkeypatch.setattr(bt, "wandb", SimpleNamespace(log=logs.append, Table=FakeTable))
    return logs


def parent_cpu(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    monkeypatch.setattr(bt, "_read_parent_cpu_seconds", lambda: values.pop(0))


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def test_init_reads_cgroup_limits(cgroup: Path, clock: FakeClock) -> None:
    sampler = ResourceSampler(cgroup / "out", interval=10.0)
    assert sampler.mem_max_gb == 16.0
    assert sampler.cpu_max == 4.0
    assert sampler.t0 == 1000.0


def test_first_row_exact(
    cgroup: Path,
    clock: FakeClock,
    wandb_logs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = cgroup / "out"
    out.mkdir()
    parent_cpu(monkeypatch, [4.0, 9.0])

    def step() -> bool:
        clock.now = 1010.0
        write_cgroup(cgroup, 26_000_000, 3 * GIB)
        return False

    sampler = ResourceSampler(out, interval=10.0)
    sampler._stop = ScriptedStop([step, lambda: True])
    sampler._run()

    assert read_rows(out / "resource_samples.csv") == [
        ["t", "elapsed_s", "cpu_cores", "mem_gb", "mem_peak_gb", "anon_gb"]
        + ["parent_cores", "n_procs", "adapter", "method", "phase_kind"],
        ["1010.0", "10.0", "2.5", "3.0", "5.0", "2.0", "0.5", "3", "", "", "setup"],
    ]
    assert wandb_logs == [
        {
            "res/cpu_cores": 2.5,
            "res/mem_gb": 3.0,
            "res/mem_peak_gb": 5.0,
            "res/anon_gb": 2.0,
            "res/parent_cores": 0.5,
            "res/n_procs": 3,
            "res/elapsed_s": 10.0,
        }
    ]


def test_phase_stamp_follows_build_phase_set(
    cgroup: Path,
    clock: FakeClock,
    wandb_logs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = cgroup / "out"
    out.mkdir()
    parent_cpu(monkeypatch, [0.0, 0.0, 0.0])

    def to_phase(t: float, adapter: str, method: str, kind: str) -> Callable[[], bool]:
        def step() -> bool:
            clock.now = t
            BuildPhase.set(adapter, method, kind)
            return False

        return step

    sampler = ResourceSampler(out, interval=10.0)
    sampler._stop = ScriptedStop(
        [
            to_phase(1010.0, "GeneAdapter", "get_nodes", "node"),
            to_phase(1020.0, "GeneAdapter", "get_edges", "edge"),
            lambda: True,
        ]
    )
    sampler._run()

    stamps = [row[-3:] for row in read_rows(out / "resource_samples.csv")[1:]]
    assert stamps == [
        ["GeneAdapter", "get_nodes", "node"],
        ["GeneAdapter", "get_edges", "edge"],
    ]
    assert BuildPhase.current() == ("GeneAdapter", "get_edges", "edge")
    assert BuildPhase.transitions() == [
        (1010.0, "GeneAdapter", "get_nodes", "node"),
        (1020.0, "GeneAdapter", "get_edges", "edge"),
    ]


def test_memory_max_unlimited_reads_as_minus_one_byte(
    cgroup: Path, clock: FakeClock
) -> None:
    (cgroup / "memory.max").write_text("max\n")
    (cgroup / "cpu.max").write_text("max 100000\n")
    sampler = ResourceSampler(cgroup / "out", interval=10.0)
    assert sampler.mem_max_gb == -1 / 1024**3
    assert sampler.cpu_max == -1.0


def test_missing_memory_max_raises(cgroup: Path, clock: FakeClock) -> None:
    (cgroup / "memory.max").unlink()
    with pytest.raises(FileNotFoundError, match="memory.max"):
        ResourceSampler(cgroup / "out", interval=10.0)


def test_missing_procs_file_raises_after_header(
    cgroup: Path,
    clock: FakeClock,
    wandb_logs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = cgroup / "out"
    out.mkdir()
    parent_cpu(monkeypatch, [0.0, 0.0])
    (cgroup / "cgroup.procs").unlink()

    def step() -> bool:
        clock.now = 1010.0
        return False

    sampler = ResourceSampler(out, interval=10.0)
    sampler._stop = ScriptedStop([step])
    with pytest.raises(FileNotFoundError, match="cgroup.procs"):
        sampler._run()
    assert len(read_rows(out / "resource_samples.csv")) == 1
    assert sampler.samples == []


def test_cpu_stat_without_usage_raises(cgroup: Path, clock: FakeClock) -> None:
    (cgroup / "cpu.stat").write_text("user_usec 7\nsystem_usec 3\n")
    sampler = ResourceSampler(cgroup / "out", interval=10.0)
    with pytest.raises(RuntimeError, match="cpu.stat carries no usage_usec"):
        sampler._run()


def test_memory_stat_without_anon_raises(cgroup: Path) -> None:
    (cgroup / "memory.stat").write_text("file 4096\n")
    with pytest.raises(RuntimeError, match="memory.stat carries no anon"):
        bt._read_anon_bytes()


def test_stop_returns_per_phase_timings(
    cgroup: Path,
    clock: FakeClock,
    wandb_logs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = cgroup / "out"
    parent_cpu(monkeypatch, [0.0] * 4)
    BuildPhase.set("GeneAdapter", "get_nodes", "node")

    def sample_at(t: float, usage: int, current: int) -> Callable[[], bool]:
        def step() -> bool:
            clock.now = t
            write_cgroup(cgroup, usage, current)
            return False

        return step

    def switch_then_sample() -> bool:
        clock.now = 1025.0
        BuildPhase.set("GeneAdapter", "get_edges", "edge")
        return sample_at(1030.0, 71_000_000, 4 * GIB)()

    def finish() -> bool:
        clock.now = 1035.0
        return True

    sampler = ResourceSampler(out, interval=10.0)
    sampler._stop = ScriptedStop(
        [
            sample_at(1010.0, 21_000_000, 3 * GIB),
            sample_at(1020.0, 61_000_000, 6 * GIB),
            switch_then_sample,
            finish,
        ]
    )
    sampler.start()
    sampler._thread.join()
    timings = sampler.stop()

    assert [s.cpu_cores for s in sampler.samples] == [2.0, 4.0, 1.0]
    expected = [
        {
            "adapter": "GeneAdapter",
            "method": "get_nodes",
            "phase_kind": "node",
            "start_t": 1000.0,
            "end_t": 1025.0,
            "seconds": 25.0,
            "cpu_core_seconds": 60.0,
            "mem_peak_gb": 6.0,
        },
        {
            "adapter": "GeneAdapter",
            "method": "get_edges",
            "phase_kind": "edge",
            "start_t": 1025.0,
            "end_t": 1035.0,
            "seconds": 10.0,
            "cpu_core_seconds": 10.0,
            "mem_peak_gb": 4.0,
        },
    ]
    assert [t.model_dump() for t in timings] == expected
    assert json.loads((out / "phase_timings.json").read_text()) == expected
    assert wandb_logs[0] == {"res/mem_limit_gb": 16.0, "res/cpu_limit": 4.0}
    table = wandb_logs[-1]["phase_timings"]
    assert table.columns == list(expected[0])
    assert table.data == [list(row.values()) for row in expected]
