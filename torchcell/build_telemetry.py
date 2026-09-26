# torchcell/build_telemetry
# [[torchcell.build_telemetry]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/build_telemetry
# Test file: tests/torchcell/test_build_telemetry.py
"""Container-level resource telemetry for the KG build, paired with build phases.

W&B's own system metrics see only the parent process: on job 2032 they reported 2% CPU
and 6 threads for a build that ran 21 worker processes plus their loader children. The
sampler here reads the cgroup v2 controllers the container (or the slurm step) is
confined to, so every sample counts every process in the build:

* ``cpu.stat`` ``usage_usec`` -> CPU cores in use over the sample interval
* ``memory.current`` / ``memory.peak`` / ``memory.max`` -> resident, peak, and the limit
* ``memory.stat`` ``anon`` -> anonymous (heap) memory, the part that OOM-kills

Each sample carries the build phase current at that instant (adapter + method), which
the adapter sets through :class:`BuildPhase` as it moves through its methods. The
sampler also records every phase transition, so a build ends with a per-method timing
table without parsing logs.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path

import wandb
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

CGROUP_ROOT = Path("/sys/fs/cgroup")


class ResourceSample(BaseModel):
    """One reading of the build's cgroup, tagged with the phase it fell in."""

    t: float = Field(description="Unix time of the sample")
    elapsed_s: float = Field(description="Seconds since the sampler started")
    cpu_cores: float = Field(
        description="Cores in use over the interval (usage delta / wall delta)"
    )
    mem_gb: float = Field(description="memory.current in GiB")
    mem_peak_gb: float = Field(description="memory.peak in GiB")
    anon_gb: float = Field(description="memory.stat anon in GiB")
    adapter: str = ""
    method: str = ""
    phase_kind: str = Field(default="", description="node | edge | setup | finish")


class PhaseTiming(BaseModel):
    """Wall time of one build phase (one adapter method, or a setup step)."""

    adapter: str
    method: str
    phase_kind: str
    start_t: float
    end_t: float
    seconds: float
    cpu_core_seconds: float = Field(
        description="Integral of cpu_cores over the phase, from the samples that fell in it"
    )
    mem_peak_gb: float = Field(
        description="Max memory.current sampled during the phase"
    )


class BuildPhase:
    """Process-global 'where is the build right now', set by the build and adapters.

    A module-level singleton rather than an argument threaded through every adapter
    method: the adapters are instantiated by name from a registry and the sampler is a
    thread, so a global is the one place both can reach.
    """

    adapter: str = ""
    method: str = ""
    kind: str = "setup"
    _lock = threading.Lock()
    _transitions: list[tuple[float, str, str, str]] = []

    @classmethod
    def set(cls, adapter: str, method: str, kind: str) -> None:
        """Record a phase change; the sampler stamps subsequent samples with it."""
        with cls._lock:
            cls.adapter, cls.method, cls.kind = adapter, method, kind
            cls._transitions.append((time.time(), adapter, method, kind))

    @classmethod
    def current(cls) -> tuple[str, str, str]:
        """Return the (adapter, method, kind) triple in effect right now."""
        with cls._lock:
            return cls.adapter, cls.method, cls.kind

    @classmethod
    def transitions(cls) -> list[tuple[float, str, str, str]]:
        """Return every recorded transition, oldest first."""
        with cls._lock:
            return list(cls._transitions)


def _read_cpu_usage_usec() -> int:
    for line in (CGROUP_ROOT / "cpu.stat").read_text().splitlines():
        key, value = line.split()
        if key == "usage_usec":
            return int(value)
    raise RuntimeError("cpu.stat carries no usage_usec")


def _read_int(path: Path) -> int:
    text = path.read_text().strip()
    return -1 if text == "max" else int(text)


def _read_anon_bytes() -> int:
    for line in (CGROUP_ROOT / "memory.stat").read_text().splitlines():
        key, value = line.split()
        if key == "anon":
            return int(value)
    raise RuntimeError("memory.stat carries no anon")


GIB = 1024**3


class ResourceSampler:
    """Background thread sampling the cgroup every ``interval`` seconds.

    Samples go to W&B under ``res/`` and to ``<out_dir>/resource_samples.csv``; on
    :meth:`stop` the phase transitions are folded into a per-phase timing table saved as
    ``<out_dir>/phase_timings.json`` and logged as a W&B table.
    """

    def __init__(self, out_dir: Path, interval: float = 5.0) -> None:
        """Prepare a sampler writing under ``out_dir``; nothing runs until start()."""
        self.out_dir = out_dir
        self.interval = interval
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="resource-sampler", daemon=True
        )
        self.t0 = time.time()
        self.mem_max_gb = _read_int(CGROUP_ROOT / "memory.max") / GIB
        self.cpu_max = _read_cpu_quota()

    def start(self) -> None:
        """Start sampling; logs the cgroup limits once so the arm records its box."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        wandb.log({"res/mem_limit_gb": self.mem_max_gb, "res/cpu_limit": self.cpu_max})
        log.info(
            "resource sampler: cpu limit %s, memory limit %.1f GiB",
            self.cpu_max,
            self.mem_max_gb,
        )
        self._thread.start()

    def _run(self) -> None:
        last_usage = _read_cpu_usage_usec()
        last_t = time.time()
        csv_path = self.out_dir / "resource_samples.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(ResourceSample.model_fields)
            )
            writer.writeheader()
            while not self._stop.wait(self.interval):
                now = time.time()
                usage = _read_cpu_usage_usec()
                cores = (usage - last_usage) / 1e6 / (now - last_t)
                last_usage, last_t = usage, now
                adapter, method, kind = BuildPhase.current()
                sample = ResourceSample(
                    t=now,
                    elapsed_s=now - self.t0,
                    cpu_cores=cores,
                    mem_gb=_read_int(CGROUP_ROOT / "memory.current") / GIB,
                    mem_peak_gb=_read_int(CGROUP_ROOT / "memory.peak") / GIB,
                    anon_gb=_read_anon_bytes() / GIB,
                    adapter=adapter,
                    method=method,
                    phase_kind=kind,
                )
                self.samples.append(sample)
                writer.writerow(sample.model_dump())
                handle.flush()
                wandb.log(
                    {
                        "res/cpu_cores": sample.cpu_cores,
                        "res/mem_gb": sample.mem_gb,
                        "res/mem_peak_gb": sample.mem_peak_gb,
                        "res/anon_gb": sample.anon_gb,
                        "res/elapsed_s": sample.elapsed_s,
                    }
                )

    def stop(self) -> list[PhaseTiming]:
        """Stop sampling, write the phase timing table, and return it."""
        self._stop.set()
        self._thread.join(timeout=self.interval * 3)
        timings = self.phase_timings()
        (self.out_dir / "phase_timings.json").write_text(
            json.dumps([t.model_dump() for t in timings], indent=1)
        )
        names = list(PhaseTiming.model_fields)
        columns: list[str | int] = list(names)
        table = wandb.Table(
            columns=columns, data=[[getattr(t, c) for c in names] for t in timings]
        )
        wandb.log({"phase_timings": table})
        return timings

    def phase_timings(self) -> list[PhaseTiming]:
        """Fold the phase transitions and samples into one row per phase."""
        transitions = BuildPhase.transitions()
        end_marker = time.time()
        rows: list[PhaseTiming] = []
        for i, (start_t, adapter, method, kind) in enumerate(transitions):
            end_t = transitions[i + 1][0] if i + 1 < len(transitions) else end_marker
            inside = [s for s in self.samples if start_t <= s.t < end_t]
            rows.append(
                PhaseTiming(
                    adapter=adapter,
                    method=method,
                    phase_kind=kind,
                    start_t=start_t,
                    end_t=end_t,
                    seconds=end_t - start_t,
                    cpu_core_seconds=sum(s.cpu_cores * self.interval for s in inside),
                    mem_peak_gb=max((s.mem_gb for s in inside), default=0.0),
                )
            )
        return rows


def _read_cpu_quota() -> float:
    """CPU limit in cores from cpu.max (``max`` when unlimited -> -1)."""
    quota, period = (CGROUP_ROOT / "cpu.max").read_text().split()
    if quota == "max":
        return -1.0
    return int(quota) / int(period)
