# experiments/030-solid-growth-multi/scripts/loader_throughput_030.py
# [[experiments.030-solid-growth-multi.scripts.loader_throughput_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/loader_throughput_030
r"""Measure the training loader's throughput on one node: records per second by worker
count and by store, without a GPU and without an epoch.

WHY. The 030 Q arm on IGB compute-5-7 (job 2427397) ran epochs 1 to 2 at 84 to 94 min and
epochs 4 to 10 at 145 to 188 min; the probe found 27 of 28 loader workers in GPFS's
``mmapLock`` with the GPUs at 39%. Whether fewer workers per rank or a RAM-disk pool store
(``make_pool_store_030.py``) restores the pace is a loader question, and an epoch is the
wrong instrument for it: the first epoch of a job is slower for its own reasons (cold
page cache, worker spawn), and one setting per 90-minute epoch cannot sweep anything. This
script builds the arm's real dataset and data module (``arm_030``), draws the training
loader with each worker count in turn, discards ``warmup`` batches and times ``batches``
batches, so a setting costs minutes and the cold start is excluded. While a setting runs
it samples the worker processes' ``/proc/<pid>/stat`` state and ``wchan`` so the GPFS lock
shows up as a number beside the rate.

Settings are the cross product of ``bench.roots`` (dataset roots: ``config`` means the
config's ``dataset.root_rel``; an absolute path such as a ``/dev/shm`` staging of a pool
store is used as given) and ``bench.workers``. Per setting it reports records/s, entry
rows/s, batch-time quantiles, the fraction of worker samples in D state and the top wchan,
and the epoch time that rate projects for the arm's training set at 1 and 2 ranks (a
projection: it assumes the rate scales with ranks, which under lock contention it does not).

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/loader_throughput_030.py \\
        --config-name cgt_030_s3_q_tok_embfit_004 '+bench.workers=[14,8,4,2]' \\
        '+bench.roots=[config]' +bench.batches=60 +bench.warmup=15 +bench.label=gpfs

Writes ``results/loader_throughput/<label>_<host>_<job>.json`` (``LoaderBenchReport``) and
prints a table.
"""

from __future__ import annotations

import gc
import json
import os
import os.path as osp
import socket
import statistics
import sys
import time
from collections import Counter
from typing import Any

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import (  # noqa: E402
    EXPERIMENT,
    build_dataset,
    make_data_module,
    resolve_arm,
)


class WorkerSample(BaseModel):
    """One sweep over the loader's worker processes."""

    n_workers: int
    n_state_d: int
    wchan: dict[str, int]


class LoaderSetting(BaseModel):
    """Throughput of one (root, workers) setting."""

    root: str
    store: str
    num_workers: int
    batch_size: int
    warmup_batches: int
    measured_batches: int
    records: int
    rows: int
    seconds: float
    records_per_s: float
    rows_per_s: float
    batch_s_median: float
    batch_s_p90: float
    batch_s_max: float
    warmup_seconds: float
    worker_samples: int
    frac_workers_in_d: float
    top_wchan: dict[str, int]
    projected_epoch_min_1rank: float
    projected_epoch_min_2rank: float


class LoaderBenchReport(BaseModel):
    """Every setting of one job, with the node and the arm's training-set size."""

    label: str
    host: str
    slurm_job_id: str
    config_name: str
    n_train_records: int
    cpus: int | None
    settings: list[LoaderSetting]


def _workers_of(pid: int) -> list[int]:
    """PIDs whose parent is ``pid`` (the spawned DataLoader workers)."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == pid:
            out.append(int(name))
    return out


def _sample_workers(pid: int) -> WorkerSample:
    states: Counter[str] = Counter()
    wchans: Counter[str] = Counter()
    pids = _workers_of(pid)
    for p in pids:
        try:
            with open(f"/proc/{p}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
            with open(f"/proc/{p}/wchan") as f:
                wchan = f.read().strip() or "0"
        except OSError:
            continue
        states[state] += 1
        if state == "D":
            wchans[wchan] += 1
    return WorkerSample(n_workers=len(pids), n_state_d=states["D"], wchan=dict(wchans))


def _rows_in(batch: Any) -> int:
    pv = batch["gene"].phenotype_values
    return int(pv.shape[0])


def run_setting(
    dataset: Any,
    arm: Any,
    seed: int,
    dm_cfg: dict[str, Any],
    follow_batch: list[str],
    root: str,
    num_workers: int,
    warmup: int,
    batches: int,
    n_train: int,
) -> LoaderSetting:
    cfg = dict(dm_cfg) | {"num_workers": num_workers}
    dm = make_data_module(dataset, arm, seed, cfg, follow_batch)
    loader = dm.train_dataloader()
    it = iter(loader)
    t0 = time.time()
    for _ in range(warmup):
        next(it)
    warm_s = time.time() - t0
    times: list[float] = []
    records = rows = 0
    samples: list[WorkerSample] = []
    pid = os.getpid()
    for b in range(batches):
        t = time.time()
        batch = next(it)
        times.append(time.time() - t)
        records += int(batch.num_graphs)
        rows += _rows_in(batch)
        if b % 5 == 0 and num_workers > 0:
            samples.append(_sample_workers(pid))
    seconds = sum(times)
    rate = records / seconds
    n_samples = sum(s.n_workers for s in samples)
    n_d = sum(s.n_state_d for s in samples)
    wchan: Counter[str] = Counter()
    for s in samples:
        wchan.update(s.wchan)
    setting = LoaderSetting(
        root=root,
        store="pool_store" if dataset._pool_store is not None else "full_build",
        num_workers=num_workers,
        batch_size=int(cfg["batch_size"]),
        warmup_batches=warmup,
        measured_batches=batches,
        records=records,
        rows=rows,
        seconds=seconds,
        records_per_s=rate,
        rows_per_s=rows / seconds,
        batch_s_median=statistics.median(times),
        batch_s_p90=sorted(times)[int(0.9 * (len(times) - 1))],
        batch_s_max=max(times),
        warmup_seconds=warm_s,
        worker_samples=n_samples,
        frac_workers_in_d=(n_d / n_samples) if n_samples else 0.0,
        top_wchan=dict(wchan.most_common(3)),
        projected_epoch_min_1rank=n_train / rate / 60,
        projected_epoch_min_2rank=n_train / rate / 2 / 60,
    )
    print(
        f"  workers={num_workers:>2} {setting.store:<10} {rate:7.1f} rec/s "
        f"{setting.rows_per_s:8.1f} rows/s  batch median {setting.batch_s_median:.2f}s "
        f"p90 {setting.batch_s_p90:.2f}s  D-state {setting.frac_workers_in_d:.0%} "
        f"{setting.top_wchan}  -> epoch {setting.projected_epoch_min_2rank:.0f} min at 2 ranks",
        flush=True,
    )
    # Persistent workers live on the iterator; dropping it shuts them down.
    del it, loader, dm
    gc.collect()
    time.sleep(5)
    return setting


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="cgt_030_s3_q_tok_embfit_004",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    bench = raw["bench"]
    workers = [int(w) for w in bench["workers"]]
    roots = [str(r) for r in bench.get("roots", ["config"])]
    warmup = int(bench.get("warmup", 15))
    batches = int(bench.get("batches", 60))
    label = str(bench.get("label", "bench"))
    seed = int(raw.get("seed", 42))
    config_name = str(HydraConfig.get().job.config_name)

    arm = resolve_arm(raw["subset"])
    n_train = len(arm.train_records())
    with open(
        osp.join(experiment_root, EXPERIMENT, "queries/001_multi_measurement.cql")
    ) as f:
        query = f.read()
    dm_cfg = dict(raw["data_module"])
    follow_batch = ["perturbation_indices", "phenotype_values"]
    print(
        f"bench {label}: roots={roots} workers={workers} batch_size={dm_cfg['batch_size']} "
        f"warmup={warmup} batches={batches} train records={n_train:,}"
    )
    settings: list[LoaderSetting] = []
    for root in roots:
        dataset_root = (
            osp.join(data_root, raw["dataset"]["root_rel"])
            if root == "config"
            else root
        )
        t0 = time.time()
        dataset, _, _, _ = build_dataset(
            data_root=data_root,
            dataset_root=dataset_root,
            query=query,
            graph_names=list(raw["cell_dataset"]["graphs"]),
            node_embedding_names=[],
            phenotype_labels=list(raw["cell_dataset"]["phenotype_labels"]),
            dataset_vocabulary=arm.dataset_vocabulary,
        )
        dataset._init_lmdb_read()
        store = "pool_store" if dataset._pool_store is not None else "full_build"
        dataset.close_lmdb()
        print(
            f"root {dataset_root}: {store}, len {len(dataset):,}, built in {time.time() - t0:.0f}s"
        )
        for w in workers:
            settings.append(
                run_setting(
                    dataset,
                    arm,
                    seed,
                    dm_cfg,
                    follow_batch,
                    dataset_root,
                    w,
                    warmup,
                    batches,
                    n_train,
                )
            )
    report = LoaderBenchReport(
        label=label,
        host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        config_name=config_name,
        n_train_records=n_train,
        cpus=int(os.environ["SLURM_CPUS_PER_TASK"])
        if "SLURM_CPUS_PER_TASK" in os.environ
        else None,
        settings=settings,
    )
    out_dir = osp.join(experiment_root, EXPERIMENT, "results", "loader_throughput")
    os.makedirs(out_dir, exist_ok=True)
    out = osp.join(
        out_dir, f"{label}_{report.host}_{report.slurm_job_id or 'local'}.json"
    )
    with open(out, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(
        "\n| store | workers | rec/s | rows/s | batch median s | p90 s | D-state | epoch min @2 ranks |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for s in settings:
        print(
            f"| {s.store} | {s.num_workers} | {s.records_per_s:.1f} | {s.rows_per_s:.1f} | "
            f"{s.batch_s_median:.2f} | {s.batch_s_p90:.2f} | {s.frac_workers_in_d:.0%} | "
            f"{s.projected_epoch_min_2rank:.0f} |"
        )
    print(f"wrote {out}")
    print(json.dumps({"n_settings": len(settings)}))


if __name__ == "__main__":
    main()
