# experiments/database/scripts/bacterial_generation_profile.py
# [[experiments.database.scripts.bacterial_generation_profile]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/bacterial_generation_profile
r"""Generation cost of the bacterial tail of the KG build order, and the full-array projection.

Job 4901 (the serial KG 4.0 generation) died in ``EnvChemgenHoepfner2014Adapter`` having
walked the first 124 positions of the build order, so the 46 adapters after
``EnvChemgenShiver2016Adapter`` (index 123) had never been generated at all, and the
standard tier's own comment said their per-task peak was unmeasured. This script reads
what one per-adapter generation array wrote for those 46 slots and folds it into one row
per adapter: wall time, peak memory, cores, pool recycles, node and edge rows, and the
fraction of its tier each one used.

It reads only what the array already wrote (the ``DONE`` markers, each task's
``telemetry/resource_samples.csv``, and the generate logs); nothing here runs a
generation. The membership is recomputed from ``build_adapter_map`` every run, never read
off a list in prose: ``--tail-from`` names the adapter the measured prefix ended on and
every slot after it is in the tail.

Run from the repo root::

    PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python \\
        experiments/database/scripts/bacterial_generation_profile.py \\
        --out-dir /scratch/projects/torchcell/database/bacterial-gen-profile/biocypher-out/bacterial-gen-2026-10-15 \\
        --job 4974 --slurm-output /scratch/projects/torchcell/database/slurm/output \\
        --extra-index 108 --label bacterial-tail
    ... --table      # the Markdown tables of the note
    ... --list       # the membership alone, before any array has run

``--extra-index`` measures a slot outside the tail as well (108,
``PhageRbTnseqMutalik2020Adapter``, which lane P flagged at 37 G of store-build RSS
against the 64 G standard tier and which job 4901 generated only inside its cumulative
container, so its per-task peak was unmeasured too).

Outputs ``experiments/database/results/bacterial_generation_profile.json`` (every row,
the tier verdicts and the projection) and, with ``--table``, the Markdown the note
carries. A row whose peak passes ``--tier-warn-fraction`` of its tier's memory is
reported with the tier it should move to, which is the next tier in the table that holds
it under the same fraction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_build_adapter_resources import (  # type: ignore[import-not-found]  # noqa: E402  # a sibling script, not a package
    PhaseResources,
    fold_samples,
    read_recycles,
)

from torchcell.data.experiment_dataset import ALL_ORGANISMS  # noqa: E402
from torchcell.knowledge_graphs.generation_array import (  # noqa: E402
    AdapterSlot,
    DoneMarker,
    TierTable,
    adapter_order,
    load_tier_table,
    read_done,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_TAIL_FROM = "EnvChemgenShiver2016Adapter"


class AdapterGeneration(BaseModel):
    """One adapter's generation, as its own array task measured it."""

    index: int
    dataset_class: str
    adapter_class: str
    tier: str
    tier_cpus: int
    tier_mem_gb: int
    records: int = Field(description="the dataset length the task read")
    n_nodes: int
    n_edges: int
    wall_s: float = Field(description="the DONE marker's generation_wall_s")
    peak_memory_gb: float = Field(description="the DONE marker's peak_memory_gb")
    tier_memory_fraction: float = Field(description="peak / the tier's mem_gb")
    cores_mean: float | None = Field(
        default=None, description="mean cores over every phase, from the telemetry"
    )
    cores_max: float | None = None
    mem_max_gb_telemetry: float | None = Field(
        default=None, description="max cgroup memory the 5 s sampler saw"
    )
    recycles: int | None = Field(
        default=None, description="pool recycles the generate log reports"
    )
    phases: list[PhaseResources] = Field(default_factory=list)
    startup_s: float | None = Field(
        default=None, description="container start to the generator's first line"
    )
    torchcell_version: str
    build_commit: str
    in_tail: bool = Field(description="after the adapter job 4901 reached")

    @property
    def rows_per_s(self) -> float:
        """Node plus edge rows written per second of generation."""
        return (self.n_nodes + self.n_edges) / self.wall_s if self.wall_s else 0.0


class TierVerdict(BaseModel):
    """What one adapter's measured peak says about the tier it is keyed to."""

    adapter_class: str
    tier: str
    peak_memory_gb: float
    tier_mem_gb: int
    fraction: float
    over_warn: bool
    proposed_tier: str = Field(description="the tier it should move to, or its own")
    reason: str


class TierProjection(BaseModel):
    """A per-tier wall-clock projection of the full array at its concurrency."""

    tier: str
    tasks: int
    concurrency: int
    measured_tasks: int
    measured_task_seconds: float
    assumed_task_seconds: float = Field(
        description="seconds assumed for the tasks with no measurement"
    )
    startup_s: float
    total_task_seconds: float
    wall_hours: float
    basis: str


class Profile(BaseModel):
    """The whole readout: membership, rows, verdicts, projection."""

    label: str
    out_dir: str
    job: str | None
    build_commit: str | None
    tail_from: str
    tail_indices: list[int]
    extra_indices: list[int]
    tier_warn_fraction: float
    rows: list[AdapterGeneration]
    missing: list[int] = Field(description="slots asked for with no DONE marker")
    verdicts: list[TierVerdict]
    projection: list[TierProjection]
    projection_critical_path: str
    projection_startup_s: float


# ------------------------------------------------------------------- membership


def tail_slots(tail_from: str, include_private: bool = True) -> list[AdapterSlot]:
    """Every slot of the build order after ``tail_from``, recomputed from the map.

    The prose lists of the bacterial tail disagree with the map (the 2026-10-15 lane
    reports name 45 adapters by author and omit ``IsoprenolTiterBanerjee2024Adapter``,
    which is a second Banerjee store), so the membership is always this function's.
    """
    order = adapter_order(include_private, ALL_ORGANISMS)
    names = [slot.adapter_class for slot in order]
    if tail_from not in names:
        raise ValueError(f"{tail_from} is not in the build order of {len(order)}")
    cut = names.index(tail_from)
    return order[cut + 1 :]


# ---------------------------------------------------------------------- reading


def generation_startup_seconds(log: Path) -> float | None:
    """Container start to the generator's first logged line, from one task log.

    The ``---------------- 1 GENERATE ... <local date>`` line is written by the host
    shell just before ``docker run``; the generator's own
    ``[<utc>][__main__][INFO] - Generating adapter`` line is the first thing inside the
    container. Their difference is everything the container paid before any record was
    read: the three pip installs on the install path (255.9 s, lane C2) or the image's
    own start. None when either line is absent.
    """
    if not log.is_file():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    started = re.search(
        r"^-+ 1 GENERATE \S+ in \S+(?: from \S+)? (.+)$", text, re.MULTILINE
    )
    first = re.search(
        r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\[__main__\]"
        r"\[INFO\] - Generating adapter ",
        text,
        re.MULTILINE,
    )
    if started is None or first is None:
        return None
    for fmt in ("%a %b %d %I:%M:%S %p %Z %Y", "%a %b %d %H:%M:%S %Z %Y"):
        try:
            t0 = datetime.strptime(started.group(1).strip(), fmt)
        except ValueError:
            continue
        # The node is CDT (UTC-5) and the container logs UTC; both are wall clock.
        t0 = t0.replace(tzinfo=timezone(timedelta(hours=-5)))
        t1 = datetime.strptime(first.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        return round((t1 - t0).total_seconds(), 1)
    return None


def collect(
    out_dir: Path,
    slots: list[AdapterSlot],
    tiers: TierTable,
    *,
    tail_indices: set[int],
    slurm_output: Path | None,
    job: str | None,
) -> tuple[list[AdapterGeneration], list[int]]:
    """One row per slot that carries a DONE marker; the indices of those that do not."""
    rows: list[AdapterGeneration] = []
    missing: list[int] = []
    for slot in slots:
        marker: DoneMarker | None = read_done(out_dir, slot)
        if marker is None:
            missing.append(slot.index)
            continue
        tier = tiers.tier_for(slot.adapter_class)
        samples = (
            out_dir / "adapters" / slot.dirname / "telemetry" / "resource_samples.csv"
        )
        log = (
            slurm_output / f"{job}_{slot.index}_generate.log"
            if slurm_output and job
            else None
        )
        phases: list[PhaseResources] = []
        if samples.is_file():
            recycles = read_recycles(log if log and log.is_file() else None)
            phases = fold_samples(samples, recycles)
        task_log = (
            slurm_output / f"{job}_{slot.index}_bact_gen_standard.out"
            if slurm_output and job
            else None
        )
        rows.append(
            AdapterGeneration(
                index=slot.index,
                dataset_class=slot.dataset_class,
                adapter_class=slot.adapter_class,
                tier=tier.name,
                tier_cpus=tier.cpus,
                tier_mem_gb=tier.mem_gb,
                records=marker.dataset_len,
                n_nodes=marker.n_nodes,
                n_edges=marker.n_edges,
                wall_s=marker.generation_wall_s,
                peak_memory_gb=marker.peak_memory_gb,
                tier_memory_fraction=round(marker.peak_memory_gb / tier.mem_gb, 4),
                cores_mean=(
                    round(
                        sum(p.cores_mean * p.samples for p in phases)
                        / sum(p.samples for p in phases),
                        2,
                    )
                    if phases and sum(p.samples for p in phases)
                    else None
                ),
                cores_max=round(max(p.cores_max for p in phases), 2)
                if phases
                else None,
                mem_max_gb_telemetry=(
                    round(max(p.mem_max_gb for p in phases), 2) if phases else None
                ),
                recycles=sum(p.recycles for p in phases) if phases else None,
                phases=phases,
                startup_s=(generation_startup_seconds(task_log) if task_log else None),
                torchcell_version=marker.torchcell_version,
                build_commit=marker.build_commit,
                in_tail=slot.index in tail_indices,
            )
        )
    return rows, missing


# --------------------------------------------------------------------- verdicts


def _next_tier_holding(
    tiers: TierTable, peak_gb: float, fraction: float, current: str
) -> tuple[str, str]:
    """The smallest tier whose memory holds ``peak_gb`` under ``fraction``."""
    ordered = sorted(tiers.tiers.items(), key=lambda kv: kv[1].mem_gb)
    for name, tier in ordered:
        if peak_gb / tier.mem_gb <= fraction:
            if name == current:
                return current, (
                    f"{peak_gb:.1f} GiB is {peak_gb / tier.mem_gb:.0%} of the "
                    f"{tier.mem_gb} G {name} tier, inside {fraction:.0%}"
                )
            return name, (
                f"{peak_gb:.1f} GiB is {peak_gb / tier.mem_gb:.0%} of the "
                f"{tier.mem_gb} G {name} tier against "
                f"{peak_gb / tiers.tiers[current].mem_gb:.0%} of {current}"
            )
    largest = ordered[-1]
    return largest[0], (
        f"{peak_gb:.1f} GiB is {peak_gb / largest[1].mem_gb:.0%} of the largest tier "
        f"({largest[0]}, {largest[1].mem_gb} G); it needs a tier of its own"
    )


def verdicts(
    rows: list[AdapterGeneration], tiers: TierTable, warn_fraction: float
) -> list[TierVerdict]:
    """One verdict per row, flagged when the peak passes ``warn_fraction`` of its tier."""
    out = []
    for row in sorted(rows, key=lambda r: -r.tier_memory_fraction):
        over = row.tier_memory_fraction > warn_fraction
        proposed, reason = (
            _next_tier_holding(tiers, row.peak_memory_gb, warn_fraction, row.tier)
            if over
            else (
                row.tier,
                f"{row.peak_memory_gb:.1f} GiB is "
                f"{row.tier_memory_fraction:.0%} of the {row.tier_mem_gb} G "
                f"{row.tier} tier",
            )
        )
        out.append(
            TierVerdict(
                adapter_class=row.adapter_class,
                tier=row.tier,
                peak_memory_gb=row.peak_memory_gb,
                tier_mem_gb=row.tier_mem_gb,
                fraction=row.tier_memory_fraction,
                over_warn=over,
                proposed_tier=proposed,
                reason=reason,
            )
        )
    return out


# ------------------------------------------------------------------- projection


#: The 5 s sampling interval of ``torchcell.build_telemetry``: an adapter the serial
#: run produced no sample for ran for less than one interval, so one interval is an
#: upper bound on its duration and is what the projection charges it.
SAMPLER_INTERVAL_S = 5.0


def serial_adapter_seconds(telemetry: Path) -> dict[str, float]:
    """Per-adapter seconds of a serial generation run, folded from its cgroup samples.

    Job 4901 is the only run that has ever generated the yeast adapters, and it ran
    them in ONE 64 CPU container, not in per-adapter 16 CPU containers, so these
    seconds are a measurement of a different configuration: for the adapters that were
    core-bound in it (Costanzo DMF at 60.8 of 64 cores) the array's smaller tier will
    be slower, and the projection says so rather than hiding it.
    """
    phases = fold_samples(telemetry, [])
    seconds: dict[str, float] = {}
    for phase in phases:
        seconds[phase.adapter] = seconds.get(phase.adapter, 0.0) + phase.hours * 3600
    return seconds


class MeasuredElsewhere(BaseModel):
    """A measured task seconds figure this run did not produce, with its source."""

    adapter_class: str
    tier: str
    seconds: float
    source: str


def project(
    rows: list[AdapterGeneration],
    tiers: TierTable,
    *,
    order: list[AdapterSlot],
    elsewhere: list[MeasuredElsewhere],
    serial_seconds: dict[str, float],
    startup_s: float,
) -> tuple[list[TierProjection], str, dict[str, float]]:
    """Per-tier wall clock of the full array, charging every task a measured number.

    Each of the 170 tasks is charged, in this order of preference: this run's own
    measurement at the task's real tier; a measurement another run made at that tier
    (``elsewhere``: the Hoepfner study's 320 G baseline, lane P's Caudal); the serial
    run's per-adapter seconds; and ``SAMPLER_INTERVAL_S`` for an adapter the serial run
    sampled not even once, which bounds it above. Nothing is extrapolated, and the
    second return value names the adapter that sets the array's floor.
    """
    own = {row.adapter_class: row.wall_s for row in rows}
    other = {item.adapter_class: item.seconds for item in elsewhere}
    charged: dict[str, float] = {}
    basis_counts: dict[str, dict[str, int]] = {}
    per_tier: dict[str, list[float]] = {}
    for slot in order:
        slot_tier = tiers.tier_for(slot.adapter_class).name
        adapter = slot.adapter_class
        if adapter in own:
            seconds, source = own[adapter], "this run"
        elif adapter in other:
            seconds, source = other[adapter], "another run at this tier"
        elif adapter in serial_seconds:
            seconds, source = serial_seconds[adapter], "the serial run"
        else:
            seconds, source = SAMPLER_INTERVAL_S, "under one sampling interval"
        charged[adapter] = seconds
        per_tier.setdefault(slot_tier, []).append(seconds + startup_s)
        counts = basis_counts.setdefault(slot_tier, {})
        counts[source] = counts.get(source, 0) + 1
    projections: list[TierProjection] = []
    for tier_name, tier_spec in tiers.tiers.items():
        if tier_name not in per_tier:
            continue
        charges = per_tier[tier_name]
        basis = basis_counts[tier_name]
        measured_here = basis.get("this run", 0) + basis.get(
            "another run at this tier", 0
        )
        projections.append(
            TierProjection(
                tier=tier_name,
                tasks=len(charges),
                concurrency=tier_spec.concurrency,
                measured_tasks=measured_here,
                measured_task_seconds=round(
                    sum(
                        charged[slot.adapter_class]
                        for slot in order
                        if tiers.tier_for(slot.adapter_class).name == tier_name
                        and (slot.adapter_class in own or slot.adapter_class in other)
                    ),
                    1,
                ),
                assumed_task_seconds=round(
                    (sum(charges) / len(charges)) - startup_s, 1
                ),
                startup_s=startup_s,
                total_task_seconds=round(sum(charges), 1),
                wall_hours=round(sum(charges) / tier_spec.concurrency / 3600, 2),
                basis=", ".join(
                    f"{count} {reason}" for reason, count in sorted(basis.items())
                )
                + f"; {startup_s:.0f} s of startup each, at %{tier_spec.concurrency}",
            )
        )
    slowest = max(charged.items(), key=lambda kv: kv[1])
    return (
        projections,
        f"{slowest[0]} at {slowest[1]:.0f} s ({slowest[1] / 3600:.2f} h)",
        charged,
    )


# ----------------------------------------------------------------------- tables


def membership_table(slots: list[AdapterSlot], tiers: TierTable) -> str:
    """The tail's membership, as the note carries it."""
    lines = ["| index | dataset | adapter | tier |", "|---:|---|---|---|"]
    for slot in slots:
        lines.append(
            f"| {slot.index} | {slot.dataset_class} | {slot.adapter_class} "
            f"| {tiers.tier_for(slot.adapter_class).name} |"
        )
    return "\n".join(lines)


def rows_table(rows: list[AdapterGeneration]) -> str:
    """Per-adapter generation table, in build order."""
    lines = [
        "| index | adapter | records | node rows | edge rows | wall s | peak GiB "
        "| % of tier | cores mean | recycles |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r.index):
        lines.append(
            f"| {row.index} | {row.adapter_class} | {row.records:,} | "
            f"{row.n_nodes:,} | {row.n_edges:,} | {row.wall_s:.0f} | "
            f"{row.peak_memory_gb:.2f} | {row.tier_memory_fraction:.1%} | "
            f"{'-' if row.cores_mean is None else f'{row.cores_mean:.1f}'} | "
            f"{'-' if row.recycles is None else row.recycles} |"
        )
    return "\n".join(lines)


def extremes_table(rows: list[AdapterGeneration], n: int = 5) -> str:
    """The slowest and the largest rows."""
    slow = sorted(rows, key=lambda r: -r.wall_s)[:n]
    big = sorted(rows, key=lambda r: -r.peak_memory_gb)[:n]
    lines = ["| rank | slowest (wall s) | largest (peak GiB) |", "|---:|---|---|"]
    for i in range(n):
        lines.append(
            f"| {i + 1} | {slow[i].adapter_class} {slow[i].wall_s:.0f} s "
            f"({slow[i].records:,} records) | {big[i].adapter_class} "
            f"{big[i].peak_memory_gb:.2f} GiB ({big[i].tier_memory_fraction:.1%} "
            f"of {big[i].tier_mem_gb} G) |"
        )
    return "\n".join(lines)


def verdict_table(verdicts_: list[TierVerdict], warn_fraction: float) -> str:
    """Every adapter over the warning fraction, and the tier proposed for it."""
    over = [v for v in verdicts_ if v.over_warn]
    if not over:
        return (
            f"No adapter of the measured set passes {warn_fraction:.0%} of its tier's "
            f"memory. The largest is {verdicts_[0].adapter_class} at "
            f"{verdicts_[0].peak_memory_gb:.2f} GiB, {verdicts_[0].fraction:.1%} of "
            f"the {verdicts_[0].tier_mem_gb} G {verdicts_[0].tier} tier."
        )
    lines = [
        "| adapter | tier | peak GiB | % of tier | proposed tier | reason |",
        "|---|---|---:|---:|---|---|",
    ]
    for v in over:
        lines.append(
            f"| {v.adapter_class} | {v.tier} | {v.peak_memory_gb:.2f} | "
            f"{v.fraction:.1%} | {v.proposed_tier} | {v.reason} |"
        )
    return "\n".join(lines)


def projection_table(projections: list[TierProjection], critical: str) -> str:
    """Per-tier projected wall clock of the full array."""
    lines = [
        "| tier | tasks | concurrency | measured tasks | task seconds | "
        "projected wall h | basis |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for p in projections:
        lines.append(
            f"| {p.tier} | {p.tasks} | {p.concurrency} | {p.measured_tasks} | "
            f"{p.total_task_seconds:,.0f} | {p.wall_hours:.2f} | {p.basis} |"
        )
    total = sum(p.wall_hours for p in projections)
    lines.append(
        f"\nSerial sum of the tiers: **{total:.2f} h projected**; the tiers run "
        f"concurrently in the real build, so the array's calendar is at least the "
        f"largest row. Critical path adapter: {critical}."
    )
    return "\n".join(lines)


# -------------------------------------------------------------------------- CLI


def main() -> None:
    """Read one array's output, write the JSON, print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--tiers", type=Path, default=None)
    parser.add_argument("--tail-from", default=DEFAULT_TAIL_FROM)
    parser.add_argument(
        "--extra-index",
        type=int,
        action="append",
        default=[],
        help="a slot outside the tail to measure as well",
    )
    parser.add_argument("--job", default=None)
    parser.add_argument("--slurm-output", type=Path, default=None)
    parser.add_argument(
        "--serial-telemetry",
        type=Path,
        default=None,
        help="resource_samples.csv of the serial run (job 4901) for the other tasks",
    )
    parser.add_argument("--label", default="bacterial-tail")
    parser.add_argument("--tier-warn-fraction", type=float, default=0.5)
    parser.add_argument(
        "--startup-s",
        type=float,
        default=15.8,
        help="container startup charged per task in the projection (lane C2's image)",
    )
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    tiers = load_tier_table(args.tiers) if args.tiers else load_tier_table()
    tail = tail_slots(args.tail_from)
    order = adapter_order(True, ALL_ORGANISMS)
    by_index = {slot.index: slot for slot in order}
    extra = [by_index[i] for i in sorted(set(args.extra_index))]
    slots = extra + tail

    if args.list:
        print(f"{len(tail)} adapters after {args.tail_from}")
        print(membership_table(tail, tiers))
        if extra:
            print(f"\nplus {len(extra)} outside the tail")
            print(membership_table(extra, tiers))
        return

    if args.out_dir is None:
        parser.error("--out-dir is required unless --list")
    rows, missing = collect(
        args.out_dir,
        slots,
        tiers,
        tail_indices={slot.index for slot in tail},
        slurm_output=args.slurm_output,
        job=args.job,
    )
    verds = verdicts(rows, tiers, args.tier_warn_fraction)
    elsewhere = [
        MeasuredElsewhere(
            adapter_class="EnvChemgenHoepfner2014Adapter",
            tier="hoepfner",
            seconds=4558.0,
            source="job 4919_64, the Hoepfner study baseline at 320 G",
        ),
        MeasuredElsewhere(
            adapter_class="CaudalPanTranscriptome2024Adapter",
            tier="large",
            seconds=727.0,
            source="lane P, job 4962",
        ),
    ]
    serial = (
        serial_adapter_seconds(args.serial_telemetry) if args.serial_telemetry else {}
    )
    projections, critical, charged = project(
        rows,
        tiers,
        order=order,
        elsewhere=elsewhere,
        serial_seconds=serial,
        startup_s=args.startup_s,
    )
    profile = Profile(
        label=args.label,
        out_dir=str(args.out_dir),
        job=args.job,
        build_commit=rows[0].build_commit if rows else None,
        tail_from=args.tail_from,
        tail_indices=[slot.index for slot in tail],
        extra_indices=[slot.index for slot in extra],
        tier_warn_fraction=args.tier_warn_fraction,
        rows=rows,
        missing=missing,
        verdicts=verds,
        projection=projections,
        projection_critical_path=critical,
        projection_startup_s=args.startup_s,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"bacterial_generation_profile_{args.label}.json"
    payload: dict[str, Any] = profile.model_dump()
    payload["measured_elsewhere"] = [item.model_dump() for item in elsewhere]
    payload["projection_seconds_charged"] = {
        k: round(v, 1) for k, v in sorted(charged.items(), key=lambda kv: -kv[1])
    }
    payload["serial_run_adapters_sampled"] = len(serial)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"{len(rows)} adapters measured, {len(missing)} without a DONE marker "
        f"{missing if missing else ''} -> {out}"
    )
    if args.table:
        print("\n#### Membership\n")
        print(membership_table(tail, tiers))
        print("\n#### Per adapter\n")
        print(rows_table(rows))
        print("\n#### Extremes\n")
        print(extremes_table(rows))
        print("\n#### Tier verdicts\n")
        print(verdict_table(verds, args.tier_warn_fraction))
        print("\n#### Projection\n")
        print(projection_table(projections, critical))


if __name__ == "__main__":
    main()
