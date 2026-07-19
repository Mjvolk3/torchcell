# experiments/database/scripts/project_build_time.py
# [[experiments.database.scripts.project_build_time]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/project_build_time
r"""Project the tcdb KG BUILD (CSV-generation) time for several subset configs.

Calibrated on job 966 (config ``kg_full``: dmf/dmi Costanzo capped to 100k, all
other datasets full). Prints, for each config, a per-adapter contribution table
(adapter, records, rate s/rec, projected h, % of total) and a headline total, then
saves everything to ``experiments/database/results/build_time_projection.json``.

Configs projected:
  (a) kg_full        -- reproduces the measured ~9.22 h (measured-vs-projected self-check)
  (b) uncapped       -- every dataset full (dmf/dmi Costanzo at 20.7M each)
  (c) kg_small       -- subset.size = 1000 (fast test build)
  (d) kuzmin_100k    -- cap the whole Kuzmin family to 100k, else full
  (e) costanzo_500k  -- cap dmf/dmi Costanzo to 500k, else full

Run from the repo root:
  python experiments/database/scripts/project_build_time.py
"""

from __future__ import annotations

import json
from pathlib import Path

from torchcell.knowledge_graphs.build_time_projection import (
    DATASET_FULL_RECORDS,
    KG_FULL_CONFIG,
    AdapterRate,
    BuildTimeProjection,
    SubsetConfig,
    calibrate,
    load_timings,
    project_build_time,
)

SCRIPT = Path(__file__).resolve()
TIMINGS_PATH = SCRIPT.parent / "build966_timings.json"
RESULTS = SCRIPT.parent.parent / "results"

KUZMIN_FAMILY = [
    "SmfKuzmin2018Dataset",
    "DmfKuzmin2018Dataset",
    "TmfKuzmin2018Dataset",
    "DmiKuzmin2018Dataset",
    "TmiKuzmin2018Dataset",
    "SmfKuzmin2020Dataset",
    "DmfKuzmin2020Dataset",
    "TmfKuzmin2020Dataset",
    "DmiKuzmin2020Dataset",
    "TmiKuzmin2020Dataset",
]

# (label, human title, config, rows to print [None => all])
CONFIGS: list[tuple[str, str, SubsetConfig, int | None]] = [
    (
        "kg_full",
        "(a) kg_full  [calibration config -- self-check]",
        KG_FULL_CONFIG,
        None,
    ),
    (
        "uncapped",
        "(b) uncapped  [every dataset full; dmf/dmi Costanzo 20.7M]",
        SubsetConfig(subset_size=None, per_dataset={}),
        None,
    ),
    (
        "kg_small",
        "(c) kg_small  [subset.size = 1000]",
        SubsetConfig(subset_size=1000, per_dataset={}),
        15,
    ),
    (
        "kuzmin_100k",
        "(d) kuzmin_100k  [Kuzmin family capped to 100k, else full]",
        SubsetConfig(subset_size=None, per_dataset={n: 100000 for n in KUZMIN_FAMILY}),
        15,
    ),
    (
        "costanzo_500k",
        "(e) costanzo_500k  [dmf/dmi Costanzo capped to 500k, else full]",
        SubsetConfig(
            subset_size=None,
            per_dataset={
                "DmfCostanzo2016Dataset": 500000,
                "DmiCostanzo2016Dataset": 500000,
            },
        ),
        15,
    ),
]


def fmt_hms(sec: float) -> str:
    """Format seconds as ``H.HH h (Hh Mm)`` (and days when large)."""
    h = sec / 3600.0
    if h < 24:
        return f"{h:.2f} h ({int(sec // 3600)}h {int((sec % 3600) // 60)}m)"
    return f"{h:.2f} h ({h / 24:.2f} days)"


def format_table(proj: BuildTimeProjection, top: int | None) -> str:
    """Render a projection's per-adapter contribution table (top rows, or all)."""
    rows = proj.contributions if top is None else proj.contributions[:top]
    head = (
        f"    {'#':>2}  {'adapter':<38} {'records':>12}  "
        f"{'rate s/rec':>11}  {'proj h':>9}  {'% tot':>6}"
    )
    lines = [head, "    " + "-" * (len(head) - 4)]
    for i, c in enumerate(rows, 1):
        lines.append(
            f"    {i:>2}  {c.adapter:<38} {c.records:>12,}  "
            f"{c.rate_s_per_rec:>11.5f}  {c.projected_hours:>9.3f}  {c.pct_of_total:>6.2f}"
        )
    if top is not None and len(proj.contributions) > top:
        rest = proj.contributions[top:]
        rest_h = sum(c.projected_hours for c in rest)
        rest_pct = sum(c.pct_of_total for c in rest)
        lines.append(
            f"    ..  {f'({len(rest)} more adapters)':<38} {'':>12}  "
            f"{'':>11}  {rest_h:>9.3f}  {rest_pct:>6.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Calibrate on job 966, project every config, print tables, and save JSON."""
    timings = load_timings(TIMINGS_PATH)
    rates: list[AdapterRate] = calibrate(timings, DATASET_FULL_RECORDS)

    measured = timings.generation_total_sec
    projections: dict[str, BuildTimeProjection] = {}

    print("=" * 84)
    print(
        f"tcdb KG build-time projection  (calibrated on job 966; measured "
        f"generation = {measured:,.0f} s = {measured / 3600:.2f} h)"
    )
    print("=" * 84)

    for label, title, cfg, top in CONFIGS:
        ms = measured if label == "kg_full" else None
        proj = project_build_time(
            rates, DATASET_FULL_RECORDS, cfg, label=label, measured_sec=ms
        )
        projections[label] = proj
        print(f"\n{title}")
        print(f"    TOTAL: {proj.total_sec:,.0f} s  =  {fmt_hms(proj.total_sec)}")
        if proj.measured_sec is not None and proj.error_pct is not None:
            print(
                f"    SELF-CHECK: measured {proj.measured_sec:,.0f} s vs projected "
                f"{proj.total_sec:,.0f} s  ->  error {proj.error_pct:+.4f} %"
            )
        print(format_table(proj, top))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "build_time_projection.json"
    payload = {
        "calibration_job": 966,
        "measured_generation_sec": measured,
        "measured_generation_hours": measured / 3600.0,
        "rates": [r.model_dump() for r in rates],
        "projections": {k: v.model_dump() for k, v in projections.items()},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
