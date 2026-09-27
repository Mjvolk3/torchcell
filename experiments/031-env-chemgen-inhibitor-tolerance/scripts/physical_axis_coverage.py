# experiments/031-env-chemgen-inhibitor-tolerance/scripts/physical_axis_coverage.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.physical_axis_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/physical_axis_coverage
"""Measure the physical and medium environment fields the served records carry, per dataset.

A representation that gives temperature, aerobicity, duration, pH, solvent and medium their
own input channels only learns something from them if they VARY within a dataset. If every
value is constant inside a dataset and differs between datasets, the channel is the dataset
identity written another way. This script reads the flattened served records
(``results/records_<dataset>.parquet`` from ``flatten_records.py``) and writes

- ``results/physical_axis_summary.csv``: one row per dataset with, for each field, the count
  of distinct stated values, the share of records with a stated value, and the values;
- ``results/physical_axis_fields.csv``: one row per (field, value) with the datasets that
  carry it, which is what says whether any value crosses a dataset boundary.

Every number is read from the served record. ``physical`` is the flattener's rendering of the
environment's physical perturbations (``factor=value unit[agent]``), from which the pH term is
parsed; the duration is stored either in hours or in generations, and both are reported.
"""

from __future__ import annotations

import os
import os.path as osp
import re

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
DATASETS = [
    "vanacloig2022",
    "hillenmeyer2008_hom",
    "hillenmeyer2008_het",
    "hoepfner2014",
    "wildenhain2015",
]
COLUMNS = [
    "media_base",
    "temperature_c",
    "aerobicity",
    "duration_hours",
    "duration_generations",
    "physical",
    "solvent",
    "measurement_type",
    "assay_type",
]
PH_RE = re.compile(r"pH=([0-9.]+)")


def ph_values(physical: pd.Series) -> pd.Series:
    """The pH stated as a physical perturbation, or NA when the record states none."""
    return physical.astype(str).str.extract(PH_RE, expand=False).astype(float)


def stated(series: pd.Series) -> pd.Series:
    """The series with the flattener's empty renderings turned into NA."""
    s = series.astype(object)
    s = s.where(~s.isna(), None)
    s = s.map(lambda v: None if v in ("", "None", "nan", "|") else v)
    return s.dropna()


def field_summary(name: str, values: pd.Series, n_records: int) -> dict[str, object]:
    """Distinct stated values, the share of records stating one, and the values."""
    v = stated(values)
    distinct = sorted({str(x) for x in v})
    return {
        f"{name}_n_distinct": len(distinct),
        f"{name}_frac_stated": round(len(v) / n_records, 4),
        f"{name}_values": "|".join(distinct),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    for name in DATASETS:
        df = pd.read_parquet(
            osp.join(RESULTS_DIR, f"records_{name}.parquet"), columns=COLUMNS
        )
        n = len(df)
        fields: dict[str, pd.Series] = {
            "media_base": df["media_base"],
            "temperature_c": df["temperature_c"],
            "aerobicity": df["aerobicity"],
            "duration_hours": df["duration_hours"],
            "duration_generations": df["duration_generations"],
            "ph": ph_values(df["physical"]),
            "solvent": df["solvent"],
            "measurement_type": df["measurement_type"],
            "assay_type": df["assay_type"],
        }
        row: dict[str, object] = {"dataset": name, "records": n}
        for field, values in fields.items():
            row.update(field_summary(field, values, n))
            for value, count in stated(values).astype(str).value_counts().items():
                field_rows.append(
                    {
                        "field": field,
                        "value": value,
                        "dataset": name,
                        "records": int(count),
                        "frac_of_dataset": round(int(count) / n, 4),
                    }
                )
        rows.append(row)
        print(
            f"{name}: {n:,} records; "
            + "; ".join(
                f"{f} {row[f + '_n_distinct']} distinct, "
                f"{100 * float(row[f + '_frac_stated']):.0f}% stated"
                for f in fields
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(osp.join(RESULTS_DIR, "physical_axis_summary.csv"), index=False)
    fields_df = pd.DataFrame(field_rows)
    fields_df.to_csv(osp.join(RESULTS_DIR, "physical_axis_fields.csv"), index=False)

    # How many stated values of each field appear in more than one dataset: a field whose
    # every value belongs to exactly one dataset is dataset identity under another name.
    print()
    for field, g in fields_df.groupby("field"):
        per_value = g.groupby("value")["dataset"].nunique()
        shared = per_value[per_value > 1]
        print(
            f"{field}: {len(per_value)} distinct values across the five datasets, "
            f"{len(shared)} in more than one dataset"
            + (f" ({', '.join(shared.index)})" if len(shared) else "")
        )


if __name__ == "__main__":
    main()
