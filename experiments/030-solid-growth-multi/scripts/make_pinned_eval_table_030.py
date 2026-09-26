# experiments/030-solid-growth-multi/scripts/make_pinned_eval_table_030.py
# [[experiments.030-solid-growth-multi.scripts.make_pinned_eval_table_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/make_pinned_eval_table_030
"""The one (target, source) per pinned triple per label that val/test scoring uses.

The 030 build keeps every source entry under a genotype, so a pinned triple can carry
a Kuzmin 2018 and a Kuzmin 2020 entry of the same label. Validation and test score
exactly one target per (record, label): the entry ``LabelPolicy`` chooses (precedence
kuzmin2018 before kuzmin2020, converted zeros refused where a measurement exists,
same-source replicates combined by inverse variance when every SE is positive, else by
plain mean). In-batch the trainer averages the same-source rows with a PLAIN mean, so
this table records, per pair, how many same-source entries exist and
|plain mean - policy value|, making the deviation from inverse-variance weighting a
logged number rather than an assumption.

Entries come from ``closure/entries.parquet`` (the same fields ``entries_of_record``
reads from the LMDB bytes: dataset, experiment type, value, temperature, sd, n_samples,
p, strain_id). ``LabelPolicy`` requires a name, which enters the hash; the name used is
the module's own documented default, ``kuzmin-first``, and the resulting ``policy_id``
is recorded.

Writes (``experiments/030-solid-growth-multi/results/``):

- ``pinned_eval_table_030.parquet``: columns idx, split, label, dataset, source, value,
  n_entries_available, n_entries_combined, plain_mean_of_same_source, abs_diff. Pairs
  the policy leaves empty are counted in the summary and absent from the table.
- ``pinned_eval_table_030_summary.json``: ``PinnedEvalSummary``.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/make_pinned_eval_table_030.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import os.path as osp
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.data.label_policy import LabelEntry, LabelPolicy, source_key

POLICY_NAME = "kuzmin-first"
LABELS = ("fitness", "gene_interaction")
LABEL_OF_EXP_TYPE = {"fitness": "fitness", "gene interaction": "gene_interaction"}


class PinnedEvalSummary(BaseModel):
    """Counts over the pinned val/test (record, label) pairs under the policy."""

    policy_name: str
    policy_id: str
    policy: dict[str, Any]
    entry_source: str
    n_pinned_records: dict[str, int]
    n_pairs_expected: int
    n_rows: int
    n_pairs_empty: int
    n_pairs_multi_same_source: int
    n_pairs_multi_available: int
    max_abs_diff: float
    mean_abs_diff_over_multi: float
    rows_by_label_and_source: dict[str, int]
    rows_by_dataset: dict[str, int]


def _finite(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def entries_with_datasets(
    rows: list[dict[str, Any]],
) -> tuple[list[LabelEntry], list[str]]:
    """``LabelEntry`` objects of one record's parquet rows, with the dataset per entry.

    Mirrors ``label_policy.entries_from_records`` but keeps the dataset name beside each
    entry, because the policy reports a ``source_key`` and the table needs the dataset.
    Rows with a non-finite value are skipped, as the policy's own normalizer does.
    """
    entries: list[LabelEntry] = []
    datasets: list[str] = []
    for r in rows:
        value = _finite(r["value"])
        if value is None:
            continue
        n = _finite(r["n_samples"])
        entries.append(
            LabelEntry(
                source=source_key(str(r["dataset"]), _finite(r["temp"])),
                label=LABEL_OF_EXP_TYPE[str(r["exp_type"])],
                value=value,
                sd=_finite(r["sd"]),
                n_samples=None if n is None else int(n),
                p_value=_finite(r["p"]),
                strain_id=None if r["strain_id"] is None else str(r["strain_id"]),
            )
        )
        datasets.append(str(r["dataset"]))
    return entries, datasets


def chosen_pool(
    policy: LabelPolicy, entries: list[LabelEntry], label: str, source: str
) -> list[int]:
    """Positions of the entries the policy combined for ``label`` from ``source``.

    Reproduces ``LabelPolicy.select``'s pool: the label's entries, restricted to measured
    ones when a measurement exists and the policy refuses converted zeros beside one.
    """
    pool = [i for i, e in enumerate(entries) if e.label == label]
    measured = [i for i in pool if not entries[i].is_converted_zero]
    if policy.converted_zero_only_without_measurement and measured:
        pool = measured
    return [i for i in pool if entries[i].source == source]


def main() -> None:
    """Apply the policy to every pinned val/test record and write the table."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    entries_path = osp.join(
        data_root,
        "data/torchcell/experiments/030-solid-growth-multi/closure/entries.parquet",
    )
    results = osp.join(experiment_root, "030-solid-growth-multi", "results")

    with gzip.open(
        osp.join(results, "pinned_splits_from_010_seed_42.json.gz"), "rt"
    ) as f:
        pinned: dict[str, list[int]] = json.load(f)["pinned"]
    split_of = {i: s for s in ("val", "test") for i in pinned[s]}
    assert len(split_of) == len(pinned["val"]) + len(pinned["test"]), "val/test overlap"

    policy = LabelPolicy(name=POLICY_NAME)
    print(f"policy {POLICY_NAME} id {policy.policy_id}", flush=True)

    entries = pd.read_parquet(
        entries_path,
        columns=[
            "idx",
            "order",
            "dataset",
            "exp_type",
            "temp",
            "value",
            "sd",
            "n_samples",
            "p",
            "strain_id",
        ],
    )
    entries = entries[entries["idx"].isin(split_of)]
    assert (entries["order"] == 3).all(), "a pinned record is not a triple"
    assert entries["idx"].nunique() == len(split_of), "a pinned record has no entries"

    rows: list[dict[str, Any]] = []
    n_empty = 0
    for _, group in entries.groupby("idx", sort=True):
        idx = int(group["idx"].iloc[0])
        records: list[dict[str, Any]] = [
            {str(k): v for k, v in r.items()} for r in group.to_dict("records")
        ]
        record_entries, datasets = entries_with_datasets(records)
        for label in LABELS:
            choice = policy.select(record_entries, label)
            if choice is None:
                n_empty += 1
                continue
            same = chosen_pool(policy, record_entries, label, choice.source)
            assert len(same) == choice.n_entries_combined, (idx, label, len(same))
            names = {datasets[i] for i in same}
            assert len(names) == 1, (idx, label, names)
            plain_mean = sum(record_entries[i].value for i in same) / len(same)
            rows.append(
                {
                    "idx": int(idx),
                    "split": split_of[int(idx)],
                    "label": label,
                    "dataset": names.pop(),
                    "source": choice.source,
                    "value": choice.value,
                    "n_entries_available": choice.n_entries_available,
                    "n_entries_combined": choice.n_entries_combined,
                    "plain_mean_of_same_source": plain_mean,
                    "abs_diff": abs(plain_mean - choice.value),
                }
            )
    table = pd.DataFrame(rows)
    table.to_parquet(osp.join(results, "pinned_eval_table_030.parquet"), index=False)

    multi = table[table["n_entries_combined"] > 1]
    by_label_source = table.groupby(["label", "source"]).size()
    summary = PinnedEvalSummary(
        policy_name=POLICY_NAME,
        policy_id=policy.policy_id,
        policy=policy.model_dump(mode="json"),
        entry_source=entries_path,
        n_pinned_records={s: len(pinned[s]) for s in ("val", "test")},
        n_pairs_expected=len(LABELS) * len(split_of),
        n_rows=len(table),
        n_pairs_empty=n_empty,
        n_pairs_multi_same_source=len(multi),
        n_pairs_multi_available=int((table["n_entries_available"] > 1).sum()),
        max_abs_diff=float(table["abs_diff"].max()),
        mean_abs_diff_over_multi=float(multi["abs_diff"].mean()) if len(multi) else 0.0,
        rows_by_label_and_source={
            f"{key[0]}/{key[1]}": int(n)
            for key, n in zip(
                by_label_source.index.tolist(), by_label_source.tolist(), strict=True
            )
        },
        rows_by_dataset={
            str(k): int(v) for k, v in table["dataset"].value_counts().items()
        },
    )
    with open(osp.join(results, "pinned_eval_table_030_summary.json"), "w") as f:
        f.write(summary.model_dump_json(indent=2))
    print(summary.model_dump_json(indent=2))
    assert summary.n_rows + summary.n_pairs_empty == summary.n_pairs_expected
    print("finished: pinned eval table written")


if __name__ == "__main__":
    main()
