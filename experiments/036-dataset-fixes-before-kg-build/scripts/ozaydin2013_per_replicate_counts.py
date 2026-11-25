# experiments/036-dataset-fixes-before-kg-build/scripts/ozaydin2013_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.ozaydin2013_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/ozaydin2013_per_replicate_counts
"""Prove the per-replicate ``CarotenoidOzaydin2013Dataset`` counts against the release.

Reads two already-built stores, record by record with interned ``$ref`` pointers
resolved: ``--before-root`` (origin/main, Sheet 1 only, 4,607) and ``--after-root``
(Sheet 1 plus every numeric ``2nd Screen`` cell of ``2ndRoundOfTransformations``). Both
are built into scratch roots, never the shared dev tree. It reports:

- the release side, read from the SI itself (sha256 verified first): Sheet 1 rows and
  numeric Color cells; second-round strain rows, numeric and text ``2nd Screen`` cells,
  repeated ORFs; the ``1st Screen`` cells that are not among the Sheet 1 Color cells of
  their ORF; and the TOP200 Color cells that differ from Sheet 1 at the same
  (Plate, Row, Col), a second in-release disagreement over the first round;
- records per ``replicate_id`` kind (round 1 / round 2) and the total, before and after;
- that every before-record equals the after-record at the same index once the two
  fields this change adds (``replicate_id``, ``provenance_gaps``) are set aside, and
  which after-records are new;
- the mean-reproduction check: after-records grouped by (genotype, environment),
  averaged and compared with the before-records of that group. A second-round deletion
  carries no collection position, so its genotype is its own group; the same check with
  every ``construction`` set aside (one group per deleted gene) shows how far the second
  round moves a gene's mean, which is what a model sees when it pools by gene.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/ozaydin2013_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/ozaydin2013_per_replicate_counts.py \
        --before-root <scratch>/before/carotenoid_ozaydin2013 \
        --after-root <scratch>/after/carotenoid_ozaydin2013
"""

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter, defaultdict
from typing import Any

import lmdb
import pandas as pd

from torchcell.data import verify_raw_files
from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.ozaydin2013 import (
    _SI_SHA256,
    ROUND_1_SHEET,
    ROUND_2_SHEET,
    TOP200_SHEET,
    CarotenoidOzaydin2013Dataset,
)

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
ADDED_FIELDS = ("replicate_id", "provenance_gaps")


def read_store(root: str) -> list[dict[str, Any]]:
    """Every record of ``<root>/processed/lmdb`` in index order, interned refs resolved."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        records = [
            resolve_interned(pickle.loads(txn.get(str(i).encode())), interned)
            for i in range(n)
        ]
    env.close()
    return records


def _dump(record: dict[str, Any]) -> str:
    """A canonical JSON string of one whole record."""
    return json.dumps(record, sort_keys=True, default=str)


def _without_added_fields(record: dict[str, Any]) -> str:
    """The record with the phenotype fields this change sets put back to their defaults."""
    stripped = json.loads(_dump(record))
    phenotype = stripped["experiment"]["phenotype"]
    phenotype["replicate_id"] = None
    phenotype["provenance_gaps"] = []
    return _dump(stripped)


def _key(experiment: dict[str, Any], keep_construction: bool) -> str:
    """The (genotype, environment) group key, optionally without each construction."""
    perturbations = [
        {k: v for k, v in p.items() if keep_construction or k != "construction"}
        for p in experiment["genotype"]["perturbations"]
    ]
    return json.dumps([perturbations, experiment["environment"]], sort_keys=True)


def _value(record: dict[str, Any]) -> float:
    return float(record["experiment"]["phenotype"]["visual_score"])


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not pd.isna(value)


def _text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _release_side(xlsx: str) -> dict[str, Any]:
    """Counts read straight off the SI workbook."""
    verify_raw_files(
        osp.dirname(xlsx), {CarotenoidOzaydin2013Dataset.si_filename: _SI_SHA256}
    )
    sheet1 = pd.read_excel(xlsx, sheet_name=ROUND_1_SHEET)
    sheet1["ORF"] = sheet1["ORF name"].astype(str).str.strip().str.upper()
    top200 = pd.read_excel(xlsx, sheet_name=TOP200_SHEET)
    top200 = top200[top200["ORF name"].notna()].copy()
    top200["ORF"] = top200["ORF name"].astype(str).str.strip().str.upper()
    raw2 = pd.read_excel(xlsx, sheet_name=ROUND_2_SHEET, header=None)
    body = raw2.iloc[3:, :4]
    body.columns = ["Systematic name", "Gene Name", "1st Screen", "2nd Screen"]
    body = body[body.notna().any(axis=1)]
    numeric2 = body["2nd Screen"].map(_is_number)
    wt = body["Systematic name"].isna() & (body["Gene Name"] == "WT")
    sys_names = body["Systematic name"].dropna().astype(str).str.strip().str.upper()
    repeated = sys_names.value_counts()
    colors = sheet1.groupby("ORF")["Color"].apply(lambda s: [_text(v) for v in s])
    first_off = []
    for position, cells in body[~wt].iterrows():
        orf = (
            str(cells["Systematic name"]).strip().upper()
            if pd.notna(cells["Systematic name"])
            else str(cells["Gene Name"]).strip().upper()
        )
        sheet1_colors = colors.get(orf, [])
        if _text(cells["1st Screen"]) not in sheet1_colors:
            first_off.append(
                {
                    "sheet_row": int(str(position)) + 1,
                    "orf": orf,
                    "first_screen": _text(cells["1st Screen"]),
                    "sheet1_colors": sheet1_colors,
                }
            )
    top200_off = []
    for _, r in top200.iterrows():
        at = sheet1[
            (sheet1["ORF"] == r["ORF"])
            & (sheet1["Plate"] == r["Plate"])
            & (sheet1["Row"].astype(str).str.upper() == str(r["Row"]).upper())
            & (sheet1["Col"] == r["Col"])
        ]
        if len(at) != 1:
            raise RuntimeError(f"TOP200 {r['ORF']} has {len(at)} Sheet 1 positions")
        if _text(at["Color"].iloc[0]) != _text(r["Color"]):
            top200_off.append(
                {
                    "orf": r["ORF"],
                    "plate_row_col": f"{_text(r['Plate'])}/{r['Row']}/{_text(r['Col'])}",
                    "top200_color": _text(r["Color"]),
                    "sheet1_color": _text(at["Color"].iloc[0]),
                }
            )
    return {
        "sheet1_rows": len(sheet1),
        "sheet1_numeric_color": int(sheet1["Color"].map(_is_number).sum()),
        "round2_rows": len(body),
        "round2_strain_rows": int((~wt).sum()),
        "round2_strain_rows_without_systematic_name": int(
            (body["Systematic name"].isna() & ~wt).sum()
        ),
        "round2_wt_rows": int(wt.sum()),
        "round2_numeric_second_screen": int(numeric2.sum()),
        "round2_numeric_second_screen_strain_rows": int((numeric2 & ~wt).sum()),
        "round2_text_second_screen": int((~numeric2).sum()),
        "round2_numeric_second_screen_values": {
            _text(k): int(v)
            for k, v in sorted(Counter(body.loc[numeric2, "2nd Screen"]).items())
        },
        "round2_orfs_on_more_than_one_row": {
            str(k): int(v) for k, v in repeated[repeated > 1].items()
        },
        "round2_first_screen_not_among_sheet1_colors": first_off,
        "top200_color_differs_from_sheet1_at_same_position": top200_off,
    }


def main() -> None:
    """Read both stores and the SI, compute counts and the mean check, write JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()

    release = _release_side(
        osp.join(args.after_root, "raw", CarotenoidOzaydin2013Dataset.si_filename)
    )
    before = read_store(args.before_root)
    after = read_store(args.after_root)

    same_index = [
        _without_added_fields(a) == _dump(b)
        for a, b in zip(after, before, strict=False)
    ]
    added = after[len(before) :]

    def kind(record: dict[str, Any]) -> str:
        rid = record["experiment"]["phenotype"]["replicate_id"]
        return "none" if rid is None else rid.split(",")[0]

    checks: dict[str, Any] = {}
    for label, keep in (
        ("by_genotype_with_construction", True),
        ("by_genotype_without_construction", False),
    ):
        groups_after: dict[str, list[float]] = defaultdict(list)
        groups_before: dict[str, list[float]] = defaultdict(list)
        for r in after:
            groups_after[_key(r["experiment"], keep)].append(_value(r))
        for r in before:
            groups_before[_key(r["experiment"], keep)].append(_value(r))
        if set(groups_before) - set(groups_after):
            raise RuntimeError(f"{label}: a before-group has no after-records")
        diffs = {
            k: abs(sum(v) / len(v) - sum(groups_before[k]) / len(groups_before[k]))
            for k, v in groups_after.items()
            if k in groups_before
        }
        differing = {k: d for k, d in diffs.items() if d > 0}
        checks[label] = {
            "groups_before": len(groups_before),
            "groups_after": len(groups_after),
            "groups_after_not_in_before": len(set(groups_after) - set(groups_before)),
            "max_abs_diff": max(diffs.values()),
            "n_groups_differing": len(differing),
        }

    summary = {
        "release": release,
        "before_records": len(before),
        "after_records": len(after),
        "before_by_replicate_kind": dict(Counter(kind(r) for r in before)),
        "after_by_replicate_kind": dict(Counter(kind(r) for r in after)),
        "before_records_equal_after_at_same_index_without_added_fields": sum(
            same_index
        ),
        "after_records_new": len(added),
        "after_new_wild_type_records": sum(
            1
            for r in added
            if all(
                p["perturbation_type"] == "gene_addition"
                for p in r["experiment"]["genotype"]["perturbations"]
            )
        ),
        "after_new_released_identifier_records": [
            p["systematic_gene_name"]
            for r in added
            for p in r["experiment"]["genotype"]["perturbations"]
            if p["perturbation_type"] == "released_identifier"
        ],
        "after_replicate_ids_distinct_among_new": len(
            {r["experiment"]["phenotype"]["replicate_id"] for r in added}
        ),
        "references_equal_to_before": {_dump(r["reference"]) for r in after}
        == {_dump(r["reference"]) for r in before},
        "mean_reproduction": checks,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(
        osp.join(RESULTS, "ozaydin2013_per_replicate_counts.json"), "w"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
