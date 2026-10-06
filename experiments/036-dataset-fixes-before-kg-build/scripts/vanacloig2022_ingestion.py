# experiments/036-dataset-fixes-before-kg-build/scripts/vanacloig2022_ingestion.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.vanacloig2022_ingestion]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/vanacloig2022_ingestion
r"""Measure the Vanacloig 2022 ingestion fixes of issue #501 (and #500) before vs after.

Reads two built LMDB stores of ``EnvChemgenVanacloig2022Dataset`` directly (pickled
records + the sibling ``interned`` env, no dataset class, so a store built by older code
reads as stored):

- ``--before-root``: the store as it was (default: the dev tree,
  ``$DATA_ROOT/data/torchcell/env_chemgen_vanacloig2022``);
- ``--after-root``: the store built by the fixed loader (a scratch root before the slurm
  rebuild; the dev tree after it).

Reports, for each store: record count, conditions served (canonical compound names),
perturbations per record, the reference genome's strain and background (mating type,
ploidy, alleles and which of them are sourced vs pending), and per condition the median
response and the fraction of cells below zero. Across the two stores: the per-condition
median offset (after minus before) for the audit's six compounds and every other shared
condition, and the Spearman rank correlation of the response over the (gene, condition)
cells both stores serve. In the after store: the Pearson correlation of every
condition's response profile with the DMSO profile, over the genes both serve.

With ``--edger-tsv`` (a per-token edgeR glmQLFit logFC table with columns ``gene``,
``token``, ``logFC``, as written by the #501 audit's ``edger.R``) it also reports the
median edgeR logFC per condition and the Spearman of each store against it.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/vanacloig2022_ingestion.json``
and ``vanacloig2022_ingestion_per_condition.csv``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/vanacloig2022_ingestion.py \
        --after-root <scratch>/env_chemgen_vanacloig2022
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datamodels.compound_identity import resolved_compound

RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")

#: The audit's offset table (#501 finding 1), as matrix tokens.
AUDIT_SIX = ["CV", "NAO", "EtOH", "Acetosyringone", "GVL", "FeruloylAmide"]
#: The audit's DMSO-similar compounds (#501 finding 3), as matrix tokens.
AUDIT_DMSO_SIMILAR = [
    "FeruloylAmide",
    "Acetovanillone",
    "Acetosyringone",
    "4OHAcetophenone",
    "Acetamide",
    "CoumaroylAmide",
]


def _name(token: str) -> str:
    """The canonical compound name a matrix token is served under."""
    return resolved_compound(token).name


def read_store(root: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per record (gene, condition, response, sd, n_perturbations) + a reference."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = pickle.loads(value)
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    rows: list[tuple[str, str, float, float, int]] = []
    reference: dict[str, Any] = {}
    with env.begin() as txn:
        for _, value in txn.cursor():
            rec = resolve_interned(pickle.loads(value), interned)
            experiment = rec["experiment"]
            perts = experiment["genotype"]["perturbations"]
            (screened,) = [
                p for p in perts if p["perturbation_type"] == "barcoded_kanmx_deletion"
            ]
            (compound,) = [
                p["compound"]["name"]
                for p in experiment["environment"]["perturbations"]
                if p["perturbation_type"] == "small_molecule"
            ]
            phenotype = experiment["phenotype"]
            rows.append(
                (
                    screened["systematic_gene_name"],
                    compound,
                    float(phenotype["environment_response"]),
                    float(phenotype["environment_response_uncertainty"]),
                    len(perts),
                )
            )
            if not reference:
                reference = rec["reference"]
    env.close()
    frame = pd.DataFrame(
        rows, columns=["gene", "condition", "response", "sd", "n_perturbations"]
    )
    return frame, reference


def background_summary(reference: dict[str, Any]) -> dict[str, Any]:
    """Strain, mating type, ploidy and alleles of a reference genome."""
    genome = reference["genome_reference"]
    background = genome.get("background")
    summary: dict[str, Any] = {"strain": genome["strain"], "ploidy": genome["ploidy"]}
    if background is None:
        summary["background"] = None
        return summary
    summary["background"] = {
        "name": background["name"],
        "mating_type": background["mating_type"],
        "parents": background["parents"],
        "sourced_alleles": [
            a["allele_name"] for a in background["alleles"] if not a["provenance_gaps"]
        ],
        "pending_source_review_alleles": [
            a["allele_name"] for a in background["alleles"] if a["provenance_gaps"]
        ],
    }
    environment = reference["environment_reference"]
    summary["culture_format"] = environment.get("culture_format")
    summary["environment_gaps"] = [
        g["field"] for g in environment.get("provenance_gaps", [])
    ]
    summary["media_dropouts"] = [c["name"] for c in environment["media"]["dropouts"]]
    return summary


def store_summary(frame: pd.DataFrame, reference: dict[str, Any]) -> dict[str, Any]:
    """Counts, conditions, perturbations per record and the background."""
    return {
        "records": len(frame),
        "genes": int(frame.gene.nunique()),
        "conditions": sorted(frame.condition.unique()),
        "n_conditions": int(frame.condition.nunique()),
        "perturbations_per_record": {
            str(k): v
            for k, v in sorted(Counter(frame.n_perturbations.tolist()).items())
        },
        "reference": background_summary(reference),
    }


def main() -> None:
    """Read both stores, compare, and write the JSON + per-condition CSV."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-root",
        default=osp.join(
            os.environ["DATA_ROOT"], "data/torchcell/env_chemgen_vanacloig2022"
        ),
    )
    parser.add_argument("--after-root", required=True)
    parser.add_argument("--edger-tsv", default=None)
    args = parser.parse_args()

    before, before_ref = read_store(args.before_root)
    after, after_ref = read_store(args.after_root)

    per_condition: list[dict[str, Any]] = []
    for condition in sorted(set(before.condition) | set(after.condition)):
        b = before[before.condition == condition]
        a = after[after.condition == condition]
        row: dict[str, Any] = {
            "condition": condition,
            "records_before": len(b),
            "records_after": len(a),
            "median_before": float(b.response.median()) if len(b) else None,
            "median_after": float(a.response.median()) if len(a) else None,
            "frac_negative_before": float((b.response < 0).mean()) if len(b) else None,
            "frac_negative_after": float((a.response < 0).mean()) if len(a) else None,
            "median_sd_before": float(b.sd.median()) if len(b) else None,
            "median_sd_after": float(a.sd.median()) if len(a) else None,
        }
        if len(b) and len(a):
            joined = b.merge(a, on="gene", suffixes=("_b", "_a"))
            row["shared_cells"] = len(joined)
            row["spearman_after_vs_before"] = float(
                spearmanr(joined.response_b, joined.response_a)[0]
            )
            row["median_offset_after_minus_before"] = (
                row["median_after"] - row["median_before"]
            )
        per_condition.append(row)
    table = pd.DataFrame(per_condition)

    # DMSO profile correlation in the after store (genes x conditions mean response).
    wide = after.pivot(index="gene", columns="condition", values="response")
    dmso = _name("DMSO")
    dmso_corr: dict[str, float] = {}
    if dmso in wide.columns:
        for condition in wide.columns:
            if condition == dmso:
                continue
            pair = wide[[dmso, condition]].dropna()
            dmso_corr[condition] = float(pearsonr(pair[dmso], pair[condition])[0])
    table["pearson_with_dmso_profile_after"] = table.condition.map(dmso_corr)

    edger: dict[str, Any] | None = None
    if args.edger_tsv is not None:
        e = pd.read_csv(args.edger_tsv, sep="\t")
        e["gene"] = e.gene.str.split("_").str[0]
        e["condition"] = e.token.map(lambda t: resolved_compound(t).name)
        edger = {}
        for label, frame in (("before", before), ("after", after)):
            joined = frame.merge(e, on=["gene", "condition"])
            edger[label] = {
                c: {
                    "spearman": float(spearmanr(g.response, g.logFC)[0]),
                    "median_response_minus_median_logfc": float(
                        g.response.median() - g.logFC.median()
                    ),
                    "cells": len(g),
                }
                for c, g in joined.groupby("condition")
            }
        table["median_edger_logfc"] = table.condition.map(
            e.groupby("condition").logFC.median()
        )

    six = {t: _name(t) for t in AUDIT_SIX}
    shared = table.dropna(subset=["spearman_after_vs_before"])
    result: dict[str, Any] = {
        "before_root": args.before_root,
        "after_root": args.after_root,
        "before": store_summary(before, before_ref),
        "after": store_summary(after, after_ref),
        "conditions_added": sorted(set(after.condition) - set(before.condition)),
        "conditions_removed": sorted(set(before.condition) - set(after.condition)),
        "audit_six": {
            token: table[table.condition == name].iloc[0].to_dict()
            for token, name in six.items()
        },
        "spearman_after_vs_before": {
            "median_over_conditions": float(shared.spearman_after_vs_before.median()),
            "min_over_conditions": float(shared.spearman_after_vs_before.min()),
            "min_condition": shared.loc[
                shared.spearman_after_vs_before.idxmin(), "condition"
            ],
            "pooled_all_shared_cells": float(
                spearmanr(
                    *before.merge(after, on=["gene", "condition"])[
                        ["response_x", "response_y"]
                    ].T.to_numpy()
                )[0]
            ),
        },
        "abs_median_offset_after_minus_before": {
            "max_outside_audit_six": float(
                shared[~shared.condition.isin(six.values())]
                .median_offset_after_minus_before.abs()
                .max()
            )
        },
        "pearson_with_dmso_profile_after": {
            "audit_dmso_similar": {
                t: dmso_corr.get(_name(t)) for t in AUDIT_DMSO_SIMILAR
            },
            "median_over_other_conditions": float(
                np.median(
                    [
                        v
                        for k, v in dmso_corr.items()
                        if k not in {_name(t) for t in AUDIT_DMSO_SIMILAR}
                    ]
                )
            )
            if dmso_corr
            else None,
        },
        "edger": edger,
    }
    os.makedirs(RESULTS, exist_ok=True)
    table.to_csv(
        osp.join(RESULTS, "vanacloig2022_ingestion_per_condition.csv"), index=False
    )
    with open(osp.join(RESULTS, "vanacloig2022_ingestion.json"), "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(json.dumps({k: result[k] for k in ("audit_six",)}, indent=1, default=str))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "spearman_after_vs_before",
                    "abs_median_offset_after_minus_before",
                    "pearson_with_dmso_profile_after",
                    "conditions_added",
                    "conditions_removed",
                )
            },
            indent=1,
        )
    )
    for label in ("before", "after"):
        summary = result[label]
        print(
            label,
            summary["records"],
            summary["n_conditions"],
            summary["perturbations_per_record"],
            json.dumps(summary["reference"], default=str)[:1500],
        )


if __name__ == "__main__":
    main()
