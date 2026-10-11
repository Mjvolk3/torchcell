# experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_table.py
# [[experiments.030-solid-growth-multi.scripts.triple_noise_ceiling_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_table

"""Render the trigenic-score reproducibility table of the S3 closure document.

Reads ``results/triple_noise_ceiling_030.json`` written by
``triple_noise_ceiling_030.py`` and writes
``notes-tex/025-s3-closure/tables/t10-triple-reproducibility.tex``: one row per pair class
of re-measured triples, with the count, Pearson, Spearman and its bootstrap interval, and
the median absolute difference between the two measurements. The p-derived noise model of
the same results file is not tabulated; the document explains why it is not trusted.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_table.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import results_dir  # noqa: E402

# class key in the results file -> (what differs between the two measurements, screen)
ROWS = [
    ("same_dataset_same_strains", "same strains, two screens of one year", "Kuzmin 2020"),
    (
        "same_dataset_same_query_different_array_allele",
        "same query, different array allele",
        "both",
    ),
    ("kuzmin2018_only", "any re-measurement within the year", "Kuzmin 2018"),
    ("cross_dataset_2018_vs_2020", "same triple in both years", "2018 vs 2020"),
    ("all_pairs", "every pair", "both"),
]


def main() -> None:
    """Write the LaTeX table from the pair classes of the results file."""
    load_dotenv()
    src = osp.join(results_dir(), "triple_noise_ceiling_030.json")
    with open(src) as f:
        res = json.load(f)
    classes = res["pair_classes"]
    table = osp.join(
        osp.dirname(os.environ["EXPERIMENT_ROOT"]),
        "notes-tex/025-s3-closure/tables/t10-triple-reproducibility.tex",
    )
    head = (
        "%% SOURCE: experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_030.py "
        "(results/triple_noise_ceiling_030.json), rendered by "
        "experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_table.py "
        "-- GENERATED, do not edit\n"
    )
    best_sp = max(classes[k]["spearman"] for k, _, _ in ROWS if k in classes)
    lines = [
        head,
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"what differs & screen & triples & $r$ & $\rho$ (95\% CI) & median $|\Delta\tau|$ \\",
        r"\midrule",
    ]
    for key, what, screen in ROWS:
        if key not in classes:
            continue
        c = classes[key]
        lo, hi = c["spearman_ci95"]
        sp = f"{c['spearman']:.2f} ({lo:.2f} to {hi:.2f})"
        if c["spearman"] == best_sp:
            sp = r"\textbf{" + f"{c['spearman']:.2f}" + "}" + f" ({lo:.2f} to {hi:.2f})"
        lines.append(
            f"{what} & {screen} & {c['n_triples']:,} & {c['pearson']:.2f} & {sp} & "
            f"{c['median_abs_diff']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(osp.dirname(table), exist_ok=True)
    with open(table, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
