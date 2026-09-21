# experiments/025-solid-growth/scripts/s3_closure_asymmetric_table.py
# [[experiments.025-solid-growth.s3-closure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/s3_closure_asymmetric_table

"""Render the asymmetric-versus-symmetric table of the S3 closure document.

Reads the per-screen results written by
``experiments/029-solid-growth-ko/scripts/closure_recompute_asymmetric.py`` and writes
``notes-tex/025-s3-closure/tables/t9-asymmetric.tex``. The document's other tables are
written by the script that computes them; this one sits here because the computation
lives in the 029 experiment while the document lives with 025, and a table in the paper
must still come from a committed script rather than be typed by hand.

    python experiments/025-solid-growth/scripts/s3_closure_asymmetric_table.py
"""

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
CSV = osp.join(
    EXPERIMENT_ROOT, "029-solid-growth-ko/results/closure_asymmetric_by_screen.csv"
)
TABLE = osp.join(
    osp.dirname(EXPERIMENT_ROOT), "notes-tex/025-s3-closure/tables/t9-asymmetric.tex"
)
SCREEN = {"kuzmin2018": "Kuzmin 2018", "kuzmin2020": "Kuzmin 2020"}
FORM = {
    "asymmetric (published)": "asymmetric, as published",
    "symmetric": "symmetric",
    "asymmetric, array single set to 1": "array single forced to 1",
}
ORDER = list(FORM)


def main() -> None:
    """Write the LaTeX table from the per-screen results."""
    df = pd.read_csv(CSV)
    head = (
        "%% SOURCE: experiments/029-solid-growth-ko/scripts/closure_recompute_asymmetric.py "
        "(results/closure_asymmetric_by_screen.csv), rendered by "
        "experiments/025-solid-growth/scripts/s3_closure_asymmetric_table.py "
        "-- GENERATED, do not edit\n"
    )
    lines = [
        head,
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"screen & form & $n$ & $r$ & slope & rmse \\",
        r"\midrule",
    ]
    for si, (screen, label) in enumerate(SCREEN.items()):
        g = df[df["screen"] == screen].set_index("form")
        best = max(g.loc[f, "pearson"] for f in ORDER if f in g.index)
        for fi, form in enumerate(ORDER):
            if form not in g.index:
                continue
            row = g.loc[form]
            r_txt = f"{row['pearson']:.3f}"
            if row["pearson"] == best:
                r_txt = r"\textbf{" + r_txt + "}"
            name = label if fi == 0 else ""
            lines.append(
                f"{name} & {FORM[form]} & {int(row['n']):,} & {r_txt} & "
                f"{row['slope']:.2f} & {row['rmse']:.3f} \\\\"
            )
        if si == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(osp.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
