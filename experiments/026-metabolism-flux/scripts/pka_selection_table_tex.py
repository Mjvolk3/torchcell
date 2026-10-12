# experiments/026-metabolism-flux/scripts/pka_selection_table_tex.py
# [[experiments.026-metabolism-flux.scripts.pka_selection_table_tex]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/pka_selection_table_tex.py

r"""Emit the pKa method-selection tables of the 026 kinetics-thermodynamics document.

``pka_method_selection.tex``
    One row per candidate method: failure rate, worst per-pH p90 and the pH it occurs
    at, the pH 7.2 median and mean, ladder-length agreement, cost, and whether the rule
    selected it or which clause rejected it.
``pka_p90_by_ph.tex``
    The same p90 statistic at each of the eight working compartment pHs, one row per
    method, which is the table behind the rule's insistence on the worst pH.

Both read ``results/pka_method_selection.json`` (the decision, its rule and the ranking)
and ``results/pka_method_selection_figure.json`` (the per-pH statistic the panels draw).
Nothing is recomputed.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/pka_selection_table_tex.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import cast

from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
REPO = osp.dirname(EXPERIMENT_ROOT)
TABLE_DIR = osp.join(REPO, "notes-tex", "026-kinetics-thermodynamics", "tables")

DECISION_JSON = osp.join(RESULTS, "pka_method_selection.json")
FIGURE_JSON = osp.join(RESULTS, "pka_method_selection_figure.json")

#: Display name per method. A method absent from here raises rather than printing a slug.
METHOD_LABEL = {
    "cascade_top3": "cascade, three members",
    "cascade_top2": "cascade, two members",
    "ensemble_top2": "median of two",
    "ensemble_top3": "median of three",
    "per_group_best": "best tool per functional group",
    "unipka-simple-template": "Uni-pKa, simple template",
    "molgpka": "MolGpKa",
    "pkasolver": "pKasolver-lite",
}

HEADER = (
    "%% notes-tex/026-kinetics-thermodynamics/tables/{name}\n"
    "%% AUTO-GENERATED -- do not hand-edit. Regenerate with\n"
    "%%   PYTHONPATH=. python experiments/026-metabolism-flux/scripts/"
    "pka_selection_table_tex.py\n"
    "%% SOURCE: {source}\n"
)


def _read(path: str) -> dict[str, object]:
    with open(path) as handle:
        return cast(dict[str, object], json.load(handle))


def _write(name: str, source: str, body: str) -> None:
    path = osp.join(TABLE_DIR, name)
    with open(path, "w") as handle:
        handle.write(HEADER.format(name=name, source=source) + body)
    print(f"wrote {path}")


def main() -> None:
    decision = _read(DECISION_JSON)
    figure = _read(FIGURE_JSON)

    rule = cast(dict[str, object], decision["rule"])
    ranking = cast(list[dict[str, object]], decision["ranking"])
    selected = cast(str, decision["selected"])
    rejected = {
        entry.split(" (")[0]: entry.split("(")[1].rstrip(")")
        for entry in cast(list[str], decision["rejected_for_failure_rate"])
    }
    max_failure = float(cast(float, rule["max_failure_rate"]))
    n_evaluated = int(cast(int, decision["n_evaluated"]))
    n_held_out = f"{n_evaluated:,}".replace(",", "{,}")

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[pKa method selection]{Every candidate p$K_a$ method against the"
        " eQuilibrator compound cache's own ChemAxon constants, on the"
        f" {n_held_out}"
        " held-out compounds. The rule selects the smallest \\emph{worst} per-pH p90 of"
        " $|\\Delta\\Delta_f G'^{\\circ}|$ among methods whose failure rate is at most"
        f" {max_failure:.2f}, breaking ties on the pH 7.2 median and then on cost. Failure"
        " rate is the fraction of compounds a method returns no ladder for, and it is a"
        " bound rather than a scored error because a method that cannot answer leaves a"
        " gap rather than a cheap answer. \\emph{Right count} is the fraction whose ladder"
        " length matches the cache's, reported and deliberately outside the rule: a ladder"
        " can carry the right number of constants and the wrong values, and the transform"
        " is what the flux layer consumes.}",
        "  \\label{tab:pka-selection}",
        "  \\begin{tabular}{@{}llrrrrrrl@{}}",
        "    \\toprule",
        "    & & & \\multicolumn{2}{c}{worst pH} & \\multicolumn{2}{c}{at pH 7.2} & &"
        " \\\\",
        "    Method & Kind & Fail & p90 & pH & Median & Mean & Right count & Verdict"
        " \\\\",
        "    \\midrule",
    ]
    for entry in ranking:
        method = cast(str, entry["method"])
        if method not in METHOD_LABEL:
            raise KeyError(f"method {method!r} has no display name in this script")
        if method == selected:
            verdict = "\\textbf{selected}"
        elif method in rejected:
            verdict = "rejected, " + rejected[method]
        else:
            verdict = "ranked below"
        lines.append(
            f"    {METHOD_LABEL[method]} & {entry['kind']} & "
            f"{float(cast(float, entry['failure_rate'])):.4f} & "
            f"{float(cast(float, entry['worst_working_p90'])):.2f} & "
            f"{entry['worst_working_ph']} & "
            f"{float(cast(float, entry['median_ph72'])):.3f} & "
            f"{float(cast(float, entry['mae_ph72'])):.2f} & "
            f"{float(cast(float, entry['frac_count_match'])):.3f} & {verdict} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\vspace{0.3em}\\\\",
        "  {\\footnotesize p90, median and mean are of"
        " $|\\Delta\\Delta_f G'^{\\circ}|$ in kJ\\,mol$^{-1}$.}",
        "\\end{table}",
        "",
    ]
    _write(
        "pka_method_selection.tex",
        "results/pka_method_selection.json, rule and ranking, written by "
        "external_pka_method_selection.py decide",
        "\n".join(lines),
    )

    panels = cast(list[dict[str, object]], figure["panels"])
    per_ph = next(p for p in panels if p["panel"] == "b")
    values = cast(dict[str, object], per_ph["values"])
    working_ph = [float(v) for v in cast(list[float], values["working_ph"])]
    by_method = cast(dict[str, list[float]], values["p90_by_method"])

    header = " & ".join(f"{ph:g}" for ph in working_ph)
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[pKa error by compartment pH]{The p90 of"
        " $|\\Delta\\Delta_f G'^{\\circ}|$, in kJ\\,mol$^{-1}$, at each pH the pipeline"
        " transforms a metabolite at. A metabolite is priced at its own compartment's pH,"
        " so a method has to hold at all eight; scoring at pH 7.2 alone understates the"
        " selected cascade's tail by 3.4 kJ\\,mol$^{-1}$ and ranks MolGpKa above Uni-pKa."
        " The mitochondrial matrix at pH 7.8 is the worst column for every method but"
        " one.}",
        "  \\label{tab:pka-by-ph}",
        "  \\begin{tabular}{@{}l" + "r" * len(working_ph) + "@{}}",
        "    \\toprule",
        "    & \\multicolumn{" + str(len(working_ph)) + "}{c}{compartment pH} \\\\",
        f"    Method & {header} \\\\",
        "    \\midrule",
    ]
    for method, series in sorted(by_method.items(), key=lambda kv: max(kv[1])):
        if method not in METHOD_LABEL:
            raise KeyError(f"method {method!r} has no display name in this script")
        cells = " & ".join(f"{value:.2f}" for value in series)
        name = METHOD_LABEL[method]
        if method == selected:
            name = f"\\textbf{{{name}}}"
        lines.append(f"    {name} & {cells} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    _write(
        "pka_p90_by_ph.tex",
        "results/pka_method_selection_figure.json, panel b, written by "
        "plot_pka_method_selection.py",
        "\n".join(lines),
    )


if __name__ == "__main__":
    main()
