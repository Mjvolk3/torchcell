# experiments/026-metabolism-flux/scripts/external_pka_table_tex.py
# [[experiments.026-metabolism-flux.external-pka-route]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/external_pka_table_tex.py

r"""Render the external-pKa benchmark table into the 026 document, from the benchmark.

The numbers come from ``external_pka_benchmark.py table``, which is the committed reader of
``results/external_pka_benchmark.json``. That subcommand emits GitHub-flavored Markdown and
has no ``--out`` and no LaTeX mode, and it is owned elsewhere, so this script runs it and
translates its FIRST table rather than reading the JSON a second time. Going through the
subcommand is deliberate: two readers of one JSON drift, and the note and the document then
disagree about a number with no way to tell which is stale.

Output is a complete float, caption and label included, so the section is a one-line
``\input`` and no benchmark number is ever typed into a ``.tex`` file.

Run from the repo root with the torchcell env (the equilibrator env is needed for the
benchmark's ``sample``/``predict``/``score`` steps, not for ``table``, which only reads
JSON):

    python experiments/026-metabolism-flux/scripts/external_pka_table_tex.py
"""

import argparse
import os
import os.path as osp
import re
import subprocess
import sys
from typing import cast

from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
SCRIPTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
BENCHMARK = osp.join(SCRIPTS, "external_pka_benchmark.py")
REPO = osp.abspath(osp.join(EXPERIMENT_ROOT, ".."))
DEFAULT_OUT = osp.join(
    REPO, "notes-tex", "026-kinetics-thermodynamics", "tables", "external_pka_tools.tex"
)

#: The header the benchmark prints, mapped to what the column is called on the page. The
#: keys are matched exactly, so a rename upstream fails loudly here rather than silently
#: shifting a column.
HEADERS = {
    "tool": "Tool",
    "n attempted": "$n$",
    "n scored": "Scored",
    "right pKa count": "Right count",
    "pKa MAE": "p$K_a$ MAE",
    "dGf'0 MAE (kJ/mol)": "MAE",
    "dGf'0 median (kJ/mol)": "Median",
    "dGf'0 p90 (kJ/mol)": "p90",
}

DISPLAY = {
    "pkasolver-lite": r"pKasolver-lite",
    "molgpka": r"MolGpKa",
    "unipka": r"Uni-pKa, full template",
    "unipka-simple-template": r"Uni-pKa, simple template",
    "qupkake": r"QupKake",
    "qupkake-truncated": r"QupKake, truncated",
}


def run_table() -> str:
    """Capture the benchmark's own table output, verbatim."""
    result = subprocess.run(
        [sys.executable, BENCHMARK, "table"], capture_output=True, text=True, check=True
    )
    return result.stdout


def parse_first_table(text: str) -> tuple[str, list[str], list[list[str]]]:
    """The scoring-conditions line, the header cells, and the rows of the first table."""
    lines = [line.rstrip() for line in text.splitlines()]
    scored_at = next(line for line in lines if line.startswith("Scored at"))
    start = next(i for i, line in enumerate(lines) if line.startswith("| tool |"))
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return scored_at, header, rows


def conditions(scored_at: str) -> str:
    """Pull the sample size and the state from the benchmark's own preamble line."""
    return re.sub(r"^Scored at \S+ ", "", scored_at).rstrip(".")


def render(scored_at: str, header: list[str], rows: list[list[str]]) -> str:
    unknown = [h for h in header if h not in HEADERS]
    if unknown:
        raise ValueError(f"benchmark emitted unrecognized columns: {unknown}")
    names = [HEADERS[h] for h in header]
    body = []
    for row in rows:
        cells = [DISPLAY.get(row[0], row[0]), *row[1:]]
        body.append("    " + " & ".join(cells) + r" \\")
    return "\n".join(
        [
            "%% notes-tex/026-kinetics-thermodynamics/tables/external_pka_tools.tex",
            "%% AUTO-GENERATED -- do not hand-edit. Regenerate with",
            "%%   python experiments/026-metabolism-flux/scripts/external_pka_table_tex.py",
            "%% SOURCE: experiments/026-metabolism-flux/results/external_pka_benchmark.json,",
            "%% read through `external_pka_benchmark.py table`.",
            r"\begin{table}[H]",
            r"  \centering",
            r"  \footnotesize",
            r"  \caption[External pKa predictors]{Open-source p$K_a$ predictors against the"
            r" compound cache's own ChemAxon-derived constants, " + conditions(scored_at)
            + r". The last three columns are the change in $\Delta_f G'^{\circ}$ when the"
            r" predicted constants replace the cache's, with the reference microspecies and"
            r" the untransformed formation energy held fixed, so they isolate the p$K_a$"
            r" contribution. \textbf{The two Uni-pKa rows are the same weights and the same"
            r" code}, and differ only in the microstate enumeration template, which fixes how"
            r" many rungs a ladder can have before the network is consulted."
            r" \textbf{QupKake is unmeasured on this set}: a rung costs a full"
            r" xTB wave, the run reached 4 of up to 14 rungs, and only 1 of 60 ladders had"
            r" terminated. Its truncated row scores all 60 with the rungs that ran, so those"
            r" numbers bound it rather than measure it.}",
            r"  \label{tab:external-pka}",
            r"  \begin{tabular}{@{}l" + "r" * (len(names) - 1) + r"@{}}",
            r"    \hline",
            r"    \multicolumn{5}{@{}l}{} & \multicolumn{3}{c@{}}{"
            r"$\Delta_f G'^{\circ}$ error (kJ\,mol$^{-1}$)} \\",
            "    " + " & ".join(names) + r" \\",
            r"    \hline",
            *body,
            r"    \hline",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    scored_at, header, rows = parse_first_table(run_table())
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write(render(scored_at, header, rows))
    print(f"wrote {args.out} ({len(rows)} tools)")


if __name__ == "__main__":
    main()
