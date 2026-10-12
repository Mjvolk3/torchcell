# experiments/026-metabolism-flux/scripts/kinetic_tier_tables_tex.py
# [[experiments.026-metabolism-flux.scripts.kinetic_tier_tables_tex]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kinetic_tier_tables_tex.py

r"""Emit the kinetic-tier tables of the 026 kinetics-thermodynamics document.

``kinetic_tiers.tex``
    One row per tier of the resolved $k_{cat}$ and $K_M$ tables: how the tier is keyed,
    what it is evidence about, and how many of the 3,728 catalytic units it fills.
``kinetic_default_units.tex``
    The 18 units still on the organism default, one row each, with the substrate that
    blocks them and why no open-source predictor reaches it.

Both read ``results/kinetic_coverage_closure.json``. The tier definitions are declared
here, because a tier's meaning is documentation; the counts are read.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/kinetic_tier_tables_tex.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
REPO = osp.dirname(EXPERIMENT_ROOT)
TABLE_DIR = osp.join(REPO, "notes-tex", "026-kinetics-thermodynamics", "tables")

CLOSURE_JSON = osp.join(RESULTS, "kinetic_coverage_closure.json")
SHIPPED_VARIANT = "verified_substrate_key_with_homolog"


class TierDoc(BaseModel):
    """The prose half of one tier's row."""

    model_config = ConfigDict(extra="forbid")

    display: str
    keyed_on: str
    evidence: str
    what_it_is_evidence_about: str


TIERS: dict[str, TierDoc] = {
    "measured": TierDoc(
        display="measured, this organism",
        keyed_on=(
            "the unit's own UniProt accession in the Open Enzyme Database mirror, and the"
            " assay substrate matched by InChIKey connectivity block to a substrate the"
            " reaction consumes"
        ),
        evidence="measured",
        what_it_is_evidence_about="this protein doing this chemistry",
    ),
    "homolog": TierDoc(
        display="measured, same EC, another organism",
        keyed_on=(
            "an EC number annotated on the reaction plus the same substrate match, in any"
            " organism"
        ),
        evidence="measured",
        what_it_is_evidence_about=(
            "the chemistry, not this protein, so it is never counted as a measurement of"
            " the modeled organism"
        ),
    ),
    "predicted": TierDoc(
        display="predicted",
        keyed_on="the protein sequence and the substrate structure",
        evidence="estimated",
        what_it_is_evidence_about=(
            "the median over the predictors the within-class study admits, which is"
            " EZKinetics, EITLEM and UniKP for $k_{cat}$ and EZKinetics alone for $K_M$"
        ),
    ),
    "default": TierDoc(
        display="organism default",
        keyed_on="nothing; it is the fallback",
        evidence="tabulated",
        what_it_is_evidence_about=(
            "the median of the values the database tiers placed in this same table, so it"
            " is a summary of the organism rather than of the unit"
        ),
    ),
}

HEADER = (
    "%% notes-tex/026-kinetics-thermodynamics/tables/{name}\n"
    "%% AUTO-GENERATED -- do not hand-edit. Regenerate with\n"
    "%%   PYTHONPATH=. python experiments/026-metabolism-flux/scripts/"
    "kinetic_tier_tables_tex.py\n"
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


#: Every character LaTeX reads as markup. Substrate names carry braces, brackets and
#: arrows straight out of the GEM, so nothing may reach the table unescaped.
_REPLACEMENTS = [
    ("\\", "\\textbackslash{}"),
    ("{", "\\{"),
    ("}", "\\}"),
    ("_", "\\_"),
    ("%", "\\%"),
    ("&", "\\&"),
    ("#", "\\#"),
    ("$", "\\$"),
    ("^", "\\textasciicircum{}"),
    ("~", "\\textasciitilde{}"),
    (">", "\\textgreater{}"),
    ("<", "\\textless{}"),
]


def _escape(text: str) -> str:
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _num(value: int) -> str:
    """Thousands separator LaTeX can typeset without a space after the comma."""
    return f"{value:,}".replace(",", "{,}")


def main() -> None:
    closure = _read(CLOSURE_JSON)
    kcat = cast(
        dict[str, object], cast(dict[str, object], closure["parameters"])["k_cat"]
    )
    n_units = int(cast(int, kcat["n_units"]))
    variants = cast(dict[str, dict[str, object]], kcat["variants"])
    counts = cast(dict[str, int], variants[SHIPPED_VARIANT]["tier_counts"])

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Kinetic tiers]{The four tiers a turnover number or Michaelis"
        " constant is drawn from, tried in this order over the"
        f" {_num(n_units)} catalytic units of yeast-GEM 9.0.2."
        " $k_{cat}$ and $K_M$ resolve on the same"
        " keys and land on the same counts, because a unit that has one measured has the"
        " other. The homolog tier is measured and is nonetheless folded in with the"
        " predictions wherever an experimental fraction is reported, since a measurement"
        " of another organism's enzyme is not a measurement of this one.}",
        "  \\label{tab:kinetic-tiers}",
        "  \\begin{tabular}{@{}p{0.18\\textwidth}p{0.27\\textwidth}"
        "p{0.28\\textwidth}rr@{}}",
        "    \\toprule",
        "    Tier & Keyed on & Evidence about & Units & Share \\\\",
        "    \\midrule",
    ]
    for tier, doc in TIERS.items():
        count = counts[tier]
        lines.append(
            f"    {doc.display} & {doc.keyed_on} & {doc.what_it_is_evidence_about} & "
            f"{_num(count)} & {100.0 * count / n_units:.1f}\\% \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    _write(
        "kinetic_tiers.tex",
        "results/kinetic_coverage_closure.json, parameters.k_cat.variants."
        f"{SHIPPED_VARIANT}.tier_counts; tier definitions declared in "
        "kinetic_tier_tables_tex.py",
        "\n".join(lines),
    )

    ceiling = cast(dict[str, object], closure["ceiling"])
    remaining = cast(list[dict[str, object]], ceiling["remaining"])
    by_unit: dict[int, list[dict[str, object]]] = {}
    for row in remaining:
        by_unit.setdefault(int(cast(int, row["unit_id"])), []).append(row)

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Units on the organism default]{Every catalytic unit still carrying"
        " the organism default, and the substrate that keeps it there. A unit needs a"
        " structure for at least one substrate its reaction consumes; these have none any"
        " mirrored predictor can take. \\emph{Rows} is how many of the unit's substrates"
        " fail. None of the four classes is reachable with an open-source tool: an R-group"
        " placeholder is a curation decision in the model rather than a missing structure,"
        " a macromolecular substrate needs a predictor that accepts a protein and all"
        " seven mirrored models take a small-molecule SMILES, and the structure writer was"
        " run on the last two classes and refused them.}",
        "  \\label{tab:kinetic-defaults}",
        "  \\begin{tabular}{@{}rllp{0.30\\textwidth}p{0.22\\textwidth}r@{}}",
        "    \\toprule",
        "    Unit & Reaction & Genes & Substrate & Why & Rows \\\\",
        "    \\midrule",
    ]
    for unit_id in sorted(by_unit):
        rows = by_unit[unit_id]
        first = rows[0]
        genes = ", ".join(cast(list[str], first["gene_ids"]))
        substrates = "; ".join(sorted({cast(str, r["substrate"])[:38] for r in rows}))
        substrates = substrates[:78] + ("..." if len(substrates) > 78 else "")
        reason = cast(str, first["reason"])
        reason = reason[:56] + ("..." if len(reason) > 56 else "")
        lines.append(
            f"    {unit_id} & \\texttt{{{_escape(cast(str, first['reaction_id']))}}} & "
            f"{_escape(genes)} & {_escape(substrates)} & {_escape(reason)} & "
            f"{len(rows)} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    _write(
        "kinetic_default_units.tex",
        "results/kinetic_coverage_closure.json, ceiling.remaining, written by "
        "improve_kinetic_coverage.py",
        "\n".join(lines),
    )


if __name__ == "__main__":
    main()
