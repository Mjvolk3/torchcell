# experiments/019-perturb-seq-costing/scripts/cost_per_cell_table.py
# [[experiments.019-perturb-seq-costing.scripts.cost_per_cell_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/cost_per_cell_table
"""Cost per cell across single-cell RNA-seq methods, with citations.

Emits two tables:

* ``cost_per_cell.csv`` -- the published/derived LIBRARY-PREP cost per cell,
  one row per method, each carrying its citation key and verbatim quote.
* ``cost_per_cell_loaded.csv`` -- the same platforms costed end to end, per
  *usable* cell, from ``cost_model``.

The two differ by more than an order of magnitude for split-pool, and that gap
is the point. A quoted per-cell figure is reagents divided by cells *processed*.
The number that decides a budget is total spend divided by cells that survive QC
and carry an identified guide. Split-pool wins the first comparison by ~60x and
the second by ~2x, because it spends the most reads per usable UMI (94% of its
reads are rRNA as published) and discards the largest share of what it
sequences.

Also emits ``cost_per_cell.md`` -- the same table as markdown, so the note never
hand-transcribes a number.

Run:  python experiments/019-perturb-seq-costing/scripts/cost_per_cell_table.py
"""

from __future__ import annotations

import os
import os.path as osp
import textwrap

import pandas as pd
from dotenv import load_dotenv

import cost_data as CD
import cost_model as CM

load_dotenv()
RESULTS_DIR = osp.join(
    os.environ["EXPERIMENT_ROOT"], "019-perturb-seq-costing", "results"
)

# Short citation strings for the note. Keys match the tc-lit mirror / Zotero.
CITE = {
    "brettnerUltraHighthroughputMassively2024": "Brettner 2024, *Yeast* 41:242",
    "jarianiNewProtocolSinglecell2020": "Jariani 2020, *eLife* 9:e55320",
    "gaisserHighthroughputSinglecellTranscriptomics2024": (
        "Gaisser 2024, *Nat Protoc*, Suppl. Table 2"
    ),
}


def library_prep_table() -> pd.DataFrame:
    rows = []
    for c in sorted(CD.PER_CELL_COSTS, key=lambda x: -x.usd_per_cell):
        rows.append(
            {
                "method": c.label,
                "organism": c.organism,
                "isolation": c.isolation.replace("_", "-"),
                "usd_per_cell": c.usd_per_cell,
                "scope": c.scope,
                "sourced_or_derived": "derived" if c.derived else "quoted",
                "derivation": c.derivation or "",
                "citation": CITE[c.citation_key],
                "citation_key": c.citation_key,
                "line": c.line or "",
                "quote": c.quote,
            }
        )
    return pd.DataFrame(rows)


def loaded_table(cells_per_gene: int = 250) -> pd.DataFrame:
    """End-to-end cost per usable cell, from the genome-scale model."""
    design = CM.ScreenDesign(cells_per_gene=cells_per_gene)
    rows = []
    for plat in CM.PLATFORMS:
        b = CM.budget_for(design, plat)
        # Quoted-style figure for the same platform: reagents only, per cell
        # PROCESSED -- the like-for-like comparison against the table above.
        reagents = b.protocol_usd + b.sublibrary_usd
        rows.append(
            {
                "platform": plat.name,
                "projected": plat.projected,
                "usable_cells": b.usable_cells,
                "sequenced_cells": b.sequenced_cells,
                "usable_fraction": plat.usable_fraction,
                "reagents_usd": round(reagents),
                "sequencing_usd": round(b.sequencing_usd),
                "total_usd": round(b.recurring_usd),
                "reagent_usd_per_cell_processed": round(
                    reagents / b.sequenced_cells, 5
                ),
                "loaded_usd_per_usable_cell": round(
                    b.recurring_usd / b.usable_cells, 4
                ),
                "hidden_multiplier": round(
                    (b.recurring_usd / b.usable_cells)
                    / (reagents / b.sequenced_cells),
                    1,
                ),
                "sequencing_share_pct": round(
                    100 * b.sequencing_usd / b.recurring_usd, 1
                ),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Markdown table of the library-prep costs, formatted for the note."""
    lines = [
        "| Method | Organism | Isolation | $ / cell | Basis | Source |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for _, r in df.iterrows():
        usd = r["usd_per_cell"]
        # Four orders of magnitude in one column: a fixed 2-dp format collapses
        # the three cheapest methods onto "$0.00" and hides the whole point.
        if usd >= 1:
            s = f"${usd:,.2f}"
        elif usd >= 0.01:
            s = f"${usd:.4f}"
        else:
            s = f"${usd:.5f}"
        basis = "quoted" if r["sourced_or_derived"] == "quoted" else "derived"
        lines.append(
            f"| {r['method']} | *{r['organism']}* | {r['isolation']} | {s} | "
            f"{basis} | {r['citation']} |"
        )
    return "\n".join(lines)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lp = library_prep_table()
    lp.to_csv(osp.join(RESULTS_DIR, "cost_per_cell.csv"), index=False)

    ld = loaded_table()
    ld.to_csv(osp.join(RESULTS_DIR, "cost_per_cell_loaded.csv"), index=False)

    md = to_markdown(lp)
    with open(osp.join(RESULTS_DIR, "cost_per_cell.md"), "w") as f:
        f.write("<!-- generated by cost_per_cell_table.py -- do not hand-edit -->\n")
        f.write(md + "\n")

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("=== Library-prep cost per cell (as quoted / derived from sources) ===")
    print(
        lp[
            [
                "method",
                "organism",
                "isolation",
                "usd_per_cell",
                "sourced_or_derived",
                "citation",
            ]
        ].to_string(index=False)
    )
    print(
        "\nspread, cheapest to dearest: "
        f"{CD.PER_CELL_SPREAD:,.0f}x\n"
    )
    print("=== End-to-end, per USABLE cell (6,000 genes, 250 cells/gene) ===")
    print(
        ld[
            [
                "platform",
                "reagent_usd_per_cell_processed",
                "loaded_usd_per_usable_cell",
                "hidden_multiplier",
                "sequencing_share_pct",
            ]
        ].to_string(index=False)
    )
    print(
        textwrap.dedent(
            """
            'hidden_multiplier' is how much larger the real per-cell cost is than
            the reagents-only figure a paper would quote for the same platform.
            """
        )
    )
    print(f"Wrote CSVs + cost_per_cell.md to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
