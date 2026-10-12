# experiments/036-dataset-fixes-before-kg-build/scripts/load_time_name_resolution_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.load_time_name_resolution_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/load_time_name_resolution_inventory.py
"""Every loader that still resolves gene names against a genome at LOAD time.

Owner, 2026-10-12: "the data is the data!!!! it is what they reported ... the genome is
separate.. it doesn't gate anything... we do that at modeling time if it is an issue." A
loader stores each record under the names its release gives; resolution against the
annotation is downstream. This script reads every module under ``torchcell/datasets``
with ``ast`` (nothing is imported) and lists, per module, each call that resolves a name
against a genome, split by where it runs:

- ``load``: inside the build path (``process``, ``create_experiment``, readers, helpers);
  these are the calls the sweep removes or moves downstream;
- ``verify``: inside a verification function (``verify*``, ``run_verification``,
  ``_l0_``..``_l4_`` and ``l0_``..``l4_`` helpers, ``*_rule``, ``stored_*``), which are
  reported findings since the same day and stay.

Resolution calls counted: ``.resolve_gene_name(``, ``resolve_gene_name_strict``,
``check_ambiguous_aliases``, ``candidate_orfs``, ``check_pin``, ``reconcile_locus_tags``
and ``require_resolved`` (the bacterial reconciliation and its whole-dataset threshold),
``sga_systematic_name_currency`` / ``SystematicNameCurrency`` (the SGA rename-or-refuse
table), and any use of ``SGA_SYSTEMATIC_REFUSALS`` / ``MERGED_INTO_LOCUS`` /
``GeneNameRefused``. Each module is assigned the branch that holds it (the coordinator's
list of 2026-10-12); a module no branch holds is left to the sweep.

Writes ``results/load_time_name_resolution_inventory.json`` and prints the table.
"""

from __future__ import annotations

import ast
import json
import os.path as osp
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parents[3]
DATASETS = REPO / "torchcell" / "datasets"
RESULTS = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "load_time_name_resolution_inventory.json"
)

#: Attribute or name references that resolve a gene name against a genome.
RESOLUTION_CALLS = frozenset(
    {
        "resolve_gene_name",
        "resolve_gene_name_strict",
        "check_ambiguous_aliases",
        "candidate_orfs",
        "check_pin",
        "reconcile_locus_tags",
        "require_resolved",
        # systematic_name_currency's entry points: map or refuse released SGA names
        "sga_systematic_name_currency",
        "SystematicNameCurrency",
    }
)
#: Names whose mere use is a load-time refusal decision.
REFUSAL_NAMES = frozenset(
    {"SGA_SYSTEMATIC_REFUSALS", "MERGED_INTO_LOCUS", "GeneNameRefused"}
)
VERIFY_FUNCTION = re.compile(
    r"^(_?verify|run_verification|run_.*verification|_?l[0-4]_|.*_rule$|stored_|"
    r"every_locus|_check_|check_|_l4|_assert_)"
)

#: Who holds each module on 2026-10-12 (coordinator's list). Modules not here go to
#: the sweep.
HOLDERS: dict[str, str] = {
    "scerevisiae/costanzo2016.py": "fix/audit-yeast-refusals-1045",
    "scerevisiae/kuzmin2018.py": "fix/audit-yeast-refusals-1045, fix/keep-records-887-sd-zero",
    "scerevisiae/kuzmin2020.py": "fix/audit-yeast-refusals-1045, fix/keep-records-887-sd-zero",
    "scerevisiae/yoshikawa2009.py": "fix/audit-yeast-refusals-1045",
    "scerevisiae/parsons2006.py": "fix/audit-yeast-refusals-1045",
    "scerevisiae/muenzner2024.py": "fix/audit-yeast-refusals-1045",
    "scerevisiae/xue2025.py": "fix/keep-records-887-sd-zero",
    "scerevisiae/nguessan2025.py": "fix/nguessan2025-register-1050",
    "systematic_name_currency.py": "fix/audit-yeast-refusals-1045",
    "ecoli/babu2014.py": "fix/audit-bacterial-refusals-1045",
    "ecoli/butland2008.py": "fix/audit-bacterial-refusals-1045",
    "ecoli/girgis2009.py": "fix/audit-bacterial-refusals-1045",
    "bacteria_common.py": "fix/audit-bacterial-refusals-1045",
}
BACTERIAL_SET = "fix/audit-bacterial-refusals-1045 (bacterial set)"


class CallSite(BaseModel):
    """One resolution call, the function it sits in, and whether that is load time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    line: int
    function: str
    phase: str


class ModuleInventory(BaseModel):
    """One module's resolution calls and its holder."""

    model_config = ConfigDict(extra="forbid")

    module: str
    holder: str
    load: list[CallSite] = Field(default_factory=list)
    verify: list[CallSite] = Field(default_factory=list)


def _calls(path: Path) -> list[CallSite]:
    """Every resolution reference in ``path`` with its enclosing function."""
    tree = ast.parse(path.read_text())
    sites: list[CallSite] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                else function
            )
            name = None
            if isinstance(child, ast.Attribute) and child.attr in (
                RESOLUTION_CALLS | REFUSAL_NAMES
            ):
                name = child.attr
            elif isinstance(child, ast.Name) and child.id in (
                RESOLUTION_CALLS | REFUSAL_NAMES
            ):
                name = child.id
            if name is not None and function != "<module>":
                phase = "verify" if VERIFY_FUNCTION.match(function) else "load"
                sites.append(
                    CallSite(
                        name=name, line=child.lineno, function=function, phase=phase
                    )
                )
            visit(child, inner)

    visit(tree, "<module>")
    return sites


def holder_of(relative: str) -> str:
    """The branch holding a module, the bacterial set, or the sweep."""
    if relative in HOLDERS:
        return HOLDERS[relative]
    if relative.startswith(("ecoli/", "pputida/")):
        return BACTERIAL_SET
    return "sweep (unheld)"


def inventory() -> list[ModuleInventory]:
    """Every module under ``torchcell/datasets`` with at least one resolution call."""
    out: list[ModuleInventory] = []
    for path in sorted(DATASETS.rglob("*.py")):
        relative = str(path.relative_to(DATASETS))
        if relative == "gene_alias_resolution.py":
            continue
        sites = _calls(path)
        if not sites:
            continue
        row = ModuleInventory(module=relative, holder=holder_of(relative))
        for site in sites:
            (row.load if site.phase == "load" else row.verify).append(site)
        out.append(row)
    return out


def render(rows: list[ModuleInventory]) -> str:
    """The load-time table, grouped by holder, as markdown."""
    load_rows = [r for r in rows if r.load]
    lines = [
        f"{len(load_rows)} modules resolve names at load time; "
        f"{sum(1 for r in rows if r.verify and not r.load)} only in verification.",
        "",
        "| Module | Holder | Load-time calls (function: names) |",
        "|---|---|---|",
    ]
    for row in sorted(load_rows, key=lambda r: (r.holder, r.module)):
        by_function: dict[str, set[str]] = {}
        for site in row.load:
            by_function.setdefault(site.function, set()).add(site.name)
        calls = "; ".join(
            f"`{fn}`: {', '.join(sorted(names))}"
            for fn, names in sorted(by_function.items())
        )
        lines.append(f"| `{row.module}` | {row.holder} | {calls} |")
    return "\n".join(lines)


def main() -> int:
    """Write the inventory and print the table."""
    rows = inventory()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps([r.model_dump() for r in rows], indent=2) + "\n", encoding="utf-8"
    )
    print(render(rows))
    print(f"\n-> {osp.relpath(RESULTS, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
